"""
DiT Blocks V2 for Diffusion Snake

Upgraded components based on latest 2024-2025 research:
  [M2] SeparatePointEmbedding — Avoids coordinate drowning in high-dim features
  [M3] CyclicRoPE1D          — Per-layer rotary PE for closed contours (FLUX-style)
  [M4] DiTBlockV2            — RMSNorm + QK-Norm + SwiGLU + 9-param adaLN-Zero

References:
  - DiT (ICCV 2023): adaLN-Zero, zero-init
  - FLUX.1 (2024): per-layer RoPE, parallel attn+FFN
  - SiT (ECCV 2024): velocity prediction
  - SwiGLU (Shazeer 2020): gated FFN
  - QK-Norm (2024): attention stability

Author: DiffSnake Team
Date: 2026-04-02
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Utility: adaLN modulation (shared with V1)
# ---------------------------------------------------------------------------
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer-norm modulation: x * (1 + scale) + shift."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# RMSNorm — faster than LayerNorm, no mean centering (LLaMA/PaLM standard)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, NeurIPS 2019).

    Widely adopted in LLaMA 3, PaLM 2, Mistral, DeepSeek as of 2024.
    Eliminates mean-centering for efficiency while preserving scale invariance.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# SwiGLU FFN — gated linear unit with Swish (LLaMA/PaLM standard)
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network: (x·W1 ⊙ Swish(x·V)) · W2

    Reference: Shazeer, "GLU Variants Improve Transformer" (2020).
    Uses the 2/3 rule to maintain FLOP parity with standard SiLU-MLP.
    Hidden dim rounded to nearest multiple of 64 for GPU efficiency.
    """

    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        # Round to nearest multiple of 64 for hardware efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.v = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.v(x)) * self.w1(x)))


# ---------------------------------------------------------------------------
# Cyclic RoPE 1D — Rotary Position Embedding for closed contours
# ---------------------------------------------------------------------------
class CyclicRoPE1D(nn.Module):
    """1D Cyclic Rotary Position Embedding for closed contours.

    Position angles: θᵢ = 2π × i / P  (cyclic, matching Snake topology).
    Applied to Q and K in every attention layer (FLUX-style per-layer injection).

    Properties:
    - Naturally encodes that point_0 and point_{P-1} are adjacent
    - Encodes *relative* position (i-j) mod P → compatible with start-point alignment
    - No conflict with the contour alignment algorithm in pretrain_evolution.py

    References:
    - Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    - FLUX.1 (2024): RoPE applied per-layer to Q/K (vs SD3 input-only PE)
    """

    def __init__(self, head_dim: int, num_points: int = 128):
        super().__init__()
        self.head_dim = head_dim
        self.num_points = num_points
        # Frequency bands: geometric series 1 / 10000^(2i/d)
        freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('freqs', freqs)  # (head_dim // 2,)

    def _build_cache(self, P: int, device: torch.device):
        """Build rotation matrices for P cyclic positions."""
        # Cyclic positions: [0, 2π/P, 4π/P, ..., 2π(P-1)/P]
        positions = torch.arange(P, device=device, dtype=torch.float32) * (2.0 * math.pi / P)
        # Outer product: (P, head_dim // 2)
        angles = torch.outer(positions, self.freqs.to(device))
        return angles.cos(), angles.sin()

    def apply_rotary(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cyclic rotary embedding to tensor.

        Args:
            x: (N, num_heads, P, head_dim)
        Returns:
            Rotated x with same shape.
        """
        P = x.shape[2]
        cos_cache, sin_cache = self._build_cache(P, x.device)
        # Split into even/odd pairs
        x1 = x[..., 0::2]  # (N, H, P, head_dim//2)
        x2 = x[..., 1::2]
        # Broadcast: (1, 1, P, head_dim//2)
        c = cos_cache[:P].unsqueeze(0).unsqueeze(0)
        s = sin_cache[:P].unsqueeze(0).unsqueeze(0)
        # Rotate
        out1 = x1 * c - x2 * s
        out2 = x1 * s + x2 * c
        return torch.stack([out1, out2], dim=-1).flatten(-2)


# ---------------------------------------------------------------------------
# SeparatePointEmbedding — Avoids coord info being drowned by features
# ---------------------------------------------------------------------------
class SeparatePointEmbedding(nn.Module):
    """Separate embedding for coordinates and sampled features.

    Problem with V1: `cat(x_t[2], feat[64]) → Linear(66, 256)` causes
    the 2-dim coordinate to be "drowned" by the 64-dim features.

    Solution: Independent MLPs with dedicated capacity for each modality.
    Coordinate gets 1/4 of state_dim, feature gets 3/4.

    Reference: ContourFormer (CVPR 2025) — separate encoding strategy.
    """

    def __init__(self, state_dim: int = 256, feature_dim: int = 64):
        super().__init__()
        coord_dim = state_dim // 4  # 64
        feat_dim = state_dim - coord_dim  # 192

        self.coord_embed = nn.Sequential(
            nn.Linear(2, coord_dim),
            nn.SiLU(),
            nn.Linear(coord_dim, coord_dim),
        )
        self.feat_embed = nn.Sequential(
            nn.Linear(feature_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, x_t: torch.Tensor, sampled_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t:          (N, P, 2)  — noisy displacement / contour coordinates
            sampled_feat: (N, C, P)  — sampled local features (channel-first)
        Returns:
            (N, P, state_dim)
        """
        coord_emb = self.coord_embed(x_t)                       # (N, P, coord_dim)
        feat_emb = self.feat_embed(sampled_feat.transpose(1, 2))  # (N, P, feat_dim)
        return torch.cat([coord_emb, feat_emb], dim=-1)


# ---------------------------------------------------------------------------
# DiTBlockV2 — Full upgrade: RMSNorm + QK-Norm + CyclicRoPE + SwiGLU + 9-param adaLN
# ---------------------------------------------------------------------------
class DiTBlockV2(nn.Module):
    """Advanced DiT Block V2 with comprehensive upgrades.

    Improvements over V1:
      1. RMSNorm replaces LayerNorm (efficiency + stability)
      2. QK-RMSNorm prevents attention logit explosion
      3. CyclicRoPE injected per-layer on Q/K (FLUX-style)
      4. Cross-Attention now uses adaLN-Zero gate (time-conditional)
      5. SwiGLU replaces SiLU-MLP (better expressivity, same FLOPs)
      6. 9 modulation parameters: 3 for SA, 3 for CA, 3 for FFN
      7. Dropout=0 by default (DiT paper recommendation)

    Structure:
      1. adaLN → Self-Attention (QK-Norm + CyclicRoPE) → gated residual
      2. adaLN → Cross-Attention (QK-Norm) → gated residual
      3. adaLN → SwiGLU FFN → gated residual
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_points: int = 128,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # === Normalization (RMSNorm, no learned affine — adaLN provides it) ===
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)

        # === QK-Norm for attention stability ===
        self.qk_norm = RMSNorm(head_dim)

        # === Cyclic-RoPE ===
        self.rope = CyclicRoPE1D(head_dim=head_dim, num_points=num_points)

        # === Self-Attention (manual projections for RoPE + QK-Norm) ===
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.sa_out_proj = nn.Linear(dim, dim, bias=False)

        # === Cross-Attention ===
        self.cross_q_proj = nn.Linear(dim, dim, bias=False)
        self.cross_k_proj = nn.Linear(dim, dim, bias=False)
        self.cross_v_proj = nn.Linear(dim, dim, bias=False)
        self.ca_out_proj = nn.Linear(dim, dim, bias=False)

        # === SwiGLU FFN ===
        self.mlp = SwiGLU(dim=dim, dropout=dropout)

        # === adaLN-Zero: 9 parameters ===
        # (shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_ff, scale_ff, gate_ff)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim, bias=True),
        )
        # Zero-init for stable training (DiT paper: "blocks start as identity")
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention with QK-Norm and CyclicRoPE."""
        N, P, D = x.shape
        H = self.num_heads
        hd = self.head_dim

        q = self.q_proj(x).view(N, P, H, hd).transpose(1, 2)  # (N, H, P, hd)
        k = self.k_proj(x).view(N, P, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(N, P, H, hd).transpose(1, 2)

        # QK-Norm → CyclicRoPE
        q = self.rope.apply_rotary(self.qk_norm(q))
        k = self.rope.apply_rotary(self.qk_norm(k))

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(N, P, D)
        return self.sa_out_proj(out)

    def _cross_attention(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Cross-attention with QK-Norm (no RoPE on cross — different sequence)."""
        N, P, D = x.shape
        L = context.shape[1]
        H = self.num_heads
        hd = self.head_dim

        q = self.cross_q_proj(x).view(N, P, H, hd).transpose(1, 2)
        k = self.cross_k_proj(context).view(N, L, H, hd).transpose(1, 2)
        v = self.cross_v_proj(context).view(N, L, H, hd).transpose(1, 2)

        # QK-Norm (no RoPE on cross-attention — context has different positional semantics)
        q = self.qk_norm(q)
        k = self.qk_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(N, P, D)
        return self.ca_out_proj(out)

    def forward(
        self,
        x: torch.Tensor,
        image_context: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:             (N, P, dim) — Snake point features
            image_context: (N, L, dim) — Visual context (global or local)
            t_emb:         (N, dim)    — Time embedding
        Returns:
            (N, P, dim)
        """
        # --- Predict 9 modulation parameters ---
        mod = self.adaLN_modulation(t_emb)  # (N, 9 * dim)
        (shift_sa, scale_sa, gate_sa,
         shift_ca, scale_ca, gate_ca,
         shift_ff, scale_ff, gate_ff) = mod.chunk(9, dim=1)

        # --- 1. Self-Attention ---
        x_sa = modulate(self.norm1(x), shift_sa, scale_sa)
        x = x + gate_sa.unsqueeze(1) * self._self_attention(x_sa)

        # --- 2. Cross-Attention (time-conditioned gate — new in V2) ---
        x_ca = modulate(self.norm2(x), shift_ca, scale_ca)
        x = x + gate_ca.unsqueeze(1) * self._cross_attention(x_ca, image_context)

        # --- 3. SwiGLU FFN ---
        x_ff = modulate(self.norm3(x), shift_ff, scale_ff)
        x = x + gate_ff.unsqueeze(1) * self.mlp(x_ff)

        return x


# ---------------------------------------------------------------------------
# FinalLayer — adaLN-modulated output projection (DiT paper best practice)
# ---------------------------------------------------------------------------
class FinalLayer(nn.Module):
    """Final output layer with adaLN modulation.

    The DiT paper found that applying adaLN at the final layer matters.
    Zero-init on both the modulation and the output linear for stable training.
    """

    def __init__(self, dim: int, out_dim: int = 2):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )
        # Zero-init everything for identity-like startup
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(t_emb).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)
