"""
DiT Denoiser V3.5 for Diffusion Snake — Fourier-Space Diffusion

Instead of predicting 128-point displacement directly, V3.5 operates in
Fourier space: the denoiser predicts noise on K Fourier coefficients,
and IFFT reconstructs the 128-point displacement.

Key changes from V3:
  - Sequence length is K (Fourier coefficients) instead of 128 (points)
  - Output dimension is 4 (real+imag for x,y) instead of 2
  - Learnable frequency embeddings replace point embeddings
  - CNN features are aggregated from 128 points via a pooling bridge

This guarantees smooth output by construction: only K low-frequency
components can be represented, making high-frequency jitter impossible.

Author: DiffSnake Team
Date: 2026-04-16
"""

import torch
import torch.nn as nn
from typing import Tuple

from .dit_blocks import SinusoidalTimeEmbedding, PerceiverCompressor
from .dit_blocks_v2 import FinalLayer, RMSNorm, modulate
from .dit_blocks_v3 import DiTBlockV3


class FourierPointBridge(nn.Module):
    """Bridge from 128-point spatial features to K frequency tokens.

    Takes per-point sampled CNN features (N, C, 128) and produces
    K frequency-aware tokens (N, K, state_dim) via cross-attention.
    """

    def __init__(self, feature_dim: int = 64, state_dim: int = 256, num_freq: int = 16):
        super().__init__()
        self.num_freq = num_freq
        # Learnable frequency queries
        self.freq_queries = nn.Parameter(torch.randn(num_freq, state_dim) * 0.02)
        # Project point features to state_dim
        self.point_proj = nn.Sequential(
            nn.Linear(feature_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )
        # Cross-attention: freq queries attend to point features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=state_dim, num_heads=8, batch_first=True
        )
        self.norm_q = RMSNorm(state_dim)
        self.norm_kv = RMSNorm(state_dim)

    def forward(self, sampled_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sampled_feat: (N, C, P) — per-point CNN features (P=128)
        Returns:
            (N, K, state_dim) — frequency tokens
        """
        N = sampled_feat.shape[0]
        # Project point features: (N, P, state_dim)
        kv = self.point_proj(sampled_feat.transpose(1, 2))
        # Expand freq queries: (N, K, state_dim)
        q = self.freq_queries.unsqueeze(0).expand(N, -1, -1)
        # Cross-attention
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        out, _ = self.cross_attn(q_norm, kv_norm, kv_norm)
        return out + q  # Residual connection


class FourierCoeffEmbedding(nn.Module):
    """Embed noisy Fourier coefficients into state_dim tokens.

    Input x_t has shape (N, K, 4) where 4 = (real_x, imag_x, real_y, imag_y).
    Also incorporates frequency index information via learnable embeddings.
    """

    def __init__(self, state_dim: int = 256, num_freq: int = 16):
        super().__init__()
        coord_dim = state_dim // 4
        feat_dim = state_dim - coord_dim

        self.coeff_embed = nn.Sequential(
            nn.Linear(4, coord_dim),
            nn.SiLU(),
            nn.Linear(coord_dim, coord_dim),
        )
        # Learnable frequency position embedding
        self.freq_pos_embed = nn.Embedding(num_freq, feat_dim)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: (N, K, 4) — noisy Fourier coefficients
        Returns:
            (N, K, state_dim)
        """
        N, K, _ = x_t.shape
        coeff_emb = self.coeff_embed(x_t)  # (N, K, coord_dim)
        freq_idx = torch.arange(K, device=x_t.device)
        pos_emb = self.freq_pos_embed(freq_idx).unsqueeze(0).expand(N, -1, -1)  # (N, K, feat_dim)
        return torch.cat([coeff_emb, pos_emb], dim=-1)


class FourierFinalLayer(nn.Module):
    """Final output layer for Fourier coefficients.

    Outputs (N, K, 4) — real/imag parts for x,y displacement Fourier coefficients.
    Zero-initialized for stable training start.
    """

    def __init__(self, dim: int, out_dim: int = 4):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )
        # Zero-init for identity-like startup
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(t_emb).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class DiTDenoiserV3_5(nn.Module):
    """
    DiT Denoiser V3.5 — Fourier-Space Diffusion.

    The denoiser operates on K Fourier coefficient tokens instead of 128 point tokens.
    This guarantees smooth output by construction.

    Architecture:
        1. FourierPointBridge: 128-point CNN features → K frequency tokens (local context)
        2. PerceiverCompressor: Full feature map → 256 global tokens (global context)
        3. FourierCoeffEmbedding: Noisy Fourier coefficients → K state tokens
        4. DiTBlockV3 layers: Self-Attn → Cross-Attn with alternating global/local context
        5. FourierFinalLayer: → K × 4 (real/imag for x,y)
    """

    def __init__(
        self,
        state_dim: int = 256,
        feature_dim: int = 64,
        num_layers: int = 6,
        num_heads: int = 8,
        time_dim: int = 256,
        num_points: int = 128,
        num_queries: int = 256,
        num_fourier_k: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.num_fourier_k = num_fourier_k
        self.num_points = num_points

        # 1. Time Embedding (same as V3)
        self.time_emb_net = nn.Sequential(
            SinusoidalTimeEmbedding(dim=state_dim // 4),
            nn.Linear(state_dim // 4, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )

        # 2. Global Context: Perceiver IO (same as V3)
        self.global_compressor = PerceiverCompressor(
            in_dim=feature_dim,
            out_dim=state_dim,
            num_queries=num_queries,
        )

        # 3. Local Context: Bridge from 128 points → K frequency tokens
        self.fourier_bridge = FourierPointBridge(
            feature_dim=feature_dim,
            state_dim=state_dim,
            num_freq=num_fourier_k,
        )

        # 4. Fourier Coefficient Embedding
        self.coeff_embed = FourierCoeffEmbedding(
            state_dim=state_dim,
            num_freq=num_fourier_k,
        )

        # 5. DiT Blocks V3 (Self → Cross Flow)
        # Note: num_points=num_fourier_k since sequence length is K, not 128
        self.dit_layers = nn.ModuleList([
            DiTBlockV3(
                dim=state_dim,
                num_heads=num_heads,
                num_points=num_fourier_k,
                dropout=0.0,
            )
            for _ in range(num_layers)
        ])

        # 6. Final Layer → K × 4
        self.final_layer = FourierFinalLayer(dim=state_dim, out_dim=4)

    def forward(
        self,
        cnn_feature: torch.Tensor,
        sampled_feat: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        adj=None,
        polys=None,
        py_ind: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            cnn_feature:  (B, 64, H, W)  — Full P2 feature map
            sampled_feat: (N, 64, P)      — Sampled features at 128 point locations
            x_t:          (N, K, 4)       — Noisy Fourier coefficients
            t:            (N,)            — Timesteps
            py_ind:       (N,)            — Maps each contour to its image index

        Returns:
            eps_pred: (N, K, 4) — Predicted noise in Fourier space
        """
        assert x_t.dim() == 3 and x_t.shape[-1] == 4, \
            f"Expected x_t shape (N, K, 4), got {x_t.shape}"
        assert x_t.shape[1] == self.num_fourier_k, \
            f"Expected K={self.num_fourier_k}, got {x_t.shape[1]}"

        N = x_t.shape[0]
        t_emb = self.time_emb_net(t)

        # Ensure cnn_feature is 4D
        if cnn_feature.dim() == 3:
            cnn_feature = cnn_feature.unsqueeze(0)

        # Global context via Perceiver
        global_ctx = self.global_compressor(cnn_feature)
        if py_ind is not None:
            global_ctx = global_ctx[py_ind]
        elif global_ctx.shape[0] != N:
            if global_ctx.shape[0] == 1:
                global_ctx = global_ctx.expand(N, -1, -1)
            else:
                raise ValueError(
                    f"Batch mismatch: global_ctx={global_ctx.shape[0]}, N={N}"
                )

        # Local context: bridge from 128 point features → K frequency tokens
        local_ctx = self.fourier_bridge(sampled_feat)  # (N, K, state_dim)

        # Fourier coefficient embedding
        x = self.coeff_embed(x_t)  # (N, K, state_dim)

        # Process through DiT Blocks (alternating global/local context)
        for i, dit_layer in enumerate(self.dit_layers):
            context = global_ctx if (i % 2 == 0) else local_ctx
            x = dit_layer(x, context, t_emb)

        # Final output
        pred = self.final_layer(x, t_emb)  # (N, K, 4)
        L = torch.zeros(1, device=x_t.device, dtype=x_t.dtype)
        return pred, L
