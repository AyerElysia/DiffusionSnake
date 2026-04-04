"""
DiT Blocks V3 for Diffusion Snake
Evolutionary Dynamic Network Components.

Key Upgrades from V2:
- Reversed Attention Flow (Cross-Attention -> Self-Attention).
  First locates the boundaries using image context, then coordinates
  the contour geometry globally among the 128 points.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .dit_blocks_v2 import (
    RMSNorm,
    SwiGLU,
    CyclicRoPE1D,
    modulate
)

class DiTBlockV3(nn.Module):
    """
    Advanced DiT Block V3
    Sequence: Cross-Attention (Find Edges) -> Self-Attention (Smooth & Coordinate) -> Local Conv -> FFN
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

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)
        
        self.qk_norm = RMSNorm(head_dim)
        self.rope = CyclicRoPE1D(head_dim=head_dim, num_points=num_points)

        # Cross-Attention
        self.cross_q_proj = nn.Linear(dim, dim, bias=False)
        self.cross_k_proj = nn.Linear(dim, dim, bias=False)
        self.cross_v_proj = nn.Linear(dim, dim, bias=False)
        self.ca_out_proj = nn.Linear(dim, dim, bias=False)

        # Self-Attention
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.sa_out_proj = nn.Linear(dim, dim, bias=False)

        # SwiGLU FFN
        self.mlp = SwiGLU(dim=dim, dropout=dropout)

        # Local Topology Enhancer: Circular Depth-wise Conv (K=3)
        # Bridges the gap between Global Attention and Local Geometry.
        self.local_smooth = nn.Conv1d(
            dim, dim, kernel_size=3, padding=1, 
            groups=dim, padding_mode='circular', bias=False
        )


        # adaLN-Zero: 9 parameters
        # (shift_ca, scale_ca, gate_ca, shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def _cross_attention(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        N, P, D = x.shape
        L = context.shape[1]
        H = self.num_heads
        hd = self.head_dim

        q = self.cross_q_proj(x).view(N, P, H, hd).transpose(1, 2)
        k = self.cross_k_proj(context).view(N, L, H, hd).transpose(1, 2)
        v = self.cross_v_proj(context).view(N, L, H, hd).transpose(1, 2)

        q = self.qk_norm(q)
        k = self.qk_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(N, P, D)
        return self.ca_out_proj(out)

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        N, P, D = x.shape
        H = self.num_heads
        hd = self.head_dim

        q = self.q_proj(x).view(N, P, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(N, P, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(N, P, H, hd).transpose(1, 2)

        q = self.rope.apply_rotary(self.qk_norm(q))
        k = self.rope.apply_rotary(self.qk_norm(k))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(N, P, D)
        return self.sa_out_proj(out)

    def forward(self, x: torch.Tensor, image_context: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(t_emb)
        (shift_ca, scale_ca, gate_ca,
         shift_sa, scale_sa, gate_sa,
         shift_ff, scale_ff, gate_ff) = mod.chunk(9, dim=1)

        # V3 Design: Cross-Attention first (locate boundaries), then Self-Attention (coordinate geometry)
        # --- 1. Cross-Attention (Sense Image Boundaries) ---
        x_ca = modulate(self.norm1(x), shift_ca, scale_ca)
        x = x + gate_ca.unsqueeze(1) * self._cross_attention(x_ca, image_context)

        # --- 2. Self-Attention (Coordinate Internally) ---
        x_sa = modulate(self.norm2(x), shift_sa, scale_sa)
        x = x + gate_sa.unsqueeze(1) * self._self_attention(x_sa)

        # 3. Local Smoothing & SwiGLU FFN
        x_ff = modulate(self.norm3(x), shift_ff, scale_ff)
        x_ff_smooth = self.local_smooth(x_ff.transpose(1, 2)).transpose(1, 2)
        x = x + gate_ff.unsqueeze(1) * self.mlp(x_ff_smooth)

        return x
