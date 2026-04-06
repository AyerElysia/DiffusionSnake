"""
DiT Blocks V3 for Diffusion Snake (Rollback Version 3.1)
--------------------------------------------------------------
Key Upgrades:
- RESTORED Attention Flow (Self-Attention -> Cross-Attention).
- REMOVED Local Smoothing Conv (Redundant with Self-Attention).
- RETAINS Octagon Initialization (External Decoders).
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
    Advanced DiT Block V3 (Reverted design to V2 matching Flow Matching stability)
    Sequence: Self-Attention (Coordinate) -> Cross-Attention (Image) -> FFN
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

        # Normalization
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)
        
        # QK-Norm & RoPE
        self.qk_norm = RMSNorm(head_dim)
        self.rope = CyclicRoPE1D(head_dim=head_dim, num_points=num_points)

        # Self-Attention
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.sa_out_proj = nn.Linear(dim, dim, bias=False)

        # Cross-Attention
        self.cross_q_proj = nn.Linear(dim, dim, bias=False)
        self.cross_k_proj = nn.Linear(dim, dim, bias=False)
        self.cross_v_proj = nn.Linear(dim, dim, bias=False)
        self.ca_out_proj = nn.Linear(dim, dim, bias=False)

        # MLP
        self.mlp = SwiGLU(dim=dim, dropout=dropout)

        # adaLN-Zero: 9 parameters (Back to V2 Order)
        # (shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_ff, scale_ff, gate_ff)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

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

    def forward(self, x: torch.Tensor, image_context: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(t_emb)
        (shift_sa, scale_sa, gate_sa,
         shift_ca, scale_ca, gate_ca,
         shift_ff, scale_ff, gate_ff) = mod.chunk(9, dim=1)

        # 1. Self-Attention (Coordinate internally first - SAME AS V2)
        x_sa = modulate(self.norm1(x), shift_sa, scale_sa)
        x = x + gate_sa.unsqueeze(1) * self._self_attention(x_sa)

        # 2. Cross-Attention (Interact with Image Context - SAME AS V2)
        x_ca = modulate(self.norm2(x), shift_ca, scale_ca)
        x = x + gate_ca.unsqueeze(1) * self._cross_attention(x_ca, image_context)

        # 3. FFN (No local_smooth - SAME AS V2)
        x_ff = modulate(self.norm3(x), shift_ff, scale_ff)
        x = x + gate_ff.unsqueeze(1) * self.mlp(x_ff)

        return x
