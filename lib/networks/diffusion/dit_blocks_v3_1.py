"""
DiT Blocks V3.1 for Diffusion Snake

Based on V3 blocks, adapted for Patchify image tokens.

Components:
  1. PatchifyEmbedding - ViT-style patch embedding
  2. DiTBlockV3_1 - Same as V3 (Self -> Cross -> FFN)

Author: DiffSnake Team
Date: 2026-04-06
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .dit_blocks_v2 import (
    RMSNorm,
    SwiGLU,
    CyclicRoPE1D,
    modulate,
)


# ---------------------------------------------------------------------------
# Patchify Embedding (ViT-style, from V2.2)
# ---------------------------------------------------------------------------
class PatchifyEmbedding(nn.Module):
    """
    Standard ViT-style Patchify for image features.
    No overlapping patches. Converts HxW into (H/p * W/p) tokens.
    Adds absolute 2D sine-cosine positional embedding.

    For 128x128 feature map with patch_size=8: 16x16 = 256 patches
    """

    def __init__(
        self,
        in_channels: int = 64,
        patch_size: int = 8,
        out_dim: int = 256,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_dim = out_dim

        # Projection: equivalent to non-overlapping convolutions
        self.proj = nn.Conv2d(in_channels, out_dim, kernel_size=patch_size, stride=patch_size)

        # Maximum grid size (assume feature map <= 128x128, patch_size=8 -> max 16x16=256)
        max_grid = 16
        self.pos_embed = nn.Parameter(torch.zeros(1, max_grid * max_grid, out_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] image feature map
        Returns:
            [B, num_patches, out_dim] patch tokens
        """
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, out_dim, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, out_dim]

        # Add positional embedding
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]
        return x


# ---------------------------------------------------------------------------
# DiT Block V3.1 (Same as V3: Self -> Cross -> FFN)
# ---------------------------------------------------------------------------
class DiTBlockV3_1(nn.Module):
    """
    DiT Block V3.1.

    Same structure as V3:
    1. Self-Attention (with CyclicRoPE)
    2. Cross-Attention (with image context)
    3. FFN (SwiGLU)

    Uses adaLN-Zero with 9 modulation parameters.
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

        # adaLN-Zero: 9 parameters
        # (shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_ff, scale_ff, gate_ff)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def _self_attention(self, x: torch.Tensor, point_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Self-attention with QK-Norm and CyclicRoPE.

        Args:
            x: (N, P, D) point features
            point_mask: (N, P) V3.10: point validity mask (1=valid, 0=padding)
        """
        N, P, D = x.shape
        H = self.num_heads
        hd = self.head_dim

        q = self.q_proj(x).view(N, P, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(N, P, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(N, P, H, hd).transpose(1, 2)

        q = self.rope.apply_rotary(self.qk_norm(q))
        k = self.rope.apply_rotary(self.qk_norm(k))

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # V3.10: Apply attention mask if provided
        if point_mask is not None:
            # point_mask: [N, P] where 1=valid, 0=padding
            # Create attention mask: [N, 1, 1, P] for broadcasting to [N, H, P, P]
            # We want to mask out attention TO padding positions (columns)
            attn_mask = point_mask.unsqueeze(1).unsqueeze(2)  # [N, 1, 1, P]
            # Mask out invalid positions with large negative value
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)

        # Handle NaN from softmax of all -inf (all padding row)
        if point_mask is not None:
            attn = torch.nan_to_num(attn, nan=0.0)

        out = (attn @ v).transpose(1, 2).contiguous().view(N, P, D)
        return self.sa_out_proj(out)

    def _cross_attention(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Cross-attention with QK-Norm (no RoPE on cross)."""
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

    def forward(
        self,
        x: torch.Tensor,
        image_context: torch.Tensor,
        t_emb: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,  # V3.10: add point mask
    ) -> torch.Tensor:
        """
        Args:
            x:             (N, P, dim) - Snake point features
            image_context: (N, L, dim) - Image patch tokens (from Patchify)
            t_emb:         (N, dim)    - Time embedding
            point_mask:    (N, P)      - V3.10: Point validity mask (1=valid, 0=padding)

        Returns:
            (N, P, dim)
        """
        mod = self.adaLN_modulation(t_emb)
        (shift_sa, scale_sa, gate_sa,
         shift_ca, scale_ca, gate_ca,
         shift_ff, scale_ff, gate_ff) = mod.chunk(9, dim=1)

        # 1. Self-Attention (with point mask)
        x_sa = modulate(self.norm1(x), shift_sa, scale_sa)
        x = x + gate_sa.unsqueeze(1) * self._self_attention(x_sa, point_mask=point_mask)

        # 2. Cross-Attention
        x_ca = modulate(self.norm2(x), shift_ca, scale_ca)
        x = x + gate_ca.unsqueeze(1) * self._cross_attention(x_ca, image_context)

        # 3. FFN
        x_ff = modulate(self.norm3(x), shift_ff, scale_ff)
        x = x + gate_ff.unsqueeze(1) * self.mlp(x_ff)

        return x