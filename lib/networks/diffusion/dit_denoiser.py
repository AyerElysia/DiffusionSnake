"""
DiT Denoiser for Diffusion Snake

纯 DiT 架构的去噪器，完全基于 Transformer 的全局建模

作者: DiffSnake Team
日期: 2026-03-11
"""

import torch
import torch.nn as nn
from typing import Tuple
from .dit_blocks import (
    SinusoidalTimeEmbedding,
    SnakePosEncoding,
    PerceiverCompressor,
    DiTBlock
)


class DiTDenoiser(nn.Module):
    """
    Advanced DiT Denoiser for Diffusion Snake (MMDiT-inspired)

    Core Design:
    1. Perceiver Global Compression: Extracts features from the entire P2 map.
    2. Snake Positional Encoding: Maintains contour topology.
    3. adaLN-Zero Blocks: Performs time-conditioned denoising in every layer.
    4. Residual Local Features: Combines sampled local features with global context.
    """
    def __init__(
        self,
        state_dim: int = 256,
        feature_dim: int = 64,
        num_layers: int = 6,
        num_heads: int = 8,
        time_dim: int = 256,
        num_points: int = 128
    ):
        super().__init__()

        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.time_dim = time_dim

        # === 1. Time Embedding ===
        self.time_emb_net = nn.Sequential(
            SinusoidalTimeEmbedding(dim=state_dim // 4),
            nn.Linear(state_dim // 4, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )

        # === 2. Global Compression (Perceiver) ===
        # Compresses (64, H, W) -> (256, state_dim)
        self.visual_encoder = PerceiverCompressor(
            in_dim=feature_dim,
            out_dim=state_dim,
            num_queries=256
        )

        # === 3. Point & Positional Embedding ===
        self.point_proj = nn.Linear(2 + feature_dim, state_dim)
        self.pos_enc = SnakePosEncoding(dim=state_dim, num_points=num_points)

        # === 4. DiT Blocks (adaLN-Zero) ===
        self.dit_layers = nn.ModuleList([
            DiTBlock(
                dim=state_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])

        # === 5. Output Head ===
        self.output_proj = nn.Sequential(
            nn.LayerNorm(state_dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(state_dim, 2)
        )
        # Initialize output projection to zero (for delta prediction stability)
        nn.init.constant_(self.output_proj[-1].weight, 0)
        nn.init.constant_(self.output_proj[-1].bias, 0)

    def forward(
        self,
        cnn_feature: torch.Tensor,
        sampled_feat: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        adj: torch.Tensor = None,
        polys=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cnn_feature: Full P2 feature map, (N, 64, H, W)
            sampled_feat: Sampled features at point locations, (N, 64, P)
            x_t: Noisy displacement/contour, (N, P, 2)
            t: Timesteps, (N,)
        """
        N, P, _ = x_t.shape

        # 1. Time Embedding
        t_emb = self.time_emb_net(t)  # (N, state_dim)

        # 2. Global Feature Compression
        global_context = self.visual_encoder(cnn_feature)  # (N, 256, state_dim)

        # 3. Point Embedding (Local Context + Coords + PosEncoding)
        # Combine local sampled features with coordinates
        x = torch.cat([x_t, sampled_feat.transpose(1, 2)], dim=-1)  # (N, P, 2 + 64)
        x = self.point_proj(x)  # (N, P, state_dim)
        x = self.pos_enc(x)     # Inject topology information

        # 4. Process through DiT Blocks
        for dit_layer in self.dit_layers:
            x = dit_layer(x, global_context, t_emb)

        # 5. Predict Noise/Epsilon
        eps_pred = self.output_proj(x)  # (N, P, 2)

        L = torch.zeros(1, device=x_t.device, dtype=x_t.dtype)
        return eps_pred, L
