"""
DiT Denoiser V3 for Diffusion Snake
Evolutionary Dynamic Network

Key Features:
  [M1] Self -> Cross Attention Flow (standard Transformer decoder order)
  [M2] Perceiver IO for global semantic compression (256 learnable queries)
  [M3] Separate Point Embeddings & Cyclic-RoPE
  [M4] Octagon Initialization support (handled externally in snake_decode.py)

Global Context: 256 learnable query tokens cross-attend to H*W image tokens.
Local Context: Per-point sampled features from grid_sample.

Author: DiffSnake Team
Date: 2026-04-02
"""

import torch
import torch.nn as nn
from typing import Tuple

from .dit_blocks import SinusoidalTimeEmbedding, PerceiverCompressor
from .dit_blocks_v2 import SeparatePointEmbedding, FinalLayer
from .dit_blocks_v3 import DiTBlockV3


class DiTDenoiserV3(nn.Module):
    """
    DiT Denoiser V3 for Diffusion Snake.

    Uses Perceiver IO (learnable queries) for global semantic compression.
    Attention flow: Self-Attention -> Cross-Attention -> FFN (standard decoder order).
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
        **kwargs,  # Accept but ignore extra kwargs for compatibility
    ):
        super().__init__()
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.time_dim = time_dim

        # 1. Time Embedding
        self.time_emb_net = nn.Sequential(
            SinusoidalTimeEmbedding(dim=state_dim // 4),
            nn.Linear(state_dim // 4, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )

        # 2. Global Context: Perceiver IO (256 learnable queries)
        self.global_compressor = PerceiverCompressor(
            in_dim=feature_dim,
            out_dim=state_dim,
            num_queries=num_queries,
        )

        # 3. Local Context: per-point sampled features projection
        self.local_proj = nn.Sequential(
            nn.Linear(feature_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )

        # 4. Separate Point Embedding
        self.point_embed = SeparatePointEmbedding(
            state_dim=state_dim, feature_dim=feature_dim
        )

        # 5. DiT Blocks V3 (Self -> Cross Flow)
        self.dit_layers = nn.ModuleList([
            DiTBlockV3(
                dim=state_dim,
                num_heads=num_heads,
                num_points=num_points,
                dropout=0.0,
            )
            for _ in range(num_layers)
        ])

        # 6. Final Layer
        self.final_layer = FinalLayer(dim=state_dim, out_dim=2)

    def forward(
        self,
        cnn_feature: torch.Tensor,
        sampled_feat: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        adj=None,
        polys=None,
        py_ind: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            cnn_feature:  (B, 64, H, W)  - Full P2 feature map
            sampled_feat: (N, 64, P)      - Sampled features at point locations
            x_t:          (N, P, 2)       - Noisy displacement
            t:            (N,)            - Timesteps
            py_ind:       (N,)            - Maps each contour to its image index

        Returns:
            eps_pred: (N, P, 2) - Predicted noise
            L:       scalar     - Auxiliary loss placeholder
        """
        N, P, _ = x_t.shape
        t_emb = self.time_emb_net(t)

        # Ensure cnn_feature is 4D [Batch, Channels, H, W]
        if cnn_feature.dim() == 3:
            cnn_feature = cnn_feature.unsqueeze(0)

        # Global context via Perceiver (256 learnable queries)
        global_ctx = self.global_compressor(cnn_feature)

        # Expand global_ctx from Image batch (B) to Contour batch (N)
        if py_ind is not None:
            global_ctx = global_ctx[py_ind]
        elif global_ctx.shape[0] != N:
            if global_ctx.shape[0] == 1:
                global_ctx = global_ctx.expand(N, -1, -1)
            else:
                raise ValueError(
                    f"Batch dimension mismatch: global_ctx={global_ctx.shape[0]}, N={N}"
                )

        # Local context from sampled features
        local_ctx = self.local_proj(sampled_feat.transpose(1, 2))

        # Point embedding
        x = self.point_embed(x_t, sampled_feat)

        # Process through DiT Blocks (alternating global/local context)
        for i, dit_layer in enumerate(self.dit_layers):
            context = global_ctx if (i % 2 == 0) else local_ctx
            x = dit_layer(x, context, t_emb)

        # Final output
        pred = self.final_layer(x, t_emb)
        L = torch.zeros(1, device=x_t.device, dtype=x_t.dtype)
        return pred, L