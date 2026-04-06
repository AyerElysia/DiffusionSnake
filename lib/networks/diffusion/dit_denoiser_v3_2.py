"""
DiT Flow Matching V3.2 for Diffusion Snake

Based on V3.1 architecture (Self -> Cross -> FFN) with Flow Matching training.

Key Features:
  - Patchify for image tokens (ViT-style)
  - Self-Attention + Cross-Attention (no Joint Attention redundancy)
  - Flow Matching training objective (predict velocity v)
  - Alternating global/local context injection
  - ODE Solver for inference (10-20 steps)

Author: DiffSnake Team
Date: 2026-04-06
"""

import torch
import torch.nn as nn
from typing import Tuple

from .dit_blocks import SinusoidalTimeEmbedding
from .dit_blocks_v2 import SeparatePointEmbedding, FinalLayer
from .dit_blocks_v3_1 import PatchifyEmbedding, DiTBlockV3_1


class DiTFlowMatchingV3_2(nn.Module):
    """
    DiT Flow Matching V3.2 for Diffusion Snake.

    Architecture:
      - Self-Attention -> Cross-Attention -> FFN (efficient, no redundancy)
      - Patchify for global image tokens
      - Alternating global/local context

    Training: Predict velocity field v (Flow Matching)
    Inference: Euler ODE solver, 10-20 steps
    """

    def __init__(
        self,
        state_dim: int = 256,
        feature_dim: int = 64,
        num_layers: int = 6,
        num_heads: int = 8,
        time_dim: int = 256,
        num_points: int = 128,
        patch_size: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.time_dim = time_dim
        self.patch_size = patch_size

        # 1. Time Embedding
        self.time_emb_net = nn.Sequential(
            SinusoidalTimeEmbedding(dim=state_dim // 4),
            nn.Linear(state_dim // 4, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )

        # 2. Global Context: Patchify (ViT-style)
        self.image_embed = PatchifyEmbedding(
            in_channels=feature_dim,
            patch_size=patch_size,
            out_dim=state_dim,
        )

        # 3. Local Context: per-point sampled features projection
        self.local_proj = nn.Sequential(
            nn.Linear(feature_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )

        # 4. Separate Point Embedding
        self.point_embed = SeparatePointEmbedding(
            state_dim=state_dim,
            feature_dim=feature_dim,
        )

        # 5. DiT Blocks (Self -> Cross -> FFN)
        self.dit_layers = nn.ModuleList([
            DiTBlockV3_1(
                dim=state_dim,
                num_heads=num_heads,
                num_points=num_points,
                dropout=0.0,
            )
            for _ in range(num_layers)
        ])

        # 6. Final Layer (predict velocity v)
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
        Forward pass for Flow Matching.

        Args:
            cnn_feature:  (B, 64, H, W)  - Full P2 feature map
            sampled_feat: (N, 64, P)      - Sampled features at point locations
            x_t:          (N, P, 2)       - Current state on the flow path
            t:            (N,)            - Time in [0, 1] (scaled by 1000 for embedding)
            py_ind:       (N,)            - Maps each contour to its image index

        Returns:
            v_pred: (N, P, 2) - Predicted velocity field
            L:     scalar     - Auxiliary loss placeholder
        """
        N, P, _ = x_t.shape
        t_emb = self.time_emb_net(t)

        # Ensure cnn_feature is 4D [Batch, Channels, H, W]
        if cnn_feature.dim() == 3:
            cnn_feature = cnn_feature.unsqueeze(0)

        # Global context via Patchify
        global_ctx = self.image_embed(cnn_feature)

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

        # Predict velocity field
        v_pred = self.final_layer(x, t_emb)
        L = torch.zeros(1, device=x_t.device, dtype=x_t.dtype)
        return v_pred, L