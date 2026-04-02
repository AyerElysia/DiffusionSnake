"""
DiT Denoiser V2 for Diffusion Snake

Complete upgraded denoiser integrating all V2 improvements:
  [M1] Multi-scale visual features (Global Perceiver + Local per-point)
  [M2] Separate point embedding (coord/feat independent MLPs)
  [M3] Cyclic-RoPE (per-layer, closed contour topology)
  [M4] DiTBlockV2 (RMSNorm + QK-Norm + SwiGLU + 9-param adaLN)
  [M6] Final adaLN output head

Interface: 100% compatible with V1 DiTDenoiser — same forward() signature.

References:
  - DiT (ICCV 2023), FLUX.1 (2024), SiT (2024), DyDiT (2024)
  - SwiGLU (Shazeer 2020), QK-Norm (2024), RMSNorm (2019)
  - Point Transformer V3 (CVPR 2024), ContourFormer (CVPR 2025)

Author: DiffSnake Team
Date: 2026-04-02
"""

import torch
import torch.nn as nn
from typing import Tuple

from .dit_blocks import SinusoidalTimeEmbedding, PerceiverCompressor
from .dit_blocks_v2 import (
    SeparatePointEmbedding,
    DiTBlockV2,
    FinalLayer,
)


class DiTDenoiserV2(nn.Module):
    """Advanced DiT Denoiser V2 for Diffusion Snake.

    Upgrades over V1:
      [M1] Multi-scale visual context — retains both global Perceiver
           compressed features AND local per-point sampled features.
           Odd layers use global context (shape semantics), even layers
           use local context (boundary-level details).
      [M2] Separate point embedding — coordinates and features processed
           by independent MLPs to avoid coordinate drowning.
      [M3] CyclicRoPE per-layer — superior to additive SnakePosEncoding;
           naturally handles closed contour topology and is compatible
           with start-point alignment.
      [M4] DiTBlockV2 — RMSNorm, QK-Norm, SwiGLU, 9-param adaLN-Zero
           (SA + CA + FFN all get time-conditioned gates).
      [M6] Final adaLN output head — time-aware output projection.
    """

    def __init__(
        self,
        state_dim: int = 256,
        feature_dim: int = 64,
        num_layers: int = 6,
        num_heads: int = 8,
        time_dim: int = 256,
        num_points: int = 128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.time_dim = time_dim

        # === 1. Time Embedding (reuse V1's proven design) ===
        self.time_emb_net = nn.Sequential(
            SinusoidalTimeEmbedding(dim=state_dim // 4),
            nn.Linear(state_dim // 4, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )

        # === 2. Multi-Scale Visual Encoder [M1] ===
        # Global path: Perceiver compression → shape-level semantics
        self.global_compressor = PerceiverCompressor(
            in_dim=feature_dim,
            out_dim=state_dim,
            num_queries=256,
        )
        # Local path: project sampled features → boundary-level details
        self.local_proj = nn.Sequential(
            nn.Linear(feature_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )

        # === 3. Separate Point Embedding [M2] ===
        self.point_embed = SeparatePointEmbedding(
            state_dim=state_dim,
            feature_dim=feature_dim,
        )

        # === 4. DiT Blocks V2 [M3 + M4] ===
        self.dit_layers = nn.ModuleList([
            DiTBlockV2(
                dim=state_dim,
                num_heads=num_heads,
                num_points=num_points,
                dropout=0.0,
            )
            for _ in range(num_layers)
        ])

        # === 5. Final Layer [M6] ===
        self.final_layer = FinalLayer(dim=state_dim, out_dim=2)

    def forward(
        self,
        cnn_feature: torch.Tensor,
        sampled_feat: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        adj: torch.Tensor = None,
        polys=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass — interface identical to V1 DiTDenoiser.

        Args:
            cnn_feature:  (N, 64, H, W)  — Full P2 feature map
            sampled_feat: (N, 64, P)      — Sampled features at point locations
            x_t:          (N, P, 2)       — Noisy displacement/contour
            t:            (N,)            — Timesteps
            adj:          Unused (kept for interface compatibility)
            polys:        Unused (kept for interface compatibility)

        Returns:
            eps_pred: (N, P, 2) — Predicted noise (or velocity)
            L:        scalar    — Auxiliary loss placeholder
        """
        N, P, _ = x_t.shape

        # 1. Time Embedding
        t_emb = self.time_emb_net(t)  # (N, state_dim)

        # 2. Multi-Scale Visual Context [M1]
        global_ctx = self.global_compressor(cnn_feature)  # (N, 256, state_dim)
        local_ctx = self.local_proj(
            sampled_feat.transpose(1, 2)
        )  # (N, P, state_dim)

        # 3. Point Embedding [M2]
        x = self.point_embed(x_t, sampled_feat)  # (N, P, state_dim)

        # 4. Process through DiT Blocks V2 [M3 + M4]
        # Alternate between global and local context for cross-attention
        for i, dit_layer in enumerate(self.dit_layers):
            context = global_ctx if (i % 2 == 0) else local_ctx
            x = dit_layer(x, context, t_emb)

        # 5. Final output [M6]
        pred = self.final_layer(x, t_emb)  # (N, P, 2)

        L = torch.zeros(1, device=x_t.device, dtype=x_t.dtype)
        return pred, L
