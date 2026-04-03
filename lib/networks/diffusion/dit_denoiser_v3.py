"""
DiT Denoiser V3 for Diffusion Snake
Evolutionary Dynamic Network

Upgrades over V2.0:
  [M1] Re-aligned Attention Flow (Cross -> Self) for better geometric synergy
  [M2] Retains robust Perceiver/Anchor logic for image contexts
  [M3] Retains Separate Point Embeddings & Cyclic-RoPE

Author: DiffSnake Team
Date: 2026-04-02
"""

import torch
import torch.nn as nn
from typing import Tuple

from .dit_blocks import SinusoidalTimeEmbedding, PerceiverCompressor
from .dit_blocks_v2 import SeparatePointEmbedding, FinalLayer, SpatialAnchorCompressor
from .dit_blocks_v3 import DiTBlockV3

class DiTDenoiserV3(nn.Module):
    """
    Advanced DiT Denoiser V3.
    Re-aligns the attention flow and optionally uses Octagon Initialization.
    Does not use MM-DiT Length-Agnostic Joint Attention (keeps sequence lengths static).
    """

    def __init__(
        self,
        state_dim: int = 256,
        feature_dim: int = 64,
        num_layers: int = 6,
        num_heads: int = 8,
        time_dim: int = 256,
        num_points: int = 128,
        use_v2_1: bool = False,
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

        # 2. Multi-Scale Visual Encoder
        if use_v2_1:
            self.global_compressor = SpatialAnchorCompressor(
                in_dim=feature_dim, out_dim=state_dim, anchor_size=16
            )
        else:
            self.global_compressor = PerceiverCompressor(
                in_dim=feature_dim, out_dim=state_dim, num_queries=256
            )
            
        self.local_proj = nn.Sequential(
            nn.Linear(feature_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )

        # 3. Separate Point Embedding
        self.point_embed = SeparatePointEmbedding(
            state_dim=state_dim, feature_dim=feature_dim
        )

        # 4. DiT Blocks V3 (Cross -> Self Flow)
        self.dit_layers = nn.ModuleList([
            DiTBlockV3(
                dim=state_dim,
                num_heads=num_heads,
                num_points=num_points,
                dropout=0.0,
            )
            for _ in range(num_layers)
        ])

        # 5. Final Layer
        self.final_layer = FinalLayer(dim=state_dim, out_dim=2)

    def forward(
        self, cnn_feature, sampled_feat, x_t, t, adj=None, polys=None, py_ind=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        N, P, _ = x_t.shape
        t_emb = self.time_emb_net(t)

        # Ensure cnn_feature is 4D [Batch, Channels, H, W]
        if cnn_feature.dim() == 3:
            cnn_feature = cnn_feature.unsqueeze(0)

        global_ctx = self.global_compressor(cnn_feature)

        if py_ind is not None:
            global_ctx = global_ctx[py_ind]
        elif global_ctx.shape[0] != N:
            if global_ctx.shape[0] == 1:
                global_ctx = global_ctx.expand(N, -1, -1)
            else:
                raise ValueError(f"Batch dimension mismatch: global_ctx={global_ctx.shape[0]}, N={N}")
                
        local_ctx = self.local_proj(sampled_feat.transpose(1, 2))
        x = self.point_embed(x_t, sampled_feat)

        # Process through V3 Blocks (alternating contexts)
        for i, dit_layer in enumerate(self.dit_layers):
            context = global_ctx if (i % 2 == 0) else local_ctx
            x = dit_layer(x, context, t_emb)

        pred = self.final_layer(x, t_emb)
        L = torch.zeros(1, device=x_t.device, dtype=x_t.dtype)
        return pred, L
