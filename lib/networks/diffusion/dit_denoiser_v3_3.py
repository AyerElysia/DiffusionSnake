"""
DiT Denoiser V3.3 for Diffusion Snake
Based on V3 with Circular Convolution for smoothness

Key Features:
  [M1] Self -> Cross Attention Flow (from V3)
  [M2] Perceiver IO for global semantic compression (from V3)
  [M3] Separate Point Embeddings & Cyclic-RoPE (from V3)
  [M4] Circular Conv 1D before final layer (NEW in V3.3)
       - Enhances neighborhood consistency
       - Prevents outlier vertices from flying away

Author: DiffSnake Team
Date: 2026-04-16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .dit_blocks import SinusoidalTimeEmbedding, PerceiverCompressor
from .dit_blocks_v2 import SeparatePointEmbedding, FinalLayer
from .dit_blocks_v3 import DiTBlockV3


class CircularConv1d(nn.Module):
    """
    1D Convolution with circular padding for closed contours.

    This ensures that the first and last points are treated as neighbors,
    which is essential for closed contour modeling.
    """
    def __init__(self, dim: int, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=0)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, P, D) - contour features
        Returns:
            x: (N, P, D) - smoothed features
        """
        # Transpose to (N, D, P) for Conv1d
        x_t = x.transpose(1, 2)

        # Circular padding
        pad_size = self.kernel_size // 2
        x_padded = F.pad(x_t, (pad_size, pad_size), mode='circular')

        # Apply convolution
        x_conv = self.conv(x_padded)

        # Transpose back to (N, P, D)
        x_out = x_conv.transpose(1, 2)

        # Layer norm for stability
        x_out = self.norm(x_out)

        # Residual keeps the original contour geometry and lets the conv act as a refinement.
        return x + 0.1 * x_out


class DiTDenoiserV3_3(nn.Module):
    """
    DiT Denoiser V3.3 for Diffusion Snake.

    Extends V3 with Circular Convolution before the final prediction layer
    to enhance contour smoothness and prevent outlier vertices.
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
        circular_conv_kernel: int = 5,  # NEW parameter
        **kwargs,
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

        # 6. Circular Convolution (NEW in V3.3)
        self.circular_conv = CircularConv1d(
            dim=state_dim,
            kernel_size=circular_conv_kernel
        )

        # 7. Final Layer
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
    ) -> torch.Tensor:
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
        """
        # Input validation
        assert x_t.dim() == 3 and x_t.shape[-1] == 2, \
            f"Expected x_t shape (N, P, 2), got {x_t.shape}"
        assert t.dim() == 1, f"Expected t shape (N,), got {t.shape}"
        assert sampled_feat.dim() == 3, \
            f"Expected sampled_feat shape (N, C, P), got {sampled_feat.shape}"

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

        # Apply Circular Convolution for smoothness (NEW in V3.3)
        x = self.circular_conv(x)

        # Final output
        pred = self.final_layer(x, t_emb)
        L = torch.zeros(1, device=x_t.device, dtype=x_t.dtype)
        return pred, L
