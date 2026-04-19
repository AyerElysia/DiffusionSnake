"""
DiT Flow Matching V3.7 for Diffusion Snake — Anti-Burr Enhancement.

V3.7 inherits the V3.0 global semantic backbone (Perceiver, adaLN-zero)
and adds three anti-burr mechanisms:

  1. **Circular 1D Convolution Smoothing** in hidden space before the
     final projection.  Enforces local spatial coherence along the closed
     contour so that adjacent points predict consistent velocities.

  2. **Laplacian Regularization** on the predicted velocity field.
     Penalises second-order differences (zigzag patterns) and is
     returned as the auxiliary loss *L* so the training wrapper can
     incorporate it.

  3. **Learnable Smoothing Gate** — a scalar sigmoid gate that controls
     how much of the circular-conv smoothed hidden state is blended
     with the raw DiT output.  Initialised to favour smoothing.

Author: Copilot / DiffSnake Team
Date: 2026-04-19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit_denoiser_v3 import DiTDenoiserV3


# -------------------------------------------------------------------
# Circular 1D Convolution — handles closed-contour topology
# -------------------------------------------------------------------
class CircularConv1d(nn.Module):
    """Conv1d with circular (wrap-around) padding for closed contours."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, bias: bool = True):
        super().__init__()
        self.pad_size = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, P)
        x = F.pad(x, (self.pad_size, self.pad_size), mode='circular')
        return self.conv(x)


# -------------------------------------------------------------------
# V3.7 Denoiser
# -------------------------------------------------------------------
class DiTFlowMatchingV3_7(DiTDenoiserV3):
    """Flow-Matching denoiser with built-in anti-burr smoothing.

    Parameters
    ----------
    smooth_kernel_size : int
        Kernel width for circular conv smoothing layers (default 9).
    num_smooth_layers : int
        Number of circular conv blocks (default 2).
    laplacian_weight : float
        Coefficient for the Laplacian regularisation loss (default 0.1).
    """

    def __init__(self, *args,
                 smooth_kernel_size: int = 9,
                 num_smooth_layers: int = 2,
                 laplacian_weight: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.laplacian_weight = laplacian_weight

        # Build circular-conv smoothing stack
        layers = []
        for i in range(num_smooth_layers):
            layers.append(
                CircularConv1d(self.state_dim, self.state_dim,
                               kernel_size=smooth_kernel_size)
            )
            if i < num_smooth_layers - 1:
                layers.append(nn.SiLU())
        self.smooth_layers = nn.Sequential(*layers)

        # Learnable gate — init > 0 so smoothing is strong at start
        # sigmoid(2.0) ≈ 0.88
        self.smooth_gate = nn.Parameter(torch.tensor(2.0))

    # ----------------------------------------------------------------
    def forward(
        self,
        cnn_feature: torch.Tensor,
        sampled_feat: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        adj=None,
        polys=None,
        py_ind: torch.Tensor = None,
    ) -> tuple:
        """Forward pass — identical to V3.0 except for the smoothing
        layer and Laplacian regularisation appended at the end."""

        assert x_t.dim() == 3 and x_t.shape[-1] == 2, \
            f"Expected x_t (N,P,2), got {x_t.shape}"
        assert t.dim() == 1, f"Expected t (N,), got {t.shape}"
        assert sampled_feat.dim() == 3, \
            f"Expected sampled_feat (N,C,P), got {sampled_feat.shape}"

        N, P, _ = x_t.shape
        t_emb = self.time_emb_net(t)

        # ---- global context via Perceiver (inherited from V3.0) ----
        if cnn_feature.dim() == 3:
            cnn_feature = cnn_feature.unsqueeze(0)
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

        # ---- local context ----
        local_ctx = self.local_proj(sampled_feat.transpose(1, 2))

        # ---- point embedding ----
        x = self.point_embed(x_t, sampled_feat)

        # ---- DiT transformer blocks (alternating global / local) ----
        for i, dit_layer in enumerate(self.dit_layers):
            context = global_ctx if (i % 2 == 0) else local_ctx
            x = dit_layer(x, context, t_emb)

        # ---- V3.7: circular-conv smoothing in hidden space ----
        x_perm = x.permute(0, 2, 1)                 # (N, D, P)
        x_smooth = self.smooth_layers(x_perm)        # (N, D, P)
        x_smooth = x_smooth.permute(0, 2, 1)         # (N, P, D)

        gate = torch.sigmoid(self.smooth_gate)        # scalar in (0, 1)
        x = (1.0 - gate) * x + gate * x_smooth

        # ---- final projection to (N, P, 2) ----
        pred = self.final_layer(x, t_emb)

        # ---- Laplacian regularisation (training only) ----
        if self.training and self.laplacian_weight > 0:
            prev_pt = torch.roll(pred, 1, dims=1)
            next_pt = torch.roll(pred, -1, dims=1)
            laplacian = pred - (prev_pt + next_pt) * 0.5
            L = self.laplacian_weight * laplacian.pow(2).mean()
        else:
            L = torch.zeros(1, device=pred.device, dtype=pred.dtype)

        return pred, L
