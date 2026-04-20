"""
DiT Flow Matching V3.7 for Diffusion Snake — Per-Point Output Head.

The core insight: with a shared Linear(256->2) output layer, points with
nearly identical hidden states MUST produce nearly identical displacements.
On small contours, many nearby points share the same bilinear-interpolated
features -> same hidden states -> same predictions -> burrs/inaccuracy.

V3.7 replaces the shared output layer with per-point independent output heads.
Each of the 128 points gets its own Linear(256->2), enabling the model to
predict distinct displacements even when hidden representations are similar.

Author: Copilot / DiffSnake Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit_denoiser_v3_2 import DiTFlowMatchingV3_2
from .dit_blocks_v2 import RMSNorm, FinalLayer, modulate


class PerPointFinalLayer(nn.Module):
    """Per-point output heads with adaLN modulation.

    Each contour point has its own Linear(dim->out_dim) output mapping,
    enabling distinct predictions even for identical hidden states.
    Uses the same adaLN modulation as the original FinalLayer.
    """

    def __init__(self, dim: int = 256, out_dim: int = 2, num_points: int = 128,
                 use_float64: bool = False):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.num_points = num_points
        self.use_float64 = use_float64

        self.norm = RMSNorm(dim)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )

        # Per-point weight: (num_points, out_dim, dim) and bias: (num_points, out_dim)
        dtype = torch.float64 if use_float64 else torch.float32
        self.per_point_weight = nn.Parameter(torch.zeros(num_points, out_dim, dim, dtype=dtype))
        self.per_point_bias = nn.Parameter(torch.zeros(num_points, out_dim, dtype=dtype))

        # Zero-init for stable start
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)

    def init_from_shared(self, shared_final_layer: FinalLayer):
        """Initialize from a pretrained shared FinalLayer for warm start."""
        with torch.no_grad():
            shared_w = shared_final_layer.linear.weight.data  # (out_dim, dim)
            shared_b = shared_final_layer.linear.bias.data    # (out_dim,)
            dtype = self.per_point_weight.dtype
            for p in range(self.num_points):
                self.per_point_weight.data[p] = shared_w.clone().to(dtype)
                self.per_point_bias.data[p] = shared_b.clone().to(dtype)

            # Copy adaLN weights
            for src_p, dst_p in zip(shared_final_layer.adaLN.parameters(),
                                     self.adaLN.parameters()):
                dst_p.data.copy_(src_p.data)

            # Copy norm weights
            self.norm.weight.data.copy_(shared_final_layer.norm.weight.data)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, P, D) hidden states
            t_emb: (N, D) timestep embedding
        Returns:
            pred: (N, P, out_dim) — float64 if use_float64 else float32
        """
        shift, scale = self.adaLN(t_emb).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)  # (N, P, D)

        P = x.shape[1]
        w = self.per_point_weight[:P]
        b = self.per_point_bias[:P]

        if self.use_float64:
            # Cast hidden states to float64 for precision-critical final computation
            x = x.double()

        pred = torch.einsum('npd,pod->npo', x, w) + b.unsqueeze(0)
        return pred


class RegularizedPerPointFinalLayer(nn.Module):
    """Shared output + per-point delta corrections for generalization.

    pred = shared_linear(x) + delta_scale * delta_p(x)

    The shared component captures the generalizable mapping from hidden
    states to displacement. The per-point deltas learn fine corrections
    that differentiate nearby points with similar features (the root
    cause of burrs on small contours). L2 regularization on the delta
    weights prevents overfitting while preserving their disambiguating
    power.
    """

    def __init__(self, dim: int = 256, out_dim: int = 2,
                 num_points: int = 128, delta_scale: float = 0.1):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.num_points = num_points
        self.delta_scale = delta_scale

        # Shared components (same structure as FinalLayer)
        self.norm = RMSNorm(dim)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )
        self.shared_linear = nn.Linear(dim, out_dim)

        # Per-point delta corrections (zero-init → starts as pure shared)
        self.delta_weight = nn.Parameter(torch.zeros(num_points, out_dim, dim))
        self.delta_bias = nn.Parameter(torch.zeros(num_points, out_dim))

        # Zero-init adaLN for stable start
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)

    def init_from_shared(self, shared_final_layer: FinalLayer):
        """Initialize shared weights from a pretrained FinalLayer."""
        with torch.no_grad():
            self.shared_linear.weight.data.copy_(shared_final_layer.linear.weight.data)
            self.shared_linear.bias.data.copy_(shared_final_layer.linear.bias.data)
            for src_p, dst_p in zip(shared_final_layer.adaLN.parameters(),
                                     self.adaLN.parameters()):
                dst_p.data.copy_(src_p.data)
            self.norm.weight.data.copy_(shared_final_layer.norm.weight.data)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(t_emb).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)  # (N, P, D)

        P = x.shape[1]

        # Shared prediction (generalizable)
        shared_pred = self.shared_linear(x)  # (N, P, out_dim)

        # Per-point delta correction (small, regularized)
        w = self.delta_weight[:P]   # (P, out_dim, dim)
        b = self.delta_bias[:P]     # (P, out_dim)
        delta_pred = torch.einsum('npd,pod->npo', x, w) + b.unsqueeze(0)

        return shared_pred + self.delta_scale * delta_pred

    def delta_reg_loss(self) -> torch.Tensor:
        """L2 regularization on per-point delta parameters."""
        return self.delta_weight.pow(2).mean() + self.delta_bias.pow(2).mean()


class DiTFlowMatchingV3_7(DiTFlowMatchingV3_2):
    """Flow-Matching denoiser with per-point output heads."""

    def __init__(self, *args,
                 num_points: int = 128,
                 use_per_point_head: bool = True,
                 use_float64_head: bool = False,
                 use_regularized_per_point: bool = False,
                 delta_scale: float = 0.1,
                 delta_reg_weight: float = 0.001,
                 point_embed_scale: float = 0.1,
                 laplacian_weight: float = 0.0,
                 inject_at_input: bool = False,
                 inject_at_output: bool = False,
                 use_scale_conditioning: bool = False,
                 **kwargs):
        super().__init__(*args, num_points=num_points, **kwargs)

        self.num_points = num_points
        self.laplacian_weight = laplacian_weight
        self.delta_reg_weight = delta_reg_weight
        self.inject_at_input = inject_at_input
        self.inject_at_output = inject_at_output
        self.use_per_point_head = use_per_point_head
        self.use_scale_conditioning = use_scale_conditioning

        # Scale conditioning: embeds log(contour_scale) and adds to t_emb.
        # Zero-initialized output so it starts as identity (safe for warm start).
        if use_scale_conditioning:
            self.scale_embed_net = nn.Sequential(
                nn.Linear(1, self.state_dim),
                nn.SiLU(),
                nn.Linear(self.state_dim, self.state_dim),
            )
            nn.init.zeros_(self.scale_embed_net[-1].weight)
            nn.init.zeros_(self.scale_embed_net[-1].bias)

        if use_per_point_head:
            self._shared_final_layer = self.final_layer
            if use_regularized_per_point:
                self.final_layer = RegularizedPerPointFinalLayer(
                    dim=self.state_dim, out_dim=2, num_points=num_points,
                    delta_scale=delta_scale)
            else:
                self.final_layer = PerPointFinalLayer(
                    dim=self.state_dim, out_dim=2, num_points=num_points,
                    use_float64=use_float64_head)

        if inject_at_input:
            self.point_idx_embed_in = nn.Embedding(num_points, self.state_dim)
            nn.init.normal_(self.point_idx_embed_in.weight, std=point_embed_scale)

        if inject_at_output:
            self.point_idx_embed_out = nn.Embedding(num_points, self.state_dim)
            nn.init.normal_(self.point_idx_embed_out.weight, std=point_embed_scale)

        self.register_buffer('_point_indices',
                             torch.arange(num_points, dtype=torch.long))

    def init_per_point_from_checkpoint(self):
        """Initialize per-point heads from the loaded shared FinalLayer weights."""
        if self.use_per_point_head and hasattr(self, '_shared_final_layer'):
            self.final_layer.init_from_shared(self._shared_final_layer)
            del self._shared_final_layer

    def forward(
        self,
        cnn_feature: torch.Tensor,
        sampled_feat: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        adj=None,
        polys=None,
        py_ind: torch.Tensor = None,
        contour_scale: torch.Tensor = None,
    ) -> tuple:
        assert x_t.dim() == 3 and x_t.shape[-1] == 2
        assert t.dim() == 1
        assert sampled_feat.dim() == 3

        # Auto-cast inputs to match parameter dtype (supports full-denoiser float64)
        param_dtype = next(self.parameters()).dtype
        if t.dtype != param_dtype:
            t = t.to(param_dtype)
        if x_t.dtype != param_dtype:
            x_t = x_t.to(param_dtype)
        if sampled_feat.dtype != param_dtype:
            sampled_feat = sampled_feat.to(param_dtype)
        if cnn_feature.dtype != param_dtype:
            cnn_feature = cnn_feature.to(param_dtype)

        N, P, _ = x_t.shape
        t_emb = self.time_emb_net(t)

        # Scale conditioning: add log(contour_scale) embedding to t_emb
        if self.use_scale_conditioning and contour_scale is not None:
            log_scale = torch.log(contour_scale.view(-1, 1).to(param_dtype) + 1e-6)
            t_emb = t_emb + self.scale_embed_net(log_scale)

        if cnn_feature.dim() == 3:
            cnn_feature = cnn_feature.unsqueeze(0)
        global_ctx = self.image_embed(cnn_feature)

        if py_ind is not None:
            global_ctx = global_ctx[py_ind]
        elif global_ctx.shape[0] != N:
            if global_ctx.shape[0] == 1:
                global_ctx = global_ctx.expand(N, -1, -1)
            else:
                raise ValueError(
                    f"Batch mismatch: global_ctx={global_ctx.shape[0]}, N={N}"
                )

        local_ctx = self.local_proj(sampled_feat.transpose(1, 2))
        x = self.point_embed(x_t, sampled_feat)

        if self.inject_at_input:
            point_ids = self._point_indices[:P].unsqueeze(0).expand(N, -1)
            x = x + self.point_idx_embed_in(point_ids)

        for i, dit_layer in enumerate(self.dit_layers):
            context = global_ctx if (i % 2 == 0) else local_ctx
            x = dit_layer(x, context, t_emb)

        if self.inject_at_output:
            point_ids = self._point_indices[:P].unsqueeze(0).expand(N, -1)
            x = x + self.point_idx_embed_out(point_ids)

        pred = self.final_layer(x, t_emb)

        L = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        if self.training:
            if self.laplacian_weight > 0:
                prev_pt = torch.roll(pred, 1, dims=1)
                next_pt = torch.roll(pred, -1, dims=1)
                laplacian = pred - (prev_pt + next_pt) * 0.5
                L = L + self.laplacian_weight * laplacian.pow(2).mean()
            if self.delta_reg_weight > 0 and hasattr(self.final_layer, 'delta_reg_loss'):
                L = L + self.delta_reg_weight * self.final_layer.delta_reg_loss()

        return pred, L
