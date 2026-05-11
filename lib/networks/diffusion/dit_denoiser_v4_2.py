"""V4.2 flow-matching denoiser.

V4.2 extends the V4.1 recipe with curvature-aware conditioning and a gated
per-point delta head, while keeping the V3.4-compatible trunk for reuse.
"""

import torch
import torch.nn as nn

from .dit_blocks_v2 import RMSNorm
from .dit_denoiser_v3_4 import DiTFlowMatchingV3_4
from .dit_denoiser_v4 import PerPointDeltaHead


class GatedPerPointDeltaHead(PerPointDeltaHead):
    def __init__(
        self,
        dim: int = 256,
        out_dim: int = 2,
        num_points: int = 128,
        delta_scale: float = 0.10,
        reg_weight: float = 0.0,
        gate_bias: float = -2.0,
    ):
        super().__init__(
            dim=dim,
            out_dim=out_dim,
            num_points=num_points,
            delta_scale=delta_scale,
            reg_weight=reg_weight,
        )
        self.gate = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )
        nn.init.zeros_(self.gate[1].weight)
        nn.init.zeros_(self.gate[1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        delta = super().forward(x, t_emb)
        gate = torch.sigmoid(self.gate(x))
        return gate * delta


class DiTFlowMatchingV4_2(DiTFlowMatchingV3_4):
    def __init__(
        self,
        *args,
        num_points: int = 128,
        use_detail_context: bool = False,
        detail_feature_dim: int = 320,
        use_per_point_delta: bool = True,
        per_point_delta_scale: float = 0.10,
        per_point_delta_reg_weight: float = 0.0,
        use_curvature_conditioning: bool = True,
        curvature_embed_scale: float = 0.10,
        use_delta_gate: bool = True,
        delta_gate_bias: float = -2.0,
        **kwargs,
    ):
        super().__init__(
            *args,
            num_points=num_points,
            use_detail_context=use_detail_context,
            detail_feature_dim=detail_feature_dim,
            **kwargs,
        )
        self.num_points = int(num_points)
        self.use_per_point_delta = bool(use_per_point_delta)
        self.use_curvature_conditioning = bool(use_curvature_conditioning)
        self.curvature_embed_scale = float(curvature_embed_scale)

        if self.use_curvature_conditioning:
            self.curvature_local_proj = nn.Sequential(
                nn.Linear(1, self.state_dim),
                nn.SiLU(),
                nn.Linear(self.state_dim, self.state_dim),
            )
            self.curvature_point_proj = nn.Sequential(
                nn.Linear(1, self.state_dim),
                nn.SiLU(),
                nn.Linear(self.state_dim, self.state_dim),
            )
            nn.init.zeros_(self.curvature_local_proj[-1].weight)
            nn.init.zeros_(self.curvature_local_proj[-1].bias)
            nn.init.zeros_(self.curvature_point_proj[-1].weight)
            nn.init.zeros_(self.curvature_point_proj[-1].bias)

        if self.use_per_point_delta:
            head_cls = GatedPerPointDeltaHead if use_delta_gate else PerPointDeltaHead
            head_kwargs = dict(
                dim=self.state_dim,
                out_dim=2,
                num_points=self.num_points,
                delta_scale=per_point_delta_scale,
                reg_weight=per_point_delta_reg_weight,
            )
            if use_delta_gate:
                head_kwargs['gate_bias'] = delta_gate_bias
            self.per_point_delta_head = head_cls(**head_kwargs)

    @staticmethod
    def _compute_curvature(polys: torch.Tensor) -> torch.Tensor:
        prev_pt = torch.roll(polys, 1, dims=1)
        next_pt = torch.roll(polys, -1, dims=1)
        curvature = (prev_pt - 2.0 * polys + next_pt).norm(dim=-1, keepdim=True)
        return curvature / (curvature.amax(dim=1, keepdim=True) + 1e-6)

    def forward(
        self,
        cnn_feature,
        sampled_feat,
        x_t,
        t,
        adj=None,
        polys=None,
        py_ind=None,
        contour_scale=None,
        detail_feat=None,
        x_self_cond=None,
    ):
        param_dtype = next(self.parameters()).dtype
        if cnn_feature.dtype != param_dtype:
            cnn_feature = cnn_feature.to(param_dtype)
        sampled_feat = sampled_feat.to(param_dtype)
        x_t = x_t.to(param_dtype)
        t = t.to(param_dtype)
        if detail_feat is not None and detail_feat.dtype != param_dtype:
            detail_feat = detail_feat.to(param_dtype)
        t_emb = self.time_emb_net(t)

        n_contours, _, _ = x_t.shape
        if cnn_feature.dim() == 3:
            cnn_feature = cnn_feature.unsqueeze(0)

        global_ctx = self.global_compressor(cnn_feature)
        if py_ind is not None:
            global_ctx = global_ctx[py_ind]
        elif global_ctx.shape[0] != n_contours:
            if global_ctx.shape[0] == 1:
                global_ctx = global_ctx.expand(n_contours, -1, -1)
            else:
                raise ValueError(f"Batch dimension mismatch: global_ctx={global_ctx.shape[0]}, N={n_contours}")

        local_ctx = self.local_proj(sampled_feat.transpose(1, 2))
        x = self.point_embed(x_t, sampled_feat)

        if self.use_detail_context and detail_feat is not None:
            detail_ctx = detail_feat.transpose(1, 2)
            local_ctx = local_ctx + self.detail_local_proj(detail_ctx)
            x = x + self.detail_point_proj(detail_ctx)

        if self.use_curvature_conditioning and polys is not None:
            curvature = self._compute_curvature(polys.to(dtype=param_dtype))
            local_ctx = local_ctx + self.curvature_embed_scale * self.curvature_local_proj(curvature)
            x = x + self.curvature_embed_scale * self.curvature_point_proj(curvature)

        for i, dit_layer in enumerate(self.dit_layers):
            context = global_ctx if (i % 2 == 0) else local_ctx
            x = dit_layer(x, context, t_emb)

        pred = self.final_layer(x, t_emb)
        reg_loss = pred.new_zeros(())

        if self.use_per_point_delta:
            pred = pred + self.per_point_delta_head(x, t_emb)
            reg_loss = reg_loss + self.per_point_delta_head.reg_loss().to(pred.device, pred.dtype)
        return pred, reg_loss