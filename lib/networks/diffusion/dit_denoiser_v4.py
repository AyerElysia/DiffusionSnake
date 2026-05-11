"""V4.0 flow-matching denoiser with multi-scale detail context.

This keeps the V3.4 backbone intact for checkpoint reuse, then adds two
zero-init residual improvements:
1. detail-context fusion fed by the FM wrapper's local detail sampler
2. a per-point delta head on top of the shared final head
"""

import torch
import torch.nn as nn

from .dit_blocks_v2 import RMSNorm, modulate
from .dit_denoiser_v3 import DiTDenoiserV3


class PerPointDeltaHead(nn.Module):
    """Zero-init residual per-point head for local shape disambiguation."""

    def __init__(
        self,
        dim: int = 256,
        out_dim: int = 2,
        num_points: int = 128,
        delta_scale: float = 0.25,
        reg_weight: float = 0.0,
    ):
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.reg_weight = float(reg_weight)
        self.norm = RMSNorm(dim)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )
        self.delta_weight = nn.Parameter(torch.zeros(num_points, out_dim, dim))
        self.delta_bias = nn.Parameter(torch.zeros(num_points, out_dim))

        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(t_emb).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        num_points = x.shape[1]
        delta_weight = self.delta_weight[:num_points]
        delta_bias = self.delta_bias[:num_points]
        delta = torch.einsum('npd,pod->npo', x, delta_weight) + delta_bias.unsqueeze(0)
        return self.delta_scale * delta

    def reg_loss(self) -> torch.Tensor:
        if self.reg_weight <= 0:
            return self.delta_weight.new_zeros(())
        reg = self.delta_weight.pow(2).mean() + self.delta_bias.pow(2).mean()
        return self.reg_weight * reg


class DiTFlowMatchingV4(DiTDenoiserV3):
    """V4.0 adapter: V3.4 warm start + detail fusion + per-point delta head."""

    def __init__(
        self,
        *args,
        num_points: int = 128,
        use_detail_context: bool = False,
        detail_feature_dim: int = 192,
        use_per_point_delta: bool = True,
        per_point_delta_scale: float = 0.25,
        per_point_delta_reg_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, num_points=num_points, **kwargs)
        self.num_points = int(num_points)
        self.use_detail_context = bool(use_detail_context)
        self.detail_feature_dim = int(detail_feature_dim)
        self.use_per_point_delta = bool(use_per_point_delta)

        if self.use_detail_context:
            self.detail_local_proj = nn.Sequential(
                nn.Linear(self.detail_feature_dim, self.state_dim),
                nn.SiLU(),
                nn.Linear(self.state_dim, self.state_dim),
            )
            self.detail_point_proj = nn.Sequential(
                nn.Linear(self.detail_feature_dim, self.state_dim),
                nn.SiLU(),
                nn.Linear(self.state_dim, self.state_dim),
            )
            nn.init.zeros_(self.detail_local_proj[-1].weight)
            nn.init.zeros_(self.detail_local_proj[-1].bias)
            nn.init.zeros_(self.detail_point_proj[-1].weight)
            nn.init.zeros_(self.detail_point_proj[-1].bias)

        if self.use_per_point_delta:
            self.per_point_delta_head = PerPointDeltaHead(
                dim=self.state_dim,
                out_dim=2,
                num_points=self.num_points,
                delta_scale=per_point_delta_scale,
                reg_weight=per_point_delta_reg_weight,
            )

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
        assert x_t.dim() == 3 and x_t.shape[-1] == 2, \
            f"Expected x_t shape (N, P, 2), got {x_t.shape}"
        assert t.dim() == 1, f"Expected t shape (N,), got {t.shape}"
        assert sampled_feat.dim() == 3, \
            f"Expected sampled_feat shape (N, C, P), got {sampled_feat.shape}"

        param_dtype = next(self.parameters()).dtype
        if cnn_feature.dtype != param_dtype:
            cnn_feature = cnn_feature.to(param_dtype)
        if sampled_feat.dtype != param_dtype:
            sampled_feat = sampled_feat.to(param_dtype)
        if x_t.dtype != param_dtype:
            x_t = x_t.to(param_dtype)
        if t.dtype != param_dtype:
            t = t.to(param_dtype)
        if detail_feat is not None and detail_feat.dtype != param_dtype:
            detail_feat = detail_feat.to(param_dtype)

        n_contours, _, _ = x_t.shape
        t_emb = self.time_emb_net(t)

        if cnn_feature.dim() == 3:
            cnn_feature = cnn_feature.unsqueeze(0)

        global_ctx = self.global_compressor(cnn_feature)
        if py_ind is not None:
            global_ctx = global_ctx[py_ind]
        elif global_ctx.shape[0] != n_contours:
            if global_ctx.shape[0] == 1:
                global_ctx = global_ctx.expand(n_contours, -1, -1)
            else:
                raise ValueError(
                    f"Batch dimension mismatch: global_ctx={global_ctx.shape[0]}, N={n_contours}"
                )

        local_ctx = self.local_proj(sampled_feat.transpose(1, 2))
        x = self.point_embed(x_t, sampled_feat)

        if self.use_detail_context and detail_feat is not None:
            detail_ctx = detail_feat.transpose(1, 2)
            local_ctx = local_ctx + self.detail_local_proj(detail_ctx)
            x = x + self.detail_point_proj(detail_ctx)

        for i, dit_layer in enumerate(self.dit_layers):
            context = global_ctx if (i % 2 == 0) else local_ctx
            x = dit_layer(x, context, t_emb)

        pred = self.final_layer(x, t_emb)
        reg_loss = torch.zeros(1, device=x_t.device, dtype=x_t.dtype)
        if self.use_per_point_delta:
            pred = pred + self.per_point_delta_head(x, t_emb)
            reg_loss = reg_loss + self.per_point_delta_head.reg_loss().to(x_t.device, x_t.dtype)
        return pred, reg_loss