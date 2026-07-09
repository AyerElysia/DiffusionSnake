"""
DiT Flow Matching V3.4 adapter.

Keeps the original V3 backbone and accepts the FM wrapper's extra
arguments so V3.4 can be trained with Flow Matching without changing
the core backbone behavior.
"""

import torch
import torch.nn as nn

from .dit_denoiser_v3 import DiTDenoiserV3


class DiTFlowMatchingV3_4(DiTDenoiserV3):
    """V3.4 FM adapter with optional zero-init detail context."""

    def __init__(
        self,
        *args,
        use_detail_context: bool = False,
        detail_feature_dim: int = 192,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_detail_context = bool(use_detail_context)
        self.detail_feature_dim = int(detail_feature_dim)

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
        return pred, reg_loss


