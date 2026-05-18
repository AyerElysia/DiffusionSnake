"""V4.1 flow-matching denoiser.

V4.1 keeps the proven V3.4 detail backbone for checkpoint reuse and adds only a
zero-init per-point residual head for small local contour corrections.
"""

from .dit_denoiser_v3_4 import DiTFlowMatchingV3_4
from .dit_denoiser_v4 import LatentLoopBlock, MoEFinalHead, PerPointDeltaHead, StrongPerPointDeltaHead


class DiTFlowMatchingV4_1(DiTFlowMatchingV3_4):
    """V3.4 detail adapter plus a conservative per-point delta head."""

    def __init__(
        self,
        *args,
        num_points: int = 128,
        use_per_point_delta: bool = True,
        per_point_delta_scale: float = 0.10,
        per_point_delta_reg_weight: float = 0.0,
        per_point_delta_head_type: str = 'linear',
        per_point_delta_hidden_mult: float = 2.0,
        per_point_delta_use_cyclic_mixer: bool = True,
        final_head_type: str = 'standard',
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_balance_weight: float = 1e-3,
        moe_expert_init_std: float = 1e-4,
        moe_router_noise_std: float = 0.01,
        moe_use_point_embed: bool = True,
        moe_use_cyclic_router: bool = True,
        moe_use_shared_expert: bool = False,
        moe_routed_expert_scale: float = 1.0,
        moe_expert_type: str = 'linear',
        moe_expert_hidden_dim: int = 256,
        use_latent_loop: bool = False,
        latent_loop_steps: int = 4,
        **kwargs,
    ):
        super().__init__(*args, num_points=num_points, **kwargs)
        self.num_points = int(num_points)
        self.final_head_type = str(final_head_type).strip().lower()
        self.use_moe_final_head = self.final_head_type in ('moe', 'moe_final', 'deepseek_moe')
        self.use_per_point_delta = bool(use_per_point_delta) and not self.use_moe_final_head
        self.per_point_delta_head_type = str(per_point_delta_head_type).strip().lower()
        self.use_latent_loop = bool(use_latent_loop)
        self.latent_loop_steps = int(max(1, latent_loop_steps))

        if self.use_moe_final_head:
            self.final_layer = MoEFinalHead(
                dim=self.state_dim,
                out_dim=2,
                num_points=self.num_points,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                balance_weight=moe_balance_weight,
                expert_init_std=moe_expert_init_std,
                router_noise_std=moe_router_noise_std,
                use_point_embed=moe_use_point_embed,
                use_cyclic_router=moe_use_cyclic_router,
                use_shared_expert=moe_use_shared_expert,
                routed_expert_scale=moe_routed_expert_scale,
                expert_type=moe_expert_type,
                expert_hidden_dim=moe_expert_hidden_dim,
            )

        if self.use_latent_loop:
            self.latent_loop = LatentLoopBlock(
                dim=self.state_dim,
                num_heads=getattr(self.dit_layers[0], 'num_heads', 8),
                num_points=self.num_points,
                dropout=0.0,
            )

        if self.use_per_point_delta:
            if self.per_point_delta_head_type in ('strong', 'mlp', 'local'):
                self.per_point_delta_head = StrongPerPointDeltaHead(
                    dim=self.state_dim,
                    out_dim=2,
                    num_points=self.num_points,
                    delta_scale=per_point_delta_scale,
                    reg_weight=per_point_delta_reg_weight,
                    hidden_mult=per_point_delta_hidden_mult,
                    use_cyclic_mixer=per_point_delta_use_cyclic_mixer,
                )
            else:
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

        if self.use_latent_loop:
            for _ in range(self.latent_loop_steps):
                x = self.latent_loop(x, t_emb)

        pred = self.final_layer(x, t_emb)
        reg_loss = pred.new_zeros(())
        if hasattr(self.final_layer, 'reg_loss'):
            reg_loss = reg_loss + self.final_layer.reg_loss().to(pred.device, pred.dtype)
        if self.use_per_point_delta:
            pred = pred + self.per_point_delta_head(x, t_emb)
            reg_loss = reg_loss + self.per_point_delta_head.reg_loss().to(pred.device, pred.dtype)
        return pred, reg_loss
