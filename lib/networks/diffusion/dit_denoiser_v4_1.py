"""V4.1 flow-matching denoiser.

V4.1 keeps the proven V3.4 detail backbone for checkpoint reuse and adds only a
zero-init per-point residual head for small local contour corrections.
"""

import torch
import torch.nn as nn

from .dit_blocks import SinusoidalTimeEmbedding
from .dit_denoiser_v3_4 import DiTFlowMatchingV3_4
from .dit_denoiser_v4 import (
    DenseResidualFinalHead,
    LatentLoopBlock,
    ModernSparseResidualHead,
    MoEFinalHead,
    PerPointDeltaHead,
    SharedDenseSparseResidualHead,
    StrongPerPointDeltaHead,
)


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
        moe_balance_mode: str = 'legacy',
        moe_hard_phi_ema_decay: float = 0.99,
        moe_expert_init_std: float = 1e-4,
        moe_router_noise_std: float = 0.01,
        moe_use_point_embed: bool = True,
        moe_use_cyclic_router: bool = True,
        moe_use_shared_expert: bool = False,
        moe_route_shared_expert: bool = False,
        moe_route_shared_init_bias: float = 0.0,
        moe_routed_expert_scale: float = 1.0,
        moe_expert_type: str = 'linear',
        moe_expert_hidden_dim: int = 256,
        dense_residual_hidden_dim: int = 1024,
        shared_sparse_num_experts: int = 4,
        shared_sparse_expert_hidden_dim: int = 128,
        shared_sparse_router_temperature: float = 0.50,
        shared_sparse_load_ema_decay: float = 0.99,
        shared_sparse_balance_bias_step: float = 1e-3,
        shared_sparse_balance_bias_limit: float = 0.10,
        shared_sparse_expert_scale: float = 1.0,
        modern_output_num_experts: int = 4,
        modern_output_top_k: int = 2,
        modern_output_hidden_dim: int = 256,
        modern_output_router_temperature: float = 0.20,
        modern_output_balance_weight: float = 1e-3,
        modern_output_phi_ema_decay: float = 0.99,
        modern_output_contrastive_weight: float = 1e-3,
        modern_output_expert_init_std: float = 1e-4,
        use_latent_loop: bool = False,
        latent_loop_steps: int = 4,
        use_s_conditioning: bool = False,
        **kwargs,
    ):
        super().__init__(*args, num_points=num_points, **kwargs)
        self.num_points = int(num_points)
        self.final_head_type = str(final_head_type).strip().lower()
        self.use_moe_final_head = self.final_head_type in ('moe', 'moe_final', 'deepseek_moe')
        self.use_dense_residual_final_head = self.final_head_type in (
            'dense_residual', 'dense_residual_mlp'
        )
        self.use_shared_sparse_final_head = self.final_head_type in (
            'shared_sparse', 'shared_sparse_residual', 'dense_sparse_residual'
        )
        self.use_modern_sparse_final_head = self.final_head_type in (
            'modern_moe', 'modern_sparse_moe', 'contour_moe'
        )
        self.use_custom_final_head = (
            self.use_moe_final_head
            or self.use_dense_residual_final_head
            or self.use_shared_sparse_final_head
            or self.use_modern_sparse_final_head
        )
        self.use_per_point_delta = bool(use_per_point_delta) and not self.use_custom_final_head
        self.per_point_delta_head_type = str(per_point_delta_head_type).strip().lower()
        self.use_latent_loop = bool(use_latent_loop)
        self.latent_loop_steps = int(max(1, latent_loop_steps))

        if use_s_conditioning:
            self.s_emb_net = nn.Sequential(
                SinusoidalTimeEmbedding(dim=self.state_dim // 4),
                nn.Linear(self.state_dim // 4, self.state_dim),
                nn.SiLU(),
                nn.Linear(self.state_dim, self.state_dim),
            )
            nn.init.zeros_(self.s_emb_net[-1].weight)
            nn.init.zeros_(self.s_emb_net[-1].bias)

        if self.use_moe_final_head:
            self.final_layer = MoEFinalHead(
                dim=self.state_dim,
                out_dim=2,
                num_points=self.num_points,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                balance_weight=moe_balance_weight,
                balance_mode=moe_balance_mode,
                hard_phi_ema_decay=moe_hard_phi_ema_decay,
                expert_init_std=moe_expert_init_std,
                router_noise_std=moe_router_noise_std,
                use_point_embed=moe_use_point_embed,
                use_cyclic_router=moe_use_cyclic_router,
                use_shared_expert=moe_use_shared_expert,
                route_shared_expert=moe_route_shared_expert,
                route_shared_init_bias=moe_route_shared_init_bias,
                routed_expert_scale=moe_routed_expert_scale,
                expert_type=moe_expert_type,
                expert_hidden_dim=moe_expert_hidden_dim,
            )
        elif self.use_dense_residual_final_head:
            self.final_layer = DenseResidualFinalHead(
                dim=self.state_dim,
                out_dim=2,
                hidden_dim=dense_residual_hidden_dim,
                residual_init_std=modern_output_expert_init_std,
            )
        elif self.use_shared_sparse_final_head:
            self.final_layer = SharedDenseSparseResidualHead(
                dim=self.state_dim,
                out_dim=2,
                shared_hidden_dim=dense_residual_hidden_dim,
                num_experts=shared_sparse_num_experts,
                expert_hidden_dim=shared_sparse_expert_hidden_dim,
                router_temperature=shared_sparse_router_temperature,
                load_ema_decay=shared_sparse_load_ema_decay,
                balance_bias_step=shared_sparse_balance_bias_step,
                balance_bias_limit=shared_sparse_balance_bias_limit,
                expert_scale=shared_sparse_expert_scale,
            )
        elif self.use_modern_sparse_final_head:
            self.final_layer = ModernSparseResidualHead(
                dim=self.state_dim,
                out_dim=2,
                num_experts=modern_output_num_experts,
                top_k=modern_output_top_k,
                expert_hidden_dim=modern_output_hidden_dim,
                router_temperature=modern_output_router_temperature,
                balance_weight=modern_output_balance_weight,
                phi_ema_decay=modern_output_phi_ema_decay,
                contrastive_weight=modern_output_contrastive_weight,
                expert_init_std=modern_output_expert_init_std,
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
        locate_point_ctx=None,
        locate_global_ctx=None,
        locate_only: bool = False,
        s: torch.Tensor = None,
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
        if s is not None and s.dtype != param_dtype:
            s = s.to(param_dtype)
        if detail_feat is not None and detail_feat.dtype != param_dtype:
            detail_feat = detail_feat.to(param_dtype)
        if locate_point_ctx is not None and locate_point_ctx.dtype != param_dtype:
            locate_point_ctx = locate_point_ctx.to(param_dtype)
        if locate_global_ctx is not None and locate_global_ctx.dtype != param_dtype:
            locate_global_ctx = locate_global_ctx.to(param_dtype)

        n_contours, _, _ = x_t.shape
        t_emb = self.time_emb_net(t)
        cond_emb = t_emb
        if s is not None and hasattr(self, 's_emb_net'):
            s_scaled = (s * 1000.0).to(device=t.device, dtype=t_emb.dtype)
            s_emb = self.s_emb_net(s_scaled)
            cond_emb = t_emb + s_emb

        if cnn_feature.dim() == 3:
            cnn_feature = cnn_feature.unsqueeze(0)

        use_locate = locate_point_ctx is not None or locate_global_ctx is not None
        if locate_only and use_locate:
            sampled_for_point = torch.zeros_like(sampled_feat)
        else:
            sampled_for_point = sampled_feat

        if locate_global_ctx is not None:
            global_ctx = locate_global_ctx
        else:
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

        if locate_only and locate_point_ctx is not None:
            local_ctx = locate_point_ctx
        else:
            local_ctx = self.local_proj(sampled_feat.transpose(1, 2))
            if locate_point_ctx is not None:
                local_ctx = local_ctx + locate_point_ctx
        x = self.point_embed(x_t, sampled_for_point)
        if locate_point_ctx is not None:
            x = x + locate_point_ctx

        if (not (locate_only and use_locate)) and self.use_detail_context and detail_feat is not None:
            detail_ctx = detail_feat.transpose(1, 2)
            local_ctx = local_ctx + self.detail_local_proj(detail_ctx)
            x = x + self.detail_point_proj(detail_ctx)

        for i, dit_layer in enumerate(self.dit_layers):
            context = global_ctx if (i % 2 == 0) else local_ctx
            x = dit_layer(x, context, cond_emb)

        if self.use_latent_loop:
            for _ in range(self.latent_loop_steps):
                x = self.latent_loop(x, cond_emb)

        if hasattr(self.final_layer, 'set_conditional_routing_context'):
            self.final_layer.set_conditional_routing_context(
                diffusion_t=t,
                contour_scale=contour_scale,
            )
        pred = self.final_layer(x, cond_emb)
        reg_loss = pred.new_zeros(())
        if hasattr(self.final_layer, 'reg_loss'):
            reg_loss = reg_loss + self.final_layer.reg_loss().to(pred.device, pred.dtype)
        for dit_layer in self.dit_layers:
            if hasattr(dit_layer, 'reg_loss'):
                reg_loss = reg_loss + dit_layer.reg_loss().to(pred.device, pred.dtype)
        if self.use_per_point_delta:
            pred = pred + self.per_point_delta_head(x, cond_emb)
            reg_loss = reg_loss + self.per_point_delta_head.reg_loss().to(pred.device, pred.dtype)
        return pred, reg_loss
