"""V4.0 flow-matching denoiser with multi-scale detail context.

This keeps the V3.4 backbone intact for checkpoint reuse, then adds two
zero-init residual improvements:
1. detail-context fusion fed by the FM wrapper's local detail sampler
2. a per-point delta head on top of the shared final head
"""

import torch
import torch.nn as nn

from .dit_blocks_v2 import CyclicRoPE1D, RMSNorm, SwiGLU, modulate
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


class StrongPerPointDeltaHead(nn.Module):
    """Stronger zero-init per-point residual head with cyclic local mixing.

    The shared final head remains the stable base predictor.  This module adds a
    point-index-specific residual with local contour context, but starts as a
    no-op so old checkpoints remain safe to reuse.
    """

    def __init__(
        self,
        dim: int = 256,
        out_dim: int = 2,
        num_points: int = 128,
        delta_scale: float = 0.20,
        reg_weight: float = 0.0,
        hidden_mult: float = 2.0,
        use_cyclic_mixer: bool = True,
    ):
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.reg_weight = float(reg_weight)
        self.num_points = int(num_points)
        hidden_dim = max(dim, int(round(dim * float(hidden_mult))))

        self.norm = RMSNorm(dim)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )
        self.point_embed = nn.Parameter(torch.zeros(1, self.num_points, dim))
        self.use_cyclic_mixer = bool(use_cyclic_mixer)
        if self.use_cyclic_mixer:
            self.local_mixer = nn.Conv1d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                padding_mode='circular',
                bias=True,
            )
        else:
            self.local_mixer = None

        self.pre = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Deliberately avoid the legacy names delta_weight/delta_bias so older
        # linear delta checkpoints cannot be partially copied into this head.
        self.out_weight = nn.Parameter(torch.zeros(self.num_points, out_dim, hidden_dim))
        self.out_bias = nn.Parameter(torch.zeros(self.num_points, out_dim))

        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)
        nn.init.normal_(self.point_embed, std=0.02)
        if self.local_mixer is not None:
            nn.init.zeros_(self.local_mixer.weight)
            nn.init.zeros_(self.local_mixer.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(t_emb).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        num_points = x.shape[1]
        x = x + self.point_embed[:, :num_points].to(device=x.device, dtype=x.dtype)
        if self.local_mixer is not None:
            x = x + self.local_mixer(x.transpose(1, 2)).transpose(1, 2)
        h = self.pre(x)
        delta_weight = self.out_weight[:num_points]
        delta_bias = self.out_bias[:num_points]
        delta = torch.einsum('nph,poh->npo', h, delta_weight) + delta_bias.unsqueeze(0)
        return self.delta_scale * delta

    def reg_loss(self) -> torch.Tensor:
        if self.reg_weight <= 0:
            return self.out_weight.new_zeros(())
        reg = self.out_weight.pow(2).mean() + self.out_bias.pow(2).mean()
        return self.reg_weight * reg


class MoEFinalHead(nn.Module):
    """Pure MoE replacement for the final displacement head.

    The legacy FinalLayer parameter names are kept for clean checkpoint reuse:
    ``norm``, ``adaLN`` and ``linear`` load directly from V3.4/V4.1 checkpoints.
    Routed experts are parameterized as small deviations from the loaded linear
    head, so the initial prediction stays close to the proven no-curv model
    while the head can specialize during training.
    """

    def __init__(
        self,
        dim: int = 256,
        out_dim: int = 2,
        num_points: int = 128,
        num_experts: int = 8,
        top_k: int = 2,
        balance_weight: float = 1e-3,
        expert_init_std: float = 1e-4,
        router_noise_std: float = 0.01,
        use_point_embed: bool = True,
        use_cyclic_router: bool = True,
        use_shared_expert: bool = False,
        route_shared_expert: bool = False,
        route_shared_init_bias: float = 0.0,
        routed_expert_scale: float = 1.0,
        expert_type: str = 'linear',
        expert_hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_points = int(num_points)
        self.num_experts = int(num_experts)
        self.balance_weight = float(balance_weight)
        self.router_noise_std = float(router_noise_std)
        self.use_point_embed = bool(use_point_embed)
        self.use_cyclic_router = bool(use_cyclic_router)
        self.use_shared_expert = bool(use_shared_expert)
        self.route_shared_expert = bool(route_shared_expert) and self.use_shared_expert
        self.route_shared_init_bias = float(route_shared_init_bias)
        self.router_num_experts = self.num_experts + (1 if self.route_shared_expert else 0)
        self.top_k = int(max(1, min(top_k, self.router_num_experts)))
        self.routed_expert_scale = float(routed_expert_scale)
        self.expert_type = str(expert_type).strip().lower()
        self.use_mlp_experts = self.expert_type in ('mlp', 'mlp_all', 'mlp_experts')
        self.expert_hidden_dim = int(expert_hidden_dim)

        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )

        if self.use_mlp_experts:
            hidden_dim = max(16, self.expert_hidden_dim)
            self.shared_mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, dim),
            )
            self.expert_fc1_weight = nn.Parameter(torch.empty(self.num_experts, hidden_dim, dim))
            self.expert_fc1_bias = nn.Parameter(torch.empty(self.num_experts, hidden_dim))
            self.expert_fc2_weight = nn.Parameter(torch.empty(self.num_experts, out_dim, hidden_dim))
            self.expert_fc2_bias = nn.Parameter(torch.empty(self.num_experts, out_dim))
            self.register_parameter('expert_delta_weight', None)
            self.register_parameter('expert_delta_bias', None)
        else:
            self.shared_mlp = None
            self.expert_delta_weight = nn.Parameter(torch.empty(self.num_experts, out_dim, dim))
            self.expert_delta_bias = nn.Parameter(torch.empty(self.num_experts, out_dim))
            self.register_parameter('expert_fc1_weight', None)
            self.register_parameter('expert_fc1_bias', None)
            self.register_parameter('expert_fc2_weight', None)
            self.register_parameter('expert_fc2_bias', None)

        if self.use_point_embed:
            self.point_embed = nn.Parameter(torch.empty(1, self.num_points, dim))
        else:
            self.register_parameter('point_embed', None)

        if self.use_cyclic_router:
            self.router_mixer = nn.Conv1d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                padding_mode='circular',
                bias=True,
            )
        else:
            self.router_mixer = None
        self.router_norm = RMSNorm(dim)
        self.router_time = nn.Linear(dim, dim, bias=False)
        self.router = nn.Linear(dim, self.router_num_experts, bias=True)
        self._last_aux_loss = None

        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        if self.use_mlp_experts:
            nn.init.xavier_uniform_(self.shared_mlp[0].weight)
            nn.init.zeros_(self.shared_mlp[0].bias)
            nn.init.zeros_(self.shared_mlp[-1].weight)
            nn.init.zeros_(self.shared_mlp[-1].bias)
            nn.init.xavier_uniform_(self.expert_fc1_weight)
            nn.init.zeros_(self.expert_fc1_bias)
            nn.init.normal_(self.expert_fc2_weight, std=float(expert_init_std))
            nn.init.normal_(self.expert_fc2_bias, std=float(expert_init_std))
        else:
            nn.init.normal_(self.expert_delta_weight, std=float(expert_init_std))
            nn.init.normal_(self.expert_delta_bias, std=float(expert_init_std))
        if self.point_embed is not None:
            nn.init.normal_(self.point_embed, std=0.02)
        if self.router_mixer is not None:
            nn.init.zeros_(self.router_mixer.weight)
            nn.init.zeros_(self.router_mixer.bias)
        nn.init.normal_(self.router_time.weight, std=1e-3)
        nn.init.normal_(self.router.weight, std=1e-3)
        nn.init.zeros_(self.router.bias)
        if self.route_shared_expert and self.route_shared_init_bias != 0:
            with torch.no_grad():
                self.router.bias[-1].fill_(self.route_shared_init_bias)

    def _compute_balance_loss(self, probs: torch.Tensor, selected_idx: torch.Tensor) -> torch.Tensor:
        if self.balance_weight <= 0:
            return probs.new_zeros(())
        importance = probs.mean(dim=(0, 1))
        selected = torch.zeros_like(probs)
        selected.scatter_(-1, selected_idx, 1.0)
        load = selected.mean(dim=(0, 1)) / float(self.top_k)
        target = probs.new_full((self.router_num_experts,), 1.0 / float(self.router_num_experts))
        balance = (importance - target).pow(2).mean() + (load - target).pow(2).mean()
        return balance * self.balance_weight

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(t_emb).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        if self.use_mlp_experts:
            base = self.linear(x + self.shared_mlp(x))
            expert_hidden = torch.einsum('npd,ehd->npeh', x, self.expert_fc1_weight)
            expert_hidden = expert_hidden + self.expert_fc1_bias.view(1, 1, self.num_experts, -1)
            expert_hidden = torch.nn.functional.silu(expert_hidden)
            expert_delta = torch.einsum('npeh,eoh->npeo', expert_hidden, self.expert_fc2_weight)
            expert_delta = expert_delta + self.expert_fc2_bias.view(1, 1, self.num_experts, -1)
        else:
            base = self.linear(x)
            expert_delta = torch.einsum('npd,eod->npeo', x, self.expert_delta_weight)
            expert_delta = expert_delta + self.expert_delta_bias.view(1, 1, self.num_experts, -1)

        if self.route_shared_expert:
            # Equal-level routing: all candidates are full displacement heads.
            # Routed experts are initialized as base + delta; the shared expert is base.
            expert_out = torch.cat([base.unsqueeze(2) + expert_delta, base.unsqueeze(2)], dim=2)
        elif self.use_shared_expert:
            expert_out = expert_delta
        else:
            expert_out = base.unsqueeze(2) + expert_delta

        router_x = x
        num_points = x.shape[1]
        if self.point_embed is not None:
            router_x = router_x + self.point_embed[:, :num_points].to(device=x.device, dtype=x.dtype)
        router_x = router_x + self.router_time(t_emb).unsqueeze(1)
        if self.router_mixer is not None:
            router_x = router_x + self.router_mixer(router_x.transpose(1, 2)).transpose(1, 2)
        router_logits = self.router(self.router_norm(router_x))
        if self.training and self.router_noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_noise_std

        probs = torch.softmax(router_logits, dim=-1)
        top_vals, top_idx = torch.topk(router_logits, k=self.top_k, dim=-1)
        top_gates = torch.softmax(top_vals, dim=-1)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, -1, expert_out.shape[-1])
        chosen = torch.gather(expert_out, dim=2, index=gather_idx)
        routed_out = (chosen * top_gates.unsqueeze(-1)).sum(dim=2)
        if self.route_shared_expert:
            out = routed_out
        elif self.use_shared_expert:
            out = base + self.routed_expert_scale * routed_out
        else:
            out = routed_out

        self._last_aux_loss = self._compute_balance_loss(probs, top_idx)
        return out

    def reg_loss(self) -> torch.Tensor:
        if self._last_aux_loss is None:
            param = self.linear.weight
            return param.new_zeros(())
        aux_loss = self._last_aux_loss
        self._last_aux_loss = None
        return aux_loss


class LatentLoopBlock(nn.Module):
    """Shared latent reasoning block for repeated contour-token refinement.

    The block is intentionally residual-gated with zero-initialized gates, so a
    checkpoint trained without it starts close to the original model.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_points: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        head_dim = self.dim // self.num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.norm_attn = RMSNorm(self.dim)
        self.norm_mlp = RMSNorm(self.dim)
        self.qk_norm = RMSNorm(head_dim)
        self.rope = CyclicRoPE1D(head_dim=head_dim, num_points=num_points)

        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.mlp = SwiGLU(dim=self.dim, dropout=dropout)

        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, 6 * self.dim, bias=True),
        )
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        n, p, d = x.shape
        h = self.num_heads
        hd = self.head_dim
        q = self.q_proj(x).view(n, p, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(n, p, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(n, p, h, hd).transpose(1, 2)
        q = self.rope.apply_rotary(self.qk_norm(q))
        k = self.rope.apply_rotary(self.qk_norm(k))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(n, p, d)
        return self.out_proj(out)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        (shift_attn, scale_attn, gate_attn,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN(t_emb).chunk(6, dim=1)
        x_attn = modulate(self.norm_attn(x), shift_attn, scale_attn)
        x = x + gate_attn.unsqueeze(1) * self._self_attention(x_attn)
        x_mlp = modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mlp)
        return x


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
