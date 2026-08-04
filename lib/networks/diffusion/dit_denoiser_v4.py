"""V4.0 flow-matching denoiser with multi-scale detail context.

This keeps the V3.4 backbone intact for checkpoint reuse, then adds two
zero-init residual improvements:
1. detail-context fusion fed by the FM wrapper's local detail sampler
2. a per-point delta head on top of the shared final head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class DenseResidualFinalHead(nn.Module):
    """Dense capacity control for output-head MoE experiments.

    The legacy final-layer names are retained so the proven shared linear
    predictor loads exactly.  A single zero-near-initialized MLP predicts a
    residual displacement.  Choosing hidden_dim=1024 matches the routed MLP
    parameter pool of four hidden-256 experts to within a few parameters.
    """

    def __init__(
        self,
        dim: int = 256,
        out_dim: int = 2,
        hidden_dim: int = 1024,
        residual_init_std: float = 1e-4,
    ):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )
        self.residual_mlp = nn.Sequential(
            nn.Linear(dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), out_dim),
        )

        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.xavier_uniform_(self.residual_mlp[0].weight)
        nn.init.zeros_(self.residual_mlp[0].bias)
        nn.init.normal_(self.residual_mlp[-1].weight, std=float(residual_init_std))
        nn.init.normal_(self.residual_mlp[-1].bias, std=float(residual_init_std))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(t_emb).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x) + self.residual_mlp(x)

    def reg_loss(self) -> torch.Tensor:
        return self.linear.weight.new_zeros(())


class SharedDenseSparseResidualHead(nn.Module):
    """Dense shared predictor plus a small, truly sparse contour adapter.

    The shared path is identical to :class:`DenseResidualFinalHead`.  A single
    expert is selected for each complete contour and only selected experts are
    executed.  Expert output layers are zero initialized, so copying a trained
    dense head into this module preserves its function exactly at step zero.

    Expert-load balancing uses a small non-trainable routing bias updated from
    an EMA of hard assignments.  This avoids an auxiliary loss competing with
    the displacement objective while still pushing persistently idle experts
    back toward the decision boundary.
    """

    def __init__(
        self,
        dim: int = 256,
        out_dim: int = 2,
        shared_hidden_dim: int = 1024,
        num_experts: int = 4,
        expert_hidden_dim: int = 128,
        router_temperature: float = 0.50,
        load_ema_decay: float = 0.99,
        balance_bias_step: float = 1e-3,
        balance_bias_limit: float = 0.10,
        expert_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.out_dim = int(out_dim)
        self.num_experts = int(max(2, num_experts))
        self.router_temperature = float(max(router_temperature, 1e-4))
        self.load_ema_decay = float(min(max(load_ema_decay, 0.0), 0.99999))
        self.balance_bias_step = float(max(balance_bias_step, 0.0))
        self.balance_bias_limit = float(max(balance_bias_limit, 0.0))
        self.expert_scale = float(expert_scale)

        # Keep these names identical to DenseResidualFinalHead for direct,
        # function-preserving state transfer.
        self.norm = RMSNorm(self.dim)
        self.linear = nn.Linear(self.dim, self.out_dim)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, 2 * self.dim, bias=True),
        )
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.dim, int(shared_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(shared_hidden_dim), self.out_dim),
        )

        hidden_dim = int(max(16, expert_hidden_dim))
        self.router_norm = RMSNorm(self.dim)
        self.router = nn.Linear(self.dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.out_dim),
            )
            for _ in range(self.num_experts)
        ])

        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.xavier_uniform_(self.residual_mlp[0].weight)
        nn.init.zeros_(self.residual_mlp[0].bias)
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)
        nn.init.normal_(self.router.weight, std=1e-3)
        for expert in self.experts:
            nn.init.xavier_uniform_(expert[0].weight)
            nn.init.zeros_(expert[0].bias)
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

        uniform = torch.full((self.num_experts,), 1.0 / self.num_experts)
        self.register_buffer("load_ema", uniform.clone())
        self.register_buffer("routing_bias", torch.zeros(self.num_experts))
        self.register_buffer("_diag_hard_sum", torch.zeros(self.num_experts), persistent=False)
        self.register_buffer("_diag_contours", torch.zeros(()), persistent=False)
        self.register_buffer("_diag_calls", torch.zeros(()), persistent=False)
        self._routing_diagnostics_enabled = True

    def enable_routing_diagnostics(self, enabled: bool = True) -> None:
        self._routing_diagnostics_enabled = bool(enabled)

    @torch.no_grad()
    def reset_routing_diagnostics(self) -> None:
        self._diag_hard_sum.zero_()
        self._diag_contours.zero_()
        self._diag_calls.zero_()

    @torch.no_grad()
    def _observe_and_balance(self, top1_idx: torch.Tensor) -> None:
        hard = F.one_hot(top1_idx, num_classes=self.num_experts).float().mean(dim=0)
        if self.training:
            self.load_ema.mul_(self.load_ema_decay).add_(
                hard * (1.0 - self.load_ema_decay)
            )
            if self.balance_bias_step > 0:
                target = hard.new_full(hard.shape, 1.0 / self.num_experts)
                self.routing_bias.add_(self.balance_bias_step * (target - hard))
                self.routing_bias.sub_(self.routing_bias.mean())
                self.routing_bias.clamp_(
                    min=-self.balance_bias_limit,
                    max=self.balance_bias_limit,
                )
        if self.training or self._routing_diagnostics_enabled:
            contours = float(top1_idx.numel())
            self._diag_hard_sum.add_(hard * contours)
            self._diag_contours.add_(contours)
            self._diag_calls.add_(1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(t_emb).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        base = self.linear(x) + self.residual_mlp(x)

        descriptor = self.router_norm(x.float().mean(dim=1))
        # Cosine routing bounds logit magnitude.  The balancing bias can then
        # always recover an idle expert instead of being overwhelmed by an
        # unconstrained router weight norm.
        descriptor = F.normalize(descriptor, dim=-1)
        router_weight = F.normalize(self.router.weight.float(), dim=-1)
        logits = F.linear(descriptor, router_weight) / self.router_temperature
        logits = logits + self.routing_bias.to(device=logits.device, dtype=logits.dtype)
        probs = torch.softmax(logits, dim=-1)
        top1_idx = logits.argmax(dim=-1)
        hard_gate = F.one_hot(top1_idx, num_classes=self.num_experts).to(probs.dtype)
        # Exact one-hot forward routing with a softmax straight-through gradient.
        gates = hard_gate + probs - probs.detach()

        delta = torch.zeros_like(base)
        for expert_id, expert in enumerate(self.experts):
            contour_ids = torch.nonzero(top1_idx == expert_id, as_tuple=False).flatten()
            if contour_ids.numel() == 0:
                continue
            expert_out = expert(x[contour_ids])
            weight = gates[contour_ids, expert_id].to(x.dtype).view(-1, 1, 1)
            delta.index_add_(0, contour_ids, expert_out * weight)
        self._observe_and_balance(top1_idx)
        return base + self.expert_scale * delta

    def reg_loss(self) -> torch.Tensor:
        return self.linear.weight.new_zeros(())

    @torch.no_grad()
    def routing_diagnostics(self):
        hard = self._diag_hard_sum / self._diag_contours.clamp_min(1.0)
        entropy = -(hard.clamp_min(1e-12) * hard.clamp_min(1e-12).log()).sum()
        return {
            "soft_load": self.load_ema.detach().cpu(),
            "hard_load": hard.detach().cpu(),
            "top1_load": hard.detach().cpu(),
            "normalized_entropy": (entropy / math.log(self.num_experts)).detach().cpu(),
            "hard_cv": (
                hard.std(unbiased=False) / hard.mean().clamp_min(1e-12)
            ).detach().cpu(),
            "dead_experts_lt_1pct": (hard < 0.01).sum().detach().cpu(),
            "contours": self._diag_contours.detach().cpu(),
            "calls": self._diag_calls.detach().cpu(),
            "routing_bias": self.routing_bias.detach().cpu(),
        }


class ModernSparseResidualHead(nn.Module):
    """Contour-consistent, truly sparse residual displacement experts.

    A shared linear head models the common vector field.  A contour-level
    prototype router selects a small set of nonlinear residual experts.  The
    selected contour indices are formed before expert execution, so unselected
    experts do not run.  This differs from the legacy output MoE, which computes
    every expert before its Top-K gather.
    """

    def __init__(
        self,
        dim: int = 256,
        out_dim: int = 2,
        num_experts: int = 4,
        top_k: int = 2,
        expert_hidden_dim: int = 256,
        router_temperature: float = 0.20,
        balance_weight: float = 1e-3,
        phi_ema_decay: float = 0.99,
        contrastive_weight: float = 1e-3,
        contrastive_temperature: float = 0.07,
        expert_init_std: float = 1e-4,
    ):
        super().__init__()
        self.dim = int(dim)
        self.out_dim = int(out_dim)
        self.num_experts = int(max(2, num_experts))
        self.top_k = int(max(1, min(top_k, self.num_experts)))
        self.router_temperature = float(max(router_temperature, 1e-4))
        self.balance_weight = float(max(balance_weight, 0.0))
        self.phi_ema_decay = float(min(max(phi_ema_decay, 0.0), 0.99999))
        self.contrastive_weight = float(max(contrastive_weight, 0.0))
        self.contrastive_temperature = float(max(contrastive_temperature, 1e-4))

        # Compatibility names: these load from the same V3.4/V4.6 checkpoint
        # tensors as the standard and legacy-MoE final heads.
        self.norm = RMSNorm(self.dim)
        self.linear = nn.Linear(self.dim, self.out_dim)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, 2 * self.dim, bias=True),
        )

        hidden_dim = int(max(16, expert_hidden_dim))
        self.prototypes = nn.Parameter(torch.empty(self.num_experts, self.dim))
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.out_dim),
            )
            for _ in range(self.num_experts)
        ])

        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.normal_(self.prototypes, std=0.02)
        for expert in self.experts:
            nn.init.xavier_uniform_(expert[0].weight)
            nn.init.zeros_(expert[0].bias)
            nn.init.normal_(expert[-1].weight, std=float(expert_init_std))
            nn.init.normal_(expert[-1].bias, std=float(expert_init_std))

        uniform = torch.full((self.num_experts,), 1.0 / self.num_experts)
        self.register_buffer("phi_ema_prob", uniform.clone())
        self.register_buffer("_prototypes_initialized", torch.tensor(False))
        self.register_buffer("_diag_soft_sum", torch.zeros(self.num_experts), persistent=False)
        self.register_buffer("_diag_hard_sum", torch.zeros(self.num_experts), persistent=False)
        self.register_buffer("_diag_top1_sum", torch.zeros(self.num_experts), persistent=False)
        self.register_buffer("_diag_contours", torch.zeros(()), persistent=False)
        self.register_buffer("_diag_calls", torch.zeros(()), persistent=False)
        self._last_aux_loss = None
        self._routing_diagnostics_enabled = True

    def enable_routing_diagnostics(self, enabled: bool = True) -> None:
        self._routing_diagnostics_enabled = bool(enabled)

    @torch.no_grad()
    def _initialize_prototypes_from_data(self, descriptors: torch.Tensor) -> None:
        if bool(self._prototypes_initialized.item()):
            return
        points = F.normalize(descriptors.detach().float(), dim=-1)
        if points.shape[0] == 0:
            return
        centered = points - points.mean(dim=0, keepdim=True)
        first = int(centered.square().sum(dim=-1).argmax().item())
        centers = [points[first]]
        min_distance = 1.0 - points @ centers[0]
        for _ in range(1, self.num_experts):
            next_id = int(min_distance.argmax().item())
            centers.append(points[next_id])
            min_distance = torch.minimum(
                min_distance,
                1.0 - points @ centers[-1],
            )
        center_mat = torch.stack(centers)
        for _ in range(4):
            assignment = torch.argmax(points @ center_mat.transpose(0, 1), dim=-1)
            updated = []
            for expert_id in range(self.num_experts):
                mask = assignment == expert_id
                candidate = points[mask].mean(dim=0) if bool(mask.any()) else center_mat[expert_id]
                updated.append(F.normalize(candidate, dim=0))
            center_mat = torch.stack(updated)
        self.prototypes.copy_(center_mat.to(self.prototypes))
        self._prototypes_initialized.fill_(True)

    def _phi_balancing_loss(self, probs: torch.Tensor) -> torch.Tensor:
        if self.balance_weight <= 0:
            return probs.new_zeros(())
        batch_prob = probs.float().mean(dim=0)
        with torch.no_grad():
            self.phi_ema_prob.mul_(self.phi_ema_decay).add_(
                batch_prob.detach() * (1.0 - self.phi_ema_decay)
            )
            self.phi_ema_prob.div_(self.phi_ema_prob.sum().clamp_min(1e-12))
        price = self.phi_ema_prob.clamp_min(1e-8).log().add(1.0).detach()
        price = price - price.mean()
        return (
            self.balance_weight
            * float(self.num_experts)
            * torch.sum(batch_prob * price)
        ).to(probs.dtype)

    def _routing_contrastive_loss(
        self,
        descriptors: torch.Tensor,
        top1_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self.contrastive_weight <= 0:
            return descriptors.new_zeros(())
        centroids = []
        valid_ids = []
        for expert_id in range(self.num_experts):
            mask = top1_idx == expert_id
            if bool(mask.any()):
                centroids.append(descriptors[mask].mean(dim=0))
                valid_ids.append(expert_id)
        if len(valid_ids) < 2:
            return descriptors.new_zeros(())
        centroid_mat = F.normalize(torch.stack(centroids).float(), dim=-1)
        prototype_mat = F.normalize(self.prototypes[valid_ids].float(), dim=-1)
        logits = prototype_mat @ centroid_mat.transpose(0, 1)
        labels = torch.arange(len(valid_ids), device=descriptors.device)
        return F.cross_entropy(
            logits / self.contrastive_temperature,
            labels,
        ).to(descriptors.dtype)

    @torch.no_grad()
    def _update_diagnostics(self, probs: torch.Tensor, top_idx: torch.Tensor) -> None:
        contours = float(probs.shape[0])
        soft = probs.float().mean(dim=0)
        one_hot = F.one_hot(top_idx, num_classes=self.num_experts).float()
        hard = one_hot.mean(dim=(0, 1))
        top1 = one_hot[:, 0].mean(dim=0)
        self._diag_soft_sum.add_(soft * contours)
        self._diag_hard_sum.add_(hard * contours)
        self._diag_top1_sum.add_(top1 * contours)
        self._diag_contours.add_(contours)
        self._diag_calls.add_(1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(t_emb).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        base = self.linear(x)

        descriptors = x.float().mean(dim=1)
        descriptors = F.layer_norm(descriptors, (self.dim,))
        self._initialize_prototypes_from_data(descriptors)
        descriptor_unit = F.normalize(descriptors, dim=-1)
        prototype_unit = F.normalize(self.prototypes.float(), dim=-1)
        logits = (
            descriptor_unit @ prototype_unit.transpose(0, 1)
        ) / self.router_temperature

        observe_routing = self.training or self._routing_diagnostics_enabled
        probs = torch.softmax(logits, dim=-1) if observe_routing else None
        top_values, top_idx = torch.topk(logits, k=self.top_k, dim=-1)
        gates = (
            torch.ones_like(top_values)
            if self.top_k == 1
            else torch.softmax(top_values, dim=-1)
        )

        # Route first, then execute only experts that received contours.
        delta = torch.zeros_like(base)
        for expert_id, expert in enumerate(self.experts):
            contour_slot = torch.nonzero(top_idx == expert_id, as_tuple=False)
            if contour_slot.numel() == 0:
                continue
            contour_ids = contour_slot[:, 0]
            slots = contour_slot[:, 1]
            expert_out = expert(x[contour_ids])
            weights = gates[contour_ids, slots].to(x.dtype).view(-1, 1, 1)
            delta.index_add_(0, contour_ids, expert_out * weights)

        if self.training:
            phi = self._phi_balancing_loss(probs)
            contrastive = self._routing_contrastive_loss(descriptors, top_idx[:, 0])
            self._last_aux_loss = phi + self.contrastive_weight * contrastive
        else:
            self._last_aux_loss = x.new_zeros(())
        if observe_routing:
            self._update_diagnostics(probs, top_idx)
        return base + delta

    def reg_loss(self) -> torch.Tensor:
        if self._last_aux_loss is None:
            return self.prototypes.new_zeros(())
        loss = self._last_aux_loss
        self._last_aux_loss = None
        return loss

    @torch.no_grad()
    def routing_diagnostics(self):
        denom = self._diag_contours.clamp_min(1.0)
        soft = self._diag_soft_sum / denom
        hard = self._diag_hard_sum / denom
        top1 = self._diag_top1_sum / denom
        entropy = -(soft.clamp_min(1e-12) * soft.clamp_min(1e-12).log()).sum()
        return {
            "soft_load": soft.detach().cpu(),
            "hard_load": hard.detach().cpu(),
            "top1_load": top1.detach().cpu(),
            "normalized_entropy": (entropy / math.log(self.num_experts)).detach().cpu(),
            "hard_cv": (
                hard.std(unbiased=False) / hard.mean().clamp_min(1e-12)
            ).detach().cpu(),
            "dead_experts_lt_1pct": (hard < 0.01).sum().detach().cpu(),
            "contours": self._diag_contours.detach().cpu(),
            "calls": self._diag_calls.detach().cpu(),
            "phi_ema_prob": self.phi_ema_prob.detach().cpu(),
        }


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
        balance_mode: str = 'legacy',
        hard_phi_ema_decay: float = 0.99,
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
        self.balance_mode = str(balance_mode).strip().lower()
        self.hard_phi_ema_decay = float(
            min(max(hard_phi_ema_decay, 0.0), 0.99999)
        )
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
        self.register_buffer("_diag_soft_sum", torch.zeros(self.router_num_experts), persistent=False)
        self.register_buffer("_diag_hard_sum", torch.zeros(self.router_num_experts), persistent=False)
        self.register_buffer("_diag_top1_sum", torch.zeros(self.router_num_experts), persistent=False)
        self.register_buffer("_diag_tokens", torch.zeros(()), persistent=False)
        self.register_buffer(
            "hard_phi_ema_load",
            torch.full(
                (self.router_num_experts,),
                1.0 / float(self.router_num_experts),
            ),
        )
        # Optional evaluation-only event capture. Disabled by default and never
        # changes routing or checkpoint contents.
        self._capture_conditional_routing = False
        self._conditional_routing_context = None
        self._conditional_routing_events = []
        self._routing_diagnostics_enabled = True

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

    def enable_conditional_routing_capture(self, enabled: bool = True) -> None:
        self._capture_conditional_routing = bool(enabled)
        self._conditional_routing_context = None
        self._conditional_routing_events = []

    def enable_routing_diagnostics(self, enabled: bool = True) -> None:
        """Toggle inference-only routing observers without changing outputs."""
        self._routing_diagnostics_enabled = bool(enabled)

    def set_conditional_routing_context(
        self,
        diffusion_t: torch.Tensor = None,
        contour_scale: torch.Tensor = None,
    ) -> None:
        if not self._capture_conditional_routing:
            self._conditional_routing_context = None
            return
        self._conditional_routing_context = {
            "diffusion_t": (
                diffusion_t.detach().float().reshape(-1)
                if torch.is_tensor(diffusion_t)
                else None
            ),
            "contour_scale": (
                contour_scale.detach().float().reshape(-1)
                if torch.is_tensor(contour_scale)
                else None
            ),
        }

    def drain_conditional_routing_events(self):
        events = self._conditional_routing_events
        self._conditional_routing_events = []
        self._conditional_routing_context = None
        return events

    def _compute_balance_loss(self, probs: torch.Tensor, selected_idx: torch.Tensor) -> torch.Tensor:
        if self.balance_weight <= 0:
            return probs.new_zeros(())
        importance = probs.mean(dim=(0, 1))
        selected = torch.zeros_like(probs)
        selected.scatter_(-1, selected_idx, 1.0)
        load = selected.mean(dim=(0, 1)) / float(self.top_k)
        if self.balance_mode in ('hard_phi', 'hard-phi', 'population_hard_phi'):
            with torch.no_grad():
                self.hard_phi_ema_load.mul_(self.hard_phi_ema_decay).add_(
                    load.detach() * (1.0 - self.hard_phi_ema_decay)
                )
                self.hard_phi_ema_load.div_(
                    self.hard_phi_ema_load.sum().clamp_min(1e-12)
                )
            # The hard Top-K load determines a detached population-level
            # congestion price; the current soft probabilities carry its
            # gradient. This avoids the legacy no-op hard-count penalty.
            price = self.hard_phi_ema_load.clamp_min(1e-8).log().detach()
            price = price - price.mean()
            return (
                self.balance_weight
                * float(self.router_num_experts)
                * torch.sum(importance * price)
            ).to(probs.dtype)
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

        observe_routing = (
            self.training
            or self._routing_diagnostics_enabled
            or self._capture_conditional_routing
        )
        if not observe_routing:
            self._last_aux_loss = None
            return out

        probs = torch.softmax(router_logits, dim=-1)
        self._last_aux_loss = self._compute_balance_loss(probs, top_idx)
        with torch.no_grad():
            tokens = float(probs.shape[0] * probs.shape[1])
            self._diag_soft_sum.add_(probs.float().mean(dim=(0, 1)) * tokens)
            one_hot = torch.nn.functional.one_hot(
                top_idx, num_classes=self.router_num_experts
            ).float()
            self._diag_hard_sum.add_(one_hot.mean(dim=(0, 1, 2)) * tokens)
            self._diag_top1_sum.add_(one_hot[:, :, 0].mean(dim=(0, 1)) * tokens)
            self._diag_tokens.add_(tokens)
            if self._capture_conditional_routing:
                n_contours, n_points, _ = probs.shape
                point_bins = min(8, n_points)
                point_bin_ids = (
                    torch.arange(n_points, device=probs.device) * point_bins
                ) // max(n_points, 1)
                point_top1 = torch.zeros(
                    n_contours,
                    point_bins,
                    self.router_num_experts,
                    device=probs.device,
                    dtype=torch.float32,
                )
                point_top1.scatter_add_(
                    1,
                    point_bin_ids.view(1, -1, 1).expand(
                        n_contours, -1, self.router_num_experts
                    ),
                    one_hot[:, :, 0].float(),
                )
                context = self._conditional_routing_context or {}
                diffusion_t = context.get("diffusion_t")
                contour_scale = context.get("contour_scale")
                if diffusion_t is None or diffusion_t.numel() != n_contours:
                    diffusion_t = probs.new_full((n_contours,), float("nan")).float()
                if contour_scale is None or contour_scale.numel() != n_contours:
                    contour_scale = probs.new_full((n_contours,), float("nan")).float()
                delta_float = expert_delta.float()
                self._conditional_routing_events.append({
                    "soft_sum": probs.float().sum(dim=1).cpu(),
                    "hard_sum": one_hot.float().sum(dim=(1, 2)).cpu(),
                    "top1_sum": one_hot[:, :, 0].float().sum(dim=1).cpu(),
                    "point_top1": point_top1.cpu(),
                    "expert_delta_l2_sum": delta_float.norm(dim=-1).sum(dim=1).cpu(),
                    "expert_delta_cross": torch.einsum(
                        "npeo,npfo->nef", delta_float, delta_float
                    ).cpu(),
                    "diffusion_t": diffusion_t.cpu(),
                    "contour_scale": contour_scale.cpu(),
                    "points": int(n_points),
                    "top_k": int(self.top_k),
                })
        return out

    def reg_loss(self) -> torch.Tensor:
        if self._last_aux_loss is None:
            param = self.linear.weight
            return param.new_zeros(())
        aux_loss = self._last_aux_loss
        self._last_aux_loss = None
        return aux_loss

    @torch.no_grad()
    def routing_diagnostics(self):
        denom = self._diag_tokens.clamp_min(1.0)
        soft = self._diag_soft_sum / denom
        hard = self._diag_hard_sum / denom
        top1 = self._diag_top1_sum / denom
        entropy = -(soft.clamp_min(1e-12) * soft.clamp_min(1e-12).log()).sum()
        return {
            "soft_load": soft.detach().cpu(),
            "hard_load": hard.detach().cpu(),
            "top1_load": top1.detach().cpu(),
            "normalized_entropy": (
                entropy / math.log(self.router_num_experts)
            ).detach().cpu(),
            "hard_cv": (
                hard.std(unbiased=False) / hard.mean().clamp_min(1e-12)
            ).detach().cpu(),
            "dead_experts_lt_1pct": (hard < 0.01).sum().detach().cpu(),
            "tokens": self._diag_tokens.detach().cpu(),
            "hard_phi_ema_load": self.hard_phi_ema_load.detach().cpu(),
            "balance_mode": self.balance_mode,
        }


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
