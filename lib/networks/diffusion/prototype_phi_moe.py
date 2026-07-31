"""Prototype-routed MoE with population-level phi balancing.

This is an isolated 2026 research path.  It intentionally does not reuse the
legacy linear/noisy/cyclic router.  Routing is cosine similarity to learned
expert prototypes, and collapse control uses an EMA of soft routing
probabilities rather than per-minibatch hard assignment counts.
"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit_blocks_v2 import SwiGLU


class PrototypePhiMoE(nn.Module):
    """Sparse SwiGLU FFN replacement with prototype routing and phi-balancing."""

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        top_k: int = 1,
        expert_hidden_dim: int = 0,
        router_temperature: float = 0.20,
        phi_weight: float = 1e-3,
        phi_ema_decay: float = 0.99,
        contrastive_weight: float = 1e-3,
        contrastive_temperature: float = 0.07,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_experts = int(max(2, num_experts))
        self.top_k = int(max(1, min(top_k, self.num_experts)))
        if int(expert_hidden_dim) <= 0:
            expert_hidden_dim = int(math.ceil((self.dim * 8.0 / 3.0) / 64.0) * 64)
        self.expert_hidden_dim = int(expert_hidden_dim)
        self.router_temperature = float(max(router_temperature, 1e-4))
        self.phi_weight = float(max(phi_weight, 0.0))
        self.phi_ema_decay = float(min(max(phi_ema_decay, 0.0), 0.99999))
        self.contrastive_weight = float(max(contrastive_weight, 0.0))
        self.contrastive_temperature = float(max(contrastive_temperature, 1e-4))

        self.prototypes = nn.Parameter(torch.empty(self.num_experts, self.dim))
        self.experts = nn.ModuleList([
            SwiGLU(dim=self.dim, hidden_dim=self.expert_hidden_dim)
            for _ in range(self.num_experts)
        ])
        nn.init.normal_(self.prototypes, std=0.02)

        uniform = torch.full((self.num_experts,), 1.0 / self.num_experts)
        self.register_buffer("phi_ema_prob", uniform.clone())
        self.register_buffer("_prototypes_initialized", torch.tensor(False))

        # Non-persistent diagnostics: these never affect checkpoints or routing.
        self.register_buffer("_diag_soft_sum", torch.zeros(self.num_experts), persistent=False)
        self.register_buffer("_diag_hard_sum", torch.zeros(self.num_experts), persistent=False)
        self.register_buffer("_diag_top1_sum", torch.zeros(self.num_experts), persistent=False)
        self.register_buffer("_diag_tokens", torch.zeros(()), persistent=False)
        self.register_buffer("_diag_calls", torch.zeros(()), persistent=False)
        self._last_aux_loss = None

    @torch.no_grad()
    def _initialize_prototypes_from_data(self, descriptors: torch.Tensor) -> None:
        """One-time, data-aware prototype initialization.

        Random prototypes are fragile when contour descriptors share a strong
        common direction: tiny score offsets can make one expert win almost
        every hard Top-K decision. Farthest-point seeding followed by a few
        spherical k-means updates gives every prototype a real data mode before
        sparse routing starts, with no persistent inference-time machinery.
        """
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
            distance = 1.0 - points @ centers[-1]
            min_distance = torch.minimum(min_distance, distance)
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

    def _routing_contrastive_loss(
        self,
        descriptors: torch.Tensor,
        top1_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self.contrastive_weight <= 0:
            return descriptors.new_zeros(())
        flat_x = descriptors.reshape(-1, self.dim)
        flat_idx = top1_idx.reshape(-1)
        centroids = []
        valid_ids = []
        for expert_id in range(self.num_experts):
            mask = flat_idx == expert_id
            if bool(mask.any()):
                centroids.append(flat_x[mask].mean(dim=0))
                valid_ids.append(expert_id)
        if len(valid_ids) < 2:
            return descriptors.new_zeros(())
        centroid_mat = F.normalize(torch.stack(centroids).float(), dim=-1)
        prototype_mat = F.normalize(self.prototypes[valid_ids].float(), dim=-1)
        logits = prototype_mat @ centroid_mat.transpose(0, 1)
        labels = torch.arange(len(valid_ids), device=descriptors.device)
        return F.cross_entropy(logits / self.contrastive_temperature, labels).to(descriptors.dtype)

    def _phi_balancing_loss(self, probs: torch.Tensor) -> torch.Tensor:
        if self.phi_weight <= 0:
            return probs.new_zeros(())
        batch_prob = probs.float().reshape(-1, self.num_experts).mean(dim=0)
        with torch.no_grad():
            self.phi_ema_prob.mul_(self.phi_ema_decay).add_(
                batch_prob.detach() * (1.0 - self.phi_ema_decay)
            )
            self.phi_ema_prob.div_(self.phi_ema_prob.sum().clamp_min(1e-12))
        # Negative Shannon entropy potential:
        # grad phi(m) = log(m) + 1.  The EMA-derived price is detached so only
        # the current smooth routing probabilities receive gradients.
        price = self.phi_ema_prob.clamp_min(1e-8).log().add(1.0).detach()
        # Potentials are defined up to an additive constant. Centering leaves
        # the router gradient unchanged because routing probabilities sum to
        # one, while keeping the reported auxiliary loss near zero at balance.
        price = price - price.mean()
        return (
            self.phi_weight
            * float(self.num_experts)
            * torch.sum(batch_prob * price)
        ).to(probs.dtype)

    @torch.no_grad()
    def _update_diagnostics(self, probs: torch.Tensor, top_idx: torch.Tensor) -> None:
        flat_probs = probs.float().reshape(-1, self.num_experts)
        flat_idx = top_idx.reshape(-1, self.top_k)
        soft = flat_probs.mean(dim=0)
        one_hot = F.one_hot(top_idx, num_classes=self.num_experts).float()
        hard = one_hot.reshape(-1, self.top_k, self.num_experts).mean(dim=(0, 1))
        top1 = F.one_hot(flat_idx[:, 0], num_classes=self.num_experts).float().mean(dim=0)
        tokens = float(flat_probs.shape[0])
        self._diag_soft_sum.add_(soft * tokens)
        self._diag_hard_sum.add_(hard * tokens)
        self._diag_top1_sum.add_(top1 * tokens)
        self._diag_tokens.add_(tokens)
        self._diag_calls.add_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A contour, not an individual point, is the semantic unit here.
        # Pooling prevents 128 correlated points from being counted as
        # independent routing decisions. LayerNorm removes the shared channel
        # direction that otherwise lets one random prototype win globally.
        descriptors = x.float().mean(dim=1)
        descriptors = F.layer_norm(descriptors, (self.dim,))
        self._initialize_prototypes_from_data(descriptors)
        x_float = F.normalize(descriptors, dim=-1)
        prototypes = F.normalize(self.prototypes.float(), dim=-1)
        logits = (x_float @ prototypes.transpose(0, 1)) / self.router_temperature
        probs = torch.softmax(logits, dim=-1)
        top_values, top_idx = torch.topk(logits, k=self.top_k, dim=-1)
        if self.top_k == 1:
            gates = torch.ones_like(top_values)
        else:
            gates = torch.softmax(top_values, dim=-1)
        # Every point in a contour follows the same semantic route.
        top_idx = top_idx.unsqueeze(1).expand(-1, x.shape[1], -1)
        gates = gates.unsqueeze(1).expand(-1, x.shape[1], -1)

        flat_x = x.reshape(-1, self.dim)
        flat_idx = top_idx.reshape(-1, self.top_k)
        flat_gates = gates.to(x.dtype).reshape(-1, self.top_k)
        output = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            token_slot = torch.nonzero(flat_idx == expert_id, as_tuple=False)
            if token_slot.numel() == 0:
                continue
            token_ids = token_slot[:, 0]
            slots = token_slot[:, 1]
            expert_out = expert(flat_x[token_ids])
            weights = flat_gates[token_ids, slots].unsqueeze(-1)
            output.index_add_(0, token_ids, expert_out * weights)

        if self.training:
            phi_loss = self._phi_balancing_loss(probs)
            contrastive = self._routing_contrastive_loss(descriptors, top_idx[:, 0, 0])
            self._last_aux_loss = phi_loss + self.contrastive_weight * contrastive
        else:
            self._last_aux_loss = x.new_zeros(())
        self._update_diagnostics(probs, top_idx[:, 0])
        return output.view_as(x)

    def reg_loss(self) -> torch.Tensor:
        if self._last_aux_loss is None:
            return self.prototypes.new_zeros(())
        loss = self._last_aux_loss
        self._last_aux_loss = None
        return loss

    @torch.no_grad()
    def routing_diagnostics(self) -> Dict[str, torch.Tensor]:
        denom = self._diag_tokens.clamp_min(1.0)
        soft = self._diag_soft_sum / denom
        hard = self._diag_hard_sum / denom
        top1 = self._diag_top1_sum / denom
        entropy = -(soft.clamp_min(1e-12) * soft.clamp_min(1e-12).log()).sum()
        return {
            "soft_load": soft.detach().cpu(),
            "hard_load": hard.detach().cpu(),
            "top1_load": top1.detach().cpu(),
            "normalized_entropy": (entropy / math.log(self.num_experts)).detach().cpu(),
            "hard_cv": (hard.std(unbiased=False) / hard.mean().clamp_min(1e-12)).detach().cpu(),
            "dead_experts_lt_1pct": (hard < 0.01).sum().detach().cpu(),
            "tokens": self._diag_tokens.detach().cpu(),
            "calls": self._diag_calls.detach().cpu(),
            "phi_ema_prob": self.phi_ema_prob.detach().cpu(),
        }
