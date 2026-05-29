"""
DiT Blocks V3 for Diffusion Snake (Rollback Version 3.1)
--------------------------------------------------------------
Key Upgrades:
- RESTORED Attention Flow (Self-Attention -> Cross-Attention).
- REMOVED Local Smoothing Conv (Redundant with Self-Attention).
- RETAINS Octagon Initialization (External Decoders).
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .dit_blocks_v2 import (
    RMSNorm,
    SwiGLU,
    CyclicRoPE1D,
    modulate
)


class RoutedFFNMoE(nn.Module):
    """Routed experts used inside the DiT FFN branch.

    The original FFN remains outside this module as the shared expert, so older
    checkpoints keep their learned FFN weights and the routed experts can adapt
    on top of that stable path.
    """

    def __init__(
        self,
        dim: int = 256,
        num_points: int = 128,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_dim: int = 256,
        balance_weight: float = 1e-3,
        router_noise_std: float = 0.01,
        expert_init_std: float = 1e-4,
        routed_scale: float = 1.0,
        use_point_embed: bool = True,
        use_cyclic_router: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_points = int(num_points)
        self.num_experts = int(num_experts)
        self.top_k = int(max(1, min(top_k, self.num_experts)))
        self.hidden_dim = int(max(16, hidden_dim))
        self.balance_weight = float(balance_weight)
        self.router_noise_std = float(router_noise_std)
        self.routed_scale = float(routed_scale)

        self.expert_fc1_weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.dim))
        self.expert_fc1_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.expert_fc2_weight = nn.Parameter(torch.empty(self.num_experts, self.dim, self.hidden_dim))
        self.expert_fc2_bias = nn.Parameter(torch.empty(self.num_experts, self.dim))

        if use_point_embed:
            self.point_embed = nn.Parameter(torch.empty(1, self.num_points, self.dim))
        else:
            self.register_parameter('point_embed', None)
        self.router_norm = RMSNorm(self.dim)
        self.router_time = nn.Linear(self.dim, self.dim, bias=False)
        self.router = nn.Linear(self.dim, self.num_experts, bias=True)
        if use_cyclic_router:
            self.router_mixer = nn.Conv1d(
                self.dim,
                self.dim,
                kernel_size=3,
                padding=1,
                padding_mode='circular',
                bias=True,
            )
        else:
            self.router_mixer = None
        self._last_aux_loss = None

        nn.init.xavier_uniform_(self.expert_fc1_weight)
        nn.init.zeros_(self.expert_fc1_bias)
        nn.init.normal_(self.expert_fc2_weight, std=float(expert_init_std))
        nn.init.normal_(self.expert_fc2_bias, std=float(expert_init_std))
        if self.point_embed is not None:
            nn.init.normal_(self.point_embed, std=0.02)
        nn.init.normal_(self.router_time.weight, std=1e-3)
        nn.init.normal_(self.router.weight, std=1e-3)
        nn.init.zeros_(self.router.bias)
        if self.router_mixer is not None:
            nn.init.zeros_(self.router_mixer.weight)
            nn.init.zeros_(self.router_mixer.bias)

    def _compute_balance_loss(self, probs: torch.Tensor, selected_idx: torch.Tensor) -> torch.Tensor:
        if self.balance_weight <= 0:
            return probs.new_zeros(())
        importance = probs.mean(dim=(0, 1))
        selected = torch.zeros_like(probs)
        selected.scatter_(-1, selected_idx, 1.0)
        load = selected.mean(dim=(0, 1)) / float(self.top_k)
        target = probs.new_full((self.num_experts,), 1.0 / float(self.num_experts))
        balance = (importance - target).pow(2).mean() + (load - target).pow(2).mean()
        return balance * self.balance_weight

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        hidden = torch.einsum('npd,ehd->npeh', x, self.expert_fc1_weight)
        hidden = hidden + self.expert_fc1_bias.view(1, 1, self.num_experts, -1)
        hidden = torch.nn.functional.silu(hidden)
        expert_out = torch.einsum('npeh,edh->nped', hidden, self.expert_fc2_weight)
        expert_out = expert_out + self.expert_fc2_bias.view(1, 1, self.num_experts, -1)

        router_x = x + self.router_time(t_emb).unsqueeze(1)
        num_points = x.shape[1]
        if self.point_embed is not None:
            router_x = router_x + self.point_embed[:, :num_points].to(device=x.device, dtype=x.dtype)
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
        routed = (chosen * top_gates.unsqueeze(-1)).sum(dim=2)
        self._last_aux_loss = self._compute_balance_loss(probs, top_idx)
        return self.routed_scale * routed

    def reg_loss(self) -> torch.Tensor:
        if self._last_aux_loss is None:
            return self.router.weight.new_zeros(())
        return self._last_aux_loss


class DiTBlockV3(nn.Module):
    """
    Advanced DiT Block V3 (Reverted design to V2 matching Flow Matching stability)
    Sequence: Self-Attention (Coordinate) -> Cross-Attention (Image) -> FFN
    """
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_points: int = 128,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_ffn_moe: bool = False,
        ffn_moe_num_experts: int = 4,
        ffn_moe_top_k: int = 2,
        ffn_moe_hidden_dim: int = 256,
        ffn_moe_balance_weight: float = 1e-3,
        ffn_moe_router_noise_std: float = 0.01,
        ffn_moe_expert_init_std: float = 1e-4,
        ffn_moe_routed_scale: float = 1.0,
        ffn_moe_use_point_embed: bool = True,
        ffn_moe_use_cyclic_router: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Normalization
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)
        
        # QK-Norm & RoPE
        self.qk_norm = RMSNorm(head_dim)
        self.rope = CyclicRoPE1D(head_dim=head_dim, num_points=num_points)

        # Self-Attention
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.sa_out_proj = nn.Linear(dim, dim, bias=False)

        # Cross-Attention
        self.cross_q_proj = nn.Linear(dim, dim, bias=False)
        self.cross_k_proj = nn.Linear(dim, dim, bias=False)
        self.cross_v_proj = nn.Linear(dim, dim, bias=False)
        self.ca_out_proj = nn.Linear(dim, dim, bias=False)

        # MLP
        self.mlp = SwiGLU(dim=dim, dropout=dropout)
        self.use_ffn_moe = bool(use_ffn_moe)
        if self.use_ffn_moe:
            self.ffn_moe = RoutedFFNMoE(
                dim=dim,
                num_points=num_points,
                num_experts=ffn_moe_num_experts,
                top_k=ffn_moe_top_k,
                hidden_dim=ffn_moe_hidden_dim,
                balance_weight=ffn_moe_balance_weight,
                router_noise_std=ffn_moe_router_noise_std,
                expert_init_std=ffn_moe_expert_init_std,
                routed_scale=ffn_moe_routed_scale,
                use_point_embed=ffn_moe_use_point_embed,
                use_cyclic_router=ffn_moe_use_cyclic_router,
            )
        else:
            self.ffn_moe = None

        # adaLN-Zero: 9 parameters (Back to V2 Order)
        # (shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_ff, scale_ff, gate_ff)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        N, P, D = x.shape
        H = self.num_heads
        hd = self.head_dim

        q = self.q_proj(x).view(N, P, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(N, P, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(N, P, H, hd).transpose(1, 2)

        q = self.rope.apply_rotary(self.qk_norm(q))
        k = self.rope.apply_rotary(self.qk_norm(k))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(N, P, D)
        return self.sa_out_proj(out)

    def _cross_attention(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        N, P, D = x.shape
        L = context.shape[1]
        H = self.num_heads
        hd = self.head_dim

        q = self.cross_q_proj(x).view(N, P, H, hd).transpose(1, 2)
        k = self.cross_k_proj(context).view(N, L, H, hd).transpose(1, 2)
        v = self.cross_v_proj(context).view(N, L, H, hd).transpose(1, 2)

        q = self.qk_norm(q)
        k = self.qk_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(N, P, D)
        return self.ca_out_proj(out)

    def forward(self, x: torch.Tensor, image_context: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(t_emb)
        (shift_sa, scale_sa, gate_sa,
         shift_ca, scale_ca, gate_ca,
         shift_ff, scale_ff, gate_ff) = mod.chunk(9, dim=1)

        # 1. Self-Attention (Coordinate internally first - SAME AS V2)
        x_sa = modulate(self.norm1(x), shift_sa, scale_sa)
        x = x + gate_sa.unsqueeze(1) * self._self_attention(x_sa)

        # 2. Cross-Attention (Interact with Image Context - SAME AS V2)
        x_ca = modulate(self.norm2(x), shift_ca, scale_ca)
        x = x + gate_ca.unsqueeze(1) * self._cross_attention(x_ca, image_context)

        # 3. FFN (No local_smooth - SAME AS V2)
        x_ff = modulate(self.norm3(x), shift_ff, scale_ff)
        ffn_out = self.mlp(x_ff)
        if self.ffn_moe is not None:
            ffn_out = ffn_out + self.ffn_moe(x_ff, t_emb)
        x = x + gate_ff.unsqueeze(1) * ffn_out

        return x

    def reg_loss(self) -> torch.Tensor:
        if self.ffn_moe is None:
            return self.adaLN_modulation[-1].weight.new_zeros(())
        return self.ffn_moe.reg_loss()
