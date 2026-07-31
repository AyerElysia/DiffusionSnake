import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
from typing import Tuple, Optional, Dict, Any
import json

import lib.utils.snake.snake_gcn_utils as snake_gcn_utils
from lib.utils.snake import snake_config, snake_decode
from lib.config import cfg as global_cfg

class FlowMatchingEvolution(nn.Module):
    """
    Flow Matching (Rectified Flow) Evolution Wrapper
    Replaces DDPM diffusion with continuous vector field prediction.
    """
    @staticmethod
    def _detail_feature_multiplier(mode: str) -> int:
        if mode == 'normal':
            return 3
        if mode == 'normal_band':
            return 5
        if mode == 'normal_tangent':
            return 6
        raise ValueError(f"Unsupported detail_context_mode: {mode}")

    @staticmethod
    def _resolve_detail_context_mode(global_cfg) -> str:
        if bool(getattr(global_cfg, 'v4_2_use_detail_context', False)):
            return str(getattr(global_cfg, 'v4_2_detail_context_mode', 'normal_band')).strip().lower()
        if bool(getattr(global_cfg, 'v4_1_use_detail_context', False)):
            return str(getattr(global_cfg, 'v4_1_detail_context_mode', 'normal')).strip().lower()
        if bool(getattr(global_cfg, 'v4_use_detail_context', False)):
            return str(getattr(global_cfg, 'v4_detail_context_mode', 'normal')).strip().lower()
        if bool(getattr(global_cfg, 'v3_4_use_detail_context', False)):
            return str(getattr(global_cfg, 'v3_4_detail_context_mode', 'normal')).strip().lower()
        if bool(getattr(global_cfg, 'v3_7_use_detail_context', False)):
            return str(getattr(global_cfg, 'v3_7_detail_context_mode', 'normal')).strip().lower()
        return 'normal'

    def __init__(
        self,
        state_dim: int = 128,
        feature_dim: int = 64,
        num_points: int = 128,
        loss_weight: float = 1.0,
        loss_type: str = 'adaptive',
        dit_num_layers: int = 6,
        dit_num_heads: int = 8,
        dit_state_dim: int = 256,
        ode_steps: int = 10,
    ):
        super().__init__()
        self.num_points = num_points
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.ode_steps = ode_steps
        self._use_s_cond = bool(getattr(global_cfg, 'flow_2d_s_conditioning', False))
        self.use_iterative_refinement = bool(
            getattr(global_cfg, 'use_iterative_refinement', False)
            or getattr(global_cfg, 'use_dit_v3_4', False)
            or getattr(global_cfg, 'use_dit_v4', False)
            or getattr(global_cfg, 'use_dit_v4_1', False)
            or getattr(global_cfg, 'use_dit_v4_2', False)
        )
        self.use_fourier_smooth = int(getattr(global_cfg, 'fourier_smooth_k', 0))

        # V3.7: Per-Point Output Head (per-point embedding + per-point linear)
        if getattr(global_cfg, 'use_dit_v3_7', False):
            from .dit_denoiser_v3_7 import DiTFlowMatchingV3_7
            _pt_scale = float(getattr(global_cfg, 'v3_7_point_embed_scale', 0.1))
            _lap_w = float(getattr(global_cfg, 'v3_7_laplacian_weight', 0.0))
            _inject_in = bool(getattr(global_cfg, 'v3_7_inject_at_input', False))
            _inject_out = bool(getattr(global_cfg, 'v3_7_inject_at_output', False))
            _per_pt = bool(getattr(global_cfg, 'v3_7_use_per_point_head', True))
            _f64_head = bool(getattr(global_cfg, 'v3_7_use_float64_head', False))
            _reg_pt = bool(getattr(global_cfg, 'v3_7_use_regularized_per_point', False))
            _delta_scale = float(getattr(global_cfg, 'v3_7_delta_scale', 0.1))
            _delta_reg = float(getattr(global_cfg, 'v3_7_delta_reg_weight', 0.001))
            _scale_cond = bool(getattr(global_cfg, 'v3_7_use_scale_conditioning', False))
            _detail_ctx = bool(getattr(global_cfg, 'v3_7_use_detail_context', False))
            _detail_curve_ctx = bool(getattr(global_cfg, 'v3_7_use_detail_curve_context', False))
            _detail_curve_inject_mode = str(
                getattr(global_cfg, 'v3_7_detail_curve_inject_mode', 'both')
            ).strip().lower()
            _detail_mode = str(getattr(global_cfg, 'v3_7_detail_context_mode', 'normal')).strip().lower()
            _detail_mult = self._detail_feature_multiplier(_detail_mode) if _detail_ctx else 0
            _global_ctx_mode = str(getattr(global_cfg, 'v3_7_global_context_mode', 'patch')).strip().lower()
            _global_queries = int(getattr(global_cfg, 'v3_7_global_num_queries', 256))
            print(f"[FlowMatchingEvolution] Using DiT Flow Network V3.7 "
                  f"(per_point_head={_per_pt}, regularized={_reg_pt}, "
                  f"float64_head={_f64_head}, "
                  f"inject_in={_inject_in}, inject_out={_inject_out}, "
                  f"scale_cond={_scale_cond}, detail_ctx={_detail_ctx}, "
                  f"detail_curve_ctx={_detail_curve_ctx}, "
                  f"detail_mode={_detail_mode}, "
                  f"curve_inject={_detail_curve_inject_mode}, "
                  f"global_ctx={_global_ctx_mode}, global_queries={_global_queries}, "
                  f"ODE steps={ode_steps})")
            self.denoiser = DiTFlowMatchingV3_7(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
                use_per_point_head=_per_pt,
                use_float64_head=_f64_head,
                use_regularized_per_point=_reg_pt,
                delta_scale=_delta_scale,
                delta_reg_weight=_delta_reg,
                point_embed_scale=_pt_scale,
                laplacian_weight=_lap_w,
                inject_at_input=_inject_in,
                inject_at_output=_inject_out,
                use_scale_conditioning=_scale_cond,
                use_detail_context=_detail_ctx,
                use_detail_curve_context=_detail_curve_ctx,
                detail_curve_inject_mode=_detail_curve_inject_mode,
                detail_feature_dim=feature_dim * _detail_mult,
                use_self_conditioning=bool(getattr(global_cfg, 'v3_7_use_self_conditioning', False)),
                global_context_mode=_global_ctx_mode,
                global_num_queries=_global_queries,
            )
        elif getattr(global_cfg, 'use_dit_v3_1', False):
            from .dit_denoiser_v3_1 import DiTDenoiserV3_1
            print(f"[FlowMatchingEvolution] Using DiT Flow Network V3.1 "
                  f"(Patchify + Self->Cross, ODE steps={ode_steps})")
            self.denoiser = DiTDenoiserV3_1(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
            )
        elif getattr(global_cfg, 'use_dit_v4_2', False):
            from .dit_denoiser_v4_2 import DiTFlowMatchingV4_2
            _detail_ctx = bool(
                getattr(global_cfg, 'v4_2_use_detail_context', False)
                or getattr(global_cfg, 'v4_1_use_detail_context', False)
                or getattr(global_cfg, 'v3_4_use_detail_context', False)
            )
            _detail_mode = self._resolve_detail_context_mode(global_cfg)
            _detail_mult = self._detail_feature_multiplier(_detail_mode) if _detail_ctx else 1
            _use_pp_delta = bool(getattr(global_cfg, 'v4_2_use_per_point_delta', True))
            _pp_delta_scale = float(getattr(global_cfg, 'v4_2_per_point_delta_scale', 0.10))
            _pp_delta_reg = float(getattr(global_cfg, 'v4_2_per_point_delta_reg_weight', 0.0))
            _use_curv_cond = bool(getattr(global_cfg, 'v4_2_use_curvature_conditioning', True))
            _curv_scale = float(getattr(global_cfg, 'v4_2_curvature_embed_scale', 0.10))
            _use_delta_gate = bool(getattr(global_cfg, 'v4_2_use_delta_gate', True))
            _delta_gate_bias = float(getattr(global_cfg, 'v4_2_delta_gate_bias', -2.0))
            print(f"[FlowMatchingEvolution] Using DiT Flow Network V4.2 "
                  f"(detail_ctx={_detail_ctx}, detail_mode={_detail_mode}, "
                  f"per_point_delta={_use_pp_delta}, delta_scale={_pp_delta_scale}, "
                  f"delta_reg={_pp_delta_reg}, curvature_cond={_use_curv_cond}, "
                  f"delta_gate={_use_delta_gate}, ODE steps={ode_steps})")
            self.denoiser = DiTFlowMatchingV4_2(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
                use_detail_context=_detail_ctx,
                detail_feature_dim=feature_dim * _detail_mult,
                use_per_point_delta=_use_pp_delta,
                per_point_delta_scale=_pp_delta_scale,
                per_point_delta_reg_weight=_pp_delta_reg,
                use_curvature_conditioning=_use_curv_cond,
                curvature_embed_scale=_curv_scale,
                use_delta_gate=_use_delta_gate,
                delta_gate_bias=_delta_gate_bias,
            )
        elif getattr(global_cfg, 'use_dit_v4_1', False):
            from .dit_denoiser_v4_1 import DiTFlowMatchingV4_1
            _detail_ctx = bool(
                getattr(global_cfg, 'v4_1_use_detail_context', False)
                or getattr(global_cfg, 'v3_4_use_detail_context', False)
            )
            _detail_mode = self._resolve_detail_context_mode(global_cfg)
            _detail_mult = self._detail_feature_multiplier(_detail_mode) if _detail_ctx else 1
            _use_pp_delta = bool(getattr(global_cfg, 'v4_1_use_per_point_delta', True))
            _pp_delta_scale = float(getattr(global_cfg, 'v4_1_per_point_delta_scale', 0.10))
            _pp_delta_reg = float(getattr(global_cfg, 'v4_1_per_point_delta_reg_weight', 0.0))
            _pp_delta_head_type = str(getattr(global_cfg, 'v4_1_per_point_delta_head_type', 'linear')).strip().lower()
            _pp_delta_hidden_mult = float(getattr(global_cfg, 'v4_1_per_point_delta_hidden_mult', 2.0))
            _pp_delta_cyclic = bool(getattr(global_cfg, 'v4_1_per_point_delta_use_cyclic_mixer', True))
            _final_head_type = str(getattr(global_cfg, 'v4_1_final_head_type', 'standard')).strip().lower()
            _moe_enabled = _final_head_type in ('moe', 'moe_final', 'deepseek_moe')
            _moe_num_experts = int(getattr(global_cfg, 'v4_6_moe_num_experts', 8))
            _moe_top_k = int(getattr(global_cfg, 'v4_6_moe_top_k', 2))
            _moe_balance_weight = float(getattr(global_cfg, 'v4_6_moe_balance_weight', 1e-3))
            _moe_balance_mode = str(
                getattr(global_cfg, 'v4_6_moe_balance_mode', 'legacy')
            ).strip().lower()
            _moe_hard_phi_ema = float(
                getattr(global_cfg, 'v4_6_moe_hard_phi_ema_decay', 0.99)
            )
            _moe_expert_init_std = float(getattr(global_cfg, 'v4_6_moe_expert_init_std', 1e-4))
            _moe_router_noise_std = float(getattr(global_cfg, 'v4_6_moe_router_noise_std', 0.01))
            _moe_point_embed = bool(getattr(global_cfg, 'v4_6_moe_use_point_embed', True))
            _moe_cyclic_router = bool(getattr(global_cfg, 'v4_6_moe_use_cyclic_router', True))
            _moe_shared_expert = bool(getattr(global_cfg, 'v4_6_moe_use_shared_expert', False))
            _moe_route_shared = bool(getattr(global_cfg, 'v4_6_moe_route_shared_expert', False))
            _moe_route_shared_bias = float(getattr(global_cfg, 'v4_6_moe_route_shared_init_bias', 0.0))
            _moe_routed_scale = float(getattr(global_cfg, 'v4_6_moe_routed_expert_scale', 1.0))
            _moe_expert_type = str(getattr(global_cfg, 'v4_6_moe_expert_type', 'linear')).strip().lower()
            _moe_expert_hidden = int(getattr(global_cfg, 'v4_6_moe_expert_hidden_dim', 256))
            _latent_loop = bool(getattr(global_cfg, 'v4_7_use_latent_loop', False))
            _latent_loop_steps = int(getattr(global_cfg, 'v4_7_latent_loop_steps', 4))
            _dit_ffn_moe = bool(getattr(global_cfg, 'v4_10_use_dit_ffn_moe', False))
            _dit_ffn_moe_experts = int(getattr(global_cfg, 'v4_10_dit_ffn_moe_num_experts', 4))
            _dit_ffn_moe_top_k = int(getattr(global_cfg, 'v4_10_dit_ffn_moe_top_k', 2))
            _dit_ffn_moe_hidden = int(getattr(global_cfg, 'v4_10_dit_ffn_moe_hidden_dim', 256))
            _dit_ffn_moe_balance = float(getattr(global_cfg, 'v4_10_dit_ffn_moe_balance_weight', 1e-3))
            _dit_ffn_moe_router_noise = float(getattr(global_cfg, 'v4_10_dit_ffn_moe_router_noise_std', 0.01))
            _dit_ffn_moe_init_std = float(getattr(global_cfg, 'v4_10_dit_ffn_moe_expert_init_std', 1e-4))
            _dit_ffn_moe_scale = float(getattr(global_cfg, 'v4_10_dit_ffn_moe_routed_scale', 1.0))
            _dit_ffn_moe_point = bool(getattr(global_cfg, 'v4_10_dit_ffn_moe_use_point_embed', True))
            _dit_ffn_moe_cyclic = bool(getattr(global_cfg, 'v4_10_dit_ffn_moe_use_cyclic_router', True))
            _proto_moe = bool(getattr(global_cfg, 'v5_1_use_prototype_phi_moe', False))
            _proto_layers = str(getattr(global_cfg, 'v5_1_prototype_phi_moe_layers', 'odd'))
            _proto_experts = int(getattr(global_cfg, 'v5_1_prototype_phi_num_experts', 4))
            _proto_top_k = int(getattr(global_cfg, 'v5_1_prototype_phi_top_k', 1))
            _proto_hidden = int(getattr(global_cfg, 'v5_1_prototype_phi_hidden_dim', 0))
            _proto_temp = float(getattr(global_cfg, 'v5_1_prototype_phi_router_temperature', 0.20))
            _proto_balance = float(getattr(global_cfg, 'v5_1_prototype_phi_balance_weight', 1e-3))
            _proto_ema = float(getattr(global_cfg, 'v5_1_prototype_phi_ema_decay', 0.99))
            _proto_contrast = float(getattr(global_cfg, 'v5_1_prototype_phi_contrastive_weight', 1e-3))
            _moe_summary = (
                'final_head_moe=True(experts={}, top_k={}, balance={})'.format(
                    _moe_num_experts, _moe_top_k, _moe_balance_mode
                )
                if _moe_enabled
                else 'final_head_moe=False'
            )
            _ffn_moe_summary = (
                'dit_ffn_moe=True(experts={}, top_k={})'.format(
                    _dit_ffn_moe_experts, _dit_ffn_moe_top_k
                )
                if _dit_ffn_moe
                else 'dit_ffn_moe=False'
            )
            _proto_summary = (
                'prototype_phi_moe=True(layers={}, experts={}, top_k={})'.format(
                    _proto_layers, _proto_experts, _proto_top_k
                )
                if _proto_moe else 'prototype_phi_moe=False'
            )
            print(f"[FlowMatchingEvolution] Using DiT Flow Network V4.1 "
                  f"(detail_ctx={_detail_ctx}, "
                  f"detail_mode={_detail_mode}, per_point_delta={_use_pp_delta}, "
                  f"delta_head={_pp_delta_head_type}, delta_scale={_pp_delta_scale}, "
                  f"delta_reg={_pp_delta_reg}, "
                  f"final_head={_final_head_type}, {_moe_summary}, "
                  f"latent_loop={_latent_loop}, latent_loop_steps={_latent_loop_steps}, "
                  f"s_cond={self._use_s_cond}, {_ffn_moe_summary}, {_proto_summary}, "
                  f"ODE steps={ode_steps})")
            self.denoiser = DiTFlowMatchingV4_1(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
                use_detail_context=_detail_ctx,
                detail_feature_dim=feature_dim * _detail_mult,
                use_per_point_delta=_use_pp_delta,
                per_point_delta_scale=_pp_delta_scale,
                per_point_delta_reg_weight=_pp_delta_reg,
                per_point_delta_head_type=_pp_delta_head_type,
                per_point_delta_hidden_mult=_pp_delta_hidden_mult,
                per_point_delta_use_cyclic_mixer=_pp_delta_cyclic,
                final_head_type=_final_head_type,
                moe_num_experts=_moe_num_experts,
                moe_top_k=_moe_top_k,
                moe_balance_weight=_moe_balance_weight,
                moe_balance_mode=_moe_balance_mode,
                moe_hard_phi_ema_decay=_moe_hard_phi_ema,
                moe_expert_init_std=_moe_expert_init_std,
                moe_router_noise_std=_moe_router_noise_std,
                moe_use_point_embed=_moe_point_embed,
                moe_use_cyclic_router=_moe_cyclic_router,
                moe_use_shared_expert=_moe_shared_expert,
                moe_route_shared_expert=_moe_route_shared,
                moe_route_shared_init_bias=_moe_route_shared_bias,
                moe_routed_expert_scale=_moe_routed_scale,
                moe_expert_type=_moe_expert_type,
                moe_expert_hidden_dim=_moe_expert_hidden,
                use_latent_loop=_latent_loop,
                latent_loop_steps=_latent_loop_steps,
                use_s_conditioning=self._use_s_cond,
                use_ffn_moe=_dit_ffn_moe,
                ffn_moe_num_experts=_dit_ffn_moe_experts,
                ffn_moe_top_k=_dit_ffn_moe_top_k,
                ffn_moe_hidden_dim=_dit_ffn_moe_hidden,
                ffn_moe_balance_weight=_dit_ffn_moe_balance,
                ffn_moe_router_noise_std=_dit_ffn_moe_router_noise,
                ffn_moe_expert_init_std=_dit_ffn_moe_init_std,
                ffn_moe_routed_scale=_dit_ffn_moe_scale,
                ffn_moe_use_point_embed=_dit_ffn_moe_point,
                ffn_moe_use_cyclic_router=_dit_ffn_moe_cyclic,
                use_prototype_phi_moe=_proto_moe,
                prototype_phi_moe_layers=_proto_layers,
                prototype_phi_num_experts=_proto_experts,
                prototype_phi_top_k=_proto_top_k,
                prototype_phi_hidden_dim=_proto_hidden,
                prototype_phi_router_temperature=_proto_temp,
                prototype_phi_balance_weight=_proto_balance,
                prototype_phi_ema_decay=_proto_ema,
                prototype_phi_contrastive_weight=_proto_contrast,
            )
        elif getattr(global_cfg, 'use_dit_v4', False):
            from .dit_denoiser_v4 import DiTFlowMatchingV4
            _detail_ctx = bool(
                getattr(global_cfg, 'v4_use_detail_context', False)
                or getattr(global_cfg, 'v3_4_use_detail_context', False)
                or getattr(global_cfg, 'v3_7_use_detail_context', False)
            )
            _detail_mode = self._resolve_detail_context_mode(global_cfg)
            _detail_mult = self._detail_feature_multiplier(_detail_mode) if _detail_ctx else 1
            _use_pp_delta = bool(getattr(global_cfg, 'v4_use_per_point_delta', True))
            _pp_delta_scale = float(getattr(global_cfg, 'v4_per_point_delta_scale', 0.25))
            _pp_delta_reg = float(getattr(global_cfg, 'v4_per_point_delta_reg_weight', 0.0))
            print(f"[FlowMatchingEvolution] Using DiT Flow Network V4.0 "
                  f"(detail_ctx={_detail_ctx}, detail_mode={_detail_mode}, "
                  f"per_point_delta={_use_pp_delta}, delta_scale={_pp_delta_scale}, "
                  f"delta_reg={_pp_delta_reg}, ODE steps={ode_steps})")
            self.denoiser = DiTFlowMatchingV4(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
                use_detail_context=_detail_ctx,
                detail_feature_dim=feature_dim * _detail_mult,
                use_per_point_delta=_use_pp_delta,
                per_point_delta_scale=_pp_delta_scale,
                per_point_delta_reg_weight=_pp_delta_reg,
            )
        # V3.4: keep the original V3 backbone and iterative refinement, only swap to FM
        elif getattr(global_cfg, 'use_dit_v3_4', False):
            from .dit_denoiser_v3_4 import DiTFlowMatchingV3_4
            _detail_ctx = bool(
                getattr(global_cfg, 'v3_4_use_detail_context', False)
                or getattr(global_cfg, 'v3_7_use_detail_context', False)
            )
            _detail_mode = str(
                getattr(
                    global_cfg,
                    'v3_4_detail_context_mode',
                    getattr(global_cfg, 'v3_7_detail_context_mode', 'normal'),
                )
            ).strip().lower()
            _detail_mult = self._detail_feature_multiplier(_detail_mode) if _detail_ctx else 1
            print(f"[FlowMatchingEvolution] Using DiT Flow Network V3.4 "
                  f"(V3 backbone + iterative refinement, detail_ctx={_detail_ctx}, "
                  f"detail_mode={_detail_mode}, ODE steps={ode_steps})")
            self.denoiser = DiTFlowMatchingV3_4(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
                use_detail_context=_detail_ctx,
                detail_feature_dim=feature_dim * _detail_mult,
            )
        # V3.6: V3 global query + iterative refinement + Flow Matching
        elif getattr(global_cfg, 'use_dit_v3_6', False):
            from .dit_denoiser_v3_6 import DiTFlowMatchingV3_6
            print(f"[FlowMatchingEvolution] Using DiT Flow Network V3.6 "
                  f"(V3 global query + iterative refinement, ODE steps={ode_steps})")
            self.denoiser = DiTFlowMatchingV3_6(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
            )
        # V3.2: Efficient Self+Cross Attention with Flow Matching
        elif getattr(global_cfg, 'use_dit_v3_2', False):
            from .dit_denoiser_v3_2 import DiTFlowMatchingV3_2
            print(f"[FlowMatchingEvolution] Using DiT Flow Network V3.2 "
                  f"(Self+Cross + Patchify, ODE steps={ode_steps})")
            self.denoiser = DiTFlowMatchingV3_2(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
            )
        # Default fallback after V2 archive: use V3.2 flow denoiser.
        else:
            from .dit_denoiser_v3_2 import DiTFlowMatchingV3_2
            print(f"[FlowMatchingEvolution] Using default DiT Flow Network V3.2 "
                  f"(Self+Cross + Patchify, ODE steps={ode_steps})")
            self.denoiser = DiTFlowMatchingV3_2(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
            )

        self._locate_token_enabled = bool(getattr(global_cfg, 'use_locate_token_dit', False))
        self._locate_token_only = bool(getattr(global_cfg, 'v11_locate_only', True))
        self._locate_token_dim = int(dit_state_dim)
        self._locate_global_tokens = int(getattr(global_cfg, 'v11_locate_global_tokens', 64))
        self._locate_roi_grid = int(getattr(global_cfg, 'v11_locate_roi_grid', 6))
        self._locate_use_roi_tokens = bool(getattr(global_cfg, 'v11_locate_use_roi_tokens', True))
        self._locate_use_normal_tangent = bool(getattr(global_cfg, 'v11_locate_use_normal_tangent', True))
        self.locate_feat_proj = None
        self.locate_point_mixer = None
        self.locate_global_queries = None
        self.locate_global_attn = None
        self.locate_global_norm = None
        if self._locate_token_enabled:
            locate_in_dim = int(getattr(global_cfg, 'locate_feat_dim', 2304))
            point_samples = 5 if self._locate_use_normal_tangent else 1
            self.locate_feat_proj = nn.Sequential(
                nn.Conv2d(locate_in_dim, self._locate_token_dim, kernel_size=1, bias=False),
                nn.GroupNorm(8, self._locate_token_dim),
                nn.GELU(),
                nn.Conv2d(self._locate_token_dim, self._locate_token_dim, kernel_size=3, padding=1, bias=True),
            )
            self.locate_point_mixer = nn.Sequential(
                nn.Linear(self._locate_token_dim * point_samples, self._locate_token_dim),
                nn.SiLU(),
                nn.Linear(self._locate_token_dim, self._locate_token_dim),
            )
            self.locate_global_queries = nn.Embedding(self._locate_global_tokens, self._locate_token_dim)
            self.locate_global_attn = nn.MultiheadAttention(
                embed_dim=self._locate_token_dim,
                num_heads=max(1, int(dit_num_heads)),
                batch_first=True,
            )
            self.locate_global_norm = nn.LayerNorm(self._locate_token_dim)
            print(
                f"[FlowMatchingEvolution] V11 LocateToken-DiT enabled "
                f"(locate_only={self._locate_token_only}, dim={self._locate_token_dim}, "
                f"global_tokens={self._locate_global_tokens}, roi_grid={self._locate_roi_grid}, "
                f"normal_tangent={self._locate_use_normal_tangent})",
                flush=True,
            )

        def _env_bool(name: str, default: bool) -> bool:
            raw = os.environ.get(name, '').strip()
            if not raw:
                return bool(default)
            return raw.lower() in ('1', 'true', 'yes', 'y', 'on')

        self._use_latent_policy = _env_bool(
            'FLOW_USE_LATENT_POLICY',
            bool(getattr(global_cfg, 'flow_use_latent_policy', False)),
        )
        self._use_latent_ranker = _env_bool(
            'FLOW_USE_LATENT_RANKER',
            bool(getattr(global_cfg, 'flow_use_latent_ranker', False)),
        )
        if self._use_latent_ranker:
            self._use_latent_policy = True
        self._latent_policy_scale = float(os.environ.get(
            'FLOW_LATENT_POLICY_SCALE',
            getattr(global_cfg, 'flow_latent_policy_scale', 0.10),
        ))
        self._latent_logprob_scale = float(os.environ.get(
            'FLOW_LATENT_LOGPROB_SCALE',
            getattr(global_cfg, 'flow_latent_logprob_scale', 1.0),
        ))
        self._latent_policy_eval_mode = str(
            os.environ.get(
                'FLOW_LATENT_POLICY_EVAL_MODE',
                getattr(global_cfg, 'flow_latent_policy_eval_mode', 'sample'),
            )
        ).strip().lower()
        self._latent_selector_k = int(os.environ.get(
            'FLOW_LATENT_SELECTOR_K',
            getattr(global_cfg, 'flow_latent_selector_k', 8),
        ))
        self.latent_policy = None
        if self._use_latent_policy:
            hidden = int(getattr(global_cfg, 'flow_latent_policy_hidden_dim', feature_dim))
            self.latent_policy = nn.Sequential(
                nn.Conv1d(feature_dim, hidden, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv1d(hidden, 2, kernel_size=1),
            )
            nn.init.zeros_(self.latent_policy[-1].weight)
            nn.init.zeros_(self.latent_policy[-1].bias)
            print(
                f"[FlowMatchingEvolution] Latent x0 policy enabled "
                f"(hidden={hidden}, scale={self._latent_policy_scale}, "
                f"logprob_scale={self._latent_logprob_scale})"
            )
        self.latent_ranker = None
        self.latent_ranker_head = None
        if self._use_latent_ranker:
            rank_hidden = int(os.environ.get(
                'FLOW_LATENT_RANKER_HIDDEN_DIM',
                getattr(global_cfg, 'flow_latent_ranker_hidden_dim', feature_dim),
            ))
            self.latent_ranker = nn.Sequential(
                nn.Conv1d(feature_dim + 2, rank_hidden, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv1d(rank_hidden, rank_hidden, kernel_size=1),
                nn.SiLU(inplace=True),
            )
            self.latent_ranker_head = nn.Linear(rank_hidden, 1)
            nn.init.zeros_(self.latent_ranker_head.weight)
            nn.init.zeros_(self.latent_ranker_head.bias)
            print(
                f"[FlowMatchingEvolution] Latent x0 ranker enabled "
                f"(hidden={rank_hidden}, selector_k={self._latent_selector_k}, "
                f"eval_mode={self._latent_policy_eval_mode})"
            )

        # V6s: optimal cyclic alignment — find cyclic shift that minimises total displacement MSE
        # (greedy nearest-first-point can be far from optimal for tortuous contours)
        self._optimal_cyclic_align = bool(getattr(global_cfg, 'v3_7_optimal_cyclic_align', False))
        # V3.7: per-step ODE smoothing
        self._ode_smooth_k = int(getattr(global_cfg, 'v3_7_ode_smooth_k', 0))
        # V6p: ODE solver selection
        # 'euler'  — 1st-order Euler (default)
        # 'heun'   — 2nd-order Runge-Kutta predictor-corrector (2x NFE per step)
        # 'ab2'    — 2nd-order Adams-Bashforth (1x NFE per step, same cost as Euler)
        self._ode_solver = str(getattr(global_cfg, 'v3_7_ode_solver', 'euler')).strip().lower()
        # V3.7: spectral loss decomposition
        self._spectral_loss_k = int(getattr(global_cfg, 'v3_7_spectral_loss_k', 0))
        self._hf_loss_weight = float(getattr(global_cfg, 'v3_7_hf_loss_weight', 0.1))
        # V6o: endpoint consistency loss — weight on L_endpoint = (1-t)^2 * FM_loss
        self._endpoint_loss_weight = float(getattr(global_cfg, 'v3_7_endpoint_loss_weight', 0.0))
        # V3.7: curvature-aware point weighting for high-curvature detail
        self._use_curvature_reweight = bool(
            getattr(global_cfg, 'v4_2_use_curvature_reweight', False)
            or
            getattr(global_cfg, 'v4_1_use_curvature_reweight', False)
            or getattr(global_cfg, 'v3_7_use_curvature_reweight', False)
        )
        if bool(getattr(global_cfg, 'v4_2_use_curvature_reweight', False)):
            self._curvature_loss_weight = float(getattr(global_cfg, 'v4_2_curvature_loss_weight', 1.5))
            self._curvature_reweight_power = float(getattr(global_cfg, 'v4_2_curvature_reweight_power', 1.0))
        elif bool(getattr(global_cfg, 'v4_1_use_curvature_reweight', False)):
            self._curvature_loss_weight = float(getattr(global_cfg, 'v4_1_curvature_loss_weight', 1.5))
            self._curvature_reweight_power = float(getattr(global_cfg, 'v4_1_curvature_reweight_power', 1.0))
        else:
            self._curvature_loss_weight = float(getattr(global_cfg, 'v3_7_curvature_loss_weight', 1.0))
            self._curvature_reweight_power = float(getattr(global_cfg, 'v3_7_curvature_reweight_power', 1.0))
        self._use_detail_context = bool(
            getattr(global_cfg, 'v4_2_use_detail_context', False)
            or
            getattr(global_cfg, 'v4_1_use_detail_context', False)
            or getattr(global_cfg, 'v4_use_detail_context', False)
            or getattr(global_cfg, 'v3_4_use_detail_context', False)
            or getattr(global_cfg, 'v3_7_use_detail_context', False)
        )
        self._detail_context_mode = self._resolve_detail_context_mode(global_cfg)
        # V3.7.3: low-noise flow matching (default 1.0 = standard)
        self._flow_noise_scale = float(getattr(global_cfg, 'flow_noise_scale', 1.0))
        # Optional training-only noise scale (fallback to flow_noise_scale)
        self._flow_train_noise_scale = float(
            getattr(global_cfg, 'flow_train_noise_scale', self._flow_noise_scale)
        )
        # V3.7.3: inference averaging
        self._infer_avg_samples = int(getattr(global_cfg, 'infer_avg_samples', 0))
        self._infer_noise_scale = float(getattr(global_cfg, 'infer_noise_scale', -1.0))
        self._step_logprob_mode = str(
            getattr(global_cfg, 'flow_step_logprob_mode', 'gaussian')
        ).strip().lower()
        self._step_noise_level = float(getattr(global_cfg, 'flow_step_noise_level', 0.8))
        self._step_sde_type = str(
            getattr(global_cfg, 'flow_step_sde_type', 'sde')
        ).strip().lower()
        # Per-step noise annealing: list of noise scales for each iterative refinement step
        # e.g. [0.7, 0.5, 0.3] for coarse-to-fine. Empty = use infer_noise_scale for all steps.
        _itn = getattr(global_cfg, 'iterative_noise_scales', [])
        self._iterative_noise_scales = list(_itn) if _itn else []
        # V3.7.6: fix t=0 during training (pure direct regression)
        self._flow_fix_t0 = bool(getattr(global_cfg, 'flow_fix_t0', False))
        if bool(getattr(global_cfg, 'use_dit_v4_2', False)):
            self._small_disp_prob = float(getattr(global_cfg, 'v4_2_small_disp_prob', 0.0))
            self._small_disp_min_frac = float(getattr(global_cfg, 'v4_2_small_disp_min_frac', 0.80))
            self._small_disp_max_frac = float(getattr(global_cfg, 'v4_2_small_disp_max_frac', 0.95))
        else:
            self._small_disp_prob = float(getattr(global_cfg, 'v4_1_small_disp_prob', 0.0))
            self._small_disp_min_frac = float(getattr(global_cfg, 'v4_1_small_disp_min_frac', 0.80))
            self._small_disp_max_frac = float(getattr(global_cfg, 'v4_1_small_disp_max_frac', 0.95))
        # V4.3: soft Chamfer loss — bidirectional soft nearest-neighbour on predicted endpoint contour
        self._chamfer_loss_weight = float(getattr(global_cfg, 'v4_3_chamfer_loss_weight', 0.0))
        self._chamfer_tau = float(getattr(global_cfg, 'v4_3_chamfer_tau', 0.05))
        # V3.7/V6b-style generalization knobs used by the standalone scripts.
        self._use_contour_norm = bool(getattr(global_cfg, 'v3_7_use_contour_norm', False))
        self._flow_zero_x0_prob = float(getattr(global_cfg, 'flow_zero_x0_prob', 0.0))
        self._flow_t_beta_alpha = float(getattr(global_cfg, 'flow_t_beta_alpha', 1.0))
        self._flow_t_beta_beta = float(getattr(global_cfg, 'flow_t_beta_beta', 1.0))
        # V6p: logit-normal t-sampling (SD3-style: t = sigmoid(N(mean, std^2)))
        # Use v3_7_t_sample_mode='logit_normal' to enable; defaults to uniform
        self._t_sample_mode = str(getattr(global_cfg, 'v3_7_t_sample_mode', 'uniform')).strip().lower()
        self._logit_normal_mean = float(getattr(global_cfg, 'v3_7_logit_normal_mean', 0.0))
        self._logit_normal_std = float(getattr(global_cfg, 'v3_7_logit_normal_std', 1.0))
        # V6r: self-conditioning — model conditions on its own previous x1 estimate
        # At training: 50% of steps do a "dry run" first, then condition on that result
        # At inference: always condition on previous ODE step's x1 prediction
        self._use_self_conditioning = bool(getattr(global_cfg, 'v3_7_use_self_conditioning', False))
        self._max_disp_frac = float(getattr(global_cfg, 'fm_max_disp_frac', 0.0))
        self._geom_bridge = bool(getattr(global_cfg, 'flow_geom_bridge', False))
        self._resample_feat_at_xt = bool(getattr(global_cfg, 'flow_resample_feat_at_xt', False))
        self._geom_sched_sampling = bool(getattr(global_cfg, 'flow_geom_sched_sampling', False))
        self._geom_sched_inner_steps = int(getattr(global_cfg, 'flow_geom_sched_inner_steps', 2))
        self._geom_sched_prob = float(getattr(global_cfg, 'flow_geom_sched_prob', 1.0))
        self._geom_infer_resample_per_ode = bool(
            getattr(global_cfg, 'flow_geom_infer_resample_per_ode_step', False)
        )
        self._geom_position_flow = bool(getattr(global_cfg, 'flow_geom_position_flow', False))
        self._geom_seg_flow = bool(getattr(global_cfg, 'flow_geom_seg_flow', False))
        self._geom_t_init = float(getattr(global_cfg, 'flow_geom_t_init', 0.5))
        self._geom_noise_scale = float(getattr(global_cfg, 'flow_geom_noise_scale', 1.0))
        self._geom_x0_jitter = float(getattr(global_cfg, 'flow_geom_x0_jitter', 0.0))
        self._geom_xt_jitter_rel = float(getattr(global_cfg, 'flow_geom_xt_jitter_rel', 0.0))

        # Optional supervised displacement gate. The gate predicts how much of
        # the proposed residual should be applied for contours that are already
        # close to the target or whose predicted residual points the wrong way.
        self._use_disp_gate = bool(getattr(global_cfg, 'flow_use_disp_gate', False))
        self._disp_gate_apply_inference = bool(getattr(global_cfg, 'flow_disp_gate_apply_inference', True))
        self._disp_gate_apply_training_pred = bool(getattr(global_cfg, 'flow_disp_gate_apply_training_pred', False))
        self._disp_gate_loss_weight = float(getattr(global_cfg, 'flow_disp_gate_loss_weight', 0.0))
        self._disp_gate_detach_input = bool(getattr(global_cfg, 'flow_disp_gate_detach_input', True))
        if self._use_disp_gate:
            gate_hidden = int(getattr(global_cfg, 'flow_disp_gate_hidden_dim', 128))
            gate_hidden = max(gate_hidden, 16)
            self.disp_gate_head = nn.Sequential(
                nn.Linear(feature_dim + 4, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, 1),
            )
            # Start close to the legacy behavior: gate ~= 1.
            gate_bias = float(getattr(global_cfg, 'flow_disp_gate_init_bias', 4.0))
            nn.init.zeros_(self.disp_gate_head[-1].weight)
            nn.init.constant_(self.disp_gate_head[-1].bias, gate_bias)
        else:
            self.disp_gate_head = None

        # CMAM 先验参数保留 (以防外部调用)
        self.compute_L = True

        self._disp_norm_enabled = bool(getattr(global_cfg, 'diffusion_disp_norm', False))
        self._disp_min = None
        self._disp_max = None
        self._load_disp_stats(getattr(global_cfg, 'diffusion_disp_stats', ''))

    def _load_disp_stats(self, stats_path: str) -> None:
        if (not self._disp_norm_enabled) or (not stats_path):
            return
        try:
            if not os.path.exists(stats_path):
                raise FileNotFoundError(f"diffusion_disp_norm enabled but stats file not found: {stats_path}")
            with open(stats_path, 'r') as f:
                s = json.load(f)
            dx_min = float(s['dx_min'])
            dx_max = float(s['dx_max'])
            dy_min = float(s['dy_min'])
            dy_max = float(s['dy_max'])
            disp_min = torch.tensor([dx_min, dy_min], dtype=torch.float32).view(1, 1, 2)
            disp_max = torch.tensor([dx_max, dy_max], dtype=torch.float32).view(1, 1, 2)
            
            if hasattr(self, '_disp_min') and (not isinstance(getattr(self, '_disp_min', None), torch.Tensor)):
                try: delattr(self, '_disp_min')
                except Exception: pass
            if hasattr(self, '_disp_max') and (not isinstance(getattr(self, '_disp_max', None), torch.Tensor)):
                try: delattr(self, '_disp_max')
                except Exception: pass
            self.register_buffer('_disp_min', disp_min)
            self.register_buffer('_disp_max', disp_max)
        except Exception as e:
            self._disp_norm_enabled = False
            raise

    def _has_disp_stats(self) -> bool:
        return (
            self._disp_norm_enabled
            and isinstance(getattr(self, '_disp_min', None), torch.Tensor)
            and isinstance(getattr(self, '_disp_max', None), torch.Tensor)
        )

    def normalize_disp(self, disp: torch.Tensor) -> torch.Tensor:
        if not self._has_disp_stats():
            return disp
        denom = (self._disp_max - self._disp_min).clamp_min(1e-12)
        return (disp - self._disp_min.to(disp.device, disp.dtype)) * (2.0 / denom.to(disp.device, disp.dtype)) - 1.0

    def denormalize_disp(self, disp_norm: torch.Tensor) -> torch.Tensor:
        if not self._has_disp_stats():
            return disp_norm
        scale = (self._disp_max - self._disp_min).clamp_min(1e-12)
        return (disp_norm + 1.0) * 0.5 * scale.to(disp_norm.device, disp_norm.dtype) + self._disp_min.to(disp_norm.device, disp_norm.dtype)

    @staticmethod
    def compute_contour_scale(polys: torch.Tensor) -> torch.Tensor:
        span_x = polys[..., 0].amax(dim=1) - polys[..., 0].amin(dim=1)
        span_y = polys[..., 1].amax(dim=1) - polys[..., 1].amin(dim=1)
        contour_scale = torch.maximum(span_x, span_y).clamp_min(1.0)
        return contour_scale.view(-1, 1, 1)

    @staticmethod
    def compute_centroid(poly: torch.Tensor) -> torch.Tensor:
        return poly.mean(dim=1, keepdim=True)

    def normalize_position(
        self,
        poly: torch.Tensor,
        centroid: torch.Tensor,
        contour_scale: torch.Tensor,
    ) -> torch.Tensor:
        return (poly - centroid.to(poly.device, poly.dtype)) / contour_scale.to(poly.device, poly.dtype)

    def denormalize_position(
        self,
        pos_centered: torch.Tensor,
        centroid: torch.Tensor,
        contour_scale: torch.Tensor,
    ) -> torch.Tensor:
        return (
            pos_centered * contour_scale.to(pos_centered.device, pos_centered.dtype)
            + centroid.to(pos_centered.device, pos_centered.dtype)
        )

    def _get_geom_t_init(self) -> float:
        return min(max(float(self._geom_t_init), 1e-4), 1.0 - 1e-4)

    def normalize_target_disp(self, disp_raw: torch.Tensor, contour_scale: torch.Tensor) -> torch.Tensor:
        if self._use_contour_norm:
            return disp_raw / contour_scale.to(disp_raw.device, disp_raw.dtype)
        return self.normalize_disp(disp_raw)

    def denormalize_pred_disp(self, disp_pred: torch.Tensor, contour_scale: torch.Tensor) -> torch.Tensor:
        if self._use_contour_norm:
            return disp_pred * contour_scale.to(disp_pred.device, disp_pred.dtype)
        return self.denormalize_disp(disp_pred)

    def clamp_pred_disp(self, disp: torch.Tensor, init_poly: torch.Tensor) -> torch.Tensor:
        if self._max_disp_frac <= 0 or disp.numel() == 0:
            return disp
        limit = self.compute_contour_scale(init_poly).to(disp.device, disp.dtype) * self._max_disp_frac
        norm = disp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return disp * torch.clamp(limit / norm, max=1.0)

    def _disp_gate_features(
        self,
        sampled_feat: torch.Tensor,
        disp: torch.Tensor,
        contour_scale: torch.Tensor,
    ) -> torch.Tensor:
        pooled = sampled_feat.mean(dim=-1)
        scale = contour_scale.to(device=disp.device, dtype=disp.dtype).clamp_min(1.0)
        disp_norm = disp / scale
        mag = disp_norm.norm(dim=-1)
        mean_mag = mag.mean(dim=1, keepdim=True)
        max_mag = mag.amax(dim=1, keepdim=True)
        rms_mag = mag.pow(2).mean(dim=1, keepdim=True).sqrt()
        log_scale = torch.log(scale.view(-1, 1))
        gate_in = torch.cat([pooled, mean_mag, max_mag, rms_mag, log_scale], dim=1)
        if self._disp_gate_detach_input:
            gate_in = gate_in.detach()
        return gate_in

    def predict_disp_gate(
        self,
        sampled_feat: torch.Tensor,
        disp: torch.Tensor,
        contour_scale: torch.Tensor,
    ) -> torch.Tensor:
        if (not self._use_disp_gate) or self.disp_gate_head is None or disp.numel() == 0:
            return disp.new_ones((disp.size(0), 1, 1))
        gate_in = self._disp_gate_features(sampled_feat, disp, contour_scale)
        gate = torch.sigmoid(self.disp_gate_head(gate_in)).view(-1, 1, 1)
        return gate.to(device=disp.device, dtype=disp.dtype)

    @staticmethod
    def compute_disp_gate_target(pred_disp: torch.Tensor, target_disp: torch.Tensor) -> torch.Tensor:
        pred = pred_disp.detach()
        target = target_disp.detach().to(device=pred.device, dtype=pred.dtype)
        denom = pred.pow(2).sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        numer = (pred * target).sum(dim=(1, 2), keepdim=True)
        return (numer / denom).clamp(0.0, 1.0)

    def sample_train_t(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._flow_fix_t0:
            return torch.zeros(n, 1, 1, device=device, dtype=dtype)

        # V6p: logit-normal sampling (SD3 style) — concentrates t near 0.5 where flow is most nonlinear
        if self._t_sample_mode == 'logit_normal':
            eps = torch.randn(n, device=device, dtype=dtype)
            t = torch.sigmoid(self._logit_normal_mean + self._logit_normal_std * eps)
            return t.view(n, 1, 1)

        alpha = max(self._flow_t_beta_alpha, 1e-6)
        beta = max(self._flow_t_beta_beta, 1e-6)
        if abs(alpha - 1.0) < 1e-6 and abs(beta - 1.0) < 1e-6:
            return torch.rand(n, device=device, dtype=dtype).view(n, 1, 1)

        dist = torch.distributions.Beta(alpha, beta)
        return dist.sample((n,)).to(device=device, dtype=dtype).view(n, 1, 1)

    def sample_train_x0(self, x1: torch.Tensor) -> torch.Tensor:
        x0 = torch.randn_like(x1) * self._flow_train_noise_scale
        if self._flow_zero_x0_prob > 0:
            zero_mask = torch.rand(
                x1.size(0), 1, 1, device=x1.device, dtype=x1.dtype
            ) < self._flow_zero_x0_prob
            x0 = torch.where(zero_mask, torch.zeros_like(x0), x0)
        return x0

    def _compute_soft_chamfer(
        self, pred_pts: torch.Tensor, gt_pts: torch.Tensor, tau: float = 0.05
    ) -> torch.Tensor:
        """Differentiable bidirectional soft Chamfer distance.

        Uses weighted softmin (d weighted by softmax(-d/tau)).  Always ≥ 0.
        Inputs should be in a normalised coordinate system (~unit scale) so
        that tau is meaningful; call-site divides by contour_scale.
        """
        d = torch.cdist(pred_pts.float(), gt_pts.float())   # (N, P, P)
        w_fwd = torch.softmax(-d / tau, dim=2)              # each pred → soft-nearest GT
        w_bwd = torch.softmax(-d / tau, dim=1)              # each GT  → soft-nearest pred
        soft_fwd = (d * w_fwd).sum(dim=2)                   # (N, P)
        soft_bwd = (d * w_bwd).sum(dim=1)                   # (N, P)
        return (soft_fwd.mean() + soft_bwd.mean()) * 0.5

    def compute_curvature_weights(self, polys: torch.Tensor) -> torch.Tensor:
        prev_pt = torch.roll(polys, 1, dims=1)
        next_pt = torch.roll(polys, -1, dims=1)
        curvature = (prev_pt - 2.0 * polys + next_pt).norm(dim=-1)
        curvature_norm = curvature / (curvature.amax(dim=1, keepdim=True) + 1e-6)
        if self._curvature_reweight_power != 1.0:
            curvature_norm = curvature_norm.pow(self._curvature_reweight_power)
        return 1.0 + self._curvature_loss_weight * curvature_norm

    def sample_detail_features(
        self,
        cnn_feature: torch.Tensor,
        img_poly: torch.Tensor,
        py_ind: torch.Tensor,
        h: int,
        w: int,
        sampled_feat: Optional[torch.Tensor] = None,
        contour_scale: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self._use_detail_context:
            return None
        if img_poly.numel() == 0:
            channels = cnn_feature.size(1)
            return cnn_feature.new_zeros((0, channels * 3, 0))

        if sampled_feat is None:
            sampled_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, img_poly, py_ind, h, w)
        if contour_scale is None:
            contour_scale = self.compute_contour_scale(img_poly)

        prev_pt = torch.roll(img_poly, 1, dims=1)
        next_pt = torch.roll(img_poly, -1, dims=1)
        tangent = next_pt - prev_pt
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)

        contour_scale = contour_scale.to(img_poly.device, img_poly.dtype)
        radius_1 = torch.clamp(contour_scale / 64.0, min=0.75, max=2.0)
        radius_2 = torch.clamp(contour_scale / 32.0, min=1.5, max=4.0)

        plus_1 = snake_gcn_utils.get_gcn_feature(cnn_feature, img_poly + normal * radius_1, py_ind, h, w)
        minus_1 = snake_gcn_utils.get_gcn_feature(cnn_feature, img_poly - normal * radius_1, py_ind, h, w)
        plus_2 = snake_gcn_utils.get_gcn_feature(cnn_feature, img_poly + normal * radius_2, py_ind, h, w)
        minus_2 = snake_gcn_utils.get_gcn_feature(cnn_feature, img_poly - normal * radius_2, py_ind, h, w)

        detail_terms = [
            plus_1 - minus_1,
            plus_2 - minus_2,
            0.5 * (plus_1 + minus_1) - sampled_feat,
        ]

        if self._detail_context_mode == 'normal_band':
            radius_3 = torch.clamp(contour_scale / 16.0, min=2.5, max=6.0)
            plus_3 = snake_gcn_utils.get_gcn_feature(cnn_feature, img_poly + normal * radius_3, py_ind, h, w)
            minus_3 = snake_gcn_utils.get_gcn_feature(cnn_feature, img_poly - normal * radius_3, py_ind, h, w)
            detail_terms.extend([
                plus_3 - minus_3,
                0.5 * (plus_2 + minus_2) - sampled_feat,
            ])
        elif self._detail_context_mode == 'normal_tangent':
            tangent_plus_1 = snake_gcn_utils.get_gcn_feature(
                cnn_feature, img_poly + tangent * radius_1, py_ind, h, w
            )
            tangent_minus_1 = snake_gcn_utils.get_gcn_feature(
                cnn_feature, img_poly - tangent * radius_1, py_ind, h, w
            )
            tangent_plus_2 = snake_gcn_utils.get_gcn_feature(
                cnn_feature, img_poly + tangent * radius_2, py_ind, h, w
            )
            tangent_minus_2 = snake_gcn_utils.get_gcn_feature(
                cnn_feature, img_poly - tangent * radius_2, py_ind, h, w
            )
            detail_terms.extend([
                tangent_plus_1 - tangent_minus_1,
                tangent_plus_2 - tangent_minus_2,
                0.5 * (tangent_plus_1 + tangent_minus_1) - sampled_feat,
            ])
        elif self._detail_context_mode != 'normal':
            raise ValueError(f"Unsupported v3_7_detail_context_mode: {self._detail_context_mode}")

        return torch.cat(detail_terms, dim=1)

    @staticmethod
    def _batch_meta_tensor(batch: Dict[str, Any], key: str, device: torch.device, dtype: torch.dtype):
        if batch is None or 'meta' not in batch or key not in batch['meta']:
            return None
        value = batch['meta'][key]
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype, non_blocking=True)
        return torch.as_tensor(value, device=device, dtype=dtype)

    @staticmethod
    def _batch_tensor(batch: Dict[str, Any], key: str, device: torch.device, dtype: torch.dtype):
        if batch is None or key not in batch:
            return None
        value = batch[key]
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype, non_blocking=True)
        return torch.as_tensor(value, device=device, dtype=dtype)

    def _points_to_locate_grid(
        self,
        points: torch.Tensor,
        py_ind: torch.Tensor,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """Map output-coordinate contour points to Locate patch-grid coordinates."""
        device, dtype = points.device, points.dtype
        inv_trans = self._batch_meta_tensor(batch, 'inv_trans_input', device, dtype)
        orig_hw = self._batch_meta_tensor(batch, 'orig_hw', device, dtype)
        inp_out_hw = self._batch_meta_tensor(batch, 'inp_out_hw', device, dtype)
        flipped = self._batch_meta_tensor(batch, 'flipped', device, dtype)
        locate_scale = self._batch_tensor(batch, 'locate_feat_scale', device, dtype)
        grid_hw = self._batch_tensor(batch, 'locate_feat_grid_hw', device, dtype)
        patch_size = self._batch_tensor(batch, 'locate_feat_patch_size', device, dtype)
        locate_pad = self._batch_tensor(batch, 'locate_feat_pad', device, dtype)
        if inv_trans is None or orig_hw is None or locate_scale is None or grid_hw is None:
            raise KeyError(
                "V11 LocateToken-DiT requires meta.inv_trans_input/meta.orig_hw and "
                "locate_feat_scale/locate_feat_grid_hw in the batch."
            )

        py_ind = py_ind.to(device=device, dtype=torch.long)
        n_contours = int(points.size(0))
        if n_contours == 0:
            return points.new_zeros((0, 0, 1, 2))

        if patch_size is None:
            patch_size = torch.full((inv_trans.size(0), 1), 14.0, device=device, dtype=dtype)
        if locate_pad is None:
            locate_pad = torch.zeros((inv_trans.size(0), 4), device=device, dtype=dtype)
        if flipped is None:
            flipped = torch.zeros((inv_trans.size(0), 1), device=device, dtype=dtype)
        if locate_scale.dim() == 1:
            locate_scale = locate_scale[:, None]
        if patch_size.dim() == 1:
            patch_size = patch_size[:, None]
        if flipped.dim() == 1:
            flipped = flipped[:, None]

        if inp_out_hw is not None and inp_out_hw.size(-1) >= 4:
            sx = (inp_out_hw[:, 1] / inp_out_hw[:, 3].clamp_min(1.0))[py_ind].view(n_contours, 1)
            sy = (inp_out_hw[:, 0] / inp_out_hw[:, 2].clamp_min(1.0))[py_ind].view(n_contours, 1)
        else:
            ratio = float(getattr(global_cfg, 'down_ratio', 4.0))
            sx = points.new_full((n_contours, 1), ratio)
            sy = points.new_full((n_contours, 1), ratio)

        input_x = points[..., 0] * sx
        input_y = points[..., 1] * sy
        ones = torch.ones_like(input_x)
        input_xy1 = torch.stack([input_x, input_y, ones], dim=1)

        inv = inv_trans[py_ind]
        src_xy = torch.bmm(inv, input_xy1).transpose(1, 2)
        src_x = src_xy[..., 0]
        src_y = src_xy[..., 1]
        orig_w = orig_hw[py_ind, 1].view(n_contours, 1)
        flip_mask = flipped[py_ind].view(n_contours, 1) > 0.5
        src_x = torch.where(flip_mask, orig_w - src_x - 1.0, src_x)

        scale = locate_scale[py_ind].view(n_contours, 1)
        patch = patch_size[py_ind].view(n_contours, 1).clamp_min(1.0)
        pad_left = locate_pad[py_ind, 0].view(n_contours, 1)
        pad_top = locate_pad[py_ind, 1].view(n_contours, 1)
        feat_x = (src_x * scale + pad_left) / patch - 0.5
        feat_y = (src_y * scale + pad_top) / patch - 0.5

        gh = grid_hw[py_ind, 0].view(n_contours, 1).clamp_min(1.0)
        gw = grid_hw[py_ind, 1].view(n_contours, 1).clamp_min(1.0)
        norm_x = torch.where(gw > 1.0, (feat_x / (gw - 1.0)) * 2.0 - 1.0, torch.zeros_like(feat_x))
        norm_y = torch.where(gh > 1.0, (feat_y / (gh - 1.0)) * 2.0 - 1.0, torch.zeros_like(feat_y))
        return torch.stack([norm_x, norm_y], dim=-1).unsqueeze(2)

    def _sample_locate_points(
        self,
        locate_map: torch.Tensor,
        points: torch.Tensor,
        py_ind: torch.Tensor,
        batch: Dict[str, Any],
        contour_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self._locate_use_normal_tangent:
            grid = self._points_to_locate_grid(points, py_ind, batch)
            sampled = F.grid_sample(
                locate_map[py_ind.to(device=locate_map.device, dtype=torch.long)],
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True,
            ).squeeze(-1).transpose(1, 2)
            return self.locate_point_mixer(sampled)

        prev_pt = torch.roll(points, 1, dims=1)
        next_pt = torch.roll(points, -1, dims=1)
        tangent = next_pt - prev_pt
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
        if contour_scale is None:
            contour_scale = self.compute_contour_scale(points)
        radius = torch.clamp(contour_scale.to(points.device, points.dtype) / 64.0, min=0.75, max=3.0)
        sample_points = [
            points,
            points + normal * radius,
            points - normal * radius,
            points + tangent * radius,
            points - tangent * radius,
        ]
        feat_n = locate_map[py_ind.to(device=locate_map.device, dtype=torch.long)]
        sampled_terms = []
        for pts in sample_points:
            grid = self._points_to_locate_grid(pts, py_ind, batch)
            sampled = F.grid_sample(
                feat_n,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True,
            ).squeeze(-1).transpose(1, 2)
            sampled_terms.append(sampled)
        return self.locate_point_mixer(torch.cat(sampled_terms, dim=-1))

    def _sample_locate_roi_tokens(
        self,
        locate_map: torch.Tensor,
        points: torch.Tensor,
        py_ind: torch.Tensor,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        grid_size = max(int(self._locate_roi_grid), 1)
        n_contours = int(points.size(0))
        if n_contours == 0 or grid_size <= 0:
            return locate_map.new_zeros((n_contours, 0, self._locate_token_dim))
        x_min = points[..., 0].amin(dim=1)
        x_max = points[..., 0].amax(dim=1)
        y_min = points[..., 1].amin(dim=1)
        y_max = points[..., 1].amax(dim=1)
        xs_base = torch.linspace(0.0, 1.0, grid_size, device=points.device, dtype=points.dtype)
        ys_base = torch.linspace(0.0, 1.0, grid_size, device=points.device, dtype=points.dtype)
        yy, xx = torch.meshgrid(ys_base, xs_base, indexing='ij')
        roi_x = x_min.view(n_contours, 1, 1) + xx.view(1, grid_size, grid_size) * (
            x_max - x_min
        ).clamp_min(1.0).view(n_contours, 1, 1)
        roi_y = y_min.view(n_contours, 1, 1) + yy.view(1, grid_size, grid_size) * (
            y_max - y_min
        ).clamp_min(1.0).view(n_contours, 1, 1)
        roi_points = torch.stack([roi_x, roi_y], dim=-1).view(n_contours, grid_size * grid_size, 2)
        grid = self._points_to_locate_grid(roi_points, py_ind, batch).view(n_contours, grid_size, grid_size, 2)
        sampled = F.grid_sample(
            locate_map[py_ind.to(device=locate_map.device, dtype=torch.long)],
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        return sampled.flatten(2).transpose(1, 2)

    def build_locate_token_context(
        self,
        batch: Dict[str, Any],
        points: torch.Tensor,
        py_ind: torch.Tensor,
        contour_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if not self._locate_token_enabled:
            return {}
        if batch is None or 'locate_feat' not in batch:
            raise KeyError(
                "cfg.use_locate_token_dit=True but batch has no locate_feat. "
                "Run Locate feature extraction or check locate_feat_cache_root."
            )
        raw = batch['locate_feat']
        if not torch.is_tensor(raw):
            raw = torch.as_tensor(raw)
        param = next(self.locate_feat_proj.parameters())
        raw = raw.to(device=points.device, dtype=param.dtype, non_blocking=True)
        locate_map = self.locate_feat_proj(raw).to(dtype=points.dtype)

        point_ctx = self._sample_locate_points(
            locate_map,
            points,
            py_ind,
            batch,
            contour_scale=contour_scale,
        )

        bsz, channels, h, w = locate_map.shape
        tokens = locate_map.flatten(2).transpose(1, 2)
        queries = self.locate_global_queries.weight.to(device=points.device, dtype=points.dtype)
        queries = queries.unsqueeze(0).expand(bsz, -1, -1)
        global_ctx, _ = self.locate_global_attn(queries, tokens, tokens, need_weights=False)
        global_ctx = self.locate_global_norm(global_ctx + queries)
        global_ctx = global_ctx[py_ind.to(device=points.device, dtype=torch.long)]
        if self._locate_use_roi_tokens:
            roi_ctx = self._sample_locate_roi_tokens(locate_map, points, py_ind, batch)
            global_ctx = torch.cat([global_ctx, roi_ctx], dim=1)
        return {
            'locate_point_ctx': point_ctx,
            'locate_global_ctx': global_ctx,
            'locate_only': self._locate_token_only,
            'locate_map_absmax': locate_map.detach().abs().max(),
            'locate_point_ctx_absmax': point_ctx.detach().abs().max(),
        }

    @staticmethod
    def fourier_smooth(disp: torch.Tensor, k: int) -> torch.Tensor:
        """Apply a Fourier low-pass filter to contour displacement."""
        if k <= 0:
            return disp
        _, n, _ = disp.shape
        freq = torch.fft.rfft(disp, dim=1)
        mask = torch.zeros(freq.shape[1], device=disp.device, dtype=torch.bool)
        mask[:k + 1] = True
        freq[:, ~mask, :] = 0
        return torch.fft.irfft(freq, n=n, dim=1)

    # ------------------------------------------------------------------
    # Internal KV-cache helpers — store cross-attn K/V inside each
    # DiTBlockV3 so the normal denoiser.forward() path reuses them.
    # ------------------------------------------------------------------

    def _set_denoiser_kv_cache(
        self,
        cnn_feature: torch.Tensor,
        sampled_feat: torch.Tensor,
        py_ind=None,
        detail_feat=None,
    ) -> bool:
        """Pre-compute and store K/V in all DiTBlockV3 layers.

        Returns True if cache was successfully set, False otherwise.
        """
        from .dit_blocks_v3 import DiTBlockV3

        if not hasattr(self.denoiser, 'dit_layers'):
            return False
        if not any(isinstance(l, DiTBlockV3) for l in self.denoiser.dit_layers):
            return False

        param_dtype = next(self.denoiser.parameters()).dtype
        cnn_feature = cnn_feature.to(param_dtype)
        sampled_feat = sampled_feat.to(param_dtype)

        if cnn_feature.dim() == 3:
            cnn_feature = cnn_feature.unsqueeze(0)

        n_contours = sampled_feat.shape[0]
        global_ctx = self.denoiser.global_compressor(cnn_feature)
        if py_ind is not None:
            global_ctx = global_ctx[py_ind]
        elif global_ctx.shape[0] != n_contours:
            if global_ctx.shape[0] == 1:
                global_ctx = global_ctx.expand(n_contours, -1, -1)

        local_ctx = self.denoiser.local_proj(sampled_feat.transpose(1, 2))

        if (
            getattr(self.denoiser, 'use_detail_context', False)
            and detail_feat is not None
        ):
            detail_feat = detail_feat.to(param_dtype)
            detail_ctx = detail_feat.transpose(1, 2)
            local_ctx = local_ctx + self.denoiser.detail_local_proj(detail_ctx)

        for i, layer in enumerate(self.denoiser.dit_layers):
            if isinstance(layer, DiTBlockV3):
                context = global_ctx if (i % 2 == 0) else local_ctx
                layer.set_kv_cache(context)
        return True

    def _clear_denoiser_kv_cache(self) -> None:
        """Release KV cache from all DiTBlockV3 layers."""
        from .dit_blocks_v3 import DiTBlockV3

        if hasattr(self.denoiser, 'dit_layers'):
            for layer in self.denoiser.dit_layers:
                if isinstance(layer, DiTBlockV3):
                    layer.clear_kv_cache()

    def predict_velocity(
        self,
        cnn_feature,
        i_it_py,
        c_it_py,
        sampled_feat,
        detail_feat,
        py_ind,
        x_t,
        t_continuous,
        contour_scale: Optional[torch.Tensor] = None,
        x_self_cond: Optional[torch.Tensor] = None,
        locate_context: Optional[Dict[str, torch.Tensor]] = None,
        s: Optional[torch.Tensor] = None,
    ):
        """预测速度场 V_t。

        KV cache is now managed internally by DiTBlockV3 (set_kv_cache /
        clear_kv_cache).  The normal denoiser.forward() path is always taken;
        cached K/V tensors in each block are reused automatically.
        """
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, i_it_py.size(1), i_it_py.device)

        # t ∈ [0,1] → [0,1000] for the time embedder
        t_scaled = t_continuous * 1000.0

        denoiser_kwargs = {}
        if locate_context:
            denoiser_kwargs.update({
                'locate_point_ctx': locate_context.get('locate_point_ctx', None),
                'locate_global_ctx': locate_context.get('locate_global_ctx', None),
                'locate_only': bool(locate_context.get('locate_only', False)),
            })
        if self._use_s_cond and s is not None:
            denoiser_kwargs['s'] = s

        v_pred, L = self.denoiser(
            cnn_feature,
            sampled_feat,
            x_t,
            t_scaled,
            adj,
            polys=i_it_py,
            py_ind=py_ind,
            contour_scale=contour_scale,
            detail_feat=detail_feat,
            x_self_cond=x_self_cond,
            **denoiser_kwargs,
        )
        return v_pred, L

    def prepare_sampling_context(
        self,
        cnn_feature: torch.Tensor,
        i_it_py: torch.Tensor,
        py_ind: torch.Tensor,
        batch: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        sampled_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_py, py_ind, h, w)
        contour_scale = self.compute_contour_scale(i_it_py)
        detail_feat = self.sample_detail_features(
            cnn_feature,
            i_it_py,
            py_ind,
            h,
            w,
            sampled_feat=sampled_feat,
            contour_scale=contour_scale,
        )
        ctx = {
            'sampled_feat': sampled_feat,
            'detail_feat': detail_feat,
            'contour_scale': contour_scale,
            'h': h,
            'w': w,
        }
        if self._locate_token_enabled and batch is not None:
            ctx['locate_context'] = self.build_locate_token_context(
                batch,
                i_it_py,
                py_ind,
                contour_scale=contour_scale,
            )
        return ctx

    @staticmethod
    def _gaussian_sample_logprob(sample: torch.Tensor, mean: torch.Tensor, std) -> torch.Tensor:
        if torch.is_tensor(std):
            std_t = std.to(device=mean.device, dtype=mean.dtype)
        else:
            std_t = mean.new_full((mean.size(0), 1, 1), max(float(std), 1e-6))
        std_t = std_t.clamp_min(1e-6)
        var = std_t.pow(2)
        log_prob = -((sample.detach() - mean) ** 2) / (2.0 * var)
        log_prob = log_prob - torch.log(std_t) - 0.5 * math.log(2.0 * math.pi)
        return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    def latent_policy_mean(self, sampled_feat: torch.Tensor) -> torch.Tensor:
        if (not self._use_latent_policy) or self.latent_policy is None:
            return sampled_feat.new_zeros((sampled_feat.size(0), sampled_feat.size(2), 2))
        mean = self.latent_policy(sampled_feat).transpose(1, 2).contiguous()
        return mean * self._latent_policy_scale

    def initial_latent_logprob(
        self,
        sampled_feat: torch.Tensor,
        latent_x0: torch.Tensor,
        noise_scale: float,
    ) -> torch.Tensor:
        mean = self.latent_policy_mean(sampled_feat).to(device=latent_x0.device, dtype=latent_x0.dtype)
        return self._gaussian_sample_logprob(latent_x0, mean, float(noise_scale)) * self._latent_logprob_scale

    def sample_initial_latent_with_logprob(
        self,
        sampled_feat: torch.Tensor,
        like: torch.Tensor,
        noise_scale: float,
        generator: Optional[torch.Generator] = None,
    ):
        mean = self.latent_policy_mean(sampled_feat).to(device=like.device, dtype=like.dtype)
        std = max(float(noise_scale), 1e-6)
        noise = torch.randn(like.shape, device=like.device, dtype=like.dtype, generator=generator)
        latent_x0 = mean + std * noise
        log_prob = self._gaussian_sample_logprob(latent_x0, mean, std) * self._latent_logprob_scale
        std_t = mean.new_full((mean.size(0), 1, 1), std)
        return latent_x0, log_prob, mean, std_t

    def latent_ranker_score(self, sampled_feat: torch.Tensor, latent_x0: torch.Tensor) -> torch.Tensor:
        if (not self._use_latent_ranker) or self.latent_ranker is None or self.latent_ranker_head is None:
            return sampled_feat.new_zeros((sampled_feat.size(0),))
        x_feat = latent_x0.transpose(1, 2).to(device=sampled_feat.device, dtype=sampled_feat.dtype)
        rank_in = torch.cat([sampled_feat, x_feat], dim=1)
        pooled = self.latent_ranker(rank_in).mean(dim=-1)
        return self.latent_ranker_head(pooled).squeeze(-1)

    def sample_ranked_initial_latent(
        self,
        sampled_feat: torch.Tensor,
        like: torch.Tensor,
        noise_scale: float,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        k = max(int(self._latent_selector_k if k is None else k), 1)
        if k == 1 or (not self._use_latent_ranker):
            x0, _, _, _ = self.sample_initial_latent_with_logprob(sampled_feat, like, noise_scale)
            return x0

        candidates = []
        scores = []
        for _ in range(k):
            x0, _, _, _ = self.sample_initial_latent_with_logprob(sampled_feat, like, noise_scale)
            candidates.append(x0)
            scores.append(self.latent_ranker_score(sampled_feat, x0))
        x_stack = torch.stack(candidates, dim=0)  # (K, B, P, 2)
        score_stack = torch.stack(scores, dim=0)  # (K, B)
        best_idx = torch.argmax(score_stack, dim=0)
        gather_idx = best_idx.view(1, -1, 1, 1).expand(1, -1, x_stack.size(2), x_stack.size(3))
        return x_stack.gather(0, gather_idx).squeeze(0)

    @staticmethod
    def _gaussian_step_with_logprob(
        step_mean: torch.Tensor,
        action_std: float,
        dt: float,
        prev_sample: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        if action_std <= 0:
            if prev_sample is None:
                prev_sample = step_mean
            log_prob = step_mean.new_zeros(step_mean.size(0))
            std = step_mean.new_zeros(step_mean.size(0), 1, 1)
            return prev_sample, log_prob, step_mean, std

        std_scalar = max(float(action_std) * math.sqrt(max(abs(float(dt)), 1e-12)), 1e-6)
        std = step_mean.new_full((step_mean.size(0), 1, 1), std_scalar)
        if prev_sample is None:
            noise = torch.randn(
                step_mean.shape,
                device=step_mean.device,
                dtype=step_mean.dtype,
                generator=generator,
            )
            prev_sample = step_mean + std * noise
        else:
            prev_sample = prev_sample.to(device=step_mean.device, dtype=step_mean.dtype)

        var = std.pow(2).clamp_min(1e-12)
        log_prob = -((prev_sample.detach() - step_mean) ** 2) / (2.0 * var)
        log_prob = log_prob - torch.log(std.clamp_min(1e-12)) - 0.5 * math.log(2.0 * math.pi)
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, log_prob, step_mean, std

    @staticmethod
    def _flow_grpo_step_with_logprob(
        sample: torch.Tensor,
        model_output: torch.Tensor,
        t_value,
        total_steps: int,
        noise_level: float,
        sde_type: str = 'sde',
        prev_sample: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        total_steps = max(int(total_steps), 1)
        dt = 1.0 / float(total_steps)
        t_float = float(t_value.item()) if torch.is_tensor(t_value) else float(t_value)
        transition_mean = sample + model_output * dt
        zero_std = sample.new_zeros(sample.size(0), 1, 1)

        if noise_level <= 0:
            if prev_sample is None:
                prev_sample = transition_mean
            else:
                prev_sample = prev_sample.to(device=sample.device, dtype=sample.dtype)
            return prev_sample, sample.new_zeros(sample.size(0)), transition_mean, zero_std

        sigma = sample.new_full((sample.size(0), 1, 1), max(1.0 - t_float, 0.0))
        sigma_next = sample.new_full(
            (sample.size(0), 1, 1),
            max(1.0 - min(t_float + dt, 1.0), 0.0),
        )
        dt_sigma = sigma_next - sigma
        neg_dt_sigma = (-dt_sigma).clamp_min(1e-12)
        sde_type = str(sde_type).strip().lower()

        if sde_type == 'sde':
            safe_sigma = sigma.clamp_min(1e-6)
            sigma_ref = torch.where(
                sigma >= (1.0 - 1e-6),
                sigma_next.clamp(max=1.0 - 1e-6),
                sigma,
            ).clamp(min=1e-6, max=1.0 - 1e-6)
            std_dev_t = torch.sqrt(safe_sigma / (1.0 - sigma_ref).clamp_min(1e-6)) * float(noise_level)
            transition_mean = (
                sample * (1.0 + std_dev_t.pow(2) / (2.0 * safe_sigma) * dt_sigma)
                + model_output * (1.0 + std_dev_t.pow(2) * (1.0 - sigma) / (2.0 * safe_sigma)) * dt_sigma
            )
            transition_std = std_dev_t * torch.sqrt(neg_dt_sigma)
            if prev_sample is None:
                noise = torch.randn(
                    transition_mean.shape,
                    device=transition_mean.device,
                    dtype=transition_mean.dtype,
                    generator=generator,
                )
                prev_sample = transition_mean + transition_std * noise
            else:
                prev_sample = prev_sample.to(device=transition_mean.device, dtype=transition_mean.dtype)

            var = transition_std.pow(2).clamp_min(1e-12)
            log_prob = -((prev_sample.detach() - transition_mean) ** 2) / (2.0 * var)
            log_prob = log_prob - torch.log(transition_std.clamp_min(1e-12)) - 0.5 * math.log(2.0 * math.pi)
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            return prev_sample, log_prob, transition_mean, transition_std

        if sde_type == 'cps':
            transition_std = sigma_next.clamp_min(0.0) * math.sin(float(noise_level) * math.pi / 2.0)
            pred_original_sample = sample - sigma * model_output
            noise_estimate = sample + model_output * (1.0 - sigma)
            transition_mean = pred_original_sample * (1.0 - sigma_next) + noise_estimate * torch.sqrt(
                (sigma_next.pow(2) - transition_std.pow(2)).clamp_min(0.0)
            )
            if prev_sample is None:
                noise = torch.randn(
                    transition_mean.shape,
                    device=transition_mean.device,
                    dtype=transition_mean.dtype,
                    generator=generator,
                )
                prev_sample = transition_mean + transition_std * noise
            else:
                prev_sample = prev_sample.to(device=transition_mean.device, dtype=transition_mean.dtype)
            log_prob = -((prev_sample.detach() - transition_mean) ** 2)
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            return prev_sample, log_prob, transition_mean, transition_std

        raise ValueError(f'Unsupported flow-grpo sde_type: {sde_type}')

    def step_with_logprob(
        self,
        cnn_feature: torch.Tensor,
        i_it_py: torch.Tensor,
        c_it_py: torch.Tensor,
        py_ind: torch.Tensor,
        x_t: torch.Tensor,
        t_value,
        step_index: int,
        total_steps: int,
        action_std: float,
        prev_sample: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        sampled_feat: Optional[torch.Tensor] = None,
        detail_feat: Optional[torch.Tensor] = None,
        contour_scale: Optional[torch.Tensor] = None,
        x_self_cond: Optional[torch.Tensor] = None,
        locate_context: Optional[Dict[str, torch.Tensor]] = None,
        step_mode: Optional[str] = None,
        noise_level: Optional[float] = None,
        sde_type: Optional[str] = None,
        s: Optional[torch.Tensor] = None,
    ):
        if sampled_feat is None or contour_scale is None or (self._use_detail_context and detail_feat is None):
            ctx = self.prepare_sampling_context(cnn_feature, i_it_py, py_ind)
            sampled_feat = ctx['sampled_feat']
            detail_feat = ctx['detail_feat']
            contour_scale = ctx['contour_scale']

        total_steps = max(int(total_steps), 1)
        dt = 1.0 / float(total_steps)
        t_float = float(t_value.item()) if torch.is_tensor(t_value) else float(t_value)
        t_tensor = torch.full((x_t.size(0),), t_float, device=x_t.device, dtype=x_t.dtype)
        contour_scale_flat = contour_scale.view(-1).to(device=x_t.device, dtype=x_t.dtype)

        v_pred, _ = self.predict_velocity(
            cnn_feature,
            i_it_py,
            c_it_py,
            sampled_feat,
            detail_feat,
            py_ind,
            x_t,
            t_tensor,
            contour_scale=contour_scale_flat,
            x_self_cond=x_self_cond,
            locate_context=locate_context,
            s=s,
        )

        next_self_cond = None
        if self._use_self_conditioning:
            next_self_cond = (x_t + (1.0 - t_float) * v_pred).detach()

        use_heun = (self._ode_solver == 'heun')
        if use_heun and step_index < total_steps - 1:
            x_pred = x_t + v_pred * dt
            t_next = torch.full((x_t.size(0),), t_float + dt, device=x_t.device, dtype=x_t.dtype)
            v_pred2, _ = self.predict_velocity(
                cnn_feature,
                i_it_py,
                c_it_py,
                sampled_feat,
                detail_feat,
                py_ind,
                x_pred,
                t_next,
                contour_scale=contour_scale_flat,
                x_self_cond=next_self_cond,
                locate_context=locate_context,
                s=s,
            )
            step_velocity = (v_pred + v_pred2) * 0.5
            step_mean = x_t + step_velocity * dt
        else:
            step_velocity = v_pred
            step_mean = x_t + step_velocity * dt

        if self._ode_smooth_k > 0:
            step_mean = self.fourier_smooth(step_mean, self._ode_smooth_k)

        step_mode = (
            self._step_logprob_mode
            if step_mode is None
            else str(step_mode).strip().lower()
        )
        if step_mode == 'gaussian':
            prev_sample, log_prob, step_mean, std = self._gaussian_step_with_logprob(
                step_mean,
                action_std=action_std,
                dt=dt,
                prev_sample=prev_sample,
                generator=generator,
            )
        elif step_mode in ('flow_grpo', 'flowgrpo'):
            # flow_grpo noise is controlled by noise_level (not action_std).
            # Always call _flow_grpo_step_with_logprob so log_prob actually
            # depends on v_theta and KL can be non-zero.
            prev_sample, log_prob, step_mean, std = self._flow_grpo_step_with_logprob(
                sample=x_t,
                model_output=step_velocity,
                t_value=t_float,
                total_steps=total_steps,
                noise_level=self._step_noise_level if noise_level is None else float(noise_level),
                sde_type=self._step_sde_type if sde_type is None else str(sde_type),
                prev_sample=prev_sample,
                generator=generator,
            )
        else:
            raise ValueError(f'Unsupported step_with_logprob mode: {step_mode}')
        return prev_sample, log_prob, step_mean, std, next_self_cond

    @torch.no_grad()
    def sample_with_logprob(
        self,
        cnn_feature: torch.Tensor,
        i_it_py: torch.Tensor,
        c_it_py: torch.Tensor,
        py_ind: torch.Tensor,
        steps: Optional[int] = None,
        window_size: int = 0,
        window_range: Tuple[int, int] = (0, 0),
        noise_scale: Optional[float] = None,
        action_std: float = 0.0,
        generator: Optional[torch.Generator] = None,
        step_mode: Optional[str] = None,
        noise_level: Optional[float] = None,
        sde_type: Optional[str] = None,
        s: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if steps is None:
            steps = self.ode_steps
        steps = max(int(steps), 1)
        if noise_scale is None:
            noise_scale = self._flow_train_noise_scale

        device = i_it_py.device
        ctx = self.prepare_sampling_context(cnn_feature, i_it_py, py_ind)
        latent_log_prob = None
        latent_mean = None
        latent_std = None
        if self._use_latent_policy:
            x, latent_log_prob, latent_mean, latent_std = self.sample_initial_latent_with_logprob(
                ctx['sampled_feat'],
                i_it_py,
                float(noise_scale),
                generator=generator,
            )
        else:
            x = torch.randn_like(i_it_py) * float(noise_scale)
        latent_x0 = x.detach()

        if window_size and window_size > 0:
            start_min = int(window_range[0]) if isinstance(window_range, (tuple, list)) and len(window_range) > 0 else 0
            end_max = int(window_range[1]) if isinstance(window_range, (tuple, list)) and len(window_range) > 1 else steps
            end_max = max(end_max, window_size)
            if end_max <= start_min + window_size:
                window_start = max(0, min(start_min, max(steps - window_size, 0)))
            else:
                window_start = np.random.randint(start_min, end_max - window_size + 1)
            window_end = min(window_start + window_size, steps)
        else:
            window_start = 0
            window_end = steps

        latents_seq = []
        log_probs = []
        t_seq = []
        step_indices = []
        x_ts = []
        x_prevs = []
        x_self_conds = []

        x_self_cond = torch.zeros_like(x) if self._use_self_conditioning else None
        dt = 1.0 / float(steps)

        for idx in range(steps):
            t_value = idx * dt
            if idx == window_start:
                latents_seq.append(x.detach())
            in_policy_window = idx >= window_start and idx < window_end

            x_prev, log_prob, _, _, next_self_cond = self.step_with_logprob(
                cnn_feature,
                i_it_py,
                c_it_py,
                py_ind,
                x_t=x,
                t_value=t_value,
                step_index=idx,
                total_steps=steps,
                action_std=action_std if in_policy_window else 0.0,
                prev_sample=None,
                generator=generator,
                sampled_feat=ctx['sampled_feat'],
                detail_feat=ctx['detail_feat'],
                contour_scale=ctx['contour_scale'],
                x_self_cond=x_self_cond,
                step_mode=step_mode,
                noise_level=noise_level,
                sde_type=sde_type,
                s=s,
            )

            if in_policy_window:
                latents_seq.append(x_prev.detach())
                log_probs.append(log_prob.detach())
                t_seq.append(torch.tensor(t_value, device=device, dtype=x.dtype))
                step_indices.append(torch.tensor(idx, device=device, dtype=torch.long))
                x_ts.append(x.detach())
                x_prevs.append(x_prev.detach())
                x_self_conds.append(None if x_self_cond is None else x_self_cond.detach())

            x = x_prev.detach()
            if self._use_self_conditioning:
                x_self_cond = next_self_cond

        disp = self.denormalize_pred_disp(x, ctx['contour_scale'])
        disp = self.clamp_pred_disp(disp, i_it_py)
        py = i_it_py + disp
        return {
            'latents': latents_seq,
            'log_probs': log_probs,
            'timesteps': t_seq,
            'step_indices': step_indices,
            'x_ts': x_ts,
            'x_prevs': x_prevs,
            'x_self_conds': x_self_conds,
            'sampled_feat': ctx['sampled_feat'],
            'detail_feat': ctx['detail_feat'],
            'contour_scale': ctx['contour_scale'],
            'disp': disp,
            'py': py,
            'latent_x0': latent_x0,
            'latent_log_prob': None if latent_log_prob is None else latent_log_prob.detach(),
            'latent_mean': None if latent_mean is None else latent_mean.detach(),
            'latent_std': None if latent_std is None else latent_std.detach(),
            'latent_noise_scale': float(noise_scale),
        }

    def _sample_disp_from_sampled_feat(
        self,
        cnn_feature,
        i_it_py,
        c_it_py,
        py_ind,
        sampled_feat,
        detail_feat,
        steps=None,
        noise_scale=None,
        locate_context: Optional[Dict[str, torch.Tensor]] = None,
        s: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if steps is None:
            steps = self.ode_steps
        if noise_scale is None:
            ns = self._infer_noise_scale
            noise_scale = self._flow_noise_scale if ns < 0 else ns

        device = i_it_py.device
        N = i_it_py.size(0)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        contour_scale = self.compute_contour_scale(i_it_py)
        if self._use_latent_policy:
            if self._latent_policy_eval_mode in ('ranker', 'selector', 'select'):
                x_t = self.sample_ranked_initial_latent(
                    sampled_feat,
                    i_it_py,
                    float(noise_scale),
                    k=self._latent_selector_k,
                )
            elif self._latent_policy_eval_mode in ('mean', 'det', 'deterministic'):
                x_t = self.latent_policy_mean(sampled_feat).to(device=i_it_py.device, dtype=i_it_py.dtype)
            else:
                x_t, _, _, _ = self.sample_initial_latent_with_logprob(
                    sampled_feat,
                    i_it_py,
                    float(noise_scale),
                )
        else:
            x_t = torch.randn_like(i_it_py) * noise_scale
        contour_scale_flat = contour_scale.view(-1)
        dt = 1.0 / steps

        use_heun = (self._ode_solver == 'heun')
        use_ab2  = (self._ode_solver == 'ab2')
        # V6r: self-conditioning state — starts at zero, updated each step
        x_self_cond = torch.zeros_like(x_t) if self._use_self_conditioning else None
        # Adams-Bashforth 2nd-order: store previous velocity for multi-step correction
        v_prev: Optional[torch.Tensor] = None

        # KV-cache: pre-compute cross-attention K/V once when features are fixed.
        # Stored inside each DiTBlockV3; the normal denoiser.forward() path
        # reuses them automatically.
        # Skipped when resample_feat_at_xt is on (features change per step).
        # Set env FLOW_DISABLE_KV_CACHE=1 to force-disable for ablation.
        _has_kv_cache = False
        _kv_disabled = os.environ.get('FLOW_DISABLE_KV_CACHE', '').strip() in ('1', 'true', 'yes')
        if (
            not _kv_disabled
            and not self._resample_feat_at_xt
            and locate_context is None
        ):
            with torch.no_grad():
                _has_kv_cache = self._set_denoiser_kv_cache(
                    cnn_feature, sampled_feat, py_ind=py_ind, detail_feat=detail_feat
                )

        for i in range(steps):
            t_val = i * dt
            t_tensor = torch.full((N,), t_val, device=device, dtype=torch.float32)
            if self._resample_feat_at_xt:
                with torch.no_grad():
                    xt_disp_raw = self.denormalize_pred_disp(x_t, contour_scale)
                    cur_contour = (i_it_py + xt_disp_raw).detach()
                    cur_sampled_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, cur_contour, py_ind, h, w)
                    cur_detail_feat = self.sample_detail_features(
                        cnn_feature,
                        cur_contour,
                        py_ind,
                        h,
                        w,
                        sampled_feat=cur_sampled_feat,
                        contour_scale=contour_scale,
                    )
            else:
                cur_sampled_feat = sampled_feat
                cur_detail_feat = detail_feat
            v_pred, _ = self.predict_velocity(
                cnn_feature,
                i_it_py,
                c_it_py,
                cur_sampled_feat,
                cur_detail_feat,
                py_ind,
                x_t,
                t_tensor,
                contour_scale=contour_scale_flat,
                x_self_cond=x_self_cond,
                locate_context=locate_context,
                s=s,
            )

            # Update self-conditioning with current x1 estimate
            if self._use_self_conditioning:
                x_self_cond = (x_t + (1.0 - t_val) * v_pred).detach()

            if use_heun and i < steps - 1:
                # Heun's method (2nd-order Runge-Kutta): predictor-corrector, 2x NFE
                x_pred = x_t + v_pred * dt
                t_next = torch.full((N,), t_val + dt, device=device, dtype=torch.float32)
                if self._resample_feat_at_xt:
                    with torch.no_grad():
                        xt_disp_pred = self.denormalize_pred_disp(x_pred, contour_scale)
                        pred_contour = (i_it_py + xt_disp_pred).detach()
                        pred_sampled_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, pred_contour, py_ind, h, w)
                        pred_detail_feat = self.sample_detail_features(
                            cnn_feature,
                            pred_contour,
                            py_ind,
                            h,
                            w,
                            sampled_feat=pred_sampled_feat,
                            contour_scale=contour_scale,
                        )
                else:
                    pred_sampled_feat = sampled_feat
                    pred_detail_feat = detail_feat
                v_pred2, _ = self.predict_velocity(
                    cnn_feature,
                    i_it_py,
                    c_it_py,
                    pred_sampled_feat,
                    pred_detail_feat,
                    py_ind,
                    x_pred,
                    t_next,
                    contour_scale=contour_scale_flat,
                    x_self_cond=x_self_cond,
                    locate_context=locate_context,
                    s=s,
                )
                x_t = x_t + (v_pred + v_pred2) * 0.5 * dt
            elif use_ab2 and v_prev is not None:
                # Adams-Bashforth 2nd-order: 2nd-order accuracy at 1x NFE per step.
                # x_{n+1} = x_n + dt * (3/2 * v_n - 1/2 * v_{n-1})
                x_t = x_t + dt * (1.5 * v_pred - 0.5 * v_prev)
            else:
                # Euler step (also used for the first AB2 step to bootstrap v_prev)
                x_t = x_t + v_pred * dt

            v_prev = v_pred

            # V3.7: per-step ODE Fourier smoothing
            if self._ode_smooth_k > 0:
                x_t = self.fourier_smooth(x_t, self._ode_smooth_k)

        # Release KV cache stored in DiTBlockV3 instances
        if _has_kv_cache:
            self._clear_denoiser_kv_cache()

        disp = self.denormalize_pred_disp(x_t, contour_scale)
        return self.clamp_pred_disp(disp, i_it_py)

    def _sample_disp_geom_bridge(
        self,
        cnn_feature,
        i_it_py,
        c_it_py,
        py_ind,
        steps=None,
        noise_scale=None,
        locate_context: Optional[Dict[str, torch.Tensor]] = None,
        s: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if steps is None:
            steps = self.ode_steps

        device = i_it_py.device
        N = i_it_py.size(0)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        contour_scale = self.compute_contour_scale(i_it_py)
        contour_scale_flat = contour_scale.view(-1)
        geom_position_flow = self._geom_position_flow
        geom_seg_flow = geom_position_flow and self._geom_seg_flow
        centroid = None
        if geom_seg_flow:
            centroid = self.compute_centroid(i_it_py)
            x_t = torch.randn_like(i_it_py) * self._geom_noise_scale
        elif geom_position_flow:
            centroid = self.compute_centroid(i_it_py)
            x_t = self.normalize_position(i_it_py, centroid, contour_scale)
        else:
            x_t = torch.zeros_like(i_it_py)
        if self._geom_x0_jitter > 0:
            x_t = x_t + torch.randn_like(i_it_py) * self._geom_x0_jitter

        if geom_seg_flow:
            steps = max(int(steps), 2)
            t_init = self._get_geom_t_init()
            front_steps = int(round(float(steps) * t_init))
            front_steps = min(max(front_steps, 1), steps - 1)
            back_steps = max(steps - front_steps, 1)
            segment_specs = (
                (front_steps, 1.0 / float(front_steps), True),
                (back_steps, 1.0 / float(back_steps), False),
            )
            for seg_steps, dt_seg, use_init_feat in segment_specs:
                for i in range(seg_steps):
                    t_val = i * dt_seg
                    if use_init_feat:
                        cur = i_it_py
                    else:
                        cur = self.denormalize_position(
                            x_t, centroid, contour_scale
                        )
                    sampled_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, cur, py_ind, h, w)
                    detail_feat = self.sample_detail_features(
                        cnn_feature,
                        cur,
                        py_ind,
                        h,
                        w,
                        sampled_feat=sampled_feat,
                        contour_scale=contour_scale,
                    )
                    t_tensor = torch.full((N,), t_val, device=device, dtype=torch.float32)
                    # Segmented position flow uses per-segment time embeddings: each bridge is a local [0, 1] ODE.
                    v_pred, _ = self.predict_velocity(
                        cnn_feature,
                        i_it_py,
                        c_it_py,
                        sampled_feat,
                        detail_feat,
                        py_ind,
                        x_t,
                        t_tensor,
                        contour_scale=contour_scale_flat,
                        x_self_cond=None,
                        locate_context=locate_context,
                        s=s,
                    )
                    x_t = x_t + v_pred * dt_seg

            final_poly = self.denormalize_position(x_t, centroid, contour_scale)
            disp = final_poly - i_it_py
            return self.clamp_pred_disp(disp, i_it_py)

        dt = 1.0 / steps

        for i in range(steps):
            t_val = i * dt
            if geom_position_flow:
                cur = self.denormalize_position(x_t, centroid, contour_scale)
            else:
                cur = i_it_py + self.denormalize_pred_disp(x_t, contour_scale)
            sampled_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, cur, py_ind, h, w)
            detail_feat = self.sample_detail_features(
                cnn_feature,
                cur,
                py_ind,
                h,
                w,
                sampled_feat=sampled_feat,
                contour_scale=contour_scale,
            )
            t_tensor = torch.full((N,), t_val, device=device, dtype=torch.float32)
            # Geometric bridge inference in v4_6c skips self-conditioning, disp-gate, latent-policy, and avg-samples.
            v_pred, _ = self.predict_velocity(
                cnn_feature,
                i_it_py,
                c_it_py,
                sampled_feat,
                detail_feat,
                py_ind,
                x_t,
                t_tensor,
                contour_scale=contour_scale_flat,
                x_self_cond=None,
                locate_context=locate_context,
                s=s,
            )
            x_t = x_t + v_pred * dt

        if geom_position_flow:
            final_poly = self.denormalize_position(x_t, centroid, contour_scale)
            disp = final_poly - i_it_py
        else:
            disp = self.denormalize_pred_disp(x_t, contour_scale)
        return self.clamp_pred_disp(disp, i_it_py)

    def sample_disp(self, cnn_feature, i_it_py, c_it_py, py_ind, steps=None, batch=None) -> torch.Tensor:
        """
        Euler ODE Solver for Flow Matching.
        V3.7.3: supports multi-trajectory averaging via infer_avg_samples config.
        """
        if steps is None:
            steps = self.ode_steps
            
        device = i_it_py.device
        N = i_it_py.size(0)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        contour_scale = self.compute_contour_scale(i_it_py)

        if self._geom_bridge and self._geom_infer_resample_per_ode:
            locate_context = None
            if self._locate_token_enabled and batch is not None:
                locate_context = self.build_locate_token_context(
                    batch,
                    i_it_py,
                    py_ind,
                    contour_scale=contour_scale,
                )
            return self._sample_disp_geom_bridge(
                cnn_feature,
                i_it_py,
                c_it_py,
                py_ind,
                steps=steps,
                locate_context=locate_context,
            )
        
        # 推理时：先进行一次特征采样
        sampled_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_py, py_ind, h, w)
        detail_feat = self.sample_detail_features(
            cnn_feature,
            i_it_py,
            py_ind,
            h,
            w,
            sampled_feat=sampled_feat,
            contour_scale=contour_scale,
        )
        locate_context = None
        if self._locate_token_enabled and batch is not None:
            locate_context = self.build_locate_token_context(
                batch,
                i_it_py,
                py_ind,
                contour_scale=contour_scale,
            )

        avg_n = self._infer_avg_samples
        if avg_n > 1:
            all_disps = []
            for _ in range(avg_n):
                d = self._sample_disp_from_sampled_feat(
                    cnn_feature, i_it_py, c_it_py, py_ind, sampled_feat, detail_feat,
                    steps=steps, locate_context=locate_context)
                all_disps.append(d)
            disp = torch.stack(all_disps).mean(dim=0)
            if self._use_disp_gate and self._disp_gate_apply_inference:
                gate = self.predict_disp_gate(sampled_feat, disp, contour_scale)
                disp = disp * gate
            return disp

        disp = self._sample_disp_from_sampled_feat(
            cnn_feature, i_it_py, c_it_py, py_ind, sampled_feat, detail_feat,
            steps=steps, locate_context=locate_context
        )
        if self._use_disp_gate and self._disp_gate_apply_inference:
            gate = self.predict_disp_gate(sampled_feat, disp, contour_scale)
            disp = disp * gate
        return disp

    def sample_disp_iterative(
        self,
        cnn_feature,
        i_it_py,
        c_it_py,
        py_ind,
        num_iter_steps=3,
        fractions=None,
        ode_steps=None,
        batch=None,
    ):
        """V3.4-style iterative refinement for Flow Matching."""
        if fractions is None:
            fractions = [1.0 / (num_iter_steps - i) for i in range(num_iter_steps)]
        if ode_steps is None:
            ode_steps = self.ode_steps

        N, P, _ = i_it_py.shape
        device = i_it_py.device
        if N == 0:
            return torch.zeros((0, P, 2), device=device, dtype=i_it_py.dtype)

        current_contour = i_it_py.clone()
        total_disp = torch.zeros(N, P, 2, device=device, dtype=i_it_py.dtype)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        cumulative_s = 0.0

        for step_idx in range(num_iter_steps):
            s_tensor = None
            if self._use_s_cond:
                s_tensor = torch.full((N,), cumulative_s, device=device, dtype=i_it_py.dtype)
            sampled_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, current_contour, py_ind, h, w)
            contour_scale = self.compute_contour_scale(current_contour)
            detail_feat = self.sample_detail_features(
                cnn_feature,
                current_contour,
                py_ind,
                h,
                w,
                sampled_feat=sampled_feat,
                contour_scale=contour_scale,
            )
            locate_context = None
            if self._locate_token_enabled and batch is not None:
                locate_context = self.build_locate_token_context(
                    batch,
                    current_contour,
                    py_ind,
                    contour_scale=contour_scale,
            )
            # V6o: stochastic TTA ensemble for iterative inference
            avg_n = self._infer_avg_samples
            step_ns = self._iterative_noise_scales[step_idx] if (
                self._iterative_noise_scales and step_idx < len(self._iterative_noise_scales)
            ) else None
            if avg_n > 1:
                all_disps = []
                for _ in range(avg_n):
                    if self._geom_bridge and self._geom_infer_resample_per_ode:
                        d = self._sample_disp_geom_bridge(
                            cnn_feature,
                            current_contour,
                            c_it_py,
                            py_ind,
                            steps=ode_steps,
                            noise_scale=step_ns,
                            locate_context=locate_context,
                            s=s_tensor,
                        )
                    else:
                        d = self._sample_disp_from_sampled_feat(
                            cnn_feature, current_contour, c_it_py, py_ind,
                            sampled_feat, detail_feat, steps=ode_steps,
                            noise_scale=step_ns,
                            locate_context=locate_context,
                            s=s_tensor,
                        )
                    all_disps.append(d)
                disp = torch.stack(all_disps).mean(dim=0)
            else:
                if self._geom_bridge and self._geom_infer_resample_per_ode:
                    disp = self._sample_disp_geom_bridge(
                        cnn_feature,
                        current_contour,
                        c_it_py,
                        py_ind,
                        steps=ode_steps,
                        noise_scale=step_ns,
                        locate_context=locate_context,
                        s=s_tensor,
                    )
                else:
                    disp = self._sample_disp_from_sampled_feat(
                        cnn_feature,
                        current_contour,
                        c_it_py,
                        py_ind,
                        sampled_feat,
                        detail_feat,
                        steps=ode_steps,
                        noise_scale=step_ns,
                        locate_context=locate_context,
                        s=s_tensor,
                    )
            if self._use_disp_gate and self._disp_gate_apply_inference:
                gate = self.predict_disp_gate(sampled_feat, disp, contour_scale)
                disp = disp * gate
            frac = fractions[step_idx]
            applied_disp = disp * frac
            current_contour = current_contour + applied_disp
            total_disp = total_disp + applied_disp
            cumulative_s = cumulative_s + (1.0 - cumulative_s) * float(frac)

        return total_disp

    def sample_disp_curve(
        self,
        cnn_feature,
        i_it_py,
        c_it_py,
        py_ind,
        alpha: float = 2.0,
        steps: int = 20,
        noise_scale=None,
        batch=None,
        s_max: float = 0.97,
        resample_feat: bool = True,
    ):
        """2D FM curved inference path.
    
        Walks a power-law curve t(τ)=τ^(1/α), s(τ)=τ^α on the (s,t) plane.
        Each step combines t-direction velocity (model) and s-direction velocity (analytic, using d̂).
        Feature resampling follows the evolving contour (DeepSnake-style).
        """
        N, P, _ = i_it_py.shape
        device = i_it_py.device
        if N == 0:
            return torch.zeros((0, P, 2), device=device, dtype=i_it_py.dtype)
        if noise_scale is None:
            ns = self._infer_noise_scale
            noise_scale = self._flow_noise_scale if ns < 0 else ns

        steps = max(int(steps), 1)
        alpha = max(float(alpha), 1e-6)
        s_max = min(max(float(s_max), 0.0), 1.0 - 1e-4)
        x = torch.randn_like(i_it_py) * float(noise_scale)
        contour_scale = self.compute_contour_scale(i_it_py)
        contour_scale_flat = contour_scale.view(-1)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        dt = 1.0 / steps
        locate_context = None
        if self._locate_token_enabled and batch is not None:
            locate_context = self.build_locate_token_context(
                batch,
                i_it_py,
                py_ind,
                contour_scale=contour_scale,
            )
        sampled_feat = None
        detail_feat = None
        if not resample_feat:
            ctx = self.prepare_sampling_context(cnn_feature, i_it_py, py_ind, batch=batch)
            sampled_feat = ctx['sampled_feat']
            detail_feat = ctx['detail_feat']
            contour_scale = ctx['contour_scale']
            contour_scale_flat = contour_scale.view(-1)
            locate_context = ctx.get('locate_context', locate_context)

        t_prev = 0.0
        s_prev = 0.0
        for i in range(steps):
            tau = (i + 1) * dt
            t_val = tau ** (1.0 / alpha)
            s_val = min(tau ** alpha, s_max)
            delta_t = t_val - t_prev
            delta_s = s_val - s_prev
            t_tensor = torch.full((N,), t_val, device=device, dtype=x.dtype)
            s_tensor = torch.full((N,), s_val, device=device, dtype=x.dtype)
            if resample_feat:
                with torch.no_grad():
                    xt_disp = self.denormalize_pred_disp(x, contour_scale)
                    cur_contour = (i_it_py + xt_disp).detach()
                    cur_sampled_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, cur_contour, py_ind, h, w)
                    cur_detail_feat = self.sample_detail_features(
                        cnn_feature,
                        cur_contour,
                        py_ind,
                        h,
                        w,
                        sampled_feat=cur_sampled_feat,
                        contour_scale=contour_scale,
                    )
            else:
                cur_sampled_feat = sampled_feat
                cur_detail_feat = detail_feat
            v_t, _ = self.predict_velocity(
                cnn_feature,
                i_it_py,
                c_it_py,
                cur_sampled_feat,
                cur_detail_feat,
                py_ind,
                x,
                t_tensor,
                contour_scale=contour_scale_flat,
                locate_context=locate_context,
                s=s_tensor,
            )
            x1_hat = x + (1.0 - t_val) * v_t
            d_hat = x1_hat / max(1.0 - s_val, 1e-4)
            x = x + v_t * delta_t - d_hat * delta_s
            t_prev = t_val
            s_prev = s_val
        disp = self.denormalize_pred_disp(d_hat, contour_scale)
        if self._use_disp_gate and self._disp_gate_apply_inference:
            if sampled_feat is None:
                sampled_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_py, py_ind, h, w)
            gate = self.predict_disp_gate(sampled_feat, disp, contour_scale)
            disp = disp * gate
        return disp

    @staticmethod
    def _progress_targets_to_residual_fractions(targets):
        """Convert absolute progress targets into fractions of the current residual."""
        fractions = []
        prev = 0.0
        for target in targets:
            target = min(max(float(target), prev), 1.0)
            remaining = max(1.0 - prev, 1e-6)
            fractions.append((target - prev) / remaining)
            prev = target
        return fractions

    @staticmethod
    def _octagon_init_from_extreme(extreme_points: torch.Tensor) -> torch.Tensor:
        """Build the diffusion init contour from predicted/GT extreme points."""
        # V4.6c 推理时若 use_pred_extreme_init_for_inference=True，
        # ct_snake.attach_extreme_prediction 会先产生 output['ex']。
        # 这里把 4 个 refined extreme points 变成 octagon，再均匀上采样到 snake_config.poly_num
        # 个点，作为 flow/diffusion evolution 的初始 contour。
        # 这一步仍然依赖 detector bbox 的位置；bbox 错了，octagon 也会在错误区域附近。
        if (not torch.is_tensor(extreme_points)) or extreme_points.numel() == 0:
            if torch.is_tensor(extreme_points):
                return extreme_points.new_zeros((0, snake_config.poly_num, 2))
            return torch.zeros((0, snake_config.poly_num, 2))
        octagon = snake_decode.get_octagon(extreme_points[None])
        return snake_gcn_utils.uniform_upsample(octagon, snake_config.poly_num)[0]

    def forward(self, output: Dict[str, Any], cnn_feature: torch.Tensor, batch: Dict[str, Any]) -> Dict[str, Any]:
        ret = {}
        device = cnn_feature.device
        
        if self.training:
            train_dict = snake_gcn_utils.prepare_training(output, batch)
            from lib.utils.snake.sam_init import sam_init_enabled
            if sam_init_enabled():
                h, w = cnn_feature.size(2), cnn_feature.size(3)
                from lib.utils.snake.sam_init import maybe_replace_training_init, maybe_use_output_sam_training_init
                train_dict = maybe_use_output_sam_training_init(train_dict, output)
                if not torch.is_tensor(output.get('sam_i_it_py', None)):
                    train_dict = maybe_replace_training_init(
                        train_dict,
                        output,
                        batch,
                        device=device,
                        out_h=h,
                        out_w=w,
                    )
            init_source = str(getattr(global_cfg, 'diffusion_init_source', 'extreme')).strip().lower()
            if init_source in ('gt_box_octagon', 'gt_bbox_octagon', 'box_octagon', 'bbox_octagon'):
                train_dict = snake_gcn_utils.replace_training_init_with_gt_box_octagon(train_dict)
                ret['diffusion_init_source'] = init_source
            elif (
                bool(getattr(global_cfg, 'use_pred_extreme_init_for_diffusion', False))
                and torch.is_tensor(output.get('ex_pred', None))
            ):
                ex_for_init = output['ex_pred']
                if bool(getattr(global_cfg, 'detach_pred_extreme_init', True)):
                    ex_for_init = ex_for_init.detach()
                pred_i_it_py = self._octagon_init_from_extreme(ex_for_init)
                if pred_i_it_py.size(0) == train_dict['i_it_py'].size(0):
                    prob_cfg = float(getattr(global_cfg, 'pred_extreme_init_prob', -1.0))
                    pred_prob = 1.0 if prob_cfg < 0.0 else min(max(prob_cfg, 0.0), 1.0)
                    num_inst = int(pred_i_it_py.size(0))
                    if pred_prob >= 1.0:
                        pred_mask = torch.ones((num_inst,), device=pred_i_it_py.device, dtype=torch.bool)
                    elif pred_prob <= 0.0:
                        pred_mask = torch.zeros((num_inst,), device=pred_i_it_py.device, dtype=torch.bool)
                    else:
                        pred_mask = torch.rand((num_inst,), device=pred_i_it_py.device) < pred_prob

                    train_dict = dict(train_dict)
                    if bool(pred_mask.any()):
                        mix_mask = pred_mask.view(-1, 1, 1)
                        train_dict['i_it_py'] = torch.where(mix_mask, pred_i_it_py, train_dict['i_it_py'])
                        train_dict['c_it_py'] = snake_gcn_utils.img_poly_to_can_poly(train_dict['i_it_py'])
                    train_dict['i_pred_4py'] = ex_for_init
                    ret['pred_extreme_init_count'] = pred_mask.to(dtype=pred_i_it_py.dtype).sum()
                    ret['gt_extreme_init_count'] = (~pred_mask).to(dtype=pred_i_it_py.dtype).sum()
                    ret['pred_extreme_init_prob_effective'] = pred_i_it_py.new_tensor(float(pred_prob))
            ret.update(train_dict)
            if torch.is_tensor(output.get('ex_pred', None)):
                ret['ex_pred'] = output['ex_pred']
                if torch.is_tensor(output.get('i_gt_4py', None)):
                    ret['i_gt_4py'] = output['i_gt_4py']
            if torch.is_tensor(output.get('ex_box_jitter_count', None)):
                ret['ex_box_jitter_count'] = output['ex_box_jitter_count']
            for debug_key in ('locate_feat_residual_absmax', 'locate_feat_adapter_last_absmax'):
                if torch.is_tensor(output.get(debug_key, None)):
                    ret[debug_key] = output[debug_key].detach()
            
            i_init_train_py = train_dict['i_it_py']
            c_init_train_py = train_dict['c_it_py']
            i_gt_py = train_dict['i_gt_py'].clone()
            py_ind = train_dict['py_ind']
            point_mask_train = train_dict.get('point_mask', None)

            if i_init_train_py.numel() == 0:
                ret.update({'diff_loss': (cnn_feature.sum() * 0.0), 'py_pred': [i_init_train_py]})
                return ret

            h, w = cnn_feature.size(2), cnn_feature.size(3)

            # --- 对齐与归标化 ---
            def _signed_area(poly: torch.Tensor) -> torch.Tensor:
                x, y = poly[..., 0], poly[..., 1]
                x1, y1 = torch.roll(x, -1, 1), torch.roll(y, -1, 1)
                return 0.5 * torch.sum(x * y1 - x1 * y, dim=1)

            area_init, area_gt = _signed_area(i_init_train_py), _signed_area(i_gt_py)
            orient_mismatch = ((area_init >= 0) ^ (area_gt >= 0))
            if orient_mismatch.any():
                flipped_gt = torch.flip(i_gt_py, dims=[1])
                i_gt_py = torch.where(orient_mismatch.view(-1, 1, 1), flipped_gt, i_gt_py)

            if self._optimal_cyclic_align:
                # Optimal cyclic alignment: find shift k* = argmin_k sum||oct[i] - gt[(i+k)%N]||^2
                # Maximise cross-correlation  ↔  minimise squared displacement
                # For N=128 and typical batch sizes the O(N^2) loop is fast (<1 ms on GPU).
                # IMPORTANT: wrap in no_grad to avoid accumulating 128 autograd nodes.
                N_pts = i_gt_py.size(1)
                B_a = i_gt_py.size(0)
                with torch.no_grad():
                    oct_pts = i_init_train_py.detach()   # (B, N, 2)
                    gt_pts  = i_gt_py.detach()            # (B, N, 2)
                    shift_costs = torch.zeros(B_a, N_pts, device=device, dtype=oct_pts.dtype)
                    for k in range(N_pts):
                        diff = oct_pts - torch.roll(gt_pts, -k, 1)
                        shift_costs[:, k] = diff.pow(2).sum(dim=(1, 2))
                    best_k = shift_costs.argmin(dim=1)  # (B,)
                i_gt_py = torch.stack(
                    [torch.roll(i_gt_py[i], -int(best_k[i].item()), 0) for i in range(B_a)], 0
                )
            else:
                # Greedy: align by nearest point to oct[0] only.
                # Vectorised batched roll: no per-sample .item() (each of those
                # forces a host<->device sync and stalls the training step).
                d2 = (i_init_train_py[:, :1, :] - i_gt_py).pow(2).sum(-1)
                shift = d2.argmin(dim=1)                       # (B,)
                N_pts = i_gt_py.size(1)
                gather_idx = (
                    torch.arange(N_pts, device=i_gt_py.device).unsqueeze(0)
                    + shift.unsqueeze(1)
                ) % N_pts                                      # (B, N)
                i_gt_py = torch.gather(
                    i_gt_py,
                    1,
                    gather_idx.unsqueeze(-1).expand(-1, -1, i_gt_py.size(2)),
                )

            x1_raw = i_gt_py - i_init_train_py
            full_disp_for_s = x1_raw.clone()

            used_mixed_iter_interp = False
            if self.use_iterative_refinement and not self._geom_bridge:
                iter_steps = int(getattr(global_cfg, 'iterative_num_steps', 3))
                full_disp = x1_raw.clone()
                B = x1_raw.size(0)
                use_rich_state_sampling = bool(getattr(global_cfg, 'v4_9_use_rich_state_sampling', False))
                use_mixed_iter_interp = bool(getattr(global_cfg, 'v4_4_use_mixed_iter_interp', False))
                if use_rich_state_sampling:
                    cont_p = max(float(getattr(global_cfg, 'v4_9_continuous_state_prob', 0.60)), 0.0)
                    disc_p = max(float(getattr(global_cfg, 'v4_9_discrete_state_prob', 0.0)), 0.0)
                    small_p = max(float(getattr(global_cfg, 'v4_9_small_state_prob', 0.25)), 0.0)
                    far_p = max(float(getattr(global_cfg, 'v4_9_hard_far_state_prob', 0.10)), 0.0)
                    zero_p = max(float(getattr(global_cfg, 'v4_9_near_zero_state_prob', 0.05)), 0.0)
                    exact_zero_p = max(float(getattr(global_cfg, 'v4_9_exact_zero_state_prob', 0.0)), 0.0)
                    total_p = cont_p + disc_p + small_p + far_p + zero_p + exact_zero_p
                    if total_p > 0:
                        cont_p, disc_p, small_p, far_p, zero_p, exact_zero_p = (
                            cont_p / total_p,
                            disc_p / total_p,
                            small_p / total_p,
                            far_p / total_p,
                            zero_p / total_p,
                            exact_zero_p / total_p,
                        )
                        draw = torch.rand(B, device=device)
                        cont_mask = draw < cont_p
                        disc_mask = (draw >= cont_p) & (draw < cont_p + disc_p)
                        small_mask = (draw >= cont_p + disc_p) & (draw < cont_p + disc_p + small_p)
                        far_mask = (
                            (draw >= cont_p + disc_p + small_p)
                            & (draw < cont_p + disc_p + small_p + far_p)
                        )
                        zero_start = cont_p + disc_p + small_p + far_p
                        zero_mask = (draw >= zero_start) & (draw < zero_start + zero_p)
                        exact_zero_mask = draw >= (zero_start + zero_p)

                        # Vectorised, sync-free sampling. The previous version
                        # called mask.any() / mask.sum().item() per category;
                        # every one of those forces a host<->device sync and
                        # stalls the step. Here per-sample [lo, hi] bounds are
                        # gathered from the masks and a single uniform draw is
                        # rescaled, which is distribution-equivalent.
                        def _bounds(min_name, max_name, default_min, default_max):
                            lo_v = min(max(float(getattr(global_cfg, min_name, default_min)), 0.0), 0.999)
                            hi_v = min(max(float(getattr(global_cfg, max_name, default_max)), lo_v), 0.999)
                            return lo_v, hi_v

                        cont_lo, cont_hi = _bounds(
                            'v4_9_continuous_min_frac', 'v4_9_continuous_max_frac', 0.05, 0.85)
                        far_lo, far_hi = _bounds(
                            'v4_9_hard_far_min_frac', 'v4_9_hard_far_max_frac', 0.0, 0.20)
                        zero_lo, zero_hi = _bounds(
                            'v4_9_near_zero_min_frac', 'v4_9_near_zero_max_frac', 0.95, 0.995)
                        small_lo = min(max(self._small_disp_min_frac, 0.0), 0.999)
                        small_hi = min(max(self._small_disp_max_frac, small_lo), 0.999)

                        lo = torch.zeros(B, device=device, dtype=full_disp.dtype)
                        hi = torch.zeros(B, device=device, dtype=full_disp.dtype)
                        for _mask, (_lo, _hi) in (
                                (cont_mask, (cont_lo, cont_hi)),
                                (small_mask, (small_lo, small_hi)),
                                (far_mask, (far_lo, far_hi)),
                                (zero_mask, (zero_lo, zero_hi)),
                        ):
                            lo = torch.where(_mask, lo.new_full((), _lo), lo)
                            hi = torch.where(_mask, hi.new_full((), _hi), hi)

                        u = torch.rand(B, device=device, dtype=full_disp.dtype)
                        frac = (lo + u * (hi - lo)).view(B, 1, 1)

                        fractions = getattr(global_cfg, 'v4_9_discrete_fractions', None)
                        if fractions is None:
                            fractions = getattr(global_cfg, 'iterative_fractions', None)
                        if fractions:
                            frac_choices = torch.tensor(
                                [float(f) for f in fractions],
                                device=device,
                                dtype=full_disp.dtype,
                            ).clamp_(0.0, 0.999)
                            choice_idx = torch.randint(
                                0, frac_choices.numel(), (B,), device=device)
                            disc_frac = frac_choices[choice_idx]
                        else:
                            disc_frac = torch.randint(
                                0, iter_steps, (B,), device=device
                            ).to(dtype=full_disp.dtype) / float(max(iter_steps, 1))
                        frac = torch.where(
                            disc_mask.view(B, 1, 1), disc_frac.view(B, 1, 1), frac)
                        frac = torch.where(
                            exact_zero_mask.view(B, 1, 1), frac.new_full((), 1.0), frac)
                        used_mixed_iter_interp = True
                    else:
                        situations = torch.randint(0, iter_steps, (B,), device=device)
                        frac = situations.to(dtype=full_disp.dtype).view(B, 1, 1) / float(max(iter_steps, 1))
                elif use_mixed_iter_interp:
                    cont_p = max(float(getattr(global_cfg, 'v4_4_continuous_interp_prob', 0.70)), 0.0)
                    disc_p = max(float(getattr(global_cfg, 'v4_4_discrete_interp_prob', 0.20)), 0.0)
                    small_p = max(float(getattr(global_cfg, 'v4_4_small_interp_prob', self._small_disp_prob)), 0.0)
                    total_p = cont_p + disc_p + small_p
                    if total_p > 0:
                        cont_p, disc_p, small_p = cont_p / total_p, disc_p / total_p, small_p / total_p
                        draw = torch.rand(B, device=device)
                        cont_mask = draw < cont_p
                        disc_mask = (draw >= cont_p) & (draw < cont_p + disc_p)
                        small_mask = draw >= (cont_p + disc_p)

                        frac = torch.zeros(B, 1, 1, device=device, dtype=full_disp.dtype)

                        default_cont_max = float(max(iter_steps - 1, 0)) / float(max(iter_steps, 1))
                        cont_min = max(float(getattr(global_cfg, 'v4_4_continuous_interp_min_frac', 0.0)), 0.0)
                        cont_max = min(
                            float(getattr(global_cfg, 'v4_4_continuous_interp_max_frac', default_cont_max)),
                            0.99,
                        )
                        if cont_max < cont_min:
                            cont_max = cont_min
                        if cont_mask.any():
                            frac[cont_mask] = torch.empty(
                                int(cont_mask.sum().item()), 1, 1,
                                device=device, dtype=full_disp.dtype,
                            ).uniform_(cont_min, cont_max)

                        if disc_mask.any():
                            situations = torch.randint(
                                0, iter_steps,
                                (int(disc_mask.sum().item()),),
                                device=device,
                            )
                            frac[disc_mask] = situations.to(dtype=full_disp.dtype).view(-1, 1, 1) / float(max(iter_steps, 1))

                        if small_mask.any() and small_p > 0:
                            min_frac = min(max(self._small_disp_min_frac, 0.0), 0.99)
                            max_frac = min(max(self._small_disp_max_frac, min_frac), 0.99)
                            frac[small_mask] = torch.empty(
                                int(small_mask.sum().item()), 1, 1,
                                device=device, dtype=full_disp.dtype,
                            ).uniform_(min_frac, max_frac)
                        used_mixed_iter_interp = True
                    else:
                        situations = torch.randint(0, iter_steps, (B,), device=device)
                        frac = situations.to(dtype=full_disp.dtype).view(B, 1, 1) / float(max(iter_steps, 1))
                else:
                    situations = torch.randint(0, iter_steps, (B,), device=device)
                    frac = situations.to(dtype=full_disp.dtype).view(B, 1, 1) / float(max(iter_steps, 1))
                i_init_train_py = i_init_train_py + full_disp * frac
                x1_raw = full_disp * (1.0 - frac)

            if self._small_disp_prob > 0 and not used_mixed_iter_interp and not self._geom_bridge:
                prob = min(max(self._small_disp_prob, 0.0), 1.0)
                small_mask = torch.rand(x1_raw.size(0), device=device) < prob
                if small_mask.any():
                    min_frac = min(max(self._small_disp_min_frac, 0.0), 0.99)
                    max_frac = min(max(self._small_disp_max_frac, min_frac), 0.99)
                    frac = torch.empty(
                        int(small_mask.sum().item()), 1, 1,
                        device=device, dtype=x1_raw.dtype,
                    ).uniform_(min_frac, max_frac)
                    frac_full = torch.zeros_like(x1_raw[..., :1])
                    frac_full = frac_full.clone()
                    frac_full[small_mask] = frac
                    i_init_train_py = i_init_train_py + x1_raw * frac_full
                    x1_raw = x1_raw * (1.0 - frac_full)

            if self._use_s_cond:
                full_norm = full_disp_for_s.norm(dim=-1).mean(dim=-1).clamp_min(1e-6)
                residual_norm = x1_raw.norm(dim=-1).mean(dim=-1).clamp_min(1e-6)
                train_s = (1.0 - residual_norm / full_norm).clamp(0.0, 1.0)
            else:
                train_s = None

            c_init_train_py = snake_gcn_utils.img_poly_to_can_poly(i_init_train_py)
            contour_scale = self.compute_contour_scale(i_init_train_py)
            contour_scale_flat = contour_scale.view(-1)
            geom_position_flow = self._geom_bridge and self._geom_position_flow
            geom_seg_flow = geom_position_flow and self._geom_seg_flow
            centroid = None
            if geom_position_flow:
                centroid = self.compute_centroid(i_init_train_py)
                x1 = self.normalize_position(i_gt_py, centroid, contour_scale)
            else:
                x1 = self.normalize_target_disp(x1_raw, contour_scale)
            N = x1.size(0)

            # --- Flow Matching Core ---
            t = self.sample_train_t(N, device=device, dtype=x1.dtype)
            if geom_seg_flow:
                x_init = self.normalize_position(i_init_train_py, centroid, contour_scale)
                x_gt = x1
                x_noise = torch.randn_like(x_init) * self._geom_noise_scale
                if self._geom_x0_jitter > 0:
                    x_noise = x_noise + torch.randn_like(x_noise) * self._geom_x0_jitter
                x0 = x_noise
            elif geom_position_flow:
                x0 = self.normalize_position(i_init_train_py, centroid, contour_scale)
                if self._geom_x0_jitter > 0:
                    x0 = x0 + torch.randn_like(x1) * self._geom_x0_jitter
            elif self._geom_bridge:
                x0 = torch.zeros_like(x1)
                if self._geom_x0_jitter > 0:
                    x0 = x0 + torch.randn_like(x1) * self._geom_x0_jitter
            else:
                x0 = self.sample_train_x0(x1)
            use_geom_sched_sampling = (
                self._geom_bridge
                and self._geom_sched_sampling
                and self._resample_feat_at_xt
                and not geom_seg_flow
            )
            if use_geom_sched_sampling:
                sched_prob = min(max(self._geom_sched_prob, 0.0), 1.0)
                use_geom_sched_sampling = torch.rand(1, device=device).item() < sched_prob

            model_t = t
            if geom_seg_flow:
                t_init = self._get_geom_t_init()
                front_mask = t < t_init
                local_t_front = t / t_init
                local_t_back = (t - t_init) / (1.0 - t_init)
                model_t = torch.where(front_mask, local_t_front, local_t_back).clamp(0.0, 1.0)
                x0 = torch.where(front_mask, x_noise, x_init)
                x1 = torch.where(front_mask, x_init, x_gt)
                x_t = (1.0 - model_t) * x0 + model_t * x1
            elif use_geom_sched_sampling:
                inner_steps = max(int(self._geom_sched_inner_steps), 0)
                dt_inner = 1.0 / float(inner_steps + 1)
                k_land = int(torch.randint(inner_steps + 1, (1,), device=device).item())
                t_land = x1.new_full((N, 1, 1), float(k_land) * dt_inner)

                with torch.no_grad():
                    if geom_position_flow:
                        x_roll = x0.detach().clone()
                    else:
                        x_roll = torch.zeros_like(x1)
                    for j in range(k_land):
                        t_j_scalar = float(j) * dt_inner
                        t_j = x1.new_full((N, 1, 1), t_j_scalar)
                        if geom_position_flow:
                            cur_j = self.denormalize_position(
                                x_roll, centroid, contour_scale
                            ).detach()
                        else:
                            cur_j = (
                                i_init_train_py
                                + self.denormalize_pred_disp(x_roll, contour_scale)
                            ).detach()
                        sampled_feat_j = snake_gcn_utils.get_gcn_feature(
                            cnn_feature, cur_j, py_ind, h, w
                        )
                        detail_feat_j = self.sample_detail_features(
                            cnn_feature,
                            cur_j,
                            py_ind,
                            h,
                            w,
                            sampled_feat=sampled_feat_j,
                            contour_scale=contour_scale,
                        )
                        locate_context_j = None
                        if self._locate_token_enabled:
                            locate_context_j = self.build_locate_token_context(
                                batch,
                                cur_j,
                                py_ind,
                                contour_scale=contour_scale,
                            )
                        v_j, _ = self.predict_velocity(
                            cnn_feature,
                            i_init_train_py,
                            c_init_train_py,
                            sampled_feat_j,
                            detail_feat_j,
                            py_ind,
                            x_roll,
                            t_j.view(-1),
                            contour_scale=contour_scale_flat,
                            x_self_cond=None,
                            locate_context=locate_context_j,
                            s=train_s,
                        )
                        x_roll = x_roll + v_j * dt_inner
                x_t = x_roll.detach()
                t = t_land
                model_t = t
            else:
                x_t = (1.0 - t) * x0 + t * x1
                model_t = t

            # relative perturbation on x_t: noise scaled by per-point GT displacement magnitude.
            # sigma_rel * |x1| * randn -> auto-adapts to each point/sample, avoids guessing absolute scale.
            if self._geom_xt_jitter_rel > 0:
                if geom_position_flow:
                    x1_norm = (x1 - x0).norm(dim=-1, keepdim=True).clamp_min(1e-6)
                else:
                    x1_norm = x1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                x_t = x_t + torch.randn_like(x_t) * (self._geom_xt_jitter_rel * x1_norm)

            # --- 特征采样 & 预测 ---
            if geom_seg_flow:
                xt_feat_poly = self.denormalize_position(
                    x_t, centroid, contour_scale
                ).detach()
                feat_poly = torch.where(front_mask, i_init_train_py, xt_feat_poly)
            elif self._resample_feat_at_xt:
                if geom_position_flow:
                    feat_poly = self.denormalize_position(
                        x_t, centroid, contour_scale
                    ).detach()
                else:
                    xt_disp_raw = self.denormalize_pred_disp(x_t, contour_scale)
                    feat_poly = (i_init_train_py + xt_disp_raw).detach()
            else:
                feat_poly = i_init_train_py
            sampled_feat_curr = snake_gcn_utils.get_gcn_feature(cnn_feature, feat_poly, py_ind, h, w)
            detail_feat_curr = self.sample_detail_features(
                cnn_feature,
                feat_poly,
                py_ind,
                h,
                w,
                sampled_feat=sampled_feat_curr,
                contour_scale=contour_scale,
            )
            locate_context_curr = None
            if self._locate_token_enabled:
                locate_context_curr = self.build_locate_token_context(
                    batch,
                    feat_poly,
                    py_ind,
                    contour_scale=contour_scale,
                )
                for debug_key in ('locate_map_absmax', 'locate_point_ctx_absmax'):
                    if torch.is_tensor(locate_context_curr.get(debug_key, None)):
                        ret[debug_key] = locate_context_curr[debug_key].detach()

            # V6r: self-conditioning — 50% of training steps use a dry-run prediction
            x_self_cond = None
            if self._use_self_conditioning:
                if torch.rand(1).item() < 0.5:
                    with torch.no_grad():
                        v_dry, _ = self.predict_velocity(
                            cnn_feature,
                            i_init_train_py,
                            c_init_train_py,
                            sampled_feat_curr,
                            detail_feat_curr,
                            py_ind,
                            x_t,
                            model_t.view(-1),
                            contour_scale=contour_scale_flat,
                            x_self_cond=None,  # dry run always starts unconditioned
                            locate_context=locate_context_curr,
                            s=train_s,
                        )
                    # Self-cond = current estimate of clean displacement x1
                    x_self_cond = (x_t + (1.0 - model_t) * v_dry).detach()
                else:
                    x_self_cond = torch.zeros_like(x_t)

            v_pred, L_reg = self.predict_velocity(
                cnn_feature,
                i_init_train_py,
                c_init_train_py,
                sampled_feat_curr,
                detail_feat_curr,
                py_ind,
                x_t,
                model_t.view(-1),
                contour_scale=contour_scale_flat,
                x_self_cond=x_self_cond,
                locate_context=locate_context_curr,
                s=train_s,
            )

            # 5. 计算目标速度 V_target = X_1 - X_0
            if use_geom_sched_sampling and not geom_position_flow:
                v_target = x1
            else:
                v_target = x1 - x0
            x1_pred = x_t + (1.0 - model_t) * v_pred
            if geom_position_flow:
                pred_poly = self.denormalize_position(x1_pred, centroid, contour_scale)
                pred_disp = pred_poly - i_init_train_py
            else:
                pred_disp = self.denormalize_pred_disp(x1_pred, contour_scale)
            pred_disp = self.clamp_pred_disp(pred_disp, i_init_train_py)
            gate_loss_val = pred_disp.new_zeros(())
            disp_gate = pred_disp.new_ones((pred_disp.size(0), 1, 1))
            disp_gate_target = pred_disp.new_ones((pred_disp.size(0), 1, 1))
            pred_disp_for_contours = pred_disp
            if self._use_disp_gate and self.disp_gate_head is not None:
                disp_gate = self.predict_disp_gate(sampled_feat_curr, pred_disp, contour_scale)
                disp_gate_target = self.compute_disp_gate_target(pred_disp, x1_raw)
                gate_loss_val = F.smooth_l1_loss(disp_gate, disp_gate_target, reduction='mean')
                if self._disp_gate_apply_training_pred:
                    pred_disp_for_contours = pred_disp * disp_gate
            pred_contours = i_init_train_py + pred_disp_for_contours

            # 6. Flow Matching Loss (V3.7: optional spectral decomposition)
            if self._use_curvature_reweight:
                gt_poly = i_init_train_py + x1_raw
                point_weights = self.compute_curvature_weights(gt_poly).to(v_pred.dtype)
                point_mse = (v_pred - v_target).pow(2).mean(dim=-1)
                loss = (point_mse * point_weights).mean()
            elif self._spectral_loss_k > 0 and v_pred.size(1) > self._spectral_loss_k * 2:
                v_pred_lf = self.fourier_smooth(v_pred, self._spectral_loss_k)
                v_target_lf = self.fourier_smooth(v_target, self._spectral_loss_k)
                loss_lf = F.mse_loss(v_pred_lf, v_target_lf, reduction='mean')
                loss_hf = F.mse_loss(v_pred - v_pred_lf, v_target - v_target_lf, reduction='mean')
                loss = loss_lf + self._hf_loss_weight * loss_hf
            else:
                loss = F.mse_loss(v_pred, v_target, reduction='mean')

            # V6o: endpoint consistency loss
            # x1_pred = x_t + (1-t)*v_pred  → penalise deviation from x1
            # Equivalent to (1-t)^2 * MSE(v_pred, v_target), upweighting t≈0
            if self._endpoint_loss_weight > 0:
                x1_pred = x_t + (1.0 - model_t) * v_pred
                endpoint_loss = F.mse_loss(x1_pred, x1, reduction='mean')
                loss = loss + self._endpoint_loss_weight * endpoint_loss

            # V4.3: soft Chamfer loss — penalise predicted contour distance to GT boundary
            # Inputs are normalised by contour_scale so distances live in ~[0, 1] and
            # tau / weight are well-calibrated independently of image resolution.
            chamfer_loss_val = pred_contours.new_zeros(())
            if self._chamfer_loss_weight > 0:
                scale = contour_scale.to(pred_contours)          # (N, 1, 1)
                chamfer_loss_val = self._compute_soft_chamfer(
                    pred_contours / scale,
                    i_gt_py.to(pred_contours) / scale,
                    tau=self._chamfer_tau,
                )
                loss = loss + self._chamfer_loss_weight * chamfer_loss_val

            if self._use_disp_gate and self._disp_gate_loss_weight > 0:
                loss = loss + self._disp_gate_loss_weight * gate_loss_val

            # Add denoiser regularisation (Laplacian from V3.7, zero for others)
            loss = loss + L_reg

            ret.update({
                'diff_loss': loss,
                'diff_lossA1': loss,
                'diff_loss_total': loss,
                'diff_loss1': loss,
                'diff_loss_chamfer': chamfer_loss_val,
                'diff_loss_gate': gate_loss_val,
                'disp_gate_mean': disp_gate.mean(),
                'disp_gate_target_mean': disp_gate_target.mean(),
                'point_mask': point_mask_train,
                'py_pred': [i_init_train_py],
                'py': pred_contours,
                'pred_contours': pred_contours,
                'pred_disp': pred_disp_for_contours,
                'v_pred': v_pred.mean(), # For observation logging optional
            })
            
        else:
            # 推理时进行 ODE/flow 求解：从 i_it_py 初始轮廓出发，预测每个点的位移 disp。
            # 输出 ret['py'] = i_it_py + disp，即最终 contour；报告里的 final_contours.jsonl
            # 读取的就是这个最终多边形，而不是 detector 的 raw bbox。
            with torch.no_grad():
                if (
                    bool(getattr(global_cfg, 'use_pred_extreme_init_for_inference', False))
                    and torch.is_tensor(output.get('ex', None))
                ):
                    ex = output['ex']
                    # 当前配置优先使用 extreme refine 后的 ex 初始化：
                    # refined extreme -> octagon -> 128 点 contour -> evolution。
                    i_it_py = self._octagon_init_from_extreme(ex)
                    c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)
                    py_ind = output.get(
                        'ex_py_ind',
                        torch.zeros((i_it_py.size(0),), dtype=torch.long, device=i_it_py.device),
                    )
                    init = {
                        'i_it_4py': ex,
                        'c_it_4py': snake_gcn_utils.img_poly_to_can_poly(ex),
                        'ind': py_ind,
                        'i_it_py': i_it_py,
                        'c_it_py': c_it_py,
                        'py_ind': py_ind,
                    }
                else:
                    init = snake_gcn_utils.prepare_testing(output)
                ret.update(init)
                
                i_it_py = init['i_it_py']
                c_it_py = init['c_it_py']
                py_ind = init['py_ind']

                if i_it_py.numel() == 0:
                    disp = torch.zeros_like(i_it_py)
                    ret.update({'disp': disp, 'py': i_it_py})
                elif self.use_iterative_refinement:
                    # V4.6c 当前配置使用 iterative flow refinement：
                    # 从 init contour 出发分多段预测位移，最后 ret['py'] 才是 final contour。
                    # 这里不会删除 detector false positive；每个输入 bbox 都会尝试生成一个 contour。
                    iter_steps = int(getattr(global_cfg, 'iterative_num_steps', 3))
                    use_rich_infer_schedule = bool(
                        getattr(global_cfg, 'v4_9_use_rich_infer_schedule', False)
                    )
                    if use_rich_infer_schedule:
                        targets = list(getattr(global_cfg, 'v4_9_infer_target_fractions', []))
                        if not targets:
                            targets = [0.3333, 0.5, 0.80, 0.97, 1.0]
                        fractions = self._progress_targets_to_residual_fractions(targets)
                        iter_steps = len(fractions)
                    else:
                        fractions = list(getattr(global_cfg, 'iterative_fractions', []))
                    if not fractions:
                        fractions = [1.0 / (iter_steps - i) for i in range(iter_steps)]
                    iter_ode_steps = int(
                        getattr(
                            global_cfg,
                            'iterative_ode_steps',
                            getattr(global_cfg, 'iterative_ddim_steps', self.ode_steps),
                        )
                    )
                    if iter_ode_steps <= 0:
                        iter_ode_steps = self.ode_steps
                    disp = self.sample_disp_iterative(
                        cnn_feature,
                        i_it_py,
                        c_it_py,
                        py_ind,
                        num_iter_steps=iter_steps,
                        fractions=fractions,
                        ode_steps=iter_ode_steps,
                        batch=batch,
                    )
                    if self.use_fourier_smooth > 0:
                        disp = self.fourier_smooth(disp, self.use_fourier_smooth)
                    ret.update({'disp': disp, 'py': i_it_py + disp})
                else:
                    disp = self.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind, steps=self.ode_steps, batch=batch)
                    if self.use_fourier_smooth > 0:
                        disp = self.fourier_smooth(disp, self.use_fourier_smooth)
                    ret.update({
                        'disp': disp,
                        'py': i_it_py + disp
                    })
                
        return ret
