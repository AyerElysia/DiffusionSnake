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
from lib.utils.snake import snake_config
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
            _moe_num_experts = int(getattr(global_cfg, 'v4_6_moe_num_experts', 8))
            _moe_top_k = int(getattr(global_cfg, 'v4_6_moe_top_k', 2))
            _moe_balance_weight = float(getattr(global_cfg, 'v4_6_moe_balance_weight', 1e-3))
            _moe_expert_init_std = float(getattr(global_cfg, 'v4_6_moe_expert_init_std', 1e-4))
            _moe_router_noise_std = float(getattr(global_cfg, 'v4_6_moe_router_noise_std', 0.01))
            _moe_point_embed = bool(getattr(global_cfg, 'v4_6_moe_use_point_embed', True))
            _moe_cyclic_router = bool(getattr(global_cfg, 'v4_6_moe_use_cyclic_router', True))
            _moe_shared_expert = bool(getattr(global_cfg, 'v4_6_moe_use_shared_expert', False))
            _moe_routed_scale = float(getattr(global_cfg, 'v4_6_moe_routed_expert_scale', 1.0))
            _moe_expert_type = str(getattr(global_cfg, 'v4_6_moe_expert_type', 'linear')).strip().lower()
            _moe_expert_hidden = int(getattr(global_cfg, 'v4_6_moe_expert_hidden_dim', 256))
            _latent_loop = bool(getattr(global_cfg, 'v4_7_use_latent_loop', False))
            _latent_loop_steps = int(getattr(global_cfg, 'v4_7_latent_loop_steps', 4))
            print(f"[FlowMatchingEvolution] Using DiT Flow Network V4.1 "
                  f"(V3.4 detail backbone, detail_ctx={_detail_ctx}, "
                  f"detail_mode={_detail_mode}, per_point_delta={_use_pp_delta}, "
                  f"delta_head={_pp_delta_head_type}, delta_scale={_pp_delta_scale}, "
                  f"delta_reg={_pp_delta_reg}, "
                  f"final_head={_final_head_type}, "
                  f"moe_experts={_moe_num_experts}, moe_top_k={_moe_top_k}, "
                  f"moe_shared={_moe_shared_expert}, "
                  f"moe_expert_type={_moe_expert_type}, "
                  f"moe_expert_hidden={_moe_expert_hidden}, "
                  f"latent_loop={_latent_loop}, latent_loop_steps={_latent_loop_steps}, "
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
                moe_expert_init_std=_moe_expert_init_std,
                moe_router_noise_std=_moe_router_noise_std,
                moe_use_point_embed=_moe_point_embed,
                moe_use_cyclic_router=_moe_cyclic_router,
                moe_use_shared_expert=_moe_shared_expert,
                moe_routed_expert_scale=_moe_routed_scale,
                moe_expert_type=_moe_expert_type,
                moe_expert_hidden_dim=_moe_expert_hidden,
                use_latent_loop=_latent_loop,
                latent_loop_steps=_latent_loop_steps,
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

        # V6s: optimal cyclic alignment — find cyclic shift that minimises total displacement MSE
        # (greedy nearest-first-point can be far from optimal for tortuous contours)
        self._optimal_cyclic_align = bool(getattr(global_cfg, 'v3_7_optimal_cyclic_align', False))
        # V3.7: per-step ODE smoothing
        self._ode_smooth_k = int(getattr(global_cfg, 'v3_7_ode_smooth_k', 0))
        # V6p: ODE solver selection (euler/heun — Heun's method gives 2nd-order accuracy)
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
    ):
        """
        包装接口：预测速度场 V_t
        """
        N = x_t.size(0)
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, i_it_py.size(1), i_it_py.device)
        
        # 将 t_[0,1] 转换到类似于 1~1000 以供 time_embedder 分辨
        t_scaled = t_continuous * 1000.0
        
        # Flow denoiser 核心调用，务必传入 sampled_feat
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
        )
        return v_pred, L

    def prepare_sampling_context(
        self,
        cnn_feature: torch.Tensor,
        i_it_py: torch.Tensor,
        py_ind: torch.Tensor,
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
        return {
            'sampled_feat': sampled_feat,
            'detail_feat': detail_feat,
            'contour_scale': contour_scale,
            'h': h,
            'w': w,
        }

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
            )
            step_mean = x_t + (v_pred + v_pred2) * 0.5 * dt
        else:
            step_mean = x_t + v_pred * dt

        if self._ode_smooth_k > 0:
            step_mean = self.fourier_smooth(step_mean, self._ode_smooth_k)

        prev_sample, log_prob, step_mean, std = self._gaussian_step_with_logprob(
            step_mean,
            action_std=action_std,
            dt=dt,
            prev_sample=prev_sample,
            generator=generator,
        )
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
    ) -> Dict[str, Any]:
        if steps is None:
            steps = self.ode_steps
        steps = max(int(steps), 1)
        if noise_scale is None:
            noise_scale = self._flow_train_noise_scale

        device = i_it_py.device
        x = torch.randn_like(i_it_py) * float(noise_scale)
        ctx = self.prepare_sampling_context(cnn_feature, i_it_py, py_ind)

        if window_size and window_size > 0:
            start_min = int(window_range[0]) if isinstance(window_range, (tuple, list)) and len(window_range) > 0 else 0
            end_max = int(window_range[1]) if isinstance(window_range, (tuple, list)) and len(window_range) > 1 else steps
            end_max = max(end_max, window_size)
            if end_max <= start_min + window_size:
                s = max(0, min(start_min, max(steps - window_size, 0)))
            else:
                s = np.random.randint(start_min, end_max - window_size + 1)
            e = min(s + window_size, steps)
        else:
            s = 0
            e = steps

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
            if idx == s:
                latents_seq.append(x.detach())
            in_policy_window = idx >= s and idx < e

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
    ) -> torch.Tensor:
        if steps is None:
            steps = self.ode_steps
        if noise_scale is None:
            ns = self._infer_noise_scale
            noise_scale = self._flow_noise_scale if ns < 0 else ns

        device = i_it_py.device
        N = i_it_py.size(0)
        x_t = torch.randn_like(i_it_py) * noise_scale
        contour_scale = self.compute_contour_scale(i_it_py)
        contour_scale_flat = contour_scale.view(-1)
        dt = 1.0 / steps

        use_heun = (self._ode_solver == 'heun')
        # V6r: self-conditioning state — starts at zero, updated each step
        x_self_cond = torch.zeros_like(x_t) if self._use_self_conditioning else None

        for i in range(steps):
            t_val = i * dt
            t_tensor = torch.full((N,), t_val, device=device, dtype=torch.float32)
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
            )

            # Update self-conditioning with current x1 estimate
            if self._use_self_conditioning:
                x_self_cond = (x_t + (1.0 - t_val) * v_pred).detach()

            if use_heun and i < steps - 1:
                # Heun's method (2nd-order Runge-Kutta): predictor-corrector
                x_pred = x_t + v_pred * dt
                t_next = torch.full((N,), t_val + dt, device=device, dtype=torch.float32)
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
                    x_self_cond=x_self_cond,  # use updated self-cond for corrector step
                )
                x_t = x_t + (v_pred + v_pred2) * 0.5 * dt
            else:
                x_t = x_t + v_pred * dt

            # V3.7: per-step ODE Fourier smoothing
            if self._ode_smooth_k > 0:
                x_t = self.fourier_smooth(x_t, self._ode_smooth_k)

        disp = self.denormalize_pred_disp(x_t, contour_scale)
        return self.clamp_pred_disp(disp, i_it_py)

    def sample_disp(self, cnn_feature, i_it_py, c_it_py, py_ind, steps=None) -> torch.Tensor:
        """
        Euler ODE Solver for Flow Matching.
        V3.7.3: supports multi-trajectory averaging via infer_avg_samples config.
        """
        if steps is None:
            steps = self.ode_steps
            
        device = i_it_py.device
        N = i_it_py.size(0)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        
        # 推理时：先进行一次特征采样
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

        avg_n = self._infer_avg_samples
        if avg_n > 1:
            all_disps = []
            for _ in range(avg_n):
                d = self._sample_disp_from_sampled_feat(
                    cnn_feature, i_it_py, c_it_py, py_ind, sampled_feat, detail_feat, steps=steps)
                all_disps.append(d)
            return torch.stack(all_disps).mean(dim=0)

        return self._sample_disp_from_sampled_feat(
            cnn_feature, i_it_py, c_it_py, py_ind, sampled_feat, detail_feat, steps=steps
        )

    def sample_disp_iterative(
        self,
        cnn_feature,
        i_it_py,
        c_it_py,
        py_ind,
        num_iter_steps=3,
        fractions=None,
        ode_steps=None,
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

        for step_idx in range(num_iter_steps):
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
            # V6o: stochastic TTA ensemble for iterative inference
            avg_n = self._infer_avg_samples
            step_ns = self._iterative_noise_scales[step_idx] if (
                self._iterative_noise_scales and step_idx < len(self._iterative_noise_scales)
            ) else None
            if avg_n > 1:
                all_disps = []
                for _ in range(avg_n):
                    d = self._sample_disp_from_sampled_feat(
                        cnn_feature, current_contour, c_it_py, py_ind,
                        sampled_feat, detail_feat, steps=ode_steps,
                        noise_scale=step_ns,
                    )
                    all_disps.append(d)
                disp = torch.stack(all_disps).mean(dim=0)
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
                )
            frac = fractions[step_idx]
            applied_disp = disp * frac
            current_contour = current_contour + applied_disp
            total_disp = total_disp + applied_disp

        return total_disp

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
            ret.update(train_dict)
            
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
                # Greedy: align by nearest point to oct[0] only
                d2 = (i_init_train_py[:, :1, :] - i_gt_py).pow(2).sum(-1)
                i_gt_py = torch.stack([torch.roll(i_gt_py[i], -int(d2[i].argmin().item()), 0) for i in range(i_gt_py.size(0))], 0)

            x1_raw = i_gt_py - i_init_train_py

            used_mixed_iter_interp = False
            if self.use_iterative_refinement:
                iter_steps = int(getattr(global_cfg, 'iterative_num_steps', 3))
                full_disp = x1_raw.clone()
                B = x1_raw.size(0)
                use_rich_state_sampling = bool(getattr(global_cfg, 'v4_9_use_rich_state_sampling', False))
                use_mixed_iter_interp = bool(getattr(global_cfg, 'v4_4_use_mixed_iter_interp', False))
                if use_rich_state_sampling:
                    cont_p = max(float(getattr(global_cfg, 'v4_9_continuous_state_prob', 0.60)), 0.0)
                    small_p = max(float(getattr(global_cfg, 'v4_9_small_state_prob', 0.25)), 0.0)
                    far_p = max(float(getattr(global_cfg, 'v4_9_hard_far_state_prob', 0.10)), 0.0)
                    zero_p = max(float(getattr(global_cfg, 'v4_9_near_zero_state_prob', 0.05)), 0.0)
                    total_p = cont_p + small_p + far_p + zero_p
                    if total_p > 0:
                        cont_p, small_p, far_p, zero_p = (
                            cont_p / total_p,
                            small_p / total_p,
                            far_p / total_p,
                            zero_p / total_p,
                        )
                        draw = torch.rand(B, device=device)
                        cont_mask = draw < cont_p
                        small_mask = (draw >= cont_p) & (draw < cont_p + small_p)
                        far_mask = (draw >= cont_p + small_p) & (draw < cont_p + small_p + far_p)
                        zero_mask = draw >= (cont_p + small_p + far_p)

                        frac = torch.zeros(B, 1, 1, device=device, dtype=full_disp.dtype)

                        def _sample_frac(mask, min_name, max_name, default_min, default_max):
                            if not mask.any():
                                return
                            min_frac = min(max(float(getattr(global_cfg, min_name, default_min)), 0.0), 0.999)
                            max_frac = min(max(float(getattr(global_cfg, max_name, default_max)), min_frac), 0.999)
                            frac[mask] = torch.empty(
                                int(mask.sum().item()), 1, 1,
                                device=device, dtype=full_disp.dtype,
                            ).uniform_(min_frac, max_frac)

                        _sample_frac(
                            cont_mask,
                            'v4_9_continuous_min_frac',
                            'v4_9_continuous_max_frac',
                            0.05,
                            0.85,
                        )
                        if small_mask.any():
                            min_frac = min(max(self._small_disp_min_frac, 0.0), 0.999)
                            max_frac = min(max(self._small_disp_max_frac, min_frac), 0.999)
                            frac[small_mask] = torch.empty(
                                int(small_mask.sum().item()), 1, 1,
                                device=device, dtype=full_disp.dtype,
                            ).uniform_(min_frac, max_frac)
                        _sample_frac(
                            far_mask,
                            'v4_9_hard_far_min_frac',
                            'v4_9_hard_far_max_frac',
                            0.0,
                            0.20,
                        )
                        _sample_frac(
                            zero_mask,
                            'v4_9_near_zero_min_frac',
                            'v4_9_near_zero_max_frac',
                            0.95,
                            0.995,
                        )
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

            if self._small_disp_prob > 0 and not used_mixed_iter_interp:
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

            c_init_train_py = snake_gcn_utils.img_poly_to_can_poly(i_init_train_py)
            contour_scale = self.compute_contour_scale(i_init_train_py)
            contour_scale_flat = contour_scale.view(-1)
            x1 = self.normalize_target_disp(x1_raw, contour_scale)
            N = x1.size(0)

            # --- Flow Matching Core ---
            t = self.sample_train_t(N, device=device, dtype=x1.dtype)
            x0 = self.sample_train_x0(x1)
            x_t = (1.0 - t) * x0 + t * x1

            # --- 特征采样 & 预测 ---
            sampled_feat_curr = snake_gcn_utils.get_gcn_feature(cnn_feature, i_init_train_py, py_ind, h, w)
            detail_feat_curr = self.sample_detail_features(
                cnn_feature,
                i_init_train_py,
                py_ind,
                h,
                w,
                sampled_feat=sampled_feat_curr,
                contour_scale=contour_scale,
            )

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
                            t.view(-1),
                            contour_scale=contour_scale_flat,
                            x_self_cond=None,  # dry run always starts unconditioned
                        )
                    # Self-cond = current estimate of clean displacement x1
                    x_self_cond = (x_t + (1.0 - t) * v_dry).detach()
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
                t.view(-1),
                contour_scale=contour_scale_flat,
                x_self_cond=x_self_cond,
            )

            # 5. 计算目标速度 V_target = X_1 - X_0
            v_target = x1 - x0
            x1_pred = x_t + (1.0 - t) * v_pred
            pred_disp = self.denormalize_pred_disp(x1_pred, contour_scale)
            pred_disp = self.clamp_pred_disp(pred_disp, i_init_train_py)
            pred_contours = i_init_train_py + pred_disp

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
                x1_pred = x_t + (1.0 - t) * v_pred
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

            # Add denoiser regularisation (Laplacian from V3.7, zero for others)
            loss = loss + L_reg

            ret.update({
                'diff_loss': loss,
                'diff_lossA1': loss,
                'diff_loss_total': loss,
                'diff_loss1': loss,
                'diff_loss_chamfer': chamfer_loss_val,
                'point_mask': point_mask_train,
                'py_pred': [i_init_train_py],
                'py': pred_contours,
                'pred_contours': pred_contours,
                'pred_disp': pred_disp,
                'v_pred': v_pred.mean(), # For observation logging optional
            })
            
        else:
            # 推理时进行ODE求解
            with torch.no_grad():
                init = snake_gcn_utils.prepare_testing(output)
                ret.update(init)
                
                i_it_py = init['i_it_py']
                c_it_py = init['c_it_py']
                py_ind = init['py_ind']

                if i_it_py.numel() == 0:
                    disp = torch.zeros_like(i_it_py)
                    ret.update({'disp': disp, 'py': i_it_py})
                elif self.use_iterative_refinement:
                    iter_steps = int(getattr(global_cfg, 'iterative_num_steps', 3))
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
                    )
                    if self.use_fourier_smooth > 0:
                        disp = self.fourier_smooth(disp, self.use_fourier_smooth)
                    ret.update({'disp': disp, 'py': i_it_py + disp})
                else:
                    disp = self.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind, steps=self.ode_steps)
                    if self.use_fourier_smooth > 0:
                        disp = self.fourier_smooth(disp, self.use_fourier_smooth)
                    ret.update({
                        'disp': disp,
                        'py': i_it_py + disp
                    })
                
        return ret
