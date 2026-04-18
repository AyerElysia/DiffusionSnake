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
        )
        self.use_fourier_smooth = int(getattr(global_cfg, 'fourier_smooth_k', 0))

        # V3.6: V3 global query + iterative refinement + Flow Matching
        if getattr(global_cfg, 'use_dit_v3_6', False):
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
        # V2.3 Hybrid: Joint Attention with Odd-Even Injection
        elif getattr(global_cfg, 'use_hybrid', False):
            from .dit_denoiser_v2_3_hybrid import DiTFlowMatchingHybrid
            print(f"[FlowMatchingEvolution] Using HYBRID Flow Network V2.3 (Odd-Even Injection)")
            self.denoiser = DiTFlowMatchingHybrid(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
            )
        # V2.3 Default: Joint Attention MM-DiT
        else:
            from .dit_denoiser_v2_3 import DiTFlowMatchingV2_3
            print(f"[FlowMatchingEvolution] Using DiT Flow Network V2.3 (MM-DiT Joint)")
            self.denoiser = DiTFlowMatchingV2_3(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
            )

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

    def predict_velocity(self, cnn_feature, i_it_py, c_it_py, sampled_feat, py_ind, x_t, t_continuous):
        """
        包装接口：预测速度场 V_t
        """
        N = x_t.size(0)
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, i_it_py.size(1), i_it_py.device)
        
        # 将 t_[0,1] 转换到类似于 1~1000 以供 time_embedder 分辨
        t_scaled = t_continuous * 1000.0
        
        # DiT V2.3: 核心调用，务必传入 sampled_feat
        # 对应 DiTFlowMatchingV2_3.forward(cnn_feature, sampled_feat, x_t, t_scaled, ...)
        v_pred, L = self.denoiser(cnn_feature, sampled_feat, x_t, t_scaled, adj, polys=i_it_py, py_ind=py_ind)
        return v_pred, L

    def _sample_disp_from_sampled_feat(
        self,
        cnn_feature,
        i_it_py,
        c_it_py,
        py_ind,
        sampled_feat,
        steps=None,
    ) -> torch.Tensor:
        if steps is None:
            steps = self.ode_steps

        device = i_it_py.device
        N = i_it_py.size(0)
        x_t = torch.randn_like(i_it_py)
        dt = 1.0 / steps
        for i in range(steps):
            t_val = i * dt
            t_tensor = torch.full((N,), t_val, device=device, dtype=torch.float32)
            v_pred, _ = self.predict_velocity(
                cnn_feature, i_it_py, c_it_py, sampled_feat, py_ind, x_t, t_tensor
            )
            x_t = x_t + v_pred * dt

        return self.denormalize_disp(x_t)

    def sample_disp(self, cnn_feature, i_it_py, c_it_py, py_ind, steps=None) -> torch.Tensor:
        """
        Euler ODE Solver for Flow Matching
        """
        if steps is None:
            steps = self.ode_steps
            
        device = i_it_py.device
        N = i_it_py.size(0)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        
        # 推理时：先进行一次特征采样
        sampled_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_py, py_ind, h, w)
        return self._sample_disp_from_sampled_feat(
            cnn_feature, i_it_py, c_it_py, py_ind, sampled_feat, steps=steps
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
            disp = self._sample_disp_from_sampled_feat(
                cnn_feature,
                current_contour,
                c_it_py,
                py_ind,
                sampled_feat,
                steps=ode_steps,
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
            ret.update(train_dict)
            
            i_init_train_py = train_dict['i_it_py']
            c_init_train_py = train_dict['c_it_py']
            i_gt_py = train_dict['i_gt_py']
            py_ind = train_dict['py_ind']

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
                i_gt_py[orient_mismatch] = torch.flip(i_gt_py[orient_mismatch], dims=[1])

            d2 = (i_init_train_py[:, :1, :] - i_gt_py).pow(2).sum(-1)
            i_gt_py = torch.stack([torch.roll(i_gt_py[i], -int(d2[i].argmin().item()), 0) for i in range(i_gt_py.size(0))], 0)

            x1_raw = i_gt_py - i_init_train_py

            if self.use_iterative_refinement:
                iter_steps = int(getattr(global_cfg, 'iterative_num_steps', 3))
                full_disp = x1_raw.clone()
                B = x1_raw.size(0)
                situations = torch.randint(0, iter_steps, (B,), device=device)
                for sit in range(1, iter_steps):
                    mask = (situations == sit)
                    if mask.any():
                        frac = sit / iter_steps
                        i_init_train_py[mask] = i_init_train_py[mask] + full_disp[mask] * frac
                        x1_raw[mask] = full_disp[mask] * (1.0 - frac)

            x1 = self.normalize_disp(x1_raw)
            N = x1.size(0)

            # --- Flow Matching Core ---
            t = torch.rand(N, device=device).view(N, 1, 1)
            x0 = torch.randn_like(x1)
            x_t = (1.0 - t) * x0 + t * x1

            # --- 特征采样 & 预测 ---
            sampled_feat_curr = snake_gcn_utils.get_gcn_feature(cnn_feature, i_init_train_py, py_ind, h, w)
            v_pred, _ = self.predict_velocity(cnn_feature, i_init_train_py, c_init_train_py, sampled_feat_curr, py_ind, x_t, t.view(-1))

            # 5. 计算目标速度 V_target = X_1 - X_0
            v_target = x1 - x0

            # 6. Flow Matching Loss
            loss = F.mse_loss(v_pred, v_target, reduction='mean')

            ret.update({
                'diff_loss': loss,
                'diff_lossA1': loss,
                'diff_loss_total': loss,
                'diff_loss1': loss,
                'py_pred': [i_init_train_py],
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
                    disp = self.sample_disp_iterative(
                        cnn_feature,
                        i_it_py,
                        c_it_py,
                        py_ind,
                        num_iter_steps=iter_steps,
                        fractions=fractions,
                        ode_steps=self.ode_steps,
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
