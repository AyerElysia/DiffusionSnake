import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
from typing import Tuple, Optional, Dict, Any
import json

from .dit_denoiser_v2_3 import DiTFlowMatchingV2_3
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

        # V2.3 启用专属去噪网络
        print(f"[FlowMatchingEvolution] Using DiT Flow Network V2.3 "
              f"(layers={dit_num_layers}, heads={dit_num_heads}, dim={dit_state_dim})")
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

    def predict_velocity(self, cnn_feature, i_it_py, c_it_py, py_ind, x_t, t_continuous):
        """
        包装接口：将时间尺度放大 1000 倍，匹配 Sinusoidal Embedding 设计。
        """
        N = x_t.size(0)
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, i_it_py.size(1), i_it_py.device)
        
        # 将 t_[0,1] 转换到类似于 1~1000 以供 time_embedder 分辨
        t_scaled = t_continuous * 1000.0
        
        # DiT V2.3: 预测速度 Vector Field V_t
        v_pred, L = self.denoiser(cnn_feature, c_it_py, x_t, t_scaled, adj, polys=i_it_py, py_ind=py_ind)
        return v_pred, L

    def sample_disp(self, cnn_feature, i_it_py, c_it_py, py_ind, steps=None) -> torch.Tensor:
        """
        Euler ODE Solver for Flow Matching
        """
        if steps is None:
            steps = self.ode_steps
            
        device = i_it_py.device
        N = i_it_py.size(0)
        
        # x_0 = 标准正态分布噪声
        x_t = torch.randn_like(i_it_py)
        
        dt = 1.0 / steps
        for i in range(steps):
            # t takes values like 0, 0.1, 0.2 ... 0.9 (if steps=10)
            t_val = i * dt
            t_tensor = torch.full((N,), t_val, device=device, dtype=torch.float32)
            
            v_pred, _ = self.predict_velocity(cnn_feature, i_it_py, c_it_py, py_ind, x_t, t_tensor)
            
            # Euler 步进: X_{t + dt} = X_t + V_t * dt
            x_t = x_t + v_pred * dt

        # x_1 is the target displacement
        disp_pred_norm = x_t
        disp_pred = self.denormalize_disp(disp_pred_norm)
        return disp_pred

    def forward(self, output: Dict[str, Any], cnn_feature: torch.Tensor, batch: Dict[str, Any]) -> Dict[str, Any]:
        ret = {}
        device = cnn_feature.device
        
        if self.training:
            if 'i_gt_py' not in batch:
                raise ValueError("FlowMatchingEvolution: i_gt_py required for training")

            i_gt_py = batch['i_gt_py'][0]
            if 'i_it_py' in output and 'c_it_py' in output and 'py_ind' in output:
                i_init_train_py = output['i_it_py']
                c_init_train_py = output['c_it_py']
                py_ind = output['py_ind']
            else:
                train_dict = snake_gcn_utils.prepare_training(output, batch)
                ret.update(train_dict)
                i_init_train_py = train_dict['i_it_py']
                c_init_train_py = train_dict['c_it_py']
                py_ind = train_dict['py_ind']

            if i_init_train_py.numel() == 0:
                ret.update({'diff_loss': (cnn_feature.sum() * 0.0), 'py_pred': [i_init_train_py]})
                return ret

            h, w = cnn_feature.size(2), cnn_feature.size(3)

            # 对齐 GT 多边形方向并重排列点顺序，寻找使得 MSE 最小的点序匹配
            def _signed_area(poly: torch.Tensor) -> torch.Tensor:
                x = poly[..., 0]
                y = poly[..., 1]
                x1 = torch.roll(x, shifts=-1, dims=1)
                y1 = torch.roll(y, shifts=-1, dims=1)
                return 0.5 * torch.sum(x * y1 - x1 * y, dim=1)

            area_init = _signed_area(i_init_train_py)
            area_gt = _signed_area(i_gt_py)
            orient_mismatch = ((area_init >= 0) ^ (area_gt >= 0))
            if orient_mismatch.any():
                i_gt_py[orient_mismatch] = torch.flip(i_gt_py[orient_mismatch], dims=[1])

            d2 = (i_init_train_py[:, :1, :] - i_gt_py).pow(2).sum(-1)
            nearest = torch.argmin(d2, dim=1)
            if i_gt_py.size(0) > 0:
                rolled = []
                for i in range(i_gt_py.size(0)):
                    s = int(nearest[i].item())
                    if s != 0:
                        rolled.append(torch.roll(i_gt_py[i], shifts=-s, dims=0))
                    else:
                        rolled.append(i_gt_py[i])
                i_gt_py = torch.stack(rolled, dim=0)

            # Target Disp X_1
            x1 = i_gt_py - i_init_train_py
            x1 = self.normalize_disp(x1)

            N = x1.size(0)
            if N == 0:
                ret.update({'diff_loss': (cnn_feature.sum() * 0.0), 'py_pred': [i_init_train_py]})
                return ret

            # --- Flow Matching Core Logic ---
            # 1. 采样时间 t ~ U(0, 1)
            # 在[0,1)中进行均匀采样并转为float张量
            t = torch.rand(N, device=device, dtype=torch.float32)

            # 2. 生成完全纯噪声 X_0 ~ N(0, I)
            x0 = torch.randn_like(x1)

            # 3. 计算插值状态 X_t = (1-t)X_0 + tX_1
            # 广播 t 到 [N, 1, 1] 形状匹配 [N, P, 2]
            t_expand = t.view(N, 1, 1)
            x_t = (1.0 - t_expand) * x0 + t_expand * x1

            # 4. 预测速度 V_t
            v_pred, _ = self.predict_velocity(cnn_feature, i_init_train_py, c_init_train_py, py_ind, x_t, t)

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
                else:
                    disp = self.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind, steps=self.ode_steps)
                    ret.update({
                        'disp': disp,
                        'py': i_it_py + disp
                    })
                
        return ret
