import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler
import os
import sys
import re
import numpy as np
from typing import Tuple, Optional, Dict, Any
import json

from .snake_denoiser import SnakeDenoiser
from .dit_denoiser import DiTDenoiser
from .dit_denoiser_v2 import DiTDenoiserV2
from .dit_denoiser_v2_2 import DiTDenoiserV2_2
from .dit_denoiser_v2_2_hybrid import DiTDenoiserV2_2Hybrid
from .dit_denoiser_v3 import DiTDenoiserV3
from .dit_denoiser_v3_1 import DiTDenoiserV3_1
import lib.utils.snake.snake_gcn_utils as snake_gcn_utils
from lib.utils.snake import snake_config
from lib.config import cfg as global_cfg


def remap_legacy_state_dict(sd: dict) -> dict:
    """Remap legacy checkpoint keys to match current model architecture.

    Handles the rename of ``time_emb_1`` / ``time_emb_3`` (old separate
    nn.Linear layers) to ``time_emb_net.1`` / ``time_emb_net.3``
    (nn.Sequential).
    """
    _LEGACY_RE = re.compile(r'(\.?)time_emb_(\d)(\..*)')
    new_sd = {}
    for k, v in sd.items():
        m = _LEGACY_RE.search(k)
        if m:
            new_k = _LEGACY_RE.sub(r'\1time_emb_net.\2\3', k)
            new_sd[new_k] = v
        else:
            new_sd[k] = v
    return new_sd


def _select_denoiser_type(global_cfg, use_dit_v2, use_dit_v2_1, use_dit_v2_2,
                          use_dit_denoiser, use_hybrid=False):
    """Determine denoiser type from config flags with clear precedence.

    Returns:
        str: One of 'dit_v3_1', 'dit_v3', 'dit_v2_2_hybrid', 'dit_v2_2',
             'dit_v2_1', 'dit_v2', 'dit_v1', 'snake'
    """
    if getattr(global_cfg, 'use_dit_v3_1', False):
        return 'dit_v3_1'
    elif getattr(global_cfg, 'use_dit_v3', False):
        return 'dit_v3'
    elif use_dit_v2_2:
        if use_hybrid or getattr(global_cfg, 'use_hybrid', False):
            return 'dit_v2_2_hybrid'
        return 'dit_v2_2'
    elif use_dit_v2_1:
        return 'dit_v2_1'
    elif use_dit_v2:
        return 'dit_v2'
    elif use_dit_denoiser:
        return 'dit_v1'
    else:
        return 'snake'


class DiffusionEvolution(nn.Module):
    """
    预训练专用的扩散模型实现
    只包含预训练相关的方法和逻辑
    """
    def __init__(
        self,
        state_dim: int = 128,
        feature_dim: int = 64,
        num_points: int = 128,
        num_timesteps: int = 1000,
        use_ddim_inference: bool = True,
        loss_weight: float = 1.0,
        loss_type: str = 'adaptive',
        res_layers: int = 7,
        fusion_dim: int = 256,
        use_vm2: bool = True,
        use_dit_denoiser: bool = False,
        use_dit_v2: bool = False,         # 新增：是否使用 DiT V2 (全面升级版)
        use_dit_v2_1: bool = False,       # 新增：是否使用 DiT V2.1 (CNN Anchor Pooling 版)
        use_dit_v2_2: bool = False,       # 新增：是否使用 DiT V2.2 (MM-DiT Patchify 版)
        use_dit_v2_3: bool = False,       # 新增：是否使用 DiT V2.3
        use_flow_matching: bool = False,  # 新增：是否使用 Flow Matching
        flow_ode_steps: int = 10,         # Flow Matching ODE 步数
        dit_num_layers: int = 6,
        dit_num_heads: int = 8,
        dit_state_dim: int = 256,
    ):
        super().__init__()
        self.num_points = num_points
        self.T = num_timesteps
        self.use_ddim = use_ddim_inference
        self.loss_weight = loss_weight
        self.loss_type = loss_type

        # Determine denoiser type with clear precedence
        denoiser_type = _select_denoiser_type(
            global_cfg, use_dit_v2, use_dit_v2_1, use_dit_v2_2,
            use_dit_denoiser, use_hybrid=getattr(global_cfg, 'use_hybrid', False)
        )

        # Initialize denoiser based on type
        if denoiser_type == 'dit_v3_1':
            print(f"[DiffusionEvolution] Using DiT Denoiser V3.1 (Patchify + Self->Cross Flow) "
                  f"(layers={dit_num_layers}, heads={dit_num_heads}, dim={dit_state_dim})")
            self.denoiser = DiTDenoiserV3_1(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
            )
        elif denoiser_type == 'dit_v3':
            print(f"[DiffusionEvolution] Using DiT Denoiser V3 (Perceiver Semantics + Self->Cross Flow) "
                  f"(layers={dit_num_layers}, heads={dit_num_heads}, dim={dit_state_dim})")
            self.denoiser = DiTDenoiserV3(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
            )
        elif denoiser_type == 'dit_v2_2_hybrid':
            print(f"[DiffusionEvolution] Using HYBRID DiT Denoiser V2.2 (Odd-Even Injection)")
            self.denoiser = DiTDenoiserV2_2Hybrid(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
            )
        elif denoiser_type == 'dit_v2_2':
            print(f"[DiffusionEvolution] Using DiT Denoiser V2.2 (MM-DiT Patchify) "
                  f"(layers={dit_num_layers}, heads={dit_num_heads}, dim={dit_state_dim})")
            self.denoiser = DiTDenoiserV2_2(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
            )
        elif denoiser_type in ('dit_v2_1', 'dit_v2'):
            ver = "V2.1 (Anchor Pool)" if denoiser_type == 'dit_v2_1' else "V2"
            print(f"[DiffusionEvolution] Using DiT Denoiser {ver} "
                  f"(layers={dit_num_layers}, heads={dit_num_heads}, dim={dit_state_dim})")
            self.denoiser = DiTDenoiserV2(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
                use_v2_1=(denoiser_type == 'dit_v2_1'),
            )
        elif denoiser_type == 'dit_v1':
            print("[DiffusionEvolution] Using DiT Denoiser V1")
            self.denoiser = DiTDenoiser(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
            )
        else:  # 'snake'
            print("[DiffusionEvolution] Using GCN Snake Denoiser")
            self.denoiser = SnakeDenoiser(
                state_dim=state_dim,
                use_vm2=use_vm2,
                feature_dim=feature_dim,
                res_layers=res_layers,
                fusion_dim=fusion_dim,
            )

        self.denoiser_type = denoiser_type
        
        # 根据配置选择调度器
        if use_ddim_inference:
            self.scheduler = DDIMScheduler(
                num_train_timesteps=self.T,
                beta_schedule='linear',
                prediction_type='epsilon',
                clip_sample=False,
            )
        else:
            self.scheduler = DDPMScheduler(
                num_train_timesteps=self.T,
                beta_schedule='linear',
                prediction_type='epsilon',
                clip_sample=False,
            )

        # Cache scheduler coefficients on the active device to avoid per-step allocations.
        self._sched_cache_device = None
        self._alphas_cumprod_dev = None
        self._sqrt_alphas_cumprod_dev = None
        self._sqrt_one_minus_alphas_cumprod_dev = None
        
        # 可选：CMAM先验
        self.compute_L = True
        
        # Store configuration for Snake
        self.snake_res_layers = res_layers
        self.fusion_state_dim = fusion_dim

        self._disp_norm_enabled = bool(getattr(global_cfg, 'diffusion_disp_norm', False))
        self._disp_min = None
        self._disp_max = None
        self._load_disp_stats(getattr(global_cfg, 'diffusion_disp_stats', ''))

    def _load_disp_stats(self, stats_path: str) -> None:
        if (not self._disp_norm_enabled) or (not stats_path):
            return
        try:
            if not os.path.exists(stats_path):
                raise FileNotFoundError(f"diffusion_disp_norm is enabled but stats file not found: {stats_path}")
            with open(stats_path, 'r') as f:
                s = json.load(f)
            if not all(k in s for k in ('dx_min', 'dx_max', 'dy_min', 'dy_max')):
                raise KeyError(f"disp stats json missing keys. Expected dx_min/dx_max/dy_min/dy_max. Got keys={list(s.keys())}")
            dx_min = float(s['dx_min'])
            dx_max = float(s['dx_max'])
            dy_min = float(s['dy_min'])
            dy_max = float(s['dy_max'])
            disp_min = torch.tensor([dx_min, dy_min], dtype=torch.float32).view(1, 1, 2)
            disp_max = torch.tensor([dx_max, dy_max], dtype=torch.float32).view(1, 1, 2)
            # Avoid register_buffer name conflict if attributes were pre-created (e.g., set to None).
            if hasattr(self, '_disp_min') and (not isinstance(getattr(self, '_disp_min', None), torch.Tensor)):
                try:
                    delattr(self, '_disp_min')
                except Exception:
                    pass
            if hasattr(self, '_disp_max') and (not isinstance(getattr(self, '_disp_max', None), torch.Tensor)):
                try:
                    delattr(self, '_disp_max')
                except Exception:
                    pass
            self.register_buffer('_disp_min', disp_min)
            self.register_buffer('_disp_max', disp_max)
        except Exception as e:
            # Fail fast: if user enabled normalization, do not silently proceed without it.
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
    
    def configure_snake(self, res_layers=7, fusion_dim=256):
        """Configure Snake backbone parameters"""
        self.snake_res_layers = res_layers
        self.fusion_state_dim = fusion_dim
        
        # Reconfigure deniser's Snake if it exists
        if hasattr(self.denoiser, 'snake'):
            # Note: This would require recreating the Snake module
            # For now, we'll store the config for future use
            pass

    def _ensure_sched_cache(self, device: torch.device):
        if self._sched_cache_device is not None and self._sched_cache_device == device:
            return
        try:
            alphas_cumprod = self.scheduler.alphas_cumprod
        except Exception:
            alphas_cumprod = None

        if alphas_cumprod is None:
            self._sched_cache_device = device
            self._alphas_cumprod_dev = None
            self._sqrt_alphas_cumprod_dev = None
            self._sqrt_one_minus_alphas_cumprod_dev = None
            return

        alphas_cumprod = alphas_cumprod.to(device=device)
        self._sched_cache_device = device
        self._alphas_cumprod_dev = alphas_cumprod
        self._sqrt_alphas_cumprod_dev = torch.sqrt(alphas_cumprod)
        self._sqrt_one_minus_alphas_cumprod_dev = torch.sqrt(1.0 - alphas_cumprod)

    def _add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Manual DDPM forward process: x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*eps.

        Implemented to avoid potential tensor caching/retention inside diffusers schedulers.
        """
        self._ensure_sched_cache(x0.device)
        if self._sqrt_alphas_cumprod_dev is None:
            # Fallback to scheduler implementation if we failed to cache buffers
            return self.scheduler.add_noise(x0, noise, t)

        # gather coefficients for each sample in batch
        t = t.long()
        a = self._sqrt_alphas_cumprod_dev.index_select(0, t).view(-1, 1, 1)
        am1 = self._sqrt_one_minus_alphas_cumprod_dev.index_select(0, t).view(-1, 1, 1)
        return a * x0 + am1 * noise

    def predict_eps(self, cnn_feature: torch.Tensor, i_it_py: torch.Tensor, c_it_py: torch.Tensor,
                   py_ind: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测噪声：使用去噪器预测当前状态中的噪声分量
        """
        # 1. 采样局部 GCN 特征 (保留作为 Local Context)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        gcn_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_py, py_ind, h, w)

        # 2. 构建邻接矩阵 (仅 GCN Denoiser 需要)
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, i_it_py.size(1), i_it_py.device)

        # 3. 通过去噪器预测噪声 (使用 denoiser_type 而非 isinstance)
        if self.denoiser_type.startswith('dit'):
            eps_pred, L = self.denoiser(cnn_feature, gcn_feat, x_t, t, adj, polys=i_it_py, py_ind=py_ind)
        else:  # snake
            eps_pred, L = self.denoiser(gcn_feat, c_it_py, x_t, t, adj, polys=i_it_py)

        return eps_pred, L

    @torch.no_grad()
    def sample_disp(self, cnn_feature, i_it_py, c_it_py, py_ind, steps: int = 50):
        """扩散采样：从纯噪声逐步去噪得到位移场"""
        N, P, _ = i_it_py.shape
        device = i_it_py.device
        if N == 0:
            return torch.zeros((0, P, 2), device=device, dtype=i_it_py.dtype)
        x = torch.randn(N, P, 2, device=device)

        # 设置调度步数
        self.scheduler.set_timesteps(steps, device=device)

        for t in self.scheduler.timesteps:
            t_batch = torch.full((N,), t, device=device, dtype=torch.long)
            eps_pred, _ = self.predict_eps(cnn_feature, i_it_py, c_it_py, py_ind, x, t_batch)
            out = self.scheduler.step(model_output=eps_pred, timestep=t, sample=x)
            x = out.prev_sample

        return self.denormalize_disp(x)

    def forward(self, output, cnn_feature, batch=None):
        ret = output
        if self.training:
            with torch.no_grad():
                # 1) 准备训练数据
                init = snake_gcn_utils.prepare_training(output, batch)
                ret.update({
                    'i_it_4py': init['i_it_4py'],
                    'i_it_py': init['i_it_py'],
                    'i_gt_4py': init['i_gt_4py'],
                    'i_gt_py': init['i_gt_py']
                })

                # 2) 直接复用数据构造阶段生成的真实 init 轮廓
                device = cnn_feature.device
                i_init_train_py = init['i_it_py'].to(device)
                c_init_train_py = init['c_it_py'].to(device)
                i_gt_py = init['i_gt_py'].to(device)
                py_ind = init['py_ind']

                # 仅保留 A1 路径，不构建 B/C 变体
                h, w = cnn_feature.size(2), cnn_feature.size(3)

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
                    nearest_cpu = nearest.cpu().tolist()
                    for i in range(i_gt_py.size(0)):
                        s = nearest_cpu[i]
                        if s != 0:
                            rolled.append(torch.roll(i_gt_py[i], shifts=-s, dims=0))
                        else:
                            rolled.append(i_gt_py[i])
                    i_gt_py = torch.stack(rolled, dim=0)

                x0 = i_gt_py - i_init_train_py

                x0 = self.normalize_disp(x0)

                # 不再构建 B/C 对齐与目标

                N = x0.size(0)
                if N == 0:
                    ret.update({'diff_loss': (cnn_feature.sum() * 0.0), 'py_pred': [i_init_train_py]})
                    return ret

                # 仅保留 A1：起始轮廓与完整目标位移
                vec_full = x0
                contour1 = i_init_train_py
                c_contour1 = c_init_train_py
                contours_combined = contour1
                c_combined = c_contour1
                x0_combined = vec_full
                py_ind_combined = py_ind

                N3 = x0_combined.size(0)
                t = torch.randint(0, self.T, (N3,), device=device, dtype=torch.long)
                noise = torch.randn_like(x0_combined)
                x_t = self._add_noise(x0_combined, noise, t)

            eps_pred, _ = self.predict_eps(cnn_feature, contours_combined, c_combined, py_ind_combined, x_t, t)
            N_orig = i_init_train_py.size(0)
            eps_pred_A1 = eps_pred[0 * N_orig:1 * N_orig]
            noise_A1 = noise[0 * N_orig:1 * N_orig]

            lossA1 = F.mse_loss(eps_pred_A1, noise_A1, reduction='mean') if eps_pred_A1.numel() > 0 else (cnn_feature.sum() * 0.0)
            diff_loss = lossA1

            ret.update({
                'diff_loss': diff_loss,
                'diff_lossA1': lossA1,
                'diff_loss_total': diff_loss,
                'diff_loss1': lossA1,
                'py_pred': [i_init_train_py],
            })
            
        else:
            # 推理时使用DDIM采样
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
                    disp = self.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind, steps=50)
                    ret.update({
                        'disp': disp,
                        'py': i_it_py + disp
                    })
                
        return ret
