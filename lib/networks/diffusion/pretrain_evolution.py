import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler
import logging
import os
import sys
import re
import numpy as np
from typing import Tuple, Optional, Dict, Any
import json

from .snake_denoiser import SnakeDenoiser
from .dit_denoiser import DiTDenoiser
from .dit_denoiser_v3 import DiTDenoiserV3
from .dit_denoiser_v3_1 import DiTDenoiserV3_1
from .dit_denoiser_v3_3 import DiTDenoiserV3_3  # NEW
from .dit_denoiser_v3_5 import DiTDenoiserV3_5  # V3.5 Fourier-space
import lib.utils.snake.snake_gcn_utils as snake_gcn_utils
from lib.utils.snake import snake_config
from lib.config import cfg as global_cfg

logger = logging.getLogger(__name__)


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


def _select_denoiser_type(global_cfg, use_dit_denoiser):
    """Determine denoiser type from config flags with clear precedence.

    Returns:
        str: One of 'dit_v3_5', 'dit_v3_3', 'dit_v3_2', 'dit_v3_1',
             'dit_v3', 'dit_v1', 'snake'
    """
    if getattr(global_cfg, 'use_dit_v3_5', False):
        return 'dit_v3_5'
    elif getattr(global_cfg, 'use_dit_v3_3', False):
        return 'dit_v3_3'
    elif getattr(global_cfg, 'use_dit_v3_2', False):
        return 'dit_v3_2'
    elif getattr(global_cfg, 'use_dit_v3_1', False):
        return 'dit_v3_1'
    elif getattr(global_cfg, 'use_dit_v3', False):
        return 'dit_v3'
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
        self.use_iterative_refinement = bool(
            getattr(global_cfg, 'use_iterative_refinement', False)
            or getattr(global_cfg, 'use_dit_v3_4', False)
        )
        if self.use_iterative_refinement:
            logger.info("[DiffusionEvolution] V3.4 iterative refinement enabled")

        # Determine denoiser type with clear precedence
        denoiser_type = _select_denoiser_type(global_cfg, use_dit_denoiser)

        # V3.5: Fourier-space diffusion config
        self.use_fourier_diffusion = (denoiser_type == 'dit_v3_5')
        self.fourier_k = getattr(global_cfg, 'fourier_k', 16)
        if self.use_fourier_diffusion:
            logger.info(f"[DiffusionEvolution] V3.5 Fourier-space diffusion enabled (K={self.fourier_k})")

        # Initialize denoiser based on type
        if denoiser_type == 'dit_v3_5':
            logger.info(
                f"[DiffusionEvolution] Using DiT Denoiser V3.5 (Fourier-Space) "
                f"(layers={dit_num_layers}, heads={dit_num_heads}, dim={dit_state_dim}, K={self.fourier_k})"
            )
            self.denoiser = DiTDenoiserV3_5(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
                num_fourier_k=self.fourier_k,
            )
        elif denoiser_type == 'dit_v3_3':
            circular_conv_kernel = getattr(global_cfg, 'circular_conv_kernel', 5)
            logger.info(
                f"[DiffusionEvolution] Using DiT Denoiser V3.3 (V3 + Circular Conv) "
                f"(layers={dit_num_layers}, heads={dit_num_heads}, dim={dit_state_dim}, kernel={circular_conv_kernel})"
            )
            self.denoiser = DiTDenoiserV3_3(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
                circular_conv_kernel=circular_conv_kernel,
            )
        elif denoiser_type == 'dit_v3_1':
            logger.info(
                f"[DiffusionEvolution] Using DiT Denoiser V3.1 (Patchify + Self->Cross Flow) "
                f"(layers={dit_num_layers}, heads={dit_num_heads}, dim={dit_state_dim})"
            )
            self.denoiser = DiTDenoiserV3_1(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
            )
        elif denoiser_type == 'dit_v3':
            logger.info(
                f"[DiffusionEvolution] Using DiT Denoiser V3 (Perceiver Semantics + Self->Cross Flow) "
                f"(layers={dit_num_layers}, heads={dit_num_heads}, dim={dit_state_dim})"
            )
            self.denoiser = DiTDenoiserV3(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
            )
        elif denoiser_type == 'dit_v1':
            logger.info("[DiffusionEvolution] Using DiT Denoiser V1")
            self.denoiser = DiTDenoiser(
                state_dim=dit_state_dim,
                feature_dim=feature_dim,
                num_layers=dit_num_layers,
                num_heads=dit_num_heads,
                num_points=num_points,
            )
        else:  # 'snake'
            logger.info("[DiffusionEvolution] Using GCN Snake Denoiser")
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

        # Fourier-domain statistics for V3.5 normalization
        self._fourier_mean = None
        self._fourier_std = None
        self._load_fourier_stats(getattr(global_cfg, 'fourier_disp_stats', ''))

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

    def _load_fourier_stats(self, stats_path: str) -> None:
        """Load Fourier-domain normalization statistics (mean/std)."""
        if not stats_path:
            return
        try:
            if not os.path.exists(stats_path):
                logger.warning(f"Fourier stats file not found: {stats_path}, will use spatial-based normalization")
                return
            with open(stats_path, 'r') as f:
                s = json.load(f)
            mean_val = float(s.get('fourier_global_mean', 0.0))
            std_val = float(s.get('fourier_global_std', 1.0))
            if std_val < 1e-8:
                logger.warning(f"Fourier std too small ({std_val}), skipping Fourier stats")
                return
            fourier_mean = torch.tensor(mean_val, dtype=torch.float32)
            fourier_std = torch.tensor(std_val, dtype=torch.float32)
            for attr in ('_fourier_mean', '_fourier_std'):
                if hasattr(self, attr) and not isinstance(getattr(self, attr, None), torch.Tensor):
                    try:
                        delattr(self, attr)
                    except Exception:
                        pass
            self.register_buffer('_fourier_mean', fourier_mean)
            self.register_buffer('_fourier_std', fourier_std)
            logger.info(f"[DiffusionEvolution] Loaded Fourier stats: mean={mean_val:.4f}, std={std_val:.4f}")
        except Exception as e:
            logger.warning(f"Failed to load Fourier stats: {e}")

    def _has_fourier_stats(self) -> bool:
        return (
            isinstance(getattr(self, '_fourier_mean', None), torch.Tensor)
            and isinstance(getattr(self, '_fourier_std', None), torch.Tensor)
        )

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

    def normalize_disp_fourier(self, fourier_coeff: torch.Tensor) -> torch.Tensor:
        """Normalize Fourier coefficients for diffusion.
        Uses Fourier-domain statistics (mean/std) when available for proper
        standardization. Falls back to spatial-range-based scaling otherwise.
        """
        if self._has_fourier_stats():
            mean = self._fourier_mean.to(fourier_coeff.device, fourier_coeff.dtype)
            std = self._fourier_std.to(fourier_coeff.device, fourier_coeff.dtype)
            return (fourier_coeff - mean) / std
        if not self._has_disp_stats():
            return fourier_coeff
        spatial_range = (self._disp_max - self._disp_min).clamp_min(1e-12)
        fft_scale = spatial_range.max().to(fourier_coeff.device, fourier_coeff.dtype) * (self.num_points / 2.0)
        return fourier_coeff / (fft_scale + 1e-8)

    def denormalize_disp_fourier(self, fourier_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize Fourier coefficients back to original scale."""
        if self._has_fourier_stats():
            mean = self._fourier_mean.to(fourier_norm.device, fourier_norm.dtype)
            std = self._fourier_std.to(fourier_norm.device, fourier_norm.dtype)
            return fourier_norm * std + mean
        if not self._has_disp_stats():
            return fourier_norm
        spatial_range = (self._disp_max - self._disp_min).clamp_min(1e-12)
        fft_scale = spatial_range.max().to(fourier_norm.device, fourier_norm.dtype) * (self.num_points / 2.0)
        return fourier_norm * (fft_scale + 1e-8)
    
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

    def _predict_x0_from_eps(self, x_t: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict x0 from x_t and predicted noise eps.

        From DDPM: x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*eps
        Solve for x0: x0 = (x_t - sqrt(1-a_bar)*eps) / sqrt(a_bar)
        """
        self._ensure_sched_cache(x_t.device)
        if self._sqrt_alphas_cumprod_dev is None:
            # Fallback: use scheduler's step function
            return self.scheduler.step(eps, t[0].item(), x_t).pred_original_sample

        t = t.long()
        a = self._sqrt_alphas_cumprod_dev.index_select(0, t).view(-1, 1, 1)
        am1 = self._sqrt_one_minus_alphas_cumprod_dev.index_select(0, t).view(-1, 1, 1)
        return (x_t - am1 * eps) / a

    def predict_eps(
        self,
        cnn_feature: torch.Tensor,
        i_it_py: torch.Tensor,
        c_it_py: torch.Tensor,
        py_ind: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        batch: dict = None,
        point_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测噪声：使用去噪器预测当前状态中的噪声分量

        Args:
            batch: V3.10: batch dictionary containing point_mask
        """
        # 1. 采样局部 GCN 特征 (保留作为 Local Context)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        gcn_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_py, py_ind, h, w)

        # 2. 构建邻接矩阵 (仅 GCN Denoiser 需要)
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, i_it_py.size(1), i_it_py.device)

        # V3.10: extract point_mask from batch when not explicitly provided.
        if point_mask is None and batch is not None and 'point_mask' in batch and 'ct_01' in batch:
            try:
                ct_01 = batch['ct_01'].byte()
                point_mask = snake_gcn_utils.collect_training(batch['point_mask'], ct_01)
            except Exception:
                point_mask = None
        if point_mask is not None:
            if point_mask.dim() != 2 or point_mask.shape[0] != i_it_py.shape[0] or point_mask.shape[1] != i_it_py.shape[1]:
                point_mask = None
            else:
                point_mask = point_mask.to(device=i_it_py.device, dtype=i_it_py.dtype)

        # 3. 通过去噪器预测噪声 (使用 denoiser_type 而非 isinstance)
        if self.denoiser_type.startswith('dit'):
            # Only pass point_mask for V3.10+ which supports it
            if point_mask is not None and getattr(self.denoiser, 'supports_point_mask', False):
                eps_pred, L = self.denoiser(cnn_feature, gcn_feat, x_t, t, adj, polys=i_it_py, py_ind=py_ind, point_mask=point_mask)
            else:
                eps_pred, L = self.denoiser(cnn_feature, gcn_feat, x_t, t, adj, polys=i_it_py, py_ind=py_ind)
        else:  # snake
            eps_pred, L = self.denoiser(gcn_feat, c_it_py, x_t, t, adj, polys=i_it_py)

        return eps_pred, L

    @staticmethod
    def fourier_smooth(disp, k):
        """Fourier low-pass filter on contour displacement.
        Args:
            disp: (B, N_points, 2) displacement vectors
            k: keep lowest k frequency components per side (total 2k+1 kept)
        Returns:
            smoothed displacement (B, N_points, 2)
        """
        B, N, C = disp.shape
        freq = torch.fft.rfft(disp, dim=1)  # (B, N//2+1, 2)
        mask = torch.zeros(freq.shape[1], device=disp.device, dtype=torch.bool)
        mask[:k + 1] = True
        freq[:, ~mask, :] = 0
        return torch.fft.irfft(freq, n=N, dim=1)

    @staticmethod
    def _fill_outlier_points(disp: torch.Tensor, outlier_mask: torch.Tensor, fill_iters: int = 2) -> torch.Tensor:
        """Replace sharp outlier points by circular interpolation from valid neighbors."""
        if fill_iters <= 0 or not outlier_mask.any():
            return disp

        corrected = disp.clone()
        B, N, _ = disp.shape

        for _ in range(fill_iters):
            for b in range(B):
                mask = outlier_mask[b]
                if not mask.any():
                    continue
                if int((~mask).sum().item()) < 2:
                    continue

                outlier_idx = torch.nonzero(mask, as_tuple=False).flatten().tolist()
                for idx in outlier_idx:
                    left = idx
                    right = idx

                    found_left = False
                    for _step in range(N):
                        left = (left - 1) % N
                        if not mask[left]:
                            found_left = True
                            break

                    found_right = False
                    for _step in range(N):
                        right = (right + 1) % N
                        if not mask[right]:
                            found_right = True
                            break

                    if not (found_left and found_right):
                        continue

                    left_dist = (idx - left) % N
                    right_dist = (right - idx) % N
                    denom = left_dist + right_dist
                    if denom <= 0:
                        continue

                    left_w = right_dist / denom
                    right_w = left_dist / denom
                    corrected[b, idx] = corrected[b, left] * left_w + corrected[b, right] * right_w

        return corrected

    @staticmethod
    def _frequency_weights(num_freq: int, k: int, low_gain: float, device, dtype) -> torch.Tensor:
        """Smooth weighting that keeps low frequencies and suppresses high ones."""
        weights = torch.ones(num_freq, device=device, dtype=dtype)
        if k <= 0:
            return weights.view(1, -1, 1)

        cutoff = min(max(int(k), 1), num_freq - 1)
        idx = torch.arange(num_freq, device=device, dtype=dtype)

        # Gentle boost for low frequencies, DC kept unchanged.
        if cutoff >= 1:
            low_idx = idx[:cutoff + 1]
            low_scale = 1.0 + float(low_gain) * (1.0 - low_idx / float(cutoff + 1))
            weights[:cutoff + 1] = low_scale
            weights[0] = 1.0

        # Smooth Gaussian falloff for high frequencies.
        if cutoff + 1 < num_freq:
            sigma = max(float(cutoff) * 0.75, 1.0)
            tail = idx[cutoff + 1:] - float(cutoff)
            weights[cutoff + 1:] = torch.exp(-((tail / sigma) ** 2)).clamp_min(0.05)

        return weights.view(1, -1, 1)

    @classmethod
    def hybrid_postprocess(
        cls,
        disp: torch.Tensor,
        k: int,
        low_gain: float = 0.15,
        outlier_z: float = 2.5,
        neighbor_span: int = 1,
        fill_iters: int = 2,
        blend: float = 0.85,
    ) -> torch.Tensor:
        """Hybrid cleanup: outlier interpolation + tapered low-frequency emphasis."""
        if k <= 0:
            return disp

        if disp.numel() == 0:
            return disp

        center = disp
        if neighbor_span > 1:
            acc = center.clone()
            count = 1.0
            for shift in range(1, neighbor_span + 1):
                acc = acc + torch.roll(center, shifts=shift, dims=1) + torch.roll(center, shifts=-shift, dims=1)
                count += 2.0
            center = acc / count

        left = torch.roll(center, shifts=1, dims=1)
        right = torch.roll(center, shifts=-1, dims=1)
        linear = 0.5 * (left + right)
        dev = torch.norm(center - linear, dim=-1)
        curv = torch.norm(left - 2.0 * center + right, dim=-1)
        score = dev + 0.5 * curv

        med = score.median(dim=1, keepdim=True).values
        mad = (score - med).abs().median(dim=1, keepdim=True).values.clamp_min(1e-6)
        z_score = (score - med) / (1.4826 * mad)
        outlier_mask = z_score > outlier_z

        corrected = cls._fill_outlier_points(center, outlier_mask, fill_iters=fill_iters)

        freq = torch.fft.rfft(corrected, dim=1)
        weights = cls._frequency_weights(
            num_freq=freq.shape[1],
            k=k,
            low_gain=low_gain,
            device=disp.device,
            dtype=corrected.dtype,
        )
        filtered = torch.fft.irfft(freq * weights, n=disp.shape[1], dim=1)
        blend = float(max(0.0, min(1.0, blend)))
        return blend * filtered + (1.0 - blend) * corrected

    def _apply_postprocess(self, disp: torch.Tensor) -> torch.Tensor:
        hybrid_k = int(getattr(global_cfg, 'hybrid_postprocess_k', 0))
        if hybrid_k > 0:
            return self.hybrid_postprocess(
                disp,
                k=hybrid_k,
                low_gain=float(getattr(global_cfg, 'hybrid_postprocess_low_gain', 0.15)),
                outlier_z=float(getattr(global_cfg, 'hybrid_postprocess_outlier_z', 2.5)),
                neighbor_span=int(getattr(global_cfg, 'hybrid_postprocess_neighbor_span', 1)),
                fill_iters=int(getattr(global_cfg, 'hybrid_postprocess_fill_iters', 2)),
                blend=float(getattr(global_cfg, 'hybrid_postprocess_blend', 0.85)),
            )

        smooth_k = int(getattr(global_cfg, 'fourier_smooth_k', 0))
        if smooth_k > 0:
            return self.fourier_smooth(disp, smooth_k)
        return disp

    def disp_to_fourier(self, disp):
        """Convert (N, 128, 2) displacement to (N, K, 4) Fourier coefficients.
        The 4 channels are: [real_x, imag_x, real_y, imag_y].
        """
        K = self.fourier_k
        # FFT along point dimension for each coordinate
        freq_x = torch.fft.rfft(disp[..., 0], dim=1)  # (N, 65) complex
        freq_y = torch.fft.rfft(disp[..., 1], dim=1)  # (N, 65) complex
        # Take first K coefficients
        freq_x = freq_x[:, :K]  # (N, K) complex
        freq_y = freq_y[:, :K]  # (N, K) complex
        # Stack as (N, K, 4): [real_x, imag_x, real_y, imag_y]
        return torch.stack([freq_x.real, freq_x.imag, freq_y.real, freq_y.imag], dim=-1)

    def fourier_to_disp(self, fourier_coeff, num_points=128):
        """Convert (N, K, 4) Fourier coefficients back to (N, 128, 2) displacement."""
        K = fourier_coeff.shape[1]
        n_freq = num_points // 2 + 1  # 65 for 128 points
        device = fourier_coeff.device
        dtype = fourier_coeff.dtype
        N = fourier_coeff.shape[0]
        # Reconstruct complex coefficients, zero-pad high frequencies
        full_x = torch.zeros(N, n_freq, device=device, dtype=torch.complex64)
        full_y = torch.zeros(N, n_freq, device=device, dtype=torch.complex64)
        real_x, imag_x, real_y, imag_y = fourier_coeff.unbind(dim=-1)
        full_x[:, :K] = torch.complex(real_x.float(), imag_x.float())
        full_y[:, :K] = torch.complex(real_y.float(), imag_y.float())
        # IFFT
        disp_x = torch.fft.irfft(full_x, n=num_points, dim=1)  # (N, 128)
        disp_y = torch.fft.irfft(full_y, n=num_points, dim=1)  # (N, 128)
        return torch.stack([disp_x, disp_y], dim=-1).to(dtype)

    @torch.no_grad()
    def sample_disp_fourier(self, cnn_feature, i_it_py, c_it_py, py_ind, steps: int = 50):
        """DDIM sampling in Fourier space for V3.5."""
        N, P, _ = i_it_py.shape
        device = i_it_py.device
        K = self.fourier_k
        if N == 0:
            return torch.zeros((0, P, 2), device=device, dtype=i_it_py.dtype)
        # Start from Gaussian noise in Fourier space (N, K, 4)
        x = torch.randn(N, K, 4, device=device)
        self.scheduler.set_timesteps(steps, device=device)
        for t in self.scheduler.timesteps:
            t_batch = torch.full((N,), t, device=device, dtype=torch.long)
            eps_pred, _ = self.predict_eps(cnn_feature, i_it_py, c_it_py, py_ind, x, t_batch, batch=None)
            out = self.scheduler.step(model_output=eps_pred, timestep=t, sample=x)
            x = out.prev_sample
        # Convert Fourier coefficients back to displacement
        x0_fourier = self.denormalize_disp_fourier(x)
        return self.fourier_to_disp(x0_fourier, num_points=P)

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
            eps_pred, _ = self.predict_eps(cnn_feature, i_it_py, c_it_py, py_ind, x, t_batch, batch=None)
            out = self.scheduler.step(model_output=eps_pred, timestep=t, sample=x)
            x = out.prev_sample

        return self.denormalize_disp(x)

    @torch.no_grad()
    def sample_disp_iterative(self, cnn_feature, i_it_py, c_it_py, py_ind,
                              num_iter_steps=3, fractions=None, ddim_steps=20):
        """V3.4 multi-step iterative DDIM sampling.

        At each step, predicts displacement from the current contour position,
        applies a fraction of it, and re-samples CNN features at the updated position.
        """
        if fractions is None:
            fractions = [1.0 / (num_iter_steps - i) for i in range(num_iter_steps)]
        N, P, _ = i_it_py.shape
        device = i_it_py.device
        if N == 0:
            return torch.zeros((0, P, 2), device=device, dtype=i_it_py.dtype)

        current_contour = i_it_py.clone()
        total_disp = torch.zeros(N, P, 2, device=device, dtype=i_it_py.dtype)

        for step_idx in range(num_iter_steps):
            disp = self.sample_disp(cnn_feature, current_contour, c_it_py, py_ind, steps=ddim_steps)
            frac = fractions[step_idx]
            applied_disp = disp * frac
            current_contour = current_contour + applied_disp
            total_disp = total_disp + applied_disp

        return total_disp

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
                point_mask_train = init.get('point_mask', None)
                if point_mask_train is not None:
                    point_mask_train = point_mask_train.to(device)

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

                # --- V3.4: Multi-step iterative training (random starting point) ---
                if self.use_iterative_refinement:
                    iter_steps = getattr(global_cfg, 'iterative_num_steps', 3)
                    full_disp = x0.clone()
                    B = x0.size(0)
                    situations = torch.randint(0, iter_steps, (B,), device=device)
                    for sit in range(1, iter_steps):
                        mask = (situations == sit)
                        if mask.any():
                            frac = sit / iter_steps
                            i_init_train_py[mask] = i_init_train_py[mask] + full_disp[mask] * frac
                            x0[mask] = full_disp[mask] * (1.0 - frac)

                # --- V3.5: Convert to Fourier space ---
                if self.use_fourier_diffusion:
                    x0_fourier = self.disp_to_fourier(x0)  # (N, K, 4)
                    x0_combined = self.normalize_disp_fourier(x0_fourier)
                else:
                    x0_combined = self.normalize_disp(x0)

                # 不再构建 B/C 对齐与目标

                N = x0_combined.size(0)
                if N == 0:
                    ret.update({'diff_loss': (cnn_feature.sum() * 0.0), 'py_pred': [i_init_train_py]})
                    return ret

                # 仅保留 A1：起始轮廓与完整目标位移
                contours_combined = i_init_train_py
                c_combined = c_init_train_py
                py_ind_combined = py_ind

                N3 = x0_combined.size(0)
                t = torch.randint(0, self.T, (N3,), device=device, dtype=torch.long)
                noise = torch.randn_like(x0_combined)
                x_t = self._add_noise(x0_combined, noise, t)

            eps_pred, _ = self.predict_eps(
                cnn_feature,
                contours_combined,
                c_combined,
                py_ind_combined,
                x_t,
                t,
                batch=batch,
                point_mask=point_mask_train,
            )
            N_orig = i_init_train_py.size(0)
            eps_pred_A1 = eps_pred[0 * N_orig:1 * N_orig]
            noise_A1 = noise[0 * N_orig:1 * N_orig]

            if eps_pred_A1.numel() > 0:
                mask_A1 = point_mask_train
                if mask_A1 is not None and mask_A1.shape[:2] == eps_pred_A1.shape[:2]:
                    mask_A1 = mask_A1.to(dtype=eps_pred_A1.dtype).unsqueeze(-1)
                    diff_sq = (eps_pred_A1 - noise_A1).pow(2) * mask_A1
                    denom = (mask_A1.sum() * eps_pred_A1.size(-1)).clamp(min=1.0)
                    lossA1 = diff_sq.sum() / denom
                else:
                    lossA1 = F.mse_loss(eps_pred_A1, noise_A1, reduction='mean')
            else:
                lossA1 = (cnn_feature.sum() * 0.0)
            diff_loss = lossA1

            # Compute predicted contours for smoothness loss (V3.3)
            # Predict x0 from eps_pred and x_t
            x0_pred = self._predict_x0_from_eps(x_t[0 * N_orig:1 * N_orig], eps_pred_A1, t[:N_orig])
            if self.use_fourier_diffusion:
                x0_pred_fourier = self.denormalize_disp_fourier(x0_pred)
                x0_pred_denorm = self.fourier_to_disp(x0_pred_fourier)
            else:
                x0_pred_denorm = self.denormalize_disp(x0_pred)
            pred_contours = i_init_train_py + x0_pred_denorm

            ret.update({
                'diff_loss': diff_loss,
                'diff_lossA1': lossA1,
                'diff_loss_total': diff_loss,
                'diff_loss1': lossA1,
                'point_mask': point_mask_train,
                'py_pred': [i_init_train_py],
                'py': pred_contours,
                'pred_contours': pred_contours,  # NEW for smoothness loss
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
                    ret.update({'disp': disp, 'py': i_it_py, 'py_pred': [i_it_py], 'pred_contours': i_it_py})
                elif self.use_fourier_diffusion:
                    # V3.5: DDIM sampling in Fourier space
                    disp = self.sample_disp_fourier(
                        cnn_feature, i_it_py, c_it_py, py_ind, steps=50
                    )
                    final_py = i_it_py + disp
                    ret.update({'disp': disp, 'py': final_py, 'py_pred': [final_py], 'pred_contours': final_py})
                elif self.use_iterative_refinement:
                    iter_steps = getattr(global_cfg, 'iterative_num_steps', 3)
                    fractions = list(getattr(global_cfg, 'iterative_fractions', []))
                    if not fractions:
                        fractions = [1.0 / (iter_steps - i) for i in range(iter_steps)]
                    ddim_steps = getattr(global_cfg, 'iterative_ddim_steps', 20)
                    disp = self.sample_disp_iterative(
                        cnn_feature, i_it_py, c_it_py, py_ind,
                        num_iter_steps=iter_steps, fractions=fractions, ddim_steps=ddim_steps,
                    )
                    disp = self._apply_postprocess(disp)
                    final_py = i_it_py + disp
                    ret.update({'disp': disp, 'py': final_py, 'py_pred': [final_py], 'pred_contours': final_py})
                else:
                    disp = self.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind, steps=50)
                    disp = self._apply_postprocess(disp)
                    final_py = i_it_py + disp
                    ret.update({
                        'disp': disp,
                        'py': final_py,
                        'py_pred': [final_py],
                        'pred_contours': final_py
                    })
                
        return ret
