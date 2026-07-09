"""RL V5: three-step geometric action policy.

The default action policy is the original low-frequency normal-direction
geometric action.  The optional ``df_inner_step`` policy keeps the V5 reward,
gating, evaluation, and visualization machinery, but uses Flow-GRPO-style
inner diffusion/flow transitions as PPO actions.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--cfg_file', default='', type=str)
_pre_args, _remaining_argv = _pre_parser.parse_known_args()
if _pre_args.cfg_file:
    os.environ['CFG_FILE'] = _pre_args.cfg_file
sys.argv = [sys.argv[0]]
if _pre_args.cfg_file:
    sys.argv += ['--cfg_file', _pre_args.cfg_file]
sys.argv += _remaining_argv

import numpy as np
import torch
import torch.nn as nn
import cv2

from lib.config import cfg, args
from lib.datasets import make_data_loader
from lib.datasets.collate_batch import make_collator
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.networks import make_network
from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
from lib.train import make_optimizer
from lib.train.grpo_v2_utils import EMA, freeze_bn_running_stats, freeze_ref_flow, percentiles
from lib.train.rewards.curvature_detail_reward import compute_curvature_detail_score
from lib.train.rewards.region_reward import compute_region_score, compute_nsd_score
from lib.train.rewards.region_reward import _poly_to_mask_np, _extract_contour
from lib.train.trainers import make_trainer
from lib.utils.snake import snake_config, snake_gcn_utils


_THIS_DIR = Path(__file__).resolve().parent


def _cfg_file_used() -> str:
    return str(getattr(args, 'cfg_file', '') or os.environ.get('CFG_FILE', '')).strip()


def _cfg_stem() -> str:
    p = _cfg_file_used()
    if p:
        return Path(p).stem
    md = str(getattr(cfg, 'model_dir', '') or '').strip()
    return Path(md).name if md else 'rl_v4_three_iter'


def _project_path(p) -> Path:
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else _THIS_DIR / p


def _output_dir() -> Path:
    env_md = str(os.environ.get('RL_V4_MODEL_DIR', '') or '').strip()
    if env_md:
        return _project_path(env_md)
    md = str(getattr(cfg, 'model_dir', '') or '').strip()
    return _project_path(md) if md else _THIS_DIR / 'data' / 'outputs' / _cfg_stem()


def _resolve_checkpoint_path() -> Optional[Path]:
    candidates = []
    env_ckpt = os.environ.get('CKPT_PATH', '').strip()
    if env_ckpt:
        candidates.append(Path(env_ckpt))
    arg_ckpt = str(getattr(args, 'ckpt', '') or '').strip()
    if arg_ckpt:
        candidates.append(Path(arg_ckpt))
    resume_path = str(getattr(cfg, 'resume_path', '') or '').strip()
    if resume_path:
        candidates.append(_project_path(resume_path))
    for c in candidates:
        if c.suffix in ('.pt', '.pth') and c.exists():
            return c
    return None


def _safe_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    if tmp.exists():
        tmp.unlink()
    torch.save(obj, str(tmp))
    os.replace(str(tmp), str(path))


def _extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    for k in ('state_dict', 'model', 'net', 'network'):
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    return ckpt


def _adapt_state_dict(model, sd):
    if not isinstance(sd, dict):
        return sd
    sd = remap_legacy_state_dict(sd)
    if all(k.startswith('module.') for k in sd):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    needs_net = any(k.startswith('net.') for k in model.state_dict())
    has_net = any(k.startswith('net.') for k in sd)
    if needs_net and not has_net:
        sd = {f'net.{k}': v for k, v in sd.items()}
    return sd


def _set_seed(seed: int):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: Dict, device='cuda'):
    for k in list(batch.keys()):
        if k == 'meta':
            continue
        v = batch[k]
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
    return batch


def _flatten_valid_polys(batch: Dict, key: str, device=None) -> torch.Tensor:
    polys = batch.get(key)
    if not isinstance(polys, torch.Tensor):
        raise KeyError(f'batch missing tensor {key}')
    if polys.dim() == 4:
        if 'ct_01' in batch and isinstance(batch['ct_01'], torch.Tensor):
            polys = polys[batch['ct_01'].bool()]
        else:
            polys = polys.view(-1, polys.size(-2), polys.size(-1))
    elif polys.dim() != 3:
        raise ValueError(f'{key} must be 3D or 4D, got {tuple(polys.shape)}')
    return polys.to(device=device) if device is not None else polys


def _make_py_ind(batch: Dict, n_contours: int, device) -> torch.Tensor:
    if 'ct_01' in batch and isinstance(batch['ct_01'], torch.Tensor) and batch['ct_01'].dim() == 2:
        valid = batch['ct_01'].bool()
        inds = [
            torch.full((int(valid[bi].sum().item()),), bi, dtype=torch.long, device=device)
            for bi in range(valid.size(0))
        ]
        if inds:
            return torch.cat(inds, dim=0)
    return torch.zeros((n_contours,), dtype=torch.long, device=device)


def _signed_area(poly: torch.Tensor) -> torch.Tensor:
    x = poly[..., 0]
    y = poly[..., 1]
    x1 = torch.roll(x, shifts=-1, dims=1)
    y1 = torch.roll(y, shifts=-1, dims=1)
    return 0.5 * torch.sum(x * y1 - x1 * y, dim=1)


def _align_gt(i_init: torch.Tensor, i_gt: torch.Tensor) -> torch.Tensor:
    if i_init.size(0) == 0:
        return i_gt
    i_gt = i_gt.clone()
    mis = ((_signed_area(i_init) >= 0) ^ (_signed_area(i_gt) >= 0))
    if mis.any():
        i_gt[mis] = torch.flip(i_gt[mis], dims=[1])
    d2 = (i_init[:, :1, :] - i_gt).pow(2).sum(-1)
    nearest = torch.argmin(d2, dim=1)
    rolled = [torch.roll(i_gt[i], shifts=-int(nearest[i].item()), dims=0) for i in range(i_gt.size(0))]
    return torch.stack(rolled, dim=0)


def _contour_laplacian_px(poly: torch.Tensor, coord_scale: float) -> torch.Tensor:
    poly_px = poly * float(coord_scale)
    return torch.roll(poly_px, 1, dims=1) - 2.0 * poly_px + torch.roll(poly_px, -1, dims=1)


def _burr_penalty(final_poly: torch.Tensor, init_poly: torch.Tensor, gt_poly: torch.Tensor,
                  coord_scale: float, margin_px: float, max_px: float,
                  quantile: float = 95.0) -> tuple[torch.Tensor, torch.Tensor]:
    lap_final = _contour_laplacian_px(final_poly, coord_scale).norm(dim=-1)
    lap_gt = _contour_laplacian_px(gt_poly, coord_scale).norm(dim=-1)
    excess = torch.relu(lap_final - lap_gt - float(margin_px))
    q = min(max(float(quantile) / 100.0, 0.0), 1.0)
    raw = torch.quantile(excess, q=q, dim=1)
    penalty = torch.clamp(raw / max(float(max_px), 1e-6), min=0.0, max=2.0)
    return penalty, raw


def _as_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    return bool(v)


def _parse_env_value(raw: str, default):
    raw = str(raw).strip()
    if isinstance(default, bool):
        return raw.lower() in ('1', 'true', 'yes', 'y', 'on')
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, (tuple, list)):
        parts = raw.replace('[', '').replace(']', '').split(',')
        vals = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            try:
                vals.append(int(p))
            except ValueError:
                try:
                    vals.append(float(p))
                except ValueError:
                    vals.append(p)
        return type(default)(vals)
    return raw


def _set_requires_grad(module, flag: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = flag


def _z_logprob(z: torch.Tensor) -> torch.Tensor:
    lp = -0.5 * z.pow(2) - 0.5 * math.log(2.0 * math.pi)
    return lp.mean(dim=tuple(range(1, lp.ndim)))


def _contour_normals(poly: torch.Tensor) -> torch.Tensor:
    tangent = torch.roll(poly, shifts=-1, dims=1) - torch.roll(poly, shifts=1, dims=1)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    return normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _circular_smooth_1d(x: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    """
    1-D circular (wrap-around) Gaussian smoothing along the contour dimension (dim=1).
    x: (B, N)  →  returns (B, N) with same mean but smooth spatial structure.
    """
    import torch.nn.functional as F
    k = max(int(kernel_size) | 1, 3)  # ensure odd, min 3
    sigma_k = k / 6.0
    idx = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    kernel = torch.exp(-idx ** 2 / (2.0 * sigma_k ** 2))
    kernel = kernel / kernel.sum()
    pad = k // 2
    # circular padding: tile the contour at both ends
    x_pad = torch.cat([x[:, -pad:], x, x[:, :pad]], dim=1)  # (B, N+2*pad)
    x_pad = x_pad.unsqueeze(1)                               # (B, 1, N+2*pad)
    w = kernel.view(1, 1, -1)                                # (1, 1, k)
    smoothed = F.conv1d(x_pad, w, padding=0).squeeze(1)     # (B, N)
    return smoothed


def _lowfreq_basis(n_points: int, n_modes: int, device, dtype, damp_highfreq: bool = False) -> torch.Tensor:
    n_modes = max(int(n_modes), 1)
    theta = torch.linspace(0.0, 2.0 * math.pi, n_points + 1, device=device, dtype=dtype)[:-1]
    cols = [torch.ones_like(theta) / math.sqrt(float(n_points))]
    freqs = [0]
    freq = 1
    while len(cols) < n_modes:
        cols.append(math.sqrt(2.0 / float(n_points)) * torch.cos(float(freq) * theta))
        freqs.append(freq)
        if len(cols) >= n_modes:
            break
        cols.append(math.sqrt(2.0 / float(n_points)) * torch.sin(float(freq) * theta))
        freqs.append(freq)
        freq += 1
    basis = torch.stack(cols[:n_modes], dim=1)
    if damp_highfreq:
        decay = basis.new_tensor([1.0 / (1.0 + 0.5 * float(k)) for k in freqs[:n_modes]])
        basis = basis * decay.view(1, -1)
    return basis


def _fourier_band_basis(n_points: int, k_min: int, k_max: int, device, dtype) -> torch.Tensor:
    k_min = max(int(k_min), 1)
    k_max = max(int(k_max), k_min)
    theta = torch.linspace(0.0, 2.0 * math.pi, n_points + 1, device=device, dtype=dtype)[:-1]
    cols = []
    for freq in range(k_min, k_max + 1):
        cols.append(math.sqrt(2.0 / float(n_points)) * torch.cos(float(freq) * theta))
        cols.append(math.sqrt(2.0 / float(n_points)) * torch.sin(float(freq) * theta))
    return torch.stack(cols, dim=1)


def _poly_explorer_features(poly: torch.Tensor, frac: float, coord_scale: float = 128.0) -> torch.Tensor:
    coord_scale = max(float(coord_scale), 1.0)
    p = poly / coord_scale
    mean = p.mean(dim=1)
    std = p.std(dim=1, unbiased=False)
    min_xy = p.amin(dim=1)
    max_xy = p.amax(dim=1)
    size = max_xy - min_xy
    centered = p - mean[:, None, :]
    radius = centered.norm(dim=-1)
    radius_mean = radius.mean(dim=1, keepdim=True)
    radius_std = radius.std(dim=1, unbiased=False, keepdim=True)
    x, y = p[..., 0], p[..., 1]
    area = 0.5 * (
        x * torch.roll(y, shifts=-1, dims=1) - y * torch.roll(x, shifts=-1, dims=1)
    ).sum(dim=1, keepdim=True).abs()
    frac_t = torch.full((poly.size(0), 1), float(frac), device=poly.device, dtype=poly.dtype)
    return torch.cat([mean, std, min_xy, max_xy, size, radius_mean, radius_std, area, frac_t], dim=1)


class FourierExplorer(nn.Module):
    def __init__(self, low_modes: int, detail_modes: int, hidden_dim: int = 64,
                 mu_max: float = 0.50, logstd_min: float = -1.50, logstd_max: float = 0.70,
                 feature_dim: int = 64, feature_embed_dim: int = 32):
        super().__init__()
        self.low_modes = int(low_modes)
        self.detail_modes = int(detail_modes)
        self.mu_max = float(mu_max)
        self.logstd_min = float(logstd_min)
        self.logstd_max = float(logstd_max)
        feature_dim = max(int(feature_dim), 1)
        feature_embed_dim = max(int(feature_embed_dim), 4)
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_embed_dim),
            nn.SiLU(),
        )
        out_dim = 2 * (self.low_modes + self.detail_modes)
        self.net = nn.Sequential(
            nn.Linear(14 + feature_embed_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, poly: torch.Tensor, sampled_feat: torch.Tensor, frac: float):
        geom_feat = _poly_explorer_features(poly, frac, coord_scale=128.0)
        pooled_feat = sampled_feat.mean(dim=-1).to(device=poly.device, dtype=poly.dtype)
        img_feat = self.feature_proj(pooled_feat)
        feat = torch.cat([geom_feat, img_feat], dim=1)
        out = self.net(feat)
        n_low = self.low_modes
        n_detail = self.detail_modes
        low_mu_raw = out[:, :n_low]
        low_logstd = out[:, n_low:2 * n_low]
        off = 2 * n_low
        detail_mu_raw = out[:, off:off + n_detail]
        detail_logstd = out[:, off + n_detail:off + 2 * n_detail]
        low_mu = torch.tanh(low_mu_raw) * self.mu_max
        detail_mu = torch.tanh(detail_mu_raw) * self.mu_max
        low_logstd = low_logstd.clamp(self.logstd_min, self.logstd_max)
        detail_logstd = detail_logstd.clamp(self.logstd_min, self.logstd_max)
        return low_mu, low_logstd, detail_mu, detail_logstd


class FMVelocityPolicy(nn.Module):
    """
    Per-outer-step scalar policy aligned with FM velocity direction.
    Outputs (mu, logstd) for the scale factor applied to FM velocity direction.
    The exploration action = FM_mean_direction * scale * sigma_px
    """

    def __init__(self, outer_steps: int, feature_dim: int = 64,
                 feature_embed_dim: int = 32, hidden_dim: int = 64,
                 init_logstd: float = -1.0, logstd_min: float = -4.0,
                 logstd_max: float = 1.0):
        super().__init__()
        self.outer_steps = int(outer_steps)
        self.init_logstd = float(init_logstd)
        self.logstd_min = float(logstd_min)
        self.logstd_max = float(logstd_max)
        self.step_embed = nn.Embedding(max(int(outer_steps), 1), max(int(feature_embed_dim), 4))
        in_dim = max(int(feature_dim), 1) + max(int(feature_embed_dim), 4)
        self.net = nn.Sequential(
            nn.Linear(in_dim, max(int(hidden_dim), 16)),
            nn.ReLU(),
            nn.Linear(max(int(hidden_dim), 16), 2),  # mu, raw_logstd
        )
        # init: mu≈0, logstd≈init_logstd
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, step_idx: int, poly: torch.Tensor, c_poly: torch.Tensor,
                mean: torch.Tensor, sampled_feat: torch.Tensor, frac: float):
        """
        Returns (mu, logstd) both shape (B, 1).
        """
        B = poly.size(0)
        idx = max(0, min(int(step_idx), self.step_embed.num_embeddings - 1))
        step_t = torch.full((B,), idx, device=poly.device, dtype=torch.long)
        s_emb = self.step_embed(step_t)  # (B, feature_embed_dim)
        # use mean of sampled_feat as global poly feature
        feat = (
            sampled_feat.mean(dim=1) if sampled_feat.dim() == 3 else sampled_feat
        ).to(device=poly.device, dtype=poly.dtype)  # (B, feature_dim)
        x = torch.cat([feat, s_emb.to(dtype=poly.dtype)], dim=-1)
        out = self.net(x)  # (B, 2)
        mu = out[:, :1]  # (B, 1)
        raw_logstd = out[:, 1:2]  # (B, 1)
        logstd = (raw_logstd + self.init_logstd).clamp(self.logstd_min, self.logstd_max)
        return mu, logstd


class PerPointFMVelocityPolicy(nn.Module):
    """
    Per-point FM velocity policy.
    Global branch outputs a shared (mu, logstd); per-point branch adds a small
    tanh-limited offset to mu for each contour point.  logstd remains global —
    this avoids the O(N) ratio-variance blowup that kills per-point logstd.

    forward → (mu_pp: B×N, logstd_g: B×1)
    """

    def __init__(
        self,
        outer_steps: int,
        feature_dim: int = 128,
        feature_embed_dim: int = 32,
        hidden_dim: int = 64,
        init_logstd: float = -1.0,
        logstd_min: float = -4.0,
        logstd_max: float = 1.0,
        offset_scale: float = 0.3,
    ):
        super().__init__()
        self.outer_steps = int(outer_steps)
        self.init_logstd = float(init_logstd)
        self.logstd_min = float(logstd_min)
        self.logstd_max = float(logstd_max)
        self.offset_scale = float(offset_scale)

        self.step_embed = nn.Embedding(
            max(int(outer_steps), 1), max(int(feature_embed_dim), 4)
        )
        # ── global branch: pool(sampled_feat) + step_emb → (mu_g, raw_logstd)
        global_in = max(int(feature_dim), 1) + max(int(feature_embed_dim), 4)
        self.global_net = nn.Sequential(
            nn.Linear(global_in, max(int(hidden_dim), 16)),
            nn.ReLU(),
            nn.Linear(max(int(hidden_dim), 16), 2),  # mu_g, raw_logstd
        )
        nn.init.zeros_(self.global_net[-1].weight)
        nn.init.constant_(self.global_net[-1].bias, 0.0)

        # ── per-point branch: sampled_feat_i + step_emb + vel_mag_norm → mu_offset
        # extra vel_mag_norm feature: +1 dim
        point_in = max(int(feature_dim), 1) + max(int(feature_embed_dim), 4) + 1
        self.point_net = nn.Sequential(
            nn.Linear(point_in, max(int(hidden_dim), 16)),
            nn.ReLU(),
            nn.Linear(max(int(hidden_dim), 16), 1),  # raw mu_offset per point
        )
        nn.init.zeros_(self.point_net[-1].weight)
        nn.init.zeros_(self.point_net[-1].bias)

    def forward(
        self,
        step_idx: int,
        poly: torch.Tensor,
        c_poly: torch.Tensor,
        mean_action: torch.Tensor,  # FM velocity, shape (B, N_poly, 2)
        sampled_feat: torch.Tensor,  # (B, N_samp, C) or (B, C, N_samp)
        frac: float,
    ):
        """Returns (mu_pp: B×N_poly, logstd_g: B×1)."""
        B = poly.size(0)
        N_poly = mean_action.size(1)

        # ── normalise sampled_feat to (B, N_samp, C).
        # NOTE: cnn_feature has 64 channels while N_poly=128, so "size(1) > size(2)"
        # is NOT a reliable way to detect (B, C, N) layout — use N_poly (a known
        # ground truth) to decide which axis is the point axis instead.
        if sampled_feat.dim() == 3 and sampled_feat.size(1) != N_poly and sampled_feat.size(2) == N_poly:
            sampled_feat = sampled_feat.transpose(1, 2)  # (B, C, N_poly) → (B, N_poly, C)
        sf = sampled_feat.to(device=poly.device, dtype=poly.dtype)  # (B, N_samp, C)

        idx = max(0, min(int(step_idx), self.step_embed.num_embeddings - 1))
        step_t = torch.full((B,), idx, device=poly.device, dtype=torch.long)
        s_emb = self.step_embed(step_t).to(dtype=poly.dtype)  # (B, E)

        # ── global branch: pool over N_samp
        global_feat = sf.mean(dim=1)  # (B, C)
        g_in = torch.cat([global_feat, s_emb], dim=-1)  # (B, C+E)
        g_out = self.global_net(g_in)  # (B, 2)
        mu_g = g_out[:, :1]  # (B, 1)
        raw_logstd = g_out[:, 1:2]  # (B, 1)
        logstd_g = (raw_logstd + self.init_logstd).clamp(self.logstd_min, self.logstd_max)

        # ── FM velocity magnitude feature over N_poly contour points
        vel_mag = mean_action.norm(dim=-1, keepdim=True)  # (B, N_poly, 1)
        vel_mag_norm = vel_mag / (vel_mag.mean(dim=1, keepdim=True).clamp_min(1e-6))

        # ── per-point branch: expand global_feat + step_emb to N_poly
        #    so that the output has shape (B, N_poly), matching contour layout.
        gf_exp = global_feat.unsqueeze(1).expand(-1, N_poly, -1)   # (B, N_poly, C)
        s_emb_exp = s_emb.unsqueeze(1).expand(-1, N_poly, -1)      # (B, N_poly, E)
        p_in = torch.cat([gf_exp, s_emb_exp, vel_mag_norm], dim=-1)  # (B, N_poly, C+E+1)
        mu_offset_raw = self.point_net(p_in).squeeze(-1)  # (B, N_poly)

        # tanh-limited offset keeps per-point corrections small
        mu_pp = mu_g + torch.tanh(mu_offset_raw) * self.offset_scale  # (B, N_poly)
        return mu_pp, logstd_g


def _per_point_fm_velocity_logprob(
    scale_per_point: torch.Tensor,  # (B, N) — sampled scale
    mu_per_point: torch.Tensor,     # (B, N)
    global_logstd: torch.Tensor,    # (B, 1)
) -> torch.Tensor:
    """
    Log-prob under N(mu_per_point, exp(global_logstd)^2), averaged over N.
    Returns (B,).
    """
    var = torch.exp(2.0 * global_logstd).clamp_min(1e-12)  # (B, 1)
    lp = -0.5 * (
        (scale_per_point - mu_per_point) ** 2 / var
        + 2.0 * global_logstd
        + math.log(2.0 * math.pi)
    )  # (B, N)
    return lp.mean(dim=1)  # (B,)


class PerPointFMScalePolicy(nn.Module):
    """
    Per-point multiplicative FM scale policy — tanh-bounded parameterisation.

    Sampling:
        raw_pp ~ N(mu_pp, exp(logstd_pp)^2)          # unbounded Gaussian in raw space
        scale_pp = tanh(raw_pp) * max_scale            # bounded to (-max_scale, +max_scale)
        action   = fm_velocity * (1 + scale_pp)        # multiplier in (1-A, 1+A)

    Design rationale:
      • scale=0 → pure FM prediction (correct initialisation: last-layer bias=0)
      • multiplier always in (1-A, 1+A), e.g. (0.75, 1.25) with A=0.25
      • no reversal possible; exploration is a fine-tuning of per-point amplitude
      • tanh reparameterisation yields unbiased log_prob (no truncated-Gaussian bias)
      • stored key 'fm_velocity_scales' now holds raw_pp (pre-tanh) for PPO reuse

    forward() returns (mu_pp: B×N, logstd_pp: B×1) for the raw (pre-tanh) Gaussian.
    """

    def __init__(
        self,
        outer_steps: int,
        feature_dim: int = 128,
        feature_embed_dim: int = 32,
        hidden_dim: int = 64,
        init_logstd: float = -0.5,
        logstd_min: float = -3.0,
        logstd_max: float = 2.0,
        offset_scale: float = 0.5,
        max_scale: float = 0.25,
    ):
        super().__init__()
        self.outer_steps = int(outer_steps)
        self.init_logstd = float(init_logstd)
        self.logstd_min = float(logstd_min)
        self.logstd_max = float(logstd_max)
        self.offset_scale = float(offset_scale)
        self.max_scale = float(max_scale)  # tanh bound: scale_pp ∈ (-A,+A), multiplier ∈ (1-A,1+A)

        self.step_embed = nn.Embedding(
            max(int(outer_steps), 1), max(int(feature_embed_dim), 4)
        )
        # Global branch: pool(sampled_feat) + step_emb → (mu_g, raw_logstd_g)
        global_in = max(int(feature_dim), 1) + max(int(feature_embed_dim), 4)
        self.global_net = nn.Sequential(
            nn.Linear(global_in, max(int(hidden_dim), 16)),
            nn.ReLU(),
            nn.Linear(max(int(hidden_dim), 16), 2),  # mu_g, raw_logstd_g
        )
        nn.init.zeros_(self.global_net[-1].weight)
        nn.init.constant_(self.global_net[-1].bias, 0.0)

        # Per-point branch: per-point feat + step_emb + vel_mag_norm → mu_offset
        point_in = max(int(feature_dim), 1) + max(int(feature_embed_dim), 4) + 1
        self.point_net = nn.Sequential(
            nn.Linear(point_in, max(int(hidden_dim), 16)),
            nn.ReLU(),
            nn.Linear(max(int(hidden_dim), 16), 1),  # raw mu_offset per point
        )
        nn.init.zeros_(self.point_net[-1].weight)
        nn.init.zeros_(self.point_net[-1].bias)

    def forward(
        self,
        step_idx: int,
        poly: torch.Tensor,
        c_poly: torch.Tensor,
        mean_action: torch.Tensor,  # FM velocity (B, N_poly, 2)
        sampled_feat: torch.Tensor,  # (B, N_samp, C) or (B, C, N_samp)
        frac: float,
    ):
        """Returns (mu_pp: B×N, logstd_pp: B×1 broadcast-ready)."""
        B = poly.size(0)
        N_poly = mean_action.size(1)

        # Normalise sampled_feat to (B, N_samp, C). Use N_poly (known ground truth)
        # to pick the point axis — channel count (64) can be smaller OR larger than
        # N_poly (128) depending on config, so a size-comparison heuristic is unsafe.
        if sampled_feat.dim() == 3 and sampled_feat.size(1) != N_poly and sampled_feat.size(2) == N_poly:
            sampled_feat = sampled_feat.transpose(1, 2)
        sf = sampled_feat.to(device=poly.device, dtype=poly.dtype)

        idx = max(0, min(int(step_idx), self.step_embed.num_embeddings - 1))
        step_t = torch.full((B,), idx, device=poly.device, dtype=torch.long)
        s_emb = self.step_embed(step_t).to(dtype=poly.dtype)  # (B, E)

        # Global branch
        global_feat = sf.mean(dim=1)  # (B, C)
        g_in = torch.cat([global_feat, s_emb], dim=-1)
        g_out = self.global_net(g_in)  # (B, 2)
        mu_g = g_out[:, :1]            # (B, 1)
        raw_logstd = g_out[:, 1:2]     # (B, 1)
        logstd_pp = (raw_logstd + self.init_logstd).clamp(self.logstd_min, self.logstd_max)  # (B,1)

        # FM velocity magnitude feature
        vel_mag = mean_action.norm(dim=-1, keepdim=True)       # (B, N_poly, 1)
        vel_mag_norm = vel_mag / (vel_mag.mean(dim=1, keepdim=True).clamp_min(1e-6))

        # Per-point mu offset -- use the UN-pooled per-point feature (sf) so the
        # point_net actually sees each point's local image content, instead of
        # a copy-pasted global-pooled feature (which erased all spatial signal).
        # sf is sampled at the N_poly contour-point locations (get_gcn_feature
        # on i_it_py), so N_samp == N_poly here and no re-alignment is needed.
        s_emb_exp = s_emb.unsqueeze(1).expand(-1, N_poly, -1)      # (B, N_poly, E)
        p_in = torch.cat([sf, s_emb_exp, vel_mag_norm], dim=-1)
        mu_offset_raw = self.point_net(p_in).squeeze(-1)  # (B, N_poly)

        mu_pp = mu_g + torch.tanh(mu_offset_raw) * self.offset_scale  # (B, N_poly)
        return mu_pp, logstd_pp  # (B, N), (B, 1)


def _per_point_fm_scale_logprob(
    raw_pp: torch.Tensor,     # (B, N) — PRE-TANH sample stored at rollout time
    mu_pp: torch.Tensor,      # (B, N) — current policy mean (in raw/pre-tanh space)
    logstd_pp: torch.Tensor,  # (B, 1) or (B, N) — current policy logstd
    max_scale: float = 0.25,  # tanh bound (must match policy's max_scale)
) -> torch.Tensor:
    """
    Log-prob of tanh-bounded per-point scale under the tanh-squashed Gaussian.

    Sampling:  raw_pp ~ N(mu_pp, exp(logstd_pp)^2)
               scale_pp = tanh(raw_pp) * max_scale    ← bounded, no log_prob bias

    Change-of-variables formula:
        log p(scale_pp) = log N(raw_pp; mu_pp, sigma^2)
                        - log(max_scale)
                        - log(1 - tanh^2(raw_pp))

    raw_pp is stored at rollout time (detached); mu_pp / logstd_pp come from the
    current policy and carry gradients for PPO.  Returns per-batch mean (B,).
    """
    var = torch.exp(2.0 * logstd_pp).clamp_min(1e-12)
    # Gaussian log-prob in the raw (pre-tanh) space
    lp_gauss = -0.5 * (
        (raw_pp - mu_pp) ** 2 / var
        + 2.0 * logstd_pp
        + math.log(2.0 * math.pi)
    )  # (B, N)
    # tanh Jacobian correction: log|d scale_pp / d raw_pp| = log(max_scale*(1-tanh^2))
    tanh_raw = torch.tanh(raw_pp)
    log_jacob = math.log(max_scale) + torch.log1p(-tanh_raw.pow(2).clamp(max=1.0 - 1e-6))
    lp = lp_gauss - log_jacob   # subtract log|Jacobian| for change-of-variables
    return lp.mean(dim=1)  # (B,)


def _normal_z_logprob(z: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    var = torch.exp(2.0 * logstd).clamp_min(1e-12)
    lp = -0.5 * (((z - mu) ** 2) / var + 2.0 * logstd + math.log(2.0 * math.pi))
    return lp.mean(dim=1)



def _fm_velocity_logprob(action: torch.Tensor, mean: torch.Tensor,
                         state: torch.Tensor, scale: torch.Tensor,
                         mu: torch.Tensor, logstd: torch.Tensor,
                         sigma: float) -> torch.Tensor:
    """
    action = mean + contour_normals(state) * scale * sigma
    scale is the sampled scalar per batch, shape (B, 1)
    mu / logstd are also (B, 1)
    log_prob = sum of N(scale | mu, exp(logstd)^2) over the scalar dim -> (B,)
    """
    var = torch.exp(2.0 * logstd).clamp_min(1e-12)
    lp = -0.5 * (((scale - mu) ** 2) / var + 2.0 * logstd + math.log(2.0 * math.pi))
    return lp.sum(dim=-1)  # sum over the scalar dim -> (B,)


def _geom_delta_from_z(
    poly: torch.Tensor,
    z: torch.Tensor,
    sigma: float,
    damp_highfreq: bool = False,
) -> torch.Tensor:
    basis = _lowfreq_basis(poly.size(1), z.size(1), poly.device, poly.dtype, damp_highfreq=damp_highfreq)
    scalar = torch.matmul(z, basis.t()) * float(sigma)
    return _contour_normals(poly) * scalar.unsqueeze(-1)


def _band_detail_delta_from_z(poly: torch.Tensor, z: torch.Tensor, sigma: float,
                              k_min: int, k_max: int, gate: float) -> torch.Tensor:
    if float(gate) <= 0.0 or float(sigma) <= 0.0:
        return torch.zeros_like(poly)
    basis = _fourier_band_basis(poly.size(1), k_min, k_max, poly.device, poly.dtype)
    scalar = torch.matmul(z, basis.t()) * float(sigma) * float(gate)
    return _contour_normals(poly) * scalar.unsqueeze(-1)


def _project_geom_z(
    poly: torch.Tensor,
    residual: torch.Tensor,
    sigma: float,
    n_modes: int,
    damp_highfreq: bool = False,
) -> torch.Tensor:
    basis = _lowfreq_basis(poly.size(1), n_modes, poly.device, poly.dtype, damp_highfreq=damp_highfreq)
    scalar = (residual * _contour_normals(poly)).sum(dim=-1) / max(float(sigma), 1e-6)
    z = torch.matmul(scalar, basis)
    if damp_highfreq:
        norm_sq = basis.pow(2).sum(dim=0).clamp_min(1e-12)
        z = z / norm_sq.view(1, -1)
    return z


def _project_band_detail_z(poly: torch.Tensor, residual: torch.Tensor, sigma: float,
                           k_min: int, k_max: int, gate: float) -> torch.Tensor:
    basis = _fourier_band_basis(poly.size(1), k_min, k_max, poly.device, poly.dtype)
    denom = max(float(sigma) * max(float(gate), 0.0), 1e-6)
    scalar = (residual * _contour_normals(poly)).sum(dim=-1) / denom
    return torch.matmul(scalar, basis)


def _geom_action_logprob(action: torch.Tensor, mean: torch.Tensor, state: torch.Tensor,
                         sigma: float, n_modes: int, damp_highfreq: bool = False) -> torch.Tensor:
    z = _project_geom_z(state, action.detach() - mean, sigma, n_modes, damp_highfreq=damp_highfreq)
    return _z_logprob(z)


def _band_detail_action_logprob(action: torch.Tensor, mean: torch.Tensor, state: torch.Tensor,
                                low_sigma: float, low_modes: int, detail_sigma: float,
                                detail_k_min: int, detail_k_max: int, detail_gate: float,
                                detail_logprob_weight: float,
                                damp_highfreq: bool = False) -> torch.Tensor:
    residual = action.detach() - mean
    z_low = _project_geom_z(state, residual, low_sigma, low_modes, damp_highfreq=damp_highfreq)
    lp_low = _z_logprob(z_low)
    if float(detail_gate) <= 0.0 or float(detail_sigma) <= 0.0 or float(detail_logprob_weight) <= 0.0:
        return lp_low
    z_detail = _project_band_detail_z(
        state,
        residual,
        detail_sigma,
        detail_k_min,
        detail_k_max,
        detail_gate,
    )
    lp_detail = _z_logprob(z_detail)
    return lp_low + float(detail_logprob_weight) * lp_detail


def _flow_disp_from_latent(flow, cnn_feature, i_state, c_state, py_ind, latent, steps: int) -> torch.Tensor:
    steps = max(int(steps), 1)
    ctx = flow.prepare_sampling_context(cnn_feature, i_state, py_ind)
    x = latent
    x_self_cond = torch.zeros_like(x) if getattr(flow, '_use_self_conditioning', False) else None
    dt = 1.0 / float(steps)
    for idx in range(steps):
        t_value = idx * dt
        x, _, _, _, next_self_cond = flow.step_with_logprob(
            cnn_feature,
            i_state,
            c_state,
            py_ind,
            x_t=x,
            t_value=t_value,
            step_index=idx,
            total_steps=steps,
            action_std=0.0,
            prev_sample=None,
            sampled_feat=ctx['sampled_feat'],
            detail_feat=ctx['detail_feat'],
            contour_scale=ctx['contour_scale'],
            x_self_cond=x_self_cond,
        )
        if getattr(flow, '_use_self_conditioning', False):
            x_self_cond = next_self_cond
    disp = flow.denormalize_pred_disp(x, ctx['contour_scale'])
    return flow.clamp_pred_disp(disp, i_state)


def _outer_action_mean(flow, cnn_feature, i_state, c_state, py_ind, frac: float, steps: int) -> torch.Tensor:
    latent = torch.zeros_like(i_state)
    return _flow_disp_from_latent(flow, cnn_feature, i_state, c_state, py_ind, latent, steps) * float(frac)


def main():
    print('=' * 72)
    print('[RL-V5] Three-step geometric / diffusion-inner-step action policy')
    print('=' * 72)

    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True
    cfg.use_flow_matching = True
    cfg.use_grpo = True

    v4_cfg = getattr(cfg, 'rl_v4', None)

    def cv(name, default):
        if name == 'train_steps' and 'RL_V4_STEPS' in os.environ:
            return _parse_env_value(os.environ['RL_V4_STEPS'], default)
        env_name = f'RL_V4_{name.upper()}'
        if env_name in os.environ:
            return _parse_env_value(os.environ[env_name], default)
        if v4_cfg is not None and name in v4_cfg:
            return v4_cfg[name]
        return getattr(cfg, f'rl_v4_{name}', default)

    gpu_override = os.environ.get('RL_V4_GPU', '').strip()
    if gpu_override:
        cfg.gpus = [int(gpu_override)]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_override
        print(f'[RL-V5] Override GPU -> {gpu_override}')

    train_steps = int(cv('train_steps', 300))
    rl_resume_ckpt = str(getattr(cfg, 'rl_v4_resume_ckpt', '') or '').strip()
    k_rollouts = int(cv('k', 8))
    outer_steps = int(cv('outer_steps', 3))
    fractions = [float(x) for x in list(cv('fractions', [0.3333, 0.5, 1.0]))]
    if len(fractions) < outer_steps:
        fractions = fractions + [1.0] * (outer_steps - len(fractions))
    fractions = fractions[:outer_steps]
    ode_steps = int(cv('ode_steps', getattr(cfg, 'iterative_ode_steps', getattr(cfg, 'flow_ode_steps', 10))))
    if ode_steps <= 0:
        ode_steps = int(getattr(cfg, 'flow_ode_steps', 10))
    geom_modes = int(cv('geom_lowfreq_modes', 8))
    sigma_px_cfg = cv('geom_sigma_px', [1.2, 0.8, 0.5])
    if isinstance(sigma_px_cfg, (int, float)):
        sigma_px = [float(sigma_px_cfg)] * outer_steps
    else:
        sigma_px = [float(x) for x in list(sigma_px_cfg)]
    if len(sigma_px) < outer_steps:
        sigma_px = sigma_px + [sigma_px[-1]] * (outer_steps - len(sigma_px))
    sigma_px = sigma_px[:outer_steps]
    sigma_feat = [x / max(float(snake_config.down_ratio), 1.0) for x in sigma_px]
    action_policy = str(cv('action_policy', 'geom')).strip().lower()
    detail_k_min = int(cv('detail_k_min', 9))
    detail_k_max = int(cv('detail_k_max', 24))
    detail_sigma_px_cfg = cv('detail_sigma_px', [0.0, 0.35, 0.50])
    if isinstance(detail_sigma_px_cfg, (int, float)):
        detail_sigma_px = [float(detail_sigma_px_cfg)] * outer_steps
    else:
        detail_sigma_px = [float(x) for x in list(detail_sigma_px_cfg)]
    if len(detail_sigma_px) < outer_steps:
        detail_sigma_px = detail_sigma_px + [detail_sigma_px[-1]] * (outer_steps - len(detail_sigma_px))
    detail_sigma_px = detail_sigma_px[:outer_steps]
    detail_sigma_feat = [x / max(float(snake_config.down_ratio), 1.0) for x in detail_sigma_px]
    detail_gate_cfg = cv('detail_gate', [0.0, 0.35, 0.55])
    if isinstance(detail_gate_cfg, (int, float)):
        detail_gate = [float(detail_gate_cfg)] * outer_steps
    else:
        detail_gate = [float(x) for x in list(detail_gate_cfg)]
    if len(detail_gate) < outer_steps:
        detail_gate = detail_gate + [detail_gate[-1]] * (outer_steps - len(detail_gate))
    detail_gate = [min(max(float(x), 0.0), 1.0) for x in detail_gate[:outer_steps]]
    detail_modes = _fourier_band_basis(128, detail_k_min, detail_k_max, torch.device('cpu'), torch.float32).size(1)
    detail_logprob_weight = float(cv('detail_logprob_weight', 0.35))
    prefix_distill_weight = float(cv('prefix_distill_weight', 0.0))
    prefix_distill_steps_cfg = cv('prefix_distill_steps', [])
    if isinstance(prefix_distill_steps_cfg, (int, float)):
        prefix_distill_steps = {int(prefix_distill_steps_cfg)}
    else:
        prefix_distill_steps = {int(x) for x in list(prefix_distill_steps_cfg)}
    prefix_distill_steps = {x for x in prefix_distill_steps if 0 <= x < outer_steps}
    adaptive_explorer = _as_bool(cv('adaptive_explorer', False))
    explorer_hidden_dim = int(cv('explorer_hidden_dim', 64))
    explorer_feature_dim = int(cv('explorer_feature_dim', 64))
    explorer_feature_embed_dim = int(cv('explorer_feature_embed_dim', 32))
    explorer_mu_max = float(cv('explorer_mu_max', 0.50))
    explorer_logstd_min = float(cv('explorer_logstd_min', -1.50))
    explorer_logstd_max = float(cv('explorer_logstd_max', 0.70))
    geom_lowfreq_damp_highfreq = _as_bool(cv('geom_lowfreq_damp_highfreq', False))
    df_step_mode = str(cv('df_step_mode', 'flow_grpo')).strip().lower()
    df_sde_type = str(cv('df_sde_type', 'sde')).strip().lower()
    df_noise_level = float(cv('df_noise_level', 0.05))
    df_action_std = float(cv('df_action_std', 1.0))
    df_sde_window_size = int(cv('df_sde_window_size', 0))          # 0 = all steps (old behaviour)
    df_sde_window_range_start = int(cv('df_sde_window_range_start', 0))
    df_sde_window_range_end = int(cv('df_sde_window_range_end', 0))  # 0 = use ode_steps
    ppo_inner_epochs = int(cv('ppo_inner_epochs', 2))
    ppo_clip = float(cv('ppo_clip', 0.05))
    ppo_kl_target = float(cv('ppo_kl_target', 0.002))
    kl_beta = float(cv('kl_beta', 0.01))
    adv_clip_max = float(cv('adv_clip_max', 2.0))
    # Per-step shaped reward: blend terminal-only advantage with per-step advantage.
    # 0.0 = terminal-only (original behaviour); 1.0 = pure per-step; 0.5 = equal blend.
    per_step_reward_weight = float(cv('per_step_reward_weight', 0.0))
    # How the per-step credit is measured:
    #   'seq_delta'    : q(polys[si+1]) - q(polys[si])            (mid-contour sequential delta)
    #   'full_extrap'  : q(polys[si] + (polys[si+1]-polys[si])/frac)  (full-displacement extrapolation)
    # 'full_extrap' judges each action by the endpoint contour it *points to* (frac=1),
    # putting every step on the same yardstick regardless of its frac step size.
    per_step_credit_mode = str(cv('per_step_credit_mode', 'seq_delta')).strip().lower()
    grad_clip_norm = float(cv('grad_clip_norm', 0.3))
    lr = float(cv('lr', getattr(cfg.train, 'lr', 5e-8)))
    explorer_lr = float(cv('explorer_lr', lr))
    eval_batches_n = int(cv('eval_batches', 4))
    eval_every = int(cv('eval_every', 20))
    save_every = int(cv('save_every', 50))
    log_every = int(cv('log_every', 1))
    seed = int(cv('seed', 20260525))
    freeze_yolo = _as_bool(cv('freeze_yolo', True))
    min_load_ratio = float(cv('min_load_ratio', 95.0))
    max_contours = int(cv('max_contours', 0))
    difficulty_index_path = str(cv('difficulty_index_path', '') or '').strip()
    hard_oversample_factor = float(cv('hard_oversample_factor', 4.0))
    hard_iou_thresh = float(cv('hard_iou_thresh', 0.8))
    sigma_difficulty_scale = float(cv('sigma_difficulty_scale', 0.0))

    reward_w_region = float(cv('reward_w_region', 0.30))
    reward_w_dice = float(cv('reward_w_dice', 0.10))
    reward_w_iou = float(cv('reward_w_iou', 0.25))
    reward_w_dist = float(cv('reward_w_dist', 0.35))
    reward_dist_max_px = float(cv('reward_dist_max_px', 8.0))
    reward_dist_quantile = float(cv('reward_dist_quantile', 95.0))
    reward_dist_quantile_weight = float(cv('reward_dist_quantile_weight', 0.5))
    reward_burr_weight = float(cv('reward_burr_weight', 0.0))
    reward_burr_max_px = float(cv('reward_burr_max_px', 1.5))
    reward_burr_margin_px = float(cv('reward_burr_margin_px', 0.5))
    reward_burr_quantile = float(cv('reward_burr_quantile', 95.0))
    reward_regression_weight = float(cv('reward_regression_weight', 0.0))
    reward_regression_init_thresh = float(cv('reward_regression_init_thresh', 0.8))
    reward_global_weight = float(cv('reward_global_weight', 1.0))
    reward_detail_weight = float(cv('reward_detail_weight', 0.0))
    # delta-NSD reward: use NSD(final)-NSD(det_final) instead of absolute score
    use_delta_nsd_reward = _as_bool(cv('use_delta_nsd_reward', False))
    nsd_delta_px = float(cv('nsd_delta_px', 2.0))
    reward_detail_w_corner_dist = float(cv('reward_detail_w_corner_dist', 0.35))
    reward_detail_w_curv_match = float(cv('reward_detail_w_curv_match', 0.20))
    reward_detail_w_local_biou = float(cv('reward_detail_w_local_biou', 0.10))
    reward_detail_w_burr = float(cv('reward_detail_w_burr', 0.07))
    reward_detail_w_area = float(cv('reward_detail_w_area', 0.03))
    reward_detail_corner_dist_max_px = float(cv('reward_detail_corner_dist_max_px', 6.0))
    reward_detail_corner_dist_quantile = float(cv('reward_detail_corner_dist_quantile', 95.0))
    reward_detail_corner_dist_quantile_weight = float(cv('reward_detail_corner_dist_quantile_weight', 0.7))
    reward_detail_curvature_max_px = float(cv('reward_detail_curvature_max_px', 4.0))
    reward_detail_local_band_radius_px = int(cv('reward_detail_local_band_radius_px', 2))
    reward_detail_area_max_frac = float(cv('reward_detail_area_max_frac', 0.15))
    viz_every = int(cv('viz_every', 0))

    _set_seed(seed)

    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    net_for_load = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    ckpt_path = _resolve_checkpoint_path()
    if ckpt_path is None:
        raise FileNotFoundError('No base checkpoint found. Set resume_path or CKPT_PATH.')
    raw_ckpt = torch.load(str(ckpt_path), map_location='cpu')
    sd = _adapt_state_dict(net_for_load, _extract_state_dict(raw_ckpt))
    missing, unexpected = net_for_load.load_state_dict(sd, strict=False)
    total = len(list(net_for_load.state_dict().keys()))
    load_ratio = 100.0 * (total - len(missing)) / max(total, 1)
    print(f'[RL-V5] ckpt: {ckpt_path} | load ratio: {load_ratio:.2f}% missing={len(missing)} unexpected={len(unexpected)}')
    if load_ratio < min_load_ratio:
        raise RuntimeError(f'load ratio too low: {load_ratio:.2f}%')

    inner = net_for_load.net if hasattr(net_for_load, 'net') else net_for_load
    if freeze_yolo:
        for name in ('yolo', 'cnn_proj', 'cnn_proj_p3', 'swin_snake_feature'):
            _set_requires_grad(getattr(inner, name, None), False)
        print('[RL-V5] froze detector/feature projection parameters.')
    _set_requires_grad(inner.gcn, True)
    inner.train()
    nbn = freeze_bn_running_stats(inner)
    print(f'[RL-V5] froze BN running stats on {nbn} layers.')

    explorer = None
    if adaptive_explorer:
        if action_policy not in ('band_detail', 'geom_band_detail', 'detail_band'):
            raise ValueError('adaptive_explorer currently requires action_policy=band_detail')
        explorer_device = next(net_for_load.parameters()).device
        explorer = FourierExplorer(
            low_modes=geom_modes,
            detail_modes=detail_modes,
            hidden_dim=explorer_hidden_dim,
            mu_max=explorer_mu_max,
            logstd_min=explorer_logstd_min,
            logstd_max=explorer_logstd_max,
            feature_dim=explorer_feature_dim,
            feature_embed_dim=explorer_feature_embed_dim,
        ).to(explorer_device)
        print(
            '[RL-V5] adaptive Fourier explorer enabled '
            f'(hidden={explorer_hidden_dim}, mu_max={explorer_mu_max}, '
            f'logstd=[{explorer_logstd_min}, {explorer_logstd_max}], '
            f'feature_dim={explorer_feature_dim}, feature_embed={explorer_feature_embed_dim}).'
        )

    fm_velocity_policy = None
    if action_policy in ('fm_velocity', 'per_point_fm_velocity', 'per_point_fm_scale'):
        fmv_hidden_dim = int(cv('fm_velocity_hidden_dim', 64))
        fmv_feature_dim = int(cv('fm_velocity_feature_dim', 64))
        fmv_feature_embed_dim = int(cv('fm_velocity_feature_embed_dim', 32))
        fmv_init_logstd = float(cv('fm_velocity_init_logstd', -1.0))
        fmv_logstd_min = float(cv('fm_velocity_logstd_min', -4.0))
        fmv_logstd_max = float(cv('fm_velocity_logstd_max', 1.0))
        fmv_lr = float(cv('fm_velocity_lr', cv('explorer_lr', getattr(cfg.train, 'lr', 4e-8))))
        fmv_smooth_k = int(cv('fm_velocity_smooth_k', 9))
        fmv_entropy_weight = float(cv('fm_velocity_entropy_weight', 0.0))
        _dev = next(iter(trainer.network.parameters())).device
        if action_policy == 'per_point_fm_scale':
            # Tanh-bounded per-point FM scale: multiplier in (1-A, 1+A), no reversal
            fmv_offset_scale = float(cv('fm_velocity_offset_scale', 0.5))
            fmv_max_scale = float(cv('fm_velocity_max_scale', 0.25))
            fm_velocity_policy = PerPointFMScalePolicy(
                outer_steps=outer_steps,
                feature_dim=fmv_feature_dim,
                feature_embed_dim=fmv_feature_embed_dim,
                hidden_dim=fmv_hidden_dim,
                init_logstd=fmv_init_logstd,
                logstd_min=fmv_logstd_min,
                logstd_max=fmv_logstd_max,
                offset_scale=fmv_offset_scale,
                max_scale=fmv_max_scale,
            ).to(_dev)
            print(f'[RL-V5] per_point_fm_scale_policy: outer_steps={outer_steps}, '
                  f'hidden={fmv_hidden_dim}, feature_dim={fmv_feature_dim}, '
                  f'init_logstd={fmv_init_logstd}, offset_scale={fmv_offset_scale}, '
                  f'max_scale={fmv_max_scale}, fmv_lr={fmv_lr}, '
                  f'entropy_weight={fmv_entropy_weight}')
        elif action_policy == 'per_point_fm_velocity':
            fmv_offset_scale = float(cv('fm_velocity_offset_scale', 0.3))
            fm_velocity_policy = PerPointFMVelocityPolicy(
                outer_steps=outer_steps,
                feature_dim=fmv_feature_dim,
                feature_embed_dim=fmv_feature_embed_dim,
                hidden_dim=fmv_hidden_dim,
                init_logstd=fmv_init_logstd,
                logstd_min=fmv_logstd_min,
                logstd_max=fmv_logstd_max,
                offset_scale=fmv_offset_scale,
            ).to(_dev)
            print(f'[RL-V5] per_point_fm_velocity_policy: outer_steps={outer_steps}, '
                  f'hidden={fmv_hidden_dim}, feature_dim={fmv_feature_dim}, '
                  f'offset_scale={fmv_offset_scale}, fmv_lr={fmv_lr}')
        else:
            fm_velocity_policy = FMVelocityPolicy(
                outer_steps=outer_steps,
                feature_dim=fmv_feature_dim,
                feature_embed_dim=fmv_feature_embed_dim,
                hidden_dim=fmv_hidden_dim,
                init_logstd=fmv_init_logstd,
                logstd_min=fmv_logstd_min,
                logstd_max=fmv_logstd_max,
            ).to(_dev)
            print(f'[RL-V5] fm_velocity_policy: outer_steps={outer_steps}, hidden={fmv_hidden_dim}, '
                  f'sigma_px={sigma_px}, fmv_lr={fmv_lr}')
        if isinstance(raw_ckpt, dict) and isinstance(raw_ckpt.get('fm_velocity_policy_state_dict'), dict):
            try:
                fm_velocity_policy.load_state_dict(raw_ckpt['fm_velocity_policy_state_dict'], strict=False)
                print('[RL-V5] loaded fm_velocity_policy from checkpoint.')
            except RuntimeError as e:
                print(f'[RL-V5] WARNING: fm_velocity_policy shape mismatch vs checkpoint '
                      f'(architecture changed, e.g. feature_dim), keeping freshly-initialised '
                      f'policy weights. Error: {e}')
                _fmv_policy_shape_mismatch = True

    net_trainable = [p for p in trainer.network.parameters() if p.requires_grad]
    if net_trainable:
        optimizer = make_optimizer(cfg, trainer.network)
        for group in optimizer.param_groups:
            group['lr'] = lr
    else:
        optimizer = None
    extra_param_groups = []
    if explorer is not None:
        extra_param_groups.append({
            'params': list(explorer.parameters()),
            'lr': explorer_lr,
            'weight_decay': 0.0,
        })
    if fm_velocity_policy is not None:
        extra_param_groups.append({
            'params': list(fm_velocity_policy.parameters()),
            'lr': fmv_lr,
            'weight_decay': 0.0,
        })
    if optimizer is None:
        if not extra_param_groups:
            raise RuntimeError('No trainable parameters for RL-V5.')
        optimizer = torch.optim.AdamW(extra_param_groups, lr=lr, weight_decay=0.0)
    else:
        for group in extra_param_groups:
            optimizer.add_param_group(group)
    print(f'[RL-V5] optimizer LR set to {lr}')
    if explorer is not None:
        print(f'[RL-V5] explorer LR set to {explorer_lr}')
    ref_flow = freeze_ref_flow(inner)
    print('[RL-V5] frozen reference flow snapshot created.')

    # RL-specific resume: load model weights + optimizer state from a previous RL checkpoint.
    # The ref_flow is frozen BEFORE this, so KL is measured against the original pretrained policy.
    start_step = 0
    if rl_resume_ckpt:
        rl_resume_path = _project_path(rl_resume_ckpt)
        if rl_resume_path.exists():
            _rl_ckpt = torch.load(str(rl_resume_path), map_location='cpu')
            _sd = _rl_ckpt.get('state_dict', _rl_ckpt)
            if isinstance(_sd, dict):
                net_for_load.load_state_dict(_sd, strict=True)
            if 'optimizer' in _rl_ckpt and isinstance(_rl_ckpt['optimizer'], dict):
                try:
                    optimizer.load_state_dict(_rl_ckpt['optimizer'])
                except (RuntimeError, ValueError) as e:
                    print(f'[RL-V5] WARNING: optimizer state shape mismatch vs checkpoint '
                          f'(architecture changed), skipping optimizer state restore and '
                          f'starting fresh optimizer momentum. Error: {e}')
                # optimizer.load_state_dict does not validate tensor shapes at load time
                # (the mismatch only surfaces later inside optimizer.step()). After a
                # successful load, scrub any per-param state whose momentum-buffer shape
                # no longer matches the live parameter (e.g. fm_velocity_policy resized
                # after a feature_dim fix) so step() does not crash on a stale buffer.
                _mismatch_count = 0
                for _group in optimizer.param_groups:
                    for _p in _group['params']:
                        _st = optimizer.state.get(_p)
                        if not _st:
                            continue
                        _bad = any(
                            torch.is_tensor(_v) and _v.shape != _p.shape
                            for _v in _st.values()
                        )
                        if _bad:
                            optimizer.state[_p] = {}
                            _mismatch_count += 1
                if _mismatch_count:
                    print(f'[RL-V5] WARNING: reset optimizer momentum for {_mismatch_count} '
                          f'parameter(s) with shape mismatch vs checkpoint (fresh start for those).')
            start_step = int(_rl_ckpt.get('step', 0))
            print(f'[RL-V5] RL resume: loaded {rl_resume_path}, continuing from step={start_step}')
        else:
            print(f'[RL-V5] WARNING: rl_v4_resume_ckpt not found: {rl_resume_path}, starting from step=0')

    def _make_difficulty_train_loader():
        if not difficulty_index_path:
            return None
        path = _project_path(difficulty_index_path)
        try:
            with open(path, 'r') as f:
                index_data = json.load(f)
        except Exception as e:
            print(f'[RL-V5] WARNING: failed to load difficulty index {path}: {e}. Falling back to shuffle.')
            return None

        dataset = make_dataset(cfg, cfg.train.dataset, make_transforms(cfg, True), True)
        samples = index_data.get('samples', {}) if isinstance(index_data, dict) else {}
        if len(samples) != len(dataset):
            print(
                f'[RL-V5] WARNING: difficulty index sample count mismatch '
                f'({len(samples)} vs dataset {len(dataset)}). Falling back to shuffle.'
            )
            return None

        weights = []
        mean_ious = []
        hard_count = 0
        missing_keys = []
        hard_weight = max(float(hard_oversample_factor), 0.0)
        for idx in range(len(dataset)):
            rec = samples.get(str(idx))
            if not isinstance(rec, dict) or 'mean_iou' not in rec:
                missing_keys.append(idx)
                continue
            mean_iou = float(rec.get('mean_iou', 1.0))
            mean_ious.append(mean_iou)
            is_hard = mean_iou < float(hard_iou_thresh)
            hard_count += int(is_hard)
            weights.append(hard_weight if is_hard else 1.0)

        if missing_keys:
            print(
                f'[RL-V5] WARNING: difficulty index missing keys, first={missing_keys[:5]}. '
                'Falling back to shuffle.'
            )
            return None
        if not weights or sum(weights) <= 0.0:
            print('[RL-V5] WARNING: difficulty weights are empty/non-positive. Falling back to shuffle.')
            return None

        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler,
            int(cfg.train.batch_size),
            drop_last=False,
        )
        loader_kwargs = {
            'batch_sampler': batch_sampler,
            'num_workers': int(cfg.train.num_workers),
            'collate_fn': make_collator(cfg),
            'pin_memory': True,
        }
        if int(cfg.train.num_workers) > 0:
            loader_kwargs['persistent_workers'] = bool(
                getattr(cfg, 'dataloader_persistent_workers', True)
            )
            prefetch_factor = int(getattr(cfg, 'dataloader_prefetch_factor', 0) or 0)
            if prefetch_factor > 0:
                loader_kwargs['prefetch_factor'] = prefetch_factor

        hard_frac = 100.0 * hard_count / max(len(dataset), 1)
        mean_iou_avg = float(np.mean(mean_ious)) if mean_ious else 1.0
        print(
            f'[RL-V5] difficulty sampler enabled: {hard_count}/{len(dataset)} '
            f'hard samples ({hard_frac:.2f}%), factor={hard_weight:.2f}, '
            f'thresh={hard_iou_thresh:.3f}, mean_iou={mean_iou_avg:.4f}'
        )
        return torch.utils.data.DataLoader(dataset, **loader_kwargs)

    train_loader = _make_difficulty_train_loader()
    if train_loader is None:
        train_loader = make_data_loader(cfg, is_train=True, is_distributed=False)
    eval_batches = []
    try:
        eval_loader = make_data_loader(cfg, is_train=False, is_distributed=False)
        eval_iter = iter(eval_loader)
        for _ in range(eval_batches_n):
            eb = next(eval_iter)
            _move_batch(eb)
            eval_batches.append(eb)
        print(f'[RL-V5] fixed eval set: {len(eval_batches)} batches')
    except Exception as e:
        print(f'[RL-V5] WARNING: fixed eval unavailable: {e}')

    out_dir = _output_dir()
    ckpt_dir = out_dir / 'checkpoints'
    log_dir = out_dir / 'posttrain_rl_v5_geom_action'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'logs.jsonl'
    with open(log_dir / 'v5_hparams.json', 'w') as f:
        json.dump({
            'cfg_file': _cfg_file_used(),
            'base_ckpt': str(ckpt_path),
            'train_steps': train_steps,
            'k': k_rollouts,
            'outer_steps': outer_steps,
            'fractions': fractions,
            'ode_steps': ode_steps,
            'action_policy': action_policy,
            'geom_lowfreq_modes': geom_modes,
            'geom_lowfreq_damp_highfreq': bool(geom_lowfreq_damp_highfreq),
            'geom_sigma_px': sigma_px,
            'geom_sigma_feature': sigma_feat,
            'band_detail': {
                'enabled': action_policy in ('band_detail', 'geom_band_detail', 'detail_band'),
                'k_min': detail_k_min,
                'k_max': detail_k_max,
                'modes': detail_modes,
                'sigma_px': detail_sigma_px,
                'gate': detail_gate,
                'logprob_weight': detail_logprob_weight,
            },
            'prefix_distill': {
                'weight': prefix_distill_weight,
                'steps': sorted(prefix_distill_steps),
            },
            'adaptive_explorer': {
                'enabled': bool(adaptive_explorer),
                'hidden_dim': explorer_hidden_dim,
                'feature_dim': explorer_feature_dim,
                'feature_embed_dim': explorer_feature_embed_dim,
                'mu_max': explorer_mu_max,
                'logstd_min': explorer_logstd_min,
                'logstd_max': explorer_logstd_max,
                'lr': explorer_lr,
            },
            'df_inner_step': {
                'step_mode': df_step_mode,
                'sde_type': df_sde_type,
                'noise_level': df_noise_level,
                'action_std': df_action_std,
            },
            'ppo_inner_epochs': ppo_inner_epochs,
            'ppo_clip': ppo_clip,
            'ppo_kl_target': ppo_kl_target,
            'kl_beta': kl_beta,
            'reward_weights': {
                'region': reward_w_region,
                'dice': reward_w_dice,
                'iou': reward_w_iou,
                'dist': reward_w_dist,
            },
            'burr': {
                'mode': 'gt_relative_spike_quantile',
                'weight': reward_burr_weight,
                'max_px': reward_burr_max_px,
                'margin_px': reward_burr_margin_px,
                'quantile': reward_burr_quantile,
            },
            'anti_regression': {
                'weight': reward_regression_weight,
                'init_thresh': reward_regression_init_thresh,
            },
            'difficulty_focus': {
                'index_path': difficulty_index_path,
                'hard_oversample_factor': hard_oversample_factor,
                'hard_iou_thresh': hard_iou_thresh,
                'sigma_difficulty_scale': sigma_difficulty_scale,
            },
            'curvature_detail_reward': {
                'enabled': bool(reward_detail_weight > 0),
                'global_weight': reward_global_weight,
                'detail_weight': reward_detail_weight,
                'corner_dist': {
                    'weight': reward_detail_w_corner_dist,
                    'max_px': reward_detail_corner_dist_max_px,
                    'quantile': reward_detail_corner_dist_quantile,
                    'quantile_weight': reward_detail_corner_dist_quantile_weight,
                },
                'curv_match': {
                    'weight': reward_detail_w_curv_match,
                    'max_px': reward_detail_curvature_max_px,
                },
                'local_biou': {
                    'weight': reward_detail_w_local_biou,
                    'band_radius_px': reward_detail_local_band_radius_px,
                },
                'detail_burr': {
                    'weight': reward_detail_w_burr,
                    'max_px': reward_burr_max_px,
                    'margin_px': reward_burr_margin_px,
                    'quantile': reward_burr_quantile,
                },
                'area': {
                    'weight': reward_detail_w_area,
                    'max_frac': reward_detail_area_max_frac,
                },
            },
            'viz_every': viz_every,
            'max_contours': max_contours,
        }, f, indent=2)

    gcn = inner.gcn
    ema_reward = EMA(decay=0.95)
    best_eval_iou = -1.0

    def _manual_context(batch):
        was_training = inner.training
        inner.eval()
        with torch.no_grad():
            yolo_out = inner.yolo(batch['inp'])
            feat_list = yolo_out[1] if isinstance(yolo_out, (list, tuple)) and len(yolo_out) > 1 else None
            feat_p2 = feat_list[0] if isinstance(feat_list, (list, tuple)) else yolo_out
            cnn_feature = inner.cnn_proj(feat_p2)
            if getattr(inner, 'use_p3_features', False) and hasattr(inner, 'cnn_proj_p3'):
                if isinstance(feat_list, (list, tuple)) and len(feat_list) > 1:
                    feat_p3 = feat_list[1]
                    feat_p3_up = torch.nn.functional.interpolate(
                        feat_p3, size=feat_p2.shape[-2:], mode='bilinear', align_corners=False
                    )
                    cnn_feature = cnn_feature + inner.cnn_proj_p3(feat_p3_up)
        device = cnn_feature.device
        i_init = _flatten_valid_polys(batch, 'i_it_py', device=device)
        i_gt = _flatten_valid_polys(batch, 'i_gt_py', device=device)
        try:
            c_init = _flatten_valid_polys(batch, 'c_it_py', device=device)
        except Exception:
            c_init = snake_gcn_utils.img_poly_to_can_poly(i_init)
        py_ind = _make_py_ind(batch, i_init.size(0), device=device)
        if c_init.size(0) != i_init.size(0):
            c_init = snake_gcn_utils.img_poly_to_can_poly(i_init)
        if i_gt.size(0) != i_init.size(0):
            n = min(i_init.size(0), i_gt.size(0))
            i_init, c_init, i_gt, py_ind = i_init[:n], c_init[:n], i_gt[:n], py_ind[:n]
        if max_contours > 0 and i_init.size(0) > max_contours:
            i_init = i_init[:max_contours]
            c_init = c_init[:max_contours]
            i_gt = i_gt[:max_contours]
            py_ind = py_ind[:max_contours]
        if was_training:
            inner.train()
            freeze_bn_running_stats(inner)
        return {
            'cnn_feature': cnn_feature.detach(),
            'i_it_py': i_init.detach(),
            'c_it_py': c_init.detach(),
            'i_gt_py': _align_gt(i_init, i_gt).detach(),
            'py_ind': py_ind.detach(),
            'image_hw': (int(batch['inp'].shape[-2]), int(batch['inp'].shape[-1])),
        }

    def _score_with_components(poly, gt, image_hw):
        global_score = compute_region_score(
            poly,
            gt,
            H=int(image_hw[0]),
            W=int(image_hw[1]),
            w_boundary=reward_w_region,
            w_dice=reward_w_dice,
            w_iou=reward_w_iou,
            w_dist=reward_w_dist,
            dist_max_px=reward_dist_max_px,
            dist_quantile=reward_dist_quantile,
            dist_quantile_weight=reward_dist_quantile_weight,
            coord_scale=float(snake_config.down_ratio),
        )
        comps = {
            'global_score': global_score,
        }
        if reward_regression_weight > 0:
            comps['iou'] = compute_region_score(
                poly,
                gt,
                H=int(image_hw[0]),
                W=int(image_hw[1]),
                w_boundary=0,
                w_dice=0,
                w_iou=1,
                w_dist=0,
                coord_scale=float(snake_config.down_ratio),
            )
        if reward_detail_weight <= 0:
            return reward_global_weight * global_score, comps
        detail_score, detail_comps = compute_curvature_detail_score(
            poly,
            gt,
            H=int(image_hw[0]),
            W=int(image_hw[1]),
            coord_scale=float(snake_config.down_ratio),
            corner_dist_max_px=reward_detail_corner_dist_max_px,
            corner_dist_quantile=reward_detail_corner_dist_quantile,
            corner_dist_quantile_weight=reward_detail_corner_dist_quantile_weight,
            curvature_max_px=reward_detail_curvature_max_px,
            burr_margin_px=reward_burr_margin_px,
            burr_max_px=reward_burr_max_px,
            burr_quantile=reward_burr_quantile,
            local_band_radius_px=reward_detail_local_band_radius_px,
            area_max_frac=reward_detail_area_max_frac,
            w_corner_dist=reward_detail_w_corner_dist,
            w_curv_match=reward_detail_w_curv_match,
            w_local_biou=reward_detail_w_local_biou,
            w_burr=reward_detail_w_burr,
            w_area=reward_detail_w_area,
            return_components=True,
        )
        comps['detail_score'] = detail_score
        comps.update({f'detail_{k}': v for k, v in detail_comps.items()})
        return reward_global_weight * global_score + reward_detail_weight * detail_score, comps

    def _quality_score(poly, gt, image_hw):
        score, _ = _score_with_components(poly, gt, image_hw)
        return score

    @torch.no_grad()
    def _deterministic_three_step(output, flow):
        current = output['i_it_py'].detach()
        total_disp = torch.zeros_like(current)
        polys = [current.detach()]
        for frac in fractions:
            c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
            action = _outer_action_mean(
                flow, output['cnn_feature'], current, c_cur, output['py_ind'], float(frac), ode_steps
            )
            current = (current + action).detach()
            total_disp = total_disp + action.detach()
            polys.append(current.detach())
        return {'disp': total_disp, 'py': output['i_it_py'] + total_disp, 'polys': polys}

    def _pick_sde_window(ode_steps: int, window_size: int, range_start: int, range_end: int):
        """Return (win_start, win_end) inclusive-exclusive. window_size<=0 means all steps."""
        if window_size <= 0:
            return (0, ode_steps)
        r_end = range_end if range_end > 0 else ode_steps
        r_end = min(r_end, ode_steps)
        r_start = max(range_start, 0)
        max_start = max(r_end - window_size, r_start)
        win_start = random.randint(r_start, max_start)
        return (win_start, win_start + window_size)

    @torch.no_grad()
    def _sample_rollout(output, sigma_multiplier: float = 1.0):
        current = output['i_it_py'].detach()
        total_disp = torch.zeros_like(current)
        sigma_multiplier = max(float(sigma_multiplier), 0.0)
        traj = {
            'states': [],
            'c_states': [],
            'actions': [],
            'old_logs': [],
            'fractions': [],
            'sigmas': [],
            'detail_sigmas': [],
            'outer_indices': [],
            'geom_sampled_feats': [],
            'fm_velocity_scales': [],
            'prefix_states': [],
            'prefix_c_states': [],
            'prefix_fractions': [],
            'prefix_indices': [],
            'polys': [current.detach()],
        }
        if action_policy in ('df', 'df_inner_step', 'diffusion_inner_step', 'flow_inner_step'):
            traj.update({
                'x_ts': [],
                'x_prevs': [],
                'timesteps': [],
                'step_indices': [],
                'x_self_conds': [],
                'sampled_feats': [],
                'detail_feats': [],
                'contour_scales': [],
                'total_steps': [],
            })
            sde_win = _pick_sde_window(
                max(int(ode_steps), 1),
                df_sde_window_size,
                df_sde_window_range_start,
                df_sde_window_range_end,
            )
            traj['sde_win'] = list(sde_win)
            for frac in fractions:
                c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
                ctx = gcn.prepare_sampling_context(output['cnn_feature'], current, output['py_ind'])
                x = torch.zeros_like(current)
                x_self_cond = torch.zeros_like(x) if getattr(gcn, '_use_self_conditioning', False) else None
                dt = 1.0 / float(max(int(ode_steps), 1))
                for idx in range(max(int(ode_steps), 1)):
                    t_value = idx * dt
                    cur_action_std = df_action_std if (sde_win[0] <= idx < sde_win[1]) else 0.0
                    x_prev, log_prob, _, _, next_self_cond = gcn.step_with_logprob(
                        output['cnn_feature'],
                        current,
                        c_cur,
                        output['py_ind'],
                        x_t=x,
                        t_value=t_value,
                        step_index=idx,
                        total_steps=ode_steps,
                        action_std=cur_action_std,
                        prev_sample=None,
                        sampled_feat=ctx['sampled_feat'],
                        detail_feat=ctx['detail_feat'],
                        contour_scale=ctx['contour_scale'],
                        x_self_cond=x_self_cond,
                        step_mode=df_step_mode,
                        noise_level=df_noise_level,
                        sde_type=df_sde_type,
                    )
                    traj['states'].append(current.detach())
                    traj['c_states'].append(c_cur.detach())
                    traj['actions'].append(x_prev.detach())
                    traj['old_logs'].append(log_prob.detach())
                    traj['fractions'].append(float(frac))
                    traj['sigmas'].append(float(df_noise_level))
                    traj['x_ts'].append(x.detach())
                    traj['x_prevs'].append(x_prev.detach())
                    traj['timesteps'].append(torch.tensor(t_value, device=current.device, dtype=current.dtype))
                    traj['step_indices'].append(torch.tensor(idx, device=current.device, dtype=torch.long))
                    traj['x_self_conds'].append(None if x_self_cond is None else x_self_cond.detach())
                    traj['sampled_feats'].append(ctx['sampled_feat'].detach())
                    traj['detail_feats'].append(None if ctx.get('detail_feat') is None else ctx['detail_feat'].detach())
                    traj['contour_scales'].append(ctx['contour_scale'].detach())
                    traj['total_steps'].append(torch.tensor(ode_steps, device=current.device, dtype=torch.long))
                    x = x_prev.detach()
                    if getattr(gcn, '_use_self_conditioning', False):
                        x_self_cond = next_self_cond
                disp = gcn.denormalize_pred_disp(x, ctx['contour_scale'])
                applied = gcn.clamp_pred_disp(disp, current) * float(frac)
                current = (current + applied).detach()
                total_disp = total_disp + applied.detach()
                traj['polys'].append(current.detach())
            traj['disp'] = total_disp.detach()
            traj['py'] = output['i_it_py'] + total_disp
            return traj

        for si, frac in enumerate(fractions):
            c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
            mean = _outer_action_mean(gcn, output['cnn_feature'], current, c_cur, output['py_ind'], frac, ode_steps)
            if prefix_distill_weight > 0.0 and si in prefix_distill_steps:
                traj['prefix_states'].append(current.detach())
                traj['prefix_c_states'].append(c_cur.detach())
                traj['prefix_fractions'].append(float(frac))
                traj['prefix_indices'].append(int(si))
            if fm_velocity_policy is not None and action_policy == 'fm_velocity':
                ctx = gcn.prepare_sampling_context(output['cnn_feature'], current, output['py_ind'])
                sampled_feat = ctx['sampled_feat'].detach()
                mu_s, logstd_s = fm_velocity_policy(si, current, c_cur, mean, sampled_feat, float(frac))
                scale = mu_s + torch.exp(logstd_s) * torch.randn_like(mu_s)  # (B, 1)
                sigma_s = float(sigma_feat[si]) * sigma_multiplier
                normals = _contour_normals(current)  # (B, N, 2)
                delta = normals * (scale * sigma_s).unsqueeze(-1)  # (B, N, 2)
                action = mean + delta
                old_log = _fm_velocity_logprob(action, mean, current, scale, mu_s, logstd_s, sigma_s)
                traj['states'].append(current.detach())
                traj['c_states'].append(c_cur.detach())
                traj['actions'].append(action.detach())
                traj['old_logs'].append(old_log.detach())
                traj['fractions'].append(float(frac))
                traj['sigmas'].append(sigma_s)
                traj['detail_sigmas'].append(0.0)
                traj['outer_indices'].append(int(si))
                traj['geom_sampled_feats'].append(sampled_feat)
                traj['fm_velocity_scales'].append(scale.detach())
                current = (current + action).detach()
                total_disp = total_disp + action.detach()
                traj['polys'].append(current.detach())
                continue
            elif fm_velocity_policy is not None and action_policy == 'per_point_fm_velocity':
                ctx = gcn.prepare_sampling_context(output['cnn_feature'], current, output['py_ind'])
                sampled_feat = ctx['sampled_feat'].detach()
                mu_pp, logstd_g = fm_velocity_policy(si, current, c_cur, mean, sampled_feat, float(frac))
                eps = torch.randn_like(mu_pp)  # (B, N)
                scale_raw = mu_pp + torch.exp(logstd_g) * eps  # (B, N) — raw, used for log_prob
                # Circular spatial smoothing: prevents per-point scale from alternating signs → no burr.
                # log_prob is on scale_raw (reparameterisation); smoothed scale used for action only.
                scale_smooth = _circular_smooth_1d(scale_raw, kernel_size=fmv_smooth_k)  # (B, N)
                sigma_s = float(sigma_feat[si]) * sigma_multiplier
                # Direction: per-point FM velocity direction (NOT contour normal).
                fm_dir = mean / mean.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # (B, N, 2)
                delta = fm_dir * (scale_smooth * sigma_s).unsqueeze(-1)  # (B, N, 2)
                action = mean + delta
                old_log = _per_point_fm_velocity_logprob(scale_raw, mu_pp, logstd_g)
                traj['states'].append(current.detach())
                traj['c_states'].append(c_cur.detach())
                traj['actions'].append(action.detach())
                traj['old_logs'].append(old_log.detach())
                traj['fractions'].append(float(frac))
                traj['sigmas'].append(sigma_s)
                traj['detail_sigmas'].append(0.0)
                traj['outer_indices'].append(int(si))
                traj['geom_sampled_feats'].append(sampled_feat)
                traj['fm_velocity_scales'].append(scale_raw.detach())  # (B, N) — raw for PPO recompute
                current = (current + action).detach()
                total_disp = total_disp + action.detach()
                traj['polys'].append(current.detach())
                continue
            elif fm_velocity_policy is not None and action_policy == 'per_point_fm_scale':
                # Tanh-bounded per-point FM scale (SAC-style squashed Gaussian).
                # raw_pp ~ N(mu_pp, std_pp)                → unbounded Gaussian
                # scale_pp = tanh(raw_pp) * max_scale      → bounded (-A, +A)
                # action   = fm_velocity * (1 + scale_pp)  → multiplier in (1-A, 1+A)
                # scale=0 (raw=0) → pure FM; no reversal possible; no log_prob bias.
                ctx = gcn.prepare_sampling_context(output['cnn_feature'], current, output['py_ind'])
                sampled_feat = ctx['sampled_feat'].detach()
                mu_pp, logstd_pp = fm_velocity_policy(si, current, c_cur, mean, sampled_feat, float(frac))
                eps = torch.randn_like(mu_pp)  # (B, N)
                raw_pp = mu_pp + torch.exp(logstd_pp) * eps   # (B, N) pre-tanh raw sample
                _max_s = fm_velocity_policy.max_scale
                scale_pp = torch.tanh(raw_pp) * _max_s        # (B, N) bounded scale
                # action_i = fm_vel_i * (1 + scale_i)  — scale ∈ (-A,+A) → multiplier ∈ (1-A,1+A)
                action = mean * (1.0 + scale_pp.unsqueeze(-1))  # (B, N, 2)
                old_log = _per_point_fm_scale_logprob(raw_pp, mu_pp, logstd_pp, _max_s)
                traj['states'].append(current.detach())
                traj['c_states'].append(c_cur.detach())
                traj['actions'].append(action.detach())
                traj['old_logs'].append(old_log.detach())
                traj['fractions'].append(float(frac))
                traj['sigmas'].append(0.0)  # sigma unused for this scheme
                traj['detail_sigmas'].append(0.0)
                traj['outer_indices'].append(int(si))
                traj['geom_sampled_feats'].append(sampled_feat)
                traj['fm_velocity_scales'].append(raw_pp.detach())  # (B, N) raw pre-tanh sample for PPO recompute
                current = (current + action).detach()
                total_disp = total_disp + action.detach()
                traj['polys'].append(current.detach())
                continue
            sigma = float(sigma_feat[si]) * sigma_multiplier
            low_active = sigma > 0.0
            use_explorer = explorer is not None and action_policy in ('band_detail', 'geom_band_detail', 'detail_band')
            low_mu = low_logstd = detail_mu = detail_logstd = None
            geom_sampled_feat = None
            if use_explorer:
                ctx = gcn.prepare_sampling_context(output['cnn_feature'], current, output['py_ind'])
                geom_sampled_feat = ctx['sampled_feat'].detach()
                low_mu, low_logstd, detail_mu, detail_logstd = explorer(
                    current,
                    geom_sampled_feat,
                    float(frac),
                )
            z = torch.randn((current.size(0), geom_modes), device=current.device, dtype=current.dtype)
            if low_active and low_mu is not None:
                z = low_mu + torch.exp(low_logstd) * z
            low_delta = (
                _geom_delta_from_z(current, z, sigma, damp_highfreq=geom_lowfreq_damp_highfreq)
                if low_active else torch.zeros_like(current)
            )
            if low_active:
                old_log = (
                    _normal_z_logprob(z, low_mu, low_logstd)
                    if low_mu is not None else _z_logprob(z)
                )
            else:
                old_log = None
            if action_policy in ('band_detail', 'geom_band_detail', 'detail_band'):
                d_sigma = float(detail_sigma_feat[si]) * sigma_multiplier
                d_gate = float(detail_gate[si])
                detail_active = (
                    d_sigma > 0.0 and d_gate > 0.0 and float(detail_logprob_weight) > 0.0
                )
                z_detail = torch.randn((current.size(0), detail_modes), device=current.device, dtype=current.dtype)
                if detail_active and detail_mu is not None:
                    z_detail = detail_mu + torch.exp(detail_logstd) * z_detail
                detail_delta = (
                    _band_detail_delta_from_z(
                        current,
                        z_detail,
                        d_sigma,
                        detail_k_min,
                        detail_k_max,
                        d_gate,
                    )
                    if detail_active else torch.zeros_like(current)
                )
                action = mean + low_delta + detail_delta
                if detail_active:
                    detail_log = float(detail_logprob_weight) * (
                        _normal_z_logprob(z_detail, detail_mu, detail_logstd)
                        if detail_mu is not None else _z_logprob(z_detail)
                    )
                    old_log = detail_log if old_log is None else old_log + detail_log
            else:
                d_sigma = 0.0
                detail_active = False
                action = mean + low_delta
            if low_active or detail_active:
                traj['states'].append(current.detach())
                traj['c_states'].append(c_cur.detach())
                traj['actions'].append(action.detach())
                traj['old_logs'].append(old_log.detach())
                traj['fractions'].append(float(frac))
                traj['sigmas'].append(sigma)
                traj['detail_sigmas'].append(float(d_sigma))
                traj['outer_indices'].append(int(si))
                if use_explorer:
                    traj['geom_sampled_feats'].append(geom_sampled_feat)
            current = (current + action).detach()
            total_disp = total_disp + action.detach()
            traj['polys'].append(current.detach())
        traj['disp'] = total_disp.detach()
        traj['py'] = output['i_it_py'] + total_disp
        return traj

    @torch.no_grad()
    def _compute_eval():
        if not eval_batches:
            return {}
        vals_iou, vals_dice, vals_mbf = [], [], []
        detail_eval = {}
        for eb in eval_batches:
            out = _manual_context(eb)
            if out['i_it_py'].numel() == 0:
                continue
            det = _deterministic_three_step(out, gcn)
            pred = out['i_it_py'] + det['disp']
            gt = out['i_gt_py']
            hw = out['image_hw']
            vals_iou.append(compute_region_score(pred, gt, H=hw[0], W=hw[1], w_boundary=0, w_dice=0, w_iou=1,
                                                 coord_scale=float(snake_config.down_ratio)))
            vals_dice.append(compute_region_score(pred, gt, H=hw[0], W=hw[1], w_boundary=0, w_dice=1, w_iou=0,
                                                  coord_scale=float(snake_config.down_ratio)))
            vals_mbf.append(compute_region_score(pred, gt, H=hw[0], W=hw[1], w_boundary=1, w_dice=0, w_iou=0,
                                                 coord_scale=float(snake_config.down_ratio)))
            if reward_detail_weight > 0:
                _, comps = _score_with_components(pred, gt, hw)
                for key, value in comps.items():
                    detail_eval.setdefault(key, []).append(value.detach())
        if not vals_iou:
            return {}
        iou = torch.cat(vals_iou)
        dice = torch.cat(vals_dice)
        mbf = torch.cat(vals_mbf)
        ret = {
            'eval_iou': float(iou.mean().item()),
            'eval_dice': float(dice.mean().item()),
            'eval_mboundf': float(mbf.mean().item()),
            'eval_n': int(iou.numel()),
        }
        for key, values in detail_eval.items():
            if values:
                ret[f'eval_{key}'] = float(torch.cat(values).mean().item())
        return ret

    @torch.no_grad()
    def _diag_mu_error_correlation():
        """
        Diagnostic (per_point_fm_scale only): measure whether the policy's
        per-point mu_pp correction has learned to be informative about where
        the deterministic FM rollout actually deviates from GT.

        For each outer step, on the fixed eval set:
          1. Run the deterministic 5-step FM rollout (no policy).
          2. At each outer step, query fm_velocity_policy for mu_pp
             (the per-point offset that WOULD be applied if this step were
             sampled with the current policy).
          3. Compute each point's true per-step error: the residual between
             the deterministic FM prediction and where the point "should"
             have gone in the extrapolated full-displacement sense
             (same full_extrap definition used for per-step credit).
          4. Correlate |mu_pp| (magnitude of intended correction) against
             the true residual error magnitude, per point, pooled over the
             eval set. A rising positive correlation over training steps is
             evidence the small network is learning *where* to correct, not
             just producing noise.

        Returns a dict with 'diag_corr' (Pearson r) and 'diag_n' (#points).
        Returns {} if fm_velocity_policy is None or eval_batches is empty.
        """
        if fm_velocity_policy is None or not eval_batches or action_policy != 'per_point_fm_scale':
            return {}
        mu_abs_all = []
        err_abs_all = []
        for eb in eval_batches:
            out = _manual_context(eb)
            if out['i_it_py'].numel() == 0:
                continue
            current = out['i_it_py'].detach()
            gt = out['i_gt_py']
            for si, frac in enumerate(fractions):
                c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
                mean = _outer_action_mean(
                    gcn, out['cnn_feature'], current, c_cur, out['py_ind'], float(frac), ode_steps
                )
                ctx = gcn.prepare_sampling_context(out['cnn_feature'], current, out['py_ind'])
                sampled_feat = ctx['sampled_feat'].detach()
                mu_pp, _ = fm_velocity_policy(si, current, c_cur, mean, sampled_feat, float(frac))
                nxt = (current + mean).detach()
                # Full-extrap true target for this step: where the point would
                # land if the deterministic residual (nxt - current) were the
                # complete remaining displacement, compared against GT nearest
                # point distance at nxt (per-point boundary error in px).
                gt_np = gt.detach().cpu().numpy()
                nxt_np = nxt.detach().cpu().numpy()
                for b in range(nxt_np.shape[0]):
                    diffs = nxt_np[b][:, None, :] - gt_np[b][None, :, :]
                    d = np.sqrt((diffs ** 2).sum(-1)).min(axis=1)  # (N,) nearest-GT-point dist, in feature units
                    err_abs_all.append(d * float(snake_config.down_ratio))
                mu_abs_all.append(mu_pp.detach().abs().cpu().numpy().reshape(-1))
                current = nxt
        if not mu_abs_all:
            return {}
        mu_flat = np.concatenate(mu_abs_all)
        err_flat = np.concatenate(err_abs_all)
        n = min(mu_flat.shape[0], err_flat.shape[0])
        mu_flat, err_flat = mu_flat[:n], err_flat[:n]
        if n < 2 or mu_flat.std() < 1e-9 or err_flat.std() < 1e-9:
            return {'diag_mu_err_corr': 0.0, 'diag_mu_err_n': int(n)}
        corr = float(np.corrcoef(mu_flat, err_flat)[0, 1])
        return {'diag_mu_err_corr': corr, 'diag_mu_err_n': int(n)}

    def _save_checkpoint(path: Path, step: int, metrics: Dict):
        _safe_torch_save({
            'state_dict': net_for_load.state_dict(),
            'explorer_state_dict': None if explorer is None else explorer.state_dict(),
            'fm_velocity_policy_state_dict': None if fm_velocity_policy is None else fm_velocity_policy.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': int(step),
            'metrics': metrics,
            'cfg_file': _cfg_file_used(),
            'time': datetime.datetime.now().isoformat(),
        }, path)

    @torch.no_grad()
    def _dump_group_viz(batch, output, rollouts: List[Dict], quality: torch.Tensor,
                        baseline_score: torch.Tensor, step: int,
                        deterministic_py: Optional[torch.Tensor] = None,
                        deterministic_polys: Optional[List[torch.Tensor]] = None):
        if viz_every <= 0:
            return
        viz_dir = _THIS_DIR / 'visual' / f'rl_v5_{_cfg_stem()}'
        viz_dir.mkdir(parents=True, exist_ok=True)
        inp = batch['inp'][0].detach().float().cpu().numpy()
        if inp.shape[0] in (1, 3):
            inp = inp.transpose(1, 2, 0)
        inp = inp - inp.min()
        if inp.max() > 0:
            inp = inp / inp.max()
        img = (inp * 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        base = cv2.addWeighted(base, 0.45, np.zeros_like(base), 0.55, 0)
        H, W = base.shape[:2]
        scale = float(snake_config.down_ratio)

        py_ind = output['py_ind']
        img_mask = (py_ind.detach().long().view(-1) == 0) if py_ind.numel() == output['i_it_py'].shape[0] else torch.ones(output['i_it_py'].shape[0], device=py_ind.device, dtype=torch.bool)
        if not bool(img_mask.any().item()):
            return
        init_np = output['i_it_py'].detach().cpu().numpy()[img_mask.cpu().numpy()] * scale
        gt_np = output['i_gt_py'].detach().cpu().numpy()[img_mask.cpu().numpy()] * scale
        q_np = quality.detach().cpu().numpy()[:, img_mask.cpu().numpy()]
        order = list(range(min(len(rollouts), 6)))
        if len(rollouts) > 6:
            mean_q = q_np.mean(axis=1)
            for idx in [int(mean_q.argmax()), int(mean_q.argmin())]:
                if idx not in order:
                    order.append(idx)
        order = order[:8]
        best_idx = int(q_np.mean(axis=1).argmax())
        worst_idx = int(q_np.mean(axis=1).argmin())

        def draw(canvas, polys, color, thick):
            for poly in polys:
                pts = np.round(poly).astype(np.int32)
                pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
                loop = np.concatenate([pts, pts[:1]], axis=0)
                cv2.polylines(canvas, [loop], True, color, thick)

        det_py = deterministic_py
        if det_py is None:
            det_ret_for_viz = _deterministic_three_step(output, gcn)
            det_py = det_ret_for_viz['py']
            deterministic_polys = det_ret_for_viz.get('polys')
        det_np = det_py.detach().cpu().numpy()[img_mask.cpu().numpy()] * scale
        delta_stats = []

        panels = []
        colors = [(60, 220, 255), (80, 220, 80), (0, 170, 255), (255, 255, 255)]
        for ri in order:
            canvas = base.copy()
            draw(canvas, gt_np, (255, 0, 0), 3)
            draw(canvas, init_np, (255, 255, 0), 2)
            for si, poly_t in enumerate(rollouts[ri]['polys'][1:]):
                poly_np = poly_t.detach().cpu().numpy()[img_mask.cpu().numpy()] * scale
                draw(canvas, poly_np, colors[min(si, len(colors)-1)], 2 if si < len(rollouts[ri]['polys']) - 2 else 3)
            border = (80, 220, 80) if ri == best_idx else ((40, 40, 255) if ri == worst_idx else (90, 90, 90))
            cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), border, 4)
            label = f"k={ri} Q={float(q_np[ri].mean()):+.4f} base={float(baseline_score.mean()):.3f}"
            cv2.rectangle(canvas, (2, 2), (min(W-2, 430), 28), (0, 0, 0), -1)
            cv2.putText(canvas, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv2.LINE_AA)
            panels.append(canvas)
        if not panels:
            return
        tape = np.concatenate(panels, axis=1)
        legend = np.zeros((32, tape.shape[1], 3), dtype=np.uint8)
        cv2.putText(legend, 'cyan:init green:step1 orange:step2 white:step3 blue:GT | green border best red worst',
                    (5, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(viz_dir / f'train_group_step{step:06d}.png'), np.concatenate([legend, tape], axis=0))

        split_rows = []
        stage_names = ['init', 'step1', 'step2', 'pred']
        stage_colors = [(255, 255, 0), (80, 220, 80), (0, 170, 255), (255, 255, 255)]
        for ri in order:
            row_panels = []
            poly_stages = [output['i_it_py']] + list(rollouts[ri]['polys'][1:])
            for si, name in enumerate(stage_names):
                canvas = base.copy()
                draw(canvas, gt_np, (255, 0, 0), 3)
                poly_src = poly_stages[min(si, len(poly_stages) - 1)]
                poly_np = poly_src.detach().cpu().numpy()[img_mask.cpu().numpy()] * scale
                thick = 3 if name == 'pred' else 2
                draw(canvas, poly_np, stage_colors[si], thick)
                cv2.rectangle(canvas, (2, 2), (min(W - 2, 190), 26), (0, 0, 0), -1)
                cv2.putText(canvas, name, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (255, 255, 255), 1, cv2.LINE_AA)
                row_panels.append(canvas)
            row = np.concatenate(row_panels, axis=1)
            border = (80, 220, 80) if ri == best_idx else ((40, 40, 255) if ri == worst_idx else (90, 90, 90))
            cv2.rectangle(row, (0, 0), (row.shape[1] - 1, row.shape[0] - 1), border, 4)
            cv2.rectangle(row, (2, 28), (min(row.shape[1] - 2, 440), 55), (0, 0, 0), -1)
            cv2.putText(row, f"k={ri} Q={float(q_np[ri].mean()):+.4f} base={float(baseline_score.mean()):.3f}",
                        (7, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv2.LINE_AA)
            split_rows.append(row)
        if split_rows:
            split_tape = np.concatenate(split_rows, axis=0)
            split_legend = np.zeros((34, split_tape.shape[1], 3), dtype=np.uint8)
            cv2.putText(split_legend, 'split view: columns are init, step1, step2, pred(step3 final); blue=GT',
                        (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(viz_dir / f'train_group_step{step:06d}_split.png'),
                        np.concatenate([split_legend, split_tape], axis=0))

        delta_panels = []
        delta_magnify = 12.0
        for ri in order:
            canvas = base.copy()
            final_np = rollouts[ri]['py'].detach().cpu().numpy()[img_mask.cpu().numpy()] * scale
            diff = final_np - det_np
            diff_norm = np.linalg.norm(diff.reshape(-1, 2), axis=1)
            mean_diff = float(diff_norm.mean()) if diff_norm.size else 0.0
            max_diff = float(diff_norm.max()) if diff_norm.size else 0.0
            delta_stats.append({
                'k': int(ri),
                'mean_abs_px': mean_diff,
                'max_abs_px': max_diff,
            })
            draw(canvas, gt_np, (255, 0, 0), 3)
            draw(canvas, det_np, (180, 180, 180), 3)
            draw(canvas, final_np, (255, 255, 255), 2)
            for poly_det, poly_final in zip(det_np, final_np):
                amplified = poly_det + (poly_final - poly_det) * delta_magnify
                step_stride = max(int(poly_det.shape[0] // 16), 1)
                for pi in range(0, poly_det.shape[0], step_stride):
                    p0 = np.round(poly_det[pi]).astype(np.int32)
                    p1 = np.round(amplified[pi]).astype(np.int32)
                    p0[0] = np.clip(p0[0], 0, W - 1)
                    p0[1] = np.clip(p0[1], 0, H - 1)
                    p1[0] = np.clip(p1[0], 0, W - 1)
                    p1[1] = np.clip(p1[1], 0, H - 1)
                    cv2.arrowedLine(canvas, tuple(p0), tuple(p1), (40, 40, 255), 1, cv2.LINE_AA, tipLength=0.25)
            border = (80, 220, 80) if ri == best_idx else ((40, 40, 255) if ri == worst_idx else (90, 90, 90))
            cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), border, 4)
            label = f"k={ri} Q={float(q_np[ri].mean()):+.4f} d_mean={mean_diff:.2f}px d_max={max_diff:.2f}px"
            cv2.rectangle(canvas, (2, 2), (min(W-2, 560), 28), (0, 0, 0), -1)
            cv2.putText(canvas, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1, cv2.LINE_AA)
            delta_panels.append(canvas)
        if delta_panels:
            delta_tape = np.concatenate(delta_panels, axis=1)
            delta_legend = np.zeros((58, delta_tape.shape[1], 3), dtype=np.uint8)
            cv2.putText(delta_legend, 'delta view: gray=deterministic baseline, white=sampled final, red arrows=sample-baseline displacement x12',
                        (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(delta_legend, 'This makes conservative exploration visible; d_mean/d_max are true pixel differences before magnification.',
                        (5, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.imwrite(str(viz_dir / f'train_group_step{step:06d}_delta.png'),
                        np.concatenate([delta_legend, delta_tape], axis=0))

        step_delta_rows = []
        step_delta_stats = []
        if deterministic_polys is not None:
            det_stage_np = [
                p.detach().cpu().numpy()[img_mask.cpu().numpy()] * scale
                for p in deterministic_polys
            ]
            n_stage = min(len(det_stage_np), min(len(r['polys']) for r in rollouts))
            for ri in order:
                row_panels = []
                ri_stats = {'k': int(ri), 'stages': []}
                for si in range(1, n_stage):
                    canvas = base.copy()
                    sample_stage = rollouts[ri]['polys'][si].detach().cpu().numpy()[img_mask.cpu().numpy()] * scale
                    det_stage = det_stage_np[si]
                    diff = sample_stage - det_stage
                    diff_norm = np.linalg.norm(diff.reshape(-1, 2), axis=1)
                    mean_diff = float(diff_norm.mean()) if diff_norm.size else 0.0
                    max_diff = float(diff_norm.max()) if diff_norm.size else 0.0
                    ri_stats['stages'].append({
                        'stage': int(si),
                        'mean_abs_px': mean_diff,
                        'max_abs_px': max_diff,
                    })
                    draw(canvas, gt_np, (255, 0, 0), 2)
                    draw(canvas, det_stage, (180, 180, 180), 3)
                    draw(canvas, sample_stage, (255, 255, 255), 2)
                    for poly_det, poly_sample in zip(det_stage, sample_stage):
                        amplified = poly_det + (poly_sample - poly_det) * delta_magnify
                        step_stride = max(int(poly_det.shape[0] // 16), 1)
                        for pi in range(0, poly_det.shape[0], step_stride):
                            p0 = np.round(poly_det[pi]).astype(np.int32)
                            p1 = np.round(amplified[pi]).astype(np.int32)
                            p0[0] = np.clip(p0[0], 0, W - 1)
                            p0[1] = np.clip(p0[1], 0, H - 1)
                            p1[0] = np.clip(p1[0], 0, W - 1)
                            p1[1] = np.clip(p1[1], 0, H - 1)
                            cv2.arrowedLine(canvas, tuple(p0), tuple(p1), (40, 40, 255), 1, cv2.LINE_AA, tipLength=0.25)
                    cv2.rectangle(canvas, (2, 2), (min(W - 2, 520), 28), (0, 0, 0), -1)
                    cv2.putText(canvas, f"k={ri} step{si} d_mean={mean_diff:.2f}px d_max={max_diff:.2f}px",
                                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1, cv2.LINE_AA)
                    row_panels.append(canvas)
                if row_panels:
                    row = np.concatenate(row_panels, axis=1)
                    border = (80, 220, 80) if ri == best_idx else ((40, 40, 255) if ri == worst_idx else (90, 90, 90))
                    cv2.rectangle(row, (0, 0), (row.shape[1] - 1, row.shape[0] - 1), border, 4)
                    step_delta_rows.append(row)
                    step_delta_stats.append(ri_stats)
            if step_delta_rows:
                step_delta_tape = np.concatenate(step_delta_rows, axis=0)
                step_delta_legend = np.zeros((58, step_delta_tape.shape[1], 3), dtype=np.uint8)
                cv2.putText(step_delta_legend, 'step-delta view: columns compare sampled step1/step2/step3 with deterministic step1/step2/step3',
                            (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(step_delta_legend, 'gray=deterministic stage, white=sampled stage, red arrows=sample-stage displacement x12; true d_mean/d_max shown.',
                            (5, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.imwrite(str(viz_dir / f'train_group_step{step:06d}_stepdelta.png'),
                            np.concatenate([step_delta_legend, step_delta_tape], axis=0))

        meta = {
            'step': int(step),
            'type': 'v5_three_step_geom_action_group',
            'k_total': int(len(rollouts)),
            'k_shown': [int(x) for x in order],
            'best': int(best_idx),
            'worst': int(worst_idx),
            'quality_mean': [float(q_np[i].mean()) for i in range(len(rollouts))],
            'delta_view': {
                'magnify': float(delta_magnify),
                'stats_for_shown_k': delta_stats,
                'meaning': 'sampled final contour minus deterministic baseline final contour, measured in image pixels',
            },
            'step_delta_view': {
                'magnify': float(delta_magnify),
                'stats_for_shown_k': step_delta_stats,
                'meaning': 'sampled stage contour minus deterministic contour at the same stage, measured in image pixels',
            },
        }
        with open(viz_dir / f'train_group_step{step:06d}.json', 'w') as f:
            json.dump(meta, f, indent=2)

    train_iter = iter(train_loader)
    for step in range(start_step + 1, train_steps + 1):
        freeze_bn_running_stats(inner)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        _move_batch(batch)
        output = _manual_context(batch)
        i_init = output['i_it_py']
        i_gt = output['i_gt_py']
        if i_init.numel() == 0 or i_gt.numel() == 0:
            print(f'[RL-V5] step {step}: empty contour batch, skipping.')
            continue

        init_iou = None
        reg_mask = None
        if reward_regression_weight > 0:
            _, init_comps = _score_with_components(output['i_it_py'], i_gt, output['image_hw'])
            init_iou = init_comps['iou'].detach()
            reg_mask = (init_iou > reward_regression_init_thresh).float()

        det_ret = _deterministic_three_step(output, gcn)
        baseline_score, baseline_comps = _score_with_components(
            output['i_it_py'] + det_ret['disp'], i_gt, output['image_hw']
        )
        baseline_score = baseline_score.detach()
        baseline_comps = {k: v.detach() for k, v in baseline_comps.items()}
        # delta-NSD baseline: NSD of the deterministic rollout
        if use_delta_nsd_reward:
            baseline_nsd = compute_nsd_score(
                output['i_it_py'] + det_ret['disp'], i_gt,
                H=int(output['image_hw'][0]), W=int(output['image_hw'][1]),
                delta_px=nsd_delta_px, coord_scale=float(snake_config.down_ratio),
            ).detach()
        if sigma_difficulty_scale > 0.0 and 'iou' not in baseline_comps:
            baseline_comps['iou'] = compute_region_score(
                output['i_it_py'] + det_ret['disp'],
                i_gt,
                H=int(output['image_hw'][0]),
                W=int(output['image_hw'][1]),
                w_boundary=0,
                w_dice=0,
                w_iou=1,
                w_dist=0,
                coord_scale=float(snake_config.down_ratio),
            ).detach()
        sigma_multiplier = 1.0
        if sigma_difficulty_scale > 0.0 and 'iou' in baseline_comps:
            baseline_iou_mean = float(baseline_comps['iou'].mean().item()) if baseline_comps['iou'].numel() else 1.0
            difficulty = max(float(hard_iou_thresh) - baseline_iou_mean, 0.0)
            sigma_multiplier = 1.0 + float(sigma_difficulty_scale) * difficulty / max(float(hard_iou_thresh), 1e-6)
        if reward_regression_weight > 0:
            baseline_reg_penalty = (reg_mask * torch.relu(init_iou - baseline_comps['iou'])).detach()
            baseline_score_adj = baseline_score - reward_regression_weight * baseline_reg_penalty
        else:
            baseline_reg_penalty = torch.zeros_like(baseline_score)
            baseline_score_adj = baseline_score

        rollouts: List[Dict] = []
        final_scores = []
        final_scores_reward = []
        final_score_components: List[Dict[str, torch.Tensor]] = []
        burr_penalties = []
        burr_raws = []
        regression_penalties = []
        old_log_counts = []
        for _ in range(k_rollouts):
            ret = _sample_rollout(output, sigma_multiplier=sigma_multiplier)
            rollouts.append(ret)
            final_poly = output['i_it_py'] + ret['disp']
            score, score_comps = _score_with_components(final_poly, i_gt, output['image_hw'])
            score = score.detach()
            burr_penalty, burr_raw = _burr_penalty(
                final_poly, output['i_it_py'], i_gt,
                coord_scale=float(snake_config.down_ratio),
                margin_px=reward_burr_margin_px,
                max_px=reward_burr_max_px,
                quantile=reward_burr_quantile,
            )
            if reward_regression_weight > 0:
                regression_penalty = (reg_mask * torch.relu(init_iou - score_comps['iou'].detach())).detach()
            else:
                regression_penalty = torch.zeros_like(score)
            final_scores.append(score)
            final_score_components.append({k: v.detach() for k, v in score_comps.items()})
            if use_delta_nsd_reward:
                final_nsd = compute_nsd_score(
                    final_poly, i_gt,
                    H=int(output['image_hw'][0]), W=int(output['image_hw'][1]),
                    delta_px=nsd_delta_px, coord_scale=float(snake_config.down_ratio),
                ).detach()
                # delta NSD: improvement over deterministic baseline (burr penalty preserved)
                reward_val = (final_nsd - baseline_nsd - reward_burr_weight * burr_penalty.detach())
            else:
                reward_val = (
                    score - reward_burr_weight * burr_penalty.detach()
                    - reward_regression_weight * regression_penalty
                )
            final_scores_reward.append(reward_val.detach())
            burr_penalties.append(burr_penalty.detach())
            burr_raws.append(burr_raw.detach())
            regression_penalties.append(regression_penalty)
            old_log_counts.append(len(ret['old_logs']))

        # ===== CREDIT-ASSIGNMENT DIAGNOSTIC (offline, opt-in) =====
        # Toggle via env RL_V4_CREDIT_DIAG=1. When active we measure the
        # "partial score" at every outer-step truncation for every rollout and
        # the deterministic truncation baselines, then dump JSON and continue
        # the normal flow (no behaviour change unless RL_V4_CREDIT_DIAG_STOP
        # is set, in which case we skip the PPO update + checkpoint for this
        # iteration only). Output: <out_dir>/credit_diag_step{S}.json
        if _as_bool(os.environ.get('RL_V4_CREDIT_DIAG', '0')):
            try:
                _diag_phase = 'pre_update'
                with torch.no_grad():
                    det_polys_seq = det_ret.get('polys', [])  # [init, after_s1, ..., after_sN]
                    n_steps = min(len(fractions), max(len(det_polys_seq) - 1, 0))
                    # deterministic per-step truncation scores
                    det_partials = []
                    for t in range(1, n_steps + 1):
                        s = _quality_score(det_polys_seq[t], i_gt, output['image_hw'])
                        det_partials.append(float(s.mean().item()) if s.numel() else 0.0)
                    det_partial_init = float(_quality_score(
                        det_polys_seq[0], i_gt, output['image_hw']).mean().item()
                    ) if det_polys_seq else 0.0
                    diag_rollouts = []
                    for ri, ret in enumerate(rollouts):
                        polys_seq = ret.get('polys', [])  # [init, a1, a2, ...]
                        partials = []
                        actions_norm = []
                        olds_norm = []
                        m = len(polys_seq) - 1
                        for t in range(1, m + 1):
                            s = _quality_score(polys_seq[t], i_gt, output['image_hw'])
                            partials.append(float(s.mean().item()) if s.numel() else 0.0)
                        for t in range(min(m, len(ret.get('actions', [])))):
                            a = ret['actions'][t]
                            actions_norm.append(float(a.norm(dim=-1).mean().item()) if a.numel() else 0.0)
                            ol = ret['old_logs'][t]
                            olds_norm.append(float(ol.mean().item()) if ol.numel() else 0.0)
                        # per-step advantage proxy vs det truncation
                        step_q = []
                        for t in range(min(len(partials), len(det_partials))):
                            step_q.append(float(partials[t] - det_partials[t]))
                        diag_rollouts.append({
                            'partials': partials,
                            'actions_norm_px': [x * float(snake_config.down_ratio) for x in actions_norm],
                            'old_logs_mean': olds_norm,
                            'partial_init': det_partial_init,
                            'det_partials': det_partials,
                            'step_delta_vs_det': step_q,  # sampled_truncated - det_truncated
                        })
                    diag_payload = {
                        'phase': _diag_phase,
                        'step': int(step),
                        'outer_steps': int(outer_steps),
                        'fractions': [float(x) for x in fractions],
                        'geom_sigma_px': [float(x) for x in sigma_px],
                        'k_rollouts': int(k_rollouts),
                        'adv_clip_max': float(adv_clip_max),
                        'det_final_score': float(baseline_score.mean().item()),
                        'init_score': float(_quality_score(
                            output['i_it_py'], i_gt, output['image_hw']).mean().item()),
                        'rollouts': diag_rollouts,
                        # end-of-trajectory quantities (kept for parity with the
                        # terminal-only view used by the PPO update below)
                        'terminal_final_scores': [float(x.mean().item()) for x in final_scores],
                    }
                # we re-evaluate the terminal block independently for diag
                # (mirrors the real computation) so the JSON stands alone.
                with torch.no_grad():
                    _term_scores = torch.stack([
                        _quality_score(output['i_it_py'] + ret['disp'], i_gt, output['image_hw'])
                        for ret in rollouts
                    ], dim=0).to(i_init.device)
                    _term_burr = torch.stack([
                        _burr_penalty(
                            output['i_it_py'] + ret['disp'],
                            output['i_it_py'], i_gt,
                            coord_scale=float(snake_config.down_ratio),
                            margin_px=reward_burr_margin_px,
                            max_px=reward_burr_max_px,
                            quantile=reward_burr_quantile,
                        )[0].detach()
                        for ret in rollouts
                    ], dim=0).to(i_init.device)
                    _term_rew = _term_scores - reward_burr_weight * _term_burr
                    _term_quality = _term_rew - baseline_score_adj.unsqueeze(0)
                    _term_std = _term_quality.std(dim=0, unbiased=False).clamp_min(0.1)
                    _term_adv = (_term_quality / _term_std).clamp(-adv_clip_max, adv_clip_max)
                    # gate mechanism removed; adv_abs_mean is unmasked
                diag_payload['terminal_recompute'] = {
                    'final_scores_mean': float(_term_scores.mean().item()),
                    'burr_mean': float(_term_burr.mean().item()),
                    'quality_mean': float(_term_quality.mean().item()),
                    'quality_std_mean': float(_term_std.mean().item()),
                    'adv_mean': float(_term_adv.mean().item()),
                    'adv_abs_mean': float(_term_adv.abs().mean().item()),
                    'all_krollouts_quality_per_contour_mean': [
                        float(_term_quality[ri].mean().item()) for ri in range(len(rollouts))
                    ],
                }
                _diag_path = out_dir / f'credit_diag_step{int(step)}.json'
                with open(_diag_path, 'w') as _fh:
                    json.dump(diag_payload, _fh, indent=2)
                print(f'[RL-V5][CREDIT_DIAG] wrote {_diag_path} '
                      f'| init={diag_payload["init_score"]:.4f} '
                      f'| det_final={diag_payload["det_final_score"]:.4f} '
                      f'| terminal_q_mean={diag_payload["terminal_recompute"]["quality_mean"]:+.4f} '
                      f'| adv_abs={diag_payload["terminal_recompute"]["adv_abs_mean"]:.4f}')
                if _as_bool(os.environ.get('RL_V4_CREDIT_DIAG_STOP', '0')):
                    # consume the rollout, skip the PPO update + checkpoint.
                    # (do not del quantities defined later in the loop; rely on
                    # gc/loop-overwrite)
                    del _term_scores, _term_burr, _term_rew, _term_quality
                    del _term_std, _term_adv
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
            except Exception as _diag_e:
                import traceback as _tb
                print('[RL-V5][CREDIT_DIAG] ERROR:', _diag_e)
                _tb.print_exc()
        # ===== END CREDIT-ASSIGNMENT DIAGNOSTIC =====

        final_scores_t = torch.stack(final_scores, dim=0).to(i_init.device)
        final_scores_reward_t = torch.stack(final_scores_reward, dim=0).to(i_init.device)
        burr_penalty_t = torch.stack(burr_penalties, dim=0).to(i_init.device)
        burr_raw_t = torch.stack(burr_raws, dim=0).to(i_init.device)
        regression_penalty_t = torch.stack(regression_penalties, dim=0).to(i_init.device)
        regression_active_frac = float(reg_mask.mean().item()) if reg_mask is not None else 0.0
        quality = final_scores_reward_t if use_delta_nsd_reward else (
            final_scores_reward_t - baseline_score_adj.unsqueeze(0)
        )
        # gate mechanism removed; all rollouts contribute to advantage
        adv = quality / quality.std(dim=0, unbiased=False, keepdim=True).clamp_min(0.1)
        adv = adv.clamp(-adv_clip_max, adv_clip_max)
        reward_mean = float(quality.mean().item())
        ema_reward_val = ema_reward.update(reward_mean)

        # ===== PER-STEP SHAPED REWARD =====
        # When per_step_reward_weight > 0 we blend the terminal-only advantage
        # (already in `adv`, anchored on the final 5-step contour so the
        # optimisation target does NOT degenerate to greedy single-step) with a
        # per-step advantage that gives each outer-step its own credit signal.
        #
        # adv_ps shape: [K, n_steps, N_contours]
        #   adv_ps[ri, si] = (1-w)*terminal_adv[ri] + w*step_adv[ri, si]
        #
        # Two credit-attribution modes (rl_v4_per_step_credit_mode):
        #   'seq_delta'   : step_quality = q(polys[si+1]) - q(polys[si]) - det_seq_delta
        #                   -> measures the intermediate contour after walking
        #                      only frac[si] of the step (mixes step-length scale
        #                      noise into the signal).
        #   'full_extrap' : step_quality = q(extrap[si]) - q(det_extrap[si])
        #                   where extrap[si] = polys[si] + (polys[si+1]-polys[si])/frac[si]
        #                   -> extrapolates each step's action to the FULL
        #                      displacement (frac=1) and scores that endpoint, so
        #                      every step is judged on the same "one-shot to goal"
        #                      ruler. Attribution only; reward stays terminal.
        #
        # If per_step_reward_weight == 0 (default) this entire block is a no-op:
        # adv_ps is just adv broadcast to [K, n_steps, N_contours].
        n_outer = len(fractions)
        if per_step_reward_weight > 0.0 and n_outer > 0:
            with torch.no_grad():
                _frac_list = [max(float(fractions[_si]), 1e-3) for _si in range(n_outer)]

                def _step_quality_seq(_polys):
                    # sequential intermediate-contour deltas: [n_outer, N_contours]
                    _out = []
                    for _si in range(n_outer):
                        _p0 = _polys[_si] if _si < len(_polys) else _polys[-1]
                        _p1 = _polys[_si + 1] if (_si + 1) < len(_polys) else _polys[-1]
                        _q0 = _quality_score(_p0, i_gt, output['image_hw']).detach()
                        _q1 = _quality_score(_p1, i_gt, output['image_hw']).detach()
                        _out.append(_q1 - _q0)
                    return torch.stack(_out, dim=0)

                def _step_quality_extrap(_polys):
                    # full-displacement extrapolation endpoint score: [n_outer, N_contours]
                    _out = []
                    for _si in range(n_outer):
                        _p0 = _polys[_si] if _si < len(_polys) else _polys[-1]
                        _p1 = _polys[_si + 1] if (_si + 1) < len(_polys) else _polys[-1]
                        # applied action = p1 - p0 == full_disp * frac  ->  full_disp = (p1-p0)/frac
                        _extrap = _p0 + (_p1 - _p0) / _frac_list[_si]
                        _out.append(_quality_score(_extrap, i_gt, output['image_hw']).detach())
                    return torch.stack(_out, dim=0)

                _step_fn = (
                    _step_quality_extrap
                    if per_step_credit_mode == 'full_extrap'
                    else _step_quality_seq
                )

                det_polys = det_ret.get('polys', [])
                det_step_t = _step_fn(det_polys)  # [n_outer, N_contours]

                rollout_step = []
                for _ret in rollouts:
                    rollout_step.append(_step_fn(_ret.get('polys', [])))
                rollout_step_t = torch.stack(rollout_step, dim=0)  # [K, n_outer, N_contours]

                # Centre by deterministic baseline at the same step (same role as
                # baseline_score_adj for the terminal reward).
                step_quality = rollout_step_t - det_step_t.unsqueeze(0)

                step_adv = step_quality / step_quality.std(
                    dim=0, unbiased=False, keepdim=True
                ).clamp_min(0.1)
                step_adv = step_adv.clamp(-adv_clip_max, adv_clip_max)

                # Blend: (1-w)*terminal + w*per-step
                adv_ps = (
                    (1.0 - per_step_reward_weight) * adv.unsqueeze(1)
                    + per_step_reward_weight * step_adv
                )  # [K, n_outer, N_contours]
        else:
            # Default: broadcast terminal advantage, no extra compute
            adv_ps = adv.unsqueeze(1).expand(-1, max(n_outer, 1), -1)
            # [K, n_outer, N_contours]
        # ===== END PER-STEP SHAPED REWARD =====

        total_actions = sum(len(t['actions']) for t in rollouts)
        total_prefixes = sum(len(t.get('prefix_states', [])) for t in rollouts)
        approx_kl_hist, ratio_hist, loss_hist, prefix_loss_hist = [], [], [], []
        early_stop_epoch = ppo_inner_epochs

        for epoch in range(ppo_inner_epochs):
            optimizer.zero_grad(set_to_none=True)
            epoch_losses = []
            epoch_kls = []
            epoch_ratios = []
            for ri, traj in enumerate(rollouts):
                _n_ps = adv_ps.shape[1]  # n_outer steps
                for si, action in enumerate(traj['actions']):
                    state = traj['states'][si]
                    c_state = traj['c_states'][si]
                    frac = traj['fractions'][si]
                    old_log = traj['old_logs'][si]
                    sigma = traj['sigmas'][si]
                    entropy_bonus = None
                    if action_policy in ('df', 'df_inner_step', 'diffusion_inner_step', 'flow_inner_step'):
                        step_total = int(traj['total_steps'][si].item())
                        _sde_win = tuple(traj.get('sde_win', [0, step_total]))
                        _step_idx = int(traj['step_indices'][si].item())
                        _cur_action_std = df_action_std if (_sde_win[0] <= _step_idx < _sde_win[1]) else 0.0
                        _, lp_cur, mean_cur, std_cur, _ = gcn.step_with_logprob(
                            output['cnn_feature'],
                            state,
                            c_state,
                            output['py_ind'],
                            x_t=traj['x_ts'][si],
                            t_value=traj['timesteps'][si],
                            step_index=_step_idx,
                            total_steps=step_total,
                            action_std=_cur_action_std,
                            prev_sample=traj['x_prevs'][si],
                            sampled_feat=traj['sampled_feats'][si],
                            detail_feat=traj['detail_feats'][si],
                            contour_scale=traj['contour_scales'][si],
                            x_self_cond=traj['x_self_conds'][si],
                            step_mode=df_step_mode,
                            noise_level=df_noise_level,
                            sde_type=df_sde_type,
                        )
                    else:
                        mean_cur = _outer_action_mean(
                            gcn, output['cnn_feature'], state, c_state, output['py_ind'], frac, ode_steps
                        )
                        if action_policy == 'fm_velocity' and fm_velocity_policy is not None:
                            outer_idx = int(traj.get('outer_indices', [si])[si])
                            sampled_feat = traj['geom_sampled_feats'][si]
                            scale_old = traj['fm_velocity_scales'][si]  # (B, 1)
                            mu_s, logstd_s = fm_velocity_policy(
                                outer_idx, state, c_state, mean_cur, sampled_feat, frac
                            )
                            lp_cur = _fm_velocity_logprob(
                                action, mean_cur, state, scale_old, mu_s, logstd_s, float(sigma)
                            )
                        elif action_policy == 'per_point_fm_velocity' and fm_velocity_policy is not None:
                            outer_idx = int(traj.get('outer_indices', [si])[si])
                            sampled_feat = traj['geom_sampled_feats'][si]
                            scale_pp_old = traj['fm_velocity_scales'][si]  # (B, N)
                            mu_pp, logstd_g = fm_velocity_policy(
                                outer_idx, state, c_state, mean_cur, sampled_feat, frac
                            )
                            lp_cur = _per_point_fm_velocity_logprob(scale_pp_old, mu_pp, logstd_g)
                        elif action_policy == 'per_point_fm_scale' and fm_velocity_policy is not None:
                            outer_idx = int(traj.get('outer_indices', [si])[si])
                            sampled_feat = traj['geom_sampled_feats'][si]
                            raw_pp_old = traj['fm_velocity_scales'][si]  # (B, N) PRE-TANH raw sample
                            mu_pp, logstd_pp = fm_velocity_policy(
                                outer_idx, state, c_state, mean_cur, sampled_feat, frac
                            )
                            lp_cur = _per_point_fm_scale_logprob(
                                raw_pp_old, mu_pp, logstd_pp, fm_velocity_policy.max_scale
                            )
                            # Entropy of the underlying (pre-tanh) Gaussian, as a proxy for
                            # exploration entropy. Differentiable w.r.t. logstd_pp (network
                            # output) -> subtracting it from the loss (i.e. rewarding higher
                            # entropy) counteracts PPO's natural tendency to collapse std to 0.
                            entropy_bonus = (
                                0.5 * math.log(2.0 * math.pi * math.e) + logstd_pp
                            ).mean()
                        elif action_policy in ('band_detail', 'geom_band_detail', 'detail_band'):
                            outer_idx = int(traj.get('outer_indices', [si])[si])
                            d_idx = min(max(outer_idx, 0), len(detail_sigma_feat) - 1)
                            detail_sigmas = traj.get('detail_sigmas', [])
                            detail_sigma_cur = (
                                float(detail_sigmas[si]) if si < len(detail_sigmas)
                                else float(detail_sigma_feat[d_idx])
                            )
                            if explorer is not None:
                                residual = action.detach() - mean_cur
                                sampled_feat = traj['geom_sampled_feats'][si]
                                low_mu, low_logstd, detail_mu, detail_logstd = explorer(state, sampled_feat, frac)
                                lp_parts = []
                                if float(sigma) > 0.0:
                                    z_low = _project_geom_z(
                                        state,
                                        residual,
                                        sigma,
                                        geom_modes,
                                        damp_highfreq=geom_lowfreq_damp_highfreq,
                                    )
                                    lp_parts.append(_normal_z_logprob(z_low, low_mu, low_logstd))
                                d_sigma = detail_sigma_cur
                                d_gate = float(detail_gate[d_idx])
                                if d_sigma > 0.0 and d_gate > 0.0 and float(detail_logprob_weight) > 0.0:
                                    z_detail = _project_band_detail_z(
                                        state,
                                        residual,
                                        d_sigma,
                                        detail_k_min,
                                        detail_k_max,
                                        d_gate,
                                    )
                                    lp_parts.append(
                                        float(detail_logprob_weight)
                                        * _normal_z_logprob(z_detail, detail_mu, detail_logstd)
                                    )
                                lp_cur = sum(lp_parts) if lp_parts else torch.zeros_like(old_log)
                            else:
                                lp_cur = _band_detail_action_logprob(
                                    action,
                                    mean_cur,
                                    state,
                                    sigma,
                                    geom_modes,
                                    detail_sigma_cur,
                                    detail_k_min,
                                    detail_k_max,
                                    float(detail_gate[d_idx]),
                                    detail_logprob_weight,
                                    damp_highfreq=geom_lowfreq_damp_highfreq,
                                )
                        else:
                            lp_cur = _geom_action_logprob(
                                action,
                                mean_cur,
                                state,
                                sigma,
                                geom_modes,
                                damp_highfreq=geom_lowfreq_damp_highfreq,
                            )
                        std_cur = None
                    ratio = torch.exp(lp_cur - old_log)
                    # adv_ps: [K, n_outer, N_contours]; index by (ri, si) for per-step signal
                    _adv_si = adv_ps[ri, min(si, _n_ps - 1)].detach()
                    unclipped = -_adv_si * ratio
                    clipped = -_adv_si * torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
                    policy_loss = torch.maximum(unclipped, clipped).mean() / max(total_actions, 1)
                    if kl_beta > 0:
                        if action_policy in ('df', 'df_inner_step', 'diffusion_inner_step', 'flow_inner_step'):
                            with torch.no_grad():
                                _, _, mean_ref, _, _ = ref_flow.step_with_logprob(
                                    output['cnn_feature'],
                                    state,
                                    c_state,
                                    output['py_ind'],
                                    x_t=traj['x_ts'][si],
                                    t_value=traj['timesteps'][si],
                                    step_index=_step_idx,
                                    total_steps=step_total,
                                    action_std=_cur_action_std,
                                    prev_sample=traj['x_prevs'][si],
                                    sampled_feat=traj['sampled_feats'][si],
                                    detail_feat=traj['detail_feats'][si],
                                    contour_scale=traj['contour_scales'][si],
                                    x_self_cond=traj['x_self_conds'][si],
                                    step_mode=df_step_mode,
                                    noise_level=df_noise_level,
                                    sde_type=df_sde_type,
                                )
                            var = std_cur.pow(2).clamp_min(1e-12)
                            kl_loss = (((mean_cur - mean_ref) ** 2) / (2.0 * var)).mean() / max(total_actions, 1)
                        else:
                            with torch.no_grad():
                                mean_ref = _outer_action_mean(
                                    ref_flow, output['cnn_feature'], state, c_state, output['py_ind'], frac, ode_steps
                                )
                            if action_policy in ('per_point_fm_velocity', 'per_point_fm_scale'):
                                kl_loss = 0.5 * (mean_cur - mean_ref).pow(2).mean() / max(total_actions, 1)
                            else:
                                mean_shift_z = _project_geom_z(
                                    state,
                                    mean_cur - mean_ref,
                                    sigma,
                                    geom_modes,
                                    damp_highfreq=geom_lowfreq_damp_highfreq,
                                )
                                kl_loss = 0.5 * mean_shift_z.pow(2).mean() / max(total_actions, 1)
                        loss = policy_loss + kl_beta * kl_loss
                    else:
                        loss = policy_loss
                    if entropy_bonus is not None and fmv_entropy_weight > 0.0:
                        # Encourage logstd to stay wide: subtract entropy (maximize it).
                        # entropy_bonus is already divided by total_actions at its computation site.
                        loss = loss - fmv_entropy_weight * entropy_bonus
                    if loss.requires_grad:
                        loss.backward()
                    with torch.no_grad():
                        epoch_losses.append(float(policy_loss.detach().item()))
                        epoch_kls.append(float(0.5 * torch.mean((lp_cur - old_log) ** 2).item()))
                        epoch_ratios.append(ratio.detach())

                if prefix_distill_weight > 0.0 and traj.get('prefix_states'):
                    for pi, state in enumerate(traj['prefix_states']):
                        c_state = traj['prefix_c_states'][pi]
                        frac = traj['prefix_fractions'][pi]
                        mean_cur = _outer_action_mean(
                            gcn, output['cnn_feature'], state, c_state, output['py_ind'], frac, ode_steps
                        )
                        with torch.no_grad():
                            mean_ref = _outer_action_mean(
                                ref_flow, output['cnn_feature'], state, c_state, output['py_ind'], frac, ode_steps
                            )
                        prefix_loss = torch.nn.functional.smooth_l1_loss(
                            mean_cur,
                            mean_ref,
                            beta=1.0,
                            reduction='mean',
                        )
                        scaled_prefix_loss = (
                            float(prefix_distill_weight) * prefix_loss / max(total_prefixes, 1)
                        )
                        if scaled_prefix_loss.requires_grad:
                            scaled_prefix_loss.backward()
                        with torch.no_grad():
                            prefix_loss_hist.append(float(prefix_loss.detach().item()))

            trainable_params = [p for p in net_for_load.parameters() if p.requires_grad]
            if explorer is not None:
                trainable_params += [p for p in explorer.parameters() if p.requires_grad]
            if fm_velocity_policy is not None:
                trainable_params += [p for p in fm_velocity_policy.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=grad_clip_norm,
            )
            # Diagnostic: print fm_velocity_policy grad_norm every 10 PPO steps
            if fm_velocity_policy is not None and action_policy in ('per_point_fm_scale', 'per_point_fm_velocity') and epoch == 0:
                with torch.no_grad():
                    fmv_params = [p for p in fm_velocity_policy.parameters() if p.requires_grad and p.grad is not None]
                    if fmv_params:
                        fmv_gn = float(torch.stack([p.grad.norm() for p in fmv_params]).norm().item())
                    else:
                        fmv_gn = 0.0
                if step % 10 == 0:
                    print(f'[DIAG] step={step} fmv_policy_grad_norm={fmv_gn:.6f} '
                          f'grad_norm_total={float(grad_norm):.6f}')
            optimizer.step()
            mean_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
            approx_kl_hist.append(mean_kl)
            if epoch_ratios:
                ratio_hist.append(torch.cat(epoch_ratios))
            loss_hist.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
            if ppo_kl_target > 0 and mean_kl > ppo_kl_target:
                early_stop_epoch = epoch + 1
                break

        eval_metrics = {}
        if eval_every > 0 and (step == 1 or step % eval_every == 0):
            eval_metrics = _compute_eval()
            if eval_metrics and eval_metrics['eval_iou'] > best_eval_iou:
                best_eval_iou = float(eval_metrics['eval_iou'])
                _save_checkpoint(ckpt_dir / 'best_iou.pt', step, eval_metrics)
            if action_policy == 'per_point_fm_scale' and fm_velocity_policy is not None:
                diag_corr = _diag_mu_error_correlation()
                if diag_corr:
                    eval_metrics.update(diag_corr)
                    print(f"[DIAG-SMART] step={step} mu_err_corr={diag_corr['diag_mu_err_corr']:+.4f} "
                          f"n={diag_corr['diag_mu_err_n']}", flush=True)

        metrics = {
            'step': step,
            'reward_mean': reward_mean,
            'reward_ema': float(ema_reward_val),
            'quality_best_mean': float(quality.max(dim=0).values.mean().item()),
            'quality_p10': percentiles(quality.flatten())['p10'],
            'quality_p50': percentiles(quality.flatten())['p50'],
            'quality_p90': percentiles(quality.flatten())['p90'],
            'baseline_score_mean': float(baseline_score.mean().item()),
            'final_score_mean': float(final_scores_t.mean().item()),
            'final_score_reward_mean': float(final_scores_reward_t.mean().item()),
            'reward_global_weight': float(reward_global_weight),
            'reward_detail_weight': float(reward_detail_weight),
            'burr_penalty_mean': float(burr_penalty_t.mean().item()),
            'burr_raw_px_mean': float(burr_raw_t.mean().item()),
            'reward_burr_weight': float(reward_burr_weight),
            'reward_burr_quantile': float(reward_burr_quantile),
            'regression_penalty_mean': float(regression_penalty_t.mean().item()),
            'regression_active_frac': regression_active_frac,
            'reward_regression_weight': float(reward_regression_weight),
            'reward_regression_init_thresh': float(reward_regression_init_thresh),
            'outer_log_count_mean': float(np.mean(old_log_counts)) if old_log_counts else 0.0,
            'prefix_distill_count_mean': float(np.mean([
                len(t.get('prefix_states', [])) for t in rollouts
            ])) if rollouts else 0.0,
            'prefix_distill_weight': float(prefix_distill_weight),
            'prefix_distill_loss': float(np.mean(prefix_loss_hist)) if prefix_loss_hist else 0.0,
            'adaptive_explorer': bool(explorer is not None),
            'approx_kl': float(np.mean(approx_kl_hist)) if approx_kl_hist else 0.0,
            'policy_loss': float(np.mean(loss_hist)) if loss_hist else 0.0,
            'grad_norm': float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
            'early_stop_epoch': int(early_stop_epoch),
            'action_policy': action_policy,
            'df_step_mode': df_step_mode,
            'df_sde_type': df_sde_type,
            'df_noise_level': float(df_noise_level),
            'df_action_std': float(df_action_std),
            'geom_lowfreq_modes': int(geom_modes),
            'geom_sigma_px': [float(x) for x in sigma_px],
            'sigma_difficulty_multiplier': float(sigma_multiplier),
            'lr': lr,
        }
        for key, value in baseline_comps.items():
            metrics[f'baseline_{key}_mean'] = float(value.mean().item())
        if final_score_components:
            for key in final_score_components[0].keys():
                values = torch.stack([comp[key] for comp in final_score_components], dim=0)
                metrics[f'final_{key}_mean'] = float(values.mean().item())
        if ratio_hist:
            ratio_all = torch.cat(ratio_hist)
            metrics.update({
                'ratio_mean': float(ratio_all.mean().item()),
                'ratio_min': float(ratio_all.min().item()),
                'ratio_max': float(ratio_all.max().item()),
            })
        metrics.update(eval_metrics)

        with open(log_path, 'a') as f:
            f.write(json.dumps(metrics, sort_keys=True) + '\n')

        if step % log_every == 0:
            extra = ''
            if eval_metrics:
                extra = f" eval_iou={eval_metrics['eval_iou']:.4f} mbf={eval_metrics['eval_mboundf']:.4f}"
            print(
                f"[RL-V5] step={step}/{train_steps} reward={reward_mean:+.5f} "
                f"best={metrics['quality_best_mean']:+.5f} "
                f"dsig={sigma_multiplier:.2f} "
                f"burr={metrics['burr_penalty_mean']:.3f} reg={metrics['regression_penalty_mean']:+.3f} "
                f"detail={metrics.get('final_detail_score_mean', 0.0):+.3f} "
                f"kl={metrics['approx_kl']:.6f} logs={metrics['outer_log_count_mean']:.1f}{extra}",
                flush=True,
            )

        if viz_every > 0 and (step == 1 or step % viz_every == 0):
            try:
                _dump_group_viz(batch, output, rollouts, quality, baseline_score_adj, step, det_ret['py'], det_ret.get('polys'))
            except Exception as e:
                print(f'[RL-V5] viz failed at step {step}: {e}', flush=True)

        if step == 1 or (save_every > 0 and step % save_every == 0):
            _save_checkpoint(ckpt_dir / 'latest.pt', step, metrics)
            if save_every > 0 and step % save_every == 0:
                _save_checkpoint(ckpt_dir / f'step{step}.pt', step, metrics)

        del batch, output, rollouts, final_scores_t, quality, adv, final_score_components
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_eval = _compute_eval()
    final_metrics = {'final_eval': final_eval, 'step': train_steps}
    _save_checkpoint(ckpt_dir / 'latest.pt', train_steps, final_metrics)
    print(f'[RL-V5] done. output={out_dir}')


if __name__ == '__main__':
    main()
