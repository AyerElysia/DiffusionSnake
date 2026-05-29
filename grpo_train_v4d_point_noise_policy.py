"""RL V4d-B: per-point latent noise policy for three outer refinements."""

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

from lib.config import cfg, args
from lib.datasets import make_data_loader
from lib.networks import make_network
from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
from lib.train.grpo_v2_utils import EMA, freeze_bn_running_stats, percentiles
from lib.train.rewards.region_reward import compute_region_score
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


class PointLatentNoisePolicy(torch.nn.Module):
    """Geometry-conditioned per-point policy over each outer step's initial latent noise."""

    def __init__(self, outer_steps: int, init_std: float, min_logstd: float, max_logstd: float, hidden_dim: int = 64):
        super().__init__()
        outer_steps = max(int(outer_steps), 1)
        init_std = max(float(init_std), 1e-6)
        self.outer_steps = outer_steps
        self.init_logstd = math.log(init_std)
        self.min_logstd = float(min_logstd)
        self.max_logstd = float(max_logstd)
        self.step_embed = torch.nn.Embedding(outer_steps, 8)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(17, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 4),
        )
        last = self.mlp[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)

    @staticmethod
    def _normalize(poly: torch.Tensor) -> torch.Tensor:
        center = poly.mean(dim=1, keepdim=True)
        span = (poly.max(dim=1, keepdim=True).values - poly.min(dim=1, keepdim=True).values).amax(
            dim=-1, keepdim=True
        )
        return (poly - center) / span.clamp_min(1.0)

    def forward(self, step_index: int, state: torch.Tensor, c_state: torch.Tensor, frac: float) -> tuple[torch.Tensor, torch.Tensor]:
        idx = max(0, min(int(step_index), self.outer_steps - 1))
        n, p, _ = state.shape
        state_n = self._normalize(state)
        can_n = self._normalize(c_state)
        prev_n = torch.roll(state_n, shifts=1, dims=1)
        next_n = torch.roll(state_n, shifts=-1, dims=1)
        tangent = 0.5 * (next_n - prev_n)
        curvature = next_n - 2.0 * state_n + prev_n
        frac_feat = state.new_full((n, p, 1), float(frac))
        step_idx = torch.full((n, p), idx, dtype=torch.long, device=state.device)
        step_feat = self.step_embed(step_idx).to(dtype=state.dtype)
        feat = torch.cat([state_n, can_n, tangent, curvature, frac_feat, step_feat], dim=-1)
        out = self.mlp(feat)
        mu = out[..., :2]
        logstd_delta = out[..., 2:]
        logstd = (logstd_delta + self.init_logstd).clamp(self.min_logstd, self.max_logstd)
        return mu, logstd


class StructuredLatentNoisePolicy(torch.nn.Module):
    """Low-dimensional geometry-conditioned policy over structured latent noise fields."""

    def __init__(
        self,
        outer_steps: int,
        init_std: float,
        min_logstd: float,
        max_logstd: float,
        hidden_dim: int = 64,
        num_coeffs: int = 8,
    ):
        super().__init__()
        outer_steps = max(int(outer_steps), 1)
        init_std = max(float(init_std), 1e-6)
        self.outer_steps = outer_steps
        self.init_logstd = math.log(init_std)
        self.min_logstd = float(min_logstd)
        self.max_logstd = float(max_logstd)
        self.num_coeffs = max(int(num_coeffs), 4)
        self.step_embed = torch.nn.Embedding(outer_steps, 8)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(17, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 2 * self.num_coeffs),
        )
        last = self.mlp[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)

    @staticmethod
    def _normalize(poly: torch.Tensor) -> torch.Tensor:
        center = poly.mean(dim=1, keepdim=True)
        span = (poly.max(dim=1, keepdim=True).values - poly.min(dim=1, keepdim=True).values).amax(
            dim=-1, keepdim=True
        )
        return (poly - center) / span.clamp_min(1.0)

    def forward(self, step_index: int, state: torch.Tensor, c_state: torch.Tensor, frac: float) -> tuple[torch.Tensor, torch.Tensor]:
        idx = max(0, min(int(step_index), self.outer_steps - 1))
        n = state.size(0)
        state_n = self._normalize(state)
        can_n = self._normalize(c_state)
        prev_n = torch.roll(state_n, shifts=1, dims=1)
        next_n = torch.roll(state_n, shifts=-1, dims=1)
        tangent = 0.5 * (next_n - prev_n)
        curvature = next_n - 2.0 * state_n + prev_n
        step_idx = torch.full((n,), idx, dtype=torch.long, device=state.device)
        step_feat = self.step_embed(step_idx).to(dtype=state.dtype)
        frac_feat = state.new_full((n, 1), float(frac))
        feat = torch.cat(
            [
                state_n.std(dim=1, unbiased=False),
                can_n.std(dim=1, unbiased=False),
                tangent.norm(dim=-1).mean(dim=1, keepdim=True),
                tangent.norm(dim=-1).std(dim=1, unbiased=False, keepdim=True),
                curvature.norm(dim=-1).mean(dim=1, keepdim=True),
                curvature.norm(dim=-1).std(dim=1, unbiased=False, keepdim=True),
                frac_feat,
                step_feat,
            ],
            dim=-1,
        )
        out = self.mlp(feat)
        mu = out[..., :self.num_coeffs]
        logstd_delta = out[..., self.num_coeffs:]
        logstd = (logstd_delta + self.init_logstd).clamp(self.min_logstd, self.max_logstd)
        return mu, logstd

    def params_to_latent(self, params: torch.Tensor, state: torch.Tensor, c_state: torch.Tensor) -> torch.Tensor:
        del c_state
        n, p, _ = state.shape
        state_n = self._normalize(state)
        prev_n = torch.roll(state_n, shifts=1, dims=1)
        next_n = torch.roll(state_n, shifts=-1, dims=1)
        tangent = torch.nn.functional.normalize(0.5 * (next_n - prev_n), dim=-1, eps=1e-6)
        radial = torch.nn.functional.normalize(state_n, dim=-1, eps=1e-6)
        normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
        phase = torch.linspace(0.0, 2.0 * math.pi, p, dtype=state.dtype, device=state.device).view(1, p, 1)
        sin1 = torch.sin(phase)
        cos1 = torch.cos(phase)
        base = [
            torch.stack([torch.ones((n, p), dtype=state.dtype, device=state.device), torch.zeros((n, p), dtype=state.dtype, device=state.device)], dim=-1),
            torch.stack([torch.zeros((n, p), dtype=state.dtype, device=state.device), torch.ones((n, p), dtype=state.dtype, device=state.device)], dim=-1),
            radial,
            tangent,
            normal,
            radial * sin1,
            radial * cos1,
            normal * sin1,
        ]
        while len(base) < self.num_coeffs:
            harmonic = len(base) - 6
            wave = torch.sin(float(harmonic) * phase) if harmonic % 2 else torch.cos(float(harmonic) * phase)
            base.append(normal * wave)
        basis = torch.stack(base[:self.num_coeffs], dim=2)
        latent = (basis * params[:, None, :, None]).sum(dim=2) / math.sqrt(float(self.num_coeffs))
        return latent


def _reduce_latent_terms(x: torch.Tensor, reduction: str) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    if not dims:
        return x
    if reduction == 'sum':
        return x.sum(dim=dims)
    if reduction == 'mean':
        return x.mean(dim=dims)
    raise ValueError(f'Unsupported latent logprob reduction: {reduction}')


def _latent_logprob(latent: torch.Tensor, mean: torch.Tensor, logstd: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    std = torch.exp(logstd).clamp_min(1e-6)
    var = std.pow(2).clamp_min(1e-12)
    lp = -((latent.detach() - mean) ** 2) / (2.0 * var)
    lp = lp - logstd - 0.5 * math.log(2.0 * math.pi)
    return _reduce_latent_terms(lp, reduction)


def _latent_entropy(logstd: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    ent = logstd + 0.5 * (1.0 + math.log(2.0 * math.pi))
    return _reduce_latent_terms(ent, reduction)


def _latent_kl_to_base(mean: torch.Tensor, logstd: torch.Tensor, base_std: float, reduction: str = 'mean') -> torch.Tensor:
    base_std = max(float(base_std), 1e-6)
    std = torch.exp(logstd).clamp_min(1e-6)
    base_var = base_std ** 2
    kl = torch.log(mean.new_tensor(base_std)) - logstd + (std.pow(2) + mean.pow(2)) / (2.0 * base_var) - 0.5
    return _reduce_latent_terms(kl, reduction)


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
    print('[RL-V4d] Three outer steps with latent noise policy')
    print('=' * 72)

    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True
    cfg.use_flow_matching = True
    cfg.use_grpo = True

    v4_cfg = getattr(cfg, 'rl_v4', None)

    def cv(name, default):
        if name == 'train_steps' and 'RL_V4D_STEPS' in os.environ:
            return _parse_env_value(os.environ['RL_V4D_STEPS'], default)
        if name == 'train_steps' and 'RL_V4_STEPS' in os.environ:
            return _parse_env_value(os.environ['RL_V4_STEPS'], default)
        env_name = f'RL_V4D_{name.upper()}'
        if env_name in os.environ:
            return _parse_env_value(os.environ[env_name], default)
        env_name = f'RL_V4_{name.upper()}'
        if env_name in os.environ:
            return _parse_env_value(os.environ[env_name], default)
        if v4_cfg is not None and name in v4_cfg:
            return v4_cfg[name]
        if hasattr(cfg, f'rl_v4d_{name}'):
            return getattr(cfg, f'rl_v4d_{name}', default)
        return getattr(cfg, f'rl_v4_{name}', default)

    gpu_override = os.environ.get('RL_V4_GPU', '').strip()
    if gpu_override:
        cfg.gpus = [int(gpu_override)]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_override
        print(f'[RL-V4] Override GPU -> {gpu_override}')

    train_steps = int(cv('train_steps', 300))
    k_rollouts = int(cv('k', 8))
    outer_steps = int(cv('outer_steps', 3))
    fractions = [float(x) for x in list(cv('fractions', [0.3333, 0.5, 1.0]))]
    if len(fractions) < outer_steps:
        fractions = fractions + [1.0] * (outer_steps - len(fractions))
    fractions = fractions[:outer_steps]
    ode_steps = int(cv('ode_steps', getattr(cfg, 'iterative_ode_steps', getattr(cfg, 'flow_ode_steps', 10))))
    if ode_steps <= 0:
        ode_steps = int(getattr(cfg, 'flow_ode_steps', 10))
    noise_scale = float(cv('noise_scale', getattr(cfg, 'flow_noise_scale', 1.0)))
    latent_init_std = float(cv('latent_init_std', noise_scale))
    latent_logstd_min = float(cv('latent_logstd_min', math.log(0.20)))
    latent_logstd_max = float(cv('latent_logstd_max', math.log(2.00)))
    latent_policy_hidden_dim = int(cv('latent_policy_hidden_dim', 64))
    latent_policy_type = str(cv('policy_type', 'point')).strip().lower()
    structured_coeffs = int(cv('structured_coeffs', 8))
    logprob_reduction = str(cv('logprob_reduction', 'mean')).strip().lower()
    ppo_inner_epochs = int(cv('ppo_inner_epochs', 2))
    ppo_clip = float(cv('ppo_clip', 0.05))
    ppo_kl_target = float(cv('ppo_kl_target', 0.002))
    latent_kl_beta = float(cv('latent_kl_beta', cv('kl_beta', 0.01)))
    entropy_beta = float(cv('entropy_beta', 0.001))
    adv_clip_max = float(cv('adv_clip_max', 2.0))
    gate_margin = float(cv('gate_margin', 0.0))
    grad_clip_norm = float(cv('grad_clip_norm', 0.3))
    lr = float(cv('lr', getattr(cfg.train, 'lr', 5e-8)))
    policy_lr = float(cv('policy_lr', lr))
    eval_batches_n = int(cv('eval_batches', 4))
    eval_every = int(cv('eval_every', 20))
    save_every = int(cv('save_every', 50))
    log_every = int(cv('log_every', 1))
    seed = int(cv('seed', 20260525))
    freeze_yolo = _as_bool(cv('freeze_yolo', True))
    freeze_flow = _as_bool(cv('freeze_flow', True))
    min_load_ratio = float(cv('min_load_ratio', 95.0))

    reward_w_region = float(cv('reward_w_region', 0.30))
    reward_w_dice = float(cv('reward_w_dice', 0.10))
    reward_w_iou = float(cv('reward_w_iou', 0.25))
    reward_w_dist = float(cv('reward_w_dist', 0.35))
    reward_dist_max_px = float(cv('reward_dist_max_px', 8.0))
    reward_dist_quantile = float(cv('reward_dist_quantile', 95.0))
    reward_dist_quantile_weight = float(cv('reward_dist_quantile_weight', 0.5))

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
    print(f'[RL-V4] ckpt: {ckpt_path} | load ratio: {load_ratio:.2f}% missing={len(missing)} unexpected={len(unexpected)}')
    if load_ratio < min_load_ratio:
        raise RuntimeError(f'load ratio too low: {load_ratio:.2f}%')

    inner = net_for_load.net if hasattr(net_for_load, 'net') else net_for_load
    if freeze_yolo:
        for name in ('yolo', 'cnn_proj', 'cnn_proj_p3', 'swin_snake_feature'):
            _set_requires_grad(getattr(inner, name, None), False)
        print('[RL-V4] froze detector/feature projection parameters.')
    _set_requires_grad(inner.gcn, not freeze_flow)
    inner.train()
    nbn = freeze_bn_running_stats(inner)
    print(f'[RL-V4] froze BN running stats on {nbn} layers.')

    if latent_policy_type in ('structured', 'lowdim', 'low_dim'):
        noise_policy = StructuredLatentNoisePolicy(
            outer_steps=outer_steps,
            init_std=latent_init_std,
            min_logstd=latent_logstd_min,
            max_logstd=latent_logstd_max,
            hidden_dim=latent_policy_hidden_dim,
            num_coeffs=structured_coeffs,
        ).cuda()
        latent_policy_type = 'structured'
    elif latent_policy_type == 'point':
        noise_policy = PointLatentNoisePolicy(
            outer_steps=outer_steps,
            init_std=latent_init_std,
            min_logstd=latent_logstd_min,
            max_logstd=latent_logstd_max,
            hidden_dim=latent_policy_hidden_dim,
        ).cuda()
    else:
        raise ValueError(f'Unsupported rl_v4d_policy_type: {latent_policy_type}')
    if isinstance(raw_ckpt, dict) and isinstance(raw_ckpt.get('noise_policy'), dict):
        noise_policy.load_state_dict(raw_ckpt['noise_policy'], strict=False)
        print('[RL-V4d] loaded latent noise policy from checkpoint.')
    optim_params = list(noise_policy.parameters()) + [p for p in trainer.network.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=policy_lr, weight_decay=0.0)
    print(
        f'[RL-V4d] optimizer LR set to {policy_lr} | freeze_flow={freeze_flow} '
        f'| policy_type={latent_policy_type} | logprob_reduction={logprob_reduction}'
    )

    train_loader = make_data_loader(cfg, is_train=True, is_distributed=False)
    eval_batches = []
    try:
        eval_loader = make_data_loader(cfg, is_train=False, is_distributed=False)
        eval_iter = iter(eval_loader)
        for _ in range(eval_batches_n):
            eb = next(eval_iter)
            _move_batch(eb)
            eval_batches.append(eb)
        print(f'[RL-V4] fixed eval set: {len(eval_batches)} batches')
    except Exception as e:
        print(f'[RL-V4] WARNING: fixed eval unavailable: {e}')

    out_dir = _output_dir()
    ckpt_dir = out_dir / 'checkpoints'
    log_dir = out_dir / 'posttrain_rl_v4_three_iter'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'logs.jsonl'
    with open(log_dir / 'v4_hparams.json', 'w') as f:
        json.dump({
            'cfg_file': _cfg_file_used(),
            'base_ckpt': str(ckpt_path),
            'train_steps': train_steps,
            'k': k_rollouts,
            'outer_steps': outer_steps,
            'fractions': fractions,
            'ode_steps': ode_steps,
            'noise_scale': noise_scale,
            'latent_init_std': latent_init_std,
            'latent_logstd_min': latent_logstd_min,
            'latent_logstd_max': latent_logstd_max,
            'latent_policy_hidden_dim': latent_policy_hidden_dim,
            'latent_policy_type': latent_policy_type,
            'structured_coeffs': structured_coeffs,
            'logprob_reduction': logprob_reduction,
            'freeze_flow': freeze_flow,
            'policy_lr': policy_lr,
            'ppo_inner_epochs': ppo_inner_epochs,
            'ppo_clip': ppo_clip,
            'ppo_kl_target': ppo_kl_target,
            'latent_kl_beta': latent_kl_beta,
            'entropy_beta': entropy_beta,
            'gate_margin': gate_margin,
            'reward_weights': {
                'region': reward_w_region,
                'dice': reward_w_dice,
                'iou': reward_w_iou,
                'dist': reward_w_dist,
            },
        }, f, indent=2)

    gcn = inner.gcn
    ema_reward = EMA(decay=0.95)
    best_eval_iou = -1.0
    start_step = 1

    resume_state = _as_bool(os.environ.get('RL_V4_RESUME_STATE', False))
    if resume_state and isinstance(raw_ckpt, dict):
        resume_step = int(raw_ckpt.get('step', 0) or 0)
        resume_metrics = raw_ckpt.get('metrics') if isinstance(raw_ckpt.get('metrics'), dict) else {}
        opt_state = raw_ckpt.get('optimizer')
        if resume_step > 0:
            start_step = resume_step + 1
            eval_iou = resume_metrics.get('eval_iou')
            if eval_iou is not None:
                try:
                    best_eval_iou = float(eval_iou)
                except Exception:
                    pass
            if isinstance(opt_state, dict):
                try:
                    optimizer.load_state_dict(opt_state)
                    for group in optimizer.param_groups:
                        group['lr'] = policy_lr
                    print(f'[RL-V4d] resumed optimizer/state from step {resume_step}')
                except Exception as e:
                    print(f'[RL-V4d] WARNING: failed to resume optimizer state: {e}')
            print(f'[RL-V4d] resume mode on: start_step={start_step} train_steps={train_steps}')

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

    def _quality_score(poly, gt, image_hw):
        return compute_region_score(
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

    @torch.no_grad()
    def _deterministic_three_step(output, flow):
        current = output['i_it_py'].detach()
        total_disp = torch.zeros_like(current)
        for frac in fractions:
            c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
            action = _outer_action_mean(
                flow, output['cnn_feature'], current, c_cur, output['py_ind'], float(frac), ode_steps
            )
            current = (current + action).detach()
            total_disp = total_disp + action.detach()
        return {'disp': total_disp, 'py': output['i_it_py'] + total_disp}

    @torch.no_grad()
    def _sample_rollout(output):
        current = output['i_it_py'].detach()
        total_disp = torch.zeros_like(current)
        traj = {
            'states': [],
            'c_states': [],
            'latents': [],
            'actions': [],
            'old_logs': [],
            'fractions': [],
        }
        for step_idx, frac in enumerate(fractions):
            c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
            mean_z, logstd_z = noise_policy(step_idx, current, c_cur, frac)
            latent = mean_z + torch.exp(logstd_z) * torch.randn_like(mean_z)
            latent_field = noise_policy.params_to_latent(latent, current, c_cur) if hasattr(noise_policy, 'params_to_latent') else latent
            raw_disp = _flow_disp_from_latent(gcn, output['cnn_feature'], current, c_cur, output['py_ind'], latent_field, ode_steps)
            action = raw_disp * float(frac)
            old_log = _latent_logprob(latent, mean_z, logstd_z, logprob_reduction)
            traj['states'].append(current.detach())
            traj['c_states'].append(c_cur.detach())
            traj['latents'].append(latent.detach())
            traj['actions'].append(action.detach())
            traj['old_logs'].append(old_log.detach())
            traj['fractions'].append(float(frac))
            current = (current + action).detach()
            total_disp = total_disp + action.detach()
        traj['disp'] = total_disp.detach()
        traj['py'] = output['i_it_py'] + total_disp
        return traj

    @torch.no_grad()
    def _policy_three_step(output):
        current = output['i_it_py'].detach()
        total_disp = torch.zeros_like(current)
        for step_idx, frac in enumerate(fractions):
            c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
            mean_z, _ = noise_policy(step_idx, current, c_cur, frac)
            latent_field = noise_policy.params_to_latent(mean_z, current, c_cur) if hasattr(noise_policy, 'params_to_latent') else mean_z
            raw_disp = _flow_disp_from_latent(gcn, output['cnn_feature'], current, c_cur, output['py_ind'], latent_field, ode_steps)
            action = raw_disp * float(frac)
            current = (current + action).detach()
            total_disp = total_disp + action.detach()
        return {'disp': total_disp, 'py': output['i_it_py'] + total_disp}

    @torch.no_grad()
    def _compute_eval():
        if not eval_batches:
            return {}
        vals_iou, vals_dice, vals_mbf = [], [], []
        for eb in eval_batches:
            out = _manual_context(eb)
            if out['i_it_py'].numel() == 0:
                continue
            det = _policy_three_step(out)
            pred = out['i_it_py'] + det['disp']
            gt = out['i_gt_py']
            hw = out['image_hw']
            vals_iou.append(compute_region_score(pred, gt, H=hw[0], W=hw[1], w_boundary=0, w_dice=0, w_iou=1,
                                                 coord_scale=float(snake_config.down_ratio)))
            vals_dice.append(compute_region_score(pred, gt, H=hw[0], W=hw[1], w_boundary=0, w_dice=1, w_iou=0,
                                                  coord_scale=float(snake_config.down_ratio)))
            vals_mbf.append(compute_region_score(pred, gt, H=hw[0], W=hw[1], w_boundary=1, w_dice=0, w_iou=0,
                                                 coord_scale=float(snake_config.down_ratio)))
        if not vals_iou:
            return {}
        iou = torch.cat(vals_iou)
        dice = torch.cat(vals_dice)
        mbf = torch.cat(vals_mbf)
        return {
            'eval_iou': float(iou.mean().item()),
            'eval_dice': float(dice.mean().item()),
            'eval_mboundf': float(mbf.mean().item()),
            'eval_n': int(iou.numel()),
        }

    @torch.no_grad()
    def _compute_full_eval():
        save_dir = _project_path(os.environ.get('SAVE_DIR', f'visual/{_cfg_stem()}_v4d_eval'))
        save_dir.mkdir(parents=True, exist_ok=True)
        max_samples_env = os.environ.get('MAX_SAMPLES', '').strip()
        max_samples = int(max_samples_env) if max_samples_env else None
        full_loader = make_data_loader(cfg, is_train=False, is_distributed=False)
        dataset_size = len(full_loader.dataset) if hasattr(full_loader, 'dataset') else -1
        vals_iou, vals_dice, vals_mbf = [], [], []
        rows, failed = [], []
        for index, batch in enumerate(full_loader):
            if max_samples is not None and index >= max_samples:
                break
            try:
                _move_batch(batch)
                out = _manual_context(batch)
                if out['i_it_py'].numel() == 0:
                    raise RuntimeError('empty contour batch')
                det = _policy_three_step(out)
                pred = out['i_it_py'] + det['disp']
                gt = out['i_gt_py']
                hw = out['image_hw']
                iou = compute_region_score(pred, gt, H=hw[0], W=hw[1], w_boundary=0, w_dice=0, w_iou=1,
                                           coord_scale=float(snake_config.down_ratio))
                dice = compute_region_score(pred, gt, H=hw[0], W=hw[1], w_boundary=0, w_dice=1, w_iou=0,
                                            coord_scale=float(snake_config.down_ratio))
                mbf = compute_region_score(pred, gt, H=hw[0], W=hw[1], w_boundary=1, w_dice=0, w_iou=0,
                                           coord_scale=float(snake_config.down_ratio))
                vals_iou.append(iou.detach())
                vals_dice.append(dice.detach())
                vals_mbf.append(mbf.detach())
                rows.append({
                    'index': int(index),
                    'ok': True,
                    'mean_iou': float(iou.mean().item()),
                    'mean_dice': float(dice.mean().item()),
                    'mean_mboundf': float(mbf.mean().item()),
                })
                print(f'[{index + 1}] sample {index} iou={rows[-1]["mean_iou"]:.6f}', flush=True)
            except Exception as e:
                failed.append(int(index))
                rows.append({'index': int(index), 'ok': False, 'error': str(e)})
                print(f'[{index + 1}] sample {index} failed: {e}', flush=True)
        if vals_iou:
            iou_all = torch.cat(vals_iou)
            dice_all = torch.cat(vals_dice)
            mbf_all = torch.cat(vals_mbf)
            sample_ious = [r['mean_iou'] for r in rows if r.get('ok')]
            summary = {
                'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                'cfg_file': _cfg_file_used(),
                'ckpt': str(ckpt_path),
                'dataset': str(getattr(cfg.test, 'dataset', '')),
                'dataset_size': int(dataset_size),
                'evaluated_samples': int(len(sample_ious)),
                'failed_samples': int(len(failed)),
                'failed_indices': failed,
                'mean_iou_sample_avg': float(np.mean(sample_ious)) if sample_ious else 0.0,
                'mean_iou_contour_avg': float(iou_all.mean().item()),
                'mean_dice_sample_avg': float(np.mean([r['mean_dice'] for r in rows if r.get('ok')])) if sample_ious else 0.0,
                'mean_dice_contour_avg': float(dice_all.mean().item()),
                'mean_mboundf_sample_avg': float(np.mean([r['mean_mboundf'] for r in rows if r.get('ok')])) if sample_ious else 0.0,
                'mean_mboundf_contour_avg': float(mbf_all.mean().item()),
                'median_iou_sample_avg': float(np.median(sample_ious)) if sample_ious else 0.0,
                'std_iou_sample_avg': float(np.std(sample_ious)) if sample_ious else 0.0,
            }
        else:
            summary = {
                'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                'cfg_file': _cfg_file_used(),
                'ckpt': str(ckpt_path),
                'dataset_size': int(dataset_size),
                'evaluated_samples': 0,
                'failed_samples': int(len(failed)),
                'failed_indices': failed,
            }
        ts = summary['timestamp']
        summary_path = save_dir / f'v4d_full_eval_{ts}.json'
        rows_path = save_dir / f'v4d_full_eval_rows_{ts}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        with open(rows_path, 'w') as f:
            json.dump(rows, f, indent=2)
        print('=' * 80)
        print(f'Saved summary: {summary_path}')
        print(f'Saved rows:    {rows_path}')
        for k in ('mean_iou_sample_avg', 'mean_iou_contour_avg', 'mean_dice_sample_avg', 'mean_mboundf_sample_avg',
                  'median_iou_sample_avg', 'std_iou_sample_avg', 'failed_samples'):
            if k in summary:
                print(f'{k}: {summary[k]}')
        print('=' * 80)
        return summary

    def _save_checkpoint(path: Path, step: int, metrics: Dict):
        _safe_torch_save({
            'state_dict': net_for_load.state_dict(),
            'noise_policy': noise_policy.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': int(step),
            'metrics': metrics,
            'cfg_file': _cfg_file_used(),
            'time': datetime.datetime.now().isoformat(),
        }, path)

    if start_step > train_steps:
        print(f'[RL-V4] resume start_step={start_step} exceeds train_steps={train_steps}, nothing to do.')
        return

    if _as_bool(os.environ.get('RL_V4D_EVAL_ONLY', False)):
        _compute_full_eval()
        return

    train_iter = iter(train_loader)
    for step in range(start_step, train_steps + 1):
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
            print(f'[RL-V4] step {step}: empty contour batch, skipping.')
            continue

        det_ret = _deterministic_three_step(output, gcn)
        baseline_score = _quality_score(output['i_it_py'] + det_ret['disp'], i_gt, output['image_hw']).detach()

        rollouts: List[Dict] = []
        final_scores = []
        old_log_counts = []
        for _ in range(k_rollouts):
            ret = _sample_rollout(output)
            rollouts.append(ret)
            final_scores.append(_quality_score(output['i_it_py'] + ret['disp'], i_gt, output['image_hw']).detach())
            old_log_counts.append(len(ret['old_logs']))

        final_scores_t = torch.stack(final_scores, dim=0).to(i_init.device)
        quality = final_scores_t - baseline_score.unsqueeze(0)
        gate = (quality.max(dim=0, keepdim=True).values > gate_margin).float()
        adv = quality / quality.std(dim=0, unbiased=False, keepdim=True).clamp_min(0.1)
        adv = adv.clamp(-adv_clip_max, adv_clip_max) * gate
        reward_mean = float(quality.mean().item())
        ema_reward_val = ema_reward.update(reward_mean)
        gate_active_frac = float(gate.mean().item())

        total_actions = sum(len(t['actions']) for t in rollouts)
        approx_kl_hist, ratio_hist, loss_hist, entropy_hist, latent_kl_hist = [], [], [], [], []
        early_stop_epoch = ppo_inner_epochs

        for epoch in range(ppo_inner_epochs):
            optimizer.zero_grad(set_to_none=True)
            epoch_losses = []
            epoch_kls = []
            epoch_ratios = []
            for ri, traj in enumerate(rollouts):
                adv_ri = adv[ri].detach()
                for si, latent in enumerate(traj['latents']):
                    state = traj['states'][si]
                    old_log = traj['old_logs'][si]
                    c_state = traj['c_states'][si]
                    frac = traj['fractions'][si]
                    mean_cur, logstd_cur = noise_policy(si, state, c_state, frac)
                    lp_cur = _latent_logprob(latent, mean_cur, logstd_cur, logprob_reduction)
                    ratio = torch.exp(lp_cur - old_log)
                    unclipped = -adv_ri * ratio
                    clipped = -adv_ri * torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
                    policy_loss = torch.maximum(unclipped, clipped).mean() / max(total_actions, 1)
                    entropy = _latent_entropy(logstd_cur, logprob_reduction).mean() / max(total_actions, 1)
                    latent_kl = _latent_kl_to_base(mean_cur, logstd_cur, latent_init_std, logprob_reduction).mean() / max(total_actions, 1)
                    loss = policy_loss + latent_kl_beta * latent_kl - entropy_beta * entropy
                    if loss.requires_grad:
                        loss.backward()
                    with torch.no_grad():
                        epoch_losses.append(float(policy_loss.detach().item()))
                        epoch_kls.append(float(0.5 * torch.mean((lp_cur - old_log) ** 2).item()))
                        epoch_ratios.append(ratio.detach())
                        entropy_hist.append(float(entropy.detach().item()))
                        latent_kl_hist.append(float(latent_kl.detach().item()))

            grad_norm = torch.nn.utils.clip_grad_norm_(optim_params, max_norm=grad_clip_norm)
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
            'gate_active_frac': gate_active_frac,
            'outer_log_count_mean': float(np.mean(old_log_counts)) if old_log_counts else 0.0,
            'approx_kl': float(np.mean(approx_kl_hist)) if approx_kl_hist else 0.0,
            'latent_entropy': float(np.mean(entropy_hist)) if entropy_hist else 0.0,
            'latent_kl': float(np.mean(latent_kl_hist)) if latent_kl_hist else 0.0,
            'latent_param_abs': float(np.mean([p.detach().abs().mean().item() for p in noise_policy.parameters()])),
            'policy_loss': float(np.mean(loss_hist)) if loss_hist else 0.0,
            'grad_norm': float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
            'early_stop_epoch': int(early_stop_epoch),
            'latent_init_std': latent_init_std,
            'lr': policy_lr,
        }
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
                f"[RL-V4] step={step}/{train_steps} reward={reward_mean:+.5f} "
                f"best={metrics['quality_best_mean']:+.5f} gate={gate_active_frac:.2f} "
                f"kl={metrics['approx_kl']:.6f} logs={metrics['outer_log_count_mean']:.1f}{extra}",
                flush=True,
            )

        if step == 1 or (save_every > 0 and step % save_every == 0):
            _save_checkpoint(ckpt_dir / 'latest.pt', step, metrics)
            if save_every > 0 and step % save_every == 0:
                _save_checkpoint(ckpt_dir / f'step{step}.pt', step, metrics)

        del batch, output, rollouts, final_scores_t, quality, adv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_eval = _compute_eval()
    final_metrics = {'final_eval': final_eval, 'step': train_steps}
    _save_checkpoint(ckpt_dir / 'latest.pt', train_steps, final_metrics)
    print(f'[RL-V4] done. output={out_dir}')


if __name__ == '__main__':
    main()
