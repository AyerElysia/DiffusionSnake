"""RL V7: SeedFlow-GRPO, three-step RL with trainable initialization-noise policy."""

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
import cv2

from lib.config import cfg, args
from lib.datasets import make_data_loader
from lib.networks import make_network
from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
from lib.train.grpo_v2_utils import EMA, freeze_bn_running_stats, percentiles
from lib.train.rewards.curvature_detail_reward import compute_curvature_detail_score
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
    env_md = str(os.environ.get('RL_V7_MODEL_DIR', '') or os.environ.get('RL_V4_MODEL_DIR', '') or '').strip()
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


class SeedFlowPolicy(torch.nn.Module):
    """State-conditioned Gaussian policy for the Flow initialization latent."""

    def __init__(
        self,
        outer_steps: int,
        init_stds,
        hidden_dim: int,
        mean_scale: float,
        min_logstd: float,
        max_logstd: float,
    ):
        super().__init__()
        outer_steps = max(int(outer_steps), 1)
        hidden_dim = max(int(hidden_dim), 8)
        if isinstance(init_stds, (int, float)):
            stds = [float(init_stds)] * outer_steps
        else:
            stds = [float(x) for x in list(init_stds)]
            if not stds:
                stds = [1.0]
            if len(stds) < outer_steps:
                stds = stds + [stds[-1]] * (outer_steps - len(stds))
            stds = stds[:outer_steps]
        std_t = torch.tensor(stds, dtype=torch.float32).clamp_min(1e-6).view(outer_steps, 1, 1, 1)
        self.step_embed = torch.nn.Embedding(outer_steps, hidden_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4 + hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 2),
        )
        self.logstd = torch.nn.Parameter(std_t.log().expand(outer_steps, 1, 1, 2).clone())
        self.mean_scale = float(mean_scale)
        self.min_logstd = float(min_logstd)
        self.max_logstd = float(max_logstd)
        torch.nn.init.zeros_(self.step_embed.weight)
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, step_index: int, state: torch.Tensor, c_state: Optional[torch.Tensor] = None):
        idx = max(0, min(int(step_index), self.logstd.size(0) - 1))
        if c_state is None:
            c_state = snake_gcn_utils.img_poly_to_can_poly(state)
        center = state.mean(dim=1, keepdim=True)
        scale = (state - center).norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True).clamp_min(1.0)
        state_rel = (state - center) / scale
        step_id = torch.full((state.size(0), state.size(1)), idx, device=state.device, dtype=torch.long)
        step_feat = self.step_embed(step_id).to(dtype=state.dtype)
        inp = torch.cat([state_rel, c_state.to(dtype=state.dtype), step_feat], dim=-1)
        mean = self.net(inp) * self.mean_scale
        logstd = self.logstd[idx].clamp(self.min_logstd, self.max_logstd)
        logstd = logstd.to(device=state.device, dtype=state.dtype).expand_as(state)
        return mean, logstd


def _latent_logprob(latent: torch.Tensor, mean: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    std = torch.exp(logstd).clamp_min(1e-6)
    var = std.pow(2).clamp_min(1e-12)
    lp = -((latent.detach() - mean) ** 2) / (2.0 * var)
    lp = lp - logstd - 0.5 * math.log(2.0 * math.pi)
    return lp.mean(dim=tuple(range(1, lp.ndim)))


def _latent_entropy(logstd: torch.Tensor) -> torch.Tensor:
    ent = logstd + 0.5 * (1.0 + math.log(2.0 * math.pi))
    return ent.mean(dim=tuple(range(1, ent.ndim)))


def _latent_kl_to_base(mean: torch.Tensor, logstd: torch.Tensor, base_std: float) -> torch.Tensor:
    base_std = max(float(base_std), 1e-6)
    std = torch.exp(logstd).clamp_min(1e-6)
    base_var = base_std ** 2
    kl = torch.log(mean.new_tensor(base_std)) - logstd + (std.pow(2) + mean.pow(2)) / (2.0 * base_var) - 0.5
    return kl.mean(dim=tuple(range(1, kl.ndim)))


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
    print('[RL-V7] SeedFlow-GRPO: three-step initialization-noise policy')
    print('=' * 72)

    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True
    cfg.use_flow_matching = True
    cfg.use_grpo = True

    v7_cfg = getattr(cfg, 'rl_v7', None)
    v4_cfg = getattr(cfg, 'rl_v4', None)

    def cv(name, default):
        if name == 'train_steps' and 'RL_V7_STEPS' in os.environ:
            return _parse_env_value(os.environ['RL_V7_STEPS'], default)
        if name == 'train_steps' and 'RL_V4_STEPS' in os.environ:
            return _parse_env_value(os.environ['RL_V4_STEPS'], default)
        env_name = f'RL_V7_{name.upper()}'
        if env_name in os.environ:
            return _parse_env_value(os.environ[env_name], default)
        env_name = f'RL_V4_{name.upper()}'
        if env_name in os.environ:
            return _parse_env_value(os.environ[env_name], default)
        if v7_cfg is not None and name in v7_cfg:
            return v7_cfg[name]
        if v4_cfg is not None and name in v4_cfg:
            return v4_cfg[name]
        if hasattr(cfg, f'rl_v7_{name}'):
            return getattr(cfg, f'rl_v7_{name}', default)
        return getattr(cfg, f'rl_v4_{name}', default)

    gpu_override = os.environ.get('RL_V7_GPU', '').strip() or os.environ.get('RL_V4_GPU', '').strip()
    if gpu_override:
        cfg.gpus = [int(gpu_override)]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_override
        print(f'[RL-V7] Override GPU -> {gpu_override}')

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
    method_name = str(cv('name', 'SeedFlow-GRPO'))
    seed_std_cfg = cv('seed_std', cv('latent_std', [1.0, 0.7, 0.4]))
    if isinstance(seed_std_cfg, (int, float)):
        seed_std = [float(seed_std_cfg)] * outer_steps
    else:
        seed_std = [float(x) for x in list(seed_std_cfg)]
    if len(seed_std) < outer_steps:
        seed_std = seed_std + [seed_std[-1]] * (outer_steps - len(seed_std))
    seed_std = seed_std[:outer_steps]
    seed_policy_hidden_dim = int(cv('seed_policy_hidden_dim', 64))
    seed_policy_mean_scale = float(cv('seed_policy_mean_scale', 0.25))
    seed_logstd_min = float(cv('seed_logstd_min', math.log(0.20)))
    seed_logstd_max = float(cv('seed_logstd_max', math.log(2.00)))
    train_seed_policy = _as_bool(cv('train_seed_policy', True))
    ppo_inner_epochs = int(cv('ppo_inner_epochs', 2))
    ppo_clip = float(cv('ppo_clip', 0.05))
    ppo_kl_target = float(cv('ppo_kl_target', 0.002))
    latent_kl_beta = float(cv('latent_kl_beta', cv('kl_beta', 0.01)))
    entropy_beta = float(cv('entropy_beta', 0.001))
    flow_update_weight = float(cv('flow_update_weight', 0.0))
    adv_clip_max = float(cv('adv_clip_max', 2.0))
    gate_margin = float(cv('gate_margin', 0.0))
    grad_clip_norm = float(cv('grad_clip_norm', 0.3))
    lr = float(cv('lr', getattr(cfg.train, 'lr', 5e-8)))
    policy_lr = float(cv('policy_lr', lr))
    flow_lr = float(cv('flow_lr', lr))
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
    reward_burr_weight = float(cv('reward_burr_weight', 0.0))
    reward_burr_max_px = float(cv('reward_burr_max_px', 1.5))
    reward_burr_margin_px = float(cv('reward_burr_margin_px', 0.5))
    reward_burr_quantile = float(cv('reward_burr_quantile', 95.0))
    reward_global_weight = float(cv('reward_global_weight', 1.0))
    reward_detail_weight = float(cv('reward_detail_weight', 0.0))
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
    print(f'[RL-V7] ckpt: {ckpt_path} | load ratio: {load_ratio:.2f}% missing={len(missing)} unexpected={len(unexpected)}')
    if load_ratio < min_load_ratio:
        raise RuntimeError(f'load ratio too low: {load_ratio:.2f}%')

    inner = net_for_load.net if hasattr(net_for_load, 'net') else net_for_load
    if freeze_yolo:
        for name in ('yolo', 'cnn_proj', 'cnn_proj_p3', 'swin_snake_feature'):
            _set_requires_grad(getattr(inner, name, None), False)
        print('[RL-V7] froze detector/feature projection parameters.')
    _set_requires_grad(inner.gcn, not freeze_flow)
    inner.train()
    nbn = freeze_bn_running_stats(inner)
    print(f'[RL-V7] froze BN running stats on {nbn} layers.')

    seed_policy = SeedFlowPolicy(
        outer_steps=outer_steps,
        init_stds=seed_std,
        hidden_dim=seed_policy_hidden_dim,
        mean_scale=seed_policy_mean_scale,
        min_logstd=seed_logstd_min,
        max_logstd=seed_logstd_max,
    ).cuda()
    if isinstance(raw_ckpt, dict) and isinstance(raw_ckpt.get('seed_policy'), dict):
        seed_policy.load_state_dict(raw_ckpt['seed_policy'], strict=False)
        print('[RL-V7] loaded SeedFlow policy from checkpoint.')
    if not train_seed_policy:
        _set_requires_grad(seed_policy, False)
        print('[RL-V7] fixed random seed mode: seed policy frozen at zero-mean initial noise.')
    seed_params = [p for p in seed_policy.parameters() if p.requires_grad]
    flow_params = [p for p in trainer.network.parameters() if p.requires_grad]
    optim_params = seed_params + flow_params
    param_groups = []
    if seed_params:
        param_groups.append({'params': seed_params, 'lr': policy_lr, 'name': 'seed_policy'})
    if flow_params:
        param_groups.append({'params': flow_params, 'lr': flow_lr, 'name': 'flow'})
    if not param_groups:
        raise RuntimeError('No trainable parameters: enable train_seed_policy or unfreeze Flow.')
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
    print(
        f'[RL-V7] optimizer LR seed={policy_lr if seed_params else 0.0} '
        f'flow={flow_lr if flow_params else 0.0} | freeze_flow={freeze_flow} '
        f'train_seed_policy={train_seed_policy}'
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
        print(f'[RL-V7] fixed eval set: {len(eval_batches)} batches')
    except Exception as e:
        print(f'[RL-V7] WARNING: fixed eval unavailable: {e}')

    out_dir = _output_dir()
    ckpt_dir = out_dir / 'checkpoints'
    log_dir = out_dir / 'posttrain_rl_v7_seedflow_grpo'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'logs.jsonl'
    with open(log_dir / 'v7_hparams.json', 'w') as f:
        json.dump({
            'method': method_name,
            'cfg_file': _cfg_file_used(),
            'base_ckpt': str(ckpt_path),
            'train_steps': train_steps,
            'k': k_rollouts,
            'outer_steps': outer_steps,
            'fractions': fractions,
            'ode_steps': ode_steps,
            'action_policy': 'state_conditioned_initial_latent_noise',
            'seed_std': seed_std,
            'seed_policy_hidden_dim': seed_policy_hidden_dim,
            'seed_policy_mean_scale': seed_policy_mean_scale,
            'seed_logstd_min': seed_logstd_min,
            'seed_logstd_max': seed_logstd_max,
            'train_seed_policy': train_seed_policy,
            'freeze_flow': freeze_flow,
            'flow_lr': flow_lr,
            'ppo_inner_epochs': ppo_inner_epochs,
            'ppo_clip': ppo_clip,
            'ppo_kl_target': ppo_kl_target,
            'latent_kl_beta': latent_kl_beta,
            'entropy_beta': entropy_beta,
            'flow_update_weight': flow_update_weight,
            'gate_margin': gate_margin,
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
            'polys': [current.detach()],
        }
        for si, frac in enumerate(fractions):
            c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
            mean_z, logstd_z = seed_policy(si, current, c_cur)
            latent = mean_z + torch.exp(logstd_z) * torch.randn_like(current)
            raw_disp = _flow_disp_from_latent(
                gcn, output['cnn_feature'], current, c_cur, output['py_ind'], latent, ode_steps
            )
            action = raw_disp * float(frac)
            old_log = _latent_logprob(latent, mean_z, logstd_z)
            traj['states'].append(current.detach())
            traj['c_states'].append(c_cur.detach())
            traj['latents'].append(latent.detach())
            traj['actions'].append(action.detach())
            traj['old_logs'].append(old_log.detach())
            traj['fractions'].append(float(frac))
            current = (current + action).detach()
            total_disp = total_disp + action.detach()
            traj['polys'].append(current.detach())
        traj['disp'] = total_disp.detach()
        traj['py'] = output['i_it_py'] + total_disp
        return traj

    @torch.no_grad()
    def _policy_three_step(output):
        current = output['i_it_py'].detach()
        total_disp = torch.zeros_like(current)
        for si, frac in enumerate(fractions):
            c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
            mean_z, _ = seed_policy(si, current, c_cur)
            raw_disp = _flow_disp_from_latent(
                gcn, output['cnn_feature'], current, c_cur, output['py_ind'], mean_z, ode_steps
            )
            action = raw_disp * float(frac)
            current = (current + action).detach()
            total_disp = total_disp + action.detach()
        return {'disp': total_disp, 'py': output['i_it_py'] + total_disp}

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

    def _save_checkpoint(path: Path, step: int, metrics: Dict):
        _safe_torch_save({
            'state_dict': net_for_load.state_dict(),
            'seed_policy': seed_policy.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': int(step),
            'metrics': metrics,
            'cfg_file': _cfg_file_used(),
            'time': datetime.datetime.now().isoformat(),
        }, path)

    @torch.no_grad()
    def _dump_group_viz(batch, output, rollouts: List[Dict], quality: torch.Tensor,
                        baseline_score: torch.Tensor, step: int):
        if viz_every <= 0:
            return
        viz_dir = _THIS_DIR / 'visual' / f'rl_v7_{_cfg_stem()}'
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

        meta = {
            'step': int(step),
            'type': 'v7_seedflow_grpo_group',
            'k_total': int(len(rollouts)),
            'k_shown': [int(x) for x in order],
            'best': int(best_idx),
            'worst': int(worst_idx),
            'quality_mean': [float(q_np[i].mean()) for i in range(len(rollouts))],
        }
        with open(viz_dir / f'train_group_step{step:06d}.json', 'w') as f:
            json.dump(meta, f, indent=2)

    train_iter = iter(train_loader)
    for step in range(1, train_steps + 1):
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
            print(f'[RL-V7] step {step}: empty contour batch, skipping.')
            continue

        det_ret = _deterministic_three_step(output, gcn)
        baseline_score, baseline_comps = _score_with_components(
            output['i_it_py'] + det_ret['disp'], i_gt, output['image_hw']
        )
        baseline_score = baseline_score.detach()
        baseline_comps = {k: v.detach() for k, v in baseline_comps.items()}

        rollouts: List[Dict] = []
        final_scores = []
        final_scores_reward = []
        final_score_components: List[Dict[str, torch.Tensor]] = []
        burr_penalties = []
        burr_raws = []
        old_log_counts = []
        for _ in range(k_rollouts):
            ret = _sample_rollout(output)
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
            final_scores.append(score)
            final_score_components.append({k: v.detach() for k, v in score_comps.items()})
            final_scores_reward.append((score - reward_burr_weight * burr_penalty.detach()).detach())
            burr_penalties.append(burr_penalty.detach())
            burr_raws.append(burr_raw.detach())
            old_log_counts.append(len(ret['old_logs']))

        final_scores_t = torch.stack(final_scores, dim=0).to(i_init.device)
        final_scores_reward_t = torch.stack(final_scores_reward, dim=0).to(i_init.device)
        burr_penalty_t = torch.stack(burr_penalties, dim=0).to(i_init.device)
        burr_raw_t = torch.stack(burr_raws, dim=0).to(i_init.device)
        quality = final_scores_reward_t - baseline_score.unsqueeze(0)
        gate = (quality.max(dim=0, keepdim=True).values > gate_margin).float()
        adv = quality / quality.std(dim=0, unbiased=False, keepdim=True).clamp_min(0.1)
        adv = adv.clamp(-adv_clip_max, adv_clip_max) * gate
        reward_mean = float(quality.mean().item())
        ema_reward_val = ema_reward.update(reward_mean)
        gate_active_frac = float(gate.mean().item())

        total_actions = sum(len(t['actions']) for t in rollouts)
        approx_kl_hist, ratio_hist, loss_hist, entropy_hist, latent_kl_hist, flow_loss_hist = [], [], [], [], [], []
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
                    c_state = traj['c_states'][si]
                    old_log = traj['old_logs'][si]
                    mean_cur, logstd_cur = seed_policy(si, state, c_state)
                    lp_cur = _latent_logprob(latent, mean_cur, logstd_cur)
                    ratio = torch.exp(lp_cur - old_log)
                    unclipped = -adv_ri * ratio
                    clipped = -adv_ri * torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
                    policy_loss = torch.maximum(unclipped, clipped).mean() / max(total_actions, 1)
                    entropy = _latent_entropy(logstd_cur).mean() / max(total_actions, 1)
                    latent_kl = _latent_kl_to_base(mean_cur, logstd_cur, seed_std[si]).mean() / max(total_actions, 1)
                    loss = policy_loss + latent_kl_beta * latent_kl - entropy_beta * entropy
                    flow_loss = latent.new_tensor(0.0)
                    if flow_update_weight > 0 and not freeze_flow:
                        frac = float(traj['fractions'][si])
                        raw_disp_cur = _flow_disp_from_latent(
                            gcn,
                            output['cnn_feature'],
                            state,
                            c_state,
                            output['py_ind'],
                            latent,
                            ode_steps,
                        )
                        target_raw_disp = (i_gt - state).detach()
                        per_contour = torch.nn.functional.smooth_l1_loss(
                            raw_disp_cur * frac,
                            target_raw_disp * frac,
                            reduction='none',
                        ).mean(dim=(1, 2))
                        pos_w = torch.relu(adv_ri.detach())
                        if bool((pos_w > 0).any().item()):
                            flow_loss = (per_contour * pos_w).sum() / pos_w.sum().clamp_min(1e-6)
                            flow_loss = flow_loss / max(total_actions, 1)
                            loss = loss + flow_update_weight * flow_loss
                    if loss.requires_grad:
                        loss.backward()
                    with torch.no_grad():
                        epoch_losses.append(float(policy_loss.detach().item()))
                        epoch_kls.append(float(0.5 * torch.mean((lp_cur - old_log) ** 2).item()))
                        epoch_ratios.append(ratio.detach())
                        entropy_hist.append(float(entropy.detach().item()))
                        latent_kl_hist.append(float(latent_kl.detach().item()))
                        flow_loss_hist.append(float(flow_loss.detach().item()))

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
            'method': method_name,
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
            'gate_active_frac': gate_active_frac,
            'outer_log_count_mean': float(np.mean(old_log_counts)) if old_log_counts else 0.0,
            'approx_kl': float(np.mean(approx_kl_hist)) if approx_kl_hist else 0.0,
            'latent_entropy': float(np.mean(entropy_hist)) if entropy_hist else 0.0,
            'latent_kl': float(np.mean(latent_kl_hist)) if latent_kl_hist else 0.0,
            'flow_update_loss': float(np.mean(flow_loss_hist)) if flow_loss_hist else 0.0,
            'flow_update_weight': float(flow_update_weight),
            'latent_mu_abs': float(np.mean([
                seed_policy(si, i_init, output['c_it_py'])[0].detach().abs().mean().item()
                for si in range(outer_steps)
            ])),
            'latent_std_mean': float(torch.exp(seed_policy.logstd.detach()).mean().item()),
            'policy_loss': float(np.mean(loss_hist)) if loss_hist else 0.0,
            'grad_norm': float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
            'early_stop_epoch': int(early_stop_epoch),
            'seed_std': [float(x) for x in seed_std],
            'lr': policy_lr,
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
                f"[RL-V7] step={step}/{train_steps} reward={reward_mean:+.5f} "
                f"best={metrics['quality_best_mean']:+.5f} gate={gate_active_frac:.2f} "
                f"burr={metrics['burr_penalty_mean']:.3f} detail={metrics.get('final_detail_score_mean', 0.0):+.3f} "
                f"kl={metrics['approx_kl']:.6f} seed_mu={metrics['latent_mu_abs']:.4f} "
                f"seed_std={metrics['latent_std_mean']:.3f} flow_loss={metrics['flow_update_loss']:.5f} "
                f"logs={metrics['outer_log_count_mean']:.1f}{extra}",
                flush=True,
            )

        if viz_every > 0 and (step == 1 or step % viz_every == 0):
            try:
                _dump_group_viz(batch, output, rollouts, quality, baseline_score, step)
            except Exception as e:
                print(f'[RL-V7] viz failed at step {step}: {e}', flush=True)

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
    print(f'[RL-V7] done. output={out_dir}')


if __name__ == '__main__':
    main()
