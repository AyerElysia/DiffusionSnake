"""GRPO V2: stable PPO post-training for flow-matching contour snake.

Key differences vs grpo_train.py / DiffusionGRPONetworkWrapper._flow_grpo_loss:

1) **Real old-policy snapshot.** old_log_probs are captured at rollout time and
   used for `ratio = exp(lp_cur - old_log)` across N inner PPO epochs. v1 had
   ratio ≡ 1 because lp_cur was recomputed on the same unchanged network.
2) **PPO inner epochs (default 3) with KL early stop** when approx_kl exceeds
   `ppo_kl_target` to avoid catastrophic single-batch jumps.
3) **EMA reward baseline** on top of the per-group baseline, then optional
   advantage clipping.
4) **Absolute + delta reward blend** so the network has an anchor to high
   absolute quality, not only relative improvement.
5) **Norm-based grad clipping** (1.0).
6) **Periodic deterministic eval** (IoU / Dice / mBoundF) on a fixed val batch.
7) **Rich JSONL logging** with p10/p90 reward, ratio statistics, kl, grad norm.

Run with:
  conda activate snake1
  CFG_FILE=configs/btcv_v3_4_fm_grpo_v2_gpu1.yaml \
  CUDA_VISIBLE_DEVICES=1 \
  python grpo_train_v2.py --cfg_file configs/btcv_v3_4_fm_grpo_v2_gpu1.yaml
"""

from __future__ import annotations
import os
import gc
import json
import math
import random
import datetime
import contextlib
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from lib.config import cfg, args
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.train.optimizer import make_optimizer
from lib.datasets import make_data_loader
from lib.utils.snake import snake_config
from lib.utils.snake import snake_gcn_utils
from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
from lib.train.rewards.region_reward import compute_region_reward, compute_region_score
from lib.train.grpo_v2_utils import EMA, freeze_ref_flow, compute_eval_metrics, percentiles, freeze_bn_running_stats


_THIS_DIR = Path(__file__).resolve().parent


# ---------- path helpers (mirror grpo_train.py contract) ----------

def _cfg_file_used() -> str:
    return str(getattr(args, 'cfg_file', '') or os.environ.get('CFG_FILE', '')).strip()


def _cfg_stem() -> str:
    p = _cfg_file_used()
    if p:
        return Path(p).stem
    md = str(getattr(cfg, 'model_dir', '') or '').strip()
    if md:
        return Path(md).name
    return 'grpo_v2'


def _project_path(p) -> Path:
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else _THIS_DIR / p


def _output_dir() -> Path:
    env_md = str(os.environ.get('GRPO_V2_MODEL_DIR', '') or '').strip()
    if env_md:
        return _project_path(env_md)
    md = str(getattr(cfg, 'model_dir', '') or '').strip()
    return _project_path(md) if md else _THIS_DIR / 'data' / 'outputs' / _cfg_stem()


def _resolve_checkpoint_path() -> Optional[Path]:
    cand = []
    e = os.environ.get('CKPT_PATH', '').strip()
    if e:
        cand.append(Path(e))
    arg_ckpt = str(getattr(args, 'ckpt', '') or '').strip()
    if arg_ckpt:
        cand.append(Path(arg_ckpt))
    rp = str(getattr(cfg, 'resume_path', '') or '').strip()
    if rp:
        cand.append(_project_path(rp))
    md = str(getattr(cfg, 'model_dir', '') or '').strip()
    if md:
        cand.append(_project_path(md) / 'checkpoints' / 'latest.pt')
    for c in cand:
        if c.suffix in ('.pt', '.pth') and c.exists():
            return c
    return None


def _safe_torch_save(obj, path: Path):
    path = Path(path)
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
    needs = any(k.startswith('net.') for k in model.state_dict())
    has = any(k.startswith('net.') for k in sd)
    if needs and not has:
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


def _move_batch(batch, device='cuda'):
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
            mask = batch['ct_01'].bool()
            polys = polys[mask]
        else:
            polys = polys.view(-1, polys.size(-2), polys.size(-1))
    elif polys.dim() != 3:
        raise ValueError(f'{key} must be 3D or 4D, got {tuple(polys.shape)}')
    return polys.to(device=device) if device is not None else polys


def _make_py_ind(batch: Dict, n_contours: int, device) -> torch.Tensor:
    if 'ct_01' in batch and isinstance(batch['ct_01'], torch.Tensor) and batch['ct_01'].dim() == 2:
        valid = batch['ct_01'].bool()
        inds = []
        for bi in range(valid.size(0)):
            inds.append(torch.full((int(valid[bi].sum().item()),), bi, dtype=torch.long, device=device))
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
    a_i = _signed_area(i_init)
    a_g = _signed_area(i_gt)
    mis = ((a_i >= 0) ^ (a_g >= 0))
    if mis.any():
        i_gt[mis] = torch.flip(i_gt[mis], dims=[1])
    d2 = (i_init[:, :1, :] - i_gt).pow(2).sum(-1)
    nearest = torch.argmin(d2, dim=1)
    rolled = [torch.roll(i_gt[i], shifts=-int(nearest[i].item()), dims=0) for i in range(i_gt.size(0))]
    return torch.stack(rolled, dim=0)


def _contour_laplacian_px(poly: torch.Tensor, coord_scale: float) -> torch.Tensor:
    poly_px = poly * float(coord_scale)
    return torch.roll(poly_px, 1, dims=1) - 2.0 * poly_px + torch.roll(poly_px, -1, dims=1)


# ---------- main ----------

def main():
    print('=' * 70)
    print('[GRPO-V2] Stable RL post-training for flow-matching contour snake')
    print('=' * 70)

    # --- core cfg coerce
    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True
    cfg.use_grpo = True
    cfg.use_flow_matching = True

    # V2 hyperparameters (with sensible defaults; can override via cfg / env)
    v2 = getattr(cfg, 'grpo_v2', None)
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
            vals = [p.strip() for p in parts if p.strip()]
            out = []
            for v in vals:
                try:
                    out.append(int(v))
                except ValueError:
                    try:
                        out.append(float(v))
                    except ValueError:
                        out.append(v)
            return type(default)(out)
        return raw

    def cv(name, default):
        env_name = f"GRPO_V2_{name.upper()}"
        if env_name in os.environ:
            return _parse_env_value(os.environ[env_name], default)
        if v2 is not None and name in v2:
            return v2[name]
        return getattr(cfg, f'grpo_v2_{name}', default)
    def as_bool(v) -> bool:
        if isinstance(v, str):
            return v.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        return bool(v)
    gpu_override = os.environ.get('GRPO_V2_GPU', '').strip()
    if gpu_override:
        cfg.gpus = [int(gpu_override)]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_override
        print(f'[GRPO-V2] Override GPU -> {gpu_override}')
    train_steps  = int(os.environ.get('GRPO_V2_STEPS', cv('train_steps', 1000)))
    k            = int(cv('k', 6))
    rollout_steps= int(cv('rollout_steps', getattr(cfg, 'grpo_steps', 20)))
    rollout_noise_scale = float(cv('rollout_noise_scale', -1.0))
    rollout_noise_scale = None if rollout_noise_scale < 0 else rollout_noise_scale
    det_rollout_steps = int(cv('det_rollout_steps', 0))
    window_size  = int(cv('window_size', max(rollout_steps // 4, 3)))
    window_range = tuple(cv('window_range', (max(rollout_steps // 4, 1), rollout_steps)))
    action_std0  = float(cv('action_std', 0.15))
    action_std_min = float(cv('action_std_min', 0.05))
    action_std_decay = float(cv('action_std_decay', 0.9999))
    step_mode = str(cv('step_mode', 'gaussian')).strip().lower()
    noise_level = float(cv('noise_level', 0.8))
    sde_type = str(cv('sde_type', 'sde')).strip().lower()
    ppo_inner_epochs = int(cv('ppo_inner_epochs', 3))
    ppo_clip     = float(cv('ppo_clip', 0.2))
    ppo_kl_target= float(cv('ppo_kl_target', 0.02))
    kl_beta      = float(cv('kl_beta', 0.04))
    adv_clip_max = float(cv('adv_clip_max', 3.0))
    reward_abs_w = float(cv('reward_abs_weight', 0.5))
    reward_delta_w = float(cv('reward_delta_weight', 0.5))
    reward_w_region = float(cv('reward_w_region', 1.0))
    reward_w_dice = float(cv('reward_w_dice', 0.0))
    reward_w_iou  = float(cv('reward_w_iou', 0.0))
    reward_w_dist = float(cv('reward_w_dist', 0.0))
    reward_dist_max_px = float(cv('reward_dist_max_px', 8.0))
    reward_dist_quantile = float(cv('reward_dist_quantile', 95.0))
    reward_dist_quantile_weight = float(cv('reward_dist_quantile_weight', 0.5))
    reward_burr_weight = float(cv('reward_burr_weight', 0.0))
    reward_burr_max_px = float(cv('reward_burr_max_px', 2.0))
    reward_burr_margin_px = float(cv('reward_burr_margin_px', 0.25))
    ema_decay    = float(cv('ema_decay', 0.99))
    grad_clip_norm = float(cv('grad_clip_norm', 1.0))
    distill_weight = float(cv('distill_weight', 0.0))
    distill_min_delta = float(cv('distill_min_delta', 0.001))
    # When distill_compare_det=True, gate distillation on best-stochastic > det+margin
    # rather than best-stochastic > YOLO_init+min_delta. This ensures we only teach
    # the model trajectories that are genuinely better than its own deterministic output.
    distill_compare_det = bool(int(cv('distill_compare_det', 0)))
    distill_det_margin = float(cv('distill_det_margin', 0.002))
    # Clip individual distillation loss to prevent catastrophic update from outlier batches.
    # Set to 0 to disable. Recommended 0.02-0.05 when lr >= 1e-6 with large action_std.
    distill_loss_clip = float(cv('distill_loss_clip', 0.0))
    # Advantage-weighted multi-rollout distillation (adv_distill=1):
    # Instead of distilling only the best-of-K rollout, use ALL K rollouts that have
    # positive advantage, weighted by their normalized advantage.  This provides ~K/2
    # independent gradient targets per step (vs 1), substantially stronger signal.
    adv_distill = bool(int(cv('adv_distill', 0)))
    distill_t_late_prob = float(cv('distill_t_late_prob', 0.0))
    distill_t_late_min = float(cv('distill_t_late_min', 0.75))
    rollout_source = str(cv('rollout_source', 'manual_gt_init')).strip().lower()
    rollout_iterative = bool(cv('rollout_iterative', True))
    train_midstate_mode = rollout_source in (
        'train_state_mid', 'train_midstate', 'train_state_structured_mid',
        'train_structured_mid', 'v3b_structured', 'v3b_train_state',
    )
    structured_mode = rollout_source in (
        'train_state_structured', 'train_structured', 'structured', 'v3_structured',
        'train_state_structured_mid', 'train_structured_mid', 'v3b_structured',
    )
    onestep_mode = as_bool(cv('onestep', rollout_source in ('train_state_onestep', 'train_onestep') or structured_mode))
    onestep_steps = int(cv('onestep_steps', 1))
    structured_normal_std_px = float(cv('structured_normal_std_px', 2.0))
    structured_gt_pull_px = float(cv('structured_gt_pull_px', 2.0))
    structured_noise_std_px = float(cv('structured_noise_std_px', 0.25))
    structured_segment_min_frac = float(cv('structured_segment_min_frac', 0.08))
    structured_segment_max_frac = float(cv('structured_segment_max_frac', 0.25))
    structured_max_delta_px = float(cv('structured_max_delta_px', 4.0))
    update_only_if_beats_det = as_bool(cv('update_only_if_beats_det', False))
    dump_trajectory_viz = as_bool(cv('dump_trajectory_viz', True))
    eval_every   = int(cv('eval_every', 50))
    viz_every    = int(cv('viz_every', 50))
    save_every   = int(cv('save_every', 100))
    log_every    = int(cv('log_every', 1))
    seed         = int(cv('seed', 20260515))
    latent_policy = as_bool(cv('latent_policy', False))
    latent_ppo_weight = float(cv('latent_ppo_weight', 1.0))
    latent_elite_only = as_bool(cv('latent_elite_only', False))
    latent_elite_min_gain = float(cv('latent_elite_min_gain', 0.0))
    latent_elite_weight_clip = float(cv('latent_elite_weight_clip', 10.0))
    latent_ranker = as_bool(cv('latent_ranker', False))
    latent_ranker_weight = float(cv('latent_ranker_weight', 1.0))
    latent_ranker_temp = float(cv('latent_ranker_temp', 0.01))
    latent_ranker_top1 = as_bool(cv('latent_ranker_top1', True))
    aligned_policy_only = as_bool(cv('aligned_policy_only', False))
    if aligned_policy_only:
        if action_std0 <= 0.0:
            raise ValueError('grpo_v2_aligned_policy_only requires grpo_v2_action_std > 0.')
        if distill_weight != 0.0:
            print('[GRPO-V2] aligned policy mode: forcing distill_weight=0.0')
            distill_weight = 0.0
        if adv_distill:
            print('[GRPO-V2] aligned policy mode: forcing adv_distill=0')
            adv_distill = False
        if latent_ranker:
            print('[GRPO-V2] aligned policy mode: forcing latent_ranker=0')
            latent_ranker = False
            latent_ranker_weight = 0.0
        if latent_elite_only:
            print('[GRPO-V2] aligned policy mode: forcing latent_elite_only=0')
            latent_elite_only = False
        if latent_policy:
            print('[GRPO-V2] aligned policy mode: forcing latent_policy=0; only step logprob actions are updated')
            latent_policy = False
    if latent_policy:
        cfg.flow_use_latent_policy = True
    if latent_ranker:
        latent_policy = True
        cfg.flow_use_latent_policy = True
        cfg.flow_use_latent_ranker = True

    _set_seed(seed)

    # --- build network/trainer/optim/loader
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, trainer.network)
    grpo_lr = float(os.environ.get('GRPO_V2_LR', cv('lr', 1e-5)))
    for g in optimizer.param_groups:
        g['lr'] = grpo_lr
    print(f'[GRPO-V2] optimizer LR set to {grpo_lr}')
    data_loader = make_data_loader(cfg, is_train=True, is_distributed=False)

    # --- multi-batch fixed eval set for stable IoU estimate
    eval_batches = []
    try:
        eval_loader = make_data_loader(cfg, is_train=False, is_distributed=False)
        eval_iter = iter(eval_loader)
        n_eval_batches = int(os.environ.get('GRPO_V2_EVAL_BATCHES', cv('eval_batches', 4)))
        for _ in range(n_eval_batches):
            try:
                eb = next(eval_iter)
                _move_batch(eb)
                eval_batches.append(eb)
            except StopIteration:
                break
        print(f'[GRPO-V2] fixed eval set: {len(eval_batches)} batches')
    except Exception as e:
        print(f'[GRPO-V2] WARNING: eval batch unavailable: {e}')
    eval_batch = eval_batches[0] if eval_batches else None  # for viz

    # --- load base ckpt
    ckpt_path = _resolve_checkpoint_path()
    if ckpt_path is None:
        raise FileNotFoundError('No base checkpoint found. Set resume_path or CKPT_PATH.')
    net_for_load = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    sd = _extract_state_dict(torch.load(str(ckpt_path), map_location='cpu'))
    sd = _adapt_state_dict(net_for_load, sd)
    missing, unexpected = net_for_load.load_state_dict(sd, strict=False)
    total = len(list(net_for_load.state_dict().keys()))
    ratio = 100.0 * (total - len(missing)) / max(total, 1)
    print(f'[GRPO-V2] ckpt: {ckpt_path} | load ratio: {ratio:.2f}%  missing={len(missing)} unexpected={len(unexpected)}')
    if ratio < float(os.environ.get('GRPO_V2_MIN_LOAD_RATIO', '95.0')):
        raise RuntimeError(f'load ratio too low: {ratio:.2f}%')

    # --- freeze ref flow snapshot for KL
    inner = net_for_load.net if hasattr(net_for_load, 'net') else net_for_load
    ref_flow = freeze_ref_flow(inner)
    print('[GRPO-V2] frozen reference policy snapshot created.')

    # --- freeze BN running stats (critical for RL stability)
    from lib.train.grpo_v2_utils import freeze_bn_running_stats
    inner.train()  # need GT branch; we re-freeze BN stats below
    nbn = freeze_bn_running_stats(inner)
    print(f'[GRPO-V2] froze BN running stats on {nbn} layers (params still trainable).')

    # --- output dirs
    out_dir = _output_dir()
    ckpt_dir = out_dir / 'checkpoints'
    log_dir = out_dir / 'posttrain_grpo_v2'
    viz_dir = _THIS_DIR / 'visual' / f'grpo_v2_{_cfg_stem()}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'logs.jsonl'

    # write config snapshot once
    with open(log_dir / 'v2_hparams.json', 'w') as f:
        json.dump({
            'train_steps': train_steps, 'k': k, 'rollout_steps': rollout_steps,
            'rollout_noise_scale': rollout_noise_scale, 'det_rollout_steps': det_rollout_steps,
            'window_size': window_size, 'window_range': list(window_range),
            'action_std0': action_std0, 'action_std_min': action_std_min,
            'action_std_decay': action_std_decay,
            'step_mode': step_mode, 'noise_level': noise_level, 'sde_type': sde_type,
            'ppo_inner_epochs': ppo_inner_epochs, 'ppo_clip': ppo_clip,
            'ppo_kl_target': ppo_kl_target, 'kl_beta': kl_beta,
            'adv_clip_max': adv_clip_max,
            'reward_abs_weight': reward_abs_w, 'reward_delta_weight': reward_delta_w,
            'reward_w_region': reward_w_region, 'reward_w_dice': reward_w_dice,
            'reward_w_iou': reward_w_iou, 'reward_w_dist': reward_w_dist,
            'reward_dist_max_px': reward_dist_max_px,
            'reward_dist_quantile': reward_dist_quantile,
            'reward_dist_quantile_weight': reward_dist_quantile_weight,
            'reward_burr_weight': reward_burr_weight,
            'reward_burr_max_px': reward_burr_max_px,
            'reward_burr_margin_px': reward_burr_margin_px,
            'rollout_source': rollout_source, 'rollout_iterative': rollout_iterative,
            'train_midstate_mode': train_midstate_mode,
            'structured_mode': structured_mode,
            'structured_normal_std_px': structured_normal_std_px,
            'structured_gt_pull_px': structured_gt_pull_px,
            'structured_noise_std_px': structured_noise_std_px,
            'structured_segment_min_frac': structured_segment_min_frac,
            'structured_segment_max_frac': structured_segment_max_frac,
            'structured_max_delta_px': structured_max_delta_px,
            'onestep_mode': onestep_mode,
            'onestep_steps': onestep_steps,
            'update_only_if_beats_det': update_only_if_beats_det,
            'dump_trajectory_viz': dump_trajectory_viz,
            'distill_weight': distill_weight, 'distill_min_delta': distill_min_delta,
            'distill_compare_det': distill_compare_det, 'distill_det_margin': distill_det_margin,
            'distill_t_late_prob': distill_t_late_prob,
            'distill_t_late_min': distill_t_late_min,
            'latent_policy': latent_policy, 'latent_ppo_weight': latent_ppo_weight,
            'latent_elite_only': latent_elite_only,
            'latent_elite_min_gain': latent_elite_min_gain,
            'latent_elite_weight_clip': latent_elite_weight_clip,
            'latent_ranker': latent_ranker,
            'latent_ranker_weight': latent_ranker_weight,
            'latent_ranker_temp': latent_ranker_temp,
            'latent_ranker_top1': latent_ranker_top1,
            'aligned_policy_only': aligned_policy_only,
            'ema_decay': ema_decay, 'grad_clip_norm': grad_clip_norm,
            'eval_every': eval_every, 'viz_every': viz_every, 'save_every': save_every,
            'seed': seed, 'base_ckpt': str(ckpt_path),
        }, f, indent=2)

    # --- baselines / EMAs
    ema_reward = EMA(decay=ema_decay)
    ema_eval_iou = EMA(decay=0.9)
    best_eval_iou = -1.0  # track best on multi-batch eval set for best-ckpt saving
    fixed_eval_baseline_iou = -1.0

    # --- helpers wired to inner net (gcn)
    gcn = inner.gcn

    def _manual_gt_init_context(batch):
        """Build the same GT-init/manual inference context used by eval_v37.

        The V3.4 training branch randomly interpolates `i_it_py` toward GT for
        supervised iterative-refinement training. RL must not use that branch:
        it would optimize a teacher-forced state that is not evaluated at test
        time. This path mirrors scripts/eval_v37_full_iou.py: YOLO features
        plus dataset-provided init/GT polygons.
        """
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
                    feat_p3_up = F.interpolate(feat_p3, size=feat_p2.shape[-2:], mode='bilinear', align_corners=False)
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
            n = min(i_gt.size(0), i_init.size(0))
            i_init, c_init, i_gt, py_ind = i_init[:n], c_init[:n], i_gt[:n], py_ind[:n]
        if was_training:
            inner.train()
            freeze_bn_running_stats(inner)
        return {
            'cnn_feature': cnn_feature,
            'i_it_py': i_init,
            'c_it_py': c_init,
            'i_gt_py': i_gt,
            'py_ind': py_ind,
            'feat_hw': tuple(cnn_feature.shape[-2:]),
        }

    def _sample_pretrain_state_frac(n: int, device, dtype) -> torch.Tensor:
        """Mirror FlowMatchingEvolution.forward rich-state interpolation."""
        if n <= 0:
            return torch.zeros((0, 1, 1), device=device, dtype=dtype)
        if bool(getattr(cfg, 'v4_9_use_rich_state_sampling', False)):
            cont_p = max(float(getattr(cfg, 'v4_9_continuous_state_prob', 0.60)), 0.0)
            disc_p = max(float(getattr(cfg, 'v4_9_discrete_state_prob', 0.0)), 0.0)
            small_p = max(float(getattr(cfg, 'v4_9_small_state_prob', 0.25)), 0.0)
            far_p = max(float(getattr(cfg, 'v4_9_hard_far_state_prob', 0.10)), 0.0)
            zero_p = max(float(getattr(cfg, 'v4_9_near_zero_state_prob', 0.05)), 0.0)
            exact_zero_p = max(float(getattr(cfg, 'v4_9_exact_zero_state_prob', 0.0)), 0.0)
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
                draw = torch.rand(n, device=device)
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
                frac = torch.zeros(n, 1, 1, device=device, dtype=dtype)

                def _sample_frac(mask, min_name, max_name, default_min, default_max):
                    if not bool(mask.any().item()):
                        return
                    min_frac = min(max(float(getattr(cfg, min_name, default_min)), 0.0), 0.999)
                    max_frac = min(max(float(getattr(cfg, max_name, default_max)), min_frac), 0.999)
                    frac[mask] = torch.empty(
                        int(mask.sum().item()), 1, 1,
                        device=device, dtype=dtype,
                    ).uniform_(min_frac, max_frac)

                _sample_frac(cont_mask, 'v4_9_continuous_min_frac', 'v4_9_continuous_max_frac', 0.05, 0.85)
                if bool(disc_mask.any().item()):
                    fractions = getattr(cfg, 'v4_9_discrete_fractions', None)
                    if fractions is None:
                        fractions = getattr(cfg, 'iterative_fractions', None)
                    if fractions:
                        choices = torch.tensor([float(f) for f in fractions], device=device, dtype=dtype).clamp_(0.0, 0.999)
                        choice_idx = torch.randint(0, int(choices.numel()), (int(disc_mask.sum().item()),), device=device)
                        frac[disc_mask] = choices[choice_idx].view(-1, 1, 1)
                if bool(small_mask.any().item()):
                    min_frac = min(max(float(getattr(cfg, 'v4_1_small_disp_min_frac', 0.80)), 0.0), 0.999)
                    max_frac = min(max(float(getattr(cfg, 'v4_1_small_disp_max_frac', 0.95)), min_frac), 0.999)
                    frac[small_mask] = torch.empty(
                        int(small_mask.sum().item()), 1, 1,
                        device=device, dtype=dtype,
                    ).uniform_(min_frac, max_frac)
                _sample_frac(far_mask, 'v4_9_hard_far_min_frac', 'v4_9_hard_far_max_frac', 0.0, 0.20)
                _sample_frac(zero_mask, 'v4_9_near_zero_min_frac', 'v4_9_near_zero_max_frac', 0.95, 0.995)
                if bool(exact_zero_mask.any().item()):
                    frac[exact_zero_mask] = 1.0
                return frac

        iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
        situations = torch.randint(0, max(iter_steps, 1), (n,), device=device)
        return situations.to(dtype=dtype).view(n, 1, 1) / float(max(iter_steps, 1))

    def _train_branch_context(batch):
        """Legacy V2 context; kept only for ablation/debugging."""
        net_for_load.train()
        freeze_bn_running_stats(net_for_load)
        with torch.no_grad():
            output = inner(batch['inp'], batch)
        if train_midstate_mode:
            oct_init = output['i_it_py'].detach()
            i_gt = output.get('i_gt_py', None)
            if not isinstance(i_gt, torch.Tensor) or i_gt.numel() == 0:
                raise RuntimeError('train_midstate rollout requires i_gt_py.')
            i_gt = _align_gt(oct_init, i_gt.detach())
            frac = _sample_pretrain_state_frac(oct_init.size(0), oct_init.device, oct_init.dtype)
            mid_init = oct_init + (i_gt - oct_init) * frac
            output = dict(output)
            output['octagon_i_it_py'] = oct_init
            output['i_it_py'] = mid_init
            output['i_gt_py'] = i_gt
            output['c_it_py'] = snake_gcn_utils.img_poly_to_can_poly(mid_init)
            output['train_midstate_frac'] = frac.detach()
        return output

    def _forward_for_rollout(batch):
        if rollout_source in ('manual', 'manual_gt', 'manual_gt_init', 'eval_manual'):
            return _manual_gt_init_context(batch)
        if rollout_source in (
            'train_state_onestep', 'train_onestep', 'train_state_structured',
            'train_structured', 'structured', 'v3_structured',
            'train_state_mid', 'train_midstate', 'train_state_structured_mid',
            'train_structured_mid', 'v3b_structured', 'v3b_train_state',
        ):
            return _train_branch_context(batch)
        if rollout_source in ('train', 'train_branch', 'legacy'):
            return _train_branch_context(batch)
        raise ValueError(f'Unknown grpo_v2_rollout_source={rollout_source!r}')

    def _disp_delta_to_latent_delta(delta_raw: torch.Tensor, contour_scale: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros_like(delta_raw)
        return gcn.normalize_target_disp(delta_raw, contour_scale) - gcn.normalize_target_disp(zero, contour_scale)

    def _structured_segment_mask(n_inst: int, n_pts: int, device, dtype) -> torch.Tensor:
        masks = torch.zeros((n_inst, n_pts, 1), device=device, dtype=dtype)
        min_len = max(1, int(round(n_pts * max(0.0, structured_segment_min_frac))))
        max_len = max(min_len, int(round(n_pts * max(structured_segment_min_frac, structured_segment_max_frac))))
        max_len = min(max_len, n_pts)
        for bi in range(n_inst):
            seg_len = int(torch.randint(min_len, max_len + 1, (1,), device=device).item())
            start = int(torch.randint(0, n_pts, (1,), device=device).item())
            idx = (torch.arange(seg_len, device=device) + start) % n_pts
            if seg_len <= 1:
                win = torch.ones((1,), device=device, dtype=dtype)
            else:
                win = torch.hann_window(seg_len + 2, periodic=False, device=device, dtype=dtype)[1:-1]
                win = win.clamp_min(0.05)
            masks[bi, idx, 0] = win
        return masks

    def _structured_latent_delta(x_latent: torch.Tensor, i_init: torch.Tensor,
                                 i_gt: torch.Tensor, contour_scale: torch.Tensor) -> torch.Tensor:
        i_gt = _align_gt(i_init, i_gt)
        cur_disp = gcn.denormalize_pred_disp(x_latent, contour_scale)
        cur_poly = i_init + cur_disp
        prev_p = torch.roll(cur_poly, shifts=1, dims=1)
        next_p = torch.roll(cur_poly, shifts=-1, dims=1)
        tangent = next_p - prev_p
        normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        # Nearest-GT vector gives a supervised direction for exploration only.
        dist = torch.cdist(cur_poly.float(), i_gt.float()).to(dtype=cur_poly.dtype)
        nn_idx = dist.argmin(dim=-1)
        nearest_gt = torch.gather(i_gt, 1, nn_idx.unsqueeze(-1).expand(-1, -1, 2))
        gt_vec = nearest_gt - cur_poly
        gt_on_normal = (gt_vec * normal).sum(dim=-1, keepdim=True) * normal

        down = max(float(snake_config.down_ratio), 1.0)
        normal_std = structured_normal_std_px / down
        gt_pull = structured_gt_pull_px / down
        noise_std = structured_noise_std_px / down
        max_delta = structured_max_delta_px / down

        seg_mask = _structured_segment_mask(cur_poly.size(0), cur_poly.size(1), cur_poly.device, cur_poly.dtype)
        normal_delta = torch.randn((cur_poly.size(0), cur_poly.size(1), 1), device=cur_poly.device, dtype=cur_poly.dtype) * normal_std * normal
        gt_norm = gt_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        gt_step = gt_on_normal / gt_norm * torch.minimum(gt_norm, torch.full_like(gt_norm, gt_pull))
        noise_delta = torch.randn_like(cur_poly) * noise_std
        delta_raw = seg_mask * (normal_delta + gt_step + noise_delta)

        delta_norm = delta_raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta_raw = delta_raw * torch.clamp(max_delta / delta_norm, max=1.0)
        return _disp_delta_to_latent_delta(delta_raw, contour_scale)

    @torch.no_grad()
    def _sample_rollout(output, action_std, steps_override=None, noise_scale_override=None):
        cnn_feature = output['cnn_feature']
        i_init = output['i_it_py']
        c_init = output['c_it_py']
        py_ind = output['py_ind']
        i_gt = output.get('i_gt_py', None)
        local_rollout_steps = int(steps_override) if steps_override is not None else rollout_steps
        local_noise_scale = rollout_noise_scale if noise_scale_override is None else noise_scale_override
        if structured_mode:
            if not isinstance(i_gt, torch.Tensor) or i_gt.numel() == 0:
                raise RuntimeError('structured rollout requires i_gt_py in output.')
            local_steps = max(int(onestep_steps), 1)
            if local_noise_scale is None:
                local_noise_scale = 1.0
            ctx = gcn.prepare_sampling_context(cnn_feature, i_init, py_ind)
            x = torch.randn_like(i_init) * float(local_noise_scale)
            if window_size and window_size > 0:
                start_min = int(window_range[0]) if isinstance(window_range, (tuple, list)) and len(window_range) > 0 else 0
                end_max = int(window_range[1]) if isinstance(window_range, (tuple, list)) and len(window_range) > 1 else local_steps
                end_max = max(end_max, window_size)
                if end_max <= start_min + window_size:
                    s = max(0, min(start_min, max(local_steps - window_size, 0)))
                else:
                    s = int(np.random.randint(start_min, end_max - window_size + 1))
                e = min(s + window_size, local_steps)
            else:
                s, e = 0, local_steps

            ret = {
                'latents': [], 'log_probs': [], 'timesteps': [], 'step_indices': [],
                'x_ts': [], 'x_prevs': [], 'x_self_conds': [],
                'sampled_feat': ctx['sampled_feat'], 'detail_feat': ctx['detail_feat'],
                'contour_scale': ctx['contour_scale'],
            }
            x_self_cond = torch.zeros_like(x) if getattr(gcn, '_use_self_conditioning', False) else None
            dt = 1.0 / float(local_steps)
            for idx in range(local_steps):
                t_value = idx * dt
                in_policy_window = idx >= s and idx < e
                if in_policy_window:
                    x_mean, _, step_mean, _, next_self_cond = gcn.step_with_logprob(
                        cnn_feature, i_init, c_init, py_ind,
                        x_t=x, t_value=t_value, step_index=idx, total_steps=local_steps,
                        action_std=0.0, prev_sample=None,
                        sampled_feat=ctx['sampled_feat'], detail_feat=ctx['detail_feat'],
                        contour_scale=ctx['contour_scale'], x_self_cond=x_self_cond,
                        step_mode=step_mode, noise_level=noise_level, sde_type=sde_type,
                    )
                    x_struct = step_mean + _structured_latent_delta(step_mean, i_init, i_gt, ctx['contour_scale'])
                    x_prev, log_prob, _, _, next_self_cond = gcn.step_with_logprob(
                        cnn_feature, i_init, c_init, py_ind,
                        x_t=x, t_value=t_value, step_index=idx, total_steps=local_steps,
                        action_std=action_std, prev_sample=x_struct,
                        sampled_feat=ctx['sampled_feat'], detail_feat=ctx['detail_feat'],
                        contour_scale=ctx['contour_scale'], x_self_cond=x_self_cond,
                        step_mode=step_mode, noise_level=noise_level, sde_type=sde_type,
                    )
                    ret['log_probs'].append(log_prob.detach())
                    ret['timesteps'].append(torch.tensor(t_value, device=i_init.device, dtype=x.dtype))
                    ret['step_indices'].append(torch.tensor(idx, device=i_init.device, dtype=torch.long))
                    ret['x_ts'].append(x.detach())
                    ret['x_prevs'].append(x_prev.detach())
                    ret['x_self_conds'].append(None if x_self_cond is None else x_self_cond.detach())
                    ret['latents'].append(x.detach())
                else:
                    x_prev, _, _, _, next_self_cond = gcn.step_with_logprob(
                        cnn_feature, i_init, c_init, py_ind,
                        x_t=x, t_value=t_value, step_index=idx, total_steps=local_steps,
                        action_std=0.0, prev_sample=None,
                        sampled_feat=ctx['sampled_feat'], detail_feat=ctx['detail_feat'],
                        contour_scale=ctx['contour_scale'], x_self_cond=x_self_cond,
                        step_mode=step_mode, noise_level=noise_level, sde_type=sde_type,
                    )
                x = x_prev.detach()
                if getattr(gcn, '_use_self_conditioning', False):
                    x_self_cond = next_self_cond
            ret['disp'] = gcn.denormalize_pred_disp(x, ctx['contour_scale'])
            ret['py'] = i_init + ret['disp']
            n_log = len(ret['log_probs'])
            ret['cnn_features'] = [cnn_feature.detach()] * n_log
            ret['i_inits'] = [i_init.detach()] * n_log
            ret['c_inits'] = [c_init.detach()] * n_log
            ret['py_inds'] = [py_ind.detach()] * n_log
            ret['sampled_feats'] = [ctx['sampled_feat'].detach()] * n_log
            ret['detail_feats'] = [None if ctx.get('detail_feat') is None else ctx['detail_feat'].detach()] * n_log
            ret['contour_scales'] = [ctx['contour_scale'].detach()] * n_log
            ret['total_steps'] = [torch.tensor(local_steps, device=i_init.device, dtype=torch.long)] * n_log
            return ret
        if onestep_mode:
            local_steps = max(int(onestep_steps), 1)
            ret = gcn.sample_with_logprob(
                cnn_feature, i_init, c_init, py_ind,
                steps=local_steps,
                window_size=window_size,
                window_range=window_range,
                action_std=action_std,
                noise_scale=local_noise_scale,
                step_mode=step_mode,
                noise_level=noise_level,
                sde_type=sde_type,
            )
            n_log = len(ret.get('log_probs', []))
            ret['cnn_features'] = [cnn_feature.detach()] * n_log
            ret['i_inits'] = [i_init.detach()] * n_log
            ret['c_inits'] = [c_init.detach()] * n_log
            ret['py_inds'] = [py_ind.detach()] * n_log
            ret['sampled_feats'] = [ret['sampled_feat'].detach()] * n_log
            ret['detail_feats'] = [None if ret.get('detail_feat') is None else ret['detail_feat'].detach()] * n_log
            ret['contour_scales'] = [ret['contour_scale'].detach()] * n_log
            ret['total_steps'] = [torch.tensor(local_steps, device=i_init.device, dtype=torch.long)] * n_log
            return ret
        if rollout_iterative and getattr(gcn, 'use_iterative_refinement', False):
            iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
            fractions = list(getattr(cfg, 'iterative_fractions', []))
            if not fractions:
                fractions = [1.0 / (iter_steps - i) for i in range(iter_steps)]
            iter_ode_steps = int(getattr(cfg, 'iterative_ode_steps', getattr(cfg, 'iterative_ddim_steps', local_rollout_steps)))
            if iter_ode_steps <= 0:
                iter_ode_steps = local_rollout_steps

            current = i_init.detach()
            total_disp = torch.zeros_like(i_init)
            merged = {
                'latents': [], 'log_probs': [], 'timesteps': [], 'step_indices': [],
                'x_ts': [], 'x_prevs': [], 'x_self_conds': [],
                'cnn_features': [], 'i_inits': [], 'c_inits': [], 'py_inds': [],
                'sampled_feats': [], 'detail_feats': [], 'contour_scales': [],
                'total_steps': [],
                'latent_x0s': [], 'latent_log_probs': [], 'latent_sampled_feats': [], 'latent_noise_scales': [],
            }
            for frac in fractions[:iter_steps]:
                c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
                ret = gcn.sample_with_logprob(
                    cnn_feature, current, c_cur, py_ind,
                    steps=iter_ode_steps, window_size=window_size,
                    window_range=window_range, action_std=action_std,
                    noise_scale=local_noise_scale,
                    step_mode=step_mode,
                    noise_level=noise_level,
                    sde_type=sde_type,
                )
                n_log = len(ret.get('log_probs', []))
                if n_log > 0:
                    merged['latents'].extend(ret.get('latents', []))
                    merged['log_probs'].extend(ret['log_probs'])
                    merged['timesteps'].extend(ret['timesteps'])
                    merged['step_indices'].extend(ret['step_indices'])
                    merged['x_ts'].extend(ret['x_ts'])
                    merged['x_prevs'].extend(ret['x_prevs'])
                    merged['x_self_conds'].extend(ret['x_self_conds'])
                    merged['cnn_features'].extend([cnn_feature.detach()] * n_log)
                    merged['i_inits'].extend([current.detach()] * n_log)
                    merged['c_inits'].extend([c_cur.detach()] * n_log)
                    merged['py_inds'].extend([py_ind.detach()] * n_log)
                    merged['sampled_feats'].extend([ret['sampled_feat'].detach()] * n_log)
                    merged['detail_feats'].extend([None if ret.get('detail_feat') is None else ret['detail_feat'].detach()] * n_log)
                    merged['contour_scales'].extend([ret['contour_scale'].detach()] * n_log)
                    merged['total_steps'].extend([torch.tensor(iter_ode_steps, device=i_init.device, dtype=torch.long)] * n_log)
                if isinstance(ret.get('latent_log_prob', None), torch.Tensor):
                    merged['latent_x0s'].append(ret['latent_x0'].detach())
                    merged['latent_log_probs'].append(ret['latent_log_prob'].detach())
                    merged['latent_sampled_feats'].append(ret['sampled_feat'].detach())
                    merged['latent_noise_scales'].append(float(ret.get('latent_noise_scale', local_noise_scale or getattr(gcn, '_flow_train_noise_scale', 1.0))))
                applied = ret['disp'] * float(frac)
                current = (current + applied).detach()
                total_disp = total_disp + applied
            merged['disp'] = total_disp
            merged['py'] = i_init + total_disp
            if merged['latent_log_probs']:
                merged['latent_log_prob'] = torch.stack(merged['latent_log_probs'], dim=0).sum(dim=0)
            return merged

        ret = gcn.sample_with_logprob(
            cnn_feature, i_init, c_init, py_ind,
            steps=local_rollout_steps, window_size=window_size,
            window_range=window_range, action_std=action_std,
            noise_scale=local_noise_scale,
            step_mode=step_mode,
            noise_level=noise_level,
            sde_type=sde_type,
        )
        n_log = len(ret.get('log_probs', []))
        ret['cnn_features'] = [cnn_feature.detach()] * n_log
        ret['i_inits'] = [i_init.detach()] * n_log
        ret['c_inits'] = [c_init.detach()] * n_log
        ret['py_inds'] = [py_ind.detach()] * n_log
        ret['sampled_feats'] = [ret['sampled_feat'].detach()] * n_log
        ret['detail_feats'] = [None if ret.get('detail_feat') is None else ret['detail_feat'].detach()] * n_log
        ret['contour_scales'] = [ret['contour_scale'].detach()] * n_log
        ret['total_steps'] = [torch.tensor(local_rollout_steps, device=i_init.device, dtype=torch.long)] * n_log
        return ret

    @torch.no_grad()
    def _sample_deterministic_policy(output):
        cnn_feature = output['cnn_feature']
        i_init = output['i_it_py']
        c_init = output['c_it_py']
        py_ind = output['py_ind']
        local_det_steps = det_rollout_steps
        if onestep_mode:
            local_steps = int(local_det_steps) if int(local_det_steps) > 0 else max(int(onestep_steps), 1)
            disp = gcn.sample_disp(
                cnn_feature, i_init, c_init, py_ind,
                steps=local_steps,
            )
            return {'disp': disp, 'py': i_init + disp}
        if getattr(gcn, 'use_iterative_refinement', False):
            iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
            fractions = list(getattr(cfg, 'iterative_fractions', []))
            if not fractions:
                fractions = [1.0 / (iter_steps - i) for i in range(iter_steps)]
            iter_ode_steps = int(getattr(cfg, 'iterative_ode_steps', getattr(cfg, 'iterative_ddim_steps', rollout_steps)))
            if local_det_steps > 0:
                iter_ode_steps = local_det_steps
            disp = gcn.sample_disp_iterative(
                cnn_feature, i_init, c_init, py_ind,
                num_iter_steps=iter_steps,
                fractions=fractions,
                ode_steps=iter_ode_steps,
            )
        else:
            disp = gcn.sample_disp(
                cnn_feature, i_init, c_init, py_ind,
                steps=(local_det_steps if local_det_steps > 0 else rollout_steps),
            )
        return {'disp': disp, 'py': i_init + disp}

    def _sample_distill_t(n: int, device, dtype):
        t = gcn.sample_train_t(n, device=device, dtype=dtype)
        p = min(max(float(distill_t_late_prob), 0.0), 1.0)
        if p <= 0.0:
            return t
        late_min = min(max(float(distill_t_late_min), 0.0), 0.999)
        mask = torch.rand(n, 1, 1, device=device, dtype=dtype) < p
        late_t = torch.empty(n, 1, 1, device=device, dtype=dtype).uniform_(late_min, 1.0)
        return torch.where(mask, late_t, t)

    def _compute_rewards(output, ret):
        i_init = output['i_it_py']
        i_gt = output['i_gt_py']
        i_gt = _align_gt(i_init, i_gt)
        H = int(eval_batch['inp'].shape[-2]) if eval_batch is not None else int(384)
        W = int(eval_batch['inp'].shape[-1]) if eval_batch is not None else int(384)
        # use actual training image scale
        H = int(output.get('feat_hw', (H // snake_config.down_ratio, W // snake_config.down_ratio))[0]) if isinstance(output.get('feat_hw', None), (tuple, list)) else H
        # use image-scale reward (matches what production cares about)
        H_img = int(384)  # snake_config defines voc_input_w/h
        # fall back to batch shape
        # (we re-derive from cnn_feature to be safe)
        try:
            H_img = int(output['cnn_feature'].shape[-2] * snake_config.down_ratio)
            W_img = int(output['cnn_feature'].shape[-1] * snake_config.down_ratio)
        except Exception:
            W_img = H_img
        # init score
        init_score = compute_region_score(
            i_init, i_gt, H=H_img, W=W_img,
            w_boundary=reward_w_region, w_dice=reward_w_dice, w_iou=reward_w_iou,
            w_dist=reward_w_dist,
            dist_max_px=reward_dist_max_px,
            dist_quantile=reward_dist_quantile,
            dist_quantile_weight=reward_dist_quantile_weight,
            coord_scale=float(snake_config.down_ratio),
        ).detach()
        final_score = compute_region_reward(
            i_init, ret['disp'], i_gt, H=H_img, W=W_img,
            w1=reward_w_region, w_dice=reward_w_dice, w_iou=reward_w_iou,
            w_dist=reward_w_dist,
            dist_max_px=reward_dist_max_px,
            dist_quantile=reward_dist_quantile,
            dist_quantile_weight=reward_dist_quantile_weight,
            coord_scale=float(snake_config.down_ratio),
        ).detach()
        delta = (final_score - init_score)
        final_poly = i_init + ret['disp']
        coord_scale = float(snake_config.down_ratio)
        lap_final = _contour_laplacian_px(final_poly, coord_scale).norm(dim=-1)
        lap_gt = _contour_laplacian_px(i_gt, coord_scale).norm(dim=-1)
        lap_disp = _contour_laplacian_px(ret['disp'], coord_scale).norm(dim=-1)
        excess_burr = torch.relu(lap_final - lap_gt - float(reward_burr_margin_px))
        burr_raw = 0.5 * lap_disp.mean(dim=1) + 0.5 * excess_burr.mean(dim=1)
        burr_penalty = torch.clamp(burr_raw / max(float(reward_burr_max_px), 1e-6), min=0.0, max=2.0).detach()
        reward = reward_abs_w * final_score + reward_delta_w * delta - reward_burr_weight * burr_penalty
        return {
            'final_score': final_score, 'init_score': init_score,
            'delta_score': delta, 'reward': reward,
            'burr_penalty': burr_penalty,
            'burr_raw_px': burr_raw.detach(),
        }

    @torch.no_grad()
    def _compute_manual_eval_metrics(batch):
        output = _manual_gt_init_context(batch)
        i_init = output['i_it_py']
        i_gt = _align_gt(i_init, output['i_gt_py'])
        if i_init.numel() == 0 or i_gt.numel() == 0:
            return {}
        cnn_feature = output['cnn_feature']
        c_init = output['c_it_py']
        py_ind = output['py_ind']
        if getattr(gcn, 'use_iterative_refinement', False):
            iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
            fractions = list(getattr(cfg, 'iterative_fractions', []))
            if not fractions:
                fractions = [1.0 / (iter_steps - i) for i in range(iter_steps)]
            iter_ode_steps = int(getattr(cfg, 'iterative_ode_steps', getattr(cfg, 'iterative_ddim_steps', rollout_steps)))
            if iter_ode_steps <= 0:
                iter_ode_steps = rollout_steps
            disp = gcn.sample_disp_iterative(
                cnn_feature, i_init, c_init, py_ind,
                num_iter_steps=iter_steps,
                fractions=fractions,
                ode_steps=iter_ode_steps,
            )
        else:
            disp = gcn.sample_disp(cnn_feature, i_init, c_init, py_ind, steps=rollout_steps)
        pred = i_init + disp
        H_img = int(batch['inp'].shape[-2])
        W_img = int(batch['inp'].shape[-1])
        iou = compute_region_score(
            pred, i_gt, H=H_img, W=W_img,
            w_boundary=0.0, w_dice=0.0, w_iou=1.0,
            coord_scale=float(snake_config.down_ratio),
        )
        dice = compute_region_score(
            pred, i_gt, H=H_img, W=W_img,
            w_boundary=0.0, w_dice=1.0, w_iou=0.0,
            coord_scale=float(snake_config.down_ratio),
        )
        mbf = compute_region_score(
            pred, i_gt, H=H_img, W=W_img,
            w_boundary=1.0, w_dice=0.0, w_iou=0.0,
            coord_scale=float(snake_config.down_ratio),
        )
        return {
            'eval_iou': float(iou.mean().item()),
            'eval_dice': float(dice.mean().item()),
            'eval_mboundf': float(mbf.mean().item()),
            'eval_n': int(iou.numel()),
        }

    def _eval_fixed_batches():
        acc_iou, acc_mbf, acc_dice, acc_n = 0.0, 0.0, 0.0, 0
        for eb in eval_batches:
            em = _compute_manual_eval_metrics(eb)
            if em:
                n = em.get('eval_n', 1)
                acc_iou += em['eval_iou'] * n
                acc_mbf += em['eval_mboundf'] * n
                acc_dice += em['eval_dice'] * n
                acc_n += n
        if acc_n <= 0:
            return {}
        return {
            'eval_iou': acc_iou / acc_n,
            'eval_mboundf': acc_mbf / acc_n,
            'eval_dice': acc_dice / acc_n,
            'eval_n': int(acc_n),
        }

    if eval_batches:
        try:
            base_eval = _eval_fixed_batches()
            if base_eval:
                best_eval_iou = float(base_eval['eval_iou'])
                fixed_eval_baseline_iou = float(base_eval['eval_iou'])
                print(
                    f"[GRPO-V2] fixed-eval baseline: "
                    f"iou={base_eval['eval_iou']:.6f} "
                    f"dice={base_eval['eval_dice']:.6f} "
                    f"mbf={base_eval['eval_mboundf']:.6f} n={base_eval['eval_n']}"
                )
        except Exception as e:
            print(f'[GRPO-V2] baseline fixed-eval failed: {e}')

    # --- training loop
    scaler = GradScaler(enabled=False)  # no AMP for stability of policy gradient
    it = iter(data_loader)
    action_std = float(action_std0)

    for step in range(1, train_steps + 1):
        import time as _time_module
        _t_step_start = _time_module.time()
        # keep BN running stats frozen (eval/train mode toggles can re-enable them)
        freeze_bn_running_stats(inner)
        # ---- get batch
        try:
            batch = next(it)
        except StopIteration:
            it = iter(data_loader)
            batch = next(it)
        _move_batch(batch)
        if step <= 3:
            print(f'[TIME] step={step} batch_loaded: {_time_module.time()-_t_step_start:.2f}s', flush=True)

        # ---- forward (no grad) to get the rollout context
        output = _forward_for_rollout(batch)
        i_init = output.get('i_it_py', None)
        c_init = output.get('c_it_py', None)
        py_ind = output.get('py_ind', None)
        i_gt = output.get('i_gt_py', None)
        cnn_feature = output.get('cnn_feature', None)
        if (not isinstance(i_init, torch.Tensor)) or i_init.numel() == 0 or (not isinstance(i_gt, torch.Tensor)):
            print(f'[GRPO-V2] step {step}: empty contour batch, skipping.')
            continue
        if step <= 3:
            print(f'[TIME] step={step} fwd_done B={i_init.shape[0]}: {_time_module.time()-_t_step_start:.2f}s', flush=True)
        i_gt_aligned = _align_gt(i_init, i_gt)
        base_device = cnn_feature.device

        # ---- collect k rollouts (frozen old policy = current params)
        rollouts: List[Dict] = []
        rewards_list = []
        final_scores_list = []
        delta_scores_list = []
        burr_penalty_list = []
        burr_raw_list = []
        disp_list = []
        old_logs_list = []  # (k, B, T)
        old_latent_logs_list = []  # (k, B)
        step_log_count_list = []
        for ri_dbg in range(k):
            ret = _sample_rollout(output, action_std)
            if step <= 3 and ri_dbg == 0:
                print(f'[TIME] step={step} rollout0_done: {_time_module.time()-_t_step_start:.2f}s', flush=True)
            step_logs = ret.get('log_probs', [])
            has_step_logs = isinstance(step_logs, list) and len(step_logs) > 0
            latent_old_log = ret.get('latent_log_prob', None) if latent_policy else None
            has_latent_log = isinstance(latent_old_log, torch.Tensor)
            if (not has_step_logs) and (not has_latent_log):
                continue
            if has_step_logs:
                old_log = torch.stack(step_logs, dim=0).transpose(0, 1).contiguous().detach()
            else:
                old_log = torch.empty((i_init.size(0), 0), device=base_device, dtype=i_init.dtype)
            _t_rew = _time_module.time()
            rew = _compute_rewards(output, ret)
            if step <= 3 and ri_dbg == 0:
                print(f'[TIME] step={step} reward0: {_time_module.time()-_t_rew:.3f}s total={_time_module.time()-_t_step_start:.2f}s', flush=True)
            rollouts.append({
                'x_ts': [x.detach() for x in ret['x_ts']],
                'x_prevs': [x.detach() for x in ret['x_prevs']],
                'x_self_conds': [None if x is None else x.detach() for x in ret['x_self_conds']],
                'timesteps': [t.detach() for t in ret['timesteps']],
                'step_indices': [s.detach() for s in ret['step_indices']],
                'cnn_features': [x.detach() for x in ret.get('cnn_features', [])],
                'i_inits': [x.detach() for x in ret.get('i_inits', [])],
                'c_inits': [x.detach() for x in ret.get('c_inits', [])],
                'py_inds': [x.detach() for x in ret.get('py_inds', [])],
                'sampled_feats': [x.detach() for x in ret.get('sampled_feats', [])],
                'detail_feats': [None if x is None else x.detach() for x in ret.get('detail_feats', [])],
                'contour_scales': [x.detach() for x in ret.get('contour_scales', [])],
                'total_steps': [s.detach() for s in ret.get('total_steps', [])],
                'latent_x0': None if ret.get('latent_x0', None) is None else ret['latent_x0'].detach(),
                'latent_x0s': [x.detach() for x in ret.get('latent_x0s', [])],
                'latent_sampled_feat': None if ret.get('sampled_feat', None) is None else ret['sampled_feat'].detach(),
                'latent_sampled_feats': [x.detach() for x in ret.get('latent_sampled_feats', [])],
                'latent_noise_scale': float(ret.get('latent_noise_scale', rollout_noise_scale or getattr(gcn, '_flow_train_noise_scale', 1.0))),
                'latent_noise_scales': list(ret.get('latent_noise_scales', [])),
            })
            old_logs_list.append(old_log)
            old_latent_logs_list.append(latent_old_log.detach() if has_latent_log else None)
            step_log_count_list.append(int(old_log.size(1)))
            rewards_list.append(rew['reward'])
            final_scores_list.append(rew['final_score'])
            delta_scores_list.append(rew['delta_score'])
            burr_penalty_list.append(rew['burr_penalty'])
            burr_raw_list.append(rew['burr_raw_px'])
            disp_list.append(ret['disp'].detach())

        if len(rollouts) == 0:
            print(f'[GRPO-V2] step {step}: zero valid rollouts, skipping.')
            continue
        if aligned_policy_only and not any(n > 0 for n in step_log_count_list):
            raise RuntimeError(
                'aligned policy mode collected no step logprobs; check action_std/window_size/window_range.'
            )

        if step <= 3:
            print(f'[TIME] step={step} all_rollouts_done: {_time_module.time()-_t_step_start:.2f}s', flush=True)
        rewards = torch.stack(rewards_list, dim=0).to(base_device)         # (k, B)
        final_scores = torch.stack(final_scores_list, dim=0).to(base_device)
        delta_scores = torch.stack(delta_scores_list, dim=0).to(base_device)
        burr_penalties = torch.stack(burr_penalty_list, dim=0).to(base_device)
        burr_raw_px = torch.stack(burr_raw_list, dim=0).to(base_device)
        disp_stack = torch.stack(disp_list, dim=0).to(base_device)

        # ---- Deterministic baseline for quality-gated distillation
        # Run one deterministic (action_std=0) rollout per step so we know the
        # current model's "floor" score. The distillation gate then ensures we
        # only distill stochastic trajectories that genuinely beat this floor.
        det_scores = None
        if distill_compare_det and (distill_weight > 0 or latent_policy or aligned_policy_only):
            try:
                det_ret = _sample_deterministic_policy(output)
                rew_det = _compute_rewards(output, det_ret)
                det_scores = rew_det['final_score'].to(base_device).detach()  # (B,)
            except Exception as e:
                print(f'[GRPO-V2] step {step}: det-baseline failed: {e}')
                det_scores = None


        if distill_compare_det and det_scores is not None:
            quality_scores = final_scores - det_scores.unsqueeze(0)
        else:
            quality_scores = delta_scores
        ema_now = ema_reward.update(float(rewards.mean()))
        # When a deterministic baseline is available, compare each rollout
        # directly against the current deployed policy. This gives PPO a
        # meaningful signal even when every stochastic rollout in the group is
        # "bad" in absolute terms but some are less bad than others.
        use_det_advantage = bool(distill_compare_det and det_scores is not None and (action_std > 0.0 or latent_policy))
        if use_det_advantage:
            advantages = quality_scores
        else:
            group_baseline = rewards.mean(dim=0, keepdim=True)
            advantages = rewards - group_baseline
        # Dampened normalization: floor std so we don't amplify within-group noise
        adv_std = advantages.std(dim=0, unbiased=False, keepdim=True).clamp_min(0.1)
        advantages = (advantages / adv_std).clamp(-adv_clip_max, adv_clip_max).detach()

        # ---- Group-quality gate: zero advantage for batch indices where the
        # best rollout fails to improve over init by a positive margin. This
        # prevents the policy from being pushed toward the "least bad" sample
        # in groups where every rollout is worse than the supervised baseline.
        gate_margin = float(cfg.train.get('grpo_v2_gate_margin', 0.0))
        quality_best = quality_scores.max(dim=0, keepdim=True).values  # (1, B)
        gate_mask = (quality_best > gate_margin).float()  # (1, B)
        if update_only_if_beats_det or not use_det_advantage:
            advantages = advantages * gate_mask
        gate_active_frac = float(gate_mask.mean().item())

        # ---- prepare sampling context once
        with torch.no_grad():
            flow_ctx = gcn.prepare_sampling_context(cnn_feature.detach(), i_init, py_ind)

        # ---- PPO inner epochs
        approx_kl_history = []
        clipfrac_history = []
        ratio_history = []
        policy_loss_history = []
        kl_loss_history = []
        grad_norm_history = []
        latent_policy_loss_history = []
        latent_kl_history = []
        latent_ratio_history = []
        latent_grad_norm_history = []
        ranker_loss_val = 0.0
        ranker_grad_norm = 0.0
        ranker_top1_acc = 0.0
        early_stop_epoch = ppo_inner_epochs
        # At action_std=0, log_prob=0 everywhere so ratio=1 and PPO gradient is
        # identically zero.  Skip the entire PPO loop to avoid K×T wasted forward
        # passes (e.g. 16×60=960 ops per step).  The distillation block below is
        # the sole gradient source and is unaffected by this skip.
        _run_ppo = (action_std > 0.0)
        if aligned_policy_only and not _run_ppo:
            raise RuntimeError('aligned policy mode requires active PPO/GRPO step-action update.')
        for epoch in range(ppo_inner_epochs if _run_ppo else 0):
            total_steps_in_epoch = sum(len(t['x_ts']) for t in rollouts)
            if total_steps_in_epoch == 0:
                break
            optimizer.zero_grad(set_to_none=True)
            policy_loss_sum = 0.0
            kl_loss_sum = 0.0
            kl_terms = 0
            ep_approx_kl = []
            ep_clipfrac = []
            ep_ratio = []

            for ri, traj in enumerate(rollouts):
                adv = advantages[ri]  # (B,)
                old_logs = old_logs_list[ri].to(base_device)  # (B, T)
                T = len(traj['x_ts'])
                if T == 0:
                    continue
                for s_idx in range(T):
                    x_self_cond = traj['x_self_conds'][s_idx]
                    step_cnn = traj['cnn_features'][s_idx] if len(traj.get('cnn_features', [])) == T else cnn_feature.detach()
                    step_i_init = traj['i_inits'][s_idx] if len(traj.get('i_inits', [])) == T else i_init
                    step_c_init = traj['c_inits'][s_idx] if len(traj.get('c_inits', [])) == T else c_init
                    step_py_ind = traj['py_inds'][s_idx] if len(traj.get('py_inds', [])) == T else py_ind
                    step_sampled_feat = traj['sampled_feats'][s_idx] if len(traj.get('sampled_feats', [])) == T else flow_ctx['sampled_feat']
                    step_detail_feat = traj['detail_feats'][s_idx] if len(traj.get('detail_feats', [])) == T else flow_ctx['detail_feat']
                    step_contour_scale = traj['contour_scales'][s_idx] if len(traj.get('contour_scales', [])) == T else flow_ctx['contour_scale']
                    step_total_steps = int(traj['total_steps'][s_idx].item()) if len(traj.get('total_steps', [])) == T else rollout_steps
                    _, lp_cur, mean_cur, std_cur, _ = gcn.step_with_logprob(
                        step_cnn,
                        step_i_init,
                        step_c_init,
                        step_py_ind,
                        x_t=traj['x_ts'][s_idx],
                        t_value=traj['timesteps'][s_idx],
                        step_index=int(traj['step_indices'][s_idx].item()),
                        total_steps=step_total_steps,
                        action_std=action_std,
                        prev_sample=traj['x_prevs'][s_idx],
                        sampled_feat=step_sampled_feat,
                        detail_feat=step_detail_feat,
                        contour_scale=step_contour_scale,
                        x_self_cond=x_self_cond,
                        step_mode=step_mode,
                        noise_level=noise_level,
                        sde_type=sde_type,
                    )
                    old_log_s = old_logs[:, s_idx]
                    ratio = torch.exp(lp_cur - old_log_s)
                    unclipped = -adv * ratio
                    clipped = -adv * torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
                    policy_term = torch.maximum(unclipped, clipped).mean()
                    loss_term = policy_term / float(total_steps_in_epoch)
                    policy_loss_sum += float(policy_term.detach().item())

                    with torch.no_grad():
                        ep_approx_kl.append(0.5 * torch.mean((lp_cur - old_log_s) ** 2).item())
                        ep_clipfrac.append(((ratio - 1.0).abs() > ppo_clip).float().mean().item())
                        ep_ratio.append(ratio.detach())

                    # KL to frozen ref transition mean under the active step model.
                    if kl_beta > 0:
                        with torch.no_grad():
                            _, _, mean_ref, _, _ = ref_flow.step_with_logprob(
                                step_cnn,
                                step_i_init,
                                step_c_init,
                                step_py_ind,
                                x_t=traj['x_ts'][s_idx],
                                t_value=traj['timesteps'][s_idx],
                                step_index=int(traj['step_indices'][s_idx].item()),
                                total_steps=step_total_steps,
                                action_std=action_std,
                                prev_sample=traj['x_prevs'][s_idx],
                                sampled_feat=step_sampled_feat,
                                detail_feat=step_detail_feat,
                                contour_scale=step_contour_scale,
                                x_self_cond=x_self_cond,
                                step_mode=step_mode,
                                noise_level=noise_level,
                                sde_type=sde_type,
                            )
                        var = std_cur.pow(2).clamp_min(1e-12)
                        kl_term = (((mean_cur - mean_ref) ** 2) / (2.0 * var)).mean()
                        loss_term = loss_term + kl_beta * kl_term / float(total_steps_in_epoch)
                        kl_loss_sum += float(kl_term.detach().item())
                        kl_terms += 1
                    # At action_std=0 log_prob=0 (no grad), so policy_term
                    # is a constant – skip backward to avoid RuntimeError.
                    if loss_term.requires_grad:
                        loss_term.backward()

            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in net_for_load.parameters() if p.requires_grad],
                max_norm=grad_clip_norm,
            )
            optimizer.step()

            approx_kl_mean = float(np.mean(ep_approx_kl)) if ep_approx_kl else 0.0
            clipfrac_mean = float(np.mean(ep_clipfrac)) if ep_clipfrac else 0.0
            ratio_all = torch.cat(ep_ratio) if ep_ratio else torch.zeros(1, device=base_device)
            approx_kl_history.append(approx_kl_mean)
            clipfrac_history.append(clipfrac_mean)
            ratio_history.append(ratio_all)
            policy_loss_history.append(float(policy_loss_sum / max(total_steps_in_epoch, 1)))
            kl_loss_history.append(float(kl_loss_sum / max(kl_terms, 1)))
            grad_norm_history.append(float(gnorm.item() if hasattr(gnorm, 'item') else float(gnorm)))

            if approx_kl_mean > ppo_kl_target:
                early_stop_epoch = epoch + 1
                break

        if latent_policy and any(x is not None for x in old_latent_logs_list):
            for epoch in range(ppo_inner_epochs):
                optimizer.zero_grad(set_to_none=True)
                latent_terms = []
                ep_latent_kl = []
                ep_latent_ratio = []
                for ri, traj in enumerate(rollouts):
                    old_latent_log = old_latent_logs_list[ri]
                    latent_x0 = traj.get('latent_x0', None)
                    latent_sampled_feat = traj.get('latent_sampled_feat', None)
                    latent_x0s = traj.get('latent_x0s', [])
                    latent_sampled_feats = traj.get('latent_sampled_feats', [])
                    has_joint_latents = bool(latent_x0s and latent_sampled_feats)
                    if old_latent_log is None:
                        continue
                    if (not has_joint_latents) and (latent_x0 is None or latent_sampled_feat is None):
                        continue
                    adv = advantages[ri]
                    latent_noise_scales = traj.get('latent_noise_scales', [])
                    if has_joint_latents:
                        lp_parts = []
                        for j, (x0_j, sf_j) in enumerate(zip(latent_x0s, latent_sampled_feats)):
                            ns_j = latent_noise_scales[j] if j < len(latent_noise_scales) else traj.get('latent_noise_scale', rollout_noise_scale or getattr(gcn, '_flow_train_noise_scale', 1.0))
                            lp_parts.append(gcn.initial_latent_logprob(sf_j, x0_j, float(ns_j)))
                        lp_cur = torch.stack(lp_parts, dim=0).sum(dim=0)
                    else:
                        lp_cur = gcn.initial_latent_logprob(
                            latent_sampled_feat,
                            latent_x0,
                            float(traj.get('latent_noise_scale', rollout_noise_scale or getattr(gcn, '_flow_train_noise_scale', 1.0))),
                        )
                    old_latent_log = old_latent_log.to(device=lp_cur.device, dtype=lp_cur.dtype)
                    ratio = torch.exp(lp_cur - old_latent_log)
                    if latent_elite_only:
                        elite_w = (quality_scores[ri].detach() - latent_elite_min_gain).clamp_min(0.0)
                        if not bool((elite_w > 0).any().item()):
                            continue
                        elite_w = elite_w / elite_w.mean().clamp_min(1e-6)
                        if latent_elite_weight_clip > 0:
                            elite_w = elite_w.clamp_max(latent_elite_weight_clip)
                        latent_terms.append(-(elite_w * lp_cur).mean())
                    else:
                        unclipped = -adv * ratio
                        clipped = -adv * torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
                        latent_terms.append(torch.maximum(unclipped, clipped).mean())
                    with torch.no_grad():
                        ep_latent_kl.append(0.5 * torch.mean((lp_cur - old_latent_log) ** 2).item())
                        ep_latent_ratio.append(ratio.detach())
                if not latent_terms:
                    break
                latent_loss = latent_ppo_weight * (sum(latent_terms) / float(len(latent_terms)))
                if latent_loss.requires_grad:
                    latent_loss.backward()
                    lgnorm = torch.nn.utils.clip_grad_norm_(
                        [p for p in net_for_load.parameters() if p.requires_grad],
                        max_norm=grad_clip_norm,
                    )
                    optimizer.step()
                    latent_grad_norm_history.append(float(lgnorm.item() if hasattr(lgnorm, 'item') else float(lgnorm)))
                latent_policy_loss_history.append(float(latent_loss.detach().item()))
                latent_kl = float(np.mean(ep_latent_kl)) if ep_latent_kl else 0.0
                latent_kl_history.append(latent_kl)
                if ep_latent_ratio:
                    latent_ratio_history.append(torch.cat(ep_latent_ratio))
                if latent_kl > ppo_kl_target:
                    break

        if latent_ranker and latent_ranker_weight > 0.0:
            ranker_scores = []
            for traj in rollouts:
                latent_x0s = traj.get('latent_x0s', [])
                latent_sampled_feats = traj.get('latent_sampled_feats', [])
                if latent_x0s and latent_sampled_feats:
                    score_parts = [
                        gcn.latent_ranker_score(sf_j, x0_j)
                        for x0_j, sf_j in zip(latent_x0s, latent_sampled_feats)
                    ]
                    ranker_scores.append(torch.stack(score_parts, dim=0).sum(dim=0))
                else:
                    latent_x0 = traj.get('latent_x0', None)
                    latent_sampled_feat = traj.get('latent_sampled_feat', None)
                    if latent_x0 is None or latent_sampled_feat is None:
                        continue
                    ranker_scores.append(gcn.latent_ranker_score(latent_sampled_feat, latent_x0))
            if len(ranker_scores) == len(rollouts):
                score_stack = torch.stack(ranker_scores, dim=0)  # (K, B)
                target = quality_scores.detach()
                target_top = target.argmax(dim=0)
                if latent_ranker_top1:
                    ranker_loss = F.cross_entropy(score_stack.transpose(0, 1), target_top)
                else:
                    temp = max(float(latent_ranker_temp), 1e-6)
                    target_prob = torch.softmax(target / temp, dim=0)
                    pred_logprob = torch.log_softmax(score_stack, dim=0)
                    ranker_loss = -(target_prob * pred_logprob).sum(dim=0).mean()
                optimizer.zero_grad(set_to_none=True)
                (latent_ranker_weight * ranker_loss).backward()
                rgnorm = torch.nn.utils.clip_grad_norm_(
                    [p for p in net_for_load.parameters() if p.requires_grad],
                    max_norm=grad_clip_norm,
                )
                optimizer.step()
                ranker_loss_val = float(ranker_loss.detach().item())
                ranker_grad_norm = float(rgnorm.item() if hasattr(rgnorm, 'item') else float(rgnorm))
                with torch.no_grad():
                    ranker_top1_acc = float(
                        (score_stack.argmax(dim=0) == target_top).float().mean().item()
                    )

        # ---- decay action std
        action_std = max(action_std_min, action_std * action_std_decay)

        # ---- Trajectory distillation (best-of-K or advantage-weighted multi-rollout)
        # PPO's policy-gradient signal is very small for this already-strong model.
        # When exploration finds rollouts that improve over the init, distill their
        # final displacements into the flow-matching denoiser.
        #
        # adv_distill=0: Best-of-K – distill only the top rollout per batch element.
        # adv_distill=1: Advantage-weighted – distill ALL positive-advantage rollouts,
        #   each weighted by its normalised advantage.  Provides ~K/2 independent
        #   gradient targets per step, giving a substantially stronger learning signal.
        distill_loss_val = 0.0
        distill_active_frac = 0.0
        distill_grad_norm = 0.0
        if distill_weight > 0 and disp_stack.numel() > 0:
            if adv_distill:
                # ---- Advantage-weighted multi-rollout distillation
                # Process each contour b independently, batching K_pos rollouts
                # for that contour in a single forward pass.
                B_eff = disp_stack.size(1)
                contour_scale = gcn.compute_contour_scale(i_init).detach()  # (B, 1, 1)
                with torch.no_grad():
                    distill_ctx = gcn.prepare_sampling_context(cnn_feature.detach(), i_init, py_ind)
                loss_terms = []
                n_active_b = 0
                for b in range(B_eff):
                    if distill_compare_det and det_scores is not None:
                        gain_b = final_scores[:, b] - det_scores[b] - distill_det_margin
                    else:
                        gain_b = delta_scores[:, b] - distill_min_delta
                    quality_mask_b = (gain_b > 0)
                    if not bool(quality_mask_b.any().item()):
                        continue
                    pos_k = quality_mask_b.nonzero(as_tuple=True)[0]
                    if len(pos_k) == 0:
                        continue
                    K_pos = len(pos_k)
                    n_active_b += K_pos
                    pos_gain_raw_b = gain_b[pos_k].detach()
                    gain_scale_b = pos_gain_raw_b.mean()
                    pos_gain_b = (pos_gain_raw_b / pos_gain_raw_b.sum().clamp_min(1e-6))  # (K_pos,)
                    # K_pos rollout targets for contour b
                    pos_disps_b = disp_stack[pos_k, b].detach()  # (K_pos, N, 2)
                    cs_b = contour_scale[b:b+1].expand(K_pos, -1, -1)  # (K_pos, 1, 1)
                    x1_b = gcn.normalize_target_disp(pos_disps_b, cs_b).detach()
                    x0_b = gcn.sample_train_x0(x1_b).detach()
                    t_b = _sample_distill_t(K_pos, device=base_device, dtype=x1_b.dtype)
                    x_t_b = (1.0 - t_b) * x0_b + t_b * x1_b  # (K_pos, N, 2)
                    # Tile per-contour context K_pos times (same init polygon, K_pos noisy interpolants)
                    i_init_b = i_init[b:b+1].expand(K_pos, -1, -1)
                    c_init_b = (c_init[b:b+1].expand(K_pos, -1, -1)
                                if c_init is not None else None)
                    py_ind_b = py_ind[b:b+1].expand(K_pos)
                    sf_b = distill_ctx['sampled_feat'][b:b+1].expand(K_pos, -1, -1)
                    df_b = (distill_ctx['detail_feat'][b:b+1].expand(K_pos, -1, -1)
                            if distill_ctx['detail_feat'] is not None else None)
                    cs_flat_b = contour_scale.view(-1)[b:b+1].expand(K_pos).to(dtype=x1_b.dtype)
                    v_pred_b, _ = gcn.predict_velocity(
                        cnn_feature.detach(), i_init_b, c_init_b,
                        sf_b, df_b, py_ind_b,
                        x_t_b, t_b.view(-1),
                        contour_scale=cs_flat_b, x_self_cond=None,
                    )
                    v_target_b = (x1_b - x0_b).detach()
                    # Per-rollout MSE, weighted by the rollout's true quality gain
                    # over the comparison baseline (det or init).
                    mse_per_b = ((v_pred_b - v_target_b) ** 2).mean(dim=-1).mean(dim=-1)  # (K_pos,)
                    loss_terms.append((pos_gain_b * mse_per_b).sum() * gain_scale_b)
                distill_active_frac = float(n_active_b) / max(len(rollouts) * B_eff, 1)
                if loss_terms:
                    distill_loss = sum(loss_terms) / len(loss_terms)
                    if distill_loss_clip > 0.0:
                        clip_scale = distill_loss_clip / distill_loss.detach().clamp_min(distill_loss_clip)
                        distill_loss = distill_loss * clip_scale
                    optimizer.zero_grad(set_to_none=True)
                    (distill_weight * distill_loss).backward()
                    dgnorm = torch.nn.utils.clip_grad_norm_(
                        [p for p in net_for_load.parameters() if p.requires_grad],
                        max_norm=grad_clip_norm,
                    )
                    optimizer.step()
                    distill_loss_val = float(distill_loss.detach().item())
                    distill_grad_norm = float(dgnorm.item() if hasattr(dgnorm, 'item') else float(dgnorm))
            else:
                # ---- Best-of-K trajectory distillation (original path)
                best_idx = torch.argmax(final_scores, dim=0)  # (B,)
                arange_b = torch.arange(disp_stack.size(1), device=base_device)
                target_disp = disp_stack[best_idx, arange_b].detach()
                # Gating: only distill trajectories that genuinely beat the current
                # deterministic model (when det baseline is available), otherwise fall
                # back to the original YOLO-init delta gate.
                if distill_compare_det and det_scores is not None:
                    active = (final_scores[best_idx, arange_b] > det_scores + distill_det_margin).detach()
                else:
                    active = (delta_scores[best_idx, arange_b] > distill_min_delta).detach()
                distill_active_frac = float(active.float().mean().item()) if active.numel() else 0.0
                if active.any():
                    contour_scale = gcn.compute_contour_scale(i_init).detach()
                    x1 = gcn.normalize_target_disp(target_disp, contour_scale).detach()
                    n_active = int(active.sum().item())
                    t = _sample_distill_t(i_init.size(0), device=base_device, dtype=x1.dtype)
                    x0 = gcn.sample_train_x0(x1).detach()
                    x_t = (1.0 - t) * x0 + t * x1
                    contour_scale_flat = contour_scale.view(-1).to(device=base_device, dtype=x1.dtype)
                    with torch.no_grad():
                        distill_ctx = gcn.prepare_sampling_context(cnn_feature.detach(), i_init, py_ind)
                    v_pred, l_reg = gcn.predict_velocity(
                        cnn_feature.detach(),
                        i_init,
                        c_init,
                        distill_ctx['sampled_feat'],
                        distill_ctx['detail_feat'],
                        py_ind,
                        x_t,
                        t.view(-1),
                        contour_scale=contour_scale_flat,
                        x_self_cond=None,
                    )
                    v_target = (x1 - x0).detach()
                    distill_loss = F.mse_loss(v_pred[active], v_target[active], reduction='mean')
                    if isinstance(l_reg, torch.Tensor):
                        distill_loss = distill_loss + l_reg.mean() * 0.0
                    # Clip to prevent catastrophic updates from outlier batches
                    if distill_loss_clip > 0.0:
                        clip_scale = distill_loss_clip / distill_loss.detach().clamp_min(distill_loss_clip)
                        distill_loss = distill_loss * clip_scale
                    optimizer.zero_grad(set_to_none=True)
                    (distill_weight * distill_loss).backward()
                    dgnorm = torch.nn.utils.clip_grad_norm_(
                        [p for p in net_for_load.parameters() if p.requires_grad],
                        max_norm=grad_clip_norm,
                    )
                    optimizer.step()
                    distill_loss_val = float(distill_loss.detach().item())
                    distill_grad_norm = float(dgnorm.item() if hasattr(dgnorm, 'item') else float(dgnorm))

        # ---- compute logging
        ratio_all = torch.cat(ratio_history) if ratio_history else torch.zeros(1, device=base_device)
        latent_ratio_all = torch.cat(latent_ratio_history) if latent_ratio_history else torch.ones(1, device=base_device)
        log_item = {
            'timestamp': datetime.datetime.now().isoformat(),
            'step': int(step),
            'reward_mean': float(rewards.mean().item()),
            'reward_std': float(rewards.std(unbiased=False).item()),
            'reward_best': float(rewards.max(dim=0)[0].mean().item()),
            'reward_p10': percentiles(rewards.view(-1))['p10'],
            'reward_p50': percentiles(rewards.view(-1))['p50'],
            'reward_p90': percentiles(rewards.view(-1))['p90'],
            'final_score_mean': float(final_scores.mean().item()),
            'final_score_best': float(final_scores.max(dim=0)[0].mean().item()),
            'delta_score_mean': float(delta_scores.mean().item()),
            'burr_penalty_mean': float(burr_penalties.mean().item()),
            'burr_penalty_best': float(burr_penalties.min(dim=0)[0].mean().item()),
            'burr_raw_px_mean': float(burr_raw_px.mean().item()),
            'reward_burr_weight': float(reward_burr_weight),
            'ema_reward': float(ema_now),
            'adv_mean': float(advantages.mean().item()),
            'adv_std': float(advantages.std(unbiased=False).item()),
            'adv_max': float(advantages.abs().max().item()),
            'approx_kl_first': approx_kl_history[0] if approx_kl_history else 0.0,
            'approx_kl_last': approx_kl_history[-1] if approx_kl_history else 0.0,
            'clipfrac_first': clipfrac_history[0] if clipfrac_history else 0.0,
            'clipfrac_last': clipfrac_history[-1] if clipfrac_history else 0.0,
            'ratio_min': float(ratio_all.min().item()) if ratio_all.numel() > 0 else 1.0,
            'ratio_max': float(ratio_all.max().item()) if ratio_all.numel() > 0 else 1.0,
            'ratio_mean': float(ratio_all.mean().item()) if ratio_all.numel() > 0 else 1.0,
            'policy_loss': float(np.mean(policy_loss_history)) if policy_loss_history else 0.0,
            'kl_loss': float(np.mean(kl_loss_history)) if kl_loss_history else 0.0,
            'grad_norm': float(np.mean(grad_norm_history)) if grad_norm_history else 0.0,
            'latent_policy': int(latent_policy),
            'latent_policy_loss': float(np.mean(latent_policy_loss_history)) if latent_policy_loss_history else 0.0,
            'latent_kl_last': latent_kl_history[-1] if latent_kl_history else 0.0,
            'latent_ratio_min': float(latent_ratio_all.min().item()) if latent_ratio_all.numel() > 0 else 1.0,
            'latent_ratio_max': float(latent_ratio_all.max().item()) if latent_ratio_all.numel() > 0 else 1.0,
            'latent_ratio_mean': float(latent_ratio_all.mean().item()) if latent_ratio_all.numel() > 0 else 1.0,
            'latent_grad_norm': float(np.mean(latent_grad_norm_history)) if latent_grad_norm_history else 0.0,
            'latent_ranker': int(latent_ranker),
            'ranker_loss': ranker_loss_val,
            'ranker_grad_norm': ranker_grad_norm,
            'ranker_top1_acc': ranker_top1_acc,
            'inner_epochs': int(early_stop_epoch),
            'action_std': float(action_std),
            'k_rollouts': len(rollouts),
            'step_log_count_mean': float(np.mean(step_log_count_list)) if step_log_count_list else 0.0,
            'gate_active_frac': gate_active_frac,
            'distill_loss': distill_loss_val,
            'distill_active_frac': distill_active_frac,
            'distill_grad_norm': distill_grad_norm,
            'det_score_mean': float(det_scores.mean().item()) if det_scores is not None else 0.0,
        }

        # ---- periodic eval averaged over all fixed eval batches
        if eval_batches and (step % eval_every == 0 or step == 1):
            try:
                em = _eval_fixed_batches()
                if em:
                    log_item.update(em)
                    ema_eval_iou.update(log_item['eval_iou'])
                    log_item['ema_eval_iou'] = float(ema_eval_iou.value)
                    log_item['eval_delta_vs_baseline'] = float(log_item['eval_iou'] - fixed_eval_baseline_iou)
                    # track best; since best_eval_iou starts at the fixed-set
                    # baseline, best_iou.pt is only written after a real gain.
                    if log_item['eval_iou'] > best_eval_iou:
                        best_eval_iou = log_item['eval_iou']
                        log_item['is_best_iou'] = 1
                        # save best ckpt
                        w = net_for_load
                        sd_to_save = {k: v for k, v in w.state_dict().items()
                                      if not (k.startswith('ref_flow.') or k.startswith('ref_gcn.'))}
                        _safe_torch_save({'state_dict': sd_to_save, 'step': step,
                                          'eval_iou': best_eval_iou},
                                         ckpt_dir / 'best_iou.pt')
            except Exception as e:
                print(f'[GRPO-V2] eval failed: {e}')

        # ---- write log
        if step % log_every == 0:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_item, ensure_ascii=False) + '\n')

        if step % 20 == 0 or step == 1:
            extra = ''
            if 'eval_iou' in log_item:
                extra = f" eval_iou={log_item['eval_iou']:.4f} Δ={log_item.get('eval_delta_vs_baseline',0):+.4f}"
            print(
                f"[V2 step {step:5d}] reward={log_item['reward_mean']:.4f} "
                f"final={log_item['final_score_mean']:.4f} "
                f"burr={log_item['burr_penalty_mean']:.3f} "
                f"dl={log_item['distill_loss']:.5f} dg={log_item['distill_grad_norm']:.4f} "
                f"af={log_item['distill_active_frac']:.2f} "
                f"kl={log_item['approx_kl_first']:.4f} "
                f"gnorm={log_item['grad_norm']:.3f} "
                f"std={action_std:.3f}{extra}"
            )

        # ---- viz: dump small trajectory tape for fixed eval batch
        if eval_batch is not None and viz_every > 0 and (step % viz_every == 0 or step == 1):
            try:
                _dump_train_group_viz(batch, output, disp_stack, rewards, final_scores,
                                      det_scores, quality_scores, gate_mask, viz_dir, step)
                if dump_trajectory_viz:
                    _dump_trajectory_viz(inner, eval_batch, viz_dir, step,
                                         rollout_steps=rollout_steps, k_viz=min(k, 8),
                                         action_std=action_std,
                                         step_mode=step_mode,
                                         noise_level=noise_level,
                                         sde_type=sde_type,
                                         reward_w_region=reward_w_region,
                                         reward_w_dice=reward_w_dice,
                                         reward_w_iou=reward_w_iou,
                                         reward_w_dist=reward_w_dist,
                                         reward_dist_max_px=reward_dist_max_px,
                                         reward_dist_quantile=reward_dist_quantile,
                                         reward_dist_quantile_weight=reward_dist_quantile_weight,
                                         reward_abs_w=reward_abs_w,
                                         reward_delta_w=reward_delta_w)
            except Exception as e:
                print(f'[GRPO-V2] viz failed: {e}')

        # ---- save ckpt
        if save_every > 0 and step % save_every == 0:
            w = net_for_load
            state_dict = {k: v for k, v in w.state_dict().items() if not (k.startswith('ref_flow.') or k.startswith('ref_gcn.'))}
            ckpt = {
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
                'step': int(step),
                'timestamp': datetime.datetime.now().isoformat(),
                'cfg_file': _cfg_file_used(),
                'base_ckpt': str(ckpt_path),
                'posttrain': {'use_grpo_v2': True},
            }
            _safe_torch_save(ckpt, ckpt_dir / 'latest.pt')
            _safe_torch_save(ckpt, ckpt_dir / f'step{step}.pt')
            print(f'[GRPO-V2] saved ckpt at step {step}')

        # ---- cleanup
        del output, batch, rollouts, old_logs_list, rewards, advantages
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.no_grad()
def _dump_train_group_viz(batch, output, disp_stack, rewards, final_scores,
                          det_scores, quality_scores, gate_mask, viz_dir: Path, step: int,
                          max_rollouts: int = 8):
    """Visualize the actual training group collected in the current step."""
    import cv2

    viz_dir.mkdir(parents=True, exist_ok=True)
    i_init = output.get('i_it_py')
    oct_init = output.get('octagon_i_it_py')
    mid_frac = output.get('train_midstate_frac')
    i_gt = output.get('i_gt_py')
    py_ind = output.get('py_ind')
    if not isinstance(i_init, torch.Tensor) or not isinstance(disp_stack, torch.Tensor):
        return
    if not isinstance(i_gt, torch.Tensor) or i_gt.numel() == 0:
        return

    i_gt = _align_gt(i_init, i_gt)
    if isinstance(py_ind, torch.Tensor) and py_ind.numel() == i_init.shape[0]:
        img_mask = (py_ind.detach().long().view(-1) == 0)
    else:
        img_mask = torch.ones((i_init.shape[0],), device=i_init.device, dtype=torch.bool)
    if not bool(img_mask.any().item()):
        return

    inp = batch['inp'][0].detach().float().cpu().numpy()
    if inp.shape[0] in (1, 3):
        inp = inp.transpose(1, 2, 0)
    inp = inp - inp.min()
    if inp.max() > 0:
        inp = inp / inp.max()
    img = (inp * 255.0).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    base_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    base_img = cv2.addWeighted(base_img, 0.45, np.zeros_like(base_img), 0.55, 0)
    H, W = base_img.shape[:2]

    scale = float(snake_config.down_ratio)
    mask_np = img_mask.detach().cpu().numpy().astype(bool)
    init_np = i_init.detach().cpu().numpy()[mask_np] * scale
    oct_np = None
    init_oct_dist_px = None
    if isinstance(oct_init, torch.Tensor) and oct_init.shape == i_init.shape:
        oct_np = oct_init.detach().cpu().numpy()[mask_np] * scale
        init_oct_dist_px = float(np.mean(np.linalg.norm(init_np - oct_np, axis=-1))) if init_np.size else None
    gt_np = i_gt.detach().cpu().numpy()[mask_np] * scale
    disp_np = disp_stack.detach().cpu().numpy()[:, mask_np] * scale
    final_np = init_np[None, ...] + disp_np

    rewards_np = rewards.detach().cpu().numpy()[:, mask_np]
    final_scores_np = final_scores.detach().cpu().numpy()[:, mask_np]
    quality_np = quality_scores.detach().cpu().numpy()[:, mask_np]
    gate_np = gate_mask.detach().cpu().numpy().reshape(-1)
    gate_sel = gate_np[mask_np] if gate_np.size == mask_np.size else gate_np
    det_np = None
    if isinstance(det_scores, torch.Tensor):
        det_np = det_scores.detach().cpu().numpy()[mask_np]

    k_total = final_np.shape[0]
    k_show = min(int(max_rollouts), k_total)
    rollout_order = list(range(k_show))
    if k_total > k_show:
        mean_quality = quality_np.mean(axis=1)
        best = int(np.argmax(mean_quality))
        worst = int(np.argmin(mean_quality))
        rollout_order = []
        for idx in list(range(k_show - 2)) + [best, worst]:
            if idx not in rollout_order:
                rollout_order.append(idx)
        rollout_order = rollout_order[:k_show]

    best_idx = int(np.argmax(quality_np.mean(axis=1)))
    worst_idx = int(np.argmin(quality_np.mean(axis=1)))

    def draw_polys(canvas, polys, color, thickness):
        for m in range(polys.shape[0]):
            pts = np.round(polys[m]).astype(np.int32)
            if pts.shape[0] < 2:
                continue
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            loop = np.concatenate([pts, pts[:1]], axis=0)
            cv2.polylines(canvas, [loop], isClosed=True, color=color, thickness=thickness)

    panels = []
    for ri in rollout_order:
        canvas = base_img.copy()
        draw_polys(canvas, gt_np, (255, 0, 0), 3)
        if oct_np is not None:
            draw_polys(canvas, oct_np, (0, 180, 255), 1)
        draw_polys(canvas, init_np, (255, 255, 0), 2)
        draw_polys(canvas, final_np[ri], (255, 255, 255), 3)
        if ri == best_idx:
            border = (80, 220, 80)
            tag = 'best'
        elif ri == worst_idx:
            border = (40, 40, 255)
            tag = 'worst'
        else:
            border = (90, 90, 90)
            tag = 'mid'
        cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), border, 4)
        label = (
            f"real k={ri} {tag} "
            f"R={float(rewards_np[ri].mean()):.3f} "
            f"Q={float(quality_np[ri].mean()):+.3f} "
            f"F={float(final_scores_np[ri].mean()):.3f}"
        )
        cv2.rectangle(canvas, (2, 2), (min(W - 2, 430), 29), (0, 0, 0), -1)
        cv2.putText(canvas, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 255, 0), 1, cv2.LINE_AA)
        panels.append(canvas)

    if not panels:
        return
    tape = np.concatenate(panels, axis=1)
    legend = np.zeros((30, tape.shape[1], 3), dtype=np.uint8)
    gate_frac = float(np.mean(gate_sel)) if gate_sel.size else 0.0
    det_text = f"det={float(np.mean(det_np)):.3f}" if det_np is not None and det_np.size else "det=NA"
    cv2.putText(
        legend,
        f"REAL TRAIN GROUP step={step} | cyan:init orange:oct white:final blue:GT | {det_text} gate_frac={gate_frac:.2f}",
        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    tape = np.concatenate([legend, tape], axis=0)
    cv2.imwrite(str(viz_dir / f'train_group_step{step:06d}.png'), tape)

    meta = {
        'step': int(step),
        'type': 'actual_training_group',
        'image_index': 0,
        'num_contours_drawn': int(mask_np.sum()),
        'init_source': 'train_midstate' if init_oct_dist_px is not None else 'raw_i_it_py',
        'mean_init_to_octagon_px': init_oct_dist_px,
        'train_midstate_frac_mean': None if not isinstance(mid_frac, torch.Tensor) else float(mid_frac.detach().mean().item()),
        'train_midstate_frac_min': None if not isinstance(mid_frac, torch.Tensor) else float(mid_frac.detach().min().item()),
        'train_midstate_frac_max': None if not isinstance(mid_frac, torch.Tensor) else float(mid_frac.detach().max().item()),
        'k_total': int(k_total),
        'k_shown': [int(x) for x in rollout_order],
        'best_rollout_by_quality_mean': int(best_idx),
        'worst_rollout_by_quality_mean': int(worst_idx),
        'gate_active_frac_drawn_image': gate_frac,
        'det_score_mean_drawn_image': None if det_np is None or not det_np.size else float(np.mean(det_np)),
        'rollouts': [
            {
                'k': int(ri),
                'reward_mean': float(rewards_np[ri].mean()),
                'final_score_mean': float(final_scores_np[ri].mean()),
                'quality_vs_det_mean': float(quality_np[ri].mean()),
            }
            for ri in range(k_total)
        ],
    }
    with open(viz_dir / f'train_group_step{step:06d}.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def _dump_trajectory_viz(inner, eval_batch, viz_dir: Path, step: int,
                         rollout_steps: int, k_viz: int, action_std: float,
                         step_mode: str, noise_level: float, sde_type: str,
                         reward_w_region: float, reward_w_dice: float, reward_w_iou: float,
                         reward_w_dist: float = 0.0,
                         reward_dist_max_px: float = 8.0,
                         reward_dist_quantile: float = 95.0,
                         reward_dist_quantile_weight: float = 0.5,
                         reward_abs_w: float = 1.0,
                         reward_delta_w: float = 0.0):
    """Render a compact group comparison: one panel per rollout.

    The diagnostic target is reward ranking, so the panel intentionally shows
    only init / final / GT. Dense intermediate ODE trajectories are too noisy
    once rollouts are close to each other.
    """
    import cv2
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    was_training = inner.training
    inner.train()  # need GT processing branch to populate i_gt_py
    freeze_bn_running_stats(inner)  # but keep BN frozen
    with torch.no_grad():
        out = inner(eval_batch['inp'], eval_batch)
    cnn_feature = out.get('cnn_feature')
    i_init = out.get('i_it_py')
    c_init = out.get('c_it_py')
    py_ind = out.get('py_ind')
    i_gt = out.get('i_gt_py', None)
    if (not isinstance(i_init, torch.Tensor)) or i_init.numel() == 0:
        if was_training:
            inner.train()
        else:
            inner.eval()
        return
    if isinstance(i_gt, torch.Tensor) and i_gt.numel() > 0:
        i_gt = _align_gt(i_init, i_gt)
    else:
        i_gt = None

    # base image (CHW float in [0,1] typically)
    inp = eval_batch['inp'][0].detach().float().cpu().numpy()
    if inp.shape[0] in (1, 3):
        inp = inp.transpose(1, 2, 0)
    inp = inp - inp.min()
    if inp.max() > 0:
        inp = inp / inp.max()
    inp_img = (inp * 255.0).astype(np.uint8)
    if inp_img.ndim == 2:
        inp_img = cv2.cvtColor(inp_img, cv2.COLOR_GRAY2BGR)
    elif inp_img.shape[-1] == 1:
        inp_img = cv2.cvtColor(inp_img, cv2.COLOR_GRAY2BGR)
    elif inp_img.shape[-1] == 3:
        inp_img = cv2.cvtColor(inp_img, cv2.COLOR_RGB2BGR)
    H, W = inp_img.shape[:2]
    gray = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    inp_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    inp_img = cv2.addWeighted(inp_img, 0.45, np.zeros_like(inp_img), 0.55, 0)

    # We need full-trajectory contour evolution. The flow `sample_with_logprob`
    # already records `latents` covering the policy window. To get *all* ODE
    # steps regardless of window, do a fresh manual rollout that snapshots
    # contour each step.
    contour_evolutions = []  # list over k, each: list of (M,P,2) per ODE step
    final_polys = []
    rewards = []
    if i_gt is not None:
        init_score = compute_region_score(
            i_init, i_gt, H=H, W=W,
            w_boundary=reward_w_region, w_dice=reward_w_dice, w_iou=reward_w_iou,
            w_dist=reward_w_dist,
            dist_max_px=reward_dist_max_px,
            dist_quantile=reward_dist_quantile,
            dist_quantile_weight=reward_dist_quantile_weight,
            coord_scale=float(snake_config.down_ratio),
        ).detach().cpu().numpy()
    else:
        init_score = None

    gcn = inner.gcn
    ctx = gcn.prepare_sampling_context(cnn_feature, i_init, py_ind)
    for ki in range(k_viz):
        x = torch.randn_like(i_init)
        contour_seq = []
        x_self_cond = torch.zeros_like(x) if getattr(gcn, '_use_self_conditioning', False) else None
        dt = 1.0 / float(rollout_steps)
        for idx in range(rollout_steps):
            t_value = idx * dt
            x_prev, _, _, _, next_self_cond = gcn.step_with_logprob(
                cnn_feature, i_init, c_init, py_ind,
                x_t=x, t_value=t_value, step_index=idx,
                total_steps=rollout_steps,
                action_std=(action_std if ki > 0 else 0.0),
                prev_sample=None,
                sampled_feat=ctx['sampled_feat'],
                detail_feat=ctx['detail_feat'],
                contour_scale=ctx['contour_scale'],
                x_self_cond=x_self_cond,
                step_mode=step_mode,
                noise_level=noise_level,
                sde_type=sde_type,
            )
            # convert intermediate x to contour by denormalize_pred_disp
            disp_int = gcn.denormalize_pred_disp(x_prev, ctx['contour_scale'])
            poly_int = (i_init + disp_int).detach().cpu().numpy() * float(snake_config.down_ratio)
            contour_seq.append(poly_int)
            x = x_prev
            if x_self_cond is not None:
                x_self_cond = next_self_cond
        disp_final = gcn.denormalize_pred_disp(x, ctx['contour_scale'])
        if i_gt is not None:
            final_score = compute_region_reward(
                i_init, disp_final, i_gt, H=H, W=W,
                w1=reward_w_region, w_dice=reward_w_dice, w_iou=reward_w_iou,
                w_dist=reward_w_dist,
                dist_max_px=reward_dist_max_px,
                dist_quantile=reward_dist_quantile,
                dist_quantile_weight=reward_dist_quantile_weight,
                coord_scale=float(snake_config.down_ratio),
            ).detach().cpu().numpy()
            train_reward = reward_abs_w * final_score + reward_delta_w * (final_score - init_score)
            rewards.append(float(np.mean(train_reward)))
        else:
            rewards.append(float('nan'))
        contour_evolutions.append(contour_seq)
        final_polys.append((i_init + disp_final).detach().cpu().numpy() * float(snake_config.down_ratio))

    # --- draw the tape: one panel per rollout
    panels = []
    init_np = i_init.detach().cpu().numpy() * float(snake_config.down_ratio)
    gt_np = i_gt.detach().cpu().numpy() * float(snake_config.down_ratio) if i_gt is not None else None
    valid_rewards = [r for r in rewards if not np.isnan(r)]
    best_idx = int(np.nanargmax(np.asarray(rewards))) if valid_rewards else -1
    worst_idx = int(np.nanargmin(np.asarray(rewards))) if valid_rewards else -1
    for ki in range(k_viz):
        img = inp_img.copy()
        # initial (cyan)
        for m in range(init_np.shape[0]):
            pts = init_np[m].astype(np.int32)
            loop = np.concatenate([pts, pts[:1]], axis=0)
            cv2.polylines(img, [loop], isClosed=True, color=(255, 255, 0), thickness=2)
        # GT (blue) - only if available
        if gt_np is not None:
            for m in range(gt_np.shape[0]):
                pts = gt_np[m].astype(np.int32)
                loop = np.concatenate([pts, pts[:1]], axis=0)
                cv2.polylines(img, [loop], isClosed=True, color=(255, 0, 0), thickness=3)
        # final (white)
        fin = final_polys[ki]
        for m in range(fin.shape[0]):
            pts = fin[m].astype(np.int32)
            loop = np.concatenate([pts, pts[:1]], axis=0)
            cv2.polylines(img, [loop], isClosed=True, color=(255, 255, 255), thickness=3)
        border_color = (80, 220, 80) if ki == best_idx else ((40, 40, 255) if ki == worst_idx else (80, 80, 80))
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), border_color, 4)
        tag = 'best' if ki == best_idx else ('worst' if ki == worst_idx else 'mid')
        label = f"k={ki} {tag} reward={rewards[ki]:.3f}"
        cv2.rectangle(img, (2, 2), (330, 26), (0, 0, 0), -1)
        cv2.putText(img, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        panels.append(img)

    if len(panels) == 0:
        return
    # concat horizontally
    target_h = max(p.shape[0] for p in panels)
    resized = []
    for p in panels:
        if p.shape[0] != target_h:
            scale = target_h / p.shape[0]
            p = cv2.resize(p, (int(p.shape[1] * scale), target_h))
        resized.append(p)
    tape = np.concatenate(resized, axis=1)
    # legend bar at top
    legend_h = 24
    legend = np.zeros((legend_h, tape.shape[1], 3), dtype=np.uint8)
    cv2.putText(legend, "cyan: init | white: rollout final | blue: GT | green border: best reward | red border: worst reward",
                (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    tape = np.concatenate([legend, tape], axis=0)
    out_path = viz_dir / f'traj_step{step:06d}.png'
    cv2.imwrite(str(out_path), tape)

    # also dump reward bar chart over rollouts
    if plt is not None and init_score is not None and not all(np.isnan(rewards)):
        fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
        ax.bar(range(len(rewards)), rewards, color='tab:orange')
        init_train_reward = reward_abs_w * init_score
        ax.axhline(float(np.mean(init_train_reward)), color='tab:gray', linestyle='--', label=f'init_abs={float(np.mean(init_train_reward)):.3f}')
        ax.set_title(f'step {step}: per-rollout training reward')
        ax.set_xlabel('rollout index'); ax.set_ylabel('reward')
        ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()
        fig.savefig(str(viz_dir / f'reward_step{step:06d}.png'), dpi=80)
        plt.close(fig)

    if was_training:
        inner.train()
    else:
        inner.eval()


if __name__ == '__main__':
    main()
