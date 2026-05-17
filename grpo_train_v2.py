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
    gpu_override = os.environ.get('GRPO_V2_GPU', '').strip()
    if gpu_override:
        cfg.gpus = [int(gpu_override)]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_override
        print(f'[GRPO-V2] Override GPU -> {gpu_override}')
    train_steps  = int(os.environ.get('GRPO_V2_STEPS', cv('train_steps', 1000)))
    k            = int(cv('k', 6))
    rollout_steps= int(cv('rollout_steps', getattr(cfg, 'grpo_steps', 20)))
    window_size  = int(cv('window_size', max(rollout_steps // 4, 3)))
    window_range = tuple(cv('window_range', (max(rollout_steps // 4, 1), rollout_steps)))
    action_std0  = float(cv('action_std', 0.15))
    action_std_min = float(cv('action_std_min', 0.05))
    action_std_decay = float(cv('action_std_decay', 0.9999))
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
    rollout_source = str(cv('rollout_source', 'manual_gt_init')).strip().lower()
    rollout_iterative = bool(cv('rollout_iterative', True))
    eval_every   = int(cv('eval_every', 50))
    viz_every    = int(cv('viz_every', 50))
    save_every   = int(cv('save_every', 100))
    log_every    = int(cv('log_every', 1))
    seed         = int(cv('seed', 20260515))

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
            'window_size': window_size, 'window_range': list(window_range),
            'action_std0': action_std0, 'action_std_min': action_std_min,
            'action_std_decay': action_std_decay,
            'ppo_inner_epochs': ppo_inner_epochs, 'ppo_clip': ppo_clip,
            'ppo_kl_target': ppo_kl_target, 'kl_beta': kl_beta,
            'adv_clip_max': adv_clip_max,
            'reward_abs_weight': reward_abs_w, 'reward_delta_weight': reward_delta_w,
            'reward_w_region': reward_w_region, 'reward_w_dice': reward_w_dice, 'reward_w_iou': reward_w_iou,
            'rollout_source': rollout_source, 'rollout_iterative': rollout_iterative,
            'distill_weight': distill_weight, 'distill_min_delta': distill_min_delta,
            'distill_compare_det': distill_compare_det, 'distill_det_margin': distill_det_margin,
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

    def _train_branch_context(batch):
        """Legacy V2 context; kept only for ablation/debugging."""
        net_for_load.train()
        freeze_bn_running_stats(net_for_load)
        with torch.no_grad():
            output = inner(batch['inp'], batch)
        return output

    def _forward_for_rollout(batch):
        if rollout_source in ('manual', 'manual_gt', 'manual_gt_init', 'eval_manual'):
            return _manual_gt_init_context(batch)
        if rollout_source in ('train', 'train_branch', 'legacy'):
            return _train_branch_context(batch)
        raise ValueError(f'Unknown grpo_v2_rollout_source={rollout_source!r}')

    @torch.no_grad()
    def _sample_rollout(output, action_std):
        cnn_feature = output['cnn_feature']
        i_init = output['i_it_py']
        c_init = output['c_it_py']
        py_ind = output['py_ind']
        if rollout_iterative and getattr(gcn, 'use_iterative_refinement', False):
            iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
            fractions = list(getattr(cfg, 'iterative_fractions', []))
            if not fractions:
                fractions = [1.0 / (iter_steps - i) for i in range(iter_steps)]
            iter_ode_steps = int(getattr(cfg, 'iterative_ode_steps', getattr(cfg, 'iterative_ddim_steps', rollout_steps)))
            if iter_ode_steps <= 0:
                iter_ode_steps = rollout_steps

            current = i_init.detach()
            total_disp = torch.zeros_like(i_init)
            merged = {
                'latents': [], 'log_probs': [], 'timesteps': [], 'step_indices': [],
                'x_ts': [], 'x_prevs': [], 'x_self_conds': [],
                'cnn_features': [], 'i_inits': [], 'c_inits': [], 'py_inds': [],
                'sampled_feats': [], 'detail_feats': [], 'contour_scales': [],
                'total_steps': [],
            }
            for frac in fractions[:iter_steps]:
                c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
                ret = gcn.sample_with_logprob(
                    cnn_feature, current, c_cur, py_ind,
                    steps=iter_ode_steps, window_size=window_size,
                    window_range=window_range, action_std=action_std,
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
                applied = ret['disp'] * float(frac)
                current = (current + applied).detach()
                total_disp = total_disp + applied
            merged['disp'] = total_disp
            merged['py'] = i_init + total_disp
            return merged

        ret = gcn.sample_with_logprob(
            cnn_feature, i_init, c_init, py_ind,
            steps=rollout_steps, window_size=window_size,
            window_range=window_range, action_std=action_std,
        )
        n_log = len(ret.get('log_probs', []))
        ret['cnn_features'] = [cnn_feature.detach()] * n_log
        ret['i_inits'] = [i_init.detach()] * n_log
        ret['c_inits'] = [c_init.detach()] * n_log
        ret['py_inds'] = [py_ind.detach()] * n_log
        ret['sampled_feats'] = [ret['sampled_feat'].detach()] * n_log
        ret['detail_feats'] = [None if ret.get('detail_feat') is None else ret['detail_feat'].detach()] * n_log
        ret['contour_scales'] = [ret['contour_scale'].detach()] * n_log
        ret['total_steps'] = [torch.tensor(rollout_steps, device=i_init.device, dtype=torch.long)] * n_log
        return ret

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
            coord_scale=float(snake_config.down_ratio),
        ).detach()
        final_score = compute_region_reward(
            i_init, ret['disp'], i_gt, H=H_img, W=W_img,
            w1=reward_w_region, w_dice=reward_w_dice, w_iou=reward_w_iou,
            coord_scale=float(snake_config.down_ratio),
        ).detach()
        delta = (final_score - init_score)
        reward = reward_abs_w * final_score + reward_delta_w * delta
        return {
            'final_score': final_score, 'init_score': init_score,
            'delta_score': delta, 'reward': reward,
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
        # keep BN running stats frozen (eval/train mode toggles can re-enable them)
        freeze_bn_running_stats(inner)
        # ---- get batch
        try:
            batch = next(it)
        except StopIteration:
            it = iter(data_loader)
            batch = next(it)
        _move_batch(batch)

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
        i_gt_aligned = _align_gt(i_init, i_gt)
        base_device = cnn_feature.device

        # ---- collect k rollouts (frozen old policy = current params)
        rollouts: List[Dict] = []
        rewards_list = []
        final_scores_list = []
        delta_scores_list = []
        disp_list = []
        old_logs_list = []  # (k, B, T)
        for _ in range(k):
            ret = _sample_rollout(output, action_std)
            if not isinstance(ret.get('log_probs', None), list) or len(ret['log_probs']) == 0:
                continue
            old_log = torch.stack(ret['log_probs'], dim=0).transpose(0, 1).contiguous().detach()
            rew = _compute_rewards(output, ret)
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
            })
            old_logs_list.append(old_log)
            rewards_list.append(rew['reward'])
            final_scores_list.append(rew['final_score'])
            delta_scores_list.append(rew['delta_score'])
            disp_list.append(ret['disp'].detach())

        if len(rollouts) == 0:
            print(f'[GRPO-V2] step {step}: zero valid rollouts, skipping.')
            continue

        rewards = torch.stack(rewards_list, dim=0).to(base_device)         # (k, B)
        final_scores = torch.stack(final_scores_list, dim=0).to(base_device)
        delta_scores = torch.stack(delta_scores_list, dim=0).to(base_device)
        disp_stack = torch.stack(disp_list, dim=0).to(base_device)

        # ---- Deterministic baseline for quality-gated distillation
        # Run one deterministic (action_std=0) rollout per step so we know the
        # current model's "floor" score. The distillation gate then ensures we
        # only distill stochastic trajectories that genuinely beat this floor.
        det_scores = None
        if distill_weight > 0 and distill_compare_det:
            try:
                det_ret = _sample_rollout(output, 0.0)
                rew_det = _compute_rewards(output, det_ret)
                det_scores = rew_det['final_score'].to(base_device).detach()  # (B,)
            except Exception as e:
                print(f'[GRPO-V2] step {step}: det-baseline failed: {e}')
                det_scores = None


        # EMA tracked for monitoring only; no longer used as a bias term, since
        # subtracting (batch_reward - ema_reward) injects a non-zero advantage
        # mean and pushes the policy in arbitrary directions when the batch
        # reward fluctuates.
        group_baseline = rewards.mean(dim=0, keepdim=True)
        ema_now = ema_reward.update(float(rewards.mean()))
        advantages = rewards - group_baseline
        # Dampened normalization: floor std so we don't amplify within-group noise
        adv_std = rewards.std(dim=0, unbiased=False, keepdim=True).clamp_min(0.1)
        advantages = (advantages / adv_std).clamp(-adv_clip_max, adv_clip_max).detach()

        # ---- Group-quality gate: zero advantage for batch indices where the
        # best rollout fails to improve over init by a positive margin. This
        # prevents the policy from being pushed toward the "least bad" sample
        # in groups where every rollout is worse than the supervised baseline.
        gate_margin = float(cfg.train.get('grpo_v2_gate_margin', 0.0))
        delta_best = delta_scores.max(dim=0, keepdim=True).values  # (1, B)
        gate_mask = (delta_best > gate_margin).float()  # (1, B)
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
        early_stop_epoch = ppo_inner_epochs
        for epoch in range(ppo_inner_epochs):
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

                    # KL to frozen ref (gaussian mean alignment)
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
                            )
                        var = std_cur.pow(2).clamp_min(1e-12)
                        kl_term = (((mean_cur - mean_ref) ** 2) / (2.0 * var)).mean()
                        loss_term = loss_term + kl_beta * kl_term / float(total_steps_in_epoch)
                        kl_loss_sum += float(kl_term.detach().item())
                        kl_terms += 1
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

        # ---- decay action std
        action_std = max(action_std_min, action_std * action_std_decay)

        # ---- Best-of-k trajectory distillation
        # PPO's policy-gradient signal is very small for this already-strong
        # model. When exploration finds a rollout that truly improves over the
        # deterministic init, directly distill its final displacement into the
        # flow-matching denoiser so the deterministic mean can inherit it.
        distill_loss_val = 0.0
        distill_active_frac = 0.0
        distill_grad_norm = 0.0
        if distill_weight > 0 and disp_stack.numel() > 0:
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
                t = gcn.sample_train_t(i_init.size(0), device=base_device, dtype=x1.dtype)
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
                    distill_loss = distill_loss.clamp(max=distill_loss_clip)
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
            'inner_epochs': int(early_stop_epoch),
            'action_std': float(action_std),
            'k_rollouts': len(rollouts),
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
                extra = f" eval_iou={log_item['eval_iou']:.4f} mboundf={log_item.get('eval_mboundf',0):.4f}"
            print(
                f"[V2 step {step:5d}] reward={log_item['reward_mean']:.4f} "
                f"final={log_item['final_score_mean']:.4f} "
                f"kl_first={log_item['approx_kl_first']:.4f} "
                f"clipfrac={log_item['clipfrac_first']:.3f} "
                f"ratio[{log_item['ratio_min']:.3f},{log_item['ratio_max']:.3f}] "
                f"gnorm={log_item['grad_norm']:.3f} "
                f"std={action_std:.3f}{extra}"
            )

        # ---- viz: dump small trajectory tape for fixed eval batch
        if eval_batch is not None and viz_every > 0 and (step % viz_every == 0 or step == 1):
            try:
                _dump_trajectory_viz(inner, eval_batch, viz_dir, step,
                                     rollout_steps=rollout_steps, k_viz=min(k, 4),
                                     action_std=action_std,
                                     reward_w_region=reward_w_region,
                                     reward_w_dice=reward_w_dice,
                                     reward_w_iou=reward_w_iou)
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
def _dump_trajectory_viz(inner, eval_batch, viz_dir: Path, step: int,
                         rollout_steps: int, k_viz: int, action_std: float,
                         reward_w_region: float, reward_w_dice: float, reward_w_iou: float):
    """Render a 'trajectory tape': for the eval batch, run k rollouts with the
    current policy, draw the contour ODE-step evolution on the image with a
    colour gradient (yellow→red along ODE steps), GT in blue, initial in cyan.

    Also dumps a deterministic prediction PNG and a small reward bar chart.
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
                coord_scale=float(snake_config.down_ratio),
            ).detach().cpu().numpy()
            rewards.append(float(np.mean(final_score)))
        else:
            rewards.append(float('nan'))
        contour_evolutions.append(contour_seq)
        final_polys.append((i_init + disp_final).detach().cpu().numpy() * float(snake_config.down_ratio))

    # --- draw the tape: one panel per rollout
    panels = []
    init_np = i_init.detach().cpu().numpy() * float(snake_config.down_ratio)
    gt_np = i_gt.detach().cpu().numpy() * float(snake_config.down_ratio) if i_gt is not None else None
    for ki in range(k_viz):
        img = inp_img.copy()
        seq = contour_evolutions[ki]
        T = len(seq)
        # ODE trajectory: colour gradient from yellow (early) to red (late)
        for tidx, poly in enumerate(seq):
            frac = tidx / max(T - 1, 1)
            color = (0, int(255 * (1 - frac)), 255)
            for m in range(poly.shape[0]):
                pts = poly[m].astype(np.int32)
                loop = np.concatenate([pts, pts[:1]], axis=0)
                cv2.polylines(img, [loop], isClosed=True, color=color, thickness=1)
        # initial (cyan)
        for m in range(init_np.shape[0]):
            pts = init_np[m].astype(np.int32)
            loop = np.concatenate([pts, pts[:1]], axis=0)
            cv2.polylines(img, [loop], isClosed=True, color=(255, 255, 0), thickness=1)
        # GT (blue) - only if available
        if gt_np is not None:
            for m in range(gt_np.shape[0]):
                pts = gt_np[m].astype(np.int32)
                loop = np.concatenate([pts, pts[:1]], axis=0)
                cv2.polylines(img, [loop], isClosed=True, color=(255, 0, 0), thickness=2)
        # final (white)
        fin = final_polys[ki]
        for m in range(fin.shape[0]):
            pts = fin[m].astype(np.int32)
            loop = np.concatenate([pts, pts[:1]], axis=0)
            cv2.polylines(img, [loop], isClosed=True, color=(255, 255, 255), thickness=2)
        cv2.putText(img, f"k={ki} r={rewards[ki]:.3f} std={action_std:.2f}",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
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
    cv2.putText(legend, "yellow->red: ODE step early->late | cyan: init | white: final | blue: GT",
                (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    tape = np.concatenate([legend, tape], axis=0)
    out_path = viz_dir / f'traj_step{step:06d}.png'
    cv2.imwrite(str(out_path), tape)

    # also dump reward bar chart over rollouts
    if plt is not None and init_score is not None and not all(np.isnan(rewards)):
        fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
        ax.bar(range(len(rewards)), rewards, color='tab:orange')
        ax.axhline(float(np.mean(init_score)), color='tab:gray', linestyle='--', label=f'init={float(np.mean(init_score)):.3f}')
        ax.set_title(f'step {step}: per-rollout reward')
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
