#!/usr/bin/env python3
"""Build a training-split difficulty index for RL hard-sample focusing."""

from __future__ import annotations

import datetime
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict

_THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS_DIR))

_DEFAULT_CFG = _THIS_DIR / 'configs' / '1232_final_v7_8_antiregression_gpu0.yaml'
if not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = str(_DEFAULT_CFG)

eval_gpu = os.environ.get('EVAL_GPU', '').strip()
if eval_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = eval_gpu

import numpy as np
import torch

from lib.config import cfg
from lib.datasets.collate_batch import make_collator
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.networks import make_network
from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
from lib.train.rewards.region_reward import compute_region_score
from lib.train.trainers import make_trainer
from lib.utils.snake import snake_config, snake_gcn_utils


def _project_path(p) -> Path:
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else _THIS_DIR / p


def _extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ('state_dict', 'model', 'net', 'network'):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
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


def _move_batch(batch: Dict, device='cuda'):
    for key in list(batch.keys()):
        if key in ('meta', 'orig_img', 'img_path'):
            continue
        value = batch[key]
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)
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


def _cfg_value(name, default):
    v4_cfg = getattr(cfg, 'rl_v4', None)
    if v4_cfg is not None and name in v4_cfg:
        return v4_cfg[name]
    return getattr(cfg, f'rl_v4_{name}', default)


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


def _load_model(ckpt_path: Path):
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    wrapper = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    sd = _adapt_state_dict(wrapper, _extract_state_dict(torch.load(str(ckpt_path), map_location='cpu')))
    info = wrapper.load_state_dict(sd, strict=False)
    total = len(list(wrapper.state_dict().keys()))
    load_ratio = 100.0 * (total - len(info.missing_keys)) / max(total, 1)
    print(
        f'[*] Loaded checkpoint: {ckpt_path} | load_ratio={load_ratio:.2f}% '
        f'missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}'
    )
    wrapper.eval()
    inner = wrapper.net if hasattr(wrapper, 'net') else wrapper
    return wrapper, inner, inner.gcn


@torch.no_grad()
def _manual_context(inner, batch, max_contours: int):
    was_training = inner.training
    inner.eval()
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
    return {
        'cnn_feature': cnn_feature.detach(),
        'i_it_py': i_init.detach(),
        'c_it_py': c_init.detach(),
        'i_gt_py': _align_gt(i_init, i_gt).detach(),
        'py_ind': py_ind.detach(),
        'image_hw': (int(batch['inp'].shape[-2]), int(batch['inp'].shape[-1])),
    }


@torch.no_grad()
def _deterministic_three_step(output, flow, fractions, ode_steps: int):
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


def _set_seed(seed: int = 20260611):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def main():
    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True
    cfg.use_flow_matching = True
    cfg.use_grpo = True

    if eval_gpu:
        cfg.gpus = [int(eval_gpu)]
        print(f'[*] Override evaluation GPU -> {eval_gpu}')

    ckpt_env = os.environ.get('CKPT', '').strip()
    if not ckpt_env:
        raise FileNotFoundError('Set CKPT=/path/to/checkpoint.pt')
    ckpt_path = _project_path(ckpt_env)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    out_path = _project_path(os.environ.get('OUT_PATH', 'data/stats/rl_difficulty_index.json'))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    outer_steps = int(_cfg_value('outer_steps', 3))
    fractions_cfg = _cfg_value('fractions', [0.3333, 0.5, 1.0])
    fractions = [float(x) for x in list(fractions_cfg)]
    if len(fractions) < outer_steps:
        fractions = fractions + [1.0] * (outer_steps - len(fractions))
    fractions = fractions[:outer_steps]
    ode_steps = int(_cfg_value('ode_steps', getattr(cfg, 'iterative_ode_steps', getattr(cfg, 'flow_ode_steps', 10))))
    if ode_steps <= 0:
        ode_steps = int(getattr(cfg, 'flow_ode_steps', 10))
    max_contours = int(_cfg_value('max_contours', 0))

    _set_seed()
    _, inner, gcn = _load_model(ckpt_path)
    device = next(inner.parameters()).device

    dataset = make_dataset(cfg, cfg.train.dataset, make_transforms(cfg, True), True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=make_collator(cfg),
        pin_memory=True,
    )
    print(f'[*] Building difficulty index for {len(dataset)} samples from {cfg.train.dataset}')
    print(f'[*] CFG_FILE={os.environ.get("CFG_FILE")} | ODE steps={ode_steps} | fractions={fractions}')

    samples = {}
    mean_ious = []
    hard_thresh = 0.8
    for index, batch in enumerate(loader):
        _move_batch(batch, device=device)
        output = _manual_context(inner, batch, max_contours=max_contours)
        n_contours = int(output['i_it_py'].shape[0])
        if n_contours == 0:
            rec = {
                'per_contour_iou': [],
                'mean_iou': 1.0,
                'min_iou': 1.0,
                'num_contours': 0,
            }
        else:
            det = _deterministic_three_step(output, gcn, fractions=fractions, ode_steps=ode_steps)
            ious = compute_region_score(
                det['py'],
                output['i_gt_py'],
                H=int(output['image_hw'][0]),
                W=int(output['image_hw'][1]),
                w_boundary=0,
                w_dice=0,
                w_iou=1,
                w_dist=0,
                coord_scale=float(snake_config.down_ratio),
            ).detach().float().cpu().numpy()
            per_iou = [float(x) for x in ious.tolist()]
            rec = {
                'per_contour_iou': per_iou,
                'mean_iou': float(np.mean(ious)) if ious.size else 1.0,
                'min_iou': float(np.min(ious)) if ious.size else 1.0,
                'num_contours': n_contours,
            }
        samples[str(index)] = rec
        mean_ious.append(float(rec['mean_iou']))
        if (index + 1) % 50 == 0 or index + 1 == len(dataset):
            hard_ratio = float(np.mean(np.asarray(mean_ious) < hard_thresh)) if mean_ious else 0.0
            print(
                f'[{index + 1}/{len(dataset)}] mean_iou={np.mean(mean_ious):.4f} '
                f'hard<{hard_thresh:.1f}={100.0 * hard_ratio:.2f}%'
            )

    data = {
        'ckpt': str(ckpt_path),
        'cfg_file': os.environ.get('CFG_FILE', ''),
        'timestamp': datetime.datetime.now().isoformat(),
        'samples': samples,
    }
    tmp = out_path.with_suffix(out_path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(str(tmp), str(out_path))

    arr = np.asarray(mean_ious, dtype=np.float32)
    if arr.size:
        qs = [0, 10, 25, 50, 75, 90, 100]
        summary = {f'p{q}': float(np.percentile(arr, q)) for q in qs}
        hard_ratio = float(np.mean(arr < hard_thresh))
        print('[*] Difficulty summary:')
        print('    ' + ' '.join([f'{k}={v:.4f}' for k, v in summary.items()]))
        print(f'    mean={float(arr.mean()):.4f} hard_mean_iou<{hard_thresh:.1f}={100.0 * hard_ratio:.2f}%')
    print(f'[*] Wrote {out_path}')


if __name__ == '__main__':
    main()
