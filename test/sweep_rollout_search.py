#!/usr/bin/env python3
"""Sweep rollout-search settings against the current deterministic policy.

The goal is to find rollout settings whose best-of-k result reliably beats the
current deployed deterministic path on a fixed subset of samples. This is the
precondition for useful det-gated RL distillation.

Usage example:
  conda run -n snake1 --no-capture-output bash -lc '
    export CFG_FILE=configs/btcv_diffusion_dit_v3_4_fm_full_noleak.yaml
    export CUDA_VISIBLE_DEVICES=4
    export CANDIDATES="k8_s0_n1_o10,8,0,1.0,10;k8_s0_n0.5_o10,8,0,0.5,10"
    export MAX_SAMPLES=8
    python test/sweep_rollout_search.py
  '
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List


_ROOT = Path(__file__).resolve().parents[1]
_EARLY = argparse.ArgumentParser(add_help=False)
_EARLY.add_argument('--cfg', dest='cfg_file', default=None)
_EARLY_ARGS, _ = _EARLY.parse_known_args()
if _EARLY_ARGS.cfg_file:
    os.environ['CFG_FILE'] = _EARLY_ARGS.cfg_file
elif not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = str(_ROOT / 'configs' / 'btcv_diffusion_dit_v3_4_fm_full_noleak_yolom_gpu35.yaml')

sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from lib.config import cfg
from lib.datasets.collate_batch import make_collator
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.networks import make_network
from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
from lib.train.rewards.region_reward import compute_region_score
from lib.train.trainers import make_trainer
from lib.utils.snake import snake_config, snake_gcn_utils


DEFAULT_CANDIDATES = (
    'k8_s0_n1_o10,8,0.0,1.0,10;'
    'k8_s0_n05_o10,8,0.0,0.5,10;'
    'k8_s0_n05_o20,8,0.0,0.5,20;'
    'k8_s002_n05_o10,8,0.02,0.5,10;'
    'k8_s002_n05_o20,8,0.02,0.5,20;'
    'k16_s0_n05_o20,16,0.0,0.5,20'
)


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
        inds = []
        for bi in range(valid.size(0)):
            count = int(valid[bi].sum().item())
            inds.append(torch.full((count,), bi, dtype=torch.long, device=device))
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


def _move_batch(batch: Dict, device) -> Dict:
    for key, val in list(batch.items()):
        if isinstance(val, torch.Tensor):
            batch[key] = val.to(device, non_blocking=True)
    return batch


def _extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ('state_dict', 'model', 'net', 'network'):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


def _adapt_state_dict(model, sd):
    sd = remap_legacy_state_dict(sd)
    if all(k.startswith('module.') for k in sd):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    needs = any(k.startswith('net.') for k in model.state_dict())
    has = any(k.startswith('net.') for k in sd)
    if needs and not has:
        sd = {f'net.{k}': v for k, v in sd.items()}
    return sd


def load_model(ckpt_path: Path):
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    wrapper = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    ckpt_obj = torch.load(str(ckpt_path), map_location='cpu')
    sd = _extract_state_dict(ckpt_obj)
    sd = _adapt_state_dict(wrapper, sd)
    info = wrapper.load_state_dict(sd, strict=False)
    print(
        f'[sweep] ckpt={ckpt_path} load_ratio='
        f'{100.0 * (len(sd) - len(info.missing_keys)) / max(len(sd), 1):.2f}% '
        f'missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}'
    )
    return trainer.network.cuda().eval()


def parse_candidates(spec: str) -> List[Dict]:
    items = []
    for raw in spec.split(';'):
        raw = raw.strip()
        if not raw:
            continue
        name, k, std, noise, ode = [x.strip() for x in raw.split(',')]
        items.append({
            'name': name,
            'k': int(k),
            'action_std': float(std),
            'noise_scale': float(noise),
            'ode_steps': int(ode),
        })
    if not items:
        raise ValueError('No valid candidates parsed from CANDIDATES')
    return items


@torch.no_grad()
def build_manual_context(core, batch: Dict) -> Dict:
    yolo_out = core.yolo(batch['inp'])
    feat_list = yolo_out[1] if isinstance(yolo_out, (list, tuple)) and len(yolo_out) > 1 else None
    feat_p2 = feat_list[0] if isinstance(feat_list, (list, tuple)) else yolo_out
    cnn_feature = core.cnn_proj(feat_p2)
    if getattr(core, 'use_p3_features', False) and hasattr(core, 'cnn_proj_p3'):
        if isinstance(feat_list, (list, tuple)) and len(feat_list) > 1:
            feat_p3 = feat_list[1]
            feat_p3_up = F.interpolate(feat_p3, size=feat_p2.shape[-2:], mode='bilinear', align_corners=False)
            cnn_feature = cnn_feature + core.cnn_proj_p3(feat_p3_up)

    device = cnn_feature.device
    i_init = _flatten_valid_polys(batch, 'i_it_py', device=device)
    i_gt = _flatten_valid_polys(batch, 'i_gt_py', device=device)
    c_init = _flatten_valid_polys(batch, 'c_it_py', device=device)
    py_ind = _make_py_ind(batch, i_init.size(0), device=device)
    if c_init.size(0) != i_init.size(0):
        c_init = snake_gcn_utils.img_poly_to_can_poly(i_init)
    if i_gt.size(0) != i_init.size(0):
        n = min(i_gt.size(0), i_init.size(0))
        i_init, c_init, i_gt, py_ind = i_init[:n], c_init[:n], i_gt[:n], py_ind[:n]
    i_gt = _align_gt(i_init, i_gt)
    return {
        'cnn_feature': cnn_feature,
        'i_init': i_init,
        'c_init': c_init,
        'i_gt': i_gt,
        'py_ind': py_ind,
    }


def score_iou(pred_poly: torch.Tensor, gt_poly: torch.Tensor, batch: Dict) -> float:
    h_img = int(batch['inp'].shape[-2])
    w_img = int(batch['inp'].shape[-1])
    score = compute_region_score(
        pred_poly,
        gt_poly,
        H=h_img,
        W=w_img,
        w_boundary=0.0,
        w_dice=0.0,
        w_iou=1.0,
        coord_scale=float(snake_config.down_ratio),
    )
    return float(score.mean().item())


@torch.no_grad()
def run_candidate_rollout(gcn, ctx: Dict, action_std: float, noise_scale: float, ode_steps: int) -> torch.Tensor:
    iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
    fractions = list(getattr(cfg, 'iterative_fractions', []))
    if not fractions:
        fractions = [1.0 / (iter_steps - i) for i in range(iter_steps)]
    current = ctx['i_init'].detach()
    total_disp = torch.zeros_like(current)
    for frac in fractions[:iter_steps]:
        c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
        ret = gcn.sample_with_logprob(
            ctx['cnn_feature'],
            current,
            c_cur,
            ctx['py_ind'],
            steps=ode_steps,
            action_std=action_std,
            noise_scale=noise_scale,
        )
        applied = ret['disp'] * float(frac)
        current = (current + applied).detach()
        total_disp = total_disp + applied
    return total_disp


def main():
    parser = argparse.ArgumentParser(description='Sweep rollout generator candidates.')
    parser.add_argument('--out', default=os.environ.get('OUT', str(_ROOT / 'test' / 'rollout_sweep_results.json')))
    args = parser.parse_args()

    candidates = parse_candidates(os.environ.get('CANDIDATES', DEFAULT_CANDIDATES))
    max_samples = int(os.environ.get('MAX_SAMPLES', '8'))
    split = os.environ.get('SPLIT', 'train').strip().lower()
    seed = int(os.environ.get('SEED', '20260519'))
    ckpt_rel = os.environ.get(
        'CKPT',
        'data/outputs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolom_gpu35_reusemax/checkpoints/latest.pt',
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ckpt_path = Path(ckpt_rel)
    if not ckpt_path.is_absolute():
        ckpt_path = _ROOT / ckpt_path

    model = load_model(ckpt_path)
    core = model.net if hasattr(model, 'net') else model
    gcn = core.gcn

    if split == 'test':
        dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    else:
        dataset = make_dataset(cfg, cfg.train.dataset, make_transforms(cfg, True), True)
    collator = make_collator(cfg)

    limit = min(len(dataset), max_samples) if max_samples > 0 else len(dataset)
    print(f'[sweep] split={split} samples={limit} candidates={len(candidates)}')

    det_scores = []
    sample_rows = []
    summary = {c['name']: {'gains': [], 'best_scores': [], 'wins': 0} for c in candidates}

    for sample_idx in range(limit):
        batch = collator([dataset[sample_idx]])
        _move_batch(batch, device='cuda')
        ctx = build_manual_context(core, batch)
        if ctx['i_init'].numel() == 0:
            del batch, ctx
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        det_disp = gcn.sample_disp_iterative(
            ctx['cnn_feature'],
            ctx['i_init'],
            ctx['c_init'],
            ctx['py_ind'],
            num_iter_steps=int(getattr(cfg, 'iterative_num_steps', 3)),
            fractions=list(getattr(cfg, 'iterative_fractions', [])) or None,
            ode_steps=int(getattr(cfg, 'iterative_ode_steps', getattr(cfg, 'flow_ode_steps', 10))),
        )
        det_score = score_iou(ctx['i_init'] + det_disp, ctx['i_gt'], batch)
        det_scores.append(det_score)

        row = {'idx': sample_idx, 'det_iou': det_score, 'candidates': {}}
        print(f'[sample {sample_idx:03d}] det={det_score:.4f}')
        for cand in candidates:
            best_score = -1.0
            best_gain = -999.0
            for _ in range(cand['k']):
                disp = run_candidate_rollout(
                    gcn,
                    ctx,
                    action_std=cand['action_std'],
                    noise_scale=cand['noise_scale'],
                    ode_steps=cand['ode_steps'],
                )
                score = score_iou(ctx['i_init'] + disp, ctx['i_gt'], batch)
                if score > best_score:
                    best_score = score
                    best_gain = score - det_score
            row['candidates'][cand['name']] = {
                'best_iou': best_score,
                'gain': best_gain,
            }
            summary[cand['name']]['gains'].append(best_gain)
            summary[cand['name']]['best_scores'].append(best_score)
            if best_gain > 0:
                summary[cand['name']]['wins'] += 1
            print(
                f"  - {cand['name']}: best={best_score:.4f} gain={best_gain:+.4f} "
                f"(k={cand['k']} std={cand['action_std']} noise={cand['noise_scale']} ode={cand['ode_steps']})"
            )
        sample_rows.append(row)
        del batch, ctx, det_disp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ranked = []
    for cand in candidates:
        gains = np.array(summary[cand['name']]['gains'], dtype=np.float32)
        bests = np.array(summary[cand['name']]['best_scores'], dtype=np.float32)
        item = {
            **cand,
            'mean_gain': float(gains.mean()) if gains.size else 0.0,
            'median_gain': float(np.median(gains)) if gains.size else 0.0,
            'max_gain': float(gains.max()) if gains.size else 0.0,
            'positive_rate': float(summary[cand['name']]['wins']) / max(len(gains), 1),
            'mean_best_iou': float(bests.mean()) if bests.size else 0.0,
            'mean_det_iou': float(np.mean(det_scores)) if det_scores else 0.0,
            'n_samples': int(len(gains)),
        }
        ranked.append(item)
    ranked.sort(key=lambda x: (x['mean_gain'], x['positive_rate'], x['max_gain']), reverse=True)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'created_at': datetime.datetime.now().isoformat(),
        'cfg_file': os.environ.get('CFG_FILE', ''),
        'ckpt': str(ckpt_path),
        'split': split,
        'max_samples': limit,
        'candidates': candidates,
        'ranked': ranked,
        'samples': sample_rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print('\n[sweep] ranked candidates:')
    for item in ranked:
        print(
            f"  * {item['name']}: mean_gain={item['mean_gain']:+.4f} "
            f"median_gain={item['median_gain']:+.4f} pos_rate={item['positive_rate']:.2f} "
            f"max_gain={item['max_gain']:+.4f}"
        )
    print(f'[sweep] wrote {out_path}')


if __name__ == '__main__':
    main()
