#!/usr/bin/env python3
"""Quantitative RL signal diagnostics for V3.4-FM.

This script answers three concrete questions:

1. Which rollout source actually creates reward gains over the deployed
   deterministic policy?
2. Is that source covered by PPO log-probabilities?
3. Does the train/val signal look similar enough to justify RL training?

It intentionally does not update weights.
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
    os.environ['CFG_FILE'] = str(_ROOT / 'configs' / 'btcv_v3_4_fm_rl_v8g_ppo_gain_gpu4.yaml')

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'test'))

import numpy as np
import torch

from lib.config import cfg
from lib.datasets.collate_batch import make_collator
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.utils.snake import snake_config, snake_gcn_utils

from sweep_rollout_search import (  # noqa: E402
    build_manual_context,
    load_model,
    score_iou,
    _move_batch,
)


DEFAULT_MODES = (
    # name,k,action_std,noise_scale,ode_steps
    'x0_search_k16_n05_o20,16,0.0,0.5,20;'
    'x0_default_k8_n1_o10,8,0.0,1.0,10;'
    'step_noise_k8_s002_n05_o20,8,0.02,0.5,20;'
    'step_noise_k8_s005_n05_o20,8,0.05,0.5,20'
)


def parse_modes(spec: str) -> List[Dict]:
    modes = []
    for raw in spec.split(';'):
        raw = raw.strip()
        if not raw:
            continue
        name, k, std, noise, ode = [x.strip() for x in raw.split(',')]
        modes.append({
            'name': name,
            'k': int(k),
            'action_std': float(std),
            'noise_scale': float(noise),
            'ode_steps': int(ode),
        })
    if not modes:
        raise ValueError('No modes parsed')
    return modes


@torch.no_grad()
def sample_iterative_with_logs(gcn, ctx: Dict, action_std: float, noise_scale: float, ode_steps: int) -> Dict:
    iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
    fractions = list(getattr(cfg, 'iterative_fractions', []))
    if not fractions:
        fractions = [1.0 / (iter_steps - i) for i in range(iter_steps)]

    current = ctx['i_init'].detach()
    total_disp = torch.zeros_like(current)
    all_log_probs = []
    initial_latent_count = 0
    policy_step_count = 0

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
        initial_latent_count += 1
        if ret.get('log_probs'):
            policy_step_count += len(ret['log_probs'])
            all_log_probs.extend([x.detach().float().cpu() for x in ret['log_probs']])
        applied = ret['disp'] * float(frac)
        current = (current + applied).detach()
        total_disp = total_disp + applied

    if all_log_probs:
        lp = torch.stack(all_log_probs, dim=0)  # (T, contours)
        lp_abs_mean = float(lp.abs().mean().item())
        lp_std = float(lp.std(unbiased=False).item())
        lp_min = float(lp.min().item())
        lp_max = float(lp.max().item())
    else:
        lp_abs_mean = lp_std = lp_min = lp_max = 0.0

    return {
        'disp': total_disp,
        'initial_latent_count': initial_latent_count,
        'policy_step_count': policy_step_count,
        'logprob_abs_mean': lp_abs_mean,
        'logprob_std': lp_std,
        'logprob_min': lp_min,
        'logprob_max': lp_max,
    }


@torch.no_grad()
def deterministic_score(gcn, ctx: Dict, batch: Dict, det_ode_steps: int) -> float:
    iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
    fractions = list(getattr(cfg, 'iterative_fractions', []))
    if not fractions:
        fractions = [1.0 / (iter_steps - i) for i in range(iter_steps)]
    disp = gcn.sample_disp_iterative(
        ctx['cnn_feature'],
        ctx['i_init'],
        ctx['c_init'],
        ctx['py_ind'],
        num_iter_steps=iter_steps,
        fractions=fractions,
        ode_steps=det_ode_steps,
    )
    return score_iou(ctx['i_init'] + disp, ctx['i_gt'], batch)


def summarize(values: List[float]) -> Dict:
    arr = np.array(values, dtype=np.float32)
    if arr.size == 0:
        return {'mean': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0}
    return {
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'min': float(arr.min()),
        'max': float(arr.max()),
    }


def run_split(core, gcn, split: str, modes: List[Dict], max_samples: int, det_ode_steps: int) -> Dict:
    if split == 'val':
        dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    else:
        dataset = make_dataset(cfg, cfg.train.dataset, make_transforms(cfg, True), True)
    collator = make_collator(cfg)
    limit = min(len(dataset), max_samples)

    rows = []
    mode_acc = {
        m['name']: {
            'best_gain': [],
            'mean_gain': [],
            'positive_best': 0,
            'positive_rollout': 0,
            'n_rollouts': 0,
            'logprob_abs_mean': [],
            'logprob_std': [],
            'policy_step_count': [],
            'initial_latent_count': [],
        }
        for m in modes
    }
    det_scores = []

    print(f'[diagnose] split={split} samples={limit}')
    for idx in range(limit):
        batch = collator([dataset[idx]])
        _move_batch(batch, device='cuda')
        ctx = build_manual_context(core, batch)
        det = deterministic_score(gcn, ctx, batch, det_ode_steps)
        det_scores.append(det)
        row = {'idx': idx, 'det_iou': det, 'modes': {}}
        print(f'[{split} {idx:03d}] det={det:.4f}')

        for mode in modes:
            scores = []
            lp_abs = []
            lp_std = []
            policy_counts = []
            latent_counts = []
            for _ in range(mode['k']):
                ret = sample_iterative_with_logs(
                    gcn,
                    ctx,
                    action_std=mode['action_std'],
                    noise_scale=mode['noise_scale'],
                    ode_steps=mode['ode_steps'],
                )
                score = score_iou(ctx['i_init'] + ret['disp'], ctx['i_gt'], batch)
                scores.append(score)
                lp_abs.append(ret['logprob_abs_mean'])
                lp_std.append(ret['logprob_std'])
                policy_counts.append(ret['policy_step_count'])
                latent_counts.append(ret['initial_latent_count'])

            gains = [s - det for s in scores]
            best = max(scores)
            best_gain = best - det
            mean_gain = float(np.mean(gains))
            positive_rollouts = sum(g > 0 for g in gains)
            acc = mode_acc[mode['name']]
            acc['best_gain'].append(best_gain)
            acc['mean_gain'].append(mean_gain)
            acc['positive_best'] += int(best_gain > 0)
            acc['positive_rollout'] += int(positive_rollouts)
            acc['n_rollouts'] += len(gains)
            acc['logprob_abs_mean'].append(float(np.mean(lp_abs)))
            acc['logprob_std'].append(float(np.mean(lp_std)))
            acc['policy_step_count'].append(float(np.mean(policy_counts)))
            acc['initial_latent_count'].append(float(np.mean(latent_counts)))

            row['modes'][mode['name']] = {
                **mode,
                'best_iou': best,
                'mean_iou': float(np.mean(scores)),
                'best_gain': best_gain,
                'mean_gain': mean_gain,
                'positive_rollouts': int(positive_rollouts),
                'logprob_abs_mean': float(np.mean(lp_abs)),
                'logprob_std': float(np.mean(lp_std)),
                'policy_step_count': float(np.mean(policy_counts)),
                'initial_latent_count': float(np.mean(latent_counts)),
                'ppo_can_update': bool(mode['action_std'] > 0.0),
                'initial_latent_logprob_tracked': False,
            }
            print(
                f"  {mode['name']}: best_gain={best_gain:+.4f} "
                f"mean_gain={mean_gain:+.4f} pos={positive_rollouts}/{len(gains)} "
                f"lp_abs={float(np.mean(lp_abs)):.3f} ppo={mode['action_std'] > 0.0}"
            )

        rows.append(row)
        del batch, ctx
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        'det_iou': summarize(det_scores),
        'modes': {},
    }
    for mode in modes:
        acc = mode_acc[mode['name']]
        n_samples = max(len(acc['best_gain']), 1)
        summary['modes'][mode['name']] = {
            'config': mode,
            'best_gain': summarize(acc['best_gain']),
            'mean_gain': summarize(acc['mean_gain']),
            'sample_positive_rate': float(acc['positive_best']) / n_samples,
            'rollout_positive_sample_rate': float(acc['positive_rollout']) / n_samples,
            'logprob_abs_mean': summarize(acc['logprob_abs_mean']),
            'logprob_std': summarize(acc['logprob_std']),
            'policy_step_count': summarize(acc['policy_step_count']),
            'initial_latent_count': summarize(acc['initial_latent_count']),
            'ppo_can_update': bool(mode['action_std'] > 0.0),
            'initial_latent_logprob_tracked': False,
        }

    return {
        'split': split,
        'n_samples': len(rows),
        'summary': summary,
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser(description='Diagnose RL reward/logprob coverage.')
    parser.add_argument('--out', default=os.environ.get('OUT', str(_ROOT / 'test' / 'rl_signal_diagnostics.json')))
    args = parser.parse_args()

    max_train = int(os.environ.get('MAX_TRAIN', '4'))
    max_val = int(os.environ.get('MAX_VAL', '4'))
    det_ode_steps = int(os.environ.get('DET_ODE_STEPS', '10'))
    seed = int(os.environ.get('SEED', '20260519'))
    modes = parse_modes(os.environ.get('MODES', DEFAULT_MODES))
    ckpt_rel = os.environ.get(
        'CKPT',
        'data/outputs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolom_gpu35_reusemax/checkpoints/latest.pt',
    )
    ckpt = Path(ckpt_rel)
    if not ckpt.is_absolute():
        ckpt = _ROOT / ckpt

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = load_model(ckpt)
    core = model.net if hasattr(model, 'net') else model
    gcn = core.gcn

    payload = {
        'created_at': datetime.datetime.now().isoformat(),
        'cfg_file': os.environ.get('CFG_FILE', ''),
        'ckpt': str(ckpt),
        'det_ode_steps': det_ode_steps,
        'modes': modes,
        'train': run_split(core, gcn, 'train', modes, max_train, det_ode_steps),
        'val': run_split(core, gcn, 'val', modes, max_val, det_ode_steps),
        'interpretation': {
            'ppo_gap_definition': 'PPO can only update modes with action_std > 0 because grpo_train_v2 skips PPO when action_std=0.',
            'latent_gap_definition': 'sample_with_logprob samples initial x = randn_like(...) but does not record a logprob term for that initial latent.',
        },
    }

    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print('\n[diagnose] summary')
    for split in ('train', 'val'):
        print(f'== {split} ==')
        for name, stats in payload[split]['summary']['modes'].items():
            print(
                f"{name}: best_gain_mean={stats['best_gain']['mean']:+.4f} "
                f"best_pos={stats['sample_positive_rate']:.2f} "
                f"mean_gain_mean={stats['mean_gain']['mean']:+.4f} "
                f"ppo={stats['ppo_can_update']} "
                f"latent_lp={stats['initial_latent_logprob_tracked']}"
            )
    print(f'[diagnose] wrote {out}')


if __name__ == '__main__':
    main()
