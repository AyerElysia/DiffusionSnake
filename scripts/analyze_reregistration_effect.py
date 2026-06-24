#!/usr/bin/env python3
"""Diagnose whether better GT re-registration reduces high-frequency residuals."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT_DIR))

_DEFAULT_CFG = _ROOT_DIR / 'configs' / '1232_final_v5_geom_action_5step_from3500_gpu2.yaml'
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--cfg_file', default=str(_DEFAULT_CFG), type=str)
_pre_args, _remaining_argv = _pre_parser.parse_known_args()
os.environ['CFG_FILE'] = str(_pre_args.cfg_file)
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/Ultralytics')
os.environ.setdefault('XDG_CONFIG_HOME', '/tmp')

_ORIGINAL_ARGV = list(sys.argv)
_IMPORT_ARGV = [sys.argv[0], '--cfg_file', str(_pre_args.cfg_file)]
sys.argv = _IMPORT_ARGV

from grpo_train_v5_geom_action import (
    _align_gt,
    _contour_normals,
    _flatten_valid_polys,
    _lowfreq_basis,
    _make_py_ind,
)

from scripts.analyze_v5_gt_residual_distribution import (  # noqa: E402
    deterministic_v5,
    fourier_energy_ratio,
    load_model,
    manual_context,
    move_batch,
    run_pca,
)
sys.argv = _ORIGINAL_ARGV

from lib.config import cfg  # noqa: E402
from lib.datasets import make_data_loader  # noqa: E402
from lib.utils.snake import snake_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze whether GT point re-registration reduces high-frequency normal residual energy.'
    )
    parser.add_argument('--cfg_file', default=str(_pre_args.cfg_file), type=str)
    parser.add_argument('--max_samples', default=0, type=int, help='Limit dataset samples; <=0 means all.')
    parser.add_argument('--out_dir', default=str(_ROOT_DIR / 'data' / 'analysis' / 'v5_reregistration'))
    parser.add_argument('--gpu', default='', type=str, help='Optional CUDA_VISIBLE_DEVICES override.')
    parser.add_argument('--seed', default=20260616, type=int)
    parser.add_argument('--pca_k', default=64, type=int)
    parser.add_argument('--fourier_modes', default=8, type=int)
    parser.add_argument('--ode_steps', default=0, type=int)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def signed_area(poly):
    x = poly[..., 0]
    y = poly[..., 1]
    x1 = torch.roll(x, shifts=-1, dims=1)
    y1 = torch.roll(y, shifts=-1, dims=1)
    return 0.5 * torch.sum(x * y1 - x1 * y, dim=1)


def optimal_roll_align(pred, gt):
    if pred.size(0) == 0:
        return gt
    gt = gt.clone()
    mis = ((signed_area(pred) >= 0) ^ (signed_area(gt) >= 0))
    if mis.any():
        gt[mis] = torch.flip(gt[mis], dims=[1])

    p = int(gt.size(1))
    rolled = torch.stack([torch.roll(gt, shifts=-shift, dims=1) for shift in range(p)], dim=1)
    d2 = (rolled - pred[:, None]).pow(2).sum(dim=(-1, -2))
    best = torch.argmin(d2, dim=1)
    out = [rolled[i, int(best[i].item())] for i in range(gt.size(0))]
    return torch.stack(out, dim=0)


def _closed_arclen_fractions(poly):
    nxt = np.roll(poly, shift=-1, axis=0)
    edge = np.linalg.norm(nxt - poly, axis=1).astype(np.float64)
    total = float(edge.sum())
    if total < 1e-8:
        return None, total
    start = np.concatenate([[0.0], np.cumsum(edge[:-1])]) / total
    return start.astype(np.float64), total


def _interp_closed_by_fraction(poly, target_frac):
    p = poly.shape[0]
    edge = np.linalg.norm(np.roll(poly, shift=-1, axis=0) - poly, axis=1).astype(np.float64)
    total = float(edge.sum())
    if total < 1e-8:
        return poly.copy()

    gt_s = np.concatenate([[0.0], np.cumsum(edge) / total])
    gt_ext = np.concatenate([poly, poly[:1]], axis=0).astype(np.float64)
    target = np.mod(target_frac, 1.0)
    idx = np.searchsorted(gt_s, target, side='right') - 1
    idx = np.clip(idx, 0, p - 1)
    s0 = gt_s[idx]
    s1 = gt_s[idx + 1]
    alpha = (target - s0) / np.maximum(s1 - s0, 1e-12)
    out = gt_ext[idx] * (1.0 - alpha[:, None]) + gt_ext[idx + 1] * alpha[:, None]
    return out.astype(np.float32)


def pointwise_arclen_align(pred, gt_optimal_roll):
    pred_np = pred.detach().cpu().numpy().astype(np.float32)
    gt_np = gt_optimal_roll.detach().cpu().numpy().astype(np.float32)
    aligned = []
    for pred_i, gt_i in zip(pred_np, gt_np):
        pred_frac, pred_total = _closed_arclen_fractions(pred_i)
        if pred_frac is None or pred_total < 1e-8:
            aligned.append(gt_i)
            continue
        aligned.append(_interp_closed_by_fraction(gt_i, pred_frac))
    out = np.stack(aligned, axis=0)
    return torch.from_numpy(out).to(device=pred.device, dtype=pred.dtype)


def normal_residual(pred, gt_aligned):
    normals = _contour_normals(pred)
    return ((gt_aligned - pred) * normals).sum(dim=-1)


def summarize_method(name, r_n, down_ratio, pca_k, fourier_modes):
    pca = run_pca(r_n, pca_k)
    fourier_per, fourier_mean = fourier_energy_ratio(r_n, fourier_modes)
    abs_px = np.abs(r_n * float(down_ratio)).reshape(-1)
    cum = pca['cum_explained_variance_ratio']
    return {
        'name': name,
        'num_contours': int(r_n.shape[0]),
        'P': int(r_n.shape[1]),
        'fourier8_energy_mean': float(fourier_mean),
        'fourier8_energy_median': float(np.median(fourier_per)) if fourier_per.size else 0.0,
        'highfreq_energy_mean': float(1.0 - fourier_mean),
        'pca_n90': int(pca['n90']),
        'pca_cum6': float(cum[min(5, len(cum) - 1)]) if len(cum) else 0.0,
        'pca_cum12': float(cum[min(11, len(cum) - 1)]) if len(cum) else 0.0,
        'abs_rn_px_median': float(np.percentile(abs_px, 50)),
        'abs_rn_px_p95': float(np.percentile(abs_px, 95)),
    }


def print_table(rows):
    print('\n=== Re-registration residual high-frequency comparison ===')
    print(
        f'{"method":<18} {"Fourier8 cover":>15} {"highfreq ratio":>15} '
        f'{"PCA n90":>8} {"|r_n| med px":>13}'
    )
    for row in rows:
        print(
            f'{row["name"]:<18} '
            f'{row["fourier8_energy_mean"]:>15.4f} '
            f'{row["highfreq_energy_mean"]:>15.4f} '
            f'{row["pca_n90"]:>8d} '
            f'{row["abs_rn_px_median"]:>13.4f}'
        )


def main():
    args = parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        cfg.gpus = [int(args.gpu)]
        print(f'[*] Override GPU -> {args.gpu}')

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fractions = [0.2, 0.25, 0.3333, 0.5, 1.0]
    ode_steps = int(args.ode_steps or getattr(cfg, 'iterative_ode_steps', getattr(cfg, 'flow_ode_steps', 10)))
    down_ratio = float(snake_config.down_ratio)

    inner, flow, ckpt_path = load_model(device)
    loader = make_data_loader(cfg, is_train=True, is_distributed=False)
    print(f'[*] Dataset: {cfg.train.dataset} | max_samples={args.max_samples or "all"} | device={device}')

    residuals = {
        'baseline_current': [],
        'rereg_optimal_roll': [],
        'rereg_pointwise': [],
    }
    seen_samples = 0
    total_contours = 0

    for batch_idx, batch in enumerate(loader):
        if args.max_samples > 0 and seen_samples >= args.max_samples:
            break
        batch = move_batch(batch, device)
        output = manual_context(inner, batch)
        n_contours = int(output['i_it_py'].size(0))
        if n_contours == 0:
            seen_samples += int(batch['inp'].shape[0])
            continue

        pred = deterministic_v5(output, flow, fractions, ode_steps)
        gt_raw = output['i_gt_py']

        gt_current = _align_gt(pred, gt_raw)
        gt_opt = optimal_roll_align(pred, gt_raw)
        gt_point = pointwise_arclen_align(pred, gt_opt)

        residuals['baseline_current'].append(
            normal_residual(pred, gt_current).detach().cpu().numpy().astype(np.float32)
        )
        residuals['rereg_optimal_roll'].append(
            normal_residual(pred, gt_opt).detach().cpu().numpy().astype(np.float32)
        )
        residuals['rereg_pointwise'].append(
            normal_residual(pred, gt_point).detach().cpu().numpy().astype(np.float32)
        )

        seen_samples += int(batch['inp'].shape[0])
        total_contours += n_contours
        if batch_idx % 10 == 0:
            print(f'[*] processed samples={seen_samples} contours={total_contours}')

    if not residuals['baseline_current']:
        raise RuntimeError('No valid contours collected.')

    arrays = {key: np.concatenate(value, axis=0) for key, value in residuals.items()}
    rows = [
        summarize_method('current', arrays['baseline_current'], down_ratio, args.pca_k, args.fourier_modes),
        summarize_method('optimal_roll', arrays['rereg_optimal_roll'], down_ratio, args.pca_k, args.fourier_modes),
        summarize_method('pointwise', arrays['rereg_pointwise'], down_ratio, args.pca_k, args.fourier_modes),
    ]

    current_hf = rows[0]['highfreq_energy_mean']
    pointwise_hf = rows[2]['highfreq_energy_mean']
    rel_drop = (current_hf - pointwise_hf) / max(current_hf, 1e-12)
    if rel_drop > 0.20:
        conclusion = (
            'RE-REGISTRATION HELPS: high-freq energy is substantially registration noise (B-route GO)'
        )
    else:
        conclusion = (
            'RE-REGISTRATION MARGINAL: high-freq is mostly real signal, not registration noise (B-route NO-GO)'
        )

    summary = {
        'cfg_file': str(args.cfg_file),
        'checkpoint': str(ckpt_path),
        'samples_seen': int(seen_samples),
        'num_contours': int(arrays['baseline_current'].shape[0]),
        'P': int(arrays['baseline_current'].shape[1]),
        'down_ratio': down_ratio,
        'fourier_modes': int(args.fourier_modes),
        'pca_k': int(args.pca_k),
        'methods': rows,
        'pointwise_highfreq_relative_drop_vs_current': float(rel_drop),
        'conclusion': conclusion,
    }

    np.savez_compressed(
        out_dir / 'r_n_reregistration.npz',
        r_n_current=arrays['baseline_current'],
        r_n_optimal_roll=arrays['rereg_optimal_roll'],
        r_n_pointwise=arrays['rereg_pointwise'],
        down_ratio=np.asarray(down_ratio, dtype=np.float32),
        checkpoint=str(ckpt_path),
    )
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print_table(rows)
    print(f'\npointwise highfreq relative drop vs current: {rel_drop:.2%}')
    print(conclusion)
    print(f'outputs: {out_dir}')


if __name__ == '__main__':
    main()
