#!/usr/bin/env python3
"""Offline GT normal-residual spectrum analysis for RL V5 geom-action."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT_DIR))

_DEFAULT_CFG = _ROOT_DIR / 'configs' / '1232_final_v5_geom_action_5step_from3500_gpu2.yaml'
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--cfg_file', default=str(_DEFAULT_CFG), type=str)
_pre_args, _remaining_argv = _pre_parser.parse_known_args()
os.environ['CFG_FILE'] = str(_pre_args.cfg_file)
sys.argv = [sys.argv[0], '--cfg_file', str(_pre_args.cfg_file)]

from lib.config import cfg
from lib.datasets import make_data_loader
from lib.networks import make_network
from lib.train.trainers.make_trainer import _wrapper_factory
from lib.utils.snake import snake_config, snake_gcn_utils
from grpo_train_v5_geom_action import (
    _adapt_state_dict,
    _align_gt,
    _contour_normals,
    _extract_state_dict,
    _flatten_valid_polys,
    _lowfreq_basis,
    _make_py_ind,
    _outer_action_mean,
    _project_geom_z,
    _resolve_checkpoint_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze train-set GT residual normal spectrum for V5 geom-action.'
    )
    parser.add_argument('--cfg_file', default=str(_pre_args.cfg_file), type=str)
    parser.add_argument('--max_samples', default=0, type=int, help='Limit dataset samples; <=0 means all.')
    parser.add_argument('--out_dir', default=str(_ROOT_DIR / 'data' / 'analysis' / 'v5_residual_stats'))
    parser.add_argument('--gpu', default='', type=str, help='Optional CUDA_VISIBLE_DEVICES override.')
    parser.add_argument('--seed', default=20260616, type=int)
    parser.add_argument('--pca_k', default=32, type=int)
    parser.add_argument('--fourier_modes', default=8, type=int)
    parser.add_argument('--ode_steps', default=0, type=int)
    parser.add_argument('--skip_iou', action='store_true')
    return parser.parse_args([sys.argv[1], sys.argv[2], *_remaining_argv])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def move_batch(batch, device):
    for key in list(batch.keys()):
        if key in ('meta', 'orig_img', 'img_path'):
            continue
        value = batch[key]
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def load_model(device):
    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True
    cfg.use_flow_matching = True
    cfg.use_grpo = True

    network = make_network(cfg)
    net_for_load = _wrapper_factory(cfg, network)
    ckpt_path = _resolve_checkpoint_path()
    if ckpt_path is None:
        raise FileNotFoundError('No checkpoint found. Set cfg.resume_path or CKPT_PATH.')

    raw_ckpt = torch.load(str(ckpt_path), map_location='cpu')
    sd = _adapt_state_dict(net_for_load, _extract_state_dict(raw_ckpt))
    info = net_for_load.load_state_dict(sd, strict=False)
    total = len(list(net_for_load.state_dict().keys()))
    load_ratio = 100.0 * (total - len(info.missing_keys)) / max(total, 1)
    print(
        f'[*] Loaded checkpoint: {ckpt_path} | load_ratio={load_ratio:.2f}% '
        f'missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}'
    )
    net_for_load.to(device).eval()
    inner = net_for_load.net if hasattr(net_for_load, 'net') else net_for_load
    return inner, inner.gcn, ckpt_path


@torch.no_grad()
def manual_context(inner, batch):
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
    if was_training:
        inner.train()

    return {
        'cnn_feature': cnn_feature.detach(),
        'i_it_py': i_init.detach(),
        'c_it_py': c_init.detach(),
        'i_gt_py': i_gt.detach(),
        'py_ind': py_ind.detach(),
        'image_hw': (int(batch['inp'].shape[-2]), int(batch['inp'].shape[-1])),
    }


@torch.no_grad()
def deterministic_v5(output, flow, fractions, ode_steps):
    current = output['i_it_py'].detach()
    total_disp = torch.zeros_like(current)
    for frac in fractions:
        c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
        action = _outer_action_mean(
            flow, output['cnn_feature'], current, c_cur, output['py_ind'], float(frac), int(ode_steps)
        )
        current = (current + action).detach()
        total_disp = total_disp + action.detach()
    return output['i_it_py'] + total_disp


def flatten_class_ids(batch, n_contours):
    if 'ct_cls' not in batch or not isinstance(batch['ct_cls'], torch.Tensor):
        return np.full((n_contours,), -1, dtype=np.int32)
    ct_cls = batch['ct_cls']
    if 'ct_01' in batch and isinstance(batch['ct_01'], torch.Tensor) and ct_cls.shape == batch['ct_01'].shape:
        cls = ct_cls[batch['ct_01'].bool()]
    else:
        cls = ct_cls.reshape(-1)
    cls = cls[:n_contours].detach().cpu().numpy().astype(np.int32)
    if cls.shape[0] < n_contours:
        cls = np.pad(cls, (0, n_contours - cls.shape[0]), constant_values=-1)
    return cls


def poly_to_mask(poly, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def contour_iou(init_feat, gt_feat, image_hw, down_ratio):
    h, w = image_hw
    init_px = init_feat * float(down_ratio)
    gt_px = gt_feat * float(down_ratio)
    out = []
    for a, b in zip(init_px, gt_px):
        ma = poly_to_mask(a, h, w)
        mb = poly_to_mask(b, h, w)
        inter = np.logical_and(ma, mb).sum()
        union = np.logical_or(ma, mb).sum()
        out.append(float(inter) / float(union) if union > 0 else 0.0)
    return np.asarray(out, dtype=np.float32)


def run_pca(x, k):
    x = np.asarray(x, dtype=np.float64)
    x_centered = x - x.mean(axis=1, keepdims=True)
    x_centered = x_centered - x_centered.mean(axis=0, keepdims=True)
    _, s, vh = np.linalg.svd(x_centered, full_matrices=False)
    eig = (s ** 2) / max(x_centered.shape[0] - 1, 1)
    ratio = eig / max(float(eig.sum()), 1e-12)
    k = min(int(k), vh.shape[0])
    return {
        'singular_values': s[:k].astype(np.float32),
        'explained_variance_ratio': ratio[:k].astype(np.float32),
        'cum_explained_variance_ratio': np.cumsum(ratio)[:k].astype(np.float32),
        'components': vh[:k].astype(np.float32),
        'n90': int(np.searchsorted(np.cumsum(ratio), 0.90) + 1) if ratio.size else 0,
    }


def fourier_energy_ratio(r_n, n_modes):
    x = torch.from_numpy(np.asarray(r_n, dtype=np.float32))
    basis = _lowfreq_basis(x.shape[1], int(n_modes), x.device, x.dtype)
    coeff = torch.matmul(x, basis)
    recon = torch.matmul(coeff, basis.t())
    num = recon.pow(2).sum(dim=1)
    den = x.pow(2).sum(dim=1).clamp_min(1e-12)
    per = (num / den).numpy()
    return per.astype(np.float32), float(per.mean()) if per.size else 0.0


def save_pca_plots(out_dir, components, explained_ratio, max_modes=6):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = min(int(max_modes), int(components.shape[0]))
    x = np.arange(components.shape[1])
    saved = []
    for i in range(n):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(x, components[i], linewidth=1.5)
        ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.4)
        ax.set_title(f'PCA mode {i + 1} | explained={explained_ratio[i]:.4f}')
        ax.set_xlabel('contour point index')
        ax.set_ylabel('normal residual pattern')
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = out_dir / f'pca_mode_{i + 1:02d}.png'
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(path)
    return saved


def class_summary(r_n, class_ids, min_count=5):
    rows = []
    for cls in sorted(set(class_ids.tolist())):
        if cls < 0:
            continue
        mask = class_ids == cls
        if int(mask.sum()) < min_count:
            continue
        per_fourier, fourier_mean = fourier_energy_ratio(r_n[mask], 8)
        rows.append({
            'class_id': int(cls),
            'count': int(mask.sum()),
            'abs_median_feat': float(np.median(np.abs(r_n[mask]))),
            'fourier8_energy_mean': float(fourier_mean),
            'fourier8_energy_median': float(np.median(per_fourier)) if per_fourier.size else 0.0,
        })
    return rows


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

    residuals = []
    residuals_px = []
    class_ids = []
    init_ious = []
    sample_indices = []
    contour_counts = []
    seen_samples = 0

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
        gt_aligned = _align_gt(pred, output['i_gt_py'])
        normals = _contour_normals(pred)
        r_n = ((gt_aligned - pred) * normals).sum(dim=-1)

        r_np = r_n.detach().cpu().numpy().astype(np.float32)
        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        gt_np = gt_aligned.detach().cpu().numpy().astype(np.float32)
        residuals.append(r_np)
        residuals_px.append(r_np * down_ratio)
        class_ids.append(flatten_class_ids(batch, n_contours))
        if args.skip_iou:
            init_ious.append(np.full((n_contours,), np.nan, dtype=np.float32))
        else:
            init_np = output['i_it_py'].detach().cpu().numpy().astype(np.float32)
            init_ious.append(contour_iou(init_np, gt_np, output['image_hw'], down_ratio))
        sample_indices.append(np.full((n_contours,), batch_idx, dtype=np.int32))
        contour_counts.append(n_contours)

        seen_samples += int(batch['inp'].shape[0])
        if batch_idx % 10 == 0:
            print(f'[*] processed samples={seen_samples} contours={sum(contour_counts)}')

    if not residuals:
        raise RuntimeError('No valid contours collected.')

    r_n_all = np.concatenate(residuals, axis=0)
    r_n_px_all = np.concatenate(residuals_px, axis=0)
    class_id_all = np.concatenate(class_ids, axis=0)
    init_iou_all = np.concatenate(init_ious, axis=0)
    sample_idx_all = np.concatenate(sample_indices, axis=0)

    pca = run_pca(r_n_all, args.pca_k)
    fourier_per, fourier_mean = fourier_energy_ratio(r_n_all, args.fourier_modes)
    abs_px = np.abs(r_n_px_all).reshape(-1)
    amp_quantiles = {
        'p50': float(np.percentile(abs_px, 50)),
        'p90': float(np.percentile(abs_px, 90)),
        'p95': float(np.percentile(abs_px, 95)),
        'p99': float(np.percentile(abs_px, 99)),
    }

    np.savez_compressed(
        out_dir / 'r_n_all.npz',
        r_n=r_n_all,
        r_n_px=r_n_px_all,
        class_id=class_id_all,
        init_iou=init_iou_all,
        sample_idx=sample_idx_all,
        fourier8_energy=fourier_per,
        down_ratio=np.asarray(down_ratio, dtype=np.float32),
        checkpoint=str(ckpt_path),
    )
    np.savez_compressed(
        out_dir / 'pca_spectrum.npz',
        singular_values=pca['singular_values'],
        explained_variance_ratio=pca['explained_variance_ratio'],
        cum_explained_variance_ratio=pca['cum_explained_variance_ratio'],
        components=pca['components'],
        n90=np.asarray(pca['n90'], dtype=np.int32),
    )
    plot_paths = save_pca_plots(out_dir, pca['components'], pca['explained_variance_ratio'])

    summary = {
        'cfg_file': str(args.cfg_file),
        'checkpoint': str(ckpt_path),
        'samples_seen': int(seen_samples),
        'num_contours': int(r_n_all.shape[0]),
        'P': int(r_n_all.shape[1]),
        'down_ratio': down_ratio,
        'amp_abs_px_quantiles': amp_quantiles,
        'pca_n90': int(pca['n90']),
        'pca_cum6': float(pca['cum_explained_variance_ratio'][min(5, len(pca['cum_explained_variance_ratio']) - 1)]),
        'pca_cum12': float(pca['cum_explained_variance_ratio'][min(11, len(pca['cum_explained_variance_ratio']) - 1)]),
        'fourier_modes': int(args.fourier_modes),
        'fourier_energy_mean': float(fourier_mean),
        'fourier_energy_median': float(np.median(fourier_per)) if fourier_per.size else 0.0,
        'class_summary_min5': class_summary(r_n_all, class_id_all),
        'pca_plots': [str(p) for p in plot_paths],
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n=== V5 GT normal residual summary ===')
    print(f'P: {summary["P"]}')
    print(f'total contours: {summary["num_contours"]}')
    print(f'|r_n| px median/p95: {amp_quantiles["p50"]:.4f} / {amp_quantiles["p95"]:.4f}')
    print(f'PCA cumulative EVR top6/top12: {summary["pca_cum6"]:.4f} / {summary["pca_cum12"]:.4f}')
    print(f'PCA modes for 90% energy: {summary["pca_n90"]}')
    print(f'Fourier {args.fourier_modes}-mode energy mean/median: {fourier_mean:.4f} / {summary["fourier_energy_median"]:.4f}')
    print(f'outputs: {out_dir}')


if __name__ == '__main__':
    main()
