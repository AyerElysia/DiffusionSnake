#!/usr/bin/env python3
"""
奖励函数设计诊断脚本 (只读，不修改任何训练代码)。

用法:
    CUDA_VISIBLE_DEVICES=4 CFG_FILE=configs/1232_final_v5_geom8_extrap1p0_gpu6.yaml \
        /home/medteam/miniconda3/envs/snake1/bin/python \
        scripts/diagnose_reward_function.py \
        --cfg configs/1232_final_v5_geom8_extrap1p0_gpu6.yaml \
        --ckpt data/outputs/1232_final_v5_geom8_extrap1p0_bs6_gpu6/checkpoints/latest.pt \
        --gpu 4 \
        --max_samples 100 \
        --out_dir report

分析内容:
  1. region / dice / iou / dist_score 四个奖励分量的描述统计 & 两两 Pearson 相关
  2. detail_score 子项 (curvature_detail_reward) 的描述统计
  3. 各分量与"真实 IoU 改进量 = 精修后IoU - 初始IoU"的 Pearson 相关
  4. 在 iou > 0.85 的高质量子集上重复上述分析
  5. 各分量 histogram + 各分量 vs 真实改进量散点图 -> 保存到 out_dir
  6. 打印完整文字摘要
"""

from __future__ import annotations

import argparse
import os
import sys
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# 在 import lib 之前先设置好环境变量
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--cfg', '--cfg_file', dest='cfg_file',
                  default=str(_ROOT / 'configs' / '1232_final_v5_geom8_extrap1p0_gpu6.yaml'))
_pre.add_argument('--ckpt', default=str(
    _ROOT / 'data/outputs/1232_final_v5_geom8_extrap1p0_bs6_gpu6/checkpoints/latest.pt'))
_pre.add_argument('--gpu', default='4', type=str)
_pre.add_argument('--max_samples', default=100, type=int,
                  help='最多处理多少个样本(图像), <=0 表示全量')
_pre.add_argument('--high_iou_thresh', default=0.85, type=float)
_pre.add_argument('--out_dir', default=str(_ROOT / 'report'), type=str)
_pre.add_argument('--seed', default=20260708, type=int)
_pre_args, _remain = _pre.parse_known_args()

os.environ['CFG_FILE'] = str(_pre_args.cfg_file)
os.environ['CUDA_VISIBLE_DEVICES'] = str(_pre_args.gpu)
os.environ['GRPO_V2_GPU'] = str(_pre_args.gpu)

# sys.argv 里只留 cfg_file 参数，其余训练脚本可能有的 argparse 不干扰
sys.argv = [sys.argv[0], '--cfg_file', str(_pre_args.cfg_file)] + _remain

# ---------------------------------------------------------------------------
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.config import cfg
from lib.datasets import make_data_loader
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.train.rewards.region_reward import (
    compute_region_score,
    _poly_to_mask_np, _calc_iou, _calc_dice, _calc_mboundf,
    _calc_boundary_distance_score,
)
from lib.train.rewards.curvature_detail_reward import compute_curvature_detail_score
from lib.utils.snake import snake_config, snake_gcn_utils
from grpo_train_v5_geom_action import (
    _adapt_state_dict,
    _extract_state_dict,
    _flatten_valid_polys,
    _make_py_ind,
    _align_gt,
    _outer_action_mean,
    _burr_penalty,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
DOWN_RATIO = float(snake_config.down_ratio)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _move_batch(batch, device):
    for k in list(batch.keys()):
        if k == 'meta':
            continue
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device, non_blocking=True)
    return batch


def _load_model(ckpt_path: Path, device):
    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True
    cfg.use_flow_matching = True
    cfg.use_grpo = True

    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    net = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network

    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    raw = torch.load(str(ckpt_path), map_location='cpu')
    sd = _adapt_state_dict(net, _extract_state_dict(raw))
    missing, unexpected = net.load_state_dict(sd, strict=False)
    total = len(list(net.state_dict().keys()))
    ratio = 100.0 * (total - len(missing)) / max(total, 1)
    print(f'[diag] ckpt={ckpt_path.name}  load_ratio={ratio:.2f}%  missing={len(missing)} unexpected={len(unexpected)}')
    if ratio < 90.0:
        raise RuntimeError(f'load ratio too low: {ratio:.2f}%')
    net.to(device).eval()
    inner = net.net if hasattr(net, 'net') else net
    return inner


@torch.no_grad()
def _manual_context(inner, batch, device):
    inner.eval()
    yolo_out = inner.yolo(batch['inp'])
    feat_list = yolo_out[1] if isinstance(yolo_out, (list, tuple)) and len(yolo_out) > 1 else None
    feat_p2 = feat_list[0] if isinstance(feat_list, (list, tuple)) else yolo_out
    cnn_feat = inner.cnn_proj(feat_p2)
    if getattr(inner, 'use_p3_features', False) and hasattr(inner, 'cnn_proj_p3'):
        if isinstance(feat_list, (list, tuple)) and len(feat_list) > 1:
            feat_p3_up = torch.nn.functional.interpolate(
                feat_list[1], size=feat_p2.shape[-2:], mode='bilinear', align_corners=False)
            cnn_feat = cnn_feat + inner.cnn_proj_p3(feat_p3_up)

    i_init = _flatten_valid_polys(batch, 'i_it_py', device=device)
    i_gt   = _flatten_valid_polys(batch, 'i_gt_py', device=device)
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

    return {
        'cnn_feature': cnn_feat.detach(),
        'i_it_py':    i_init.detach(),
        'c_it_py':    c_init.detach(),
        'i_gt_py':    _align_gt(i_init, i_gt).detach(),
        'py_ind':     py_ind.detach(),
        'image_hw':   (int(batch['inp'].shape[-2]), int(batch['inp'].shape[-1])),
    }


@torch.no_grad()
def _det_rollout(output, flow, fractions, ode_steps):
    current = output['i_it_py'].detach()
    total   = torch.zeros_like(current)
    for frac in fractions:
        c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
        act   = _outer_action_mean(
            flow, output['cnn_feature'], current, c_cur,
            output['py_ind'], float(frac), int(ode_steps))
        current = (current + act).detach()
        total   = total + act.detach()
    return output['i_it_py'] + total


def _cv(name, default):
    v = getattr(cfg, f'rl_v4_{name}', None)
    if v is not None:
        return v
    v4 = getattr(cfg, 'rl_v4', None)
    if v4 is not None and name in v4:
        return v4[name]
    return default


def _get_rollout_schedule():
    outer = int(_cv('outer_steps', 5))
    fracs = [float(x) for x in list(_cv('fractions', [0.2, 0.25, 0.3333, 0.5, 1.0]))]
    if len(fracs) < outer:
        fracs = fracs + [1.0] * (outer - len(fracs))
    fracs = fracs[:outer]
    ode = int(_cv('ode_steps', getattr(cfg, 'iterative_ode_steps',
                                       getattr(cfg, 'flow_ode_steps', 10))))
    return fracs, max(ode, 1)


def _get_detail_kwargs():
    return dict(
        coord_scale=DOWN_RATIO,
        corner_dist_max_px=float(_cv('reward_detail_corner_dist_max_px', 6.0)),
        corner_dist_quantile=float(_cv('reward_detail_corner_dist_quantile', 95.0)),
        corner_dist_quantile_weight=float(_cv('reward_detail_corner_dist_quantile_weight', 0.7)),
        curvature_max_px=float(_cv('reward_detail_curvature_max_px', 4.0)),
        burr_margin_px=float(_cv('reward_burr_margin_px', 0.5)),
        burr_max_px=float(_cv('reward_burr_max_px', 1.5)),
        burr_quantile=float(_cv('reward_burr_quantile', 95.0)),
        local_band_radius_px=int(_cv('reward_detail_local_band_radius_px', 2)),
        area_max_frac=float(_cv('reward_detail_area_max_frac', 0.15)),
        w_corner_dist=float(_cv('reward_detail_w_corner_dist', 0.35)),
        w_curv_match=float(_cv('reward_detail_w_curv_match', 0.20)),
        w_local_biou=float(_cv('reward_detail_w_local_biou', 0.10)),
        w_burr=float(_cv('reward_detail_w_burr', 0.07)),
        w_area=float(_cv('reward_detail_w_area', 0.03)),
        return_components=True,
    )


def _get_reward_weights():
    return dict(
        w_region=float(_cv('reward_w_region', 0.30)),
        w_dice=float(_cv('reward_w_dice', 0.10)),
        w_iou=float(_cv('reward_w_iou', 0.25)),
        w_dist=float(_cv('reward_w_dist', 0.35)),
        dist_max_px=float(_cv('reward_dist_max_px', 8.0)),
        dist_quantile=float(_cv('reward_dist_quantile', 95.0)),
        dist_quantile_weight=float(_cv('reward_dist_quantile_weight', 0.5)),
        burr_weight=float(_cv('reward_burr_weight', 0.06)),
        burr_max_px=float(_cv('reward_burr_max_px', 1.5)),
        burr_margin_px=float(_cv('reward_burr_margin_px', 0.50)),
        burr_quantile=float(_cv('reward_burr_quantile', 95.0)),
    )


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------
def _stats(arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, min=np.nan, p10=np.nan,
                    median=np.nan, p90=np.nan, max=np.nan)
    return dict(
        n=int(finite.size),
        mean=float(np.mean(finite)),
        std=float(np.std(finite)),
        min=float(np.min(finite)),
        p10=float(np.percentile(finite, 10)),
        median=float(np.median(finite)),
        p90=float(np.percentile(finite, 90)),
        max=float(np.max(finite)),
    )


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _fmt(v, d=4):
    if v is None or not np.isfinite(v):
        return 'N/A'
    return f'{float(v):.{d}f}'


# ---------------------------------------------------------------------------
# main data-collection loop
# ---------------------------------------------------------------------------
def _collect(inner, loader, fractions, ode_steps, rw, dkw, device, max_samples):
    flow = inner.gcn
    records = []
    n_images = 0
    print(f'[diag] collecting (max_samples={max_samples}) ...')

    for batch_idx, batch in enumerate(loader):
        if max_samples > 0 and n_images >= max_samples:
            break
        batch = _move_batch(batch, device)
        try:
            ctx = _manual_context(inner, batch, device)
        except Exception as e:
            print(f'[diag] skip batch {batch_idx}: {e}')
            continue
        if ctx['i_it_py'].numel() == 0:
            continue

        i_init = ctx['i_it_py']   # (N, P, 2) feature-space
        i_gt   = ctx['i_gt_py']
        hw     = ctx['image_hw']
        H, W   = hw

        # ---- deterministic rollout: pred contour ----
        pred = _det_rollout(ctx, flow, fractions, ode_steps)   # (N, P, 2)

        # ---- initial IoU (octagon init) ----
        init_iou_t = compute_region_score(
            i_init, i_gt, H=H, W=W,
            w_boundary=0, w_dice=0, w_iou=1, w_dist=0,
            coord_scale=DOWN_RATIO)

        # ---- individual reward components (pred) ----
        region_t = compute_region_score(
            pred, i_gt, H=H, W=W,
            w_boundary=1, w_dice=0, w_iou=0, w_dist=0,
            coord_scale=DOWN_RATIO)

        dice_t = compute_region_score(
            pred, i_gt, H=H, W=W,
            w_boundary=0, w_dice=1, w_iou=0, w_dist=0,
            coord_scale=DOWN_RATIO)

        iou_t = compute_region_score(
            pred, i_gt, H=H, W=W,
            w_boundary=0, w_dice=0, w_iou=1, w_dist=0,
            coord_scale=DOWN_RATIO)

        dist_t = compute_region_score(
            pred, i_gt, H=H, W=W,
            w_boundary=0, w_dice=0, w_iou=0, w_dist=1,
            dist_max_px=rw['dist_max_px'],
            dist_quantile=rw['dist_quantile'],
            dist_quantile_weight=rw['dist_quantile_weight'],
            coord_scale=DOWN_RATIO)

        # ---- composite reward (as actually used in training) ----
        composite_t = (rw['w_region'] * region_t
                       + rw['w_dice']   * dice_t
                       + rw['w_iou']    * iou_t
                       + rw['w_dist']   * dist_t)

        # ---- burr penalty ----
        burr_pen_t, burr_raw_t = _burr_penalty(
            pred, i_init, i_gt,
            coord_scale=DOWN_RATIO,
            margin_px=rw['burr_margin_px'],
            max_px=rw['burr_max_px'],
            quantile=rw['burr_quantile'])

        # ---- detail score (currently disabled, weight=0) ----
        detail_t, detail_comps = compute_curvature_detail_score(
            pred, i_gt, H=H, W=W, **dkw)

        # ---- convert all to numpy, per contour ----
        N = i_init.size(0)
        for ci in range(N):
            rec = {
                'init_iou':    float(init_iou_t[ci].item()),
                'pred_iou':    float(iou_t[ci].item()),
                'iou_gain':    float(iou_t[ci].item()) - float(init_iou_t[ci].item()),
                'region':      float(region_t[ci].item()),
                'dice':        float(dice_t[ci].item()),
                'iou':         float(iou_t[ci].item()),
                'dist':        float(dist_t[ci].item()),
                'composite':   float(composite_t[ci].item()),
                'burr_penalty': float(burr_pen_t[ci].item()),
                'burr_raw_px':  float(burr_raw_t[ci].item()),
                'detail_score': float(detail_t[ci].item()),
                'corner_dist':  float(detail_comps['corner_dist'][ci].item()),
                'curv_match':   float(detail_comps['curv_match'][ci].item()),
                'local_biou':   float(detail_comps['local_biou'][ci].item()),
                'detail_burr':  float(detail_comps['burr_penalty'][ci].item()),
                'area_penalty': float(detail_comps['area_penalty'][ci].item()),
            }
            records.append(rec)

        n_images += 1
        if n_images % 10 == 0:
            print(f'[diag] {n_images} images  {len(records)} contours')

    print(f'[diag] done: {n_images} images  {len(records)} contours total')
    return records


# ---------------------------------------------------------------------------
# analysis: correlation matrices & stats tables
# ---------------------------------------------------------------------------
def _analyze(records, high_iou_thresh=0.85):
    if not records:
        return {}

    keys_main = ['region', 'dice', 'iou', 'dist', 'composite', 'burr_penalty',
                 'detail_score', 'corner_dist', 'curv_match', 'local_biou',
                 'detail_burr', 'area_penalty']
    keys_redundancy = ['region', 'dice', 'iou']
    keys_vs_gain = ['region', 'dice', 'iou', 'dist', 'composite',
                    'burr_penalty', 'detail_score', 'corner_dist',
                    'curv_match', 'local_biou', 'detail_burr']

    def _arr(k):
        return np.array([r[k] for r in records], dtype=np.float64)

    iou_gain = _arr('iou_gain')
    pred_iou = _arr('pred_iou')
    init_iou = _arr('init_iou')

    # --- 全量统计 ---
    stats_all = {k: _stats(_arr(k)) for k in keys_main}
    stats_all['init_iou']  = _stats(init_iou)
    stats_all['pred_iou']  = _stats(pred_iou)
    stats_all['iou_gain']  = _stats(iou_gain)

    # --- 全量两两 Pearson ---
    corr_redundancy = {}
    for i, ki in enumerate(keys_redundancy):
        for j, kj in enumerate(keys_redundancy):
            if j > i:
                corr_redundancy[f'{ki}_vs_{kj}'] = _pearson(_arr(ki), _arr(kj))

    corr_vs_gain = {f'{k}_vs_iou_gain': _pearson(_arr(k), iou_gain)
                    for k in keys_vs_gain}

    # --- 高质量子集 (iou > thresh) ---
    hi_mask = pred_iou > high_iou_thresh
    hi_n = int(hi_mask.sum())
    hi_records = [r for r, m in zip(records, hi_mask) if m]

    if hi_records:
        stats_hi = {k: _stats(_arr(k)[hi_mask]) for k in keys_main}
        stats_hi['iou_gain'] = _stats(iou_gain[hi_mask])
        corr_hi_redundancy = {}
        for i, ki in enumerate(keys_redundancy):
            for j, kj in enumerate(keys_redundancy):
                if j > i:
                    corr_hi_redundancy[f'{ki}_vs_{kj}'] = \
                        _pearson(_arr(ki)[hi_mask], _arr(kj)[hi_mask])
        corr_hi_vs_gain = {
            f'{k}_vs_iou_gain': _pearson(_arr(k)[hi_mask], iou_gain[hi_mask])
            for k in keys_vs_gain
        }
    else:
        stats_hi = {}
        corr_hi_redundancy = {}
        corr_hi_vs_gain = {}

    return {
        'n_total': len(records),
        'n_high_iou': hi_n,
        'high_iou_thresh': high_iou_thresh,
        'stats_all': stats_all,
        'corr_redundancy_all': corr_redundancy,
        'corr_vs_gain_all': corr_vs_gain,
        'stats_high_iou': stats_hi,
        'corr_redundancy_high_iou': corr_hi_redundancy,
        'corr_vs_gain_high_iou': corr_hi_vs_gain,
    }


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _plot_histograms(records, out_dir: Path):
    keys = ['region', 'dice', 'iou', 'dist', 'burr_penalty',
            'detail_score', 'corner_dist', 'curv_match',
            'local_biou', 'detail_burr', 'area_penalty']
    titles = {
        'region': 'Region (boundary F-score)',
        'dice': 'Dice',
        'iou': 'IoU',
        'dist': 'Dist Score',
        'burr_penalty': 'Burr Penalty',
        'detail_score': 'Detail Score (combined)',
        'corner_dist': 'Detail: Corner Dist',
        'curv_match': 'Detail: Curv Match',
        'local_biou': 'Detail: Local BIoU',
        'detail_burr': 'Detail: Burr',
        'area_penalty': 'Detail: Area Penalty',
    }
    n = len(keys)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for idx, k in enumerate(keys):
        vals = np.array([r[k] for r in records], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        ax = axes[idx]
        ax.hist(vals, bins=40, edgecolor='none', alpha=0.8, color='steelblue')
        ax.set_title(titles.get(k, k), fontsize=9)
        ax.set_xlabel('score', fontsize=8)
        ax.set_ylabel('count', fontsize=8)
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        ax.axvline(mu, color='red', lw=1.2, linestyle='--', label=f'μ={mu:.3f}')
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Reward Component Histograms (all samples)', fontsize=11, y=1.02)
    fig.tight_layout()
    out = out_dir / 'reward_diagnosis_histograms.png'
    fig.savefig(str(out), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[diag] saved {out}')


def _plot_scatter_vs_gain(records, out_dir: Path):
    keys = ['region', 'dice', 'iou', 'dist', 'burr_penalty',
            'detail_score', 'corner_dist', 'curv_match',
            'local_biou', 'detail_burr']
    titles = {
        'region': 'Region vs IoU gain',
        'dice': 'Dice vs IoU gain',
        'iou': 'IoU vs IoU gain',
        'dist': 'Dist Score vs IoU gain',
        'burr_penalty': 'Burr Penalty vs IoU gain',
        'detail_score': 'Detail Score vs IoU gain',
        'corner_dist': 'Corner Dist vs IoU gain',
        'curv_match': 'Curv Match vs IoU gain',
        'local_biou': 'Local BIoU vs IoU gain',
        'detail_burr': 'Detail Burr vs IoU gain',
    }
    iou_gain = np.array([r['iou_gain'] for r in records], dtype=np.float64)

    cols = 5
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()

    for idx, k in enumerate(keys):
        vals = np.array([r[k] for r in records], dtype=np.float64)
        ax = axes[idx]
        mask = np.isfinite(vals) & np.isfinite(iou_gain)
        ax.scatter(vals[mask], iou_gain[mask], s=6, alpha=0.35, color='steelblue')
        r = _pearson(vals, iou_gain)
        ax.set_title(f'{titles.get(k,k)}\nr={_fmt(r, 3)}', fontsize=9)
        ax.set_xlabel(k, fontsize=8)
        ax.set_ylabel('IoU gain', fontsize=8)
        ax.axhline(0, color='gray', lw=0.7, linestyle=':')
        ax.tick_params(labelsize=7)

    for idx in range(len(keys), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Reward Components vs IoU gain (pred_iou − init_iou)', fontsize=11, y=1.02)
    fig.tight_layout()
    out = out_dir / 'reward_diagnosis_scatter_vs_gain.png'
    fig.savefig(str(out), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[diag] saved {out}')


def _plot_corr_heatmap(results, out_dir: Path):
    """2x2 correlation matrices: all / high-iou, redundancy / vs-gain."""
    keys_r = ['region', 'dice', 'iou']
    keys_g = ['region', 'dice', 'iou', 'dist', 'composite',
              'burr_penalty', 'detail_score', 'curv_match', 'detail_burr']
    tag_pairs = [
        ('corr_redundancy_all', keys_r, 'Redundancy corr (all)'),
        ('corr_vs_gain_all', keys_g, 'vs IoU-gain corr (all)'),
        ('corr_redundancy_high_iou', keys_r, f'Redundancy corr (IoU>{results["high_iou_thresh"]:.2f})'),
        ('corr_vs_gain_high_iou', keys_g, f'vs IoU-gain corr (IoU>{results["high_iou_thresh"]:.2f})'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (corr_key, ks, title) in enumerate(tag_pairs[1:3]):  # vs-gain and redundancy side-by-side
        ax = axes[ax_idx]
        corr_dict = results.get(corr_key, {})
        if corr_key.startswith('corr_vs_gain'):
            labels = ks
            mat = np.zeros((1, len(labels)))
            for ci, k in enumerate(labels):
                v = corr_dict.get(f'{k}_vs_iou_gain', np.nan)
                mat[0, ci] = float(v) if v is not None else np.nan
            im = ax.imshow(mat, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
            ax.set_yticks([0])
            ax.set_yticklabels(['corr w/ IoU-gain'], fontsize=8)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
            for ci in range(len(labels)):
                v = mat[0, ci]
                ax.text(ci, 0, _fmt(v, 3), ha='center', va='center',
                        fontsize=7, color='black' if abs(v) < 0.6 else 'white')
        else:
            n = len(ks)
            mat = np.zeros((n, n))
            for i, ki in enumerate(ks):
                for j, kj in enumerate(ks):
                    if i == j:
                        mat[i, j] = 1.0
                    elif j > i:
                        v = corr_dict.get(f'{ki}_vs_{kj}', np.nan)
                        mat[i, j] = float(v) if v is not None else np.nan
                        mat[j, i] = mat[i, j]
            im = ax.imshow(mat, vmin=-1, vmax=1, cmap='RdBu_r')
            ax.set_xticks(range(n))
            ax.set_xticklabels(ks, rotation=35, ha='right', fontsize=8)
            ax.set_yticks(range(n))
            ax.set_yticklabels(ks, fontsize=8)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, _fmt(mat[i, j], 3), ha='center', va='center',
                            fontsize=7, color='black' if abs(mat[i, j]) < 0.6 else 'white')
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    out = out_dir / 'reward_diagnosis_corr_heatmap.png'
    fig.savefig(str(out), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[diag] saved {out}')


# ---------------------------------------------------------------------------
# text report printer
# ---------------------------------------------------------------------------
def _print_report(results, rw):
    sep = '=' * 72

    def _row(name, s):
        return (f'  {name:<20s}  n={s["n"]:4d}  mean={_fmt(s["mean"])}  '
                f'std={_fmt(s["std"])}  p10={_fmt(s["p10"])}  '
                f'p90={_fmt(s["p90"])}')

    print(sep)
    print('REWARD FUNCTION DIAGNOSIS REPORT')
    print(f'  total contours : {results["n_total"]}')
    print(f'  high-IoU (>{results["high_iou_thresh"]:.2f}) : {results["n_high_iou"]}')
    print()
    print('Active reward weights:')
    print(f'  region={rw["w_region"]}  dice={rw["w_dice"]}  '
          f'iou={rw["w_iou"]}  dist={rw["w_dist"]}  '
          f'burr_weight={rw["burr_weight"]}  detail_weight=0.0 (disabled)')
    print(sep)

    print('\n[A] COMPONENT STATISTICS – ALL SAMPLES')
    sa = results['stats_all']
    for k in ['init_iou', 'pred_iou', 'iou_gain',
              'region', 'dice', 'iou', 'dist', 'composite', 'burr_penalty',
              'detail_score', 'corner_dist', 'curv_match', 'local_biou',
              'detail_burr', 'area_penalty']:
        if k in sa:
            print(_row(k, sa[k]))

    print('\n[B] REDUNDANCY CORRELATIONS (region / dice / iou) – ALL')
    for k, v in results['corr_redundancy_all'].items():
        print(f'  {k}: {_fmt(v, 4)}')

    print('\n[C] CORRELATION WITH IoU-GAIN – ALL SAMPLES')
    for k, v in results['corr_vs_gain_all'].items():
        print(f'  {k}: {_fmt(v, 4)}')

    if results['stats_high_iou']:
        print(f'\n[D] COMPONENT STATISTICS – HIGH-IoU SUBSET (>{results["high_iou_thresh"]:.2f})')
        sh = results['stats_high_iou']
        for k in ['iou_gain', 'region', 'dice', 'iou', 'dist', 'burr_penalty',
                  'detail_score', 'corner_dist', 'curv_match', 'local_biou',
                  'detail_burr']:
            if k in sh:
                print(_row(k, sh[k]))

        print(f'\n[E] REDUNDANCY CORRELATIONS – HIGH-IoU SUBSET')
        for k, v in results['corr_redundancy_high_iou'].items():
            print(f'  {k}: {_fmt(v, 4)}')

        print(f'\n[F] CORRELATION WITH IoU-GAIN – HIGH-IoU SUBSET')
        for k, v in results['corr_vs_gain_high_iou'].items():
            print(f'  {k}: {_fmt(v, 4)}')

    print(sep)
    print('\n[VERDICT]')
    # --- auto-generate verdict ---
    sa = results['stats_all']
    sh = results.get('stats_high_iou', {})

    # redundancy: region/dice/iou pairwise corr
    rdx_vals = [v for v in results['corr_redundancy_all'].values() if np.isfinite(v)]
    rdx_mean = float(np.mean(rdx_vals)) if rdx_vals else np.nan

    # high-IoU std of region/dice/iou
    hi_std_region = sh.get('region', {}).get('std', np.nan)
    hi_std_iou    = sh.get('iou',    {}).get('std', np.nan)
    hi_std_dist   = sh.get('dist',   {}).get('std', np.nan)
    hi_std_detail = sh.get('detail_score', {}).get('std', np.nan)

    # vs-gain correlation
    cg = results['corr_vs_gain_all']
    cg_region  = cg.get('region_vs_iou_gain', np.nan)
    cg_iou     = cg.get('iou_vs_iou_gain',    np.nan)
    cg_dist    = cg.get('dist_vs_iou_gain',   np.nan)
    cg_detail  = cg.get('detail_score_vs_iou_gain', np.nan)
    cg_curv    = cg.get('curv_match_vs_iou_gain', np.nan)
    cg_dbur    = cg.get('detail_burr_vs_iou_gain', np.nan)

    cg_hi      = results.get('corr_vs_gain_high_iou', {})
    cg_hi_dist = cg_hi.get('dist_vs_iou_gain', np.nan)
    cg_hi_det  = cg_hi.get('detail_score_vs_iou_gain', np.nan)

    print(f'1. region/dice/iou 冗余度: 均值 pairwise r = {_fmt(rdx_mean, 4)}')
    if np.isfinite(rdx_mean):
        if rdx_mean > 0.95:
            print('   → 高度冗余 (r>0.95)：三项携带几乎相同信号，可大幅降权。')
        elif rdx_mean > 0.85:
            print('   → 中度冗余 (r>0.85)：有重叠，其中 dice 最可削减。')
        else:
            print('   → 冗余度适中，三项仍有独立信息。')

    print(f'\n2. 高质量子集 (IoU>{results["high_iou_thresh"]:.2f}) 各分量标准差:')
    print(f'   region std={_fmt(hi_std_region)}  iou std={_fmt(hi_std_iou)}  '
          f'dist std={_fmt(hi_std_dist)}  detail std={_fmt(hi_std_detail)}')
    if np.isfinite(hi_std_iou) and np.isfinite(hi_std_dist):
        if hi_std_iou < 0.5 * hi_std_dist:
            print('   → region/iou 在高质量区间方差显著小于 dist，精修阶段区分度不足。')
        else:
            print('   → 各分量方差在高质量区间差异有限。')

    print(f'\n3. 与 IoU-gain 的相关性 (全量):')
    print(f'   region r={_fmt(cg_region)}  iou r={_fmt(cg_iou)}  '
          f'dist r={_fmt(cg_dist)}  detail r={_fmt(cg_detail)}')
    print(f'   curv_match r={_fmt(cg_curv)}  detail_burr r={_fmt(cg_dbur)}')
    print(f'   高质量子集: dist r={_fmt(cg_hi_dist)}  detail r={_fmt(cg_hi_det)}')

    if np.isfinite(cg_dist) and np.isfinite(cg_iou):
        if cg_dist > cg_iou + 0.05:
            print('   → dist_score 与真实改进量相关性高于 iou，更适合做精修 reward。')
        elif abs(cg_dist - cg_iou) < 0.05:
            print('   → dist_score 与 iou 与真实改进量相关性相近。')
        else:
            print('   → iou 与真实改进量相关性略高于 dist（但差异可能在误差范围内）。')

    print()
    print('综合结论:')
    lines = []
    if np.isfinite(rdx_mean) and rdx_mean > 0.90:
        lines.append('- region / dice / iou 高度冗余（r>.90），当前三项合计权重 0.65 存在信号重复。')
        lines.append('  建议保留其中1项代表整体 overlap（保留 iou 或 region），降低 dice 权重至 0。')
    if np.isfinite(hi_std_dist) and np.isfinite(hi_std_iou) and hi_std_iou < hi_std_dist:
        lines.append('- 高质量子集中 dist_score 方差 > iou 方差，精修阶段 dist 仍有区分度，可适当提权。')
    cg_det = cg_hi.get('detail_score_vs_iou_gain', np.nan)
    if cg_det is None:
        cg_det = np.nan
    cg_det = float(cg_det)
    if np.isfinite(cg_det) and np.isfinite(cg_hi_dist):
        if abs(cg_det) > 0.05:
            lines.append('- detail_score 在高质量子集中与真实改进量有显著相关，建议开启小权重试验。')
        else:
            lines.append('- detail_score 在高质量子集中与真实改进量相关性弱，暂不建议高权重启用。')
    if not lines:
        lines.append('- 奖励函数设计基本合理，未发现明显冗余或盲区。')
    for l in lines:
        print(l)
    print(sep)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main():
    args = _pre_args
    _set_seed(int(args.seed))

    # GPU setup
    gpu = str(args.gpu).strip()
    if gpu:
        cfg.gpus = [int(gpu.split(',')[0])]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        print(f'[diag] GPU={gpu}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[diag] device={device}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(str(args.ckpt)).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = _ROOT / ckpt_path

    fractions, ode_steps = _get_rollout_schedule()
    rw   = _get_reward_weights()
    dkw  = _get_detail_kwargs()

    print(f'[diag] rollout fractions={fractions}  ode_steps={ode_steps}')
    print(f'[diag] reward weights: region={rw["w_region"]} dice={rw["w_dice"]} '
          f'iou={rw["w_iou"]} dist={rw["w_dist"]}')

    inner = _load_model(ckpt_path, device)

    # use test dataset (BtcvVal)
    loader = make_data_loader(cfg, is_train=False, is_distributed=False)
    print(f'[diag] test dataset: {cfg.test.dataset}  img_path={cfg.test.img_path}')

    # collect per-contour records
    records = _collect(
        inner, loader, fractions, ode_steps, rw, dkw, device,
        max_samples=int(args.max_samples),
    )

    if not records:
        print('[diag] ERROR: no records collected. Exiting.')
        return

    # analyse
    results = _analyze(records, high_iou_thresh=float(args.high_iou_thresh))

    # save JSON
    json_path = out_dir / 'reward_diagnosis_results.json'
    with open(json_path, 'w') as f:
        # convert numpy floats to native Python for JSON serialisation
        def _to_py(v):
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            if isinstance(v, dict):
                return {kk: _to_py(vv) for kk, vv in v.items()}
            return v
        json.dump({k: _to_py(v) for k, v in results.items()}, f, indent=2)
    print(f'[diag] saved {json_path}')

    # plots
    _plot_histograms(records, out_dir)
    _plot_scatter_vs_gain(records, out_dir)
    _plot_corr_heatmap(results, out_dir)

    # text report
    _print_report(results, rw)


if __name__ == '__main__':
    main()
