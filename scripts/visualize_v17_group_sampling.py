#!/usr/bin/env python3
"""Offline visualization for V17 GRPO group rollouts.

The training run used grpo_v2_viz_every=0, so this script recreates the same
manual-GT-init rollout group from a checkpoint and draws the sampled contours.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS_DIR))


def _preparse_cfg(argv):
    cfg_file = None
    for i, arg in enumerate(argv):
        if arg == '--cfg_file' and i + 1 < len(argv):
            cfg_file = argv[i + 1]
        elif arg.startswith('--cfg_file='):
            cfg_file = arg.split('=', 1)[1]
    return cfg_file


_cfg_file = _preparse_cfg(sys.argv)
if _cfg_file:
    os.environ['CFG_FILE'] = _cfg_file
elif not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = str(_THIS_DIR / 'configs' / 'btcv_v3_4_fm_rl_v17_mid3_stepgrpo_gpu2.yaml')

# lib.config has its own argparse. Hide this script's arguments from it.
_ORIG_ARGV = list(sys.argv)
sys.argv = [sys.argv[0]]

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.config import cfg
from lib.datasets import make_data_loader
from lib.networks import make_network
from lib.train.grpo_v2_utils import freeze_bn_running_stats
from lib.train.rewards.region_reward import compute_region_reward, compute_region_score
from lib.train.trainers import make_trainer
from lib.utils.snake import snake_config, snake_gcn_utils
from grpo_train_v2 import (
    _adapt_state_dict,
    _align_gt,
    _contour_laplacian_px,
    _extract_state_dict,
    _flatten_valid_polys,
    _make_py_ind,
    _move_batch,
)

sys.argv = _ORIG_ARGV


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', default=os.environ.get('CFG_FILE'))
    parser.add_argument('--ckpt', default='')
    parser.add_argument('--out_dir', default='data/outputs/btcv_v3_4_fm_rl_v17_mid3_stepgrpo_group_viz')
    parser.add_argument('--gpu', default=os.environ.get('EVAL_GPU', os.environ.get('CUDA_VISIBLE_DEVICES', '0')).split(',')[0])
    parser.add_argument('--seed', type=int, default=20260616)
    parser.add_argument('--batch_index', type=int, default=0)
    parser.add_argument('--image_index', type=int, default=0)
    parser.add_argument('--max_rollouts', type=int, default=16)
    parser.add_argument('--k', type=int, default=int(getattr(cfg, 'grpo_v2_k', 16)))
    parser.add_argument('--action_std', type=float, default=float(getattr(cfg, 'grpo_v2_action_std', 0.02)))
    parser.add_argument('--rollout_steps', type=int, default=int(getattr(cfg, 'grpo_v2_rollout_steps', getattr(cfg, 'grpo_steps', 10))))
    parser.add_argument('--window_size', type=int, default=int(getattr(cfg, 'grpo_v2_window_size', 3)))
    parser.add_argument('--window_start', type=int, default=None)
    parser.add_argument('--window_end', type=int, default=None)
    parser.add_argument('--save_prefix', default='v17_group_sampling')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else _THIS_DIR / p


def default_ckpt():
    candidates = [
        _THIS_DIR / 'data/outputs/btcv_v3_4_fm_rl_v17_mid3_stepgrpo_continue_from_best130_gpu6/checkpoints/latest.pt',
        _THIS_DIR / 'data/outputs/btcv_v3_4_fm_rl_v17_mid3_stepgrpo_continue_from_best130_gpu6/checkpoints/step50.pt',
        _THIS_DIR / 'data/outputs/btcv_v3_4_fm_rl_v17_mid3_stepgrpo_continue_from200_gpu7/checkpoints/best_iou.pt',
        _THIS_DIR / 'data/outputs/btcv_v3_4_fm_rl_v17_mid3_stepgrpo_long1200_from300_gpu7/checkpoints/best_iou.pt',
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError('No default V17 checkpoint found; pass --ckpt.')


def load_inner(ckpt_path, device):
    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True
    cfg.use_grpo = True
    cfg.use_flow_matching = True
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    net = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    sd = _extract_state_dict(torch.load(str(ckpt_path), map_location='cpu'))
    sd = _adapt_state_dict(net, sd)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    total = len(net.state_dict())
    ratio = 100.0 * (total - len(missing)) / max(total, 1)
    print(f'[*] Loaded {ckpt_path} | ratio={ratio:.2f}% missing={len(missing)} unexpected={len(unexpected)}')
    trainer.network.to(device)
    inner = net.net if hasattr(net, 'net') else net
    inner.eval()
    freeze_bn_running_stats(inner)
    return inner


@torch.no_grad()
def manual_gt_init_context(inner, batch):
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
    return {
        'cnn_feature': cnn_feature,
        'i_it_py': i_init,
        'c_it_py': c_init,
        'i_gt_py': i_gt,
        'py_ind': py_ind,
    }


@torch.no_grad()
def sample_rollout(gcn, output, action_std, rollout_steps, window_size, window_range):
    cnn_feature = output['cnn_feature']
    i_init = output['i_it_py']
    py_ind = output['py_ind']
    current = i_init.detach()
    total_disp = torch.zeros_like(i_init)
    iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
    fractions = list(getattr(cfg, 'iterative_fractions', [])) or [1.0 / (iter_steps - i) for i in range(iter_steps)]
    iter_ode_steps = int(getattr(cfg, 'iterative_ode_steps', getattr(cfg, 'iterative_ddim_steps', rollout_steps)))
    if iter_ode_steps <= 0:
        iter_ode_steps = rollout_steps
    for frac in fractions[:iter_steps]:
        c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
        ret = gcn.sample_with_logprob(
            cnn_feature,
            current,
            c_cur,
            py_ind,
            steps=iter_ode_steps,
            window_size=window_size,
            window_range=window_range,
            action_std=action_std,
            noise_scale=float(getattr(cfg, 'grpo_v2_rollout_noise_scale', 0.0)),
            step_mode=str(getattr(cfg, 'grpo_v2_step_mode', 'gaussian')).strip().lower(),
            noise_level=float(getattr(cfg, 'grpo_v2_noise_level', 0.0)),
            sde_type=str(getattr(cfg, 'grpo_v2_sde_type', 'sde')).strip().lower(),
        )
        applied = ret['disp'] * float(frac)
        current = (current + applied).detach()
        total_disp = total_disp + applied
    return {'disp': total_disp, 'py': i_init + total_disp}


@torch.no_grad()
def deterministic_policy(gcn, output, rollout_steps):
    cnn_feature = output['cnn_feature']
    i_init = output['i_it_py']
    c_init = output['c_it_py']
    py_ind = output['py_ind']
    iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
    fractions = list(getattr(cfg, 'iterative_fractions', []))
    ode_steps = int(getattr(cfg, 'iterative_ode_steps', getattr(cfg, 'iterative_ddim_steps', rollout_steps)))
    disp = gcn.sample_disp_iterative(
        cnn_feature,
        i_init,
        c_init,
        py_ind,
        num_iter_steps=iter_steps,
        fractions=fractions,
        ode_steps=ode_steps,
    )
    return {'disp': disp, 'py': i_init + disp}


@torch.no_grad()
def compute_scores(output, ret, det_score):
    i_init = output['i_it_py']
    i_gt = _align_gt(i_init, output['i_gt_py'])
    h_img = int(output['cnn_feature'].shape[-2] * snake_config.down_ratio)
    w_img = int(output['cnn_feature'].shape[-1] * snake_config.down_ratio)
    final_score = compute_region_reward(
        i_init,
        ret['disp'],
        i_gt,
        H=h_img,
        W=w_img,
        w1=float(getattr(cfg, 'grpo_v2_reward_w_region', 0.20)),
        w_dice=float(getattr(cfg, 'grpo_v2_reward_w_dice', 0.15)),
        w_iou=float(getattr(cfg, 'grpo_v2_reward_w_iou', 0.40)),
        w_dist=float(getattr(cfg, 'grpo_v2_reward_w_dist', 0.25)),
        dist_max_px=float(getattr(cfg, 'grpo_v2_reward_dist_max_px', 8.0)),
        dist_quantile=float(getattr(cfg, 'grpo_v2_reward_dist_quantile', 95.0)),
        dist_quantile_weight=float(getattr(cfg, 'grpo_v2_reward_dist_quantile_weight', 0.5)),
        coord_scale=float(snake_config.down_ratio),
    ).detach()
    init_score = compute_region_score(
        i_init,
        i_gt,
        H=h_img,
        W=w_img,
        w_boundary=float(getattr(cfg, 'grpo_v2_reward_w_region', 0.20)),
        w_dice=float(getattr(cfg, 'grpo_v2_reward_w_dice', 0.15)),
        w_iou=float(getattr(cfg, 'grpo_v2_reward_w_iou', 0.40)),
        w_dist=float(getattr(cfg, 'grpo_v2_reward_w_dist', 0.25)),
        dist_max_px=float(getattr(cfg, 'grpo_v2_reward_dist_max_px', 8.0)),
        dist_quantile=float(getattr(cfg, 'grpo_v2_reward_dist_quantile', 95.0)),
        dist_quantile_weight=float(getattr(cfg, 'grpo_v2_reward_dist_quantile_weight', 0.5)),
        coord_scale=float(snake_config.down_ratio),
    ).detach()
    final_poly = i_init + ret['disp']
    lap_final = _contour_laplacian_px(final_poly, float(snake_config.down_ratio)).norm(dim=-1)
    lap_gt = _contour_laplacian_px(i_gt, float(snake_config.down_ratio)).norm(dim=-1)
    lap_disp = _contour_laplacian_px(ret['disp'], float(snake_config.down_ratio)).norm(dim=-1)
    excess_burr = torch.relu(lap_final - lap_gt - float(getattr(cfg, 'grpo_v2_reward_burr_margin_px', 0.50)))
    burr_raw = 0.5 * lap_disp.mean(dim=1) + 0.5 * excess_burr.mean(dim=1)
    burr_penalty = torch.clamp(
        burr_raw / max(float(getattr(cfg, 'grpo_v2_reward_burr_max_px', 1.5)), 1e-6),
        min=0.0,
        max=2.0,
    )
    reward = (
        float(getattr(cfg, 'grpo_v2_reward_abs_weight', 0.0)) * final_score
        + float(getattr(cfg, 'grpo_v2_reward_delta_weight', 1.0)) * (final_score - init_score)
        - float(getattr(cfg, 'grpo_v2_reward_burr_weight', 0.03)) * burr_penalty
    )
    quality = final_score - det_score
    return final_score, reward, quality, burr_raw


def tensor_image_to_bgr(inp):
    arr = inp.detach().float().cpu().numpy()
    if arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    img = (arr * 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[-1] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def draw_polys(canvas, polys, color, thickness):
    h, w = canvas.shape[:2]
    for poly in polys:
        pts = np.round(poly).astype(np.int32)
        if pts.shape[0] < 2:
            continue
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        cv2.polylines(canvas, [np.concatenate([pts, pts[:1]], axis=0)], True, color, thickness)


def make_group_figure(batch, output, det_ret, rollout_rets, metrics, image_index, out_png):
    py_ind = output['py_ind'].detach().long()
    mask = py_ind == int(image_index)
    if not bool(mask.any().item()):
        mask = torch.ones_like(py_ind, dtype=torch.bool)
    scale = float(snake_config.down_ratio)
    init_np = output['i_it_py'][mask].detach().cpu().numpy() * scale
    gt_np = _align_gt(output['i_it_py'], output['i_gt_py'])[mask].detach().cpu().numpy() * scale
    det_np = det_ret['py'][mask].detach().cpu().numpy() * scale
    finals = [r['py'][mask].detach().cpu().numpy() * scale for r in rollout_rets]

    base = tensor_image_to_bgr(batch['inp'][int(image_index)].detach().cpu())
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    base = cv2.addWeighted(base, 0.50, np.zeros_like(base), 0.50, 0)
    h, w = base.shape[:2]

    q = np.array([float(m['quality_mean']) for m in metrics], dtype=np.float32)
    order = list(np.argsort(-q))
    k_show = min(len(order), 16)
    order = order[:k_show]
    best_idx = int(np.argmax(q))
    worst_idx = int(np.argmin(q))

    panels = []
    overview = base.copy()
    draw_polys(overview, gt_np, (255, 80, 80), 3)
    draw_polys(overview, init_np, (0, 220, 255), 2)
    draw_polys(overview, det_np, (0, 255, 0), 2)
    cv2.rectangle(overview, (2, 2), (min(w - 2, 560), 58), (0, 0, 0), -1)
    cv2.putText(overview, 'overview | blue=GT yellow=init green=det', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overview, f'contours={int(mask.sum().item())} rollouts={len(rollout_rets)}', (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    panels.append(overview)

    for ri in order:
        canvas = base.copy()
        draw_polys(canvas, gt_np, (255, 80, 80), 3)
        draw_polys(canvas, init_np, (0, 220, 255), 1)
        draw_polys(canvas, det_np, (0, 150, 0), 1)
        draw_polys(canvas, finals[ri], (255, 255, 255), 3)
        border = (80, 220, 80) if ri == best_idx else (40, 40, 255) if ri == worst_idx else (100, 100, 100)
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), border, 4)
        label = (
            f"k={ri:02d} R={metrics[ri]['reward_mean']:+.4f} "
            f"Q={metrics[ri]['quality_mean']:+.4f} F={metrics[ri]['final_score_mean']:.4f}"
        )
        cv2.rectangle(canvas, (2, 2), (min(w - 2, 520), 30), (0, 0, 0), -1)
        cv2.putText(canvas, label, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(canvas)

    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    blank = np.zeros_like(base)
    while len(panels) < rows * cols:
        panels.append(blank.copy())
    grid_rows = [np.concatenate(panels[r * cols:(r + 1) * cols], axis=1) for r in range(rows)]
    grid = np.concatenate(grid_rows, axis=0)
    legend = np.zeros((42, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        legend,
        'V17 GRPO group sampling | white=sample final, green border=best quality, red border=worst',
        (8, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    out = np.concatenate([legend, grid], axis=0)
    cv2.imwrite(str(out_png), out)


def main():
    args = parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        cfg.gpus = [int(str(args.gpu).split(',')[0])]
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = resolve_path(args.ckpt) if args.ckpt else default_ckpt()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inner = load_inner(ckpt, device)
    gcn = inner.gcn
    loader = make_data_loader(cfg, is_train=True, is_distributed=False)
    it = iter(loader)
    batch = None
    for _ in range(max(int(args.batch_index), 0) + 1):
        batch = next(it)
    _move_batch(batch, device=str(device))
    output = manual_gt_init_context(inner, batch)
    if output['i_it_py'].numel() == 0:
        raise RuntimeError('Selected batch has no valid contours.')

    window_range = getattr(cfg, 'grpo_v2_window_range', [4, 7])
    if args.window_start is not None and args.window_end is not None:
        window_range = [args.window_start, args.window_end]
    window_range = tuple(int(x) for x in window_range)

    det_ret = deterministic_policy(gcn, output, args.rollout_steps)
    det_score, _, _, _ = compute_scores(output, det_ret, torch.zeros(output['i_it_py'].shape[0], device=device))
    rollout_rets = []
    metrics = []
    for ki in range(int(args.k)):
        ret = sample_rollout(gcn, output, args.action_std, args.rollout_steps, args.window_size, window_range)
        final_score, reward, quality, burr_raw = compute_scores(output, ret, det_score)
        rollout_rets.append(ret)
        metrics.append({
            'k': ki,
            'final_score_mean': float(final_score.mean().item()),
            'reward_mean': float(reward.mean().item()),
            'quality_mean': float(quality.mean().item()),
            'burr_raw_px_mean': float(burr_raw.mean().item()),
        })

    png_path = out_dir / f'{args.save_prefix}_batch{args.batch_index}_img{args.image_index}.png'
    json_path = out_dir / f'{args.save_prefix}_batch{args.batch_index}_img{args.image_index}.json'
    make_group_figure(batch, output, det_ret, rollout_rets, metrics, args.image_index, png_path)
    meta = {
        'cfg_file': os.environ.get('CFG_FILE'),
        'ckpt': str(ckpt),
        'batch_index': int(args.batch_index),
        'image_index': int(args.image_index),
        'k': int(args.k),
        'action_std': float(args.action_std),
        'rollout_steps': int(args.rollout_steps),
        'window_size': int(args.window_size),
        'window_range': list(window_range),
        'det_score_mean': float(det_score.mean().item()),
        'rollouts': metrics,
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    best = max(metrics, key=lambda x: x['quality_mean'])
    worst = min(metrics, key=lambda x: x['quality_mean'])
    print(f'saved_png {png_path}')
    print(f'saved_json {json_path}')
    print(f"det_score_mean {meta['det_score_mean']:.6f}")
    print(f"best k={best['k']} quality={best['quality_mean']:+.6f} reward={best['reward_mean']:+.6f} final={best['final_score_mean']:.6f}")
    print(f"worst k={worst['k']} quality={worst['quality_mean']:+.6f} reward={worst['reward_mean']:+.6f} final={worst['final_score_mean']:.6f}")


if __name__ == '__main__':
    main()
