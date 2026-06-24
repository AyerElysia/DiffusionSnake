#!/usr/bin/env python3
"""Diagnose whether curvature-detail reward is saturated on eval contours."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT_DIR))

_DEFAULT_CFG = _ROOT_DIR / 'configs' / '1232_final_v5_geom8_baseline_bs6_gpu2.yaml'
_DEFAULT_CKPT = (
    _ROOT_DIR
    / 'data'
    / 'outputs'
    / '1232_final_v5_geom8_baseline_bs6_gpu2'
    / 'checkpoints'
    / 'best_iou.pt'
)
_DEFAULT_OUT_DIR = _ROOT_DIR / 'data' / 'analysis' / 'v5_detail_saturation'


def _build_pre_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Offline detail-reward saturation diagnostic for RL V5 geom-action.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--cfg', '--cfg_file', dest='cfg_file', default=str(_DEFAULT_CFG), type=str)
    parser.add_argument('--ckpt', default=str(_DEFAULT_CKPT), type=str)
    parser.add_argument('--gpu', default='', type=str, help='Optional CUDA_VISIBLE_DEVICES / cfg.gpus override.')
    parser.add_argument('--max_batches', default=0, type=int, help='Limit eval batches; <=0 means all.')
    parser.add_argument('--out_dir', default=str(_DEFAULT_OUT_DIR), type=str)
    parser.add_argument('--high_iou_thresh', default=0.88, type=float)
    parser.add_argument('--seed', default=20260617, type=int)
    return parser


_pre_parser = _build_pre_parser()
_pre_args, _remaining_argv = _pre_parser.parse_known_args()

if _pre_args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_pre_args.gpu)
    os.environ['GRPO_V2_GPU'] = str(_pre_args.gpu)
os.environ['CFG_FILE'] = str(_pre_args.cfg_file)
sys.argv = [
    sys.argv[0],
    '--cfg_file',
    str(_pre_args.cfg_file),
    '--ckpt',
    str(_pre_args.ckpt),
] + _remaining_argv

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lib.config import cfg  # noqa: E402
from lib.datasets import make_data_loader  # noqa: E402
from lib.networks import make_network  # noqa: E402
from lib.train.rewards.curvature_detail_reward import compute_curvature_detail_score  # noqa: E402
from lib.train.rewards.region_reward import compute_region_score  # noqa: E402
from lib.train.trainers import make_trainer  # noqa: E402
from lib.utils.snake import snake_config, snake_gcn_utils  # noqa: E402
from grpo_train_v5_geom_action import (  # noqa: E402
    _adapt_state_dict,
    _align_gt,
    _extract_state_dict,
    _flatten_valid_polys,
    _make_py_ind,
    _outer_action_mean,
)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _move_batch(batch, device):
    for key in list(batch.keys()):
        if key == 'meta':
            continue
        value = batch[key]
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def _as_project_path(path_like) -> Path:
    path = Path(str(path_like)).expanduser()
    return path if path.is_absolute() else (_ROOT_DIR / path)


def _cv(name, default):
    v4_cfg = getattr(cfg, 'rl_v4', None)
    env_name = f'RL_V4_{name.upper()}'
    if env_name in os.environ:
        return _parse_env_value(os.environ[env_name], default)
    if v4_cfg is not None and name in v4_cfg:
        return v4_cfg[name]
    return getattr(cfg, f'rl_v4_{name}', default)


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
        for part in parts:
            part = part.strip()
            if not part:
                continue
            try:
                vals.append(int(part))
            except ValueError:
                try:
                    vals.append(float(part))
                except ValueError:
                    vals.append(part)
        return type(default)(vals)
    return raw


def _load_model(ckpt_path: Path, device: torch.device):
    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True
    cfg.use_flow_matching = True
    cfg.use_grpo = True

    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    net_for_load = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network

    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    raw_ckpt = torch.load(str(ckpt_path), map_location='cpu')
    sd = _adapt_state_dict(net_for_load, _extract_state_dict(raw_ckpt))
    missing, unexpected = net_for_load.load_state_dict(sd, strict=False)
    total = len(list(net_for_load.state_dict().keys()))
    load_ratio = 100.0 * (total - len(missing)) / max(total, 1)
    print(
        f'[*] Loaded checkpoint: {ckpt_path} | load_ratio={load_ratio:.2f}% '
        f'missing={len(missing)} unexpected={len(unexpected)}'
    )

    net_for_load.to(device).eval()
    inner = net_for_load.net if hasattr(net_for_load, 'net') else net_for_load
    return inner, inner.gcn, {'load_ratio': load_ratio, 'missing': len(missing), 'unexpected': len(unexpected)}


@torch.no_grad()
def _manual_context(inner, batch):
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
        'i_gt_py': _align_gt(i_init, i_gt).detach(),
        'py_ind': py_ind.detach(),
        'image_hw': (int(batch['inp'].shape[-2]), int(batch['inp'].shape[-1])),
    }


@torch.no_grad()
def _deterministic_three_step(output, flow, fractions, ode_steps: int):
    current = output['i_it_py'].detach()
    total_disp = torch.zeros_like(current)
    polys = [current.detach()]
    for frac in fractions:
        c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
        action = _outer_action_mean(
            flow,
            output['cnn_feature'],
            current,
            c_cur,
            output['py_ind'],
            float(frac),
            int(ode_steps),
        )
        current = (current + action).detach()
        total_disp = total_disp + action.detach()
        polys.append(current.detach())
    return {'disp': total_disp, 'py': output['i_it_py'] + total_disp, 'polys': polys}


def _detail_kwargs():
    return {
        'corner_dist_max_px': float(_cv('reward_detail_corner_dist_max_px', 6.0)),
        'corner_dist_quantile': float(_cv('reward_detail_corner_dist_quantile', 95.0)),
        'corner_dist_quantile_weight': float(_cv('reward_detail_corner_dist_quantile_weight', 0.7)),
        'curvature_max_px': float(_cv('reward_detail_curvature_max_px', 4.0)),
        'burr_margin_px': float(_cv('reward_burr_margin_px', 0.5)),
        'burr_max_px': float(_cv('reward_burr_max_px', 1.5)),
        'burr_quantile': float(_cv('reward_burr_quantile', 95.0)),
        'local_band_radius_px': int(_cv('reward_detail_local_band_radius_px', 2)),
        'area_max_frac': float(_cv('reward_detail_area_max_frac', 0.15)),
        'w_corner_dist': float(_cv('reward_detail_w_corner_dist', 0.35)),
        'w_curv_match': float(_cv('reward_detail_w_curv_match', 0.20)),
        'w_local_biou': float(_cv('reward_detail_w_local_biou', 0.10)),
        'w_burr': float(_cv('reward_detail_w_burr', 0.07)),
        'w_area': float(_cv('reward_detail_w_area', 0.03)),
    }


def _rollout_schedule():
    outer_steps = int(_cv('outer_steps', 3))
    fractions = [float(x) for x in list(_cv('fractions', [0.3333, 0.5, 1.0]))]
    if len(fractions) < outer_steps:
        fractions = fractions + [1.0] * (outer_steps - len(fractions))
    fractions = fractions[:outer_steps]
    ode_steps = int(_cv('ode_steps', getattr(cfg, 'iterative_ode_steps', getattr(cfg, 'flow_ode_steps', 10))))
    if ode_steps <= 0:
        ode_steps = int(getattr(cfg, 'flow_ode_steps', 10))
    return fractions, ode_steps


def _stats(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {'mean': None, 'median': None, 'p10': None, 'p90': None}
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'p10': float(np.percentile(arr, 10)),
        'p90': float(np.percentile(arr, 90)),
    }


def _mean_or_none(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _fmt(value):
    if value is None:
        return 'None'
    return f'{float(value):.4f}'


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def main():
    args = _pre_args
    if args.gpu:
        cfg.gpus = [int(args.gpu)]
        print(f'[*] Override GPU -> {args.gpu}')

    _set_seed(int(args.seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = _as_project_path(args.ckpt)
    out_dir = _as_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fractions, ode_steps = _rollout_schedule()
    detail_kwargs = _detail_kwargs()
    down_ratio = float(snake_config.down_ratio)

    inner, flow, load_info = _load_model(ckpt_path, device)
    loader = make_data_loader(cfg, is_train=False, is_distributed=False)
    print(
        f'[*] Eval dataset: {cfg.test.dataset} | device={device} | '
        f'max_batches={args.max_batches or "all"}'
    )
    print(f'[*] Deterministic rollout fractions={fractions} ode_steps={ode_steps}')

    ious = []
    detail_values = {
        'corner_dist': [],
        'curv_match': [],
        'local_biou': [],
        'burr_penalty': [],
        'area_penalty': [],
    }
    processed_batches = 0
    skipped_empty_batches = 0

    for batch_idx, batch in enumerate(loader):
        if int(args.max_batches) > 0 and batch_idx >= int(args.max_batches):
            break
        batch = _move_batch(batch, device)
        output = _manual_context(inner, batch)
        if output['i_it_py'].numel() == 0 or output['i_it_py'].size(0) == 0:
            skipped_empty_batches += 1
            processed_batches += 1
            continue

        det = _deterministic_three_step(output, flow, fractions, ode_steps)
        pred = output['i_it_py'] + det['disp']
        gt = output['i_gt_py']
        hw = output['image_hw']

        iou_t = compute_region_score(
            pred,
            gt,
            H=hw[0],
            W=hw[1],
            w_boundary=0,
            w_dice=0,
            w_iou=1,
            w_dist=0,
            coord_scale=down_ratio,
        )
        _, comps = compute_curvature_detail_score(
            pred,
            gt,
            H=hw[0],
            W=hw[1],
            coord_scale=down_ratio,
            return_components=True,
            **detail_kwargs,
        )

        ious.append(_to_numpy(iou_t))
        for key in detail_values:
            detail_values[key].append(_to_numpy(comps[key]))

        processed_batches += 1
        if processed_batches % 10 == 0:
            n_contours = sum(int(x.size) for x in ious)
            print(f'[*] processed batches={processed_batches} contours={n_contours}')

    if ious:
        iou_all = np.concatenate(ious, axis=0)
        detail_all = {key: np.concatenate(vals, axis=0) for key, vals in detail_values.items()}
    else:
        iou_all = np.zeros((0,), dtype=np.float32)
        detail_all = {key: np.zeros((0,), dtype=np.float32) for key in detail_values}
    high_mask = iou_all > float(args.high_iou_thresh)

    detail_stats = {key: _stats(value) for key, value in detail_all.items()}
    iou_stats = _stats(iou_all)
    high_quality = {
        'iou_threshold': float(args.high_iou_thresh),
        'count': int(high_mask.sum()),
        'fraction': float(high_mask.mean()) if high_mask.size else 0.0,
        'mean_iou': _mean_or_none(iou_all[high_mask]),
        'corner_dist_mean': _mean_or_none(detail_all['corner_dist'][high_mask]),
        'curv_match_mean': _mean_or_none(detail_all['curv_match'][high_mask]),
        'local_biou_mean': _mean_or_none(detail_all['local_biou'][high_mask]),
        'burr_penalty_mean': _mean_or_none(detail_all['burr_penalty'][high_mask]),
        'area_penalty_mean': _mean_or_none(detail_all['area_penalty'][high_mask]),
    }
    correlations = {
        'iou_vs_corner_dist': _corr(iou_all, detail_all['corner_dist']),
        'iou_vs_curv_match': _corr(iou_all, detail_all['curv_match']),
    }

    verdict = 'no_valid_contours' if iou_all.size == 0 else 'insufficient_high_iou_contours'
    cd = high_quality['corner_dist_mean']
    cm = high_quality['curv_match_mean']
    if cd is not None and cm is not None:
        if cd < 0.85 or cm < 0.85:
            verdict = 'detail_has_clear_headroom'
        elif cd >= 0.95 and cm >= 0.95:
            verdict = 'detail_also_saturated'
        else:
            verdict = 'detail_partially_unsaturated'

    summary = {
        'cfg_file': str(args.cfg_file),
        'checkpoint': str(ckpt_path),
        'device': str(device),
        'processed_batches': int(processed_batches),
        'skipped_empty_batches': int(skipped_empty_batches),
        'contours': int(iou_all.size),
        'rollout': {'fractions': fractions, 'ode_steps': int(ode_steps)},
        'load_info': load_info,
        'region_iou': iou_stats,
        'detail_components': detail_stats,
        'high_quality_subset': high_quality,
        'correlations': correlations,
        'verdict': verdict,
        'detail_kwargs': detail_kwargs,
    }

    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n=== Detail Saturation Summary ===')
    print(f'contours={summary["contours"]} batches={processed_batches} skipped_empty={skipped_empty_batches}')
    print(
        'region_iou: '
        f'mean={_fmt(iou_stats["mean"])} median={_fmt(iou_stats["median"])} '
        f'p10={_fmt(iou_stats["p10"])} p90={_fmt(iou_stats["p90"])}'
    )
    for key in ('corner_dist', 'curv_match', 'local_biou', 'burr_penalty', 'area_penalty'):
        s = detail_stats[key]
        print(
            f'{key}: mean={_fmt(s["mean"])} median={_fmt(s["median"])} '
            f'p10={_fmt(s["p10"])} p90={_fmt(s["p90"])}'
        )
    print(
        f'high_iou>{args.high_iou_thresh:.2f}: count={high_quality["count"]} '
        f'corner_dist_mean={high_quality["corner_dist_mean"]} '
        f'curv_match_mean={high_quality["curv_match_mean"]}'
    )
    print(
        'corr: '
        f'iou_vs_corner_dist={correlations["iou_vs_corner_dist"]} '
        f'iou_vs_curv_match={correlations["iou_vs_curv_match"]}'
    )
    print(f'verdict={verdict}')
    print(f'[*] Wrote {summary_path}')


if __name__ == '__main__':
    main()
