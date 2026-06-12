#!/usr/bin/env python3
"""
Verify no-GT per-contour seed selection for best-of-K inference.

Environment:
  CFG_FILE, CKPT, EVAL_GPU
  SEEDS       comma-separated ints, default 101,202,303,404,505,606,707,808
  MAX_SAMPLES empty means full dataset
  OUT_DIR     default visual/seed_selection_verify
"""

import datetime
import json
import os
import sys

import cv2
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import eval_v37_full_iou as eval_mod
from lib.config import cfg
from lib.datasets.collate_batch import make_collator
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.utils.snake import snake_config, snake_gcn_utils


STRATEGIES = ('random', 'consensus', 'edge', 'consensus_edge', 'oracle')
DEFAULT_SEEDS = '101,202,303,404,505,606,707,808'


def parse_seeds(value):
    seeds = []
    for chunk in str(value).split(','):
        chunk = chunk.strip()
        if chunk:
            seeds.append(int(chunk))
    if not seeds:
        raise ValueError('SEEDS must contain at least one integer seed')
    return seeds


def resolve_out_dir(value):
    value = str(value).strip()
    if not value:
        value = os.path.join('visual', 'seed_selection_verify')
    if os.path.isabs(value):
        return value
    return os.path.join(_REPO_ROOT, value)


def relpath_or_abs(path):
    if not path:
        return ''
    try:
        return os.path.relpath(path, _REPO_ROOT)
    except ValueError:
        return path


def tensor_to_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {k: clone_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_value(v) for v in value)
    return value


def clone_batch(batch):
    return {key: clone_value(value) for key, value in batch.items()}


def batch_to_device(batch, device):
    for key, value in batch.items():
        if key == 'locate_feat' or str(key).startswith('locate_feat_'):
            continue
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def extract_image(batch):
    if 'orig_img' not in batch:
        return np.zeros((512, 512, 3), dtype=np.uint8)
    img_raw = batch['orig_img'][0]
    img = tensor_to_numpy(img_raw)
    return img.astype(np.uint8)


def extract_img_path(batch):
    value = batch.get('img_path', '')
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ''
    if torch.is_tensor(value):
        return str(value.detach().cpu().numpy().tolist())
    return str(value)


def poly_to_list(poly):
    if poly is None:
        return None
    arr = np.asarray(poly, dtype=np.float32)
    return [[float(x), float(y)] for x, y in arr.reshape(-1, 2)]


def clamp_roi_from_polys(poly_a, poly_b, height, width, margin=2):
    pts = []
    if poly_a is not None and len(poly_a) > 0:
        pts.append(np.asarray(poly_a, dtype=np.float32).reshape(-1, 2))
    if poly_b is not None and len(poly_b) > 0:
        pts.append(np.asarray(poly_b, dtype=np.float32).reshape(-1, 2))
    if not pts:
        return None
    all_pts = np.concatenate(pts, axis=0)
    x1 = max(0, int(np.floor(float(all_pts[:, 0].min()))) - margin)
    y1 = max(0, int(np.floor(float(all_pts[:, 1].min()))) - margin)
    x2 = min(int(width), int(np.ceil(float(all_pts[:, 0].max()))) + margin + 1)
    y2 = min(int(height), int(np.ceil(float(all_pts[:, 1].max()))) + margin + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def poly_to_mask_roi(poly, roi):
    x1, y1, x2, y2 = roi
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    if poly is None or len(poly) == 0:
        return mask
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2).copy()
    pts[:, 0] -= x1
    pts[:, 1] -= y1
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


def mask_iou_roi(poly_a, poly_b, height, width):
    if poly_a is None or poly_b is None:
        return 0.0
    roi = clamp_roi_from_polys(poly_a, poly_b, height, width)
    if roi is None:
        return 0.0
    mask_a = poly_to_mask_roi(poly_a, roi)
    mask_b = poly_to_mask_roi(poly_b, roi)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def match_predictions_like_eval(pred_polys, gt_polys, height, width, match_by_iou=False):
    gt_polys = list(gt_polys)
    pred_polys = list(pred_polys)
    if (not match_by_iou) and len(pred_polys) == len(gt_polys):
        matched_pred = list(range(len(gt_polys)))
    else:
        matched_pred = [-1 for _ in gt_polys]
        if pred_polys and gt_polys:
            iou_mat = np.zeros((len(gt_polys), len(pred_polys)), dtype=np.float32)
            for gi, gt_poly in enumerate(gt_polys):
                for pi, pred_poly in enumerate(pred_polys):
                    iou_mat[gi, pi] = mask_iou_roi(pred_poly, gt_poly, height, width)
            used_gt = set()
            used_pred = set()
            while len(used_gt) < len(gt_polys) and len(used_pred) < len(pred_polys):
                gi, pi = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[gi, pi] < 0:
                    break
                matched_pred[int(gi)] = int(pi)
                used_gt.add(int(gi))
                used_pred.add(int(pi))
                iou_mat[gi, :] = -1
                iou_mat[:, pi] = -1

    per_iou = []
    for gi, gt_poly in enumerate(gt_polys):
        pi = matched_pred[gi]
        if pi < 0 or pi >= len(pred_polys):
            per_iou.append(0.0)
        else:
            per_iou.append(mask_iou_roi(pred_polys[pi], gt_poly, height, width))
    return per_iou, matched_pred


def infer_sample_raw(model, device, batch, ode_steps=10):
    """Mirror eval_v37_full_iou.eval_sample, but return raw polygons."""
    batch_to_device(batch, device)

    dr = float(snake_config.down_ratio)
    core = model.net if hasattr(model, 'net') else model
    contour_init_method = str(getattr(cfg, 'contour_init_method', 'octagon')).strip().lower()
    sam_prompt_source = str(getattr(cfg, 'sam_prompt_source', 'yolo_box')).strip().lower()
    detector_backend = str(getattr(cfg, 'detector_backend', 'yolo')).strip().lower()
    use_sam_yolo_init = (
        contour_init_method in ('sam', 'efficient_sam')
        and sam_prompt_source == 'yolo_box'
    )
    use_sam_gt_box_init = contour_init_method in ('sam', 'efficient_sam') and sam_prompt_source == 'gt_box'
    use_full_forward = (
        not bool(getattr(cfg, 'eval_manual_gt_init', False))
    ) and (
        use_sam_yolo_init
        or detector_backend != 'yolo'
        or bool(getattr(cfg, 'eval_use_network_forward', False))
    )

    with torch.no_grad():
        gt_all = batch['i_gt_py']
        if gt_all.numel() == 0:
            raise RuntimeError('No GT polygons in batch')

        batch_size, num_contours, num_points, _ = gt_all.shape
        gt_labels = None
        pred_labels = None

        if use_full_forward:
            net_out = core(batch['inp'], batch)
            if 'py' not in net_out:
                raise RuntimeError('Network output has no predicted polygons')
            pred_polys = net_out['py'].detach().cpu().numpy() * dr
            init_src = net_out.get('i_it_py', torch.zeros_like(net_out['py']))
            init_polys = init_src.detach().cpu().numpy() * dr
            if 'ct_01' in batch:
                keep = batch['ct_01'].bool()
                gt_polys = gt_all[keep].detach().cpu().numpy() * dr
                if 'ct_cls' in batch:
                    gt_labels = batch['ct_cls'][keep].detach().cpu().numpy().astype(np.int32)
            else:
                gt_polys = gt_all.view(-1, num_points, 2).detach().cpu().numpy() * dr
                if 'ct_cls' in batch:
                    gt_labels = batch['ct_cls'].view(-1).detach().cpu().numpy().astype(np.int32)
            if 'detection' in net_out and torch.is_tensor(net_out['detection']):
                det = net_out['detection'].detach()
                if det.numel() > 0 and det.size(-1) >= 6:
                    det_flat = det.reshape(-1, det.size(-1))
                    det_flat = det_flat[det_flat[:, 4] > 1e-4]
                    pred_labels = det_flat[:, 5].detach().cpu().numpy().astype(np.int32)
            match_by_iou = not bool(getattr(cfg, 'use_gt_det', False))
        else:
            if detector_backend == 'yolo':
                if not hasattr(core, 'yolo') or core.yolo is None:
                    raise RuntimeError('Manual YOLO eval path requested but yolo module is missing')
                yolo_out = core.yolo(batch['inp'])
                feat_list = yolo_out[1] if isinstance(yolo_out, (list, tuple)) and len(yolo_out) > 1 else None
                feat_p2 = feat_list[0] if isinstance(feat_list, (list, tuple)) else yolo_out
                if getattr(core, 'use_swin_snake_feature', False):
                    if not hasattr(core, 'swin_snake_feature') or core.swin_snake_feature is None:
                        raise RuntimeError('Swin feature evaluation requested but swin_snake_feature is missing')
                    cnn_feature = core.swin_snake_feature(batch['inp'])
                else:
                    cnn_feature = core.cnn_proj(feat_p2)
                if (
                    (not getattr(core, 'use_swin_snake_feature', False))
                    and getattr(core, 'use_p3_features', False)
                    and hasattr(core, 'cnn_proj_p3')
                    and isinstance(feat_list, (list, tuple))
                    and len(feat_list) > 1
                ):
                    feat_p3 = feat_list[1]
                    feat_p3_up = torch.nn.functional.interpolate(
                        feat_p3, size=feat_p2.shape[-2:], mode='bilinear', align_corners=False
                    )
                    cnn_feature = cnn_feature + core.cnn_proj_p3(feat_p3_up)
            elif (
                detector_backend.startswith('heatmap_')
                or detector_backend.startswith('convnext')
                or detector_backend.startswith('moonvit')
            ):
                if not hasattr(core, 'heatmap_detector') or core.heatmap_detector is None:
                    raise RuntimeError(f'Manual eval path requires heatmap_detector, got detector_backend={detector_backend}')
                cnn_feature, _ct_hm, _wh, mask_logits = core.heatmap_detector(batch['inp'])
                if mask_logits is not None:
                    mask_guidance_alpha = float(getattr(cfg, 'heatmap_mask_guidance_alpha', 0.0))
                    if mask_guidance_alpha > 0.0:
                        mask_guidance = torch.sigmoid(mask_logits).amax(dim=1, keepdim=True)
                        cnn_feature = cnn_feature * (1.0 + mask_guidance_alpha * mask_guidance)
            else:
                raise RuntimeError(f'Manual GT-init eval does not support detector_backend={detector_backend}')

            locate_feat_stats = {}
            if hasattr(core, 'apply_locate_feature_injection'):
                cnn_feature, locate_feat_stats = core.apply_locate_feature_injection(cnn_feature, batch)
            if hasattr(core, 'apply_locate_feature_replacement'):
                cnn_feature, replace_stats = core.apply_locate_feature_replacement(cnn_feature, batch)
                locate_feat_stats.update(replace_stats)

            if use_sam_gt_box_init:
                i_it_py = eval_mod.build_sam_gt_box_init_polys(batch, gt_all, device)
                if i_it_py is None:
                    i_it_py = eval_mod.build_init_polys(batch, gt_all)
            else:
                i_it_py = eval_mod.build_init_polys(batch, gt_all)
            c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)
            if batch_size == 1:
                py_ind = torch.zeros(i_it_py.size(0), dtype=torch.long, device=device)
            else:
                py_ind = torch.cat([
                    torch.full((num_contours,), i, dtype=torch.long, device=device)
                    for i in range(batch_size)
                ])

            if getattr(core.gcn, 'use_iterative_refinement', False):
                iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
                use_rich_infer_schedule = bool(getattr(cfg, 'v4_9_use_rich_infer_schedule', False))
                if use_rich_infer_schedule and hasattr(core.gcn, '_progress_targets_to_residual_fractions'):
                    targets = list(getattr(cfg, 'v4_9_infer_target_fractions', []))
                    if not targets:
                        targets = [0.3333, 0.5, 0.80, 0.97, 1.0]
                    fractions = core.gcn._progress_targets_to_residual_fractions(targets)
                    iter_steps = len(fractions)
                else:
                    fractions = list(getattr(cfg, 'iterative_fractions', []))
                if not fractions:
                    fractions = [1.0 / (iter_steps - i) for i in range(iter_steps)]
                iter_ode_steps = int(
                    getattr(
                        cfg,
                        'iterative_ode_steps',
                        getattr(cfg, 'iterative_ddim_steps', ode_steps),
                    )
                )
                if iter_ode_steps <= 0:
                    iter_ode_steps = ode_steps
                if hasattr(core.gcn, 'ode_steps'):
                    disp = core.gcn.sample_disp_iterative(
                        cnn_feature,
                        i_it_py,
                        c_it_py,
                        py_ind,
                        num_iter_steps=iter_steps,
                        fractions=fractions,
                        ode_steps=iter_ode_steps,
                    )
                else:
                    disp = core.gcn.sample_disp_iterative(
                        cnn_feature,
                        i_it_py,
                        c_it_py,
                        py_ind,
                        num_iter_steps=iter_steps,
                        fractions=fractions,
                        ddim_steps=iter_ode_steps,
                    )
            else:
                disp = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind, steps=ode_steps)
            fk = int(getattr(cfg, 'fourier_smooth_k', 0))
            if fk > 0:
                from lib.networks.diffusion.flow_matching_evolution import FlowMatchingEvolution
                disp = FlowMatchingEvolution.fourier_smooth(disp, fk)

            pred_polys = (i_it_py + disp).cpu().numpy() * dr
            gt_polys = gt_all.view(-1, num_points, 2).cpu().numpy() * dr
            init_polys = i_it_py.cpu().numpy() * dr
            if 'ct_cls' in batch:
                gt_labels = batch['ct_cls'].view(-1).detach().cpu().numpy().astype(np.int32)
                pred_labels = gt_labels.copy()
            match_by_iou = False

    img = extract_image(batch)
    height, width = img.shape[:2]
    per_contour_iou, matched_pred = match_predictions_like_eval(
        pred_polys,
        gt_polys,
        height,
        width,
        match_by_iou=match_by_iou,
    )

    return {
        'img': img,
        'img_path': extract_img_path(batch),
        'gt_polys': gt_polys,
        'init_polys': init_polys,
        'pred_polys': pred_polys,
        'gt_labels': gt_labels,
        'pred_labels': pred_labels,
        'per_contour_iou': per_contour_iou,
        'matched_pred': matched_pred,
        'num_pred_contours': int(pred_polys.shape[0]),
        'num_gt_contours': int(gt_polys.shape[0]),
        'match_by_iou': bool(match_by_iou),
    }


def compute_edge_map(img):
    if img.ndim == 2:
        gray = img
    elif img.shape[2] == 1:
        gray = img[:, :, 0]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def bilinear_sample(image, points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.size == 0:
        return np.zeros((0,), dtype=np.float32)
    h, w = image.shape[:2]
    xs = np.clip(points[:, 0], 0, max(w - 1, 0))
    ys = np.clip(points[:, 1], 0, max(h - 1, 0))
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, max(w - 1, 0))
    y1 = np.clip(y0 + 1, 0, max(h - 1, 0))
    wx = xs - x0
    wy = ys - y0

    v00 = image[y0, x0]
    v01 = image[y1, x0]
    v10 = image[y0, x1]
    v11 = image[y1, x1]
    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + (1.0 - wx) * wy * v01
        + wx * (1.0 - wy) * v10
        + wx * wy * v11
    )


def edge_score(poly, edge_map):
    if poly is None or len(poly) == 0:
        return 0.0
    vals = bilinear_sample(edge_map, poly)
    return float(np.mean(vals)) if vals.size else 0.0


def zscore(values):
    arr = np.asarray(values, dtype=np.float32)
    std = float(arr.std())
    if std < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - float(arr.mean())) / std


def build_candidates_for_contour(seed_results, gt_idx, seeds, gt_poly, edge_map, height, width):
    candidates = []
    polys = []
    for seed, seed_result in zip(seeds, seed_results):
        matched = seed_result['matched_pred']
        pred_idx = int(matched[gt_idx]) if gt_idx < len(matched) else -1
        valid = 0 <= pred_idx < len(seed_result['pred_polys'])
        pred_poly = seed_result['pred_polys'][pred_idx] if valid else None
        gt_iou = mask_iou_roi(pred_poly, gt_poly, height, width) if valid else 0.0
        candidates.append({
            'seed': int(seed),
            'pred_index': int(pred_idx),
            'valid': bool(valid),
            'gt_iou': float(gt_iou),
            'consensus': 0.0,
            'edge': float(edge_score(pred_poly, edge_map)) if valid else 0.0,
            'poly': poly_to_list(pred_poly),
        })
        polys.append(pred_poly)

    k = len(candidates)
    if k > 1:
        pair_iou = np.zeros((k, k), dtype=np.float32)
        for i in range(k):
            for j in range(i + 1, k):
                pair_iou[i, j] = mask_iou_roi(polys[i], polys[j], height, width)
                pair_iou[j, i] = pair_iou[i, j]
        consensus = pair_iou.sum(axis=1) / float(k - 1)
    else:
        consensus = np.zeros((k,), dtype=np.float32)

    for idx, val in enumerate(consensus.tolist()):
        candidates[idx]['consensus'] = float(val)
    return candidates


def select_candidates(candidates):
    gt_ious = np.asarray([c['gt_iou'] for c in candidates], dtype=np.float32)
    consensus = np.asarray([c['consensus'] for c in candidates], dtype=np.float32)
    edge = np.asarray([c['edge'] for c in candidates], dtype=np.float32)
    combo = zscore(consensus) + zscore(edge)

    selection_indices = {
        'random': 0,
        'consensus': int(np.argmax(consensus)),
        'edge': int(np.argmax(edge)),
        'consensus_edge': int(np.argmax(combo)),
        'oracle': int(np.argmax(gt_ious)),
    }

    selections = {}
    oracle_iou = float(gt_ious[selection_indices['oracle']])
    for name, idx in selection_indices.items():
        selected_iou = float(gt_ious[idx])
        selections[name] = {
            'candidate_index': int(idx),
            'seed': int(candidates[idx]['seed']),
            'pred_index': int(candidates[idx]['pred_index']),
            'gt_iou': selected_iou,
            'oracle_gap': float(oracle_iou - selected_iou),
            'hit_oracle': bool(idx == selection_indices['oracle']),
        }
    return selections


def new_strategy_stats():
    return {
        name: {
            'ious': [],
            'oracle_gaps': [],
            'oracle_hits': [],
        }
        for name in STRATEGIES
    }


def update_strategy_stats(stats, selections):
    for name in STRATEGIES:
        stats[name]['ious'].append(float(selections[name]['gt_iou']))
        stats[name]['oracle_gaps'].append(float(selections[name]['oracle_gap']))
        stats[name]['oracle_hits'].append(1.0 if selections[name]['hit_oracle'] else 0.0)


def merge_strategy_stats(dst, src):
    for name in STRATEGIES:
        for key in ('ious', 'oracle_gaps', 'oracle_hits'):
            dst[name][key].extend(src[name][key])


def finalize_strategy_stats(stats):
    summary = {}
    for name in STRATEGIES:
        ious = np.asarray(stats[name]['ious'], dtype=np.float32)
        gaps = np.asarray(stats[name]['oracle_gaps'], dtype=np.float32)
        hits = np.asarray(stats[name]['oracle_hits'], dtype=np.float32)
        summary[name] = {
            'num_contours': int(ious.size),
            'mean_iou': float(ious.mean()) if ious.size else 0.0,
            'top1_oracle_acc': float(hits.mean()) if hits.size else 0.0,
            'mean_oracle_gap': float(gaps.mean()) if gaps.size else 0.0,
        }
    return summary


def process_sample(model, device, dataset, collator, index, seeds, ode_steps, stats):
    batch = collator([dataset[index]])
    seed_results = []
    for seed in seeds:
        eval_mod.set_eval_seed(seed)
        seed_results.append(infer_sample_raw(model, device, clone_batch(batch), ode_steps=ode_steps))

    base = seed_results[0]
    img = base['img']
    height, width = img.shape[:2]
    edge_map = compute_edge_map(img)
    gt_labels = base['gt_labels']
    sample_stats = new_strategy_stats()

    sample_record = {
        'index': int(index),
        'ok': True,
        'img_path': base['img_path'],
        'height': int(height),
        'width': int(width),
        'num_gt_contours': int(base['num_gt_contours']),
        'num_pred_contours_by_seed': [
            {
                'seed': int(seed),
                'num_pred_contours': int(result['num_pred_contours']),
            }
            for seed, result in zip(seeds, seed_results)
        ],
        'contours': [],
    }

    for gt_idx, gt_poly in enumerate(base['gt_polys']):
        candidates = build_candidates_for_contour(
            seed_results,
            gt_idx,
            seeds,
            gt_poly,
            edge_map,
            height,
            width,
        )
        selections = select_candidates(candidates)
        update_strategy_stats(sample_stats, selections)
        label = int(gt_labels[gt_idx]) if gt_labels is not None and gt_idx < len(gt_labels) else None
        sample_record['contours'].append({
            'gt_index': int(gt_idx),
            'gt_label': label,
            'gt_poly': poly_to_list(gt_poly),
            'candidates': candidates,
            'selections': selections,
        })

    merge_strategy_stats(stats, sample_stats)
    return sample_record


def print_summary(summary):
    strategy_summary = summary['strategy_summary']
    print('\n' + '=' * 88)
    print(f"Saved results: {summary['results_path']}")
    print(f"Evaluated samples: {summary['evaluated_samples']} / {summary['dataset_size']}")
    print(f"Failed samples:    {summary['failed_samples']}")
    print('')
    print(f"{'strategy':<16} {'mean_iou':>12} {'top1_acc':>12} {'mean_gap':>12}")
    print('-' * 56)
    for name in STRATEGIES:
        row = strategy_summary[name]
        print(
            f"{name:<16} "
            f"{row['mean_iou']:>12.6f} "
            f"{row['top1_oracle_acc']:>12.6f} "
            f"{row['mean_oracle_gap']:>12.6f}"
        )
    print('=' * 88)


def main():
    eval_mod.apply_gpu_override()
    ablation_mode = eval_mod.apply_ablation_mode()

    seeds = parse_seeds(os.environ.get('SEEDS', DEFAULT_SEEDS))
    ckpt = os.environ.get('CKPT')
    ode_steps = int(os.environ.get('ODE_STEPS', getattr(cfg, 'flow_ode_steps', 10)))
    max_samples_env = os.environ.get('MAX_SAMPLES', '').strip()
    max_samples = int(max_samples_env) if max_samples_env else None
    out_dir = resolve_out_dir(os.environ.get('OUT_DIR', ''))
    os.makedirs(out_dir, exist_ok=True)

    model, device, ckpt_path = eval_mod.load_model(ckpt)
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)

    dataset_size = len(dataset)
    limit = min(dataset_size, max_samples) if max_samples is not None else dataset_size
    stats = new_strategy_stats()
    rows = []
    failed_indices = []

    print(f'[*] Verifying seed selection on {limit} / {dataset_size} samples from {cfg.test.dataset}')
    print(f'[*] Seeds: {",".join(str(s) for s in seeds)}')
    print(f'[*] ODE steps: {ode_steps} | infer_noise_scale={float(getattr(cfg, "infer_noise_scale", -1.0))}')

    for index in range(limit):
        print(f'[{index + 1}/{limit}] sample {index}')
        try:
            rows.append(process_sample(model, device, dataset, collator, index, seeds, ode_steps, stats))
        except Exception as exc:
            failed_indices.append(int(index))
            rows.append({
                'index': int(index),
                'ok': False,
                'error': str(exc),
            })
            print(f'  [!] failed: {exc}')

    results_path = os.path.join(out_dir, 'seed_selection_results.json')
    summary = {
        'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
        'cfg_file': relpath_or_abs(os.environ.get('CFG_FILE', '')),
        'ckpt': relpath_or_abs(ckpt_path),
        'ablation_mode': ablation_mode,
        'eval_use_network_forward': bool(getattr(cfg, 'eval_use_network_forward', False)),
        'use_gt_det': bool(getattr(cfg, 'use_gt_det', False)),
        'use_pred_extreme_init_for_inference': bool(getattr(cfg, 'use_pred_extreme_init_for_inference', False)),
        'diffusion_init_source': str(getattr(cfg, 'diffusion_init_source', 'extreme')),
        'infer_noise_scale': float(getattr(cfg, 'infer_noise_scale', -1.0)),
        'ode_steps': int(ode_steps),
        'seeds': [int(s) for s in seeds],
        'dataset': cfg.test.dataset,
        'test_img_path': getattr(cfg.test, 'img_path', ''),
        'dataset_size': int(dataset_size),
        'evaluated_samples': int(limit - len(failed_indices)),
        'failed_samples': int(len(failed_indices)),
        'failed_indices': failed_indices,
        'strategy_summary': finalize_strategy_stats(stats),
        'results_path': results_path,
    }

    payload = {
        'summary': summary,
        'samples': rows,
    }
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print_summary(summary)


if __name__ == '__main__':
    main()
