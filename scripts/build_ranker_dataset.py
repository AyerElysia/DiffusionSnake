#!/usr/bin/env python3
"""
Build offline training data for a learned contour-quality ranker.

Environment:
  CFG_FILE, CKPT, EVAL_GPU
  SEEDS       comma-separated ints, default 101,202,303,404,505,606,707,808
  SPLIT       train or test, default train
  MAX_SAMPLES empty means full split
  OUT_PATH    default data/stats/ranker_dataset_<SPLIT>.npz
"""

import datetime
import json
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import eval_v37_full_iou as eval_mod
import verify_seed_selection as seed_verify
from lib.config import cfg
from lib.datasets.collate_batch import make_collator
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.utils.snake import snake_config


DEFAULT_SEEDS = '101,202,303,404,505,606,707,808'


def parse_split(value):
    split = str(value or 'train').strip().lower()
    if split not in ('train', 'test'):
        raise ValueError(f"SPLIT must be 'train' or 'test', got {value!r}")
    return split


def resolve_out_path(value, split):
    if not str(value or '').strip():
        value = os.path.join('data', 'stats', f'ranker_dataset_{split}.npz')
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


def make_split_dataset(split):
    dataset_name = cfg.train.dataset if split == 'train' else cfg.test.dataset
    dataset = make_dataset(cfg, dataset_name, make_transforms(cfg, False), False)
    return dataset_name, dataset


def candidate_bbox(poly, height, width):
    if poly is None or len(poly) == 0:
        return np.zeros((4,), dtype=np.float32)
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x1 = float(np.clip(pts[:, 0].min(), 0.0, max(float(width), 0.0)))
    y1 = float(np.clip(pts[:, 1].min(), 0.0, max(float(height), 0.0)))
    x2 = float(np.clip(pts[:, 0].max(), 0.0, max(float(width), 0.0)))
    y2 = float(np.clip(pts[:, 1].max(), 0.0, max(float(height), 0.0)))
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def zero_poly_like(reference_poly):
    ref = np.asarray(reference_poly, dtype=np.float32).reshape(-1, 2)
    return np.zeros_like(ref, dtype=np.float32)


def build_ranker_candidates(seed_results, gt_idx, seeds, gt_poly, edge_map, height, width):
    rows = []
    polys = []
    for seed, seed_result in zip(seeds, seed_results):
        matched = seed_result['matched_pred']
        pred_idx = int(matched[gt_idx]) if gt_idx < len(matched) else -1
        valid = 0 <= pred_idx < len(seed_result['pred_polys'])
        pred_poly = seed_result['pred_polys'][pred_idx] if valid else None
        gt_iou = seed_verify.mask_iou_roi(pred_poly, gt_poly, height, width) if valid else 0.0
        rows.append({
            'seed': int(seed),
            'pred_index': int(pred_idx),
            'valid': bool(valid),
            'poly': pred_poly,
            'gt_iou': float(gt_iou),
            'consensus': 0.0,
            'edge': float(seed_verify.edge_score(pred_poly, edge_map)) if valid else 0.0,
            'bbox': candidate_bbox(pred_poly, height, width) if valid else np.zeros((4,), dtype=np.float32),
        })
        polys.append(pred_poly)

    k = len(rows)
    if k > 1:
        pair_iou = np.zeros((k, k), dtype=np.float32)
        for i in range(k):
            for j in range(i + 1, k):
                pair_iou[i, j] = seed_verify.mask_iou_roi(polys[i], polys[j], height, width)
                pair_iou[j, i] = pair_iou[i, j]
        consensus = pair_iou.sum(axis=1) / float(k - 1)
    else:
        consensus = np.zeros((k,), dtype=np.float32)

    for i, val in enumerate(consensus.tolist()):
        rows[i]['consensus'] = float(val)
    return rows


def stack_candidate_group(candidates, gt_poly):
    poly_arrays = []
    for cand in candidates:
        poly = cand['poly']
        if poly is None or len(poly) == 0:
            poly_arrays.append(zero_poly_like(gt_poly))
        else:
            poly_arrays.append(np.asarray(poly, dtype=np.float32).reshape(-1, 2))

    point_counts = {arr.shape[0] for arr in poly_arrays}
    if len(point_counts) != 1:
        raise ValueError(f'Candidate polygons have inconsistent point counts: {sorted(point_counts)}')

    return {
        'poly': np.stack(poly_arrays, axis=0).astype(np.float32),
        'gt_iou': np.asarray([c['gt_iou'] for c in candidates], dtype=np.float32),
        'consensus': np.asarray([c['consensus'] for c in candidates], dtype=np.float32),
        'edge': np.asarray([c['edge'] for c in candidates], dtype=np.float32),
        'local_patch_meta': np.stack([c['bbox'] for c in candidates], axis=0).astype(np.float32),
        'seed': np.asarray([c['seed'] for c in candidates], dtype=np.int64),
        'pred_index': np.asarray([c['pred_index'] for c in candidates], dtype=np.int64),
        'valid': np.asarray([c['valid'] for c in candidates], dtype=np.bool_),
    }


def process_sample(model, device, dataset, collator, sample_row, sample_index, seeds, ode_steps):
    batch = collator([dataset[sample_index]])
    seed_results = []
    for seed in seeds:
        eval_mod.set_eval_seed(seed)
        result = seed_verify.infer_sample_raw(
            model,
            device,
            seed_verify.clone_batch(batch),
            ode_steps=ode_steps,
        )
        seed_results.append(result)

    base = seed_results[0]
    img = base['img']
    height, width = img.shape[:2]
    edge_map = seed_verify.compute_edge_map(img)
    gt_labels = base['gt_labels']

    groups = []
    for gt_idx, gt_poly in enumerate(base['gt_polys']):
        candidates = build_ranker_candidates(
            seed_results,
            gt_idx,
            seeds,
            gt_poly,
            edge_map,
            height,
            width,
        )
        group = stack_candidate_group(candidates, gt_poly)
        group.update({
            'sample_row': int(sample_row),
            'sample_index': int(sample_index),
            'contour_index': int(gt_idx),
            'gt_label': int(gt_labels[gt_idx]) if gt_labels is not None and gt_idx < len(gt_labels) else -1,
            'gt_poly': np.asarray(gt_poly, dtype=np.float32).reshape(-1, 2),
        })
        groups.append(group)

    sample_info = {
        'sample_row': int(sample_row),
        'sample_index': int(sample_index),
        'img_path': base['img_path'],
        'height': int(height),
        'width': int(width),
        'num_gt_contours': int(base['num_gt_contours']),
        'num_pred_contours_by_seed': [int(r['num_pred_contours']) for r in seed_results],
    }
    return sample_info, groups


def empty_group_arrays(num_seeds):
    default_points = int(getattr(snake_config, 'poly_num', 128))
    return {
        'poly': np.zeros((0, num_seeds, default_points, 2), dtype=np.float32),
        'gt_iou': np.zeros((0, num_seeds), dtype=np.float32),
        'consensus': np.zeros((0, num_seeds), dtype=np.float32),
        'edge': np.zeros((0, num_seeds), dtype=np.float32),
        'local_patch_meta': np.zeros((0, num_seeds, 4), dtype=np.float32),
        'seed': np.zeros((0, num_seeds), dtype=np.int64),
        'pred_index': np.zeros((0, num_seeds), dtype=np.int64),
        'valid': np.zeros((0, num_seeds), dtype=np.bool_),
        'sample_row': np.zeros((0,), dtype=np.int64),
        'sample_index': np.zeros((0,), dtype=np.int64),
        'contour_index': np.zeros((0,), dtype=np.int64),
        'gt_label': np.zeros((0,), dtype=np.int64),
        'gt_poly': np.zeros((0, default_points, 2), dtype=np.float32),
    }


def stack_groups(groups, num_seeds):
    if not groups:
        return empty_group_arrays(num_seeds)
    keys = (
        'poly',
        'gt_iou',
        'consensus',
        'edge',
        'local_patch_meta',
        'seed',
        'pred_index',
        'valid',
        'gt_poly',
    )
    arrays = {key: np.stack([g[key] for g in groups], axis=0) for key in keys}
    for key in ('sample_row', 'sample_index', 'contour_index', 'gt_label'):
        arrays[key] = np.asarray([g[key] for g in groups], dtype=np.int64)
    return arrays


def summarize_arrays(arrays, samples, failed_indices):
    gt_iou = arrays['gt_iou']
    if gt_iou.size:
        flat = gt_iou.reshape(-1)
        oracle = gt_iou.max(axis=1)
        random_first = gt_iou[:, 0]
        random_expected = gt_iou.mean(axis=1)
        dist = {
            'min': float(np.min(flat)),
            'p25': float(np.percentile(flat, 25)),
            'p50': float(np.percentile(flat, 50)),
            'p75': float(np.percentile(flat, 75)),
            'max': float(np.max(flat)),
            'mean': float(np.mean(flat)),
            'std': float(np.std(flat)),
        }
        oracle_vs_random_first_gap = float(np.mean(oracle - random_first))
        oracle_vs_random_expected_gap = float(np.mean(oracle - random_expected))
        oracle_mean = float(np.mean(oracle))
        random_first_mean = float(np.mean(random_first))
        random_expected_mean = float(np.mean(random_expected))
    else:
        dist = {k: 0.0 for k in ('min', 'p25', 'p50', 'p75', 'max', 'mean', 'std')}
        oracle_vs_random_first_gap = 0.0
        oracle_vs_random_expected_gap = 0.0
        oracle_mean = 0.0
        random_first_mean = 0.0
        random_expected_mean = 0.0

    return {
        'num_samples': int(len(samples)),
        'failed_samples': int(len(failed_indices)),
        'failed_indices': [int(i) for i in failed_indices],
        'num_contours': int(gt_iou.shape[0]),
        'candidates_per_contour': int(gt_iou.shape[1]) if gt_iou.ndim == 2 else 0,
        'gt_iou_distribution': dist,
        'oracle_mean_iou': oracle_mean,
        'random_first_seed_mean_iou': random_first_mean,
        'random_expected_mean_iou': random_expected_mean,
        'oracle_vs_random_first_seed_gap': oracle_vs_random_first_gap,
        'oracle_vs_random_expected_gap': oracle_vs_random_expected_gap,
    }


def print_summary(out_path, dataset_name, dataset_size, limit, stats):
    dist = stats['gt_iou_distribution']
    print('\n' + '=' * 88)
    print(f'Saved ranker dataset: {out_path}')
    print(f'Dataset: {dataset_name} | samples: {limit} / {dataset_size} | failed: {stats["failed_samples"]}')
    print(f'Total contours: {stats["num_contours"]}')
    print(f'Candidates per contour: {stats["candidates_per_contour"]}')
    print(
        'gt_iou distribution: '
        f'mean={dist["mean"]:.6f} std={dist["std"]:.6f} '
        f'min={dist["min"]:.6f} p25={dist["p25"]:.6f} '
        f'p50={dist["p50"]:.6f} p75={dist["p75"]:.6f} max={dist["max"]:.6f}'
    )
    print(
        'oracle-vs-random gap: '
        f'first_seed={stats["oracle_vs_random_first_seed_gap"]:.6f} '
        f'expected_random={stats["oracle_vs_random_expected_gap"]:.6f}'
    )
    print(
        'mean IoU: '
        f'oracle={stats["oracle_mean_iou"]:.6f} '
        f'first_seed={stats["random_first_seed_mean_iou"]:.6f} '
        f'expected_random={stats["random_expected_mean_iou"]:.6f}'
    )
    print('=' * 88)


def main():
    split = parse_split(os.environ.get('SPLIT', 'train'))
    if split == 'train' and 'SNAKE_DISABLE_AUG' not in os.environ:
        os.environ['SNAKE_DISABLE_AUG'] = '1'

    eval_mod.apply_gpu_override()
    ablation_mode = eval_mod.apply_ablation_mode()

    seeds = seed_verify.parse_seeds(os.environ.get('SEEDS', DEFAULT_SEEDS))
    ckpt = os.environ.get('CKPT')
    ode_steps = int(os.environ.get('ODE_STEPS', getattr(cfg, 'flow_ode_steps', 10)))
    max_samples_env = os.environ.get('MAX_SAMPLES', '').strip()
    max_samples = int(max_samples_env) if max_samples_env else None
    out_path = resolve_out_path(os.environ.get('OUT_PATH', ''), split)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    model, device, ckpt_path = eval_mod.load_model(ckpt)
    dataset_name, dataset = make_split_dataset(split)
    collator = make_collator(cfg)

    dataset_size = len(dataset)
    limit = min(dataset_size, max_samples) if max_samples is not None else dataset_size
    samples = []
    groups = []
    failed_indices = []

    print(f'[*] Building ranker dataset on split={split} dataset={dataset_name}')
    print(f'[*] Samples: {limit} / {dataset_size} | seeds: {",".join(str(s) for s in seeds)}')
    print(f'[*] ODE steps: {ode_steps} | infer_noise_scale={float(getattr(cfg, "infer_noise_scale", -1.0))}')
    if split == 'train':
        print(f'[*] SNAKE_DISABLE_AUG={os.environ.get("SNAKE_DISABLE_AUG", "")}')

    for row, sample_index in enumerate(range(limit)):
        print(f'[{row + 1}/{limit}] sample {sample_index}')
        try:
            sample_info, sample_groups = process_sample(
                model,
                device,
                dataset,
                collator,
                row,
                sample_index,
                seeds,
                ode_steps,
            )
            samples.append(sample_info)
            groups.extend(sample_groups)
        except Exception as exc:
            failed_indices.append(int(sample_index))
            samples.append({
                'sample_row': int(row),
                'sample_index': int(sample_index),
                'img_path': '',
                'height': 0,
                'width': 0,
                'num_gt_contours': 0,
                'error': str(exc),
            })
            print(f'  [!] failed: {exc}')

    arrays = stack_groups(groups, len(seeds))
    stats = summarize_arrays(arrays, samples, failed_indices)
    summary = {
        'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
        'cfg_file': relpath_or_abs(os.environ.get('CFG_FILE', '')),
        'ckpt': relpath_or_abs(ckpt_path),
        'ablation_mode': ablation_mode,
        'split': split,
        'dataset': dataset_name,
        'dataset_size': int(dataset_size),
        'evaluated_limit': int(limit),
        'seeds': [int(s) for s in seeds],
        'ode_steps': int(ode_steps),
        'eval_use_network_forward': bool(getattr(cfg, 'eval_use_network_forward', False)),
        'use_gt_det': bool(getattr(cfg, 'use_gt_det', False)),
        'use_pred_extreme_init_for_inference': bool(getattr(cfg, 'use_pred_extreme_init_for_inference', False)),
        'diffusion_init_source': str(getattr(cfg, 'diffusion_init_source', 'extreme')),
        'infer_noise_scale': float(getattr(cfg, 'infer_noise_scale', -1.0)),
        'snake_disable_aug': os.environ.get('SNAKE_DISABLE_AUG', ''),
        'stats': stats,
    }

    np.savez_compressed(
        out_path,
        poly=arrays['poly'],
        gt_iou=arrays['gt_iou'],
        consensus=arrays['consensus'],
        edge=arrays['edge'],
        local_patch_meta=arrays['local_patch_meta'],
        sample_index=arrays['sample_index'],
        sample_row=arrays['sample_row'],
        contour_index=arrays['contour_index'],
        seed=arrays['seed'],
        pred_index=arrays['pred_index'],
        valid=arrays['valid'],
        gt_label=arrays['gt_label'],
        gt_poly=arrays['gt_poly'],
        sample_indices=np.asarray([s['sample_index'] for s in samples], dtype=np.int64),
        sample_img_paths=np.asarray([s['img_path'] for s in samples], dtype=np.str_),
        sample_heights=np.asarray([s['height'] for s in samples], dtype=np.int64),
        sample_widths=np.asarray([s['width'] for s in samples], dtype=np.int64),
        seeds=np.asarray(seeds, dtype=np.int64),
        failed_indices=np.asarray(failed_indices, dtype=np.int64),
        meta_json=np.asarray(json.dumps(summary, sort_keys=True)),
    )

    print_summary(out_path, dataset_name, dataset_size, limit, stats)


if __name__ == '__main__':
    main()
