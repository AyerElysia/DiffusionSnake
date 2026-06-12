#!/usr/bin/env python3
"""
Evaluate ranker-based best-of-K contour selection on the test split.

Environment:
  CFG_FILE, CKPT, EVAL_GPU
  RANKER_PATH default data/stats/contour_ranker_best.pt
  SEEDS       comma-separated ints, default 101,202,303,404,505,606,707,808
  MAX_SAMPLES empty means full dataset
  OUT_DIR     default visual/ranker_selection_eval
"""

import datetime
import json
import os
import sys

if str(os.environ.get('EVAL_GPU', '')).strip():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(os.environ.get('EVAL_GPU', '')).strip()

import numpy as np
import torch

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
from train_contour_ranker import bbox_for_poly
from train_contour_ranker import build_ranker_inputs
from train_contour_ranker import load_contour_ranker


STRATEGIES = ('random', 'consensus', 'edge', 'consensus_edge', 'ranker', 'oracle')
DEFAULT_SEEDS = '101,202,303,404,505,606,707,808'
DEFAULT_RANKER_PATH = os.path.join('data', 'stats', 'contour_ranker_best.pt')


def resolve_path(path, default_value):
    value = str(path or '').strip()
    if not value:
        value = default_value
    if os.path.isabs(value):
        return value
    return os.path.join(_REPO_ROOT, value)


def resolve_out_dir(value):
    value = str(value or '').strip()
    if not value:
        value = os.path.join('visual', 'ranker_selection_eval')
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


def poly_to_array(poly, fallback):
    if poly is None or len(poly) == 0:
        return np.zeros_like(fallback, dtype=np.float32)
    return np.asarray(poly, dtype=np.float32).reshape(-1, 2)


def build_ranker_candidate_arrays(candidates, gt_poly, height, width):
    poly_arrays = []
    bboxes = []
    consensus = []
    edge = []
    fallback = np.asarray(gt_poly, dtype=np.float32).reshape(-1, 2)
    for cand in candidates:
        poly = poly_to_array(cand.get('poly'), fallback)
        poly_arrays.append(poly)
        consensus.append(float(cand.get('consensus', 0.0)))
        edge.append(float(cand.get('edge', 0.0)))
        bboxes.append(bbox_for_poly(poly, height, width))
    return (
        np.stack(poly_arrays, axis=0).astype(np.float32),
        np.asarray(consensus, dtype=np.float32),
        np.asarray(edge, dtype=np.float32),
        np.stack(bboxes, axis=0).astype(np.float32),
    )


def ranker_select_index(ranker, ranker_ckpt, device, candidates, gt_poly, image, height, width):
    poly_group, consensus, edge, bboxes = build_ranker_candidate_arrays(
        candidates,
        gt_poly,
        height,
        width,
    )
    scalar_mean = ranker_ckpt.get('scalar_mean', None)
    scalar_std = ranker_ckpt.get('scalar_std', None)
    poly_flat, scalars, patches = build_ranker_inputs(
        poly_group,
        consensus,
        edge,
        bboxes,
        image,
        scalar_mean=scalar_mean,
        scalar_std=scalar_std,
    )
    with torch.no_grad():
        poly_t = torch.from_numpy(poly_flat[None]).to(device)
        scalars_t = torch.from_numpy(scalars[None]).to(device)
        patches_t = torch.from_numpy(patches[None]).to(device)
        scores = ranker(poly_t, scalars_t, patches_t)
        return int(torch.argmax(scores, dim=1).detach().cpu().item())


def select_candidates(candidates, ranker_idx):
    gt_ious = np.asarray([c['gt_iou'] for c in candidates], dtype=np.float32)
    consensus = np.asarray([c['consensus'] for c in candidates], dtype=np.float32)
    edge = np.asarray([c['edge'] for c in candidates], dtype=np.float32)
    combo = seed_verify.zscore(consensus) + seed_verify.zscore(edge)
    selection_indices = {
        'random': 0,
        'consensus': int(np.argmax(consensus)),
        'edge': int(np.argmax(edge)),
        'consensus_edge': int(np.argmax(combo)),
        'ranker': int(ranker_idx),
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


def process_sample(model, device, dataset, collator, index, seeds, ode_steps,
                   stats, ranker, ranker_ckpt, ranker_device):
    batch = collator([dataset[index]])
    seed_results = []
    for seed in seeds:
        eval_mod.set_eval_seed(seed)
        seed_results.append(seed_verify.infer_sample_raw(
            model,
            device,
            seed_verify.clone_batch(batch),
            ode_steps=ode_steps,
        ))

    base = seed_results[0]
    img = base['img']
    height, width = img.shape[:2]
    edge_map = seed_verify.compute_edge_map(img)
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
        candidates = seed_verify.build_candidates_for_contour(
            seed_results,
            gt_idx,
            seeds,
            gt_poly,
            edge_map,
            height,
            width,
        )
        ranker_idx = ranker_select_index(
            ranker,
            ranker_ckpt,
            ranker_device,
            candidates,
            gt_poly,
            img,
            height,
            width,
        )
        selections = select_candidates(candidates, ranker_idx)
        update_strategy_stats(sample_stats, selections)
        label = int(gt_labels[gt_idx]) if gt_labels is not None and gt_idx < len(gt_labels) else None
        sample_record['contours'].append({
            'gt_index': int(gt_idx),
            'gt_label': label,
            'gt_poly': seed_verify.poly_to_list(gt_poly),
            'candidates': candidates,
            'selections': selections,
        })

    merge_strategy_stats(stats, sample_stats)
    return sample_record


def print_summary(summary):
    strategy_summary = summary['strategy_summary']
    print('\n' + '=' * 88)
    print("Saved results: {}".format(summary['results_path']))
    print("Evaluated samples: {} / {}".format(summary['evaluated_samples'], summary['dataset_size']))
    print("Failed samples:    {}".format(summary['failed_samples']))
    print('')
    print("{:<16} {:>12} {:>12} {:>12}".format('strategy', 'mean_iou', 'top1_acc', 'mean_gap'))
    print('-' * 56)
    for name in STRATEGIES:
        row = strategy_summary[name]
        print(
            "{:<16} {:>12.6f} {:>12.6f} {:>12.6f}".format(
                name,
                row['mean_iou'],
                row['top1_oracle_acc'],
                row['mean_oracle_gap'],
            )
        )
    print('=' * 88)


def main():
    eval_mod.apply_gpu_override()
    ablation_mode = eval_mod.apply_ablation_mode()

    seeds = seed_verify.parse_seeds(os.environ.get('SEEDS', DEFAULT_SEEDS))
    ckpt = os.environ.get('CKPT')
    ode_steps = int(os.environ.get('ODE_STEPS', getattr(cfg, 'flow_ode_steps', 10)))
    max_samples_env = os.environ.get('MAX_SAMPLES', '').strip()
    max_samples = int(max_samples_env) if max_samples_env else None
    out_dir = resolve_out_dir(os.environ.get('OUT_DIR', ''))
    ranker_path = resolve_path(os.environ.get('RANKER_PATH', ''), DEFAULT_RANKER_PATH)
    os.makedirs(out_dir, exist_ok=True)

    model, device, ckpt_path = eval_mod.load_model(ckpt)
    ranker_device = device
    ranker, ranker_ckpt = load_contour_ranker(ranker_path, device=ranker_device)
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)

    dataset_size = len(dataset)
    limit = min(dataset_size, max_samples) if max_samples is not None else dataset_size
    stats = new_strategy_stats()
    rows = []
    failed_indices = []

    print('[*] Evaluating ranker selection on {} / {} samples from {}'.format(
        int(limit),
        int(dataset_size),
        cfg.test.dataset,
    ))
    print('[*] Seeds: {}'.format(','.join(str(s) for s in seeds)))
    print('[*] ODE steps: {} | infer_noise_scale={:.6f}'.format(
        int(ode_steps),
        float(getattr(cfg, 'infer_noise_scale', -1.0)),
    ))
    print('[*] Ranker: {}'.format(ranker_path))

    for index in range(limit):
        print('[{}/{}] sample {}'.format(index + 1, int(limit), int(index)))
        try:
            rows.append(process_sample(
                model,
                device,
                dataset,
                collator,
                index,
                seeds,
                ode_steps,
                stats,
                ranker,
                ranker_ckpt,
                ranker_device,
            ))
        except Exception as exc:
            failed_indices.append(int(index))
            rows.append({
                'index': int(index),
                'ok': False,
                'error': str(exc),
            })
            print('  [!] failed: {}'.format(exc))

    results_path = os.path.join(out_dir, 'ranker_selection_results.json')
    summary = {
        'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
        'cfg_file': relpath_or_abs(os.environ.get('CFG_FILE', '')),
        'ckpt': relpath_or_abs(ckpt_path),
        'ranker_path': relpath_or_abs(ranker_path),
        'ranker_epoch': int(ranker_ckpt.get('epoch', -1)) if isinstance(ranker_ckpt, dict) else -1,
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
