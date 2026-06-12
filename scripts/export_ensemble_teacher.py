#!/usr/bin/env python3
"""Export fixed ensemble-mean contour teachers for supervised distillation."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_seeds(value: str):
    seeds = [int(x.strip()) for x in str(value).split(',') if x.strip()]
    if not seeds:
        raise ValueError('At least one seed is required')
    return seeds


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


def export_teacher(args):
    os.environ['CFG_FILE'] = args.cfg_file
    sys.argv = [sys.argv[0], '--cfg_file', args.cfg_file]
    from lib.config import cfg
    from lib.datasets.collate_batch import make_collator
    from lib.datasets.make_dataset import make_dataset
    from lib.datasets.transforms import make_transforms
    from lib.utils.snake import snake_config
    import scripts.eval_v37_full_iou as eval_mod
    import scripts.verify_seed_selection as seed_mod

    cfg.train.dataset = args.dataset
    cfg.train.num_workers = 0
    cfg.test.batch_size = 1
    cfg.infer_avg_samples = 1
    cfg.infer_noise_scale = float(args.noise_scale)
    cfg.eval_manual_gt_init = bool(args.gt_init)
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        cfg.gpus = [args.gpu]

    seeds = parse_seeds(args.seeds)
    eval_mod.set_eval_seed(seeds[0])
    model, device, ckpt_path = eval_mod.load_model(args.ckpt)

    dataset = make_dataset(cfg, args.dataset, make_transforms(cfg, is_train=False), is_train=False)
    collator = make_collator(cfg)
    max_samples = len(dataset) if args.max_samples < 0 else min(args.max_samples, len(dataset))
    out = {
        'meta': {
            'cfg_file': args.cfg_file,
            'ckpt': ckpt_path,
            'dataset': args.dataset,
            'seeds': seeds,
            'noise_scale': float(args.noise_scale),
            'gt_init': bool(args.gt_init),
            'down_ratio': float(snake_config.down_ratio),
            'num_samples': int(max_samples),
        },
        'samples': {},
    }

    failed = 0
    for index in range(max_samples):
        print(f'[{index + 1}/{max_samples}] export teacher', flush=True)
        batch = collator([dataset[index]])
        try:
            seed_results = []
            for seed in seeds:
                eval_mod.set_eval_seed(seed)
                seed_results.append(seed_mod.infer_sample_raw(model, device, clone_batch(batch), ode_steps=args.ode_steps))
            base = seed_results[0]
            num_gt = int(base['num_gt_contours'])
            teacher = []
            valid = []
            for gt_idx in range(num_gt):
                polys = []
                for result in seed_results:
                    pred_idx = int(result['matched_pred'][gt_idx]) if gt_idx < len(result['matched_pred']) else -1
                    if 0 <= pred_idx < len(result['pred_polys']):
                        polys.append(np.asarray(result['pred_polys'][pred_idx], dtype=np.float32))
                if polys:
                    teacher.append(np.stack(polys, axis=0).mean(axis=0))
                    valid.append(1)
                else:
                    teacher.append(np.asarray(base['init_polys'][gt_idx], dtype=np.float32))
                    valid.append(0)

            img_path = str(base['img_path'])
            out['samples'][img_path] = {
                'index': int(index),
                'img_path': img_path,
                'teacher_py': (np.asarray(teacher, dtype=np.float32) / float(snake_config.down_ratio)).tolist(),
                'init_py': (np.asarray(base['init_polys'][:num_gt], dtype=np.float32) / float(snake_config.down_ratio)).tolist(),
                'gt_py': (np.asarray(base['gt_polys'][:num_gt], dtype=np.float32) / float(snake_config.down_ratio)).tolist(),
                'valid': valid,
                'num_gt_contours': num_gt,
            }
        except Exception as exc:
            failed += 1
            print(f'[WARN] failed index={index}: {exc}', flush=True)

    out['meta']['failed_samples'] = int(failed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f)
    print(f'[*] saved teacher: {args.out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-file', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', default='data/teachers/v7_6_train_ensemble8_teacher.json')
    parser.add_argument('--dataset', default='BtcvTrain')
    parser.add_argument('--seeds', default='101,202,303,404,505,606,707,808')
    parser.add_argument('--noise-scale', type=float, default=0.5)
    parser.add_argument('--ode-steps', type=int, default=10)
    parser.add_argument('--max-samples', type=int, default=-1)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--gt-init', action='store_true')
    args = parser.parse_args()
    export_teacher(args)


if __name__ == '__main__':
    main()
