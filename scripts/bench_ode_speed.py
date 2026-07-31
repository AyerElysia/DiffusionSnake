#!/usr/bin/env python3
"""
Wall-clock benchmark for the ODE evolution stage (Euler vs AB2, with/without
KV cache). Isolates the sample_disp_iterative call itself (the part AB2 /
KV-cache actually touches) from YOLO detection + feature extraction, and also
reports the full eval_sample() time for context.

Usage mirrors scripts/eval_v37_full_iou.py:
    CFG_FILE=configs/eval_ep3200_euler_s10o20.yaml EVAL_GPU=6 CKPT=... \
      conda run -n snake1 python scripts/bench_ode_speed.py --n 30 --warmup 5
"""
import argparse  # noqa: F401 (kept for parity)
import os
import sys
import time

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _THIS_DIR)

_DEFAULT_CFG = os.path.join(_THIS_DIR, 'configs', 'btcv_diffusion_dit_v3_7_v6c_full_noleak.yaml')
if not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = _DEFAULT_CFG

from lib.config import cfg
from lib.datasets.collate_batch import make_collator
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms

from scripts.eval_v37_full_iou import (
    apply_gpu_override,
    apply_ablation_mode,
    load_model,
    prepare_manual_gt_init_context,
    set_eval_seed,
)


class _Args:
    pass


def main():
    args = _Args()
    args.n = int(os.environ.get('BENCH_N', '30'))
    args.warmup = int(os.environ.get('BENCH_WARMUP', '5'))
    args.seed = int(os.environ.get('EVAL_SEED', '20260504'))

    apply_gpu_override()
    apply_ablation_mode()

    ckpt = os.environ.get('CKPT')
    ode_steps = int(os.environ.get('ODE_STEPS', getattr(cfg, 'flow_ode_steps', 10)))

    model, device, ckpt_path, policy, policy_metadata = load_model(
        ckpt, return_policy=True, eval_ode_steps=ode_steps
    )
    core = model.net if hasattr(model, 'net') else model
    core.eval()

    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)
    set_eval_seed(args.seed)

    total = args.n + args.warmup
    total = min(total, len(dataset))
    indices = list(range(total))

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
        getattr(cfg, 'iterative_ode_steps', getattr(cfg, 'flow_ode_steps', ode_steps))
    )
    if iter_ode_steps <= 0:
        iter_ode_steps = ode_steps

    solver = str(getattr(cfg, 'v3_7_ode_solver', 'euler'))
    kv_disabled = os.environ.get('FLOW_DISABLE_KV_CACHE', '0') == '1'
    print(f'[*] solver={solver} iterative_ode_steps={iter_ode_steps} outer_steps={iter_steps} '
          f'kv_disabled={kv_disabled} device={device}')

    ode_times_ms = []
    full_times_ms = []

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        batch = collator([sample])
        for k, v in batch.items():
            if k == 'locate_feat' or str(k).startswith('locate_feat_'):
                continue
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        with torch.no_grad():
            t_full0 = time.perf_counter()
            manual_context = prepare_manual_gt_init_context(core, batch, device)
            cnn_feature = manual_context['cnn_feature']
            i_it_py = manual_context['i_it_py']
            c_it_py = manual_context['c_it_py']
            py_ind = manual_context['py_ind']

            torch.cuda.synchronize(device)
            t_ode0 = time.perf_counter()
            disp = core.gcn.sample_disp_iterative(
                cnn_feature,
                i_it_py,
                c_it_py,
                py_ind,
                num_iter_steps=iter_steps,
                fractions=fractions,
                ode_steps=iter_ode_steps,
            )
            torch.cuda.synchronize(device)
            t_ode1 = time.perf_counter()
            _ = disp.cpu()
            t_full1 = time.perf_counter()

        if i >= args.warmup:
            ode_times_ms.append((t_ode1 - t_ode0) * 1000.0)
            full_times_ms.append((t_full1 - t_full0) * 1000.0)
        tag = "(warmup)" if i < args.warmup else ""
        print(f'[{i + 1}/{total}] ode={(t_ode1 - t_ode0) * 1000.0:.2f}ms '
              f'full={(t_full1 - t_full0) * 1000.0:.2f}ms {tag}')

    ode_arr = np.array(ode_times_ms)
    full_arr = np.array(full_times_ms)
    print('=' * 60)
    print(f'solver={solver} kv_disabled={kv_disabled} ode_steps={iter_ode_steps} '
          f'outer_steps={iter_steps} n={len(ode_arr)}')
    print(f'ODE-only  : mean={ode_arr.mean():.2f}ms  median={np.median(ode_arr):.2f}ms  '
          f'std={ode_arr.std():.2f}ms  min={ode_arr.min():.2f}ms  max={ode_arr.max():.2f}ms')
    print(f'Full-stage: mean={full_arr.mean():.2f}ms  median={np.median(full_arr):.2f}ms  '
          f'std={full_arr.std():.2f}ms')


if __name__ == '__main__':
    main()
