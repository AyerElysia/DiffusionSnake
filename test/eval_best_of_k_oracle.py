#!/usr/bin/env python3
"""
Test-time ensemble oracle diagnostic.

Answers the critical question: "Does the stochastic rollout space (action_std=0.15)
contain genuinely better contours than the deterministic path?"

Runs 3 inference modes on the full test set (150 samples):
  1. det:       deterministic (action_std=0, k=1)
  2. avg_k:     average of k stochastic rollouts (batched, fast)
  3. best_k:    oracle best of k (GT-scored, upper bound for distillation)

Usage:
  export CFG_FILE=configs/btcv_diffusion_dit_v3_4_fm_full_noleak.yaml
  export EVAL_GPU=4
  conda run -n snake1 python test/eval_best_of_k_oracle.py

Optional env vars:
  K=16          number of stochastic rollouts (default 16)
  ACTION_STD=0.15  noise added per ODE step (default 0.15)
  MAX_SAMPLES=50   limit for quick test (default: full 150)
  CKPT=path/to/checkpoint.pt (default: latest.pt from CFG_FILE stem)
"""

import json
import math
import os
import sys
import datetime

import cv2
import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _THIS_DIR)

_DEFAULT_CFG = os.path.join(_THIS_DIR, 'configs', 'btcv_diffusion_dit_v3_4_fm_full_noleak.yaml')
if not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = _DEFAULT_CFG

eval_gpu = os.environ.get('EVAL_GPU', '').strip()
if eval_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = eval_gpu
    print(f'[*] Using GPU {eval_gpu}')

from lib.config import cfg
from lib.datasets.collate_batch import make_collator
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.utils.snake import snake_config, snake_decode, snake_gcn_utils

import torch.nn.functional as F


# ─── helpers ─────────────────────────────────────────────────────────────────

def poly_to_mask(poly_pts, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly_pts).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def compute_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def poly_iou_sample(pred_polys_px, gt_polys_px, h, w):
    """Mean IoU across matched contour pairs (greedy bipartite)."""
    if len(pred_polys_px) == 0 or len(gt_polys_px) == 0:
        return 0.0
    gt_masks = [poly_to_mask(p, h, w) for p in gt_polys_px]
    pred_masks = [poly_to_mask(p, h, w) for p in pred_polys_px]
    iou_mat = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    for gi, gm in enumerate(gt_masks):
        for pi, pm in enumerate(pred_masks):
            iou_mat[gi, pi] = compute_iou(gm, pm)
    used_g, used_p, ious = set(), set(), []
    for _ in range(min(len(gt_masks), len(pred_masks))):
        gi, pi = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        if iou_mat[gi, pi] < 0:
            break
        ious.append(float(iou_mat[gi, pi]))
        used_g.add(int(gi)); used_p.add(int(pi))
        iou_mat[gi, :] = -1; iou_mat[:, pi] = -1
    return float(np.mean(ious)) if ious else 0.0


# ─── model loading ────────────────────────────────────────────────────────────

def load_model(ckpt_path=None):
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if ckpt_path is None:
        cfg_stem = os.path.splitext(os.path.basename(os.environ.get('CFG_FILE', 'default')))[0]
        ckpt_path = os.path.join(_THIS_DIR, 'data', 'outputs', cfg_stem, 'checkpoints', 'latest.pt')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    print(f'[*] Loading checkpoint: {ckpt_path}')
    ckpt_obj = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt_obj.get('state_dict') or ckpt_obj.get('model') or ckpt_obj.get('net') or ckpt_obj
    from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
    sd = remap_legacy_state_dict(sd)

    wrapper = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    info = wrapper.load_state_dict(sd, strict=False)
    print(f'[✔] Loaded {len(sd) - len(info.missing_keys)} / {len(sd)} keys')
    return trainer.network.to(device).eval(), device, ckpt_path


# ─── per-sample evaluation ────────────────────────────────────────────────────

@torch.no_grad()
def eval_sample_ensemble(model, device, batch, k=16, action_std=0.15, ode_steps=20):
    """Return det_iou, avg_k_iou, best_k_iou (oracle GT-scored) for one sample."""
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.to(device)

    dr = float(snake_config.down_ratio)
    core = model.net if hasattr(model, 'net') else model
    gcn = core.gcn

    # ── build context (mirrors _manual_gt_init_context in grpo_train_v2) ──
    yolo_out = core.yolo(batch['inp'])
    feat_list = yolo_out[1] if isinstance(yolo_out, (list, tuple)) and len(yolo_out) > 1 else None
    feat_p2 = feat_list[0] if isinstance(feat_list, (list, tuple)) else yolo_out
    cnn_feature = core.cnn_proj(feat_p2)
    if getattr(core, 'use_p3_features', False) and hasattr(core, 'cnn_proj_p3'):
        if isinstance(feat_list, (list, tuple)) and len(feat_list) > 1:
            feat_p3 = feat_list[1]
            feat_p3_up = F.interpolate(feat_p3, size=feat_p2.shape[-2:], mode='bilinear', align_corners=False)
            cnn_feature = cnn_feature + core.cnn_proj_p3(feat_p3_up)

    i_init = batch['i_it_py'].view(-1, batch['i_it_py'].shape[-2], 2)
    c_init = snake_gcn_utils.img_poly_to_can_poly(i_init)
    py_ind = torch.zeros(i_init.size(0), dtype=torch.long, device=device)
    gt_flat = batch['i_gt_py'].view(-1, batch['i_gt_py'].shape[-2], 2)

    # GT polygons in pixel space for scoring
    gt_np = (gt_flat.cpu().numpy() * dr).astype(np.float32)  # (N_gt, P, 2)
    h_img = int(batch['inp'].shape[-2] * dr)
    w_img = int(batch['inp'].shape[-1] * dr)

    # ── Helper: run one iterative refinement pass ──
    iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
    fractions = list(getattr(cfg, 'iterative_fractions', []))
    if not fractions:
        fractions = [1.0 / (iter_steps - i) for i in range(iter_steps)]
    iter_ode_steps = int(getattr(cfg, 'iterative_ode_steps',
                                  getattr(cfg, 'iterative_ddim_steps', ode_steps)))
    if iter_ode_steps <= 0:
        iter_ode_steps = ode_steps

    use_iterative = getattr(gcn, 'use_iterative_refinement', False)

    def run_rollout_iterative(std):
        current = i_init.clone()
        total_disp = torch.zeros_like(i_init)
        for fi, frac in enumerate(fractions[:iter_steps]):
            c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
            ret = gcn.sample_with_logprob(
                cnn_feature, current, c_cur, py_ind,
                steps=iter_ode_steps, action_std=std,
            )
            applied = ret['disp'] * float(frac)
            current = (current + applied).detach()
            total_disp = total_disp + applied
        return (i_init + total_disp).cpu().numpy() * dr

    def run_rollout_simple(std):
        ret = gcn.sample_with_logprob(
            cnn_feature, i_init, c_init, py_ind,
            steps=iter_ode_steps, action_std=std,
        )
        return (i_init + ret['disp']).cpu().numpy() * dr

    run_rollout = run_rollout_iterative if use_iterative else run_rollout_simple

    # ── Deterministic baseline ──
    det_pred = run_rollout(0.0)
    det_iou = poly_iou_sample(det_pred, gt_np, h_img, w_img)

    # ── k stochastic rollouts ──
    stoch_preds = []
    stoch_ious = []
    for _ in range(k):
        pred = run_rollout(action_std)
        iou = poly_iou_sample(pred, gt_np, h_img, w_img)
        stoch_preds.append(pred)
        stoch_ious.append(iou)

    # average-of-k contour (average displacements, then score)
    avg_pred = np.mean(stoch_preds, axis=0)
    avg_k_iou = poly_iou_sample(avg_pred, gt_np, h_img, w_img)

    # oracle best-of-k
    best_k_iou = max(stoch_ious)

    return {
        'det_iou': det_iou,
        'avg_k_iou': avg_k_iou,
        'best_k_iou': best_k_iou,
        'stoch_ious': stoch_ious,
    }


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    k = int(os.environ.get('K', '16'))
    action_std = float(os.environ.get('ACTION_STD', '0.15'))
    max_samples_env = os.environ.get('MAX_SAMPLES', '')
    max_samples = int(max_samples_env) if max_samples_env else None
    ckpt = os.environ.get('CKPT', None)
    save_dir = os.environ.get('SAVE_DIR', os.path.join(_THIS_DIR, 'visual', 'ensemble_oracle'))
    os.makedirs(save_dir, exist_ok=True)

    print(f'[*] k={k}  action_std={action_std}  max_samples={max_samples}')
    model, device, ckpt_path = load_model(ckpt)

    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)
    limit = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
    print(f'[*] Evaluating {limit}/{len(dataset)} test samples')

    results = []
    det_ious, avg_k_ious, best_k_ious = [], [], []

    for idx in range(limit):
        batch = collator([dataset[idx]])
        try:
            r = eval_sample_ensemble(model, device, batch, k=k, action_std=action_std)
            results.append({'idx': idx, **r})
            det_ious.append(r['det_iou'])
            avg_k_ious.append(r['avg_k_iou'])
            best_k_ious.append(r['best_k_iou'])
            if idx % 10 == 0 or idx < 5:
                print(f'[{idx+1:3d}/{limit}] det={r["det_iou"]:.4f}  '
                      f'avg{k}={r["avg_k_iou"]:.4f}  '
                      f'best{k}={r["best_k_iou"]:.4f}')
        except Exception as e:
            print(f'[{idx+1:3d}/{limit}] FAILED: {e}')
            results.append({'idx': idx, 'error': str(e)})

    # ── summary ──
    def stats(lst, name):
        a = np.array(lst)
        print(f'{name:15s}: n={len(a):3d}  '
              f'median={np.median(a):.5f}  '
              f'mean={np.mean(a):.5f}  '
              f'std={np.std(a):.5f}')

    print('\n' + '='*60)
    print(f'ENSEMBLE ORACLE RESULTS  (k={k}, action_std={action_std})')
    print('='*60)
    stats(det_ious, 'det (k=1)')
    stats(avg_k_ious, f'avg_k (k={k})')
    stats(best_k_ious, f'best_k oracle (k={k})')
    print()
    if det_ious:
        delta_avg = np.median(avg_k_ious) - np.median(det_ious)
        delta_best = np.median(best_k_ious) - np.median(det_ious)
        print(f'Δ(avg_k - det)  = {delta_avg:+.5f}')
        print(f'Δ(best_k - det) = {delta_best:+.5f}  ← upper bound for distillation')
        print()
        if delta_best < 0.003:
            print('⚠ best-of-k oracle barely improves over det → stochastic exploration')
            print('  space does NOT contain meaningfully better paths at action_std=%.2f' % action_std)
            print('  → Distillation approach is fundamentally limited. Consider alternatives.')
        elif delta_best < 0.010:
            print('✓ Moderate oracle gain. Distillation CAN help but signal is weak.')
            print('  Expect +0.001 to +0.005 from distillation if done correctly.')
        else:
            print('✓✓ Strong oracle gain. Stochastic space has much better contours.')
            print('  Distillation should give clear +0.005+ improvement if converged.')

    # save results
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(save_dir, f'ensemble_oracle_k{k}_std{action_std:.2f}_{ts}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'k': k, 'action_std': action_std, 'ckpt': ckpt_path,
            'n_samples': limit,
            'median_det': float(np.median(det_ious)) if det_ious else None,
            'median_avg_k': float(np.median(avg_k_ious)) if avg_k_ious else None,
            'median_best_k': float(np.median(best_k_ious)) if best_k_ious else None,
            'mean_det': float(np.mean(det_ious)) if det_ious else None,
            'mean_avg_k': float(np.mean(avg_k_ious)) if avg_k_ious else None,
            'mean_best_k': float(np.mean(best_k_ious)) if best_k_ious else None,
            'per_sample': results,
        }, f, indent=2)
    print(f'\n[*] Saved to {out_path}')


if __name__ == '__main__':
    main()
