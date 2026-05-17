#!/usr/bin/env python3
"""
Generate offline pseudo-labels for the training set by running k stochastic
rollouts per image and selecting the best one by GT reward.

This creates fixed "best-of-k" targets for each training sample.  Offline
targets allow much higher LR in subsequent fine-tuning because there is no
moving-target / distribution-shift instability.

Usage:
    CUDA_VISIBLE_DEVICES=4 CFG_FILE=configs/btcv_v3_4_fm_rl_v5b_gpu4.yaml \
    K=8 MAX_SAMPLES=0 \
    CKPT=data/outputs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolom_gpu35_reusemax/checkpoints/latest.pt \
    OUT=data/pseudo_labels/btcv_train_k8.json \
        python scripts/gen_train_pseudo_labels.py

Output JSON:
    {
      "meta": { "k": 8, "ckpt": "...", "n_samples": 720 },
      "samples": [
        {
          "idx": 0,
          "best_iou": 0.9123,
          "det_iou": 0.9005,
          "best_disp": [[...], ...],   # raw displacement, shape [N_contours, 128, 2]
          "contour_scale": [...]        # shape [N_contours], for re-normalization
        }, ...
      ]
    }
"""

import os, sys, json, datetime
import numpy as np
import cv2
import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _THIS_DIR)

_DEFAULT_CFG = os.path.join(_THIS_DIR, 'configs', 'btcv_v3_4_fm_rl_v5b_gpu4.yaml')
if not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = _DEFAULT_CFG

from lib.config import cfg
from lib.datasets.collate_batch import make_collator
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.train.rewards.region_reward import compute_region_score
from lib.utils.snake import snake_config, snake_gcn_utils
import torch.nn.functional as F


# ─── helpers ──────────────────────────────────────────────────────────────────

def poly_to_mask(poly_pts, h, w):
    m = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly_pts).astype(np.int32)
    if pts.ndim == 2 and pts.shape[0] > 2:
        cv2.fillPoly(m, [pts.reshape(-1, 1, 2)], 1)
    return m


def compute_iou_np(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


def poly_iou_sample(pred_polys_px, gt_polys_px, h, w):
    if len(pred_polys_px) == 0 or len(gt_polys_px) == 0:
        return 0.0
    gt_masks = [poly_to_mask(p, h, w) for p in gt_polys_px]
    pred_masks = [poly_to_mask(p, h, w) for p in pred_polys_px]
    iou_mat = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    for gi, gm in enumerate(gt_masks):
        for pi, pm in enumerate(pred_masks):
            iou_mat[gi, pi] = compute_iou_np(gm, pm)
    used_g, used_p, ious = set(), set(), []
    for _ in range(min(len(gt_masks), len(pred_masks))):
        gi, pi = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        if iou_mat[gi, pi] < 0:
            break
        ious.append(float(iou_mat[gi, pi]))
        used_g.add(int(gi)); used_p.add(int(pi))
        iou_mat[gi, :] = -1; iou_mat[:, pi] = -1
    return float(np.mean(ious)) if ious else 0.0


# ─── reward (same weights as V5b training) ────────────────────────────────────
REWARD_W_REGION = 0.2
REWARD_W_DICE   = 0.2
REWARD_W_IOU    = 0.6


def reward_fn_gt(pred_poly_py, gt_poly_py, H, W, coord_scale):
    """GT-based reward matching grpo_train_v2 configuration."""
    score = compute_region_score(
        pred_poly_py, gt_poly_py, H, W,
        w_boundary=REWARD_W_REGION, w_dice=REWARD_W_DICE, w_iou=REWARD_W_IOU,
        coord_scale=coord_scale,
    )
    return score  # (N_contours,)


# ─── model loading ────────────────────────────────────────────────────────────

def load_model(ckpt_path):
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    ckpt_obj = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt_obj.get('state_dict') or ckpt_obj.get('model') or ckpt_obj.get('net') or ckpt_obj
    from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
    sd = remap_legacy_state_dict(sd)

    wrapper = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    info = wrapper.load_state_dict(sd, strict=False)
    print(f'[✔] Loaded {len(sd) - len(info.missing_keys)}/{len(sd)} keys')
    return trainer.network.to(device).eval(), device


# ─── per-sample pseudo-label generation ──────────────────────────────────────

@torch.no_grad()
def gen_sample_pseudo_label(model, device, batch, k=8, ode_steps=10):
    """
    Run k rollouts (fresh x_0 each time), pick the best by GT reward.

    Returns dict with:
      - best_iou:    float, GT IoU of the best rollout
      - det_iou:     float, GT IoU of the deterministic rollout
      - best_disp:   np.ndarray [N_contours, 128, 2] raw pixel-space displacement
      - contour_scale: np.ndarray [N_contours] for normalization at fine-tune time
    """
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.to(device)

    dr = float(snake_config.down_ratio)
    core = model.net if hasattr(model, 'net') else model
    gcn = core.gcn

    # --- extract CNN features once ---
    yolo_out = core.yolo(batch['inp'])
    feat_list = yolo_out[1] if isinstance(yolo_out, (list, tuple)) and len(yolo_out) > 1 else None
    feat_p2 = feat_list[0] if isinstance(feat_list, (list, tuple)) else yolo_out
    cnn_feature = core.cnn_proj(feat_p2)
    if getattr(core, 'use_p3_features', False) and hasattr(core, 'cnn_proj_p3'):
        if isinstance(feat_list, (list, tuple)) and len(feat_list) > 1:
            feat_p3 = feat_list[1]
            feat_p3_up = F.interpolate(feat_p3, size=feat_p2.shape[-2:],
                                        mode='bilinear', align_corners=False)
            cnn_feature = cnn_feature + core.cnn_proj_p3(feat_p3_up)

    i_init = batch['i_it_py'].view(-1, batch['i_it_py'].shape[-2], 2)
    c_init = snake_gcn_utils.img_poly_to_can_poly(i_init)
    py_ind = torch.zeros(i_init.size(0), dtype=torch.long, device=device)
    gt_flat = batch['i_gt_py'].view(-1, batch['i_gt_py'].shape[-2], 2)

    gt_np = (gt_flat.cpu().numpy() * dr).astype(np.float32)
    h_img = int(batch['inp'].shape[-2] * dr)
    w_img = int(batch['inp'].shape[-1] * dr)

    contour_scale = gcn.compute_contour_scale(i_init).detach()  # (N, 1, 1)

    # --- iterative refinement setup (mirrors grpo_train_v2) ---
    iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
    fractions = list(getattr(cfg, 'iterative_fractions', []))
    if not fractions:
        fractions = [1.0 / (iter_steps - i) for i in range(iter_steps)]
    iter_ode_steps = int(getattr(cfg, 'iterative_ode_steps',
                                  getattr(cfg, 'iterative_ddim_steps', ode_steps)))
    if iter_ode_steps <= 0:
        iter_ode_steps = ode_steps

    use_iterative = getattr(gcn, 'use_iterative_refinement', False)

    def run_one_rollout(std=0.0):
        if use_iterative:
            current = i_init.clone()
            total_disp = torch.zeros_like(i_init)
            for frac in fractions[:iter_steps]:
                c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
                ret = gcn.sample_with_logprob(
                    cnn_feature, current, c_cur, py_ind,
                    steps=iter_ode_steps, action_std=std,
                )
                applied = ret['disp'] * float(frac)
                current = (current + applied).detach()
                total_disp = total_disp + applied
            return total_disp  # raw displacement (polygon coords)
        else:
            ret = gcn.sample_with_logprob(
                cnn_feature, i_init, c_init, py_ind,
                steps=iter_ode_steps, action_std=std,
            )
            return ret['disp']

    # --- deterministic baseline ---
    det_disp = run_one_rollout(0.0)
    det_pred_px = ((i_init + det_disp).cpu().numpy() * dr).astype(np.float32)
    det_iou = poly_iou_sample(det_pred_px, gt_np, h_img, w_img)

    # --- k stochastic rollouts (std=0, fresh x_0 each time via gcn.sample_with_logprob) ---
    best_iou = -1.0
    best_disp = det_disp.clone()  # fallback to det if nothing beats it

    for _ in range(k):
        disp = run_one_rollout(0.0)
        pred_px = ((i_init + disp).cpu().numpy() * dr).astype(np.float32)
        iou = poly_iou_sample(pred_px, gt_np, h_img, w_img)
        if iou > best_iou:
            best_iou = iou
            best_disp = disp.clone()

    return {
        'det_iou': float(det_iou),
        'best_iou': float(best_iou),
        'best_disp': best_disp.cpu().float().numpy().tolist(),
        'contour_scale': contour_scale.squeeze(-1).squeeze(-1).cpu().float().numpy().tolist(),
    }


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    k = int(os.environ.get('K', '8'))
    ckpt_rel = os.environ.get('CKPT',
        'data/outputs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolom_gpu35_reusemax/checkpoints/latest.pt')
    out_rel = os.environ.get('OUT', 'data/pseudo_labels/btcv_train_k8.json')
    max_samples = int(os.environ.get('MAX_SAMPLES', '0'))
    resume_path = os.environ.get('RESUME', '')

    out_path = os.path.join(_THIS_DIR, out_rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Check for resume
    done_idxs = set()
    existing_samples = []
    if resume_path and os.path.exists(resume_path):
        with open(resume_path) as f:
            prev = json.load(f)
        existing_samples = prev.get('samples', [])
        done_idxs = {s['idx'] for s in existing_samples}
        print(f'[*] Resuming from {resume_path}: {len(done_idxs)} samples already done')

    ckpt_path = os.path.join(_THIS_DIR, ckpt_rel)
    print(f'[*] k={k}  ckpt={ckpt_path}')

    model, device = load_model(ckpt_path)

    # Use the TRAINING dataset
    dataset = make_dataset(cfg, cfg.train.dataset, make_transforms(cfg, True), True)
    collator = make_collator(cfg)
    limit = min(len(dataset), max_samples) if max_samples > 0 else len(dataset)
    print(f'[*] Training set: {limit}/{len(dataset)} samples')

    samples = list(existing_samples)
    det_ious, best_ious = [], []
    for s in existing_samples:
        det_ious.append(s['det_iou'])
        best_ious.append(s['best_iou'])

    for idx in range(limit):
        if idx in done_idxs:
            continue
        batch = collator([dataset[idx]])
        try:
            r = gen_sample_pseudo_label(model, device, batch, k=k)
            r['idx'] = idx
            samples.append(r)
            det_ious.append(r['det_iou'])
            best_ious.append(r['best_iou'])

            if idx % 20 == 0 or idx < 5:
                gain = r['best_iou'] - r['det_iou']
                running_gain = np.mean(best_ious) - np.mean(det_ious) if det_ious else 0
                print(f'[{idx+1:4d}/{limit}] det={r["det_iou"]:.4f}  '
                      f'best{k}={r["best_iou"]:.4f}  '
                      f'gain={gain:+.4f}  '
                      f'running_gain={running_gain:+.4f}')

            # Save checkpoint every 50 samples
            if (idx + 1) % 50 == 0:
                _save(out_path, k, ckpt_path, limit, samples, det_ious, best_ious)
        except Exception as e:
            print(f'[{idx+1:4d}/{limit}] FAILED idx={idx}: {e}')
            samples.append({'idx': idx, 'error': str(e)})

    _save(out_path, k, ckpt_path, limit, samples, det_ious, best_ious)
    print(f'\n[*] DONE. Saved to {out_path}')


def _save(out_path, k, ckpt_path, limit, samples, det_ious, best_ious):
    a = np.array(det_ious) if det_ious else np.array([0.0])
    b = np.array(best_ious) if best_ious else np.array([0.0])
    summary = {
        'k': k,
        'ckpt': ckpt_path,
        'n_planned': limit,
        'n_done': len([s for s in samples if 'error' not in s]),
        'median_det': float(np.median(a)),
        'median_best': float(np.median(b)),
        'mean_det': float(np.mean(a)),
        'mean_best': float(np.mean(b)),
        'mean_gain': float(np.mean(b - a)),
        'generated_at': datetime.datetime.now().isoformat(),
    }
    print(f'  [save] median_det={summary["median_det"]:.5f}  '
          f'median_best={summary["median_best"]:.5f}  '
          f'mean_gain={summary["mean_gain"]:+.5f}  '
          f'n={summary["n_done"]}')
    with open(out_path, 'w') as f:
        json.dump({'meta': summary, 'samples': samples}, f)


if __name__ == '__main__':
    main()
