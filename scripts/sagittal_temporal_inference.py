#!/usr/bin/env python3
"""
Sagittal temporal inference: evaluate the V4.6c + MoonViT model on the
validation split using GT detections, with optional prev-frame contour init.

Usage:
    python scripts/sagittal_temporal_inference.py \
        --cfg_file configs/sagittal_2d_v4_6c_moonvit_train.yaml \
        --checkpoint data/outputs/sagittal_2d_v4_6c_moonvit/checkpoints/latest.pt \
        --gpu 7

Toggle temporal propagation off for ablation:
    ... --no_temporal
"""
import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.utils.snake.prev_contour_init import cache_previous_predictions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg_file', default='configs/sagittal_2d_v4_6c_moonvit_train.yaml')
    p.add_argument('--checkpoint', default='')
    p.add_argument('--gpu', default='7')
    p.add_argument('--no_temporal', action='store_true',
                   help='Disable prev-frame contour init (ablation)')
    p.add_argument('--output_dir', default='data/outputs/sagittal_temporal_eval')
    return p.parse_args()


def poly_to_mask(poly_pts, h, w, stride=4):
    # poly_pts in feature coords (image/stride) -> scale back to image
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly_pts.astype(np.float32) * stride).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask


def compute_dice(mask_pred, mask_gt):
    inter = float((mask_pred & mask_gt).sum())
    denom = float(mask_pred.sum() + mask_gt.sum())
    return 2.0 * inter / denom if denom > 0 else 1.0


def gt_poly_to_mask(poly_pts, h, w):
    # poly_pts already in image coords (from dataset, output space)
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly_pts.astype(np.float32)).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['CFG_FILE'] = os.path.join(_REPO, args.cfg_file)

    from lib.config import cfg
    cfg.merge_from_file(os.path.join(_REPO, args.cfg_file))
    # force test mode with GT detection
    cfg.train_or_test = 'test'
    cfg.use_gt_det = True
    cfg.use_gt_det_train_only = False  # use GT also at test time

    from lib.networks import make_network
    from lib.datasets.make_dataset import make_data_loader

    device = torch.device('cuda:0')

    # Build network
    network = make_network(cfg).to(device)
    network.eval()

    # Load checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        ckpt_path = os.path.join(
            _REPO, 'data/outputs/sagittal_2d_v4_6c_moonvit/checkpoints/latest.pt'
        )
    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu')
        sd = state.get('net', state.get('model', state))
        network.load_state_dict(sd, strict=False)
        print('Loaded checkpoint:', ckpt_path)
    else:
        print('WARNING: checkpoint not found at', ckpt_path, '— running with random weights')

    use_temporal = not args.no_temporal
    print('Temporal propagation:', 'ON' if use_temporal else 'OFF')

    # Val data loader (batch_size=1, shuffle=False keeps slice order within cases)
    val_loader = make_data_loader(cfg, is_train=False, is_distributed=False)

    os.makedirs(args.output_dir, exist_ok=True)

    # Metrics
    dice_scores = []         # (case_id, slice_idx, cls_id, dice)
    prev_contour_cache = {}  # {cls_id: np.ndarray [P, 2] feature coords}
    prev_case_id = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move tensors to device
            for k in list(batch.keys()):
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            meta = batch.get('meta', {})
            case_id = str(meta.get('case_id', ['?'])[0] if isinstance(meta.get('case_id'), (list, torch.Tensor)) else meta.get('case_id', '?'))
            slice_idx = int(meta.get('slice_idx', [0])[0] if isinstance(meta.get('slice_idx'), (list, torch.Tensor)) else meta.get('slice_idx', 0))

            # Reset cache on new case
            if case_id != prev_case_id:
                prev_contour_cache = {}
                prev_case_id = case_id
                if batch_idx > 0:
                    print()

            # Inject prev-frame cache
            if use_temporal:
                batch['prev_contour_cache'] = prev_contour_cache

            # Forward
            output = network(batch['inp'], batch)

            # Extract predictions
            py = output.get('py')         # [M, P, 2] feature coords
            py_cls = output.get('py_cls') # [M] int tensor

            # Update cache for next slice
            if use_temporal:
                prev_contour_cache = cache_previous_predictions(output)

            # Compute Dice against GT contours
            if py is not None and py_cls is not None and py.numel() > 0:
                py_np = py.detach().cpu().numpy()   # [M, P, 2] feature coords
                cls_np = py_cls.detach().cpu().numpy()

                # GT contours from batch (image coords, output_space)
                i_gt_py = batch.get('i_gt_py')  # [1, M_gt, P, 2]
                ct_cls = batch.get('ct_cls')    # [1, M_gt] class ids

                if i_gt_py is not None and ct_cls is not None and i_gt_py.numel() > 0:
                    gt_np = i_gt_py[0].cpu().numpy()   # [M_gt, P, 2] image coords
                    gt_cls_np = ct_cls[0].cpu().numpy().astype(int)  # [M_gt]

                    inp_hw = batch['inp'].shape[-2:]  # (H_inp, W_inp)
                    h_img = int(inp_hw[0])
                    w_img = int(inp_hw[1])

                    # Match predicted to GT by class ID
                    gt_by_cls = {}
                    for j, gc in enumerate(gt_cls_np):
                        gt_by_cls[int(gc)] = gt_np[j]

                    for i in range(len(cls_np)):
                        pred_cls = int(cls_np[i])
                        if pred_cls not in gt_by_cls:
                            continue
                        pred_mask = poly_to_mask(py_np[i], h_img, w_img, stride=4)
                        gt_poly_img = gt_by_cls[pred_cls]  # image coords
                        gt_mask = gt_poly_to_mask(gt_poly_img, h_img, w_img)
                        d = compute_dice(pred_mask.astype(bool), gt_mask.astype(bool))
                        dice_scores.append((case_id, slice_idx, pred_cls, d))

            if batch_idx % 50 == 0:
                recent = [x[3] for x in dice_scores[-100:]] if dice_scores else [0.0]
                print(f'\r[{batch_idx}] case={case_id} slice={slice_idx:04d} '
                      f'recent_dice={np.mean(recent):.4f}', end='', flush=True)

    print('\n\n=== Results ===')
    if not dice_scores:
        print('No dice scores computed (no matching GT found)')
        return

    all_dice = [x[3] for x in dice_scores]
    print(f'Mean Dice (foreground): {np.mean(all_dice):.4f}')
    print(f'Median Dice: {np.median(all_dice):.4f}')
    print(f'N instances: {len(all_dice)}')

    # Per-case breakdown
    from collections import defaultdict
    per_case = defaultdict(list)
    for case_id, slice_idx, cls_id, d in dice_scores:
        per_case[case_id].append(d)
    print('\nPer-case mean Dice:')
    for cid in sorted(per_case.keys()):
        print(f'  {cid}: {np.mean(per_case[cid]):.4f} ({len(per_case[cid])} instances)')

    # Save results
    label = 'temporal' if use_temporal else 'baseline'
    out_file = os.path.join(args.output_dir, f'dice_{label}.txt')
    with open(out_file, 'w') as f:
        f.write(f'temporal={use_temporal}\n')
        f.write(f'mean_dice={np.mean(all_dice):.6f}\n')
        f.write(f'median_dice={np.median(all_dice):.6f}\n')
        f.write(f'n={len(all_dice)}\n')
        f.write('case_id,slice_idx,cls_id,dice\n')
        for case_id, slice_idx, cls_id, d in dice_scores:
            f.write(f'{case_id},{slice_idx},{cls_id},{d:.6f}\n')
    print(f'\nSaved to {out_file}')


if __name__ == '__main__':
    main()
