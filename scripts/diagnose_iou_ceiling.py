#!/usr/bin/env python3
"""
Diagnostic: What is the maximum IoU achievable with PERFECT displacement prediction?

This tests whether the IoU computation pipeline (coordinate transforms, rasterization)
introduces precision loss that limits IoU below 99.9%.

Tests:
1. GT displacement → IoU (should be 100%)
2. GT displacement + tiny noise → IoU (sensitivity analysis)
3. Direct GT polygon → IoU (bypass displacement entirely)
"""
import sys, os
import numpy as np
import cv2
import torch

_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _THIS_DIR)

if not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = os.path.join(_THIS_DIR, 'configs', 'btcv_diffusion_dit_v3_7_9_single_overfit.yaml')

from lib.config import cfg
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.datasets.make_dataset import make_dataset
from lib.datasets.collate_batch import make_collator
from lib.datasets.transforms import make_transforms
from lib.utils.snake import snake_config, snake_decode, snake_gcn_utils


def poly_to_mask(poly_pts, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly_pts).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask

def compute_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dr = float(snake_config.down_ratio)
    H_img, W_img = 512, 512
    
    # Load dataset
    dataset = make_dataset(cfg, cfg.train.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)
    batch = collator([dataset[0]])
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    
    gt_all = batch['i_gt_py']  # (B, M, P, 2) in feature space
    B, M, P, _ = gt_all.shape
    print(f"GT shape: B={B}, M={M}, P={P}")
    print(f"down_ratio: {dr}")
    
    gt_feat = gt_all.view(-1, P, 2)  # in feature space (128x128 coords)
    gt_img = gt_feat.cpu().numpy() * dr  # in image space (512x512 coords)
    
    # Get init polys (octagons)
    if 'i_it_py' in batch and batch['i_it_py'].numel() > 0:
        i_it_py = batch['i_it_py'].view(-1, P, 2)
    else:
        from lib.utils.snake.snake_gcn_utils import get_extreme_points_torch
        poly_flat = gt_all.view(B * M, P, 2)
        ex = get_extreme_points_torch(poly_flat)
        init_polys = snake_decode.get_octagon(ex).view(B, M, 12, 2)
        i_it_py = snake_gcn_utils.uniform_upsample(init_polys, 128)[0]

    # Orientation + point alignment (same as finetune script)
    def _signed_area(poly):
        x, y = poly[..., 0], poly[..., 1]
        x1, y1 = torch.roll(x, -1, 1), torch.roll(y, -1, 1)
        return 0.5 * torch.sum(x * y1 - x1 * y, dim=1)
    
    i_gt_py = gt_feat.clone()
    area_init, area_gt = _signed_area(i_it_py), _signed_area(i_gt_py)
    orient_mismatch = ((area_init >= 0) ^ (area_gt >= 0))
    if orient_mismatch.any():
        i_gt_py[orient_mismatch] = torch.flip(i_gt_py[orient_mismatch], dims=[1])
    
    d2 = (i_it_py[:, :1, :] - i_gt_py).pow(2).sum(-1)
    i_gt_py = torch.stack([
        torch.roll(i_gt_py[i], -int(d2[i].argmin().item()), 0)
        for i in range(i_gt_py.size(0))
    ], 0)
    
    gt_disp = i_gt_py - i_it_py  # perfect GT displacement in feature space

    print("\n=== TEST 1: GT polygon self-IoU ===")
    print("(GT mask compared against itself - should be 100%)")
    for idx in range(gt_img.shape[0]):
        mask = poly_to_mask(gt_img[idx], H_img, W_img)
        iou = compute_iou(mask, mask)
        area = mask.sum()
        print(f"  Contour {idx}: area={area}px, self-IoU={iou*100:.3f}%")

    print("\n=== TEST 2: GT displacement → IoU ===")
    print("(init_poly + gt_disp should exactly reconstruct GT)")
    pred_perfect = (i_it_py + gt_disp).cpu().numpy() * dr  # should == gt_img
    for idx in range(gt_img.shape[0]):
        gt_mask = poly_to_mask(gt_img[idx], H_img, W_img)
        pred_mask = poly_to_mask(pred_perfect[idx], H_img, W_img)
        iou = compute_iou(pred_mask, gt_mask)
        # Check numeric precision
        max_diff = np.abs(pred_perfect[idx] - gt_img[idx]).max()
        mean_diff = np.abs(pred_perfect[idx] - gt_img[idx]).mean()
        print(f"  Contour {idx}: IoU={iou*100:.3f}%, max_diff={max_diff:.6f}px, mean_diff={mean_diff:.6f}px")
    mean_iou = np.mean([compute_iou(poly_to_mask(pred_perfect[i], H_img, W_img),
                                     poly_to_mask(gt_img[i], H_img, W_img))
                         for i in range(gt_img.shape[0])])
    print(f"  Mean IoU: {mean_iou*100:.3f}%")

    print("\n=== TEST 3: Aligned GT polygon → IoU ===")
    print("(Aligned GT points directly, no displacement pathway)")
    aligned_gt_img = i_gt_py.cpu().numpy() * dr
    for idx in range(gt_img.shape[0]):
        gt_mask = poly_to_mask(gt_img[idx], H_img, W_img)
        aligned_mask = poly_to_mask(aligned_gt_img[idx], H_img, W_img)
        iou = compute_iou(aligned_mask, gt_mask)
        max_diff = np.abs(aligned_gt_img[idx] - gt_img[idx]).max()
        print(f"  Contour {idx}: IoU={iou*100:.3f}%, max_point_diff={max_diff:.6f}px")
    mean_iou = np.mean([compute_iou(poly_to_mask(aligned_gt_img[i], H_img, W_img),
                                     poly_to_mask(gt_img[i], H_img, W_img))
                         for i in range(gt_img.shape[0])])
    print(f"  Mean IoU: {mean_iou*100:.3f}%")

    print("\n=== TEST 4: Effect of point alignment on IoU ===")
    print("(GT WITHOUT orientation/point alignment)")
    raw_gt_feat = gt_all.view(-1, P, 2)
    raw_gt_img = raw_gt_feat.cpu().numpy() * dr
    for idx in range(gt_img.shape[0]):
        gt_mask = poly_to_mask(gt_img[idx], H_img, W_img)
        raw_mask = poly_to_mask(raw_gt_img[idx], H_img, W_img)
        iou = compute_iou(raw_mask, gt_mask)
        print(f"  Contour {idx}: IoU={iou*100:.3f}%")
    mean_iou = np.mean([compute_iou(poly_to_mask(raw_gt_img[i], H_img, W_img),
                                     poly_to_mask(gt_img[i], H_img, W_img))
                         for i in range(gt_img.shape[0])])
    print(f"  Mean IoU: {mean_iou*100:.3f}%")

    print("\n=== TEST 5: Displacement normalization roundtrip ===")
    print("(Check if normalize → unnormalize preserves precision)")
    # Load model to access normalize/unnormalize functions
    model = make_network(cfg)
    trainer = make_trainer(cfg, model)
    if os.environ.get('CKPT'):
        ckpt = torch.load(os.environ['CKPT'], map_location='cpu')
        state = ckpt.get('net', ckpt)
        model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    core = model.net if hasattr(model, 'net') else model
    gcn = core.gcn
    
    # Normalize then unnormalize
    gt_disp_norm = gcn.normalize_disp(gt_disp)
    gt_disp_unnorm = gcn.unnormalize_disp(gt_disp_norm)
    roundtrip_error = (gt_disp_unnorm - gt_disp).abs()
    print(f"  Normalization roundtrip max error: {roundtrip_error.max().item():.10f}")
    print(f"  Normalization roundtrip mean error: {roundtrip_error.mean().item():.10f}")
    
    # Check IoU after roundtrip
    pred_roundtrip = (i_it_py + gt_disp_unnorm).cpu().numpy() * dr
    for idx in range(gt_img.shape[0]):
        gt_mask = poly_to_mask(gt_img[idx], H_img, W_img)
        pred_mask = poly_to_mask(pred_roundtrip[idx], H_img, W_img)
        iou = compute_iou(pred_mask, gt_mask)
        max_diff = np.abs(pred_roundtrip[idx] - gt_img[idx]).max()
        print(f"  Contour {idx}: IoU={iou*100:.3f}%, roundtrip_diff={max_diff:.8f}px")
    mean_iou = np.mean([compute_iou(poly_to_mask(pred_roundtrip[i], H_img, W_img),
                                     poly_to_mask(gt_img[i], H_img, W_img))
                         for i in range(gt_img.shape[0])])
    print(f"  Mean IoU after normalization roundtrip: {mean_iou*100:.3f}%")

    print("\n=== TEST 6: What IoU does the finetune's compute_iou_from_pred give with GT? ===")
    print("(Using exact same code path as finetune script)")
    # Simulate what compute_iou_from_pred does
    gt_polys_img_finetune = gt_all.view(-1, P, 2).cpu().numpy() * dr  # from extract_fixed_data
    pred_finetune = (i_it_py + gt_disp).detach().cpu().numpy() * dr  # init + gt_disp
    for idx in range(gt_img.shape[0]):
        gt_mask = poly_to_mask(gt_polys_img_finetune[idx], H_img, W_img)
        pred_mask = poly_to_mask(pred_finetune[idx], H_img, W_img)
        iou = compute_iou(pred_mask, gt_mask)
        # Compare polygon vertices
        max_diff = np.abs(pred_finetune[idx] - gt_polys_img_finetune[idx]).max()
        print(f"  Contour {idx}: IoU={iou*100:.3f}%, max_vertex_diff={max_diff:.8f}px")
    mean_iou = np.mean([compute_iou(poly_to_mask(pred_finetune[i], H_img, W_img),
                                     poly_to_mask(gt_polys_img_finetune[i], H_img, W_img))
                         for i in range(gt_img.shape[0])])
    print(f"  Mean IoU (finetune code path): {mean_iou*100:.3f}%")

    print("\n=== TEST 7: GT stored in batch vs GT after alignment ===")
    print("(The gt_polys_img uses ORIGINAL gt, pred uses ALIGNED gt)")
    print("This tests if orientation/point alignment changes the polygon shape")
    for idx in range(gt_img.shape[0]):
        orig = gt_all.view(-1, P, 2)[idx].cpu().numpy() * dr
        aligned = i_gt_py[idx].cpu().numpy() * dr
        orig_mask = poly_to_mask(orig, H_img, W_img)
        aligned_mask = poly_to_mask(aligned, H_img, W_img)
        iou = compute_iou(orig_mask, aligned_mask)
        max_diff = np.abs(orig - aligned).max()
        # Check if points are the same (just reordered)
        print(f"  Contour {idx}: IoU(orig,aligned)={iou*100:.3f}%, max_coord_diff={max_diff:.4f}px")

    print("\n=== CRITICAL TEST: finetune GT path mismatch ===")
    print("In finetune, gt_polys_img uses ORIGINAL gt, but pred=init+disp uses ALIGNED gt.")
    print("If alignment changes point order (rotation/flip), the POLYGON SHAPE stays same")
    print("but let's verify...")
    for idx in range(gt_img.shape[0]):
        orig_gt = gt_polys_img_finetune[idx]  # original, unaligned
        pred_gt = pred_finetune[idx]  # init + aligned_gt_disp
        orig_mask = poly_to_mask(orig_gt, H_img, W_img)
        pred_mask = poly_to_mask(pred_gt, H_img, W_img)
        iou = compute_iou(orig_mask, pred_mask)
        diff_pixels = np.abs(orig_mask.astype(int) - pred_mask.astype(int)).sum()
        orig_area = orig_mask.sum()
        pred_area = pred_mask.sum()
        print(f"  Contour {idx}: IoU={iou*100:.3f}%, diff_pixels={diff_pixels}, "
              f"orig_area={orig_area}, pred_area={pred_area}")


if __name__ == '__main__':
    main()
