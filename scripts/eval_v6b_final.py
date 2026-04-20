#!/usr/bin/env python3
"""
Final evaluation of V6b best checkpoint with visualization.
Evaluates best.pt (ep 3899, 97.38%) and generates contour overlays.
"""
import sys, os, time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import numpy as np
import cv2
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _THIS_DIR)
os.environ['CFG_FILE'] = os.path.join(_THIS_DIR, 'configs', 'btcv_diffusion_dit_v3_7_gen_single_overfit.yaml')

from lib.config import cfg
cfg.merge_from_file(os.path.join(_THIS_DIR, 'configs', 'btcv_diffusion_dit_v3_7_gen_single_overfit.yaml'))
cfg.v3_7_use_regularized_per_point = False
cfg.v3_7_use_scale_conditioning = True

from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.datasets.make_dataset import make_dataset
from lib.datasets.collate_batch import make_collator
from lib.datasets.transforms import make_transforms
from lib.utils.snake import snake_config, snake_decode, snake_gcn_utils
from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict


def poly_to_mask(poly_pts, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly_pts).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask

def compute_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def get_extreme_points_torch(pts, thresh=0.02):
    N, P, _ = pts.shape
    l = pts[..., 0].min(dim=-1)[0]
    t = pts[..., 1].min(dim=-1)[0]
    r = pts[..., 0].max(dim=-1)[0]
    b = pts[..., 1].max(dim=-1)[0]
    w = r - l + 1
    h = b - t + 1
    ex_pts = torch.stack([l - w * thresh, t, r + w * thresh, t,
                          r, t - h * thresh, r, b + h * thresh,
                          l, t - h * thresh, l, b + h * thresh,
                          l - w * thresh, (t + b) / 2, r + w * thresh, (t + b) / 2,
                          (l + r) / 2, t - h * thresh, (l + r) / 2, b + h * thresh,
                          l, (t + b) / 2, r, (t + b) / 2], dim=-1)
    return ex_pts.view(N, 12, 2)

def extract_fixed_data(model, device, batch):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    dr = float(snake_config.down_ratio)
    core = model.net if hasattr(model, 'net') else model
    model.eval()
    with torch.no_grad():
        yolo_out = core.yolo(batch['inp'])
        feat_p2 = yolo_out[1][0] if isinstance(yolo_out, (list, tuple)) and len(yolo_out) > 1 else yolo_out
        cnn_feature = core.cnn_proj(feat_p2)
        gt_all = batch['i_gt_py']
        B, M, P, _ = gt_all.shape
        if 'i_it_py' in batch and batch['i_it_py'].numel() > 0:
            i_it_py = batch['i_it_py'].view(-1, P, 2)
        else:
            poly_flat = gt_all.view(B * M, P, 2)
            ex = get_extreme_points_torch(poly_flat)
            init_polys = snake_decode.get_octagon(ex).view(B, M, 12, 2)
            i_it_py = snake_gcn_utils.uniform_upsample(init_polys, 128)[0]
        c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)
        py_ind = torch.zeros(i_it_py.size(0), dtype=torch.long, device=device)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        sampled_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_py, py_ind, h, w)
        i_gt_py = gt_all.view(-1, P, 2)
        # Orient + roll alignment
        def _signed_area(poly):
            x, y = poly[..., 0], poly[..., 1]
            x1, y1 = torch.roll(x, -1, 1), torch.roll(y, -1, 1)
            return 0.5 * torch.sum(x * y1 - x1 * y, dim=1)
        area_init, area_gt = _signed_area(i_it_py), _signed_area(i_gt_py)
        orient_mismatch = ((area_init >= 0) ^ (area_gt >= 0))
        if orient_mismatch.any():
            i_gt_py[orient_mismatch] = torch.flip(i_gt_py[orient_mismatch], dims=[1])
        d2 = (i_it_py[:, :1, :] - i_gt_py).pow(2).sum(-1)
        i_gt_py = torch.stack([
            torch.roll(i_gt_py[i], -int(d2[i].argmin().item()), 0)
            for i in range(i_gt_py.size(0))
        ], 0)
        x1_raw = i_gt_py - i_it_py
        N_contours = i_it_py.size(0)
        contour_spans = []
        for ci in range(N_contours):
            span_x = i_it_py[ci, :, 0].max() - i_it_py[ci, :, 0].min()
            span_y = i_it_py[ci, :, 1].max() - i_it_py[ci, :, 1].min()
            span = max(span_x.item(), span_y.item(), 1.0)
            contour_spans.append(span)
        contour_spans = torch.tensor(contour_spans, device=device, dtype=torch.float32)
        contour_scale = contour_spans.view(N_contours, 1, 1)
        x1_norm = x1_raw / contour_scale
    return {
        'cnn_feature': cnn_feature.detach(),
        'i_it_py': i_it_py.detach(),
        'c_it_py': c_it_py.detach(),
        'py_ind': py_ind.detach(),
        'sampled_feat': sampled_feat.detach(),
        'x1_raw': x1_raw.detach(),
        'x1_norm': x1_norm.detach(),
        'contour_scale': contour_scale.detach(),
        'i_gt_py': i_gt_py.detach(),
        'gt_polys_img': gt_all.view(-1, P, 2).cpu().numpy() * dr,
        'h': h, 'w': w,
    }

def raw_predict_velocity(denoiser, cnn_feature, i_it_py, c_it_py, sampled_feat,
                          py_ind, x_t, t_tensor, contour_scale=None):
    adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, i_it_py.size(1), i_it_py.device)
    t_scaled = t_tensor * 1000.0
    v_pred, L = denoiser(cnn_feature, sampled_feat, x_t, t_scaled, adj,
                          polys=i_it_py, py_ind=py_ind, contour_scale=contour_scale)
    return v_pred, L

def ode_inference_raw(denoiser, cnn_feature, i_it_py, c_it_py, py_ind,
                       sampled_feat, h, w, contour_scale, steps=10):
    device = i_it_py.device
    N = i_it_py.size(0)
    x_t = torch.zeros(N, i_it_py.size(1), 2, device=device)
    dt = 1.0 / steps
    cs_flat = contour_scale.view(-1)
    for i in range(steps):
        t_val = i * dt
        t_tensor = torch.full((N,), t_val, device=device, dtype=torch.float32)
        v_pred, _ = raw_predict_velocity(
            denoiser, cnn_feature, i_it_py, c_it_py, sampled_feat,
            py_ind, x_t, t_tensor, contour_scale=cs_flat
        )
        x_t = x_t + v_pred * dt
    return x_t * contour_scale

def compute_true_iou(pred_disp, data, dr):
    i_it_py_d = data['i_it_py'].double()
    pred_polys = (i_it_py_d + pred_disp.double()).detach().cpu().numpy() * dr
    gt_polys = data['gt_polys_img']
    ious = []
    for idx in range(pred_polys.shape[0]):
        gt_mask = poly_to_mask(gt_polys[idx], 512, 512)
        pred_mask = poly_to_mask(pred_polys[idx], 512, 512)
        ious.append(compute_iou(pred_mask, gt_mask))
    return ious, float(np.mean(ious)), pred_polys

def main():
    device = torch.device('cuda:0')
    
    # Build model
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    model = trainer.network.to(device)
    
    # Load best checkpoint
    ckpt_path = os.path.join(_THIS_DIR, 'data/outputs/v37v6b_scale/checkpoints/best.pt')
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt.get('state_dict') or ckpt.get('model') or ckpt
    sd = remap_legacy_state_dict(sd)
    wrapper = model.module if hasattr(model, 'module') else model
    info = wrapper.load_state_dict(sd, strict=False)
    loaded = len(sd) - len(info.missing_keys)
    print(f"  Loaded {loaded}/{len(sd)} keys")
    if 'epoch' in ckpt:
        print(f"  Epoch: {ckpt['epoch']}")
    if 'iou' in ckpt:
        print(f"  Saved IoU: {ckpt['iou']*100:.2f}%")
    
    # Load dataset
    dataset = make_dataset(cfg, cfg.train.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)
    batch = collator([dataset[0]])
    
    # Extract features
    data = extract_fixed_data(model, device, batch)
    N = data['x1_raw'].shape[0]
    dr = float(snake_config.down_ratio)
    
    core = model.net if hasattr(model, 'net') else model
    denoiser = core.gcn.denoiser
    
    print(f"\nContours: {N}, Points: 128, dr={dr}")
    print(f"Scales: {data['contour_scale'].view(-1).tolist()}")
    print(f"Scale conditioning: {denoiser.use_scale_conditioning}")
    
    # Load image for visualization
    data_path = cfg.train.data_path
    list_file = os.path.join(data_path, 'train_list.txt')
    with open(list_file) as f:
        img_name = f.readline().strip()
    img_path = os.path.join(data_path, 'images', img_name)
    img_orig = cv2.imread(img_path) if os.path.exists(img_path) else np.zeros((512, 512, 3), dtype=np.uint8)
    print(f"Image: {img_path} ({img_orig.shape})")
    
    out_dir = Path(_THIS_DIR) / 'data' / 'outputs' / 'v37v6b_scale' / 'visualizations'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    contour_scale = data['contour_scale']
    colors = [
        (255, 100, 100),   # C0
        (100, 100, 255),   # C1
        (255, 255, 100),   # C2
        (100, 255, 255),   # C3
        (255, 100, 255),   # C4
        (180, 255, 180),   # C5
    ]
    
    # Evaluate with different ODE steps
    print(f"\n{'='*60}")
    print(f"  V6b Best Checkpoint — Comprehensive Evaluation")
    print(f"{'='*60}")
    
    for steps in [5, 10, 20, 50]:
        denoiser.eval()
        with torch.no_grad():
            disp = ode_inference_raw(denoiser, data['cnn_feature'], data['i_it_py'],
                                      data['c_it_py'], data['py_ind'], data['sampled_feat'],
                                      data['h'], data['w'], contour_scale, steps=steps)
            ious, mean_iou, pred_polys_img = compute_true_iou(disp, data, dr)
        
        per_str = ' '.join([f'C{i}={v*100:.1f}%' for i, v in enumerate(ious)])
        print(f"\n  {steps:3d} steps: IoU={mean_iou*100:.2f}% | {per_str}")
        
        # Generate visualization
        vis = img_orig.copy()
        gt_polys = data['gt_polys_img']
        
        for c in range(N):
            # GT in green (thin dashed-like)
            gt_pts = np.round(gt_polys[c]).astype(np.int32)
            cv2.polylines(vis, [gt_pts], True, (0, 200, 0), 1, cv2.LINE_AA)
            
            # Pred in color
            pred_pts = np.round(pred_polys_img[c]).astype(np.int32)
            cv2.polylines(vis, [pred_pts], True, colors[c], 2, cv2.LINE_AA)
            
            # Label at contour center
            cx, cy = pred_pts.mean(0).astype(int)
            label = f"C{c}:{ious[c]*100:.1f}%"
            cv2.putText(vis, label, (cx-25, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[c], 1, cv2.LINE_AA)
        
        # Title
        title = f"V6b best.pt | {steps} steps | IoU={mean_iou*100:.2f}%"
        cv2.putText(vis, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        
        save_path = out_dir / f'v6b_best_{steps}steps.png'
        cv2.imwrite(str(save_path), vis)
        print(f"  Saved: {save_path}")
    
    # Also compute octagon baseline IoU
    print(f"\n  Octagon Baseline:")
    oct_polys = data['i_it_py'].cpu().numpy() * dr
    gt_polys = data['gt_polys_img']
    oct_ious = []
    for c in range(N):
        gt_mask = poly_to_mask(gt_polys[c], 512, 512)
        oct_mask = poly_to_mask(oct_polys[c], 512, 512)
        oct_ious.append(compute_iou(oct_mask, gt_mask))
    oct_str = ' '.join([f'C{i}={v*100:.1f}%' for i, v in enumerate(oct_ious)])
    print(f"  Octagon: IoU={np.mean(oct_ious)*100:.2f}% | {oct_str}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Octagon baseline: {np.mean(oct_ious)*100:.2f}%")
    with torch.no_grad():
        disp10 = ode_inference_raw(denoiser, data['cnn_feature'], data['i_it_py'],
                                    data['c_it_py'], data['py_ind'], data['sampled_feat'],
                                    data['h'], data['w'], contour_scale, steps=10)
        ious10, mean10, _ = compute_true_iou(disp10, data, dr)
    print(f"  V6b 10 steps:     {mean10*100:.2f}%")
    print(f"  Improvement:      +{(mean10 - np.mean(oct_ious))*100:.2f}%")
    print(f"\n  Per-contour improvement:")
    for c in range(N):
        delta = ious10[c] - oct_ious[c]
        print(f"    C{c}: {oct_ious[c]*100:.1f}% → {ious10[c]*100:.1f}% (Δ={delta*100:+.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
