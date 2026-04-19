#!/usr/bin/env python3
"""
V3.7 Inference + IoU Computation Script.

Loads a trained V3.7 (or any version) checkpoint, runs inference on the
single-overfit sample, computes per-contour and mean IoU via mask
rasterisation, and saves a visualisation overlay.

Usage:
    CUDA_VISIBLE_DEVICES=2 CFG_FILE=configs/btcv_diffusion_dit_v3_7_single_overfit.yaml \
        python scripts/infer_v3_7_iou.py [--ckpt PATH] [--ode_steps 50] [--save_dir DIR]
"""

import sys, os, argparse, datetime, json
import numpy as np
import cv2
import torch

_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _THIS_DIR)

# Config must be loaded before other lib imports
_DEFAULT_CFG = os.path.join(_THIS_DIR, 'configs', 'btcv_diffusion_dit_v3_7_single_overfit.yaml')
if not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = _DEFAULT_CFG

from lib.config import cfg, args as lib_args
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.datasets.make_dataset import make_dataset
from lib.datasets.collate_batch import make_collator
from lib.datasets.transforms import make_transforms
from lib.utils.snake import snake_config, snake_decode, snake_gcn_utils


# ------------------------------------------------------------------ #
# Utility: polygon → binary mask → IoU
# ------------------------------------------------------------------ #
def poly_to_mask(poly_pts, h, w):
    """Rasterise a closed polygon into a binary mask of shape (h, w)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly_pts).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def compute_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


# ------------------------------------------------------------------ #
# Model loading (reused from infer_v3_final.py)
# ------------------------------------------------------------------ #
def load_model(ckpt_path=None):
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if ckpt_path is None:
        cfg_stem = os.path.splitext(os.path.basename(
            os.environ.get('CFG_FILE', 'default')))[0]
        ckpt_path = os.path.join(
            _THIS_DIR, 'data', 'outputs', cfg_stem, 'checkpoints', 'latest.pt')

    print(f"[*] Loading checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print(f"[!] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt_obj = torch.load(ckpt_path, map_location='cpu')
    sd = (ckpt_obj.get('state_dict')
          or ckpt_obj.get('model')
          or ckpt_obj.get('net')
          or ckpt_obj)

    from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
    sd = remap_legacy_state_dict(sd)

    wrapper = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    info = wrapper.load_state_dict(sd, strict=False)
    n_ok = len(sd) - len(info.missing_keys)
    print(f"[✔] Loaded {n_ok} / {len(sd)} keys "
          f"(missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)})")
    return trainer.network.to(device).eval(), device


# ------------------------------------------------------------------ #
# Extreme-point extraction (matches training)
# ------------------------------------------------------------------ #
def get_extreme_points_torch(pts, thresh=0.02):
    N, P, _ = pts.shape
    device = pts.device
    l = pts[..., 0].min(dim=-1)[0]
    t = pts[..., 1].min(dim=-1)[0]
    r = pts[..., 0].max(dim=-1)[0]
    b = pts[..., 1].max(dim=-1)[0]
    w = r - l + 1
    h = b - t + 1

    results = []
    for i in range(N):
        poly_i = pts[i]
        def _find_ex(dim_idx, is_min, other_dim_range):
            if is_min:
                idx = torch.argmin(poly_i[:, dim_idx])
                val = poly_i[idx, dim_idx]
                def cond(j): return poly_i[j, dim_idx] - val <= thresh * other_dim_range
            else:
                idx = torch.argmax(poly_i[:, dim_idx])
                val = poly_i[idx, dim_idx]
                def cond(j): return val - poly_i[j, dim_idx] <= thresh * other_dim_range
            idxs = [idx.item()]
            tmp = (idx + 1) % P
            while tmp != idx and cond(tmp):
                idxs.append(tmp.item())
                tmp = (tmp + 1) % P
            tmp = (idx - 1) % P
            while tmp != idx and cond(tmp):
                idxs.append(tmp.item())
                tmp = (tmp - 1) % P
            return torch.tensor(idxs, device=device)

        t_idxs = _find_ex(1, True, h[i])
        tt_x = (poly_i[t_idxs, 0].max() + poly_i[t_idxs, 0].min()) / 2
        tt = torch.stack([tt_x, t[i]])

        b_idxs = _find_ex(1, False, h[i])
        bb_x = (poly_i[b_idxs, 0].max() + poly_i[b_idxs, 0].min()) / 2
        bb = torch.stack([bb_x, b[i]])

        l_idxs = _find_ex(0, True, w[i])
        ll_y = (poly_i[l_idxs, 1].max() + poly_i[l_idxs, 1].min()) / 2
        ll = torch.stack([l[i], ll_y])

        r_idxs = _find_ex(0, False, w[i])
        rr_y = (poly_i[r_idxs, 1].max() + poly_i[r_idxs, 1].min()) / 2
        rr = torch.stack([r[i], rr_y])

        results.append(torch.stack([tt, ll, bb, rr]))
    return torch.stack(results)


# ------------------------------------------------------------------ #
# Main inference + IoU
# ------------------------------------------------------------------ #
def run_inference_iou(model, device, batch, save_dir, ode_steps=50):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    dr = float(snake_config.down_ratio)
    core = model.net if hasattr(model, 'net') else model

    with torch.no_grad():
        # 1. Feature extraction
        yolo_out = core.yolo(batch['inp'])
        feat_p2 = yolo_out[1][0] if isinstance(yolo_out, (list, tuple)) and len(yolo_out) > 1 else yolo_out
        cnn_feature = core.cnn_proj(feat_p2)

        gt_all = batch['i_gt_py']
        if gt_all.numel() == 0:
            print("  [!] No GT polygons")
            return {}

        B, M, P, _ = gt_all.shape

        # 2. Initial contour (octagon from dataset)
        if 'i_it_py' in batch and batch['i_it_py'].numel() > 0:
            i_it_py = batch['i_it_py'].view(-1, P, 2)
        else:
            poly_flat = gt_all.view(B * M, P, 2)
            ex = get_extreme_points_torch(poly_flat)
            init_polys = snake_decode.get_octagon(ex).view(B, M, 12, 2)
            i_it_py = snake_gcn_utils.uniform_upsample(init_polys, 128)[0]

        c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)
        py_ind = torch.zeros(i_it_py.size(0), dtype=torch.long, device=device) if B == 1 else \
            torch.cat([torch.full((M,), i, dtype=torch.long, device=device) for i in range(B)])

        # 3. Sample displacement
        use_iter = getattr(cfg, 'use_iterative_refinement', False)
        if use_iter:
            iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
            fractions = list(getattr(cfg, 'iterative_fractions', []))
            if not fractions:
                fractions = [1.0 / (iter_steps - i) for i in range(iter_steps)]
            disp = core.gcn.sample_disp_iterative(
                cnn_feature, i_it_py, c_it_py, py_ind,
                num_iter_steps=iter_steps, fractions=fractions,
                ode_steps=ode_steps)
        else:
            disp = core.gcn.sample_disp(
                cnn_feature, i_it_py, c_it_py, py_ind, steps=ode_steps)

        # 4. Optional Fourier smoothing
        fk = int(getattr(cfg, 'fourier_smooth_k', 0))
        if fk > 0:
            from lib.networks.diffusion.flow_matching_evolution import FlowMatchingEvolution
            disp = FlowMatchingEvolution.fourier_smooth(disp, fk)

        pred_polys_affine = (i_it_py + disp)  # in affine coords
        pred_polys = pred_polys_affine.cpu().numpy() * dr
        gt_polys = gt_all.view(-1, P, 2).cpu().numpy() * dr
        init_polys_np = i_it_py.cpu().numpy() * dr

    # 5. Compute IoU
    if 'orig_img' in batch:
        img_raw = batch['orig_img'][0]
        img = img_raw.detach().cpu().numpy() if torch.is_tensor(img_raw) else img_raw
        img = img.astype(np.uint8)
    else:
        img = np.zeros((512, 512, 3), dtype=np.uint8)

    H_img, W_img = img.shape[:2]
    ious = []
    for idx in range(pred_polys.shape[0]):
        gt_mask = poly_to_mask(gt_polys[idx], H_img, W_img)
        pred_mask = poly_to_mask(pred_polys[idx], H_img, W_img)
        iou = compute_iou(pred_mask, gt_mask)
        ious.append(iou)
        print(f"  Contour {idx}: IoU = {iou:.6f} ({iou*100:.3f}%)")

    mean_iou = float(np.mean(ious)) if ious else 0.0
    print(f"\n  ★ Mean IoU = {mean_iou:.6f} ({mean_iou*100:.3f}%)")

    # 6. Visualise
    vis = img.copy()
    for poly in gt_polys:
        cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 255, 0), 2)
    for poly in init_polys_np:
        cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 255, 255), 1)
    for poly in pred_polys:
        cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 0, 255), 2)

    # Add IoU text
    for idx, iou in enumerate(ious):
        cx, cy = int(pred_polys[idx, :, 0].mean()), int(pred_polys[idx, :, 1].mean())
        cv2.putText(vis, f"IoU:{iou*100:.1f}%", (cx-30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f"v3_7_iou_{ts}.png")
    cv2.imwrite(save_path, vis)
    print(f"  [*] Saved visualisation: {save_path}")

    # 7. Save metrics JSON
    metrics = {
        'timestamp': ts,
        'ode_steps': ode_steps,
        'mean_iou': mean_iou,
        'per_contour_iou': ious,
        'fourier_smooth_k': fk,
        'use_iterative_refinement': use_iter,
    }
    json_path = os.path.join(save_dir, f"v3_7_metrics_{ts}.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  [*] Saved metrics: {json_path}")

    return metrics


def main():
    # Use env vars for extra params since lib.config consumes sys.argv
    ckpt = os.environ.get('CKPT', None)
    ode_steps = int(os.environ.get('ODE_STEPS', getattr(cfg, 'flow_ode_steps', 50)))
    save_dir = os.environ.get('SAVE_DIR', os.path.join(_THIS_DIR, 'visual', 'v3_7_eval'))
    index = int(os.environ.get('SAMPLE_INDEX', 0))

    model, device = load_model(ckpt)

    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)

    os.makedirs(save_dir, exist_ok=True)

    avg_n = int(getattr(cfg, 'infer_avg_samples', 1))
    noise_s = float(getattr(cfg, 'infer_noise_scale', 1.0))
    print(f"\n{'='*60}")
    print(f"V3.7 Inference + IoU Evaluation")
    print(f"  ODE steps: {ode_steps}  |  avg_samples: {avg_n}  |  noise_scale: {noise_s}")
    print(f"  Save dir:  {save_dir}")
    print(f"{'='*60}\n")

    batch = collator([dataset[index]])
    metrics = run_inference_iou(model, device, batch, save_dir,
                                ode_steps=ode_steps)
    return metrics


if __name__ == '__main__':
    main()
