#!/usr/bin/env python3
"""
Thorough evaluation of V3.7-gen checkpoint.
Tests multiple ODE step counts, deterministic vs stochastic start, multi-sample averaging.
"""
import sys, os, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _THIS_DIR)


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
                vals = poly_i[:, dim_idx]
                threshold = vals.min() + thresh * other_dim_range
                mask = vals <= threshold
            else:
                vals = poly_i[:, dim_idx]
                threshold = vals.max() - thresh * other_dim_range
                mask = vals >= threshold
            if mask.sum() == 0:
                mask[vals.argmin() if is_min else vals.argmax()] = True
            return poly_i[mask].mean(0)

        wi, hi = w[i].item(), h[i].item()
        tt = _find_ex(1, True, hi)
        ll = _find_ex(0, True, wi)
        bb = _find_ex(1, False, hi)
        rr = _find_ex(0, False, wi)
        results.append(torch.stack([tt, ll, bb, rr]))
    return torch.stack(results)


def extract_fixed_data(model, device, batch):
    from lib.utils.snake import snake_config, snake_decode, snake_gcn_utils

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

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
        gcn = core.gcn
        x1 = gcn.normalize_disp(x1_raw)

    return {
        'cnn_feature': cnn_feature.detach(),
        'i_it_py': i_it_py.detach(),
        'c_it_py': c_it_py.detach(),
        'py_ind': py_ind.detach(),
        'sampled_feat': sampled_feat.detach(),
        'x1': x1.detach(),
        'x1_raw': x1_raw.detach(),
        'i_gt_py': i_gt_py.detach(),
        'gt_polys_img': gt_all.view(-1, P, 2).cpu().numpy() * dr,
        'h': h, 'w': w,
    }


def ode_inference(gcn, cnn_feature, i_it_py, c_it_py, py_ind, sampled_feat,
                  steps=10, noise_scale=0.3, deterministic=False, seed=None):
    device = i_it_py.device
    N = i_it_py.size(0)
    if deterministic:
        x_t = torch.zeros(N, i_it_py.size(1), 2, device=device)
    else:
        if seed is not None:
            torch.manual_seed(seed)
        x_t = torch.randn(N, i_it_py.size(1), 2, device=device) * noise_scale
    dt = 1.0 / steps
    for i in range(steps):
        t_val = i * dt
        t_tensor = torch.full((N,), t_val, device=device, dtype=torch.float32)
        v_pred, _ = gcn.predict_velocity(
            cnn_feature, i_it_py, c_it_py, sampled_feat, py_ind, x_t, t_tensor
        )
        x_t = x_t + v_pred * dt
    return gcn.denormalize_disp(x_t)


def compute_iou_from_pred(pred_disp, data, dr):
    i_it_py_d = data['i_it_py'].double()
    pred_polys = (i_it_py_d + pred_disp.double()).detach().cpu().numpy() * dr
    gt_polys = (i_it_py_d + data['x1'].double()).detach().cpu().numpy() * dr
    H_img, W_img = 512, 512
    ious = []
    for idx in range(pred_polys.shape[0]):
        gt_mask = poly_to_mask(gt_polys[idx], H_img, W_img)
        pred_mask = poly_to_mask(pred_polys[idx], H_img, W_img)
        ious.append(compute_iou(pred_mask, gt_mask))
    return ious, float(np.mean(ious))


def main():
    ckpt_path = os.environ.get('CKPT', 'data/outputs/v37gen_ns03/checkpoints/best.pt')
    cfg_file = os.environ.get('CFG_FILE', 'configs/btcv_diffusion_dit_v3_7_gen_single_overfit.yaml')
    noise_scale = float(os.environ.get('FM_NOISE_SCALE', '0.3'))

    # Load config
    os.environ.setdefault('CFG_FILE', os.path.join(_THIS_DIR, cfg_file))
    from lib.config import cfg
    cfg.merge_from_file(os.path.join(_THIS_DIR, cfg_file))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from lib.networks import make_network
    from lib.train.trainers import make_trainer
    from lib.datasets.make_dataset import make_dataset
    from lib.datasets.collate_batch import make_collator
    from lib.datasets.transforms import make_transforms
    from lib.utils.snake import snake_config

    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    model = trainer.network.to(device)

    # Load checkpoint
    ckpt_full = os.path.join(_THIS_DIR, ckpt_path)
    print(f"Loading checkpoint: {ckpt_full}")
    ckpt = torch.load(ckpt_full, map_location='cpu')
    sd = ckpt.get('state_dict') or ckpt.get('model') or ckpt
    from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
    sd = remap_legacy_state_dict(sd)
    wrapper = model.module if hasattr(model, 'module') else model
    info = wrapper.load_state_dict(sd, strict=False)
    print(f"  Loaded {len(sd)-len(info.missing_keys)}/{len(sd)} keys")
    if info.missing_keys:
        print(f"  Missing: {info.missing_keys[:5]}...")
    print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}, IoU: {ckpt.get('iou', '?')}")

    # Load data
    dataset = make_dataset(cfg, cfg.train.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)
    batch = collator([dataset[0]])

    data = extract_fixed_data(model, device, batch)
    core = model.net if hasattr(model, 'net') else model
    gcn = core.gcn
    denoiser = gcn.denoiser
    dr = float(snake_config.down_ratio)
    N = data['x1'].shape[0]

    denoiser.eval()

    print(f"\n{'='*70}")
    print(f"  V3.7-gen Evaluation | noise_scale={noise_scale}")
    print(f"  Contours: {N}, Points: {data['x1'].shape[1]}")
    print(f"{'='*70}\n")

    # Test configurations
    configs = [
        # (description, ode_steps, deterministic, n_samples)
        ("Deterministic x0=0, 10 steps", 10, True, 1),
        ("Deterministic x0=0, 20 steps", 20, True, 1),
        ("Deterministic x0=0, 50 steps", 50, True, 1),
        ("Deterministic x0=0, 100 steps", 100, True, 1),
        ("Stochastic 3-sample avg, 10 steps", 10, False, 3),
        ("Stochastic 10-sample avg, 10 steps", 10, False, 10),
        ("Stochastic 20-sample avg, 10 steps", 10, False, 20),
        ("Stochastic 20-sample avg, 20 steps", 20, False, 20),
        ("Stochastic 20-sample avg, 50 steps", 50, False, 20),
    ]

    with torch.no_grad():
        for desc, steps, determ, n_samples in configs:
            all_ious = []
            t0 = time.time()
            for s in range(n_samples):
                disp = ode_inference(
                    gcn, data['cnn_feature'], data['i_it_py'], data['c_it_py'],
                    data['py_ind'], data['sampled_feat'],
                    steps=steps, noise_scale=noise_scale,
                    deterministic=determ, seed=s*42 if not determ else None
                )
                ious_i, _ = compute_iou_from_pred(disp, data, dr)
                all_ious.append(ious_i)
            elapsed = time.time() - t0

            # Compute mean per-contour IoU
            mean_per_c = [float(np.mean([all_ious[r][c] for r in range(n_samples)]))
                          for c in range(N)]
            mean_iou = float(np.mean(mean_per_c))

            # Also compute std for stochastic
            if n_samples > 1:
                std_per_c = [float(np.std([all_ious[r][c] for r in range(n_samples)]))
                             for c in range(N)]
                std_iou = float(np.mean(std_per_c))
            else:
                std_iou = 0.0

            per_c_str = ' '.join([f'C{i}={v*100:.1f}%' for i, v in enumerate(mean_per_c)])
            std_str = f" ± {std_iou*100:.1f}%" if n_samples > 1 else ""
            print(f"  [{desc}]")
            print(f"    Mean IoU = {mean_iou*100:.2f}%{std_str} | {per_c_str} | {elapsed:.1f}s")
            print()


if __name__ == '__main__':
    main()
