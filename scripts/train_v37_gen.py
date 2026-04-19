#!/usr/bin/env python3
"""
Standalone Flow Matching training for V3.7-gen (generalizable anti-burr denoiser).

Caches backbone features once, then trains only the denoiser with proper FM loss.
~100x faster than train_net.py for single-sample validation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_v37_gen.py \
        --cfg configs/btcv_diffusion_dit_v3_7_gen_single_overfit.yaml \
        --epochs 10000 --lr 3e-4 --save_dir data/outputs/v37gen
"""
import sys, os, argparse, json, time, math
import numpy as np
import cv2
import torch
import torch.nn as nn
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


def extract_fixed_data(model, device, batch):
    """Extract backbone features and GT data in eval mode (cached for reuse)."""
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

        # Orientation alignment
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


def ode_inference(gcn, cnn_feature, i_it_py, c_it_py, py_ind, sampled_feat,
                  steps=10, noise_scale=0.3):
    """Run ODE integration for inference (mirrors _sample_disp_from_sampled_feat)."""
    device = i_it_py.device
    N = i_it_py.size(0)
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


def main():
    # Use env vars to avoid lib.config argparse conflict
    class Args:
        cfg = os.environ.get('CFG_FILE', 'configs/btcv_diffusion_dit_v3_7_gen_single_overfit.yaml')
        epochs = int(os.environ.get('FM_EPOCHS', '10000'))
        lr = float(os.environ.get('FM_LR', '3e-4'))
        eta_min = float(os.environ.get('FM_ETA_MIN', '1e-7'))
        save_dir = os.environ.get('FM_SAVE_DIR', 'data/outputs/v37gen')
        eval_every = int(os.environ.get('FM_EVAL_EVERY', '200'))
        save_every = int(os.environ.get('FM_SAVE_EVERY', '2000'))
        ode_steps = int(os.environ.get('FM_ODE_STEPS', '10'))
        noise_scale = float(os.environ['FM_NOISE_SCALE']) if 'FM_NOISE_SCALE' in os.environ else None
        scope = os.environ.get('FM_SCOPE', 'denoiser')
        ckpt = os.environ.get('CKPT', '')
    args = Args()

    # Load config
    if not os.environ.get('CFG_FILE'):
        os.environ['CFG_FILE'] = os.path.join(_THIS_DIR, args.cfg)
    from lib.config import cfg
    cfg.merge_from_file(os.path.join(_THIS_DIR, args.cfg))

    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)

    print(f"=== V3.7-gen Flow Matching Training ===")
    print(f"  Config: {args.cfg}")
    print(f"  LR: {args.lr} → {args.eta_min}")
    print(f"  Epochs: {args.epochs}")
    print(f"  ODE steps: {args.ode_steps}")
    print(f"  Scope: {args.scope}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    from lib.networks import make_network
    from lib.train.trainers import make_trainer
    from lib.datasets.make_dataset import make_dataset
    from lib.datasets.collate_batch import make_collator
    from lib.datasets.transforms import make_transforms
    from lib.utils.snake import snake_config

    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    model = trainer.network.to(device)

    # Load checkpoint if provided
    if args.ckpt and os.path.exists(args.ckpt):
        print(f"  Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location='cpu')
        sd = ckpt.get('state_dict') or ckpt.get('model') or ckpt
        from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
        sd = remap_legacy_state_dict(sd)
        wrapper = model.module if hasattr(model, 'module') else model
        info = wrapper.load_state_dict(sd, strict=False)
        print(f"  Loaded {len(sd)-len(info.missing_keys)}/{len(sd)} keys")

    # Load single sample
    dataset = make_dataset(cfg, cfg.train.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)
    batch = collator([dataset[0]])

    print("[*] Extracting backbone features (cached)...")
    data = extract_fixed_data(model, device, batch)
    print(f"  Feature shape: {data['cnn_feature'].shape}")
    print(f"  Contours: {data['x1'].shape[0]}, Points: {data['x1'].shape[1]}")
    print(f"  GT disp range: [{data['x1'].min():.4f}, {data['x1'].max():.4f}]")

    # Get evolution module
    core = model.net if hasattr(model, 'net') else model
    gcn = core.gcn
    denoiser = gcn.denoiser
    dr = float(snake_config.down_ratio)

    # Flow Matching params
    noise_scale = args.noise_scale if args.noise_scale is not None else gcn._flow_train_noise_scale
    spectral_k = gcn._spectral_loss_k
    hf_weight = gcn._hf_loss_weight
    print(f"  Noise scale: {noise_scale}")
    print(f"  Spectral loss: k={spectral_k}, hf_weight={hf_weight}")

    # Freeze backbone, train denoiser
    for p in model.parameters():
        p.requires_grad = False

    train_params = []
    if args.scope == 'final':
        for p in denoiser.final_layer.parameters():
            p.requires_grad = True
            train_params.append(p)
    elif args.scope == 'all':
        for p in model.parameters():
            p.requires_grad = True
            train_params.append(p)
    else:  # denoiser
        for p in denoiser.parameters():
            p.requires_grad = True
            train_params.append(p)

    n_train = sum(p.numel() for p in train_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable: {n_train:,} params | Frozen: {n_frozen:,} params")

    # Optimizer
    wd = float(getattr(cfg.train, 'weight_decay', 0.01))
    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=wd, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min
    )

    # Cached data
    cnn_feature = data['cnn_feature']
    i_it_py = data['i_it_py']
    c_it_py = data['c_it_py']
    py_ind = data['py_ind']
    sampled_feat = data['sampled_feat']
    x1 = data['x1']  # normalized GT displacement
    N = x1.size(0)

    # Training
    log_path = os.path.join(args.save_dir, 'train_log.jsonl')
    best_iou = 0.0
    best_epoch = 0

    print(f"\n[*] Starting FM training ({args.epochs} epochs)...\n")
    t_start = time.time()

    for epoch in range(args.epochs):
        denoiser.train()

        # --- Flow Matching Forward ---
        # Random t ~ U[0,1]
        t = torch.rand(N, device=device).view(N, 1, 1)
        # Random noise x_0
        x0 = torch.randn_like(x1) * noise_scale
        # Interpolated state x_t
        x_t = (1.0 - t) * x0 + t * x1
        # Target velocity
        v_target = x1 - x0

        # Predict velocity
        v_pred, L_reg = gcn.predict_velocity(
            cnn_feature, i_it_py, c_it_py, sampled_feat, py_ind,
            x_t, t.view(-1)
        )

        # Velocity loss (with optional spectral decomposition)
        if spectral_k > 0 and v_pred.size(1) > spectral_k * 2:
            v_pred_lf = gcn.fourier_smooth(v_pred, spectral_k)
            v_target_lf = gcn.fourier_smooth(v_target, spectral_k)
            loss_lf = F.mse_loss(v_pred_lf, v_target_lf)
            loss_hf = F.mse_loss(v_pred - v_pred_lf, v_target - v_target_lf)
            fm_loss = loss_lf + hf_weight * loss_hf
        else:
            fm_loss = F.mse_loss(v_pred, v_target)

        # Total loss = FM + regularization (Laplacian + delta reg from denoiser)
        loss = fm_loss + L_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        lr = optimizer.param_groups[0]['lr']
        log_entry = {'epoch': epoch, 'loss': loss.item(), 'fm_loss': fm_loss.item(),
                     'reg': L_reg.item() if isinstance(L_reg, torch.Tensor) else float(L_reg),
                     'lr': lr}

        # Periodic evaluation with ODE inference
        if (epoch + 1) % args.eval_every == 0 or epoch == 0:
            denoiser.eval()
            with torch.no_grad():
                # Average over 3 random initializations for stable IoU
                all_ious = []
                for _ in range(3):
                    disp = ode_inference(gcn, cnn_feature, i_it_py, c_it_py, py_ind,
                                         sampled_feat, steps=args.ode_steps,
                                         noise_scale=noise_scale)
                    ious_i, _ = compute_iou_from_pred(disp, data, dr)
                    all_ious.append(ious_i)
                # Mean IoU across runs
                mean_ious = [float(np.mean([all_ious[r][c] for r in range(3)]))
                             for c in range(N)]
                mean_iou = float(np.mean(mean_ious))

            log_entry['mean_iou'] = mean_iou
            log_entry['per_iou'] = mean_ious

            if mean_iou > best_iou:
                best_iou = mean_iou
                best_epoch = epoch
                sd = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save({
                    'state_dict': sd, 'epoch': epoch,
                    'loss': loss.item(), 'iou': mean_iou,
                }, os.path.join(args.save_dir, 'checkpoints', 'best.pt'))

            elapsed = time.time() - t_start
            per_iou_str = ' '.join([f'C{i}={v*100:.1f}%' for i, v in enumerate(mean_ious)])
            print(f"  Ep {epoch:5d} | loss={loss.item():.4e} fm={fm_loss.item():.4e} "
                  f"reg={log_entry['reg']:.4e} | lr={lr:.2e} | "
                  f"IoU={mean_iou*100:.2f}% (best={best_iou*100:.2f}%@{best_epoch}) | "
                  f"{per_iou_str} | {elapsed:.0f}s")

        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            sd = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({
                'state_dict': sd, 'epoch': epoch, 'loss': loss.item(),
            }, os.path.join(args.save_dir, 'checkpoints', f'epoch_{epoch+1}.pt'))

    # Final evaluation
    denoiser.eval()
    with torch.no_grad():
        disp = ode_inference(gcn, cnn_feature, i_it_py, c_it_py, py_ind,
                             sampled_feat, steps=args.ode_steps, noise_scale=noise_scale)
        ious, mean_iou = compute_iou_from_pred(disp, data, dr)

    print(f"\n=== Final Results ===")
    for idx, iou in enumerate(ious):
        print(f"  Contour {idx}: IoU = {iou*100:.3f}%")
    print(f"  Mean IoU = {mean_iou*100:.3f}%")
    print(f"  Best IoU = {best_iou*100:.3f}% @ epoch {best_epoch}")


if __name__ == '__main__':
    main()
