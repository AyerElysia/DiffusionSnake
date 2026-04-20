#!/usr/bin/env python3
"""
V6b: Scale Conditioning + IoU-Triggered LR Decay.

Two key innovations over V5f:
  1. Scale conditioning: adds log(contour_scale) to DiT's time embedding via
     a zero-initialized MLP. This lets the DiT learn different attention patterns
     for different contour sizes, resolving the structural cause of oscillation.
  2. IoU-triggered decay: constant LR until ALL contours exceed a threshold,
     then rapid cosine decay to lock in the balanced state.

Fine-tunes from V5f checkpoint (91.10%) with scale conditioning zero-initialized.

Usage:
    CUDA_VISIBLE_DEVICES=3 FM_SAVE_DIR=data/outputs/v37v6b_scale \
        python -u scripts/train_v37_gen_v6b.py
"""
import sys, os, time, math, json, copy
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


def ode_inference_raw(denoiser, cnn_feature, i_it_py, c_it_py, py_ind,
                       sampled_feat, h, w, contour_scale, steps=10,
                       deterministic=True):
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


def raw_predict_velocity(denoiser, cnn_feature, i_it_py, c_it_py, sampled_feat,
                          py_ind, x_t, t_tensor, contour_scale=None):
    from lib.utils.snake import snake_gcn_utils, snake_config
    adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, i_it_py.size(1), i_it_py.device)
    t_scaled = t_tensor * 1000.0
    v_pred, L = denoiser(cnn_feature, sampled_feat, x_t, t_scaled, adj,
                          polys=i_it_py, py_ind=py_ind, contour_scale=contour_scale)
    return v_pred, L


def compute_true_iou(pred_disp, data, dr):
    i_it_py_d = data['i_it_py'].double()
    pred_polys = (i_it_py_d + pred_disp.double()).detach().cpu().numpy() * dr
    gt_polys = data['gt_polys_img']
    H_img, W_img = 512, 512
    ious = []
    for idx in range(pred_polys.shape[0]):
        gt_mask = poly_to_mask(gt_polys[idx], H_img, W_img)
        pred_mask = poly_to_mask(pred_polys[idx], H_img, W_img)
        ious.append(compute_iou(pred_mask, gt_mask))
    return ious, float(np.mean(ious))


def main():
    class Args:
        cfg = os.environ.get('CFG_FILE', 'configs/btcv_diffusion_dit_v3_7_gen_single_overfit.yaml')
        epochs = int(os.environ.get('FM_EPOCHS', '10000'))
        lr = float(os.environ.get('FM_LR', '5e-4'))
        eta_min = float(os.environ.get('FM_ETA_MIN', '1e-7'))
        save_dir = os.environ.get('FM_SAVE_DIR', 'data/outputs/v37v6b_scale')
        eval_every = int(os.environ.get('FM_EVAL_EVERY', '100'))
        save_every = int(os.environ.get('FM_SAVE_EVERY', '2000'))
        ode_steps = int(os.environ.get('FM_ODE_STEPS', '10'))
        scope = os.environ.get('FM_SCOPE', 'denoiser')
        ckpt = os.environ.get('CKPT', 'data/outputs/v37v5f_pph_fm/checkpoints/best.pt')
        ema_decay = float(os.environ.get('FM_EMA_DECAY', '0.999'))
        zero_x0_prob = float(os.environ.get('FM_ZERO_X0_PROB', '0.5'))
        t_beta_alpha = float(os.environ.get('FM_T_BETA_ALPHA', '0.3'))
        # IoU trigger: when min(per_contour_iou) > this, start decay
        trigger_min_iou = float(os.environ.get('FM_TRIGGER_MIN_IOU', '0.88'))
        # After trigger: cosine decay over this many additional epochs
        decay_after_trigger = int(os.environ.get('FM_DECAY_EPOCHS', '3000'))
    args = Args()

    if not os.environ.get('CFG_FILE'):
        os.environ['CFG_FILE'] = os.path.join(_THIS_DIR, args.cfg)
    from lib.config import cfg
    cfg.merge_from_file(os.path.join(_THIS_DIR, args.cfg))
    cfg.v3_7_use_regularized_per_point = False
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)

    print(f"=== V6b: Scale Conditioning + IoU-Triggered Decay ===")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  LR: {args.lr} → triggered cosine decay ({args.decay_after_trigger} ep after trigger)")
    print(f"  Max epochs: {args.epochs} | ODE steps: {args.ode_steps}")
    print(f"  Trigger: when min(per_contour_iou) > {args.trigger_min_iou*100:.0f}%")
    print(f"  Zero x_0 probability: {args.zero_x0_prob}")
    print(f"  t ~ Beta({args.t_beta_alpha}, 1.0)")
    print(f"  EMA decay: {args.ema_decay}")
    print(f"  Scope: {args.scope}")
    print(f"  Scale conditioning: ENABLED")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from lib.networks import make_network
    from lib.train.trainers import make_trainer
    from lib.datasets.make_dataset import make_dataset
    from lib.datasets.collate_batch import make_collator
    from lib.datasets.transforms import make_transforms
    from lib.utils.snake import snake_config

    # Enable scale conditioning in config
    cfg.v3_7_use_scale_conditioning = True

    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    model = trainer.network.to(device)

    # Load V5f checkpoint — scale_embed_net will be missing (stays zero-initialized)
    if args.ckpt and os.path.exists(args.ckpt):
        print(f"  Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location='cpu')
        sd = ckpt.get('state_dict') or ckpt.get('model') or ckpt
        from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
        sd = remap_legacy_state_dict(sd)
        wrapper = model.module if hasattr(model, 'module') else model
        info = wrapper.load_state_dict(sd, strict=False)
        loaded = len(sd) - len(info.missing_keys)
        print(f"  Loaded {loaded}/{len(sd)} keys, missing {len(info.missing_keys)} (expected: scale_embed_net)")
        if info.missing_keys:
            print(f"  Missing keys: {info.missing_keys[:10]}")
        loaded_iou = ckpt.get('iou', 0)
        print(f"  Checkpoint IoU: {loaded_iou*100:.2f}%")
    else:
        print(f"  WARNING: No checkpoint, training from scratch!")

    dataset = make_dataset(cfg, cfg.train.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)
    batch = collator([dataset[0]])

    print("[*] Extracting backbone features (cached)...")
    data = extract_fixed_data(model, device, batch)
    N = data['x1_raw'].shape[0]
    print(f"  Contours: {N}, Points: {data['x1_raw'].shape[1]}")
    print(f"  Per-contour scales: {data['contour_scale'].view(-1).tolist()}")

    core = model.net if hasattr(model, 'net') else model
    gcn = core.gcn
    denoiser = gcn.denoiser
    dr = float(snake_config.down_ratio)

    print(f"  FinalLayer type: {type(denoiser.final_layer).__name__}")
    print(f"  Has scale_embed_net: {hasattr(denoiser, 'scale_embed_net')}")
    print(f"  use_scale_conditioning: {denoiser.use_scale_conditioning}")

    x1_norm = data['x1_norm']
    x1_raw = data['x1_raw']
    contour_scale = data['contour_scale']
    cs_flat = contour_scale.view(-1)  # (N,) for passing to denoiser
    noise_scale = float(x1_norm.std()) * 0.5

    # Freeze backbone, train denoiser
    for p in model.parameters():
        p.requires_grad = False
    train_params = []
    for p in denoiser.parameters():
        p.requires_grad = True
        train_params.append(p)

    n_train = sum(p.numel() for p in train_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    scale_params = sum(p.numel() for p in denoiser.scale_embed_net.parameters())
    print(f"  Trainable: {n_train:,} params (scale_embed: {scale_params:,})")

    ema_state = {k: v.clone() for k, v in denoiser.state_dict().items()}

    wd = float(getattr(cfg.train, 'weight_decay', 0.01))
    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=wd, betas=(0.9, 0.999))

    cnn_feature = data['cnn_feature']
    i_it_py = data['i_it_py']
    c_it_py = data['c_it_py']
    py_ind = data['py_ind']
    sampled_feat = data['sampled_feat']
    h_feat, w_feat = data['h'], data['w']

    # Initial eval
    denoiser.eval()
    with torch.no_grad():
        disp = ode_inference_raw(denoiser, cnn_feature, i_it_py, c_it_py,
                                  py_ind, sampled_feat, h_feat, w_feat,
                                  contour_scale=contour_scale, steps=args.ode_steps)
        init_ious, init_mean = compute_true_iou(disp, data, dr)
    per_str = ' '.join([f'C{i}={v*100:.1f}%' for i, v in enumerate(init_ious)])
    print(f"  Initial IoU: {init_mean*100:.2f}% | {per_str}")

    from torch.distributions import Beta
    t_dist = Beta(args.t_beta_alpha, 1.0)

    log_path = os.path.join(args.save_dir, 'train_log.jsonl')
    best_iou = 0.0
    best_epoch = 0
    best_ema_iou = 0.0
    best_ema_epoch = 0

    # Trigger state
    triggered = False
    trigger_epoch = -1
    total_after_trigger = args.decay_after_trigger
    lr_base = args.lr

    print(f"\n[*] Starting V6b training ({args.epochs} max epochs, trigger={args.trigger_min_iou*100:.0f}%)...\n")
    t_start = time.time()

    for epoch in range(args.epochs):
        # LR scheduling: constant until trigger, then cosine decay
        if triggered:
            progress_after = (epoch - trigger_epoch) / max(total_after_trigger, 1)
            progress_after = min(progress_after, 1.0)
            lr = args.eta_min + 0.5 * (lr_base - args.eta_min) * (1 + math.cos(math.pi * progress_after))
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            # Stop if decay is complete
            if epoch - trigger_epoch >= total_after_trigger:
                print(f"\n  [TRIGGER] Decay complete at epoch {epoch}. Stopping.")
                break
        else:
            lr = args.lr

        denoiser.train()
        optimizer.zero_grad()

        # FM velocity loss
        t = t_dist.sample((N,)).to(device).view(N, 1, 1)

        if torch.rand(1).item() < args.zero_x0_prob:
            x0 = torch.zeros_like(x1_norm)
        else:
            x0 = torch.randn_like(x1_norm) * noise_scale

        x_t = (1.0 - t) * x0 + t * x1_norm
        v_target = x1_norm - x0

        v_pred, _ = raw_predict_velocity(
            denoiser, cnn_feature, i_it_py, c_it_py, sampled_feat,
            py_ind, x_t, t.view(-1), contour_scale=cs_flat
        )
        fm_loss = F.mse_loss(v_pred, v_target)
        fm_loss.backward()

        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optimizer.step()

        # Update EMA
        with torch.no_grad():
            for name, param in denoiser.named_parameters():
                if name in ema_state:
                    ema_state[name].mul_(args.ema_decay).add_(param.data, alpha=1 - args.ema_decay)

        log_entry = {'epoch': epoch, 'fm_loss': fm_loss.item(), 'lr': lr, 'triggered': triggered}

        # Periodic evaluation
        if (epoch + 1) % args.eval_every == 0 or epoch == 0:
            denoiser.eval()
            with torch.no_grad():
                disp = ode_inference_raw(denoiser, cnn_feature, i_it_py, c_it_py,
                                          py_ind, sampled_feat, h_feat, w_feat,
                                          contour_scale=contour_scale, steps=args.ode_steps)
                ious, mean_iou = compute_true_iou(disp, data, dr)

                orig_state = {k: v.clone() for k, v in denoiser.state_dict().items()}
                denoiser.load_state_dict(ema_state)
                disp_ema = ode_inference_raw(denoiser, cnn_feature, i_it_py, c_it_py,
                                              py_ind, sampled_feat, h_feat, w_feat,
                                              contour_scale=contour_scale, steps=args.ode_steps)
                ious_ema, mean_iou_ema = compute_true_iou(disp_ema, data, dr)
                denoiser.load_state_dict(orig_state)

            log_entry['mean_iou'] = mean_iou
            log_entry['per_iou'] = ious
            log_entry['ema_iou'] = mean_iou_ema

            if mean_iou > best_iou:
                best_iou = mean_iou
                best_epoch = epoch
                sd = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save({
                    'state_dict': sd, 'epoch': epoch, 'iou': mean_iou,
                    'raw_mode': True, 'contour_norm': True, 'per_point_head': True,
                    'scale_conditioning': True,
                }, os.path.join(args.save_dir, 'checkpoints', 'best.pt'))

            if mean_iou_ema > best_ema_iou:
                best_ema_iou = mean_iou_ema
                best_ema_epoch = epoch
                ema_sd = copy.deepcopy(model.state_dict())
                for name in ema_state:
                    full_key = None
                    for k in ema_sd:
                        if k.endswith(name) or k.endswith('.' + name):
                            full_key = k
                            break
                    if full_key:
                        ema_sd[full_key] = ema_state[name].cpu()
                torch.save({
                    'state_dict': ema_sd, 'epoch': epoch, 'iou': mean_iou_ema,
                    'ema': True, 'raw_mode': True, 'contour_norm': True,
                    'per_point_head': True, 'scale_conditioning': True,
                }, os.path.join(args.save_dir, 'checkpoints', 'best_ema.pt'))

            # Check trigger condition
            min_contour_iou = min(ious)
            if not triggered and min_contour_iou > args.trigger_min_iou:
                triggered = True
                trigger_epoch = epoch
                lr_base = optimizer.param_groups[0]['lr']
                print(f"\n  *** TRIGGERED at epoch {epoch}! min(IoU)={min_contour_iou*100:.1f}% > {args.trigger_min_iou*100:.0f}%")
                print(f"  *** Starting cosine decay from LR={lr_base:.2e} over {total_after_trigger} epochs")
                print(f"  *** Expected completion at epoch {epoch + total_after_trigger}\n")

            elapsed = time.time() - t_start
            per_iou_str = ' '.join([f'C{i}={v*100:.1f}%' for i, v in enumerate(ious)])
            trigger_str = f" [DECAY ep {epoch-trigger_epoch}/{total_after_trigger}]" if triggered else ""
            print(f"  Ep {epoch:5d} | fm={fm_loss.item():.4e} | lr={lr:.2e} | "
                  f"IoU={mean_iou*100:.2f}% EMA={mean_iou_ema*100:.2f}% "
                  f"(best={best_iou*100:.2f}%@{best_epoch} ema={best_ema_iou*100:.2f}%@{best_ema_epoch}) "
                  f"| {per_iou_str}{trigger_str} | {elapsed:.0f}s")

        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        if (epoch + 1) % args.save_every == 0:
            sd = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({
                'state_dict': sd, 'epoch': epoch,
                'raw_mode': True, 'contour_norm': True, 'per_point_head': True,
                'scale_conditioning': True,
            }, os.path.join(args.save_dir, 'checkpoints', f'epoch_{epoch+1}.pt'))

    # Final evaluation
    denoiser.eval()
    print(f"\n=== Final Evaluation ===")
    with torch.no_grad():
        for label, use_ema in [("Current", False), ("EMA", True)]:
            if use_ema:
                denoiser.load_state_dict(ema_state)
            for steps in [5, 10, 20]:
                disp = ode_inference_raw(denoiser, cnn_feature, i_it_py, c_it_py,
                                          py_ind, sampled_feat, h_feat, w_feat,
                                          contour_scale=contour_scale, steps=steps)
                ious, mean_iou = compute_true_iou(disp, data, dr)
                per_str = ' '.join([f'C{i}={v*100:.1f}%' for i, v in enumerate(ious)])
                rmse = (disp - x1_raw).pow(2).mean().sqrt().item()
                print(f"  [{label}] {steps:3d} steps: IoU={mean_iou*100:.2f}% RMSE={rmse:.4f} | {per_str}")

    print(f"\n  Best IoU = {best_iou*100:.3f}% @ epoch {best_epoch}")
    print(f"  Best EMA IoU = {best_ema_iou*100:.3f}% @ epoch {best_ema_epoch}")
    print(f"  Initial IoU = {init_mean*100:.2f}%")
    if triggered:
        print(f"  Trigger fired at epoch {trigger_epoch}")


if __name__ == '__main__':
    main()
