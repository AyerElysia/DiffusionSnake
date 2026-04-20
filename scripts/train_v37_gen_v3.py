#!/usr/bin/env python3
"""
V3 of FM training for V3.7-gen denoiser.
Key innovations over V2:
  1. Mixed x_0: alternate between random noise and x_0=0 (deterministic)
     - Ensures model learns both noisy and clean starting trajectories
     - Prevents overfitting to one distribution
  2. Trajectory loss: periodically run full ODE from x_0=0, optimize endpoint
     - Directly optimizes what we evaluate
     - Gradient flows through entire ODE trajectory
  3. Stable LR: lower constant (2e-4) with cosine tail
  4. EMA (exponential moving average) of model weights for stable evaluation

Usage:
    CUDA_VISIBLE_DEVICES=0 FM_NOISE_SCALE=0.1 FM_EPOCHS=20000 FM_SAVE_DIR=data/outputs/v37v3 \
        python -u scripts/train_v37_gen_v3.py
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
                  steps=10, noise_scale=0.1, deterministic=True):
    device = i_it_py.device
    N = i_it_py.size(0)
    if deterministic:
        x_t = torch.zeros(N, i_it_py.size(1), 2, device=device)
    else:
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


def ode_inference_differentiable(gcn, cnn_feature, i_it_py, c_it_py, py_ind,
                                  sampled_feat, steps=5):
    """Differentiable ODE from x_0=0 for trajectory loss."""
    device = i_it_py.device
    N = i_it_py.size(0)
    x_t = torch.zeros(N, i_it_py.size(1), 2, device=device, requires_grad=False)
    dt = 1.0 / steps
    for i in range(steps):
        t_val = i * dt
        t_tensor = torch.full((N,), t_val, device=device, dtype=torch.float32)
        v_pred, _ = gcn.predict_velocity(
            cnn_feature, i_it_py, c_it_py, sampled_feat, py_ind, x_t, t_tensor
        )
        x_t = x_t + v_pred * dt
    return x_t  # normalized displacement


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


class ConstantCosineScheduler:
    def __init__(self, optimizer, total_epochs, constant_frac=0.5, eta_min=1e-7):
        self.optimizer = optimizer
        self.total = total_epochs
        self.const_end = int(total_epochs * constant_frac)
        self.decay_epochs = total_epochs - self.const_end
        self.base_lr = optimizer.param_groups[0]['lr']
        self.eta_min = eta_min
        self._step = 0

    def step(self):
        self._step += 1
        if self._step <= self.const_end:
            lr = self.base_lr
        else:
            progress = (self._step - self.const_end) / max(self.decay_epochs, 1)
            lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


def main():
    class Args:
        cfg = os.environ.get('CFG_FILE', 'configs/btcv_diffusion_dit_v3_7_gen_single_overfit.yaml')
        epochs = int(os.environ.get('FM_EPOCHS', '20000'))
        lr = float(os.environ.get('FM_LR', '2e-4'))
        eta_min = float(os.environ.get('FM_ETA_MIN', '1e-7'))
        save_dir = os.environ.get('FM_SAVE_DIR', 'data/outputs/v37v3')
        eval_every = int(os.environ.get('FM_EVAL_EVERY', '300'))
        save_every = int(os.environ.get('FM_SAVE_EVERY', '5000'))
        ode_steps = int(os.environ.get('FM_ODE_STEPS', '10'))
        noise_scale = float(os.environ.get('FM_NOISE_SCALE', '0.1'))
        scope = os.environ.get('FM_SCOPE', 'denoiser')
        ckpt = os.environ.get('CKPT', '')
        # V3 params
        constant_frac = float(os.environ.get('FM_CONST_FRAC', '0.5'))
        delta_reg_weight = float(os.environ.get('FM_DELTA_REG', '0.001'))
        lap_weight = float(os.environ.get('FM_LAP_WEIGHT', '0.01'))
        traj_every = int(os.environ.get('FM_TRAJ_EVERY', '5'))  # trajectory loss every N steps
        traj_weight = float(os.environ.get('FM_TRAJ_WEIGHT', '1.0'))
        traj_steps = int(os.environ.get('FM_TRAJ_STEPS', '5'))  # ODE steps for trajectory loss
        zero_x0_prob = float(os.environ.get('FM_ZERO_X0_PROB', '0.5'))  # prob of x_0=0
        ema_decay = float(os.environ.get('FM_EMA_DECAY', '0.999'))
    args = Args()

    if not os.environ.get('CFG_FILE'):
        os.environ['CFG_FILE'] = os.path.join(_THIS_DIR, args.cfg)
    from lib.config import cfg
    cfg.merge_from_file(os.path.join(_THIS_DIR, args.cfg))
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)

    print(f"=== V3.7-gen FM Training V3 (Trajectory-Aware) ===")
    print(f"  Config: {args.cfg}")
    print(f"  LR: {args.lr} (constant {args.constant_frac*100:.0f}% → cosine → {args.eta_min})")
    print(f"  Epochs: {args.epochs} | ODE steps: {args.ode_steps}")
    print(f"  Noise scale: {args.noise_scale}")
    print(f"  Zero x_0 probability: {args.zero_x0_prob}")
    print(f"  Trajectory loss: every {args.traj_every} steps, weight={args.traj_weight}, "
          f"ODE={args.traj_steps} steps")
    print(f"  Delta reg: {args.delta_reg_weight} | Lap weight: {args.lap_weight}")
    print(f"  EMA decay: {args.ema_decay}")
    print(f"  Scope: {args.scope}")

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

    if args.ckpt and os.path.exists(args.ckpt):
        print(f"  Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location='cpu')
        sd = ckpt.get('state_dict') or ckpt.get('model') or ckpt
        from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
        sd = remap_legacy_state_dict(sd)
        wrapper = model.module if hasattr(model, 'module') else model
        info = wrapper.load_state_dict(sd, strict=False)
        print(f"  Loaded {len(sd)-len(info.missing_keys)}/{len(sd)} keys")

    dataset = make_dataset(cfg, cfg.train.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)
    batch = collator([dataset[0]])

    print("[*] Extracting backbone features (cached)...")
    data = extract_fixed_data(model, device, batch)
    print(f"  Feature shape: {data['cnn_feature'].shape}")
    print(f"  Contours: {data['x1'].shape[0]}, Points: {data['x1'].shape[1]}")
    print(f"  GT disp range: [{data['x1'].min():.4f}, {data['x1'].max():.4f}]")

    core = model.net if hasattr(model, 'net') else model
    gcn = core.gcn
    denoiser = gcn.denoiser
    dr = float(snake_config.down_ratio)

    if hasattr(denoiser, '_delta_reg_weight'):
        denoiser._delta_reg_weight = args.delta_reg_weight
    if hasattr(denoiser, '_laplacian_weight'):
        denoiser._laplacian_weight = args.lap_weight

    noise_scale = args.noise_scale

    # Freeze backbone
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
    else:
        for p in denoiser.parameters():
            p.requires_grad = True
            train_params.append(p)

    n_train = sum(p.numel() for p in train_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable: {n_train:,} params | Frozen: {n_frozen:,} params")

    # EMA model
    ema_state = {k: v.clone() for k, v in denoiser.state_dict().items()}

    wd = float(getattr(cfg.train, 'weight_decay', 0.01))
    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=wd, betas=(0.9, 0.999))
    scheduler = ConstantCosineScheduler(optimizer, args.epochs, args.constant_frac, args.eta_min)

    cnn_feature = data['cnn_feature']
    i_it_py = data['i_it_py']
    c_it_py = data['c_it_py']
    py_ind = data['py_ind']
    sampled_feat = data['sampled_feat']
    x1 = data['x1']
    N = x1.size(0)

    log_path = os.path.join(args.save_dir, 'train_log.jsonl')
    best_iou = 0.0
    best_epoch = 0
    best_ema_iou = 0.0
    best_ema_epoch = 0

    print(f"\n[*] Starting V3 FM training ({args.epochs} epochs)...\n")
    t_start = time.time()

    for epoch in range(args.epochs):
        denoiser.train()

        # --- FM velocity loss with mixed x_0 ---
        t = torch.rand(N, device=device).view(N, 1, 1)

        # Mixed starting point: sometimes x_0=0, sometimes random noise
        if torch.rand(1).item() < args.zero_x0_prob:
            x0 = torch.zeros_like(x1)
        else:
            x0 = torch.randn_like(x1) * noise_scale

        x_t = (1.0 - t) * x0 + t * x1
        v_target = x1 - x0

        v_pred, L_reg = gcn.predict_velocity(
            cnn_feature, i_it_py, c_it_py, sampled_feat, py_ind,
            x_t, t.view(-1)
        )
        fm_loss = F.mse_loss(v_pred, v_target)

        # --- Trajectory loss (periodic) ---
        traj_loss = torch.tensor(0.0, device=device)
        if epoch % args.traj_every == 0:
            x_pred = ode_inference_differentiable(
                gcn, cnn_feature, i_it_py, c_it_py, py_ind,
                sampled_feat, steps=args.traj_steps
            )
            traj_loss = F.mse_loss(x_pred, x1) * args.traj_weight

        loss = fm_loss + L_reg + traj_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optimizer.step()
        scheduler.step()

        # Update EMA
        with torch.no_grad():
            for name, param in denoiser.named_parameters():
                if name in ema_state:
                    ema_state[name].mul_(args.ema_decay).add_(param.data, alpha=1 - args.ema_decay)

        lr = optimizer.param_groups[0]['lr']
        log_entry = {
            'epoch': epoch,
            'loss': loss.item(),
            'fm_loss': fm_loss.item(),
            'traj_loss': traj_loss.item() if isinstance(traj_loss, torch.Tensor) else 0.0,
            'reg': L_reg.item() if isinstance(L_reg, torch.Tensor) else float(L_reg),
            'lr': lr
        }

        # Periodic evaluation
        if (epoch + 1) % args.eval_every == 0 or epoch == 0:
            denoiser.eval()
            with torch.no_grad():
                # Eval current model
                disp = ode_inference(gcn, cnn_feature, i_it_py, c_it_py, py_ind,
                                     sampled_feat, steps=args.ode_steps,
                                     noise_scale=noise_scale, deterministic=True)
                ious, mean_iou = compute_iou_from_pred(disp, data, dr)

                # Eval EMA model
                orig_state = {k: v.clone() for k, v in denoiser.state_dict().items()}
                denoiser.load_state_dict(ema_state)
                disp_ema = ode_inference(gcn, cnn_feature, i_it_py, c_it_py, py_ind,
                                         sampled_feat, steps=args.ode_steps,
                                         noise_scale=noise_scale, deterministic=True)
                ious_ema, mean_iou_ema = compute_iou_from_pred(disp_ema, data, dr)
                denoiser.load_state_dict(orig_state)

            log_entry['mean_iou'] = mean_iou
            log_entry['per_iou'] = ious
            log_entry['ema_iou'] = mean_iou_ema

            if mean_iou > best_iou:
                best_iou = mean_iou
                best_epoch = epoch
                sd = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save({
                    'state_dict': sd, 'epoch': epoch,
                    'loss': loss.item(), 'iou': mean_iou,
                }, os.path.join(args.save_dir, 'checkpoints', 'best.pt'))

            if mean_iou_ema > best_ema_iou:
                best_ema_iou = mean_iou_ema
                best_ema_epoch = epoch
                # Save EMA state into model, then save
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
                    'state_dict': ema_sd, 'epoch': epoch,
                    'loss': loss.item(), 'iou': mean_iou_ema,
                    'ema': True,
                }, os.path.join(args.save_dir, 'checkpoints', 'best_ema.pt'))

            elapsed = time.time() - t_start
            per_iou_str = ' '.join([f'C{i}={v*100:.1f}%' for i, v in enumerate(ious)])
            traj_str = f" traj={traj_loss.item():.4e}" if traj_loss.item() > 0 else ""
            print(f"  Ep {epoch:5d} | loss={loss.item():.4e} fm={fm_loss.item():.4e} "
                  f"reg={log_entry['reg']:.4e}{traj_str} | lr={lr:.2e} | "
                  f"IoU={mean_iou*100:.2f}% EMA={mean_iou_ema*100:.2f}% "
                  f"(best={best_iou*100:.2f}%@{best_epoch} ema={best_ema_iou*100:.2f}%@{best_ema_epoch}) | "
                  f"{per_iou_str} | {elapsed:.0f}s")

        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        if (epoch + 1) % args.save_every == 0:
            sd = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({
                'state_dict': sd, 'epoch': epoch, 'loss': loss.item(),
            }, os.path.join(args.save_dir, 'checkpoints', f'epoch_{epoch+1}.pt'))

    # Final evaluation
    denoiser.eval()
    print(f"\n=== Final Evaluation ===")
    with torch.no_grad():
        for label, use_ema in [("Current", False), ("EMA", True)]:
            if use_ema:
                denoiser.load_state_dict(ema_state)
            for steps in [5, 10, 20]:
                disp = ode_inference(gcn, cnn_feature, i_it_py, c_it_py, py_ind,
                                     sampled_feat, steps=steps, noise_scale=noise_scale,
                                     deterministic=True)
                ious, mean_iou = compute_iou_from_pred(disp, data, dr)
                per_str = ' '.join([f'C{i}={v*100:.1f}%' for i, v in enumerate(ious)])
                print(f"  [{label}] Det {steps:3d} steps: IoU={mean_iou*100:.2f}% | {per_str}")

            # Stochastic 20-sample
            all_ious = []
            for _ in range(20):
                disp = ode_inference(gcn, cnn_feature, i_it_py, c_it_py, py_ind,
                                     sampled_feat, steps=10, noise_scale=noise_scale,
                                     deterministic=False)
                ious_i, _ = compute_iou_from_pred(disp, data, dr)
                all_ious.append(ious_i)
            ious = [float(np.mean([all_ious[r][c] for r in range(20)])) for c in range(N)]
            mean_iou = float(np.mean(ious))
            per_str = ' '.join([f'C{i}={v*100:.1f}%' for i, v in enumerate(ious)])
            print(f"  [{label}] Stoch 20-avg: IoU={mean_iou*100:.2f}% | {per_str}")

    print(f"\n  Best IoU = {best_iou*100:.3f}% @ epoch {best_epoch}")
    print(f"  Best EMA IoU = {best_ema_iou*100:.3f}% @ epoch {best_ema_epoch}")


if __name__ == '__main__':
    main()
