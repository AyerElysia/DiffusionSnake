#!/usr/bin/env python3
"""
Standalone fine-tuning script for denoiser precision optimization.

Uses EVAL-mode features (frozen backbone + BN statistics) to match
inference conditions exactly. Only trains the denoiser module.

This avoids the training pipeline's BN-in-train-mode mismatch that
causes loss discrepancies between training and inference.

Usage:
    CUDA_VISIBLE_DEVICES=2 CFG_FILE=configs/btcv_diffusion_dit_v3_7_9_single_overfit.yaml \
        python scripts/finetune_denoiser.py \
            --ckpt data/outputs/btcv_diffusion_dit_v3_7_9_single_overfit/checkpoints/epoch_700.pt \
            --lr 1e-5 --epochs 5000 --save_dir data/outputs/btcv_diffusion_dit_v3_7_12_finetune
"""
import sys, os, argparse, json, time, math
import numpy as np
import cv2
import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _THIS_DIR)

_DEFAULT_CFG = os.path.join(_THIS_DIR, 'configs', 'btcv_diffusion_dit_v3_7_9_single_overfit.yaml')
if not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = _DEFAULT_CFG

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


def load_model(ckpt_path):
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_obj = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt_obj.get('state_dict') or ckpt_obj.get('model') or ckpt_obj

    from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
    sd = remap_legacy_state_dict(sd)

    wrapper = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    
    # For V3.7 with PerPointFinalLayer: handle two cases:
    # Case A: Checkpoint has shared FinalLayer (linear.weight/bias) → load into _shared_final_layer
    # Case B: Checkpoint already has per_point_weight/bias → load directly, skip _shared_final_layer
    gcn = wrapper.net.gcn if hasattr(wrapper.net, 'gcn') else None
    denoiser = gcn.denoiser if gcn else None
    skip_per_point_init = False
    if denoiser and hasattr(denoiser, '_shared_final_layer'):
        fl_prefix = 'net.gcn.denoiser.final_layer.'
        fl_keys = {k: v for k, v in sd.items() if k.startswith(fl_prefix)}
        if fl_keys:
            # Check if checkpoint has per_point keys (Case B)
            has_per_point = any('per_point_weight' in k for k in fl_keys)
            if has_per_point:
                print(f"[✔] Checkpoint has per-point head weights — loading directly")
                skip_per_point_init = True
                # Convert float32 per-point weights to float64 if model expects float64
                if hasattr(denoiser.final_layer, 'use_float64') and denoiser.final_layer.use_float64:
                    for k in list(sd.keys()):
                        if 'per_point_weight' in k or 'per_point_bias' in k:
                            if sd[k].dtype == torch.float32:
                                sd[k] = sd[k].double()
                                print(f"  Promoted {k} to float64")
            else:
                # Case A: shared FinalLayer → load into _shared_final_layer for later init
                fl_sd = {k.replace(fl_prefix, ''): v for k, v in fl_keys.items()}
                denoiser._shared_final_layer.load_state_dict(fl_sd, strict=True)
                print(f"[✔] Loaded shared FinalLayer weights for V3.7 init ({len(fl_sd)} keys)")
    
    info = wrapper.load_state_dict(sd, strict=False)
    n_loaded = len(sd) - len(info.missing_keys)
    print(f"[✔] Loaded checkpoint: {n_loaded}/{len(sd)} keys"
          f" (missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)})")
    
    # Store flag so we can skip init_per_point_from_checkpoint later
    if denoiser:
        denoiser._skip_per_point_init = skip_per_point_init
    
    return trainer.network.to(device), device


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
    """Extract all fixed data (features, init contours, GT) in eval mode.
    
    Returns dict with everything needed for denoiser training.
    """
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    dr = float(snake_config.down_ratio)
    core = model.net if hasattr(model, 'net') else model

    model.eval()
    with torch.no_grad():
        # Feature extraction (eval mode for correct BN)
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

        # Compute GT displacement (same as training pipeline)
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

        # Point alignment (rotate GT to best match init)
        d2 = (i_it_py[:, :1, :] - i_gt_py).pow(2).sum(-1)
        i_gt_py = torch.stack([
            torch.roll(i_gt_py[i], -int(d2[i].argmin().item()), 0)
            for i in range(i_gt_py.size(0))
        ], 0)

        x1_raw = i_gt_py - i_it_py  # GT displacement

        # Normalize displacement (same as FlowMatchingEvolution.normalize_disp)
        gcn = core.gcn
        x1 = gcn.normalize_disp(x1_raw)

    return {
        'cnn_feature': cnn_feature.detach(),
        'i_it_py': i_it_py.detach(),
        'c_it_py': c_it_py.detach(),
        'py_ind': py_ind.detach(),
        'sampled_feat': sampled_feat.detach(),
        'x1': x1.detach(),  # normalized GT displacement
        'x1_raw': x1_raw.detach(),
        'i_gt_py': i_gt_py.detach(),
        'gt_polys_img': gt_all.view(-1, P, 2).cpu().numpy() * dr,
        'h': h, 'w': w,
        'batch': batch,
    }


def compute_iou_from_pred(pred_disp, data, dr):
    """Compute IoU from predicted displacement.
    Uses float64 polygon arithmetic to avoid float32 rounding at .5 boundaries.
    Both pred and GT use the same float64 addition path for consistency.
    """
    # Use float64 for precise polygon coordinate computation
    i_it_py_d = data['i_it_py'].double()
    pred_polys = (i_it_py_d + pred_disp.double()).detach().cpu().numpy() * dr
    
    # GT: use same float64 path as prediction for consistent rounding
    x1_key = 'x1'  # In RAW mode this is x1_raw
    gt_polys = (i_it_py_d + data[x1_key].double()).detach().cpu().numpy() * dr
    
    H_img, W_img = 512, 512
    ious = []
    for idx in range(pred_polys.shape[0]):
        gt_mask = poly_to_mask(gt_polys[idx], H_img, W_img)
        pred_mask = poly_to_mask(pred_polys[idx], H_img, W_img)
        ious.append(compute_iou(pred_mask, gt_mask))
    return ious, float(np.mean(ious))


def main():
    # Use env vars to avoid lib.config argparse conflict
    class Args:
        ckpt = os.environ.get('CKPT', '')
        lr = float(os.environ.get('FT_LR', '1e-5'))
        eta_min = float(os.environ.get('FT_ETA_MIN', '1e-8'))
        epochs = int(os.environ.get('FT_EPOCHS', '5000'))
        save_dir = os.environ.get('FT_SAVE_DIR', 'data/outputs/btcv_diffusion_dit_v3_7_12_finetune')
        eval_every = int(os.environ.get('FT_EVAL_EVERY', '100'))
        save_every = int(os.environ.get('FT_SAVE_EVERY', '500'))
        mode = os.environ.get('FT_MODE', 'denoiser')  # 'denoiser' or 'residual'
        residual_lr = float(os.environ.get('FT_RESIDUAL_LR', '0.01'))
    args = Args()

    print(f"=== Denoiser Fine-Tuning ===")
    print(f"  Mode: {args.mode}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  LR: {args.lr} → {args.eta_min} (cosine)")
    print(f"  Epochs: {args.epochs}")
    print(f"  Save dir: {args.save_dir}")

    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)

    # Load model
    model, device = load_model(args.ckpt)
    core = model.net if hasattr(model, 'net') else model
    gcn = core.gcn  # FlowMatchingEvolution module

    # Load dataset and extract fixed data
    dataset = make_dataset(cfg, cfg.train.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)
    batch = collator([dataset[0]])
    
    print("[*] Extracting fixed features (eval mode)...")
    data = extract_fixed_data(model, device, batch)
    print(f"  Feature shape: {data['cnn_feature'].shape}")
    print(f"  Contours: {data['x1'].shape[0]}, Points: {data['x1'].shape[1]}")
    print(f"  GT disp range: [{data['x1'].min():.4f}, {data['x1'].max():.4f}]")
    print(f"  GT raw disp range: [{data['x1_raw'].min():.4f}, {data['x1_raw'].max():.4f}]")

    # RAW mode: bypass normalization for exact polygon reconstruction
    use_raw = os.environ.get('FT_RAW', '0') == '1'
    
    # V3.7: Initialize per-point heads from loaded shared FinalLayer weights
    # Skip if checkpoint already had per-point weights (they were loaded directly)
    denoiser = gcn.denoiser
    if hasattr(denoiser, 'init_per_point_from_checkpoint') and not getattr(denoiser, '_skip_per_point_init', False):
        print("\n  [V3.7] Initializing per-point heads from shared FinalLayer...")
        denoiser.init_per_point_from_checkpoint()
    elif getattr(denoiser, '_skip_per_point_init', False):
        print("\n  [V3.7] Skipping per-point init — checkpoint already has per-point weights")
    
    # Optional: convert ENTIRE denoiser to float64 for maximum precision
    # This ensures all hidden states (not just the final layer) are computed in float64,
    # eliminating float32 accumulation errors that limit coordinate precision.
    use_f64_denoiser = os.environ.get('FT_F64_DENOISER', '0') == '1'
    if use_f64_denoiser:
        print("\n  [F64 DENOISER] Converting entire denoiser to float64...")
        denoiser.double()
        n_f64 = sum(1 for p in denoiser.parameters() if p.dtype == torch.float64)
        n_total = sum(1 for p in denoiser.parameters())
        print(f"  {n_f64}/{n_total} parameters now float64")
    
    # Determine if the checkpoint was already RAW-transformed
    # (per-point checkpoints from RAW training are already in raw space)
    ckpt_already_raw = getattr(denoiser, '_skip_per_point_init', False) and use_raw
    
    if use_raw and gcn._has_disp_stats() and not ckpt_already_raw:
        print("\n  [RAW MODE] Transforming model to predict raw displacement...")
        disp_min = gcn._disp_min.squeeze()  # (2,)
        disp_max = gcn._disp_max.squeeze()  # (2,)
        scale = (disp_max - disp_min) / 2.0  # (2,)
        offset = (disp_max + disp_min) / 2.0  # (2,)
        
        fl = denoiser.final_layer
        # Transform FinalLayer weights depending on type
        if hasattr(fl, 'per_point_weight'):
            # PerPointFinalLayer: transform each point's weight and bias
            with torch.no_grad():
                dev = fl.per_point_weight.device
                dtype = fl.per_point_weight.dtype  # float32 or float64
                s = scale.to(dev, dtype=dtype)  # (2,)
                o = offset.to(dev, dtype=dtype)  # (2,)
                # weight: (P, 2, D) — multiply each output dim by its scale
                fl.per_point_weight.data *= s.unsqueeze(0).unsqueeze(-1)
                # bias: (P, 2) — bias * scale + offset
                fl.per_point_bias.data = fl.per_point_bias.data * s.unsqueeze(0) + o.unsqueeze(0)
            print(f"  Transformed PerPointFinalLayer weights ({fl.per_point_weight.shape[0]} points, dtype={dtype})")
        elif hasattr(fl, 'linear') and hasattr(fl.linear, 'weight'):
            with torch.no_grad():
                fl.linear.weight.data *= scale.unsqueeze(1).to(fl.linear.weight.device)
                fl.linear.bias.data = fl.linear.bias.data * scale.to(fl.linear.bias.device) + offset.to(fl.linear.bias.device)
            print(f"  Transformed FinalLayer.linear weights")
        elif hasattr(fl, 'mlp'):
            last_layer = fl.mlp[-1]
            with torch.no_grad():
                last_layer.weight.data *= scale.unsqueeze(1).to(last_layer.weight.device)
                last_layer.bias.data = last_layer.bias.data * scale.to(last_layer.bias.device) + offset.to(last_layer.bias.device)
            print(f"  Transformed MLPOutputHead last layer weights")
        else:
            print(f"  WARNING: Unknown final_layer type {type(fl)}, skipping weight transform")
        
        # Disable normalization so sample_disp skips denormalize
        gcn._disp_min = None
        gcn._disp_max = None
        
        # Switch training target to raw displacement
        data['x1'] = data['x1_raw']
        
        print(f"  Scale: {scale.cpu().numpy()}")
        print(f"  Offset: {offset.cpu().numpy()}")
        print(f"  New target range: [{data['x1'].min():.4f}, {data['x1'].max():.4f}]")
        
        # Also update gt_polys_img to use ALIGNED GT (exact match for 100% IoU ceiling)
        data['gt_polys_img'] = data['i_gt_py'].cpu().numpy() * float(snake_config.down_ratio)
    elif use_raw and ckpt_already_raw:
        print("\n  [RAW MODE] Checkpoint already RAW-transformed — skipping weight transform")
        # Just disable normalization and set target to raw
        gcn._disp_min = None
        gcn._disp_max = None
        data['x1'] = data['x1_raw']
        data['gt_polys_img'] = data['i_gt_py'].cpu().numpy() * float(snake_config.down_ratio)
        print(f"  Target range: [{data['x1'].min():.4f}, {data['x1'].max():.4f}]")
    elif use_raw and not gcn._has_disp_stats():
        print("\n  [RAW MODE] No disp stats — treating as already-raw model")
        data['x1'] = data['x1_raw']
        data['gt_polys_img'] = data['i_gt_py'].cpu().numpy() * float(snake_config.down_ratio)

    # Freeze everything except denoiser
    for p in model.parameters():
        p.requires_grad = False
    
    denoiser = gcn.denoiser
    
    # Determine which parameters to train
    train_scope = os.environ.get('FT_SCOPE', 'denoiser')  # 'denoiser', 'final', 'all', 'pt_id'
    denoiser_params = []
    if train_scope == 'final':
        # Only train final_layer (514 params) — most direct path
        for p in denoiser.final_layer.parameters():
            p.requires_grad = True
            denoiser_params.append(p)
    elif train_scope == 'per_point':
        # Train ONLY per-point heads in final_layer (per_point_weight + per_point_bias)
        fl = denoiser.final_layer
        for name in ['per_point_weight', 'per_point_bias']:
            p = getattr(fl, name, None)
            if p is not None:
                p.requires_grad = True
                denoiser_params.append(p)
    elif train_scope == 'pt_id':
        # Train point embeddings + final_layer (V3.7 specific)
        for name in ['point_idx_embed', 'point_idx_embed_in', 'point_idx_embed_out']:
            mod = getattr(denoiser, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True
                    denoiser_params.append(p)
        for p in denoiser.final_layer.parameters():
            p.requires_grad = True
            denoiser_params.append(p)
    elif train_scope == 'all':
        # Train entire model including backbone
        for p in model.parameters():
            p.requires_grad = True
            denoiser_params.append(p)
    else:
        for p in denoiser.parameters():
            p.requires_grad = True
            denoiser_params.append(p)
    
    print(f"  Train scope: {train_scope}")
    print(f"  Trainable params: {sum(p.numel() for p in denoiser_params):,}")
    print(f"  Frozen params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

    # Optimizer setup
    optim_type = os.environ.get('FT_OPTIM', 'adam')  # 'adam', 'lbfgs', 'sgd'

    # Pre-compute fixed inputs for denoiser
    cnn_feature = data['cnn_feature']
    i_it_py = data['i_it_py']
    c_it_py = data['c_it_py']
    py_ind = data['py_ind']
    sampled_feat = data['sampled_feat']
    x1 = data['x1']
    h, w = data['h'], data['w']
    N = x1.size(0)
    dr = float(snake_config.down_ratio)

    # Fixed t=0 and x_t=0 for pure direct regression
    t_zeros = torch.zeros(N, device=device)
    x_t = torch.zeros_like(x1)

    # Mode-specific setup
    residual = None
    base_pred = None
    scheduler = None

    if args.mode == 'residual':
        # Compute base prediction (frozen denoiser)
        denoiser.eval()
        with torch.no_grad():
            base_pred, _ = gcn.predict_velocity(
                cnn_feature, i_it_py, c_it_py, sampled_feat, py_ind, x_t, t_zeros
            )
            base_pred = base_pred.detach()
        
        # Freeze denoiser too, only train residual
        for p in denoiser_params:
            p.requires_grad = False
        
        # Residual initialized to zero (will converge to optimal correction)
        # Use float64 for residual to break float32 precision barriers
        residual = torch.nn.Parameter(torch.zeros(N, x1.size(1), 2, device=device, dtype=torch.float64))
        
        # Use Adam for residual (SGD is too slow for tiny gradients)
        residual_optim = os.environ.get('FT_OPTIM', 'adam')
        if residual_optim == 'sgd':
            optimizer = torch.optim.SGD([residual], lr=args.residual_lr, momentum=0.9)
        else:
            optimizer = torch.optim.Adam([residual], lr=args.lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.eta_min
        )
        print(f"  Residual mode: {residual.numel()} correction params")
        print(f"  Base pred error: {F.mse_loss(base_pred, x1).item():.2e}")
        print(f"  Residual optimizer: {residual_optim}, LR: {optimizer.param_groups[0]['lr']}")
    elif optim_type == 'lbfgs':
        print(f"  Using L-BFGS optimizer (ideal for deterministic single-sample)")
        optimizer = torch.optim.LBFGS(
            denoiser_params, lr=args.lr,
            max_iter=20, history_size=50,
            line_search_fn='strong_wolfe'
        )
        scheduler = None
    elif optim_type == 'sgd':
        optimizer = torch.optim.SGD(denoiser_params, lr=args.lr, momentum=0.9, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.eta_min
        )
    else:
        optimizer = torch.optim.AdamW(denoiser_params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.eta_min
        )

    # Compute per-contour weights inversely proportional to area  
    use_weighted = os.environ.get('FT_WEIGHTED', '0') == '1'
    contour_weights = None
    if use_weighted and N > 0:
        # Compute approximate area for each contour to weight small ones higher
        gt_polys_img = data['gt_polys_img']
        areas = []
        for idx in range(N):
            m = poly_to_mask(gt_polys_img[idx], 512, 512)
            areas.append(max(m.sum(), 1))
        areas_t = torch.tensor(areas, dtype=torch.float32, device=device)
        # Weight = 1/area, normalized so mean weight = 1
        contour_weights = (1.0 / areas_t)
        contour_weights = contour_weights / contour_weights.mean()
        print(f"  Weighted loss enabled. Weights per contour:")
        for idx in range(N):
            print(f"    Contour {idx}: area={areas[idx]}, weight={contour_weights[idx].item():.3f}")

    # Training loop
    log_path = os.path.join(args.save_dir, 'logs.jsonl')
    best_iou = 0.0
    best_epoch = 0
    
    print(f"\n[*] Starting fine-tuning...")
    t_start = time.time()

    def compute_loss(v_pred, v_target):
        """Compute weighted or unweighted MSE loss.
        Auto-promotes to float64 if v_pred is float64 (from float64 per-point heads).
        """
        # Match dtypes for precise loss computation
        if v_pred.dtype != v_target.dtype:
            v_target = v_target.to(dtype=v_pred.dtype)
        if contour_weights is not None:
            per_point_mse = (v_pred - v_target).pow(2).mean(dim=[1, 2])  # (N,)
            loss = (per_point_mse * contour_weights.to(dtype=v_pred.dtype)).mean()
        else:
            loss = F.mse_loss(v_pred, v_target, reduction='mean')
        return loss

    no_clip = os.environ.get('FT_NO_CLIP', '0') == '1'

    for epoch in range(args.epochs):
        if args.mode != 'residual':
            denoiser.train()

        if args.mode == 'residual':
            optimizer.zero_grad()
            combined = base_pred + residual
            loss = compute_loss(combined, x1)
            loss.backward()
            optimizer.step()
        elif optim_type == 'lbfgs':
            def closure():
                optimizer.zero_grad()
                v_pred, _ = gcn.predict_velocity(
                    cnn_feature, i_it_py, c_it_py, sampled_feat, py_ind, x_t, t_zeros
                )
                loss = compute_loss(v_pred, x1)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
        else:
            optimizer.zero_grad()
            v_pred, L_reg = gcn.predict_velocity(
                cnn_feature, i_it_py, c_it_py, sampled_feat, py_ind, x_t, t_zeros
            )
            loss = compute_loss(v_pred, x1)
            loss.backward()
            if not no_clip:
                torch.nn.utils.clip_grad_norm_(denoiser_params, 1.0)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        log_entry = {
            'epoch': epoch,
            'loss': loss.item(),
            'lr': current_lr,
        }

        # Periodic evaluation
        if (epoch + 1) % args.eval_every == 0 or epoch == 0:
            denoiser.eval()
            model.eval()
            with torch.no_grad():
                if args.mode == 'residual':
                    # Apply base_pred + residual, then denormalize
                    combined_norm = base_pred + residual
                    disp = gcn.denormalize_disp(combined_norm)
                else:
                    # Run inference to get displacement
                    disp = gcn.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind, steps=1)
                ious, mean_iou = compute_iou_from_pred(disp, data, dr)
            
            log_entry['mean_iou'] = mean_iou
            log_entry['per_iou'] = ious
            
            if mean_iou > best_iou:
                best_iou = mean_iou
                best_epoch = epoch
                # Save best checkpoint
                sd = {k: v.cpu() for k, v in model.state_dict().items()}
                save_dict = {
                    'state_dict': sd,
                    'epoch': epoch,
                    'loss': loss.item(),
                    'iou': mean_iou,
                }
                if residual is not None:
                    save_dict['point_residual'] = residual.detach().cpu()
                torch.save(save_dict, os.path.join(args.save_dir, 'checkpoints', 'best.pt'))

            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:5d} | loss={loss.item():.2e} | lr={current_lr:.2e} | "
                  f"IoU={mean_iou*100:.3f}% (best={best_iou*100:.3f}%@{best_epoch}) | "
                  f"{elapsed:.0f}s")

        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            sd = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({
                'state_dict': sd,
                'epoch': epoch,
                'loss': loss.item(),
            }, os.path.join(args.save_dir, 'checkpoints', f'epoch_{epoch+1}.pt'))

    # Final evaluation
    denoiser.eval()
    model.eval()
    with torch.no_grad():
        if args.mode == 'residual' and residual is not None:
            combined = base_pred + residual
            disp = gcn.denormalize_disp(combined)
        else:
            disp = gcn.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind, steps=1)
        ious, mean_iou = compute_iou_from_pred(disp, data, dr)
    
    print(f"\n=== Final Results ===")
    for idx, iou in enumerate(ious):
        print(f"  Contour {idx}: IoU = {iou*100:.3f}%")
    print(f"  Mean IoU = {mean_iou*100:.3f}%")
    print(f"  Best IoU = {best_iou*100:.3f}% @ epoch {best_epoch}")

    # Save final checkpoint
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    save_dict = {
        'state_dict': sd,
        'epoch': args.epochs,
        'loss': loss.item(),
        'iou': mean_iou,
    }
    if residual is not None:
        save_dict['point_residual'] = residual.detach().cpu()
    torch.save(save_dict, os.path.join(args.save_dir, 'checkpoints', 'final.pt'))


if __name__ == '__main__':
    main()
