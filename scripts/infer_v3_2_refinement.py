import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
"""
DiT V3.2 Flow Matching Batch Inference Script
--------------------------------------------------------------
Features: Self+Cross Attention (Efficient) + Flow Matching (10 steps ODE)
Usage: Runs on 5 random samples to evaluate the latest V3.2 model.
"""
import os
import sys
import cv2
import torch
import numpy as np
import random
import datetime
from pathlib import Path

# Set default V3.2 config
_THIS_DIR = os.path.dirname(__file__)
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..'))
_DEFAULT_CFG = os.path.join(_ROOT_DIR, 'configs', 'btcv_diffusion_dit_v3_2.yaml')

if not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = _DEFAULT_CFG

from lib.config import cfg, args
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.datasets.collate_batch import make_collator
from lib.utils.snake import snake_config, snake_decode, snake_gcn_utils
from lib.utils import data_utils

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)

def draw_poly(img, poly, color, thickness=2, closed=True):
    if poly is None or len(poly) == 0: return
    pts = poly.astype(np.int32)
    cv2.polylines(img, [pts], isClosed=closed, color=color, thickness=thickness)

def draw_results(img, pred_poly, init_poly=None, gt_poly=None, save_path=None):
    img = img.copy()
    # 1. GT (Blue)
    if gt_poly is not None:
        for poly in gt_poly:
            draw_poly(img, poly, (255, 0, 0), thickness=2)
    # 2. V3.2 Initial Octagon (Yellow)
    if init_poly is not None:
        for poly in init_poly:
            draw_poly(img, poly, (0, 255, 255), thickness=1)
    # 3. V3.2 Refined Contour (Red)
    if pred_poly is not None:
        for poly in pred_poly:
            draw_poly(img, poly, (0, 0, 255), thickness=2)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)

def load_v3_2_model():
    # Ensure V3.2 config is set (driven by config file)
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Checkpoint path
    ckpt_path = getattr(args, 'ckpt', '') or os.path.join(_ROOT_DIR, 'data', 'outputs', 'btcv_diffusion_dit_v3_2', 'checkpoints', 'latest.pt')
    print(f"[*] Loading V3.2 Flow Matching Weights from: {ckpt_path}")
    if os.path.exists(ckpt_path):
        ckpt_obj = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt_obj.get('state_dict') or ckpt_obj.get('model') or ckpt_obj.get('net') or ckpt_obj
        model = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
        model.load_state_dict(sd, strict=False)
    else:
        print(f"[!] Checkpoint not found at {ckpt_path}")
        sys.exit(1)
    return trainer.network.to(device).eval(), device

def extract_true_extreme_points(poly):
    B, M, P, _ = poly.shape
    poly_flat = poly.view(B * M, P, 2)
    t_idx = torch.argmin(poly_flat[..., 1], dim=-1)
    l_idx = torch.argmin(poly_flat[..., 0], dim=-1)
    b_idx = torch.argmax(poly_flat[..., 1], dim=-1)
    r_idx = torch.argmax(poly_flat[..., 0], dim=-1)
    batch_idx = torch.arange(B * M, device=poly.device)
    t, l, b, r = poly_flat[batch_idx, t_idx], poly_flat[batch_idx, l_idx], poly_flat[batch_idx, b_idx], poly_flat[batch_idx, r_idx]
    ex = torch.stack([t, l, b, r], dim=1)
    return ex.view(B, M, 4, 2)

def prepare_v3_2_init(batch, dr):
    gt_all = batch['i_gt_py']
    B, M = gt_all.size(0), gt_all.size(1)
    extreme_pts = extract_true_extreme_points(gt_all)
    extreme_pts = extreme_pts + 0.5
    x1, y1 = gt_all[..., 0].min(dim=-1).values, gt_all[..., 1].min(dim=-1).values
    x2, y2 = gt_all[..., 0].max(dim=-1).values, gt_all[..., 1].max(dim=-1).values
    valid_mask = (x2 - x1) > 0.1
    ex_flat = extreme_pts.view(-1, 4, 2)
    init_polys_flat = snake_decode.get_octagon(ex_flat)
    init_polys = init_polys_flat.view(B, M, 12, 2)
    i_it_py = snake_gcn_utils.uniform_upsample(init_polys[valid_mask].unsqueeze(0), snake_config.poly_num)[0]
    img_inds = []
    for b in range(B):
        num_v = int(valid_mask[b].sum().item())
        if num_v > 0: img_inds.append(torch.full((num_v,), b, dtype=torch.long, device=gt_all.device))
    ind = torch.cat(img_inds) if img_inds else torch.zeros((0,), dtype=torch.long, device=gt_all.device)
    return i_it_py, ind, valid_mask

def run_inference(model, device, batch, index, save_dir, ode_steps=10):
    dr = float(snake_config.down_ratio)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor): batch[k] = v.to(device)
    with torch.no_grad():
        core = model.net if hasattr(model, 'net') else model
        yolo_out = core.yolo(batch['inp'])
        p2 = yolo_out[1][0] if isinstance(yolo_out, tuple) and len(yolo_out) > 1 else None
        cnn_feature = core.cnn_proj(p2)
        i_it_py, ind, valid_mask = prepare_v3_2_init(batch, dr)
        pred_polys = None
        if i_it_py.size(0) > 0:
            c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)
            # V3.2: Flow Matching ODE solver (10 steps by default)
            disp = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, ind, steps=ode_steps)
            pred_polys = (i_it_py + disp).cpu().numpy() * dr

    orig_img = to_numpy(batch['orig_img'][0]).astype(np.uint8)
    init_np = i_it_py.cpu().numpy() * dr if i_it_py.numel() > 0 else None
    gt_poly_raw = batch['i_gt_py'][0][valid_mask[0]].cpu().numpy() * dr
    save_path = os.path.join(save_dir, f"{datetime.datetime.now().strftime('%H%M%S')}_v3_2_idx{index}.png")
    draw_results(orig_img, pred_polys, init_np, gt_poly_raw, save_path)
    print(f"[*] Processed index {index} (ODE {ode_steps} steps) -> {save_path}")

def main():
    model, device = load_v3_2_model()
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, is_train=False), is_train=False)
    save_dir = os.path.join(_THIS_DIR, 'visual', 'v3_2_flow_matching_eval')
    os.makedirs(save_dir, exist_ok=True)

    # Get ODE steps from config
    ode_steps = getattr(cfg, 'flow_ode_steps', 10)
    print(f"[*] V3.2 Flow Matching: ODE steps = {ode_steps}")

    # Run 5 random samples
    indices = random.sample(range(len(dataset)), 5)
    print(f"[*] Starting Batch Inference for 5 samples: {indices}")
    for idx in indices:
        batch = make_collator(cfg)([dataset[idx]])
        run_inference(model, device, batch, idx, save_dir, ode_steps=ode_steps)
    print(f"\n[✔] Inference Complete! All results in: {save_dir}")

if __name__ == '__main__':
    main()