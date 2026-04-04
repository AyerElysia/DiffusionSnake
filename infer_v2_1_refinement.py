"""
V2.1 (Anchor Pool) Refinement Inference Script
-----------------------------------------------
Loads the V2.1 model specifically.
Uses 4 true extreme points to build initial octagon (128 pts).
"""
import os
import sys
import cv2
import torch
import numpy as np
import random
import datetime
from pathlib import Path

# Config setup
_THIS_DIR = os.path.dirname(__file__)
_DEFAULT_CFG = os.path.join(_THIS_DIR, 'configs', 'btcv_diffusion_dit_v2.yaml')
if not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = _DEFAULT_CFG

from lib.config import cfg, args
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.datasets.collate_batch import make_collator
from lib.utils.snake import snake_config, snake_decode, snake_gcn_utils

def to_numpy(x):
    return x.detach().cpu().numpy().astype(np.float32) if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float32)

def draw_poly(img, poly, color, thickness=2):
    if poly is None or len(poly) == 0: return
    cv2.polylines(img, [poly.astype(np.int32)], True, color, thickness)

def load_v2_1_model():
    cfg.use_diffusion_evolution = True
    cfg.use_dit_v2 = True
    cfg.use_dit_v2_1 = True # V2.1 Anchor Pool mode
    cfg.use_dit_v3 = False  # Ensure V3 is OFF
    
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ckpt_path = os.path.join(_THIS_DIR, 'data', 'outputs', 'btcv_diffusion_dit_v2_1', 'checkpoints', 'latest.pt')
    print(f"[*] Loading V2.1 Weights from: {ckpt_path}")
    if os.path.exists(ckpt_path):
        ckpt_obj = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt_obj.get('state_dict') or ckpt_obj.get('model') or ckpt_obj.get('net') or ckpt_obj
        model = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
        model.load_state_dict(sd, strict=False)
    else:
        print(f"[!] Checkpoint not found: {ckpt_path}")
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
    ex = torch.stack([poly_flat[batch_idx, t_idx], poly_flat[batch_idx, l_idx], 
                      poly_flat[batch_idx, b_idx], poly_flat[batch_idx, r_idx]], dim=1)
    return ex.view(B, M, 4, 2)

def main():
    model, device = load_v2_1_model()
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, is_train=False), is_train=False)
    index = random.randint(0, len(dataset) - 1)
    batch = make_collator(cfg)([dataset[index]])
    
    dr = float(snake_config.down_ratio)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            
    with torch.no_grad():
        core = model.net if hasattr(model, 'net') else model
        yolo_out = core.yolo(batch['inp'])
        p2 = yolo_out[1][0] if isinstance(yolo_out, tuple) and len(yolo_out) > 1 else None
        cnn_feature = core.cnn_proj(p2)
        
        # V2.1 initialization matching
        gt_all = batch['i_gt_py']
        valid_mask = (gt_all[..., 0].max(dim=-1).values - gt_all[..., 0].min(dim=-1).values) > 0.1
        extreme_pts = extract_true_extreme_points(gt_all)
        # Apply +0.5 for sub-pixel consistency
        ex_flat = extreme_pts.view(-1, 4, 2) + 0.5
        init_polys_flat = snake_decode.get_octagon(ex_flat)
        i_it_py = snake_gcn_utils.uniform_upsample(init_polys_flat.unsqueeze(0), 128)[0]
        
        # Dummy batch ind
        ind = torch.zeros(i_it_py.size(0), dtype=torch.long, device=device)
        
        # Sample
        c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)
        disp = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, ind, steps=50)
        pred_polys = (i_it_py + disp).cpu().numpy() * dr
        init_np = i_it_py.cpu().numpy() * dr
        gt_np = batch['i_gt_py'][0].cpu().numpy() * dr

    orig_img = to_numpy(batch['orig_img'][0]).astype(np.uint8)
    save_path = f"visual/v2_1_refinement_{index}.png"
    os.makedirs("visual", exist_ok=True)
    
    img_viz = orig_img.copy()
    for poly in gt_np: draw_poly(img_viz, poly, (255, 0, 0), 2)
    for poly in init_np: draw_poly(img_viz, poly, (0, 255, 255), 1)
    for poly in pred_polys: draw_poly(img_viz, poly, (0, 0, 255), 2)
    cv2.imwrite(save_path, img_viz)
    print(f"[*] V2.1 Result Saved: {save_path}")

if __name__ == '__main__':
    main()
