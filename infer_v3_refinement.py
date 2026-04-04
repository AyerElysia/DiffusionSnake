"""
Evolutionary Dynamic Network (V3) Inference Script
--------------------------------------------------
专注于测试 V3 的扩散细化能力，使用极点八边形 (Octagon) 初始化。
"""
import os
import sys
import cv2
import torch
import numpy as np
import random
import datetime
from pathlib import Path

# 默认启用 V3 配置文件
_THIS_DIR = os.path.dirname(__file__)
_DEFAULT_CFG = os.path.join(_THIS_DIR, 'configs', 'btcv_diffusion_dit_v3.yaml')

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
    # 1. GT 轮廓 (蓝色 - 参考系)
    if gt_poly is not None:
        for poly in gt_poly:
            draw_poly(img, poly, (255, 0, 0), thickness=2)
    
    # 2. V3 初始八边形 (黄色 - 观测解剖初始准确度)
    if init_poly is not None:
        for poly in init_poly:
            draw_poly(img, poly, (0, 255, 255), thickness=1)

    # 3. V3 最终演化轮廓 (红色 - 观测拓扑平滑度)
    if pred_poly is not None:
        for poly in pred_poly:
            draw_poly(img, poly, (0, 0, 255), thickness=2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)

def load_v3_model():
    """加载 V3 模型"""
    cfg.use_diffusion_evolution = True
    cfg.use_dit_v3 = True # 强制开启 V3
    
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 寻找 checkpoint
    cfg_stem = Path(os.environ.get('CFG_FILE', 'v3')).stem
    ckpt_path = getattr(args, 'ckpt', '') or os.path.join(_THIS_DIR, 'data', 'outputs', cfg_stem, 'checkpoints', 'latest.pt')
    
    print(f"[*] Loading V3 Weights from: {ckpt_path}")
    if os.path.exists(ckpt_path):
        ckpt_obj = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt_obj.get('state_dict') or ckpt_obj.get('model') or ckpt_obj.get('net') or ckpt_obj
        model = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
        model.load_state_dict(sd, strict=False)
    else:
        print("[!] Warning: Checkpoint not found, using random weights (testing logic only).")
    
    return trainer.network.to(device).eval(), device

def gt_to_octagon_init(batch, dr):
    """V3 核心：从 GT 框提取八边形种子"""
    gt_all = batch['i_gt_py'] # [B, M, P, 2]
    B, M = gt_all.size(0), gt_all.size(1)
    
    # 构造检测框 [B, M, 4]
    x1, y1 = gt_all[..., 0].min(dim=-1).values * dr, gt_all[..., 1].min(dim=-1).values * dr
    x2, y2 = gt_all[..., 0].max(dim=-1).values * dr, gt_all[..., 1].max(dim=-1).values * dr
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1) # [B, M, 4]
    
    # 模拟检测掩码 (排除填充的 0)
    valid_mask = (x2 - x1) > 1.0
    
    # 使用 V3 的八边形解码器 (Bbox -> Quadrangle -> Octagon)
    ex_points = snake_decode.get_quadrangle(bboxes)
    rect4_all = snake_decode.get_octagon(ex_points) / dr # 在特征尺度下初始化

    
    # 提取有效实例
    rect4_valid = rect4_all[valid_mask] # [N_instances, 4, 2]
    if rect4_valid.size(0) > 0:
        i_it_py = snake_gcn_utils.uniform_upsample(rect4_valid.unsqueeze(0), snake_config.poly_num)[0]
    else:
        i_it_py = torch.zeros((0, snake_config.poly_num, 2), device=gt_all.device)
        
    # 构造 batch 索引
    img_inds = []
    for b in range(B):
        num_valid = int(valid_mask[b].sum().item())
        if num_valid > 0:
            img_inds.append(torch.full((num_valid,), b, dtype=torch.long, device=gt_all.device))
    ind = torch.cat(img_inds) if img_inds else torch.zeros((0,), dtype=torch.long, device=gt_all.device)
    
    return i_it_py, ind, valid_mask

def main():
    model, device = load_v3_model()
    
    # 1. 准备数据
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, is_train=False), is_train=False)
    index = random.randint(0, len(dataset) - 1)
    sample = dataset[index]
    batch = make_collator(cfg)([sample])
    
    # 转 GPU
    for k, v in batch.items():
        if isinstance(v, torch.Tensor): batch[k] = v.to(device)
    
    # 2. V3 推理流程
    dr = float(snake_config.down_ratio)
    model.eval()
    with torch.no_grad():
        core = model.net if hasattr(model, 'net') else model
        
        # A. 提取图像特征 (Perceiver/Anchor)
        yolo_out = core.yolo(batch['inp'])
        p2 = yolo_out[1][0] if isinstance(yolo_out, tuple) and len(yolo_out) > 1 else None
        cnn_feature = core.cnn_proj(p2)
        
        # B. V3 初始化 (八边形)
        i_it_py, ind, valid_mask = gt_to_octagon_init(batch, dr)
        
        # C. 扩散演化 (Refinement)
        pred_polys = None
        if i_it_py.size(0) > 0:
            c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)
            # steps 可以设为 50, 100 等进行测试
            disp = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, ind, steps=50)
            pred_polys = (i_it_py + disp).cpu().numpy() * dr
            
    # 3. 可视化
    # 获取原始图像用于保存 (鲁棒性改进)
    orig_img = to_numpy(batch['orig_img'][0]).astype(np.uint8)

    init_np = i_it_py.cpu().numpy() * dr if i_it_py.numel() > 0 else None
    
    # 获取 GT 轮廓
    gt_poly_raw = batch['i_gt_py'][0][valid_mask[0]].cpu().numpy() * dr
    
    # 保存结果
    save_dir = os.path.join(_THIS_DIR, 'visual', 'v3_refinement_test')
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"v3_refine_{index}_{datetime.datetime.now().strftime('%H%M%S')}.png"
    save_path = os.path.join(save_dir, save_name)
    
    draw_results(orig_img, pred_polys, init_np, gt_poly_raw, save_path)
    print(f"[*] V3 Inference Done! Result saved to: {save_path}")
    print(f"[*] Initialized with {init_np.shape[0]} Octagon instances.")

if __name__ == '__main__':
    main()
