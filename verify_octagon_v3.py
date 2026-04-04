import os
import sys
import torch
import numpy as np
import cv2
import random
from pathlib import Path

# 设置配置环境
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CFG = os.path.join(_THIS_DIR, 'configs', 'btcv_diffusion_dit_v3.yaml')
os.environ['CFG_FILE'] = _DEFAULT_CFG

from lib.config import cfg
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.datasets.collate_batch import make_collator
from lib.utils.snake import snake_config, snake_decode, snake_gcn_utils

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def get_true_extreme_points(poly):
    """从多边形中提取真实的 4 个极点 (Top, Left, Bottom, Right)"""
    t_idx = np.argmin(poly[:, 1])
    l_idx = np.argmin(poly[:, 0])
    b_idx = np.argmax(poly[:, 1])
    r_idx = np.argmax(poly[:, 0])
    # 返回顺序: 0:T, 1:L, 2:B, 3:R (应与 snake_decode.py 中的顺序一致)
    return np.stack([poly[t_idx], poly[l_idx], poly[b_idx], poly[r_idx]], axis=0)

def draw_poly(img, poly, color, thickness=1, closed=True):
    if poly is None or len(poly) == 0: return
    pts = poly.astype(np.int32)
    cv2.polylines(img, [pts], isClosed=closed, color=color, thickness=thickness)

def main():
    # 1. 加载数据
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, is_train=False), is_train=False)
    # 固定个索引，方便调试
    index = random.randint(0, len(dataset) - 1)
    sample = dataset[index]
    batch = make_collator(cfg)([sample])
    
    # 彻底确定尺度: orig_img 是原始图, i_gt_py 应与之对应
    img = to_numpy(batch['orig_img'][0]).astype(np.uint8)
    dr = float(snake_config.down_ratio) # 默认为 4
    
    # 获取有效实例
    # 注意：在某些 BTCV 数据加载器中，i_gt_py 是在 特征图尺度（如 128）下的。
    # 我们根据图像大小判断是否需要乘以 dr。
    gt_all = batch['i_gt_py'][0]
    h_img, w_img = img.shape[:2]
    val_max = gt_all.max().item()
    
    # 如果 GT 的最大值远小于图像尺寸，说明它在特征尺度下，需要映射到图像尺度。
    if val_max < (max(h_img, w_img) / 2.0):
        scale_factor = dr
    else:
        scale_factor = 1.0 # 已经是图像尺度了
        
    valid_mask = (gt_all.sum(dim=-1).sum(dim=-1) != 0)
    valid_gt = gt_all[valid_mask].cpu().numpy() * scale_factor

    if len(valid_gt) == 0:
        return main()

    os.makedirs('visual', exist_ok=True)

    # --- 任务 1: octagon_comparison.png (特写对比) ---
    inst_idx = 0
    gt_cur = valid_gt[inst_idx]
    ex_cur = get_true_extreme_points(gt_cur)
    
    # 在 1.0 比例下生成标准四边形和八边形
    # get_octagon 预期 [..., 4, 2]
    ex_torch = torch.from_numpy(ex_cur).float().unsqueeze(0).unsqueeze(0)
    octa_torch = snake_decode.get_octagon(ex_torch)
    octa_np = octa_torch[0, 0].cpu().numpy()
    
    # 上采样
    upsampled_torch = snake_gcn_utils.uniform_upsample(octa_torch, snake_config.poly_num)
    upsampled_np = upsampled_torch[0, 0].cpu().numpy()
    
    # 构造特写图
    roi_pad = 40
    b_min, b_max = gt_cur.min(0), gt_cur.max(0)
    x_s, y_s = max(0, int(b_min[0]-roi_pad)), max(0, int(b_min[1]-roi_pad))
    x_e, y_e = min(w_img, int(b_max[0]+roi_pad)), min(h_img, int(b_max[1]+roi_pad))
    roi = img[y_s:y_e, x_s:x_e].copy()
    off = np.array([x_s, y_s])
    
    draw_poly(roi, gt_cur - off, (255, 0, 0), thickness=2) # 蓝色: GT
    # 极点四边形 (T -> R -> B -> L)
    quad_np = np.stack([ex_cur[0], ex_cur[3], ex_cur[2], ex_cur[1]], axis=0)
    draw_poly(roi, quad_np - off, (0, 255, 0), thickness=2) # 绿色: 极点四边形
    draw_poly(roi, octa_np - off, (0, 255, 255), thickness=1) # 黄色: 标准八边形
    
    # 极点本体标记 (白色实心圆)
    for p in ex_cur:
        cv2.circle(roi, tuple((p - off).astype(np.int32)), 3, (255, 255, 255), -1)
    
    # 采样点 (红色点)
    for p in upsampled_np:
        cv2.circle(roi, tuple((p - off).astype(np.int32)), 2, (0, 0, 255), -1)
        
    cv2.imwrite('visual/octagon_comparison.png', roi)
    print(f"[*] Visual 1: Characterized on instance 0 of sample {index}")

    # --- 任务 2: octagon_multi_boxes.png (全局对比) ---
    multi_img = img.copy()
    for poly in valid_gt:
        ex = get_true_extreme_points(poly)
        ex_t = torch.from_numpy(ex).float().unsqueeze(0).unsqueeze(0)
        oct_t = snake_decode.get_octagon(ex_t)
        up_t = snake_gcn_utils.uniform_upsample(oct_t, snake_config.poly_num)
        
        # 绘制
        draw_poly(multi_img, poly, (255, 0, 0), thickness=1) # 蓝色: GT 基准
        draw_poly(multi_img, oct_t[0, 0].cpu().numpy(), (0, 255, 255), thickness=1) # 黄色: 八边形
        draw_poly(multi_img, up_t[0, 0].cpu().numpy(), (0, 0, 255), thickness=1) # 红色点
        
    cv2.imwrite('visual/octagon_multi_boxes.png', multi_img)
    print(f"[*] Visual 2: All {len(valid_gt)} instances processed.")

if __name__ == '__main__':
    main()
