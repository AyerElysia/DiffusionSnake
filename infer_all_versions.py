"""
Full-Version Comparison Inference Script (DiT-Snake Ensemble)
-------------------------------------------------------------
自动化遍历 V1~V3 所有版本，并对同一个样本进行扩散演化对比。
"""
import os
import sys
import cv2
import torch
import numpy as np
import random
import datetime
from pathlib import Path

# 默认配置文件引导 (防止 lib.config 找不到默认项)
_THIS_DIR = os.path.dirname(__file__)
if not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = os.path.join(_THIS_DIR, 'configs', 'btcv_diffusion_dit_v3.yaml')

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

def load_version_model(cfg_path):
    """根据配置文件动态加载对应的模型和权重"""
    # 彻底重置 cfg，防止跨版本污染
    from lib.config import cfg as global_cfg
    global_cfg.defrost()
    global_cfg.merge_from_file(cfg_path)
    
    # 强制设置必要开关
    global_cfg.use_diffusion_evolution = True
    
    network = make_network(global_cfg)
    trainer = make_trainer(global_cfg, network)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 自动定位权重
    cfg_stem = Path(cfg_path).stem
    ckpt_path = os.path.join(_THIS_DIR, 'data', 'outputs', cfg_stem, 'checkpoints', 'latest.pt')
    
    if os.path.exists(ckpt_path):
        print(f"[*] Loading {cfg_stem} from: {ckpt_path}")
        ckpt_obj = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt_obj.get('state_dict') or ckpt_obj.get('model') or ckpt_obj.get('net') or ckpt_obj
        model = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
        model.load_state_dict(sd, strict=False)
    else:
        print(f"[!] Warning: No checkpoint found for {cfg_stem}, skipping.")
        return None, None, None

    model = trainer.network.to(device).eval()
    return model, device, global_cfg

def run_inference(model, device, version_cfg, batch):
    """执行单个版本的推理流程"""
    dr = float(snake_config.down_ratio)
    model.eval()
    with torch.no_grad():
        core = model.net if hasattr(model, 'net') else model
        
        # 提取图像特征
        yolo_out = core.yolo(batch['inp'])
        p2 = yolo_out[1][0] if isinstance(yolo_out, tuple) and len(yolo_out) > 1 else None
        cnn_feature = core.cnn_proj(p2)
        
        # 准备初始化 (根据版本选择策略)
        gt_all = batch['i_gt_py']
        x1, y1 = gt_all[..., 0].min(dim=-1).values * dr, gt_all[..., 1].min(dim=-1).values * dr
        x2, y2 = gt_all[..., 0].max(dim=-1).values * dr, gt_all[..., 1].max(dim=-1).values * dr
        bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
        valid_mask = (x2 - x1) > 1.0
        
        # V3 专属八边形策略，其它版本默认圆形
        if version_cfg.get('use_dit_v3', False):
            # V3 流程: Bbox -> Quadrangle(4 Extreme Points) -> Octagon
            ex_points = snake_decode.get_quadrangle(bboxes)
            rect4_all = snake_decode.get_octagon(ex_points) / dr
        else:
            # 其它版本: Bbox -> Circle/Box
            rect4_all = snake_decode.get_box(bboxes) / dr

            
        i_it_py = snake_gcn_utils.uniform_upsample(rect4_all[valid_mask].unsqueeze(0), snake_config.poly_num)[0]
        
        # 构造 ind
        num_valid = int(valid_mask.sum().item())
        ind = torch.zeros((num_valid,), dtype=torch.long, device=device)
        
        # 采样位移
        if i_it_py.size(0) > 0:
            c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)
            disp = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, ind, steps=50)
            pred_polys = (i_it_py + disp).cpu().numpy() * dr
            init_poly = i_it_py.cpu().numpy() * dr
            return pred_polys, init_poly
    return None, None

def main():
    # 1. 扫描所有配置文件
    configs = [
        'configs/btcv_diffusion_dit_v2.yaml',
        'configs/btcv_diffusion_dit_v2_1.yaml',
        'configs/btcv_diffusion_dit_v2_2.yaml',
        'configs/btcv_diffusion_dit_v2_3.yaml',
        'configs/btcv_diffusion_dit_v3.yaml'
    ]
    
    # 2. 预先随机挑选一个样本
    cfg.merge_from_file(configs[0])
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, is_train=False), is_train=False)
    index = random.randint(0, len(dataset) - 1)
    sample = dataset[index]
    batch_raw = make_collator(cfg)([sample])
    
    # 获取原始图像用于保存
    img_item = batch_raw['orig_img'][0]
    orig_img = to_numpy(img_item).astype(np.uint8)
    
    save_dir = os.path.join(_THIS_DIR, 'visual', 'all_version_comparison')
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%m%d_%H%M')
    
    print(f"[*] Starting Comparison. Sample Index: {index}")
    
    # 3. 循环跑推理
    for cfg_path in configs:
        if not os.path.exists(cfg_path): continue
        
        # 加载对应的模型
        model, device, v_cfg = load_version_model(cfg_path)
        if model is None: continue
        
        # 准备 batch 数据
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_raw.items()}
        
        pred, init = run_inference(model, device, v_cfg, batch)
        if pred is not None:
            v_name = Path(cfg_path).stem.replace('btcv_diffusion_dit_', '')
            out_img = orig_img.copy()
            
            # 1. 自动根据 GT 框生成有效实例掩码 (针对当前采样样本)
            dr = float(snake_config.down_ratio)
            gt_all = batch_raw['i_gt_py']
            x1, y1 = gt_all[..., 0].min(dim=-1).values * dr, gt_all[..., 1].min(dim=-1).values * dr
            x2, y2 = gt_all[..., 0].max(dim=-1).values * dr, gt_all[..., 1].max(dim=-1).values * dr
            v_mask_main = (x2 - x1) > 1.0

            # 2. 画金标准 GT (蓝色 - 真相)
            gt_poly_raw = batch_raw['i_gt_py'][0][v_mask_main[0]].cpu().numpy() * dr

            for p in gt_poly_raw:
                p_pts = p.astype(np.int32)
                cv2.polylines(out_img, [p_pts], isClosed=True, color=(255, 0, 0), thickness=2)
            
            # 2. 画初始 (黄色 - 起点)
            for p in init:
                p_pts = p.astype(np.int32)
                cv2.polylines(out_img, [p_pts], isClosed=True, color=(0, 255, 255), thickness=1)
                
            # 3. 画预测 (红色 - 结果)
            for p in pred:
                p_pts = p.astype(np.int32)
                cv2.polylines(out_img, [p_pts], isClosed=True, color=(0, 0, 255), thickness=2)

            
            save_path = os.path.join(save_dir, f"{ts}_idx{index}_{v_name}.png")
            cv2.imwrite(save_path, out_img)
            print(f"[#] {v_name} comparison saved to {save_path}")

    print(f"\n[*] ALL DONE. Comparison results are in: {save_dir}")

if __name__ == '__main__':
    main()
