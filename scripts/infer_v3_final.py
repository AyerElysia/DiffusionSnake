import sys, os
import torch
import numpy as np
import cv2
import datetime
import json
from pathlib import Path

# 环境与配置初始化
_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _THIS_DIR)

from lib.config import cfg, args
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.datasets.make_dataset import make_dataset
from lib.datasets.collate_batch import make_collator
from lib.datasets.transforms import make_transforms
from lib.utils.snake import snake_config, snake_decode, snake_gcn_utils

def load_v3_model(cfg_file=None):
    if cfg_file:
        cfg.merge_from_file(cfg_file)
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取权重路径
    ckpt_path = os.path.join(_THIS_DIR, 'data/outputs', cfg.model_dir.split('/')[-1], 'checkpoints', 'latest.pt')
    print(f"[*] Loading Weights: {ckpt_path}")
    
    if os.path.exists(ckpt_path):
        ckpt_obj = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt_obj.get('state_dict') or ckpt_obj.get('model') or ckpt_obj.get('net') or ckpt_obj
        
        # Remap legacy keys (keep net. prefix for NetworkWrapper)
        from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
        sd = remap_legacy_state_dict(sd)
        wrapper = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
        
        info = wrapper.load_state_dict(sd, strict=False)
        print(f"[✔] Success: {len(sd) - len(info.missing_keys)} layers matched.")
    else:
        print(f"[!] Error: Checkpoint NOT found at {ckpt_path}!")
        sys.exit(1)
        
    return trainer.network.to(device).eval(), device

def run_inference(model, device, batch, save_dir, ver_tag, index):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor): batch[k] = v.to(device)
    
    dr = float(snake_config.down_ratio)
    core = model.net if hasattr(model, 'net') else model
    
    with torch.no_grad():
        # 1. 网络前向
        # 如果是 tuple，通常是 (None, [p2, p3, p4, p5])
        yolo_out = core.yolo(batch['inp'])
        if isinstance(yolo_out, (list, tuple)):
            feat_p2 = yolo_out[1][0] if len(yolo_out) > 1 else yolo_out[0]
        else:
            feat_p2 = yolo_out
        cnn_feature = core.cnn_proj(feat_p2)
        
        # 2. 准备初始八边形 (基于 GT 极值点快速验证)
        # 获取极值点并进行上采样
        gt_all = batch['i_gt_py']
        if gt_all.numel() == 0: return
        
        # 提取极值点
        B, M, P, _ = gt_all.shape
        poly_flat = gt_all.view(B * M, P, 2)
        
        t_idx = torch.argmin(poly_flat[..., 1], dim=-1)
        l_idx = torch.argmin(poly_flat[..., 0], dim=-1)
        b_idx = torch.argmax(poly_flat[..., 1], dim=-1)
        r_idx = torch.argmax(poly_flat[..., 0], dim=-1)
        batch_idx = torch.arange(B * M, device=device)
        ex = torch.stack([poly_flat[batch_idx, t_idx], poly_flat[batch_idx, l_idx], 
                          poly_flat[batch_idx, b_idx], poly_flat[batch_idx, r_idx]], dim=1) + 0.5
        
        init_polys = snake_decode.get_octagon(ex).view(B, M, 12, 2)
        i_it_py = snake_gcn_utils.uniform_upsample(init_polys, 128)[0]
        c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)
        py_ind = torch.zeros(i_it_py.size(0), dtype=torch.long, device=device)
        
        # 3. 核心：调用原生的 sample_disp (带分布式反归一化)
        # 注意：sample_disp 内部会自动调用 denormalize_disp
        disp = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind, steps=50)
        
        # 检查位移统计
        print(f"  [Stats] Disp Min: {disp.min().item():.3f}, Max: {disp.max().item():.3f}")
        
        # 4. 生成预测多边形
        pred_polys = (i_it_py + disp).cpu().numpy() * dr
        init_np = i_it_py.cpu().numpy() * dr
        gt_np = gt_all.cpu().numpy() * dr
        
    # 5. 渲染
    # 安全获取原图
    if 'orig_img' in batch:
        img_raw = batch['orig_img'][0]
        img = img_raw.detach().cpu().numpy() if torch.is_tensor(img_raw) else img_raw
        img = img.astype(np.uint8)
    else:
        print("  [!] Warning: 'orig_img' missing, using black background.")
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        
    # 画图：OpenCV 顺序是 BGR
    for poly in gt_np[0]: cv2.polylines(img, [poly.astype(np.int32)], True, (0, 255, 0), 2)  # GT: 绿
    for poly in init_np: cv2.polylines(img, [poly.astype(np.int32)], True, (0, 255, 255), 1) # Init: 黄
    for poly in pred_polys: cv2.polylines(img, [poly.astype(np.int32)], True, (0, 0, 255), 2) # Pred: 红
    
    save_path = os.path.join(save_dir, f"CLEAN_{ver_tag}_idx{index}_{datetime.datetime.now().strftime('%H%M%S')}.png")
    cv2.imwrite(save_path, img) 
    print(f"[*] Saved: {save_path}")

def main():
    # 参数解析已由 from lib.config import cfg, args 完成（通过 --cfg_file）
    tag = os.environ.get('TAG', 'v3')
    index = int(os.environ.get('INDEX', 0))
    
    # 无需再次加载模型配置，lib.config 已经加载到全局 cfg 中了
    model, device = load_v3_model(None) # 传入 None 即可
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)
    
    save_dir = os.path.join(_THIS_DIR, 'visual/v3_clean_eval')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[*] Starting CLEAN inference (Tag: {tag}, Index: {index})...")
    batch = collator([dataset[index]])
    run_inference(model, device, batch, save_dir, tag, index)

if __name__ == '__main__':
    main()
