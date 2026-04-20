#!/usr/bin/env python
"""
V3.4 vs V3.8 简单推理对比
直接加载checkpoint进行推理，生成可视化对比
"""
import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

os.chdir('/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30')
sys.path.insert(0, '/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30')

def load_model_and_infer(cfg_file, ckpt_path):
    """加载模型并推理"""
    # 设置环境变量
    os.environ['CFG_FILE'] = cfg_file

    # 清除缓存的模块
    for mod in list(sys.modules.keys()):
        if mod.startswith('lib.'):
            del sys.modules[mod]

    from lib.config import cfg
    from lib.networks import make_network
    from lib.train.trainers import make_trainer
    from lib.datasets.make_dataset import make_dataset
    from lib.datasets.transforms import make_transforms
    from lib.utils.snake import snake_config

    print(f"\n{'='*60}")
    print(f"配置: {Path(cfg_file).stem}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"poly_num: {snake_config.poly_num}")
    print(f"{'='*60}")

    # 加载模型
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)

    # 加载checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'net' in checkpoint:
        state_dict = checkpoint['net']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()

    # 加载测试数据
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, is_train=False), is_train=False)
    print(f"测试集大小: {len(dataset)}")

    # 推理第一个样本
    data = dataset[0]
    batch = {k: torch.tensor(v).unsqueeze(0).cuda() if isinstance(v, np.ndarray) else v
             for k, v in data.items()}

    with torch.no_grad():
        # 准备输入
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()

        output = model(batch)

        # 处理tuple输出
        if isinstance(output, tuple):
            output = output[0]  # 取第一个元素（dict）

    # 提取结果
    image = batch['inp'][0].cpu().numpy().transpose(1, 2, 0)

    # 反归一化
    mean = np.array([0.40789654, 0.44719302, 0.47026115])
    std = np.array([0.28863828, 0.27408164, 0.27809835])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)

    # 提取预测轮廓
    if 'py' in output:
        pred_polys = output['py'].cpu().numpy()  # [N, num_points, 2]
    else:
        pred_polys = None

    # 提取GT轮廓
    if 'i_gt_py' in batch:
        gt_polys = batch['i_gt_py']
        if isinstance(gt_polys, torch.Tensor):
            gt_polys = gt_polys.cpu().numpy()
        elif isinstance(gt_polys, list):
            gt_polys = np.array(gt_polys)

        # batch['i_gt_py'] shape: [batch_size, N, num_points, 2]
        # 取第一个batch的所有轮廓
        if len(gt_polys.shape) == 4:
            gt_polys = gt_polys[0]  # [N, num_points, 2]
    else:
        gt_polys = None

    return image, pred_polys, gt_polys, snake_config.poly_num

def draw_contours(image, contours, color, thickness=2):
    """绘制轮廓"""
    img = image.copy()
    if contours is not None and len(contours) > 0:
        for poly in contours:
            if isinstance(poly, torch.Tensor):
                poly = poly.cpu().numpy()
            poly = np.array(poly, dtype=np.float32)
            if len(poly.shape) == 2 and poly.shape[1] == 2:
                poly_int = poly.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [poly_int], True, color, thickness)
    return img

def calculate_metrics(pred_polys, gt_polys):
    """计算简单指标"""
    if pred_polys is None or gt_polys is None:
        return {}

    metrics = {
        'num_pred': len(pred_polys),
        'num_gt': len(gt_polys),
    }

    # 计算平均轮廓大小
    if len(pred_polys) > 0:
        areas = []
        for poly in pred_polys:
            if isinstance(poly, torch.Tensor):
                poly = poly.cpu().numpy()
            poly = np.array(poly, dtype=np.float32)
            if len(poly.shape) == 2 and poly.shape[0] > 2:
                area = cv2.contourArea(poly)
                areas.append(area)
        if areas:
            metrics['avg_area'] = np.mean(areas)
            metrics['min_area'] = np.min(areas)
            metrics['max_area'] = np.max(areas)

    return metrics

def main():
    print("="*60)
    print("V3.4 vs V3.8 推理对比")
    print("="*60)

    # V3.4推理
    print("\n推理 V3.4 (128点)...")
    img_v34, pred_v34, gt_v34, poly_num_v34 = load_model_and_infer(
        'configs/btcv_diffusion_dit_v3_4_single_overfit.yaml',
        'data/outputs/btcv_diffusion_dit_v3_4_single_overfit/checkpoints/epoch_10000.pt'
    )
    metrics_v34 = calculate_metrics(pred_v34, gt_v34)
    print(f"V3.4 指标: {metrics_v34}")

    # V3.8推理
    print("\n推理 V3.8 (64点)...")
    img_v38, pred_v38, gt_v38, poly_num_v38 = load_model_and_infer(
        'configs/btcv_diffusion_dit_v3_8_single_overfit.yaml',
        'data/outputs/btcv_diffusion_dit_v3_8_single_overfit/checkpoints/latest.pt'
    )
    metrics_v38 = calculate_metrics(pred_v38, gt_v38)
    print(f"V3.8 指标: {metrics_v38}")

    # 可视化对比
    print("\n生成可视化对比...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # V3.4
    axes[0, 0].imshow(img_v34)
    axes[0, 0].set_title(f'V3.4 Original ({poly_num_v34} points)')
    axes[0, 0].axis('off')

    img_v34_pred = draw_contours(img_v34, pred_v34, (0, 255, 0), 2)
    axes[0, 1].imshow(img_v34_pred)
    axes[0, 1].set_title(f'V3.4 Prediction\n{metrics_v34.get("num_pred", 0)} contours')
    axes[0, 1].axis('off')

    img_v34_gt = draw_contours(img_v34, gt_v34, (255, 0, 0), 2)
    axes[0, 2].imshow(img_v34_gt)
    axes[0, 2].set_title(f'V3.4 Ground Truth\n{metrics_v34.get("num_gt", 0)} contours')
    axes[0, 2].axis('off')

    # V3.8
    axes[1, 0].imshow(img_v38)
    axes[1, 0].set_title(f'V3.8 Original ({poly_num_v38} points)')
    axes[1, 0].axis('off')

    img_v38_pred = draw_contours(img_v38, pred_v38, (0, 255, 0), 2)
    axes[1, 1].imshow(img_v38_pred)
    axes[1, 1].set_title(f'V3.8 Prediction\n{metrics_v38.get("num_pred", 0)} contours')
    axes[1, 1].axis('off')

    img_v38_gt = draw_contours(img_v38, gt_v38, (255, 0, 0), 2)
    axes[1, 2].imshow(img_v38_gt)
    axes[1, 2].set_title(f'V3.8 Ground Truth\n{metrics_v38.get("num_gt", 0)} contours')
    axes[1, 2].axis('off')

    plt.tight_layout()

    output_dir = 'visual/comparison_v34_v38'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'inference_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n已保存: {output_file}")

    print("\n="*60)
    print("推理对比完成！")
    print("="*60)

if __name__ == '__main__':
    main()
