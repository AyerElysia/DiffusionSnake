#!/usr/bin/env python
"""
V3.4 vs V3.8 可视化对比脚本
生成轮廓对比图，重点关注小轮廓的毛刺情况
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

def load_and_infer(cfg_file, ckpt_path):
    """加载模型并推理"""
    os.environ['CFG_FILE'] = cfg_file

    # 重新导入配置
    import importlib
    for mod in ['lib.config', 'lib.utils.snake.snake_config']:
        if mod in sys.modules:
            del sys.modules[mod]

    from lib.config import cfg
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.snake import snake_config
    from lib.utils import net_utils

    print(f"\n加载模型: {cfg_file}")
    print(f"poly_num: {snake_config.poly_num}")

    # 加载模型
    network = make_network(cfg)

    # 直接加载checkpoint，不依赖cfg.model_dir
    print(f"加载checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cuda')
    if 'net' in checkpoint:
        network.load_state_dict(checkpoint['net'], strict=False)
    else:
        network.load_state_dict(checkpoint, strict=False)

    network = network.cuda().eval()

    # 加载数据
    data_loader = make_data_loader(cfg, is_train=False)

    # 推理
    all_images = []
    all_contours = []
    all_gt_contours = []

    with torch.no_grad():
        for batch in data_loader:
            # 移到GPU
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].cuda()

            # 推理
            output = network(batch['inp'], batch)

            print(f"Output keys: {output.keys()}")
            if 'py_pred' in output:
                print(f"py_pred type: {type(output['py_pred'])}, len: {len(output['py_pred']) if isinstance(output['py_pred'], (list, tuple)) else 'N/A'}")

            # 提取结果
            images = batch['inp'].cpu().numpy()

            # 提取预测轮廓
            if 'py' in output and isinstance(output['py'], torch.Tensor):
                pred_contours = output['py'].cpu().numpy()
                print(f"使用 output['py'], shape: {pred_contours.shape}")
            elif 'py_pred' in output and isinstance(output['py_pred'], (list, tuple)) and len(output['py_pred']) > 0:
                pred_contours = output['py_pred'][-1].cpu().numpy()
                print(f"使用 output['py_pred'][-1], shape: {pred_contours.shape}")
            else:
                print(f"警告: 无法提取预测轮廓，output keys: {output.keys()}")
                pred_contours = None

            # GT轮廓
            if 'i_gt_py' in batch:
                gt_contours = batch['i_gt_py'].cpu().numpy()
            else:
                gt_contours = None

            all_images.append(images)
            all_contours.append(pred_contours)
            all_gt_contours.append(gt_contours)

    return all_images, all_contours, all_gt_contours

def visualize_comparison(images_v34, contours_v34, images_v38, contours_v38, gt_contours, output_dir):
    """生成对比可视化"""
    os.makedirs(output_dir, exist_ok=True)

    # 处理第一个batch
    img_v34 = images_v34[0][0]  # [C, H, W]
    img_v38 = images_v38[0][0]

    # 反归一化
    mean = np.array([0.40789654, 0.44719302, 0.47026115]).reshape(3, 1, 1)
    std = np.array([0.28863828, 0.27408164, 0.27809835]).reshape(3, 1, 1)

    img_v34 = img_v34 * std + mean
    img_v38 = img_v38 * std + mean

    img_v34 = np.transpose(img_v34, (1, 2, 0))
    img_v38 = np.transpose(img_v38, (1, 2, 0))

    img_v34 = np.clip(img_v34, 0, 1)
    img_v38 = np.clip(img_v38, 0, 1)

    # 获取轮廓
    contours_v34_batch = contours_v34[0][0]  # [N, num_points, 2]
    contours_v38_batch = contours_v38[0][0]

    if gt_contours[0] is not None:
        gt_contours_batch = gt_contours[0][0]
    else:
        gt_contours_batch = None

    # 为每个轮廓生成对比图
    num_contours = contours_v34_batch.shape[0]

    for i in range(num_contours):
        contour_v34 = contours_v34_batch[i]
        contour_v38 = contours_v38_batch[i]

        # 计算面积判断是否为小轮廓
        area_v34 = cv2.contourArea(contour_v34.astype(np.float32))

        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # V3.4
        axes[0].imshow(img_v34)
        axes[0].plot(contour_v34[:, 0], contour_v34[:, 1], 'r-', linewidth=2, label='V3.4 (128 points)')
        axes[0].scatter(contour_v34[:, 0], contour_v34[:, 1], c='r', s=10, alpha=0.5)
        axes[0].set_title(f'V3.4 (128 points)\nArea: {area_v34:.1f}')
        axes[0].legend()
        axes[0].axis('off')

        # V3.8
        axes[1].imshow(img_v38)
        axes[1].plot(contour_v38[:, 0], contour_v38[:, 1], 'b-', linewidth=2, label='V3.8 (64 points)')
        axes[1].scatter(contour_v38[:, 0], contour_v38[:, 1], c='b', s=10, alpha=0.5)
        axes[1].set_title(f'V3.8 (64 points)\nArea: {area_v34:.1f}')
        axes[1].legend()
        axes[1].axis('off')

        # 叠加对比
        axes[2].imshow(img_v34)
        axes[2].plot(contour_v34[:, 0], contour_v34[:, 1], 'r-', linewidth=2, label='V3.4 (128)', alpha=0.7)
        axes[2].plot(contour_v38[:, 0], contour_v38[:, 1], 'b-', linewidth=2, label='V3.8 (64)', alpha=0.7)

        if gt_contours_batch is not None:
            gt_contour = gt_contours_batch[i]
            axes[2].plot(gt_contour[:, 0], gt_contour[:, 1], 'g--', linewidth=1, label='GT', alpha=0.5)

        axes[2].set_title('Overlay Comparison')
        axes[2].legend()
        axes[2].axis('off')

        plt.tight_layout()

        # 保存
        size_label = 'small' if area_v34 < 2000 else 'large'
        output_file = os.path.join(output_dir, f'contour_{i}_{size_label}_area{int(area_v34)}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"已保存: {output_file}")

def main():
    print("="*60)
    print("V3.4 vs V3.8 可视化对比")
    print("="*60)

    # V3.4推理
    print("\n推理 V3.4...")
    v34_ckpt = 'data/outputs/btcv_diffusion_dit_v3_4_single_overfit/checkpoints/epoch_10000.pt'
    print(f"使用checkpoint: {v34_ckpt}")
    images_v34, contours_v34, gt_v34 = load_and_infer(
        'configs/btcv_diffusion_dit_v3_4_single_overfit.yaml',
        v34_ckpt
    )

    # V3.8推理
    print("\n推理 V3.8...")
    v38_ckpt = 'data/outputs/btcv_diffusion_dit_v3_8_single_overfit/checkpoints/latest.pt'
    print(f"使用checkpoint: {v38_ckpt}")
    images_v38, contours_v38, gt_v38 = load_and_infer(
        'configs/btcv_diffusion_dit_v3_8_single_overfit.yaml',
        v38_ckpt
    )

    # 生成可视化
    print("\n生成可视化对比...")
    visualize_comparison(
        images_v34, contours_v34,
        images_v38, contours_v38,
        gt_v34,
        'visual/comparison_v34_v38'
    )

    print("\n="*60)
    print("可视化完成！")
    print("="*60)

if __name__ == '__main__':
    main()
