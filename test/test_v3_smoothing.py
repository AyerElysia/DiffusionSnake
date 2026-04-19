"""
在V3.0预测结果上应用边缘感知平滑
直接修改推理脚本，保存轮廓数据并应用平滑
"""

import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# 设置配置文件
os.environ['CFG_FILE'] = 'configs/btcv_diffusion_dit_v3.yaml'

from lib.config import cfg
from lib.networks import make_network
from lib.datasets import make_data_loader
from lib.utils.snake import snake_decode
from edge_smoothing import smooth_contours_numpy


def load_model(checkpoint_path, device='cuda:0'):
    """加载V3.0模型"""
    network = make_network(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(checkpoint['net'], strict=False)
    network.eval()
    return network, device


def run_inference_with_smoothing(model, dataloader, output_dir, device, curvature_threshold=5.0, iterations=2):
    """
    运行推理并应用边缘感知平滑

    Args:
        model: V3.0模型
        dataloader: 数据加载器
        output_dir: 输出目录
        device: 设备
        curvature_threshold: 曲率阈值
        iterations: 平滑迭代次数
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/original", exist_ok=True)
    os.makedirs(f"{output_dir}/smoothed", exist_ok=True)
    os.makedirs(f"{output_dir}/comparison", exist_ok=True)

    print(f"输出目录: {output_dir}")
    print(f"曲率阈值: {curvature_threshold}")
    print(f"迭代次数: {iterations}")
    print("=" * 60)

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            # 准备输入
            inp = batch['inp'].to(device)

            # 模型推理
            output = model(inp)

            # 获取预测轮廓
            if 'py' in output:
                pred_contours = output['py']  # (N, P, 2)
            elif 'pred_contours' in output:
                pred_contours = output['pred_contours']
            else:
                print(f"警告: 无法从输出中找到轮廓数据")
                continue

            # 转换为numpy
            pred_contours_np = pred_contours.cpu().numpy()

            # 应用边缘感知平滑
            smoothed_contours_np = []
            for contour in pred_contours_np:
                smoothed = smooth_contours_numpy(
                    contour,
                    curvature_threshold=curvature_threshold,
                    iterations=iterations
                )
                smoothed_contours_np.append(smoothed)
            smoothed_contours_np = np.array(smoothed_contours_np)

            # 获取图像
            img = batch['inp'][0].cpu().numpy().transpose(1, 2, 0)
            img = ((img * cfg.std + cfg.mean) * 255).astype(np.uint8)
            if img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # 可视化
            visualize_comparison(
                img,
                pred_contours_np,
                smoothed_contours_np,
                f"{output_dir}/comparison/idx{idx}.png"
            )

            # 保存轮廓数据
            np.save(f"{output_dir}/original/contours_idx{idx}.npy", pred_contours_np)
            np.save(f"{output_dir}/smoothed/contours_idx{idx}.npy", smoothed_contours_np)

            print(f"处理完成: idx{idx}, {len(pred_contours_np)} 个轮廓")

            # 只处理前5张图像
            if idx >= 4:
                break

    print("=" * 60)
    print(f"所有结果已保存到: {output_dir}")


def visualize_comparison(image, original_contours, smoothed_contours, save_path):
    """可视化对比"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 原始预测
    img_original = image.copy()
    for contour in original_contours:
        contour_int = contour.astype(np.int32)
        cv2.polylines(img_original, [contour_int], True, (0, 255, 0), 2)

    # 平滑后
    img_smoothed = image.copy()
    for contour in smoothed_contours:
        contour_int = contour.astype(np.int32)
        cv2.polylines(img_smoothed, [contour_int], True, (255, 0, 0), 2)

    # 叠加对比
    img_overlay = image.copy()
    for contour in original_contours:
        contour_int = contour.astype(np.int32)
        cv2.polylines(img_overlay, [contour_int], True, (0, 255, 0), 2)
    for contour in smoothed_contours:
        contour_int = contour.astype(np.int32)
        cv2.polylines(img_overlay, [contour_int], True, (255, 0, 0), 1)

    axes[0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('V3.0 Original (Green)', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(img_smoothed, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Edge-Aware Smoothed (Red)', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Overlay (Green=Original, Red=Smoothed)', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("V3.0 + 边缘感知平滑 验证脚本")
    print("=" * 60)

    # 配置
    checkpoint_path = "data/outputs/btcv_diffusion_dit_v3/checkpoints/latest.pt"
    output_dir = "visual/v3_edge_aware_smoothed_real"

    # 检查checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"错误: checkpoint不存在 {checkpoint_path}")
        print("\n可用的checkpoint:")
        ckpt_dir = "data/outputs/btcv_diffusion_dit_v3/checkpoints"
        if os.path.exists(ckpt_dir):
            for f in os.listdir(ckpt_dir):
                print(f"  - {f}")
        return

    print(f"加载模型: {checkpoint_path}")

    # 加载模型
    model, device = load_model(checkpoint_path)
    print(f"模型加载成功，使用设备: {device}")

    # 创建数据加载器
    print("创建数据加载器...")
    dataloader = make_data_loader(cfg, is_train=False)
    print(f"数据集大小: {len(dataloader)}")

    # 运行推理
    run_inference_with_smoothing(
        model,
        dataloader,
        output_dir,
        device,
        curvature_threshold=5.0,
        iterations=2
    )


if __name__ == "__main__":
    main()
