"""
快速验证脚本：对V3.0的预测结果应用边缘感知平滑

使用方法：
1. 先用V3.0模型生成预测结果（保存为.npy或可视化图像）
2. 运行此脚本对预测结果进行后处理
3. 对比平滑前后的效果

不修改任何现有代码，完全独立运行
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# 导入边缘平滑模块
from edge_smoothing import EdgeAwareSmoothing, smooth_contours_numpy


def load_contours_from_npy(npy_path):
    """从.npy文件加载轮廓"""
    data = np.load(npy_path, allow_pickle=True)
    return data


def visualize_comparison(image, original_contours, smoothed_contours, save_path):
    """
    可视化对比原始轮廓和平滑后的轮廓

    Args:
        image: 原始图像 (H, W, 3) 或 (H, W)
        original_contours: 原始轮廓列表，每个元素是 (P, 2)
        smoothed_contours: 平滑后的轮廓列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 如果是灰度图，转换为RGB
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

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
        cv2.polylines(img_overlay, [contour_int], True, (255, 0, 0), 2)

    axes[0].imshow(img_original)
    axes[0].set_title('V3.0 Original (Green)', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(img_smoothed)
    axes[1].set_title('Edge-Aware Smoothed (Red)', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(img_overlay)
    axes[2].set_title('Overlay (Green=Original, Red=Smoothed)', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"可视化结果已保存到: {save_path}")


def process_single_image(image_path, contours, curvature_threshold=5.0, iterations=2):
    """
    处理单张图像的轮廓

    Args:
        image_path: 图像路径
        contours: 轮廓列表，每个元素是 (P, 2) 的numpy数组
        curvature_threshold: 曲率阈值
        iterations: 平滑迭代次数
    Returns:
        smoothed_contours: 平滑后的轮廓列表
    """
    smoothed_contours = []

    for contour in contours:
        # 应用边缘感知平滑
        smoothed = smooth_contours_numpy(
            contour,
            curvature_threshold=curvature_threshold,
            iterations=iterations
        )
        smoothed_contours.append(smoothed)

    return smoothed_contours


def main():
    """
    主函数：批量处理V3.0的预测结果

    使用方式：
    1. 修改下面的路径配置
    2. 运行脚本
    """

    # ==================== 配置区域 ====================
    # V3.0的可视化结果目录
    visual_dir = "visual/single_sample_all_models/v3_overfit_single_gpu7_e10000"

    # 输出目录
    output_dir = "visual/v3_edge_aware_smoothed"

    # 平滑参数
    curvature_threshold = 5.0  # 曲率阈值，越小越保留尖锐转角
    iterations = 2  # 迭代次数，1-3次

    # ==================== 配置结束 ====================

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("边缘感知平滑 - 快速验证脚本")
    print("=" * 60)
    print(f"输入目录: {visual_dir}")
    print(f"输出目录: {output_dir}")
    print(f"曲率阈值: {curvature_threshold}")
    print(f"迭代次数: {iterations}")
    print("=" * 60)

    # 检查目录是否存在
    if not os.path.exists(visual_dir):
        print(f"错误：目录不存在 {visual_dir}")
        print("\n请修改脚本中的 visual_dir 路径")
        return

    # 查找所有图像文件
    image_files = list(Path(visual_dir).glob("*.png")) + list(Path(visual_dir).glob("*.jpg"))

    if len(image_files) == 0:
        print(f"警告：在 {visual_dir} 中没有找到图像文件")
        print("\n此脚本需要V3.0的可视化结果图像")
        print("如果你有.npy格式的轮廓数据，请修改脚本以支持")
        return

    print(f"\n找到 {len(image_files)} 张图像")

    # 注意：这个脚本假设你已经有V3.0的预测结果
    # 如果你有.npy格式的轮廓数据，可以直接加载
    # 如果只有可视化图像，需要先运行V3.0推理生成轮廓数据

    print("\n" + "=" * 60)
    print("重要提示：")
    print("=" * 60)
    print("此脚本需要V3.0模型的预测轮廓数据（.npy格式）")
    print("请先运行V3.0推理，保存轮廓数据，然后修改此脚本加载数据")
    print("\n建议的工作流程：")
    print("1. 运行 infer_v3_refinement.py，保存预测轮廓为.npy")
    print("2. 修改此脚本，加载.npy文件")
    print("3. 应用边缘感知平滑")
    print("4. 可视化对比")
    print("=" * 60)

    # 示例：如果你有轮廓数据
    # contours_path = "path/to/contours.npy"
    # if os.path.exists(contours_path):
    #     contours = load_contours_from_npy(contours_path)
    #     smoothed = process_single_image(image_path, contours, curvature_threshold, iterations)
    #     # 可视化
    #     visualize_comparison(image, contours, smoothed, output_path)


def demo_with_synthetic_data():
    """
    使用合成数据演示边缘感知平滑的效果
    """
    print("\n" + "=" * 60)
    print("演示：使用合成数据测试边缘感知平滑")
    print("=" * 60)

    # 创建一个带噪声的矩形轮廓
    num_points = 128

    # 矩形的四个角
    corners = np.array([
        [50, 50],
        [200, 50],
        [200, 150],
        [50, 150]
    ])

    # 在四条边上均匀采样点
    points_per_edge = num_points // 4
    contour_points = []

    for i in range(4):
        start = corners[i]
        end = corners[(i + 1) % 4]
        edge_points = np.linspace(start, end, points_per_edge, endpoint=False)
        contour_points.append(edge_points)

    contour = np.vstack(contour_points)

    # 添加噪声模拟毛刺
    noise = np.random.randn(num_points, 2) * 3
    noisy_contour = contour + noise

    # 应用平滑
    smoothed_contour = smooth_contours_numpy(
        noisy_contour,
        curvature_threshold=5.0,
        iterations=2
    )

    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 创建空白图像
    img = np.ones((250, 250, 3), dtype=np.uint8) * 255

    # 原始（无噪声）
    img1 = img.copy()
    cv2.polylines(img1, [contour.astype(np.int32)], True, (0, 0, 255), 2)
    axes[0].imshow(img1)
    axes[0].set_title('Ground Truth (Red)', fontsize=14)
    axes[0].axis('off')

    # 带噪声
    img2 = img.copy()
    cv2.polylines(img2, [noisy_contour.astype(np.int32)], True, (0, 255, 0), 2)
    axes[1].imshow(img2)
    axes[1].set_title('Noisy Prediction (Green)', fontsize=14)
    axes[1].axis('off')

    # 平滑后
    img3 = img.copy()
    cv2.polylines(img3, [smoothed_contour.astype(np.int32)], True, (255, 0, 0), 2)
    axes[2].imshow(img3)
    axes[2].set_title('Edge-Aware Smoothed (Blue)', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()

    output_path = "visual/edge_aware_smoothing_demo.png"
    os.makedirs("visual", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n演示结果已保存到: {output_path}")

    # 计算指标
    noise_error = np.mean(np.linalg.norm(noisy_contour - contour, axis=1))
    smoothed_error = np.mean(np.linalg.norm(smoothed_contour - contour, axis=1))

    print(f"\n定量评估：")
    print(f"  噪声轮廓误差: {noise_error:.4f} 像素")
    print(f"  平滑后误差:   {smoothed_error:.4f} 像素")
    print(f"  误差降低:     {(noise_error - smoothed_error) / noise_error * 100:.2f}%")


if __name__ == "__main__":
    # 先运行演示
    demo_with_synthetic_data()

    print("\n" + "=" * 60)
    print("接下来可以处理真实的V3.0预测结果")
    print("=" * 60)

    # 处理真实数据（需要先有V3.0的预测结果）
    # main()
