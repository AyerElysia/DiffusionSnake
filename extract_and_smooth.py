"""
从V3.0的可视化图像中提取轮廓，然后应用边缘感知平滑
这样不需要加载模型，避免GPU内存问题
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from edge_smoothing import smooth_contours_numpy


def extract_contours_from_image(image_path):
    """
    从可视化图像中提取绿色轮廓

    Args:
        image_path: 图像路径
    Returns:
        contours: 提取的轮廓列表
        image: 原始图像
    """
    # 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None

    # 转换到HSV空间，提取绿色
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 绿色的HSV范围
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # 创建掩码
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 转换格式
    contours_list = []
    for contour in contours:
        # 过滤太小的轮廓
        if len(contour) < 20:
            continue
        # 转换为 (P, 2) 格式
        contour_pts = contour.squeeze()
        if contour_pts.ndim == 2:
            contours_list.append(contour_pts.astype(np.float32))

    return contours_list, img


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
        cv2.polylines(img_overlay, [contour_int], True, (255, 0, 0), 2)

    axes[0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('V3.0 Original (Green)', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(img_smoothed, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Edge-Aware Smoothed (Red)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Overlay Comparison', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("从V3.0可视化图像提取轮廓并应用边缘感知平滑")
    print("=" * 60)

    # 输入输出路径
    input_image = "visual/single_sample_all_models/v3_overfit_single_gpu7_e10000/v3_overfit_single_gpu7_e10000_idx0_epoch10000.png"
    output_dir = "visual/v3_edge_aware_smoothed_extracted"

    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"输入图像: {input_image}")
    print(f"输出目录: {output_dir}")

    # 提取轮廓
    print("\n提取轮廓...")
    contours, image = extract_contours_from_image(input_image)

    if contours is None or len(contours) == 0:
        print("错误: 无法从图像中提取轮廓")
        return

    print(f"提取到 {len(contours)} 个轮廓")
    for i, c in enumerate(contours):
        print(f"  轮廓 {i}: {len(c)} 个点")

    # 应用边缘感知平滑
    print("\n应用边缘感知平滑...")
    smoothed_contours = []

    for i, contour in enumerate(contours):
        print(f"  处理轮廓 {i}...")
        smoothed = smooth_contours_numpy(
            contour,
            curvature_threshold=5.0,
            iterations=2
        )
        smoothed_contours.append(smoothed)

        # 计算平滑效果
        diff = np.mean(np.linalg.norm(contour - smoothed, axis=1))
        print(f"    平均位移: {diff:.2f} 像素")

    # 可视化
    print("\n生成可视化...")
    output_path = f"{output_dir}/comparison.png"
    visualize_comparison(image, contours, smoothed_contours, output_path)

    print(f"\n结果已保存到: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
