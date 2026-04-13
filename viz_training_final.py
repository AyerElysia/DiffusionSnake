"""
最终版训练可视化 - 放大轮廓，所有信息在一张图
"""
import os
import cv2
import torch
import numpy as np
from lib.config import cfg
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.datasets.collate_batch import make_collator
from lib.utils.snake import snake_config, snake_gcn_utils


def visualize_sample(dataset, idx, save_dir='visual'):
    """清晰可视化 - 放大轮廓区域"""

    # 准备数据
    batch = make_collator(cfg)([dataset[idx]])
    init_data = snake_gcn_utils.prepare_training(
        {'detection': torch.zeros((1, 100, 6))},
        batch
    )

    # 获取数据
    i_init = init_data['i_it_py'][0].numpy() * snake_config.down_ratio
    i_gt = init_data['i_gt_py'][0].numpy() * snake_config.down_ratio
    displacement = i_gt - i_init

    # 计算轮廓的边界框，并扩大一些
    all_points = np.vstack([i_init, i_gt])
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)

    # 扩大边界，留出边距
    margin = 50
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = x_max + margin
    y_max = y_max + margin

    # 计算放大后的画布大小（保持宽高比，但至少1000像素）
    width = x_max - x_min
    height = y_max - y_min
    scale = max(1000 / width, 1000 / height)

    canvas_w = int(width * scale)
    canvas_h = int(height * scale)

    # 创建白色画布
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # 坐标转换函数：从原始坐标到画布坐标
    def to_canvas(points):
        pts = points.copy()
        pts[:, 0] = (pts[:, 0] - x_min) * scale
        pts[:, 1] = (pts[:, 1] - y_min) * scale
        return pts.astype(int)

    i_init_canvas = to_canvas(i_init)
    i_gt_canvas = to_canvas(i_gt)

    # 计算位移统计
    disp_magnitude = np.linalg.norm(displacement, axis=1)
    mean_disp = np.mean(disp_magnitude)
    max_disp = np.max(disp_magnitude)

    # ========== 绘制 ==========

    # 1. 先画位移向量（每4个点一个，避免太密）
    step = 4
    for i in range(0, len(i_init), step):
        start = tuple(i_init_canvas[i])
        end = tuple(i_gt_canvas[i])
        mag = disp_magnitude[i]

        # 颜色
        if mag < 2:
            color = (0, 200, 0)  # 绿色
        elif mag < 5:
            color = (0, 165, 255)  # 橙色
        else:
            color = (0, 0, 255)  # 红色

        # 画粗箭头
        cv2.arrowedLine(canvas, start, end, color, 4, tipLength=0.3)

    # 2. 画初始化轮廓（黄色，虚线）
    cv2.polylines(canvas, [i_init_canvas], True, (0, 255, 255), 3, cv2.LINE_AA)

    # 3. 画GT轮廓（蓝色，实线）
    cv2.polylines(canvas, [i_gt_canvas], True, (255, 0, 0), 3, cv2.LINE_AA)

    # 4. 在初始化点上画小圆点
    for pt in i_init_canvas[::step]:
        cv2.circle(canvas, tuple(pt), 6, (0, 255, 255), -1)
        cv2.circle(canvas, tuple(pt), 6, (0, 0, 0), 1)

    # 5. 在GT点上画小圆点
    for pt in i_gt_canvas[::step]:
        cv2.circle(canvas, tuple(pt), 6, (255, 0, 0), -1)
        cv2.circle(canvas, tuple(pt), 6, (0, 0, 0), 1)

    # ========== 添加图例和统计信息 ==========
    legend_y = 40
    line_height = 50

    # 标题
    cv2.putText(canvas, f"Training Sample #{idx}", (20, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    legend_y += line_height + 20

    # 图例
    legends = [
        ("Yellow Circle + Line: Init", (0, 255, 255)),
        ("Blue Circle + Line: GT", (255, 0, 0)),
        ("Arrows: Displacement (Training Target)", (100, 100, 100)),
    ]

    for text, color in legends:
        cv2.putText(canvas, text, (20, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        legend_y += line_height

    legend_y += 20

    # 统计信息
    stats = [
        f"Mean Displacement: {mean_disp:.2f} px",
        f"Max Displacement: {max_disp:.2f} px",
        f"Num Points: {len(i_init)}",
    ]

    for text in stats:
        cv2.putText(canvas, text, (20, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        legend_y += line_height

    # 颜色说明
    legend_y += 20
    cv2.putText(canvas, "Arrow Colors:", (20, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    legend_y += line_height

    color_legends = [
        ("Green: < 2px", (0, 200, 0)),
        ("Orange: 2-5px", (0, 165, 255)),
        ("Red: > 5px", (0, 0, 255)),
    ]

    for text, color in color_legends:
        cv2.putText(canvas, text, (20, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        legend_y += line_height

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/training_sample_{idx}_zoomed.png"
    cv2.imwrite(save_path, canvas)

    print(f"✓ Sample {idx}:")
    print(f"  Mean displacement: {mean_disp:.2f}px")
    print(f"  Max displacement: {max_disp:.2f}px")
    print(f"  Canvas size: {canvas_w}x{canvas_h}")
    print(f"  Saved: {save_path}")

    return mean_disp, max_disp


def main():
    cfg.merge_from_file('configs/btcv_diffusion_dit_v3.yaml')
    cfg.train.data_path = '/mnt/sdb1/leijh/DiffusionSnake/Datasets/BTCV/btcv_png_new_snake'

    dataset = make_dataset(cfg, 'BtcvTrain', make_transforms(cfg, is_train=True), is_train=True)

    print("=" * 70)
    print("训练可视化 - 放大轮廓视图")
    print("=" * 70)
    print(f"数据集: {len(dataset)} 个样本")
    print()

    # 可视化前5个样本
    num_samples = min(5, len(dataset))
    stats = []

    for idx in range(num_samples):
        mean_disp, max_disp = visualize_sample(dataset, idx)
        stats.append((mean_disp, max_disp))
        print()

    print("=" * 70)
    print("总体统计:")
    avg_mean = np.mean([s[0] for s in stats])
    avg_max = np.mean([s[1] for s in stats])
    print(f"平均位移: {avg_mean:.2f}px")
    print(f"最大位移: {avg_max:.2f}px")
    print("=" * 70)


if __name__ == '__main__':
    main()
