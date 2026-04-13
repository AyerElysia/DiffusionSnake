"""
清晰的训练可视化 - 重点展示位移向量
简洁、大图、清晰
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
    """清晰可视化一个训练样本"""

    # 准备数据（完全按照训练流程）
    batch = make_collator(cfg)([dataset[idx]])
    init_data = snake_gcn_utils.prepare_training(
        {'detection': torch.zeros((1, 100, 6))},
        batch
    )

    # 获取数据
    i_init = init_data['i_it_py'][0].numpy() * snake_config.down_ratio
    i_gt = init_data['i_gt_py'][0].numpy() * snake_config.down_ratio
    displacement = i_gt - i_init  # 训练目标

    img = batch['orig_img'][0].astype(np.uint8).copy()
    H, W = img.shape[:2]

    # 创建3个大图
    fig_size = (W * 3 // 100, H // 100)  # 自适应大小

    # ========== 图1: 轮廓对比 ==========
    img1 = img.copy()

    # 画初始化轮廓（黄色，粗线）
    cv2.polylines(img1, [i_init.astype(int)], True, (0, 255, 255), 4)

    # 画GT轮廓（蓝色，粗线）
    cv2.polylines(img1, [i_gt.astype(int)], True, (255, 0, 0), 4)

    # 添加图例
    cv2.putText(img1, "Yellow: Init", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(img1, "Blue: GT", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    # ========== 图2: 位移向量（稀疏显示，超大箭头）==========
    img2 = img.copy()

    # 只显示每8个点，箭头才清晰
    step = 8
    for i in range(0, len(i_init), step):
        start = tuple(i_init[i].astype(int))
        end = tuple(i_gt[i].astype(int))

        # 计算位移大小
        disp_mag = np.linalg.norm(displacement[i])

        # 根据位移大小选择颜色
        if disp_mag < 2:
            color = (0, 255, 0)  # 绿色：小位移
        elif disp_mag < 5:
            color = (0, 255, 255)  # 黄色：中等位移
        else:
            color = (0, 0, 255)  # 红色：大位移

        # 画超粗箭头
        cv2.arrowedLine(img2, start, end, color, 3, tipLength=0.4)

        # 在起点画圆点
        cv2.circle(img2, start, 5, (255, 255, 255), -1)

    # 添加说明
    cv2.putText(img2, "Displacement Vectors (Training Target)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(img2, "Green: <2px | Yellow: 2-5px | Red: >5px", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # ========== 图3: 位移大小可视化 ==========
    img3 = img.copy()

    # 计算位移大小
    disp_magnitude = np.linalg.norm(displacement, axis=1)

    # 在每个点上画圆，大小和颜色表示位移
    for i in range(len(i_init)):
        pos = tuple(i_init[i].astype(int))
        mag = disp_magnitude[i]

        # 圆的半径
        radius = int(max(3, min(mag * 2, 15)))

        # 颜色
        if mag < 2:
            color = (0, 255, 0)
        elif mag < 5:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.circle(img3, pos, radius, color, -1)
        cv2.circle(img3, pos, radius, (255, 255, 255), 1)

    # 画初始化轮廓作为参考
    cv2.polylines(img3, [i_init.astype(int)], True, (128, 128, 128), 2)

    # 统计信息
    stats_text = [
        f"Mean Disp: {np.mean(disp_magnitude):.2f}px",
        f"Max Disp: {np.max(disp_magnitude):.2f}px",
        f"Std: {np.std(disp_magnitude):.2f}px",
    ]

    for i, text in enumerate(stats_text):
        cv2.putText(img3, text, (20, 40 + i*40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # ========== 保存三张图 ==========
    os.makedirs(save_dir, exist_ok=True)

    cv2.imwrite(f"{save_dir}/sample_{idx}_1_contours.png", img1)
    cv2.imwrite(f"{save_dir}/sample_{idx}_2_displacement.png", img2)
    cv2.imwrite(f"{save_dir}/sample_{idx}_3_magnitude.png", img3)

    # 也保存一个横向拼接的大图
    combined = np.hstack([img1, img2, img3])
    cv2.imwrite(f"{save_dir}/sample_{idx}_combined.png", combined)

    print(f"✓ Sample {idx}:")
    print(f"  - Mean displacement: {np.mean(disp_magnitude):.2f}px")
    print(f"  - Max displacement: {np.max(disp_magnitude):.2f}px")
    print(f"  - Saved to {save_dir}/sample_{idx}_*.png")

    return np.mean(disp_magnitude), np.max(disp_magnitude)


def main():
    cfg.merge_from_file('configs/btcv_diffusion_dit_v3.yaml')
    cfg.train.data_path = '/mnt/sdb1/leijh/DiffusionSnake/Datasets/BTCV/btcv_png_new_snake'

    dataset = make_dataset(cfg, 'BtcvTrain', make_transforms(cfg, is_train=True), is_train=True)

    print("=" * 70)
    print("清晰的训练可视化 - 位移向量")
    print("=" * 70)
    print(f"数据集: {len(dataset)} 个样本")
    print()

    # 可视化前3个样本
    num_samples = min(3, len(dataset))
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
    print()
    print("生成的文件:")
    print("  - sample_X_1_contours.png      : 轮廓对比")
    print("  - sample_X_2_displacement.png  : 位移向量（重点）")
    print("  - sample_X_3_magnitude.png     : 位移大小")
    print("  - sample_X_combined.png        : 三合一大图")
    print("=" * 70)


if __name__ == '__main__':
    main()
