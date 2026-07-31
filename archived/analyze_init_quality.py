"""
分析训练初始化质量
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

def compute_distance_metrics(init_points, gt_points):
    """计算初始化点到真实标注点的距离指标"""
    # 计算每个点的欧氏距离
    distances = np.sqrt(np.sum((init_points - gt_points) ** 2, axis=1))

    return {
        'mean_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'min_distance': np.min(distances),
        'std_distance': np.std(distances),
        'median_distance': np.median(distances),
    }

def main():
    cfg.merge_from_file('configs/btcv_diffusion_dit_v3.yaml')
    cfg.train.data_path = '/mnt/sdb1/leijh/DiffusionSnake/Datasets/BTCV/btcv_png_new_snake'

    dataset = make_dataset(cfg, 'BtcvTrain', make_transforms(cfg, is_train=True), is_train=True)

    print("=" * 70)
    print("训练初始化质量分析")
    print("=" * 70)

    # 分析前10个样本
    num_samples = min(10, len(dataset))
    all_metrics = []

    for idx in range(num_samples):
        batch = make_collator(cfg)([dataset[idx]])
        init_data = snake_gcn_utils.prepare_training({'detection': torch.zeros((1, 100, 6))}, batch)

        i_init = init_data['i_it_py'][0].numpy() * snake_config.down_ratio
        i_gt = init_data['i_gt_py'][0].numpy() * snake_config.down_ratio

        metrics = compute_distance_metrics(i_init, i_gt)
        all_metrics.append(metrics)

        print(f"\n样本 {idx}:")
        print(f"  平均距离: {metrics['mean_distance']:.2f} 像素")
        print(f"  最大距离: {metrics['max_distance']:.2f} 像素")
        print(f"  最小距离: {metrics['min_distance']:.2f} 像素")
        print(f"  标准差:   {metrics['std_distance']:.2f} 像素")
        print(f"  中位数:   {metrics['median_distance']:.2f} 像素")

        # 可视化第一个样本
        if idx == 0:
            img = batch['orig_img'][0].astype(np.uint8).copy()

            # 绘制箭头（每4个点一个）
            for i in range(0, 128, 4):
                cv2.arrowedLine(img, tuple(i_init[i].astype(int)),
                              tuple(i_gt[i].astype(int)), (255, 255, 255), 1, tipLength=0.3)

            # 绘制初始化轮廓（黄色）
            cv2.polylines(img, [i_init.astype(int)], True, (0, 255, 255), 2)

            # 绘制真实标注轮廓（蓝色）
            cv2.polylines(img, [i_gt.astype(int)], True, (255, 0, 0), 2)

            # 添加文字说明
            cv2.putText(img, f"Mean Dist: {metrics['mean_distance']:.1f}px",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Max Dist: {metrics['max_distance']:.1f}px",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, "Yellow: Init | Blue: GT",
                       (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            os.makedirs("visual", exist_ok=True)
            cv2.imwrite("visual/train_init_quality_detailed.png", img)
            print(f"\n详细可视化已保存到: visual/train_init_quality_detailed.png")

    # 计算总体统计
    print("\n" + "=" * 70)
    print("总体统计 (前 {} 个样本):".format(num_samples))
    print("=" * 70)

    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }

    print(f"平均距离: {avg_metrics['mean_distance']:.2f} ± {np.std([m['mean_distance'] for m in all_metrics]):.2f} 像素")
    print(f"最大距离: {avg_metrics['max_distance']:.2f} ± {np.std([m['max_distance'] for m in all_metrics]):.2f} 像素")
    print(f"中位数距离: {avg_metrics['median_distance']:.2f} ± {np.std([m['median_distance'] for m in all_metrics]):.2f} 像素")

    # 判断初始化质量
    print("\n" + "=" * 70)
    print("初始化质量评估:")
    print("=" * 70)

    if avg_metrics['mean_distance'] < 10:
        print("✓ 优秀: 平均距离 < 10 像素，初始化非常接近真实轮廓")
    elif avg_metrics['mean_distance'] < 20:
        print("✓ 良好: 平均距离 < 20 像素，初始化质量可接受")
    elif avg_metrics['mean_distance'] < 50:
        print("⚠ 一般: 平均距离 < 50 像素，初始化有一定偏差")
    else:
        print("✗ 较差: 平均距离 >= 50 像素，初始化偏差较大")

    if avg_metrics['max_distance'] < 30:
        print("✓ 最大偏差控制良好 (< 30 像素)")
    elif avg_metrics['max_distance'] < 100:
        print("⚠ 最大偏差中等 (< 100 像素)")
    else:
        print("✗ 存在较大偏差点 (>= 100 像素)")

if __name__ == '__main__':
    main()
