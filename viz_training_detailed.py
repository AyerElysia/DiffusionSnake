"""
详细的训练可视化 - 完全模拟训练过程
重点展示位移向量（训练目标）
"""
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from lib.config import cfg
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.datasets.collate_batch import make_collator
from lib.utils.snake import snake_config, snake_gcn_utils


def visualize_training_sample(dataset, idx, save_dir='visual'):
    """完整可视化一个训练样本，包括位移向量"""

    # 准备数据（完全按照训练流程）
    batch = make_collator(cfg)([dataset[idx]])

    # 模拟训练时的数据准备
    init_data = snake_gcn_utils.prepare_training(
        {'detection': torch.zeros((1, 100, 6))},
        batch
    )

    # 获取关键数据（转换到原始图像坐标）
    i_init = init_data['i_it_py'][0].numpy() * snake_config.down_ratio  # 初始化点
    i_gt = init_data['i_gt_py'][0].numpy() * snake_config.down_ratio    # 真实标注点

    # 计算位移向量（这就是训练目标！）
    displacement = i_gt - i_init  # shape: (128, 2)

    # 获取原始图像
    img = batch['orig_img'][0].astype(np.uint8).copy()
    H, W = img.shape[:2]

    # 创建大图：2x2布局
    fig = plt.figure(figsize=(20, 20))

    # ========== 子图1: 原始图像 + 初始化 + GT ==========
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 绘制初始化轮廓（黄色）
    init_polygon = plt.Polygon(i_init, fill=False, edgecolor='yellow',
                               linewidth=3, label='初始化轮廓')
    ax1.add_patch(init_polygon)

    # 绘制GT轮廓（蓝色）
    gt_polygon = plt.Polygon(i_gt, fill=False, edgecolor='blue',
                            linewidth=3, label='真实标注 (GT)')
    ax1.add_patch(gt_polygon)

    # 标注关键点
    ax1.scatter(i_init[:, 0], i_init[:, 1], c='yellow', s=30, alpha=0.6, zorder=5)
    ax1.scatter(i_gt[:, 0], i_gt[:, 1], c='blue', s=30, alpha=0.6, zorder=5)

    ax1.set_title('子图1: 初始化 vs 真实标注', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.axis('off')

    # ========== 子图2: 位移向量场（训练目标）==========
    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0.5)

    # 绘制初始化轮廓作为参考
    init_polygon2 = plt.Polygon(i_init, fill=False, edgecolor='yellow',
                                linewidth=2, linestyle='--', alpha=0.5)
    ax2.add_patch(init_polygon2)

    # 绘制位移向量（每2个点画一个，避免太密集）
    step = 2
    for i in range(0, len(i_init), step):
        # 计算位移大小用于颜色映射
        disp_mag = np.linalg.norm(displacement[i])

        # 颜色：红色表示大位移，绿色表示小位移
        color = plt.cm.RdYlGn_r(min(disp_mag / 30.0, 1.0))

        arrow = FancyArrowPatch(
            (i_init[i, 0], i_init[i, 1]),
            (i_gt[i, 0], i_gt[i, 1]),
            arrowstyle='->', mutation_scale=20,
            linewidth=2.5, color=color, alpha=0.8, zorder=10
        )
        ax2.add_patch(arrow)

    ax2.set_title('子图2: 位移向量场 (训练目标)', fontsize=16, fontweight='bold')
    ax2.text(10, 30, '红色=大位移, 绿色=小位移',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=12, color='black')
    ax2.axis('off')

    # ========== 子图3: 位移大小热力图 ==========
    ax3 = plt.subplot(2, 2, 3)

    # 计算每个点的位移大小
    disp_magnitude = np.linalg.norm(displacement, axis=1)

    # 创建热力图背景
    ax3.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0.3)

    # 绘制位移大小的散点图
    scatter = ax3.scatter(i_init[:, 0], i_init[:, 1],
                         c=disp_magnitude, cmap='hot',
                         s=100, alpha=0.8, edgecolors='white', linewidth=1)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('位移大小 (像素)', fontsize=12)

    # 绘制轮廓
    init_polygon3 = plt.Polygon(i_init, fill=False, edgecolor='cyan',
                                linewidth=2, linestyle='--')
    ax3.add_patch(init_polygon3)

    ax3.set_title('子图3: 位移大小分布', fontsize=16, fontweight='bold')
    ax3.axis('off')

    # ========== 子图4: 统计信息 ==========
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    # 计算统计信息
    disp_mean = np.mean(disp_magnitude)
    disp_max = np.max(disp_magnitude)
    disp_min = np.min(disp_magnitude)
    disp_std = np.std(disp_magnitude)
    disp_median = np.median(disp_magnitude)

    # 位移向量的x和y分量统计
    disp_x = displacement[:, 0]
    disp_y = displacement[:, 1]

    stats_text = f"""
训练样本统计信息
{'='*50}

【位移向量统计】（训练目标）
  平均位移: {disp_mean:.2f} 像素
  最大位移: {disp_max:.2f} 像素
  最小位移: {disp_min:.2f} 像素
  标准差:   {disp_std:.2f} 像素
  中位数:   {disp_median:.2f} 像素

【位移分量统计】
  X方向平均: {np.mean(disp_x):.2f} 像素
  Y方向平均: {np.mean(disp_y):.2f} 像素
  X方向范围: [{np.min(disp_x):.1f}, {np.max(disp_x):.1f}]
  Y方向范围: [{np.min(disp_y):.1f}, {np.max(disp_y):.1f}]

【轮廓点数】
  总点数: {len(i_init)}

【训练目标】
  模型需要学习预测从初始化点到GT点的位移向量
  Loss = MSE(predicted_disp, true_disp)

【质量评估】
  {'✓ 优秀' if disp_mean < 10 else '⚠ 一般' if disp_mean < 20 else '✗ 较差'}
  平均位移 {'< 10' if disp_mean < 10 else '< 20' if disp_mean < 20 else '>= 20'} 像素
"""

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=13, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 添加位移分布直方图
    ax4_hist = ax4.inset_axes([0.55, 0.05, 0.4, 0.35])
    ax4_hist.hist(disp_magnitude, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax4_hist.set_xlabel('位移大小 (像素)', fontsize=10)
    ax4_hist.set_ylabel('点数', fontsize=10)
    ax4_hist.set_title('位移分布直方图', fontsize=11)
    ax4_hist.grid(True, alpha=0.3)

    # 总标题
    fig.suptitle(f'训练样本 #{idx} - 完整可视化（重点：位移向量）',
                fontsize=20, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'training_detailed_sample_{idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 已保存详细可视化: {save_path}")

    return {
        'disp_mean': disp_mean,
        'disp_max': disp_max,
        'disp_std': disp_std,
    }


def main():
    cfg.merge_from_file('configs/btcv_diffusion_dit_v3.yaml')
    cfg.train.data_path = '/mnt/sdb1/leijh/DiffusionSnake/Datasets/BTCV/btcv_png_new_snake'

    dataset = make_dataset(cfg, 'BtcvTrain', make_transforms(cfg, is_train=True), is_train=True)

    print("=" * 70)
    print("训练可视化 - 重点展示位移向量（训练目标）")
    print("=" * 70)
    print(f"数据集大小: {len(dataset)} 个样本")
    print()

    # 可视化前5个样本
    num_samples = min(5, len(dataset))
    all_stats = []

    for idx in range(num_samples):
        print(f"\n处理样本 {idx}...")
        stats = visualize_training_sample(dataset, idx)
        all_stats.append(stats)

    # 总体统计
    print("\n" + "=" * 70)
    print("总体统计:")
    print("=" * 70)
    avg_disp = np.mean([s['disp_mean'] for s in all_stats])
    avg_max = np.mean([s['disp_max'] for s in all_stats])
    print(f"平均位移: {avg_disp:.2f} ± {np.std([s['disp_mean'] for s in all_stats]):.2f} 像素")
    print(f"最大位移: {avg_max:.2f} ± {np.std([s['disp_max'] for s in all_stats]):.2f} 像素")
    print()
    print("所有可视化已保存到 visual/ 目录")
    print("=" * 70)


if __name__ == '__main__':
    main()
