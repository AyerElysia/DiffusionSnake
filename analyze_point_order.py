"""
点序问题深度分析

专门分析点序是否导致毛刺的脚本：
1. 可视化点的编号和顺序
2. 检测点序跳跃（相邻点距离异常大）
3. 分析点序重排后的效果
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import cdist

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)


def detect_order_jumps(contour, threshold_multiplier=3.0):
    """
    检测点序中的异常跳跃

    Args:
        contour: (P, 2) numpy array
        threshold_multiplier: 阈值倍数

    Returns:
        jump_indices: 跳跃点的索引列表
        jump_distances: 跳跃距离列表
    """
    # 计算相邻点距离
    dists = np.linalg.norm(np.diff(contour, axis=0, append=contour[:1]), axis=1)

    # 计算阈值：均值 + threshold_multiplier * 标准差
    mean_dist = np.mean(dists)
    std_dist = np.std(dists)
    threshold = mean_dist + threshold_multiplier * std_dist

    # 找出异常跳跃
    jump_mask = dists > threshold
    jump_indices = np.where(jump_mask)[0]
    jump_distances = dists[jump_mask]

    return jump_indices, jump_distances, dists


def reorder_contour_by_nearest_neighbor(contour, start_idx=0):
    """
    使用最近邻算法重排点序

    Args:
        contour: (P, 2) numpy array
        start_idx: 起始点索引

    Returns:
        reordered: 重排后的轮廓
        order: 新的点序索引
    """
    P = len(contour)
    visited = np.zeros(P, dtype=bool)
    order = []

    current_idx = start_idx
    visited[current_idx] = True
    order.append(current_idx)

    for _ in range(P - 1):
        # 找到未访问点中距离当前点最近的
        unvisited_mask = ~visited
        unvisited_indices = np.where(unvisited_mask)[0]

        if len(unvisited_indices) == 0:
            break

        dists = np.linalg.norm(contour[unvisited_indices] - contour[current_idx], axis=1)
        nearest_local_idx = np.argmin(dists)
        nearest_idx = unvisited_indices[nearest_local_idx]

        visited[nearest_idx] = True
        order.append(nearest_idx)
        current_idx = nearest_idx

    reordered = contour[order]
    return reordered, np.array(order)


def visualize_point_order(img, contour, title, save_path, show_numbers=True, highlight_jumps=None):
    """
    可视化点序

    Args:
        img: 背景图像
        contour: (P, 2) numpy array
        title: 标题
        save_path: 保存路径
        show_numbers: 是否显示点编号
        highlight_jumps: 需要高亮的跳跃点索引
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # 显示图像
    if img is not None:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 绘制轮廓线
    ax.plot(np.append(contour[:, 0], contour[0, 0]),
            np.append(contour[:, 1], contour[0, 1]),
            'b-', linewidth=1, alpha=0.5, label='Contour')

    # 绘制点
    ax.scatter(contour[:, 0], contour[:, 1], c='red', s=30, alpha=0.6, zorder=5)

    # 标注点编号（每隔几个点标一次，避免太密集）
    if show_numbers:
        P = len(contour)
        step = max(1, P // 20)  # 最多显示20个编号
        for i in range(0, P, step):
            ax.text(contour[i, 0], contour[i, 1], str(i),
                   fontsize=8, color='yellow', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    # 高亮起始点
    ax.scatter(contour[0, 0], contour[0, 1], c='lime', s=100, marker='*',
              edgecolors='black', linewidths=2, zorder=10, label='Start (0)')

    # 高亮跳跃点
    if highlight_jumps is not None and len(highlight_jumps) > 0:
        jump_points = contour[highlight_jumps]
        ax.scatter(jump_points[:, 0], jump_points[:, 1], c='orange', s=150,
                  marker='x', linewidths=3, zorder=10, label=f'Jumps ({len(highlight_jumps)})')

        # 绘制跳跃连线
        for idx in highlight_jumps:
            next_idx = (idx + 1) % len(contour)
            ax.plot([contour[idx, 0], contour[next_idx, 0]],
                   [contour[idx, 1], contour[next_idx, 1]],
                   'r-', linewidth=3, alpha=0.7)

    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.axis('equal')
    if img is None:
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_point_order_issue(contour, img, save_prefix):
    """
    完整的点序问题分析

    Args:
        contour: (P, 2) numpy array
        img: 背景图像
        save_prefix: 保存路径前缀

    Returns:
        analysis_results: 分析结果字典
    """
    print(f"\n{'='*60}")
    print("点序问题分析")
    print(f"{'='*60}")

    # 1. 检测点序跳跃
    jump_indices, jump_distances, all_dists = detect_order_jumps(contour)

    print(f"\n[1] 点序跳跃检测:")
    print(f"    总点数: {len(contour)}")
    print(f"    平均点间距: {np.mean(all_dists):.2f}")
    print(f"    标准差: {np.std(all_dists):.2f}")
    print(f"    最大点间距: {np.max(all_dists):.2f}")
    print(f"    检测到跳跃: {len(jump_indices)} 处")

    if len(jump_indices) > 0:
        print(f"    跳跃位置: {jump_indices.tolist()}")
        print(f"    跳跃距离: {[f'{d:.2f}' for d in jump_distances]}")

    # 2. 可视化原始点序
    visualize_point_order(
        img, contour,
        f"Original Point Order (Jumps: {len(jump_indices)})",
        f"{save_prefix}_original_order.png",
        show_numbers=True,
        highlight_jumps=jump_indices
    )

    # 3. 重排点序（最近邻）
    reordered, new_order = reorder_contour_by_nearest_neighbor(contour)
    jump_indices_reordered, jump_distances_reordered, all_dists_reordered = detect_order_jumps(reordered)

    print(f"\n[2] 重排后（最近邻）:")
    print(f"    平均点间距: {np.mean(all_dists_reordered):.2f}")
    print(f"    标准差: {np.std(all_dists_reordered):.2f}")
    print(f"    最大点间距: {np.max(all_dists_reordered):.2f}")
    print(f"    检测到跳跃: {len(jump_indices_reordered)} 处")

    # 4. 可视化重排后的点序
    visualize_point_order(
        img, reordered,
        f"Reordered by Nearest Neighbor (Jumps: {len(jump_indices_reordered)})",
        f"{save_prefix}_reordered.png",
        show_numbers=True,
        highlight_jumps=jump_indices_reordered
    )

    # 5. 对比可视化
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # 原始
    axes[0].scatter(contour[:, 0], contour[:, 1], c=np.arange(len(contour)),
                   cmap='rainbow', s=30, alpha=0.8)
    axes[0].plot(np.append(contour[:, 0], contour[0, 0]),
                np.append(contour[:, 1], contour[0, 1]),
                'k-', linewidth=1, alpha=0.3)
    axes[0].set_title(f'Original Order (colored by index)\nJumps: {len(jump_indices)}', fontsize=14)
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)

    # 重排
    axes[1].scatter(reordered[:, 0], reordered[:, 1], c=np.arange(len(reordered)),
                   cmap='rainbow', s=30, alpha=0.8)
    axes[1].plot(np.append(reordered[:, 0], reordered[0, 0]),
                np.append(reordered[:, 1], reordered[0, 1]),
                'k-', linewidth=1, alpha=0.3)
    axes[1].set_title(f'Reordered (colored by new index)\nJumps: {len(jump_indices_reordered)}', fontsize=14)
    axes[1].axis('equal')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 6. 统计结果
    results = {
        'original': {
            'num_points': len(contour),
            'mean_dist': float(np.mean(all_dists)),
            'std_dist': float(np.std(all_dists)),
            'max_dist': float(np.max(all_dists)),
            'num_jumps': len(jump_indices),
            'jump_ratio': len(jump_indices) / len(contour),
        },
        'reordered': {
            'mean_dist': float(np.mean(all_dists_reordered)),
            'std_dist': float(np.std(all_dists_reordered)),
            'max_dist': float(np.max(all_dists_reordered)),
            'num_jumps': len(jump_indices_reordered),
            'jump_ratio': len(jump_indices_reordered) / len(reordered),
        },
        'improvement': {
            'jump_reduction': len(jump_indices) - len(jump_indices_reordered),
            'max_dist_reduction': float(np.max(all_dists) - np.max(all_dists_reordered)),
            'std_reduction': float(np.std(all_dists) - np.std(all_dists_reordered)),
        }
    }

    print(f"\n[3] 改进效果:")
    print(f"    跳跃减少: {results['improvement']['jump_reduction']} 处")
    print(f"    最大距离减少: {results['improvement']['max_dist_reduction']:.2f}")
    print(f"    标准差减少: {results['improvement']['std_reduction']:.2f}")

    print(f"\n{'='*60}\n")

    return results, reordered


if __name__ == "__main__":
    # 测试代码
    print("点序分析模块测试")

    # 创建一个测试轮廓（故意打乱顺序）
    num_points = 64
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radius = 50
    x = 100 + radius * np.cos(theta)
    y = 100 + radius * np.sin(theta)
    contour = np.stack([x, y], axis=1)

    # 打乱顺序（模拟点序问题）
    np.random.seed(42)
    shuffle_indices = np.random.permutation(num_points)
    contour_shuffled = contour[shuffle_indices]

    # 分析
    results, reordered = analyze_point_order_issue(
        contour_shuffled,
        None,
        "visual/test_order"
    )

    print("测试完成！")
