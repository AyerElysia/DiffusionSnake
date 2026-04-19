"""
验证假设：小轮廓毛刺是否由"位移小"导致的数值不稳定引起

实验设计：
-----------
假设A（点密度高）：小轮廓128点太密，相邻点距离小，噪声敏感
假设B（位移小）：小轮廓每步演化的位移量小，导致数值不稳定

验证逻辑：
---------
1. 统计不同大小轮廓的实际位移量（GT - Init）
2. 分析位移量与轮廓大小、毛刺程度的关系
3. 关键判断：
   - 如果小轮廓位移确实很小，且位移与曲率强相关 → 支持假设B
   - 如果小轮廓位移不小，或位移与曲率弱相关 → 不支持假设B，更可能是假设A

实验步骤：
---------
1. 加载V3.4单样本数据
2. 计算每个轮廓的：
   - 初始化轮廓（八边形）
   - GT轮廓
   - 位移向量（GT - Init）
   - 位移统计量：平均位移、最大位移、位移标准差
3. 分析位移量与轮廓大小、毛刺的相关性
4. 可视化位移分布
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

# 设置环境
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJ_DIR)

os.environ['CFG_FILE'] = os.path.join(_PROJ_DIR, 'configs/btcv_diffusion_dit_v3_4_single_overfit.yaml')

from lib.config import cfg
from lib.datasets.make_dataset import make_dataset
from lib.datasets.collate_batch import make_collator
from lib.datasets.transforms import make_transforms
from lib.utils.snake import snake_config, snake_gcn_utils


def compute_displacement_stats(init_contour, gt_contour):
    """
    计算位移统计量

    Args:
        init_contour: [N, 2] 初始化轮廓
        gt_contour: [N, 2] GT轮廓

    Returns:
        dict: 位移统计量
    """
    # 计算位移向量
    displacement = gt_contour - init_contour  # [N, 2]

    # 位移幅度
    disp_magnitude = np.linalg.norm(displacement, axis=1)  # [N]

    # 统计量
    stats = {
        'mean_disp': float(np.mean(disp_magnitude)),
        'max_disp': float(np.max(disp_magnitude)),
        'min_disp': float(np.min(disp_magnitude)),
        'std_disp': float(np.std(disp_magnitude)),
        'median_disp': float(np.median(disp_magnitude)),
        'disp_cv': float(np.std(disp_magnitude) / (np.mean(disp_magnitude) + 1e-8)),  # 变异系数
    }

    return stats, displacement, disp_magnitude


def compute_contour_size(contour):
    """计算轮廓大小"""
    # 周长
    perimeter = np.sum(np.linalg.norm(np.diff(contour, axis=0, append=contour[:1]), axis=1))

    # 面积
    x = contour[:, 0]
    y = contour[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    return perimeter, area


def main():
    print('='*100)
    print('验证假设：小轮廓毛刺是否由"位移小"导致')
    print('='*100)

    # 1. 加载数据
    print('\n[1/4] 加载数据...')
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)
    batch = collator([dataset[0]])

    # 2. 获取GT轮廓和初始化轮廓
    print('[2/4] 提取轮廓...')

    # GT轮廓
    gt_all = batch['i_gt_py']
    dr = float(snake_config.down_ratio)
    gt_np = gt_all.cpu().numpy() * dr  # [1, num_contours, 128, 2]

    # 初始化轮廓（八边形）
    init_dict = snake_gcn_utils.prepare_training({}, batch)
    init_all = init_dict['i_it_py']
    init_np = init_all.cpu().numpy() * dr  # [1, num_contours, 128, 2]

    # 加载毛刺指标
    metrics_path = os.path.join(_PROJ_DIR, 'visual/burr_v3_4_full/full_image_metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # 3. 分析每个轮廓
    print('[3/4] 分析位移统计...')

    results = []

    for i in range(len(gt_np[0])):
        gt_contour = gt_np[0][i]
        init_contour = init_np[0][i]

        # 计算位移统计
        disp_stats, displacement, disp_magnitude = compute_displacement_stats(init_contour, gt_contour)

        # 计算轮廓大小
        perimeter, area = compute_contour_size(gt_contour)

        # 获取毛刺指标
        m = metrics[i]

        results.append({
            'id': i,
            'perimeter': perimeter,
            'area': area,
            'point_density': perimeter / 128,
            'curv_max': m['raw']['curv_max'],
            'sharp_angles': m['raw']['sharp_angles'],
            **disp_stats,
            'displacement': displacement,
            'disp_magnitude': disp_magnitude,
        })

    # 4. 打印结果
    print('\n' + '='*100)
    print('位移统计结果')
    print('='*100)

    print(f"\n{'轮廓':<6} {'面积':<10} {'点密度':<10} {'平均位移':<12} {'最大位移':<12} {'位移CV':<12} {'最大曲率':<12}")
    print('-'*100)

    for r in sorted(results, key=lambda x: x['area']):
        print(f"{r['id']:<6} {r['area']:<10.1f} {r['point_density']:<10.2f} "
              f"{r['mean_disp']:<12.2f} {r['max_disp']:<12.2f} {r['disp_cv']:<12.2f} {r['curv_max']:<12.2f}")

    # 5. 相关性分析
    print('\n' + '='*100)
    print('相关性分析')
    print('='*100)

    areas = [r['area'] for r in results]
    point_densities = [r['point_density'] for r in results]
    mean_disps = [r['mean_disp'] for r in results]
    max_disps = [r['max_disp'] for r in results]
    disp_cvs = [r['disp_cv'] for r in results]
    curvs = [r['curv_max'] for r in results]

    # 计算相关系数
    corr_area_disp = np.corrcoef(areas, mean_disps)[0, 1]
    corr_disp_curv = np.corrcoef(mean_disps, curvs)[0, 1]
    corr_density_disp = np.corrcoef(point_densities, mean_disps)[0, 1]
    corr_area_curv = np.corrcoef(areas, curvs)[0, 1]

    print(f"\n面积 vs 平均位移: {corr_area_disp:.3f}")
    print(f"点密度 vs 平均位移: {corr_density_disp:.3f}")
    print(f"平均位移 vs 最大曲率: {corr_disp_curv:.3f}")
    print(f"面积 vs 最大曲率: {corr_area_curv:.3f}")

    # 6. 假设验证
    print('\n' + '='*100)
    print('假设验证')
    print('='*100)

    print("\n假设B（位移小导致不稳定）的预测：")
    print("  1. 小轮廓的位移应该明显小于大轮廓")
    print("  2. 位移量与曲率应该有强相关性（位移小 → 曲率大）")

    print("\n实际观察：")

    # 判断1：小轮廓位移是否明显更小
    small_contours = [r for r in results if r['area'] < 2000]
    large_contours = [r for r in results if r['area'] > 3000]

    if small_contours and large_contours:
        small_mean_disp = np.mean([r['mean_disp'] for r in small_contours])
        large_mean_disp = np.mean([r['mean_disp'] for r in large_contours])
        disp_ratio = small_mean_disp / large_mean_disp

        print(f"  1. 小轮廓平均位移: {small_mean_disp:.2f}")
        print(f"     大轮廓平均位移: {large_mean_disp:.2f}")
        print(f"     比值: {disp_ratio:.2f}")

        if disp_ratio < 0.7:
            print("     → 小轮廓位移明显更小（支持假设B）")
        elif disp_ratio > 0.9:
            print("     → 小轮廓位移与大轮廓相当（不支持假设B）")
        else:
            print("     → 小轮廓位移略小（弱支持假设B）")

    # 判断2：位移与曲率的相关性
    print(f"\n  2. 位移 vs 曲率相关系数: {corr_disp_curv:.3f}")

    if abs(corr_disp_curv) > 0.5:
        if corr_disp_curv < 0:
            print("     → 位移小 → 曲率大（强支持假设B）")
        else:
            print("     → 位移大 → 曲率大（不支持假设B）")
    else:
        print("     → 位移与曲率相关性弱（不支持假设B）")

    # 综合判断
    print("\n" + "="*100)
    print("综合结论")
    print("="*100)

    support_b = 0

    if small_contours and large_contours and disp_ratio < 0.7:
        support_b += 1

    if corr_disp_curv < -0.5:
        support_b += 1

    print(f"\n支持假设B的证据数量: {support_b}/2")

    if support_b >= 2:
        print("\n✓✓✓ 强支持假设B：小轮廓毛刺主要由位移小导致的数值不稳定引起")
        print("\n建议方案：")
        print("  1. 增加位移的归一化/缩放，让小轮廓和大轮廓的位移在相同尺度")
        print("  2. 对小轮廓使用更大的学习率或步长")
        print("  3. 考虑基于轮廓大小的自适应演化策略")
    elif support_b == 1:
        print("\n○ 弱支持假设B：位移小可能是一个因素，但不是主要原因")
        print("\n建议方案：")
        print("  1. 同时考虑点密度和位移两个因素")
        print("  2. 优先尝试自适应点数（假设A的方案）")
        print("  3. 如果效果不佳，再考虑位移归一化")
    else:
        print("\n✗ 不支持假设B：位移小不是主要原因")
        print("\n更可能的原因：")
        print("  1. 点密度过高（假设A）")
        print("  2. 特征图分辨率不足")
        print("  3. 模型对小目标的表达能力不足")
        print("\n建议方案：")
        print("  1. 优先尝试自适应点数（降低小轮廓的点密度）")
        print("  2. 提高特征图分辨率")
        print("  3. 增强小目标的特征提取")

    # 7. 可视化
    print('\n[4/4] 生成可视化...')

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 面积 vs 平均位移
    ax = axes[0, 0]
    ax.scatter(areas, mean_disps, s=100, alpha=0.7, c=curvs, cmap='hot')
    for r in results:
        ax.annotate(f"C{r['id']}", (r['area'], r['mean_disp']), fontsize=10)
    ax.set_xlabel('Area (pixels²)')
    ax.set_ylabel('Mean Displacement (pixels)')
    ax.set_title(f'Area vs Mean Displacement (corr={corr_area_disp:.3f})')
    ax.grid(True, alpha=0.3)

    # 2. 平均位移 vs 最大曲率
    ax = axes[0, 1]
    ax.scatter(mean_disps, curvs, s=100, alpha=0.7, c=areas, cmap='viridis')
    for r in results:
        ax.annotate(f"C{r['id']}", (r['mean_disp'], r['curv_max']), fontsize=10)
    ax.set_xlabel('Mean Displacement (pixels)')
    ax.set_ylabel('Max Curvature')
    ax.set_title(f'Displacement vs Curvature (corr={corr_disp_curv:.3f})')
    ax.grid(True, alpha=0.3)

    # 3. 点密度 vs 平均位移
    ax = axes[0, 2]
    ax.scatter(point_densities, mean_disps, s=100, alpha=0.7, c=curvs, cmap='hot')
    for r in results:
        ax.annotate(f"C{r['id']}", (r['point_density'], r['mean_disp']), fontsize=10)
    ax.set_xlabel('Point Density (perimeter/128)')
    ax.set_ylabel('Mean Displacement (pixels)')
    ax.set_title(f'Point Density vs Displacement (corr={corr_density_disp:.3f})')
    ax.grid(True, alpha=0.3)

    # 4. 位移分布（小轮廓 vs 大轮廓）
    ax = axes[1, 0]
    if small_contours and large_contours:
        small_disps = np.concatenate([r['disp_magnitude'] for r in small_contours])
        large_disps = np.concatenate([r['disp_magnitude'] for r in large_contours])
        ax.hist(small_disps, bins=30, alpha=0.5, label='Small contours', color='red')
        ax.hist(large_disps, bins=30, alpha=0.5, label='Large contours', color='blue')
        ax.set_xlabel('Displacement Magnitude (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title('Displacement Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 5. 位移CV vs 曲率
    ax = axes[1, 1]
    ax.scatter(disp_cvs, curvs, s=100, alpha=0.7, c=areas, cmap='viridis')
    for r in results:
        ax.annotate(f"C{r['id']}", (r['disp_cv'], r['curv_max']), fontsize=10)
    ax.set_xlabel('Displacement CV (std/mean)')
    ax.set_ylabel('Max Curvature')
    ax.set_title('Displacement Variability vs Curvature')
    ax.grid(True, alpha=0.3)

    # 6. 综合对比
    ax = axes[1, 2]
    x = range(len(results))
    sorted_results = sorted(results, key=lambda x: x['area'])

    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))

    areas_sorted = [r['area'] for r in sorted_results]
    disps_sorted = [r['mean_disp'] for r in sorted_results]
    curvs_sorted = [r['curv_max'] for r in sorted_results]
    labels = [f"C{r['id']}" for r in sorted_results]

    p1 = ax.plot(x, areas_sorted, 'b-o', label='Area', linewidth=2)
    p2 = ax2.plot(x, disps_sorted, 'g-s', label='Displacement', linewidth=2)
    p3 = ax3.plot(x, curvs_sorted, 'r-^', label='Curvature', linewidth=2)

    ax.set_xlabel('Contour (sorted by area)')
    ax.set_ylabel('Area (pixels²)', color='b')
    ax2.set_ylabel('Mean Displacement (pixels)', color='g')
    ax3.set_ylabel('Max Curvature', color='r')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='g')
    ax3.tick_params(axis='y', labelcolor='r')
    ax.set_title('Area vs Displacement vs Curvature')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = os.path.join(_PROJ_DIR, 'visual/displacement_analysis')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'displacement_hypothesis_verification.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    print(f'\n可视化已保存: {output_path}')

    # 保存数据
    output_json = os.path.join(output_dir, 'displacement_stats.json')
    with open(output_json, 'w') as f:
        # 移除numpy数组，只保存标量，并转换numpy类型为Python类型
        results_json = []
        for r in results:
            r_copy = {}
            for k, v in r.items():
                if not isinstance(v, np.ndarray):
                    # 转换numpy标量为Python标量
                    if isinstance(v, (np.integer, np.floating)):
                        r_copy[k] = float(v)
                    else:
                        r_copy[k] = v
            results_json.append(r_copy)
        json.dump(results_json, f, indent=2)

    print(f'数据已保存: {output_json}')

    print('\n' + '='*100)
    print('分析完成！')
    print('='*100)


if __name__ == '__main__':
    main()
