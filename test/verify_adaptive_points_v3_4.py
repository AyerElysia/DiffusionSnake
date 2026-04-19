"""
使用V3.4模型验证自适应点数方案

实验设计：
---------
1. 用V3.4模型推理得到128点的预测结果
2. 根据轮廓大小，对预测结果进行自适应下采样
3. 对比调整前后的曲率变化

验证逻辑：
---------
如果点密度是主要原因：
  → 下采样后，小轮廓的曲率应该明显降低（>30%）
  → 大轮廓的曲率变化不大（因为本来点密度就合理）

如果点密度不是主要原因：
  → 下采样后，曲率不会明显改善
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import interp1d

# 设置环境
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJ_DIR)

os.environ['CFG_FILE'] = os.path.join(_PROJ_DIR, 'configs/btcv_diffusion_dit_v3_4_single_overfit.yaml')

from lib.config import cfg
from lib.utils.snake import snake_config


def compute_curvature(contour):
    """计算轮廓的曲率"""
    # 使用三点法计算曲率
    prev = np.roll(contour, 1, axis=0)
    next = np.roll(contour, -1, axis=0)

    # 向量
    v1 = prev - contour
    v2 = next - contour

    # 归一化
    v1_norm = np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8
    v2_norm = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8

    v1 = v1 / v1_norm
    v2 = v2 / v2_norm

    # 夹角余弦
    cos_angle = np.sum(v1 * v2, axis=1)
    cos_angle = np.clip(cos_angle, -1, 1)

    # 曲率（角度变化）
    curvature = 1 - cos_angle

    return curvature


def compute_metrics(contour):
    """计算轮廓指标"""
    curvature = compute_curvature(contour)

    # 周长
    perimeter = np.sum(np.linalg.norm(np.diff(contour, axis=0, append=contour[:1]), axis=1))

    # 面积
    x = contour[:, 0]
    y = contour[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    metrics = {
        'curv_max': float(np.max(curvature)),
        'curv_mean': float(np.mean(curvature)),
        'curv_std': float(np.std(curvature)),
        'sharp_angles': int(np.sum(curvature > 0.5)),  # 夹角 < 60度
        'perimeter': float(perimeter),
        'area': float(area),
    }

    return metrics


def adaptive_downsample(contour, target_num_points):
    """
    自适应下采样轮廓

    Args:
        contour: [N, 2] 原始轮廓
        target_num_points: 目标点数

    Returns:
        [M, 2] 下采样后的轮廓
    """
    if target_num_points >= len(contour):
        return contour

    # 计算累积弧长
    dists = np.linalg.norm(np.diff(contour, axis=0, append=contour[:1]), axis=1)
    cumsum = np.concatenate([[0], np.cumsum(dists)])
    total_length = cumsum[-1]

    # 均匀采样
    target_lengths = np.linspace(0, total_length, target_num_points, endpoint=False)

    # 插值
    interp_x = interp1d(cumsum, np.concatenate([contour[:, 0], [contour[0, 0]]]), kind='linear')
    interp_y = interp1d(cumsum, np.concatenate([contour[:, 1], [contour[0, 1]]]), kind='linear')

    new_x = interp_x(target_lengths)
    new_y = interp_y(target_lengths)

    new_contour = np.stack([new_x, new_y], axis=1)

    return new_contour


def adaptive_upsample(contour, target_num_points):
    """
    自适应上采样轮廓（用于可视化对比）

    Args:
        contour: [N, 2] 原始轮廓
        target_num_points: 目标点数

    Returns:
        [M, 2] 上采样后的轮廓
    """
    if target_num_points <= len(contour):
        return contour

    # 计算累积弧长
    dists = np.linalg.norm(np.diff(contour, axis=0, append=contour[:1]), axis=1)
    cumsum = np.concatenate([[0], np.cumsum(dists)])
    total_length = cumsum[-1]

    # 均匀采样
    target_lengths = np.linspace(0, total_length, target_num_points, endpoint=False)

    # 插值
    interp_x = interp1d(cumsum, np.concatenate([contour[:, 0], [contour[0, 0]]]), kind='cubic')
    interp_y = interp1d(cumsum, np.concatenate([contour[:, 1], [contour[0, 1]]]), kind='cubic')

    new_x = interp_x(target_lengths)
    new_y = interp_y(target_lengths)

    new_contour = np.stack([new_x, new_y], axis=1)

    return new_contour


def compute_adaptive_num_points(perimeter, target_density=2.5):
    """
    根据周长计算自适应点数

    Args:
        perimeter: 轮廓周长
        target_density: 目标点密度（像素/点）

    Returns:
        int: 建议的点数
    """
    num_points = int(perimeter / target_density)
    num_points = max(32, min(num_points, 256))  # 限制在[32, 256]
    return num_points


def main():
    print('='*100)
    print('使用V3.4模型验证自适应点数方案')
    print('='*100)

    # 1. 运行V3.4推理获取预测结果
    print('\n[1/5] 运行V3.4推理...')

    import torch
    from lib.datasets.make_dataset import make_dataset
    from lib.datasets.collate_batch import make_collator
    from lib.datasets.transforms import make_transforms
    from lib.networks import make_network
    from lib.utils.net_utils import load_network

    # 加载数据
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)
    batch = collator([dataset[0]])

    # 加载模型
    network = make_network(cfg).cuda()
    ckpt_path = os.path.join(_PROJ_DIR, 'data/outputs/btcv_diffusion_dit_v3_4_single_overfit/checkpoints/latest.pt')

    if not os.path.exists(ckpt_path):
        print(f'\n错误：找不到V3.4模型: {ckpt_path}')
        print('请确认模型路径是否正确')
        return

    # 直接加载checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cuda')
    if 'network' in checkpoint:
        network.load_state_dict(checkpoint['network'], strict=False)
    else:
        network.load_state_dict(checkpoint, strict=False)
    print(f'模型加载成功: {ckpt_path}')
    network.eval()

    # 推理
    with torch.no_grad():
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        output = network(batch['inp'], batch)

    # 提取预测轮廓 - 检查可能的键名
    if 'py_pred' in output:
        py_pred = output['py_pred']
    elif 'i_it_py' in output:
        py_pred = output['i_it_py']
    elif 'ex_pred' in output:
        py_pred = output['ex_pred']
    else:
        print(f'\n可用的输出键: {list(output.keys())}')
        print('错误：找不到预测轮廓')
        return

    # [1, num_contours, 128, 2]
    dr = float(snake_config.down_ratio)
    pred_contours = py_pred[0].cpu().numpy() * dr  # [num_contours, 128, 2]

    print(f'推理得到 {len(pred_contours)} 个预测轮廓')

    # 加载原始指标
    metrics_path = os.path.join(_PROJ_DIR, 'visual/burr_v3_4_full/full_image_metrics.json')
    with open(metrics_path, 'r') as f:
        original_metrics = json.load(f)

    # 2. 对每个轮廓进行自适应调整
    print('\n[2/5] 应用自适应点数调整...')

    results = []

    for i, contour in enumerate(pred_contours):
        # 原始指标
        orig_metrics = compute_metrics(contour)
        orig_perimeter = orig_metrics['perimeter']
        orig_density = orig_perimeter / 128

        # 计算自适应点数
        adaptive_points = compute_adaptive_num_points(orig_perimeter, target_density=2.5)

        # 下采样
        adapted_contour = adaptive_downsample(contour, adaptive_points)

        # 计算调整后的指标
        adapted_metrics = compute_metrics(adapted_contour)

        # 计算改善率
        curv_improvement = (orig_metrics['curv_max'] - adapted_metrics['curv_max']) / orig_metrics['curv_max'] * 100
        sharp_improvement = (orig_metrics['sharp_angles'] - adapted_metrics['sharp_angles']) / (orig_metrics['sharp_angles'] + 1) * 100

        results.append({
            'id': i,
            'perimeter': orig_perimeter,
            'area': orig_metrics['area'],
            'orig_points': 128,
            'orig_density': orig_density,
            'adaptive_points': adaptive_points,
            'adaptive_density': orig_perimeter / adaptive_points,
            'orig_curv_max': orig_metrics['curv_max'],
            'adapted_curv_max': adapted_metrics['curv_max'],
            'curv_improvement': curv_improvement,
            'orig_sharp_angles': orig_metrics['sharp_angles'],
            'adapted_sharp_angles': adapted_metrics['sharp_angles'],
            'sharp_improvement': sharp_improvement,
            'orig_contour': contour,
            'adapted_contour': adapted_contour,
        })

    # 3. 打印结果
    print('\n' + '='*100)
    print('自适应点数调整结果')
    print('='*100)

    print(f"\n{'轮廓':<6} {'面积':<10} {'原点数':<8} {'新点数':<8} {'原曲率':<10} {'新曲率':<10} {'改善率':<10} {'等级':<12}")
    print('-'*100)

    for r in sorted(results, key=lambda x: x['area']):
        level = 'A (严重)' if r['orig_curv_max'] > 20 else 'B (中等)' if r['orig_curv_max'] > 10 else 'C (轻微)'

        print(f"{r['id']:<6} {r['area']:<10.1f} {r['orig_points']:<8} {r['adaptive_points']:<8} "
              f"{r['orig_curv_max']:<10.2f} {r['adapted_curv_max']:<10.2f} "
              f"{r['curv_improvement']:<10.1f}% {level:<12}")

    # 4. 统计分析
    print('\n' + '='*100)
    print('统计分析')
    print('='*100)

    # 按轮廓大小分组
    small_contours = [r for r in results if r['area'] < 2000]
    large_contours = [r for r in results if r['area'] > 3000]

    if small_contours:
        small_avg_improvement = np.mean([r['curv_improvement'] for r in small_contours])
        print(f"\n小轮廓（面积<2000）平均改善率: {small_avg_improvement:.1f}%")
        print(f"  样本数: {len(small_contours)}")
        print(f"  平均原曲率: {np.mean([r['orig_curv_max'] for r in small_contours]):.2f}")
        print(f"  平均新曲率: {np.mean([r['adapted_curv_max'] for r in small_contours]):.2f}")

    if large_contours:
        large_avg_improvement = np.mean([r['curv_improvement'] for r in large_contours])
        print(f"\n大轮廓（面积>3000）平均改善率: {large_avg_improvement:.1f}%")
        print(f"  样本数: {len(large_contours)}")
        print(f"  平均原曲率: {np.mean([r['orig_curv_max'] for r in large_contours]):.2f}")
        print(f"  平均新曲率: {np.mean([r['adapted_curv_max'] for r in large_contours]):.2f}")

    overall_improvement = np.mean([r['curv_improvement'] for r in results])
    print(f"\n整体平均改善率: {overall_improvement:.1f}%")

    # 5. 验证结论
    print('\n' + '='*100)
    print('验证结论')
    print('='*100)

    print("\n假设A（点密度高导致毛刺）的预测：")
    print("  1. 小轮廓改善率应该 > 30%")
    print("  2. 大轮廓改善率应该较小（< 20%）")
    print("  3. 整体改善率应该 > 25%")

    print("\n实际结果：")

    success_count = 0

    if small_contours:
        print(f"  1. 小轮廓改善率: {small_avg_improvement:.1f}%", end='')
        if small_avg_improvement > 30:
            print(" ✓ 支持假设")
            success_count += 1
        elif small_avg_improvement > 20:
            print(" ○ 弱支持假设")
        else:
            print(" ✗ 不支持假设")

    if large_contours:
        print(f"  2. 大轮廓改善率: {large_avg_improvement:.1f}%", end='')
        if large_avg_improvement < 20:
            print(" ✓ 支持假设")
            success_count += 1
        elif large_avg_improvement < 30:
            print(" ○ 弱支持假设")
        else:
            print(" ✗ 不支持假设（大轮廓也改善很多，说明原来的128点对所有轮廓都不合适）")

    print(f"  3. 整体改善率: {overall_improvement:.1f}%", end='')
    if overall_improvement > 25:
        print(" ✓ 支持假设")
        success_count += 1
    elif overall_improvement > 15:
        print(" ○ 弱支持假设")
    else:
        print(" ✗ 不支持假设")

    print(f"\n支持假设的证据数量: {success_count}/3")

    if success_count >= 2:
        print("\n✓✓✓ 强烈建议实施自适应点数方案！")
        print("\n后处理验证已经证明有效，下一步应该：")
        print("  1. 修改训练代码，集成自适应点数")
        print("  2. 在V3.4基础上重新训练")
        print("  3. 预期整体毛刺改善 > 30%")
    elif success_count == 1:
        print("\n○ 自适应点数方案有一定效果，但可能需要调整参数")
        print("\n建议：")
        print("  1. 尝试不同的target_density（1.5, 2.0, 3.0）")
        print("  2. 分析哪些轮廓改善不明显，找出原因")
    else:
        print("\n✗ 自适应点数方案效果不明显")
        print("\n可能的原因：")
        print("  1. 模型本身的预测质量问题")
        print("  2. 特征图分辨率不足")
        print("  3. 需要从训练阶段就使用自适应点数")

    # 6. 可视化
    print('\n[5/5] 生成可视化...')

    # 选择几个代表性轮廓进行可视化
    small_worst = max([r for r in results if r['area'] < 2000], key=lambda x: x['orig_curv_max'])
    large_best = min([r for r in results if r['area'] > 3000], key=lambda x: x['orig_curv_max'])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 改善率 vs 面积
    ax = axes[0, 0]
    areas = [r['area'] for r in results]
    improvements = [r['curv_improvement'] for r in results]
    colors = [r['orig_curv_max'] for r in results]

    scatter = ax.scatter(areas, improvements, s=100, alpha=0.7, c=colors, cmap='hot')
    for r in results:
        ax.annotate(f"C{r['id']}", (r['area'], r['curv_improvement']), fontsize=10)
    ax.axhline(y=30, color='g', linestyle='--', label='Target 30%')
    ax.set_xlabel('Area (pixels²)')
    ax.set_ylabel('Curvature Improvement (%)')
    ax.set_title('Area vs Improvement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Original Curvature')

    # 2. 点数变化
    ax = axes[0, 1]
    x = range(len(results))
    sorted_results = sorted(results, key=lambda x: x['area'])
    orig_points = [r['orig_points'] for r in sorted_results]
    adaptive_points = [r['adaptive_points'] for r in sorted_results]
    labels = [f"C{r['id']}" for r in sorted_results]

    ax.plot(x, orig_points, 'ro-', label='Original (128)', linewidth=2, markersize=8)
    ax.plot(x, adaptive_points, 'bs-', label='Adaptive', linewidth=2, markersize=8)
    ax.set_xlabel('Contour (sorted by area)')
    ax.set_ylabel('Number of Points')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('Point Number Adjustment')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 曲率对比
    ax = axes[0, 2]
    orig_curvs = [r['orig_curv_max'] for r in sorted_results]
    adapted_curvs = [r['adapted_curv_max'] for r in sorted_results]

    ax.plot(x, orig_curvs, 'ro-', label='Original', linewidth=2, markersize=8)
    ax.plot(x, adapted_curvs, 'bs-', label='Adaptive', linewidth=2, markersize=8)
    ax.set_xlabel('Contour (sorted by area)')
    ax.set_ylabel('Max Curvature')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('Curvature Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 小轮廓示例（最差的）
    ax = axes[1, 0]
    orig_contour = small_worst['orig_contour']
    adapted_contour = small_worst['adapted_contour']
    # 上采样回128点用于可视化
    adapted_upsampled = adaptive_upsample(adapted_contour, 128)

    ax.plot(orig_contour[:, 0], orig_contour[:, 1], 'r-', linewidth=2, label=f'Original (curv={small_worst["orig_curv_max"]:.1f})')
    ax.plot(adapted_upsampled[:, 0], adapted_upsampled[:, 1], 'b-', linewidth=2, label=f'Adaptive (curv={small_worst["adapted_curv_max"]:.1f})')
    ax.scatter(adapted_contour[:, 0], adapted_contour[:, 1], c='blue', s=20, alpha=0.5, label=f'Adaptive points ({len(adapted_contour)})')
    ax.set_title(f'Small Contour C{small_worst["id"]} (Improvement: {small_worst["curv_improvement"]:.1f}%)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # 5. 大轮廓示例（最好的）
    ax = axes[1, 1]
    orig_contour = large_best['orig_contour']
    adapted_contour = large_best['adapted_contour']
    adapted_upsampled = adaptive_upsample(adapted_contour, 128)

    ax.plot(orig_contour[:, 0], orig_contour[:, 1], 'r-', linewidth=2, label=f'Original (curv={large_best["orig_curv_max"]:.1f})')
    ax.plot(adapted_upsampled[:, 0], adapted_upsampled[:, 1], 'b-', linewidth=2, label=f'Adaptive (curv={large_best["adapted_curv_max"]:.1f})')
    ax.scatter(adapted_contour[:, 0], adapted_contour[:, 1], c='blue', s=20, alpha=0.5, label=f'Adaptive points ({len(adapted_contour)})')
    ax.set_title(f'Large Contour C{large_best["id"]} (Improvement: {large_best["curv_improvement"]:.1f}%)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # 6. 改善率分布
    ax = axes[1, 2]
    ax.hist(improvements, bins=10, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=30, color='g', linestyle='--', linewidth=2, label='Target 30%')
    ax.axvline(x=np.mean(improvements), color='r', linestyle='-', linewidth=2, label=f'Mean {np.mean(improvements):.1f}%')
    ax.set_xlabel('Curvature Improvement (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Improvement Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = os.path.join(_PROJ_DIR, 'visual/adaptive_points_verification')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'v3_4_adaptive_points_verification.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    print(f'\n可视化已保存: {output_path}')

    # 保存结果
    output_json = os.path.join(output_dir, 'v3_4_adaptive_results.json')
    with open(output_json, 'w') as f:
        results_json = []
        for r in results:
            r_copy = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
            results_json.append(r_copy)
        json.dump(results_json, f, indent=2)

    print(f'结果已保存: {output_json}')

    print('\n' + '='*100)
    print('验证完成！')
    print('='*100)


if __name__ == '__main__':
    main()
