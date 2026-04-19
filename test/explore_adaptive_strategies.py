"""
自适应点数方案探索框架

目标：找到最优的自适应点数策略，使IoU达到97%

策略探索：
1. 基于周长的线性策略（不同target_density）
2. 基于面积的策略
3. 基于周长+面积的混合策略
4. 分段策略（小/中/大轮廓不同密度）
5. 基于曲率复杂度的自适应策略
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.interpolate import interp1d
import cv2

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJ_DIR)

os.environ['CFG_FILE'] = os.path.join(_PROJ_DIR, 'configs/btcv_diffusion_dit_v3_4_single_overfit.yaml')

from lib.utils.snake import snake_config


def compute_curvature(contour):
    """计算轮廓的曲率（二阶差分）"""
    prev = np.roll(contour, 1, axis=0)
    next = np.roll(contour, -1, axis=0)
    d2 = (next - contour) - (contour - prev)
    curvature = np.linalg.norm(d2, axis=1)
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

    # 尖锐角
    prev = np.roll(contour, 1, axis=0)
    next = np.roll(contour, -1, axis=0)
    v1 = contour - prev
    v2 = next - contour
    v1_norm = np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8
    v2_norm = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8
    cos_angles = np.sum(v1 * v2, axis=1) / (v1_norm.squeeze() * v2_norm.squeeze())
    cos_angles = np.clip(cos_angles, -1, 1)
    angles = np.arccos(cos_angles) * 180 / np.pi
    sharp_angles = int(np.sum(angles > 120))

    metrics = {
        'curv_max': float(np.max(curvature)),
        'curv_mean': float(np.mean(curvature)),
        'sharp_angles': sharp_angles,
        'perimeter': float(perimeter),
        'area': float(area),
        'num_points': len(contour),
    }

    return metrics


def compute_iou(contour1, contour2, img_size=(512, 512)):
    """计算两个轮廓的IoU"""
    mask1 = np.zeros(img_size, dtype=np.uint8)
    mask2 = np.zeros(img_size, dtype=np.uint8)

    cv2.fillPoly(mask1, [contour1.astype(np.int32)], 1)
    cv2.fillPoly(mask2, [contour2.astype(np.int32)], 1)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection) / float(union)


def resample_contour(contour, num_points):
    """重采样轮廓到指定点数"""
    if num_points == len(contour):
        return contour

    # 计算累积弧长
    dists = np.linalg.norm(np.diff(contour, axis=0, append=contour[:1]), axis=1)
    cumsum = np.concatenate([[0], np.cumsum(dists)])
    total_length = cumsum[-1]

    # 均匀采样
    target_lengths = np.linspace(0, total_length, num_points, endpoint=False)

    # 插值
    interp_x = interp1d(cumsum, np.concatenate([contour[:, 0], [contour[0, 0]]]), kind='linear')
    interp_y = interp1d(cumsum, np.concatenate([contour[:, 1], [contour[0, 1]]]), kind='linear')

    new_x = interp_x(target_lengths)
    new_y = interp_y(target_lengths)

    new_contour = np.stack([new_x, new_y], axis=1)

    return new_contour


# ============================================================================
# 策略定义
# ============================================================================

class AdaptivePointsStrategy:
    """自适应点数策略基类"""
    def __init__(self, name):
        self.name = name

    def compute_num_points(self, contour, metrics):
        """计算目标点数"""
        raise NotImplementedError


class LinearPerimeterStrategy(AdaptivePointsStrategy):
    """策略1：基于周长的线性策略"""
    def __init__(self, target_density, min_points=32, max_points=256):
        super().__init__(f"Linear_Perimeter_d{target_density}")
        self.target_density = target_density
        self.min_points = min_points
        self.max_points = max_points

    def compute_num_points(self, contour, metrics):
        perimeter = metrics['perimeter']
        num_points = int(perimeter / self.target_density)
        return max(self.min_points, min(num_points, self.max_points))


class LinearAreaStrategy(AdaptivePointsStrategy):
    """策略2：基于面积的策略"""
    def __init__(self, points_per_area, min_points=32, max_points=256):
        super().__init__(f"Linear_Area_ppa{points_per_area}")
        self.points_per_area = points_per_area
        self.min_points = min_points
        self.max_points = max_points

    def compute_num_points(self, contour, metrics):
        area = metrics['area']
        num_points = int(np.sqrt(area) * self.points_per_area)
        return max(self.min_points, min(num_points, self.max_points))


class HybridStrategy(AdaptivePointsStrategy):
    """策略3：周长+面积混合策略"""
    def __init__(self, perimeter_weight=0.7, area_weight=0.3,
                 target_density=2.5, points_per_area=2.0,
                 min_points=32, max_points=256):
        super().__init__(f"Hybrid_pw{perimeter_weight}_aw{area_weight}")
        self.perimeter_weight = perimeter_weight
        self.area_weight = area_weight
        self.target_density = target_density
        self.points_per_area = points_per_area
        self.min_points = min_points
        self.max_points = max_points

    def compute_num_points(self, contour, metrics):
        perimeter = metrics['perimeter']
        area = metrics['area']

        points_from_perimeter = perimeter / self.target_density
        points_from_area = np.sqrt(area) * self.points_per_area

        num_points = int(
            self.perimeter_weight * points_from_perimeter +
            self.area_weight * points_from_area
        )
        return max(self.min_points, min(num_points, self.max_points))


class TieredStrategy(AdaptivePointsStrategy):
    """策略4：分段策略（小/中/大轮廓不同密度）"""
    def __init__(self, small_density=1.5, medium_density=2.5, large_density=3.5,
                 small_threshold=1000, large_threshold=3000,
                 min_points=32, max_points=256):
        super().__init__(f"Tiered_s{small_density}_m{medium_density}_l{large_density}")
        self.small_density = small_density
        self.medium_density = medium_density
        self.large_density = large_density
        self.small_threshold = small_threshold
        self.large_threshold = large_threshold
        self.min_points = min_points
        self.max_points = max_points

    def compute_num_points(self, contour, metrics):
        area = metrics['area']
        perimeter = metrics['perimeter']

        if area < self.small_threshold:
            density = self.small_density
        elif area < self.large_threshold:
            density = self.medium_density
        else:
            density = self.large_density

        num_points = int(perimeter / density)
        return max(self.min_points, min(num_points, self.max_points))


class CurvatureAdaptiveStrategy(AdaptivePointsStrategy):
    """策略5：基于曲率复杂度的自适应策略"""
    def __init__(self, base_density=2.5, complexity_factor=0.5,
                 min_points=32, max_points=256):
        super().__init__(f"Curvature_Adaptive_bd{base_density}_cf{complexity_factor}")
        self.base_density = base_density
        self.complexity_factor = complexity_factor
        self.min_points = min_points
        self.max_points = max_points

    def compute_num_points(self, contour, metrics):
        perimeter = metrics['perimeter']
        curv_mean = metrics['curv_mean']

        # 曲率越大，需要更多点来表示
        complexity = 1.0 + self.complexity_factor * (curv_mean / 10.0)
        adjusted_density = self.base_density / complexity

        num_points = int(perimeter / adjusted_density)
        return max(self.min_points, min(num_points, self.max_points))


# ============================================================================
# 评估函数
# ============================================================================

def evaluate_strategy(strategy, pred_contours, gt_contours):
    """评估一个策略"""
    results = []

    for i, (pred, gt) in enumerate(zip(pred_contours, gt_contours)):
        # 原始指标
        metrics_orig = compute_metrics(pred)
        iou_orig = compute_iou(pred, gt)

        # 应用策略
        target_points = strategy.compute_num_points(pred, metrics_orig)
        pred_resampled = resample_contour(pred, target_points)

        # 重采样后指标
        metrics_new = compute_metrics(pred_resampled)
        iou_new = compute_iou(pred_resampled, gt)

        # 计算改善
        curv_improvement = (metrics_orig['curv_max'] - metrics_new['curv_max']) / metrics_orig['curv_max'] * 100
        iou_improvement = (iou_new - iou_orig) / iou_orig * 100

        results.append({
            'id': i,
            'area': metrics_orig['area'],
            'perimeter': metrics_orig['perimeter'],
            'orig_points': 128,
            'new_points': target_points,
            'orig_density': metrics_orig['perimeter'] / 128,
            'new_density': metrics_new['perimeter'] / target_points,
            'orig_curv': metrics_orig['curv_max'],
            'new_curv': metrics_new['curv_max'],
            'curv_improvement': curv_improvement,
            'orig_iou': iou_orig,
            'new_iou': iou_new,
            'iou_improvement': iou_improvement,
        })

    # 统计
    avg_iou_orig = np.mean([r['orig_iou'] for r in results])
    avg_iou_new = np.mean([r['new_iou'] for r in results])
    avg_curv_improvement = np.mean([r['curv_improvement'] for r in results])
    avg_iou_improvement = np.mean([r['iou_improvement'] for r in results])

    summary = {
        'strategy_name': strategy.name,
        'avg_iou_orig': avg_iou_orig,
        'avg_iou_new': avg_iou_new,
        'avg_iou_improvement': avg_iou_improvement,
        'avg_curv_improvement': avg_curv_improvement,
        'results': results,
    }

    return summary


def main():
    print('='*100)
    print('自适应点数策略探索')
    print('目标：IoU达到97%')
    print('='*100)

    # 1. 加载数据
    print('\n[1/4] 加载数据...')
    pred_path = os.path.join(_PROJ_DIR, 'visual/burr_v3_4_full/pred_contours_raw.npy')
    pred_contours = np.load(pred_path)

    # 加载GT
    from lib.datasets.make_dataset import make_dataset
    from lib.datasets.collate_batch import make_collator
    from lib.datasets.transforms import make_transforms
    from lib.config import cfg

    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)
    batch = collator([dataset[0]])

    dr = float(snake_config.down_ratio)
    gt_all = batch['i_gt_py']
    gt_contours = gt_all.cpu().numpy()[0] * dr

    print(f'  加载了 {len(pred_contours)} 个预测轮廓')
    print(f'  加载了 {len(gt_contours)} 个GT轮廓')

    # 2. 定义策略
    print('\n[2/4] 定义策略...')
    strategies = [
        # 基线：保持128点
        LinearPerimeterStrategy(target_density=999, min_points=128, max_points=128),

        # 策略1：不同的target_density
        LinearPerimeterStrategy(target_density=1.5),
        LinearPerimeterStrategy(target_density=2.0),
        LinearPerimeterStrategy(target_density=2.5),
        LinearPerimeterStrategy(target_density=3.0),
        LinearPerimeterStrategy(target_density=3.5),

        # 策略2：基于面积
        LinearAreaStrategy(points_per_area=1.5),
        LinearAreaStrategy(points_per_area=2.0),
        LinearAreaStrategy(points_per_area=2.5),

        # 策略3：混合策略
        HybridStrategy(perimeter_weight=0.7, area_weight=0.3, target_density=2.5),
        HybridStrategy(perimeter_weight=0.8, area_weight=0.2, target_density=2.5),
        HybridStrategy(perimeter_weight=0.6, area_weight=0.4, target_density=2.5),

        # 策略4：分段策略
        TieredStrategy(small_density=1.5, medium_density=2.5, large_density=3.5),
        TieredStrategy(small_density=2.0, medium_density=2.5, large_density=3.0),
        TieredStrategy(small_density=1.0, medium_density=2.0, large_density=3.0),

        # 策略5：曲率自适应
        CurvatureAdaptiveStrategy(base_density=2.5, complexity_factor=0.3),
        CurvatureAdaptiveStrategy(base_density=2.5, complexity_factor=0.5),
        CurvatureAdaptiveStrategy(base_density=2.5, complexity_factor=0.7),
    ]

    print(f'  定义了 {len(strategies)} 个策略')

    # 3. 评估所有策略
    print('\n[3/4] 评估策略...')
    all_summaries = []

    for i, strategy in enumerate(strategies):
        print(f'  [{i+1}/{len(strategies)}] 评估策略: {strategy.name}')
        summary = evaluate_strategy(strategy, pred_contours, gt_contours)
        all_summaries.append(summary)

    # 4. 排序并展示结果
    print('\n[4/4] 结果分析...')

    # 按IoU排序
    all_summaries.sort(key=lambda x: x['avg_iou_new'], reverse=True)

    print('\n' + '='*100)
    print('策略排名（按平均IoU）')
    print('='*100)

    print(f"\n{'排名':<6} {'策略名称':<50} {'原始IoU':<12} {'新IoU':<12} {'IoU提升':<12} {'曲率改善':<12}")
    print('-'*100)

    for rank, summary in enumerate(all_summaries, 1):
        print(f"{rank:<6} {summary['strategy_name']:<50} "
              f"{summary['avg_iou_orig']:<12.4f} {summary['avg_iou_new']:<12.4f} "
              f"{summary['avg_iou_improvement']:<12.2f}% {summary['avg_curv_improvement']:<12.2f}%")

    # 找到最佳策略
    best_summary = all_summaries[0]

    print('\n' + '='*100)
    print('最佳策略详细分析')
    print('='*100)

    print(f"\n策略名称: {best_summary['strategy_name']}")
    print(f"平均IoU: {best_summary['avg_iou_orig']:.4f} → {best_summary['avg_iou_new']:.4f}")
    print(f"IoU提升: {best_summary['avg_iou_improvement']:.2f}%")
    print(f"曲率改善: {best_summary['avg_curv_improvement']:.2f}%")

    print(f"\n{'轮廓':<8} {'面积':<10} {'原点数':<10} {'新点数':<10} {'原IoU':<10} {'新IoU':<10} {'IoU提升':<10}")
    print('-'*100)

    for r in best_summary['results']:
        print(f"{r['id']:<8} {r['area']:<10.1f} {r['orig_points']:<10} {r['new_points']:<10} "
              f"{r['orig_iou']:<10.4f} {r['new_iou']:<10.4f} {r['iou_improvement']:<10.2f}%")

    # 检查是否达到目标
    print('\n' + '='*100)
    print('目标达成情况')
    print('='*100)

    target_iou = 0.97
    if best_summary['avg_iou_new'] >= target_iou:
        print(f"\n✓✓✓ 目标达成！平均IoU = {best_summary['avg_iou_new']:.4f} >= {target_iou}")
    else:
        gap = target_iou - best_summary['avg_iou_new']
        print(f"\n○ 接近目标，还差 {gap:.4f} ({gap*100:.2f}%)")
        print(f"  当前最佳IoU: {best_summary['avg_iou_new']:.4f}")
        print(f"  目标IoU: {target_iou}")

        # 分析哪些轮廓拖后腿
        print("\n拖后腿的轮廓:")
        for r in sorted(best_summary['results'], key=lambda x: x['new_iou']):
            if r['new_iou'] < target_iou:
                print(f"  轮廓{r['id']}: IoU={r['new_iou']:.4f}, 面积={r['area']:.0f}, 点数={r['new_points']}")

    # 保存结果
    output_dir = os.path.join(_PROJ_DIR, 'visual/strategy_exploration')
    os.makedirs(output_dir, exist_ok=True)

    output_json = os.path.join(output_dir, 'strategy_comparison.json')
    with open(output_json, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print(f'\n结果已保存: {output_json}')

    # 可视化
    print('\n生成可视化...')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. IoU对比
    ax = axes[0, 0]
    strategy_names = [s['strategy_name'] for s in all_summaries[:10]]
    ious = [s['avg_iou_new'] for s in all_summaries[:10]]

    bars = ax.barh(range(len(strategy_names)), ious, color='skyblue')
    ax.axvline(x=target_iou, color='red', linestyle='--', linewidth=2, label=f'Target {target_iou}')
    ax.set_yticks(range(len(strategy_names)))
    ax.set_yticklabels(strategy_names, fontsize=8)
    ax.set_xlabel('Average IoU')
    ax.set_title('Top 10 Strategies by IoU')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # 2. IoU vs 曲率改善
    ax = axes[0, 1]
    ious = [s['avg_iou_new'] for s in all_summaries]
    curvs = [s['avg_curv_improvement'] for s in all_summaries]

    ax.scatter(curvs, ious, s=100, alpha=0.6)
    for i, s in enumerate(all_summaries[:5]):
        ax.annotate(s['strategy_name'][:20], (curvs[i], ious[i]), fontsize=7)
    ax.axhline(y=target_iou, color='red', linestyle='--', linewidth=2, label=f'Target {target_iou}')
    ax.set_xlabel('Curvature Improvement (%)')
    ax.set_ylabel('Average IoU')
    ax.set_title('IoU vs Curvature Improvement')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 最佳策略的点数分配
    ax = axes[1, 0]
    best_results = best_summary['results']
    x = range(len(best_results))
    orig_points = [r['orig_points'] for r in best_results]
    new_points = [r['new_points'] for r in best_results]
    labels = [f"C{r['id']}" for r in best_results]

    ax.plot(x, orig_points, 'ro-', label='Original (128)', linewidth=2, markersize=8)
    ax.plot(x, new_points, 'bs-', label='Adaptive', linewidth=2, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Contour')
    ax.set_ylabel('Number of Points')
    ax.set_title(f'Best Strategy: {best_summary["strategy_name"]}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 最佳策略的IoU分布
    ax = axes[1, 1]
    ious_orig = [r['orig_iou'] for r in best_results]
    ious_new = [r['new_iou'] for r in best_results]

    ax.plot(x, ious_orig, 'ro-', label='Original', linewidth=2, markersize=8)
    ax.plot(x, ious_new, 'bs-', label='Adaptive', linewidth=2, markersize=8)
    ax.axhline(y=target_iou, color='green', linestyle='--', linewidth=2, label=f'Target {target_iou}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Contour')
    ax.set_ylabel('IoU')
    ax.set_title('IoU Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'strategy_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    print(f'可视化已保存: {output_path}')

    print('\n' + '='*100)
    print('探索完成！')
    print('='*100)


if __name__ == '__main__':
    main()
