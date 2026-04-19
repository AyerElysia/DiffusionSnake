"""
验证方案：对小轮廓隔点采样

实验设计：
---------
1. 使用V3.4模型正常推理得到128点预测
2. 对小轮廓（面积<2000）进行隔点采样（每隔1个点取1个，变成64点）
3. 对比采样前后的曲率变化

验证逻辑：
---------
如果点密度是主要原因：
  → 隔点采样后，小轮廓的曲率应该明显降低
  → 大轮廓保持128点，曲率变化不大
"""

import sys, os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from scipy.interpolate import interp1d

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJ_DIR)

os.environ['CFG_FILE'] = os.path.join(_PROJ_DIR, 'configs/btcv_diffusion_dit_v3_4_single_overfit.yaml')

from lib.config import cfg
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.datasets.make_dataset import make_dataset
from lib.datasets.collate_batch import make_collator
from lib.datasets.transforms import make_transforms
from lib.utils.snake import snake_config, snake_gcn_utils


def compute_curvature(contour):
    """计算轮廓的曲率"""
    prev = np.roll(contour, 1, axis=0)
    next = np.roll(contour, -1, axis=0)

    v1 = prev - contour
    v2 = next - contour

    v1_norm = np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8
    v2_norm = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8

    v1 = v1 / v1_norm
    v2 = v2 / v2_norm

    cos_angle = np.sum(v1 * v2, axis=1)
    cos_angle = np.clip(cos_angle, -1, 1)

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
        'curv_std': float(np.std(curvature)),
        'sharp_angles': sharp_angles,
        'perimeter': float(perimeter),
        'area': float(area),
        'num_points': len(contour),
    }

    return metrics


def downsample_contour(contour, step=2):
    """
    隔点采样

    Args:
        contour: [N, 2] 原始轮廓
        step: 采样步长，2表示隔1个点取1个

    Returns:
        [N//step, 2] 下采样后的轮廓
    """
    return contour[::step]


def upsample_contour(contour, target_num_points):
    """
    上采样轮廓（用于可视化）

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


def load_v3_4_model():
    """加载V3.4模型"""
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 查找checkpoint
    possible_paths = [
        os.path.join(_PROJ_DIR, 'data/outputs/btcv_diffusion_dit_v3_4_single_overfit/checkpoints/epoch_10000.pt'),
        os.path.join(_PROJ_DIR, 'data/outputs/btcv_diffusion_dit_v3_4_single_overfit/checkpoints/latest.pt'),
    ]

    ckpt_path = None
    for path in possible_paths:
        if os.path.exists(path):
            ckpt_path = path
            break

    if ckpt_path is None:
        print(f"[!] 错误: 找不到V3.4 checkpoint!")
        sys.exit(1)

    print(f"[*] 加载V3.4权重: {ckpt_path}")

    ckpt_obj = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt_obj.get('state_dict') or ckpt_obj.get('model') or ckpt_obj.get('net') or ckpt_obj

    from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
    sd = remap_legacy_state_dict(sd)
    wrapper = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network

    info = wrapper.load_state_dict(sd, strict=False)
    print(f"[✔] 加载成功: {len(sd) - len(info.missing_keys)} 层匹配")

    return trainer.network.to(device).eval(), device


def run_inference_with_downsampling(model, device, batch, area_threshold=2000, downsample_step=2):
    """
    运行推理并对小轮廓进行隔点采样

    Args:
        model: 模型
        device: 设备
        batch: 数据batch
        area_threshold: 面积阈值，小于此值的轮廓进行下采样
        downsample_step: 下采样步长，2表示隔1个点取1个
    """
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    dr = float(snake_config.down_ratio)
    core = model.net if hasattr(model, 'net') else model

    with torch.no_grad():
        # 网络前向
        yolo_out = core.yolo(batch['inp'])
        if isinstance(yolo_out, (list, tuple)):
            feat_p2 = yolo_out[1][0] if len(yolo_out) > 1 else yolo_out[0]
        else:
            feat_p2 = yolo_out
        cnn_feature = core.cnn_proj(feat_p2)

        # 获取所有轮廓
        gt_all = batch['i_gt_py']
        if gt_all.numel() == 0:
            print("  [!] 没有GT轮廓")
            return None

        B, M, P, _ = gt_all.shape
        print(f"  [*] 检测到 {M} 个轮廓")

        if 'i_it_py' not in batch or batch['i_it_py'].numel() == 0:
            print("  [!] 没有初始化轮廓")
            return None

        i_it_py = batch['i_it_py'].view(-1, P, 2)
        c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)

        if B == 1:
            py_ind = torch.zeros(i_it_py.size(0), dtype=torch.long, device=device)
        else:
            py_ind = torch.cat([torch.full((M,), i, dtype=torch.long, device=device) for i in range(B)])

        # 原始预测（128点）- 使用单步预测，不使用多步平滑
        # 注意：sample_disp会进行多步采样平滑，我们需要原始的单步预测
        # 直接调用gcn的forward获取原始位移
        from lib.utils.snake import snake_gcn_utils

        # 方法1：尝试获取原始预测（不经过多步采样）
        try:
            # 直接forward一次
            with torch.no_grad():
                disp_raw = core.gcn(cnn_feature, i_it_py, c_it_py, py_ind)
            pred_polys_128 = (i_it_py + disp_raw).cpu().numpy() * dr
            print("  [*] 使用原始单步预测（未平滑）")
        except:
            # 如果失败，使用steps=1的采样
            disp = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind, steps=1)
            pred_polys_128 = (i_it_py + disp).cpu().numpy() * dr
            print("  [*] 使用steps=1的预测")

        gt_np = gt_all.cpu().numpy() * dr

    # 获取图像
    if 'orig_img' in batch:
        img_raw = batch['orig_img'][0]
        img = img_raw.detach().cpu().numpy() if torch.is_tensor(img_raw) else img_raw
        img = img.astype(np.uint8)
    else:
        img = np.zeros((512, 512, 3), dtype=np.uint8)

    # 对每个轮廓进行处理
    results = []

    for i in range(len(pred_polys_128)):
        pred_128 = pred_polys_128[i]
        gt = gt_np[0][i]

        # 计算原始指标
        metrics_128 = compute_metrics(pred_128)

        # 判断是否为小轮廓
        is_small = metrics_128['area'] < area_threshold

        if is_small:
            # 对小轮廓进行隔点采样
            pred_downsampled = downsample_contour(pred_128, step=downsample_step)
            metrics_downsampled = compute_metrics(pred_downsampled)

            # 计算改善率
            curv_improvement = (metrics_128['curv_max'] - metrics_downsampled['curv_max']) / metrics_128['curv_max'] * 100
            sharp_improvement = (metrics_128['sharp_angles'] - metrics_downsampled['sharp_angles']) / (metrics_128['sharp_angles'] + 1) * 100
        else:
            # 大轮廓保持128点
            pred_downsampled = pred_128
            metrics_downsampled = metrics_128
            curv_improvement = 0.0
            sharp_improvement = 0.0

        results.append({
            'id': i,
            'is_small': is_small,
            'area': metrics_128['area'],
            'perimeter': metrics_128['perimeter'],
            'orig_points': 128,
            'new_points': len(pred_downsampled),
            'orig_density': metrics_128['perimeter'] / 128,
            'new_density': metrics_downsampled['perimeter'] / len(pred_downsampled),
            'orig_curv_max': metrics_128['curv_max'],
            'new_curv_max': metrics_downsampled['curv_max'],
            'curv_improvement': curv_improvement,
            'orig_sharp_angles': metrics_128['sharp_angles'],
            'new_sharp_angles': metrics_downsampled['sharp_angles'],
            'sharp_improvement': sharp_improvement,
            'pred_128': pred_128,
            'pred_downsampled': pred_downsampled,
            'gt': gt,
        })

    return results, img


def main():
    print('='*100)
    print('验证方案：对小轮廓隔点采样（使用V3.4模型）')
    print('='*100)

    # 1. 加载模型和数据
    print('\n[1/4] 加载V3.4模型和数据...')
    model, device = load_v3_4_model()
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)
    batch = collator([dataset[0]])

    # 2. 运行推理
    print('\n[2/4] 运行推理并对小轮廓隔点采样...')
    print('  策略：面积<2000的轮廓，隔1个点取1个（128点→64点）')

    results, img = run_inference_with_downsampling(
        model, device, batch,
        area_threshold=2000,
        downsample_step=2
    )

    if results is None:
        print('推理失败')
        return

    # 3. 打印结果
    print('\n' + '='*100)
    print('隔点采样结果')
    print('='*100)

    print(f"\n{'轮廓':<6} {'类型':<8} {'面积':<10} {'原点数':<8} {'新点数':<8} "
          f"{'原曲率':<10} {'新曲率':<10} {'改善率':<10}")
    print('-'*100)

    for r in sorted(results, key=lambda x: x['area']):
        contour_type = '小轮廓' if r['is_small'] else '大轮廓'
        print(f"{r['id']:<6} {contour_type:<8} {r['area']:<10.1f} {r['orig_points']:<8} {r['new_points']:<8} "
              f"{r['orig_curv_max']:<10.2f} {r['new_curv_max']:<10.2f} {r['curv_improvement']:<10.1f}%")

    # 4. 统计分析
    print('\n' + '='*100)
    print('统计分析')
    print('='*100)

    small_contours = [r for r in results if r['is_small']]
    large_contours = [r for r in results if not r['is_small']]

    if small_contours:
        small_avg_improvement = np.mean([r['curv_improvement'] for r in small_contours])
        print(f"\n小轮廓（面积<2000）:")
        print(f"  样本数: {len(small_contours)}")
        print(f"  平均原曲率: {np.mean([r['orig_curv_max'] for r in small_contours]):.2f}")
        print(f"  平均新曲率: {np.mean([r['new_curv_max'] for r in small_contours]):.2f}")
        print(f"  平均改善率: {small_avg_improvement:.1f}%")

    if large_contours:
        print(f"\n大轮廓（面积≥2000）:")
        print(f"  样本数: {len(large_contours)}")
        print(f"  平均曲率: {np.mean([r['orig_curv_max'] for r in large_contours]):.2f}")
        print(f"  （保持128点，无变化）")

    overall_improvement = np.mean([r['curv_improvement'] for r in results])
    print(f"\n整体平均改善率: {overall_improvement:.1f}%")

    # 5. 验证结论
    print('\n' + '='*100)
    print('验证结论')
    print('='*100)

    print("\n假设A（点密度高导致毛刺）的预测：")
    print("  1. 小轮廓隔点采样后，改善率应该 > 30%")
    print("  2. 整体改善率应该 > 20%")

    print("\n实际结果：")

    success_count = 0

    if small_contours:
        print(f"  1. 小轮廓平均改善率: {small_avg_improvement:.1f}%", end='')
        if small_avg_improvement > 30:
            print(" ✓✓✓ 强烈支持假设！")
            success_count += 2
        elif small_avg_improvement > 20:
            print(" ✓ 支持假设")
            success_count += 1
        elif small_avg_improvement > 10:
            print(" ○ 弱支持假设")
        else:
            print(" ✗ 不支持假设")

    print(f"  2. 整体平均改善率: {overall_improvement:.1f}%", end='')
    if overall_improvement > 20:
        print(" ✓ 支持假设")
        success_count += 1
    elif overall_improvement > 10:
        print(" ○ 弱支持假设")
    else:
        print(" ✗ 不支持假设")

    print(f"\n支持假设的证据强度: {success_count}/3")

    if success_count >= 2:
        print("\n✓✓✓ 隔点采样验证成功！点密度确实是主要原因")
        print("\n下一步建议：")
        print("  1. 实施自适应点数方案（根据轮廓周长动态调整点数）")
        print("  2. 修改训练代码，从数据准备阶段就使用自适应点数")
        print("  3. 预期整体改善 > 30%")
    elif success_count == 1:
        print("\n○ 隔点采样有一定效果")
        print("\n建议：")
        print("  1. 尝试不同的采样步长（step=3, 4）")
        print("  2. 考虑更精细的自适应策略")
    else:
        print("\n✗ 隔点采样效果不明显")
        print("\n可能的原因：")
        print("  1. 模型预测质量本身有问题")
        print("  2. 需要从训练阶段就改变点数")

    # 6. 可视化
    print('\n[4/4] 生成可视化...')

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 改善率 vs 面积
    ax = axes[0, 0]
    areas = [r['area'] for r in results]
    improvements = [r['curv_improvement'] for r in results]
    colors = ['red' if r['is_small'] else 'blue' for r in results]

    ax.scatter(areas, improvements, s=100, alpha=0.7, c=colors)
    for r in results:
        ax.annotate(f"C{r['id']}", (r['area'], r['curv_improvement']), fontsize=10)
    ax.axhline(y=30, color='g', linestyle='--', label='Target 30%')
    ax.axvline(x=2000, color='orange', linestyle='--', label='Threshold 2000')
    ax.set_xlabel('Area (pixels²)')
    ax.set_ylabel('Curvature Improvement (%)')
    ax.set_title('Area vs Improvement (Red=Small, Blue=Large)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 点数变化
    ax = axes[0, 1]
    x = range(len(results))
    sorted_results = sorted(results, key=lambda x: x['area'])
    orig_points = [r['orig_points'] for r in sorted_results]
    new_points = [r['new_points'] for r in sorted_results]
    labels = [f"C{r['id']}" for r in sorted_results]

    ax.plot(x, orig_points, 'ro-', label='Original (128)', linewidth=2, markersize=8)
    ax.plot(x, new_points, 'bs-', label='After Downsampling', linewidth=2, markersize=8)
    ax.set_xlabel('Contour (sorted by area)')
    ax.set_ylabel('Number of Points')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('Point Number Change')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 曲率对比
    ax = axes[0, 2]
    orig_curvs = [r['orig_curv_max'] for r in sorted_results]
    new_curvs = [r['new_curv_max'] for r in sorted_results]

    ax.plot(x, orig_curvs, 'ro-', label='Original (128 pts)', linewidth=2, markersize=8)
    ax.plot(x, new_curvs, 'bs-', label='After Downsampling', linewidth=2, markersize=8)
    ax.set_xlabel('Contour (sorted by area)')
    ax.set_ylabel('Max Curvature')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('Curvature Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4-6. 选择3个代表性轮廓可视化
    small_contours_sorted = sorted(small_contours, key=lambda x: x['orig_curv_max'], reverse=True)

    for idx, r in enumerate(small_contours_sorted[:3]):
        ax = axes[1, idx]

        pred_128 = r['pred_128']
        pred_down = r['pred_downsampled']
        gt = r['gt']

        # 上采样回128点用于可视化
        pred_down_upsampled = upsample_contour(pred_down, 128)

        ax.plot(gt[:, 0], gt[:, 1], 'g-', linewidth=1, alpha=0.5, label='GT')
        ax.plot(pred_128[:, 0], pred_128[:, 1], 'r-', linewidth=2, alpha=0.7, label=f'128pts (curv={r["orig_curv_max"]:.1f})')
        ax.plot(pred_down_upsampled[:, 0], pred_down_upsampled[:, 1], 'b--', linewidth=2, alpha=0.7, label=f'{r["new_points"]}pts (curv={r["new_curv_max"]:.1f})')
        ax.scatter(pred_down[:, 0], pred_down[:, 1], c='blue', s=30, alpha=0.5, zorder=10)

        ax.set_title(f'Contour {r["id"]} (Area={r["area"]:.0f}, Improvement={r["curv_improvement"]:.1f}%)')
        ax.legend(fontsize=8)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = os.path.join(_PROJ_DIR, 'visual/downsample_verification')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'v3_4_downsample_verification.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    print(f'\n可视化已保存: {output_path}')

    # 保存结果
    output_json = os.path.join(output_dir, 'v3_4_downsample_results.json')
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
