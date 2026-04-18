"""
毛刺问题分析脚本 - 散点图可视化 + 统计分析

目标：
1. 可视化原始预测的散点图（不按点序连线）
2. 可视化后处理后的散点图
3. 设计统计指标量化毛刺现象

使用V3.4模型，不加后处理先看原始效果
"""

import sys, os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json

# 环境与配置初始化
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CFG = os.path.join(_THIS_DIR, 'configs', 'btcv_diffusion_dit_v3_4.yaml')
if not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = _DEFAULT_CFG
sys.path.insert(0, _THIS_DIR)

from lib.config import cfg, args
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.datasets.make_dataset import make_dataset
from lib.datasets.collate_batch import make_collator
from lib.datasets.transforms import make_transforms
from lib.utils.snake import snake_config, snake_decode, snake_gcn_utils
from analyze_point_order import analyze_point_order_issue


def compute_burr_metrics(contour):
    """
    计算毛刺相关的统计指标

    Args:
        contour: (P, 2) numpy array

    Returns:
        dict: 包含各种统计指标
    """
    P = len(contour)

    # 1. 点间距离统计
    dists = np.linalg.norm(np.diff(contour, axis=0, append=contour[:1]), axis=1)

    # 2. 曲率统计（二阶差分）
    prev_pts = np.roll(contour, 1, axis=0)
    next_pts = np.roll(contour, -1, axis=0)
    d1 = next_pts - contour
    d1_prev = contour - prev_pts
    d2 = d1 - d1_prev
    curvatures = np.linalg.norm(d2, axis=1)

    # 3. 转角统计（向量夹角）
    v1 = contour - prev_pts
    v2 = next_pts - contour
    v1_norm = np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8
    v2_norm = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8
    cos_angles = np.sum(v1 * v2, axis=1) / (v1_norm.squeeze() * v2_norm.squeeze())
    cos_angles = np.clip(cos_angles, -1, 1)
    angles = np.arccos(cos_angles) * 180 / np.pi  # 转为角度

    # 4. 局部方向变化率（相邻边的方向差）
    directions = np.arctan2(d1[:, 1], d1[:, 0])
    direction_changes = np.abs(np.diff(directions, append=directions[:1]))
    direction_changes = np.minimum(direction_changes, 2*np.pi - direction_changes)  # 处理周期性

    # 5. 高频振荡检测（连续3点的偏离）
    # 计算每个点到其前后点连线的距离
    line_vecs = next_pts - prev_pts
    point_vecs = contour - prev_pts
    # 叉积计算点到线的距离
    cross = point_vecs[:, 0] * line_vecs[:, 1] - point_vecs[:, 1] * line_vecs[:, 0]
    line_lens = np.linalg.norm(line_vecs, axis=1) + 1e-8
    deviations = np.abs(cross) / line_lens

    metrics = {
        # 点间距离
        'dist_mean': float(np.mean(dists)),
        'dist_std': float(np.std(dists)),
        'dist_max': float(np.max(dists)),
        'dist_min': float(np.min(dists)),
        'dist_cv': float(np.std(dists) / (np.mean(dists) + 1e-8)),  # 变异系数

        # 曲率
        'curv_mean': float(np.mean(curvatures)),
        'curv_std': float(np.std(curvatures)),
        'curv_max': float(np.max(curvatures)),
        'curv_p95': float(np.percentile(curvatures, 95)),
        'curv_outliers': int(np.sum(curvatures > np.percentile(curvatures, 95) * 2)),  # 异常尖锐点数量

        # 转角
        'angle_mean': float(np.mean(angles)),
        'angle_std': float(np.std(angles)),
        'sharp_angles': int(np.sum(angles > 120)),  # 大于120度的尖锐转角数量

        # 方向变化
        'dir_change_mean': float(np.mean(direction_changes)),
        'dir_change_std': float(np.std(direction_changes)),
        'dir_change_max': float(np.max(direction_changes)),

        # 高频振荡
        'deviation_mean': float(np.mean(deviations)),
        'deviation_std': float(np.std(deviations)),
        'deviation_max': float(np.max(deviations)),
        'high_freq_points': int(np.sum(deviations > np.mean(deviations) + 2*np.std(deviations))),  # 高频振荡点数量
    }

    return metrics


def visualize_scatter_and_metrics(img, gt_poly, init_poly, pred_poly, smoothed_poly, save_prefix, index):
    """
    创建综合可视化：散点图 + 连线图 + 统计指标
    """
    fig = plt.figure(figsize=(20, 12))

    # 计算统计指标
    pred_metrics = compute_burr_metrics(pred_poly)
    smoothed_metrics = compute_burr_metrics(smoothed_poly) if smoothed_poly is not None else None

    # 1. 原始预测 - 散点图
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(pred_poly[:, 0], pred_poly[:, 1], c='red', s=10, alpha=0.6, label='Pred Points')
    ax1.scatter(gt_poly[:, 0], gt_poly[:, 1], c='green', s=5, alpha=0.3, label='GT Points')
    ax1.set_title(f'Raw Prediction - Scatter (idx={index})', fontsize=12)
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    # 2. 原始预测 - 连线图
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.polylines(img.copy(), [gt_poly.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(img.copy(), [pred_poly.astype(np.int32)], True, (0, 0, 255), 2)
    ax2.plot(np.append(pred_poly[:, 0], pred_poly[0, 0]),
             np.append(pred_poly[:, 1], pred_poly[0, 1]),
             'r-', linewidth=2, label='Pred Connected')
    ax2.plot(np.append(gt_poly[:, 0], gt_poly[0, 0]),
             np.append(gt_poly[:, 1], gt_poly[0, 1]),
             'g-', linewidth=1, alpha=0.5, label='GT')
    ax2.set_title('Raw Prediction - Connected', fontsize=12)
    ax2.legend()
    ax2.axis('off')

    # 3. 曲率热力图
    ax3 = plt.subplot(2, 3, 3)
    prev_pts = np.roll(pred_poly, 1, axis=0)
    next_pts = np.roll(pred_poly, -1, axis=0)
    d2 = (next_pts - pred_poly) - (pred_poly - prev_pts)
    curvatures = np.linalg.norm(d2, axis=1)
    scatter = ax3.scatter(pred_poly[:, 0], pred_poly[:, 1], c=curvatures, cmap='hot', s=20)
    plt.colorbar(scatter, ax=ax3, label='Curvature')
    ax3.set_title(f'Curvature Heatmap (max={pred_metrics["curv_max"]:.2f})', fontsize=12)
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)

    # 4. 后处理 - 散点图（如果有）
    ax4 = plt.subplot(2, 3, 4)
    if smoothed_poly is not None:
        ax4.scatter(smoothed_poly[:, 0], smoothed_poly[:, 1], c='blue', s=10, alpha=0.6, label='Smoothed Points')
        ax4.scatter(pred_poly[:, 0], pred_poly[:, 1], c='red', s=5, alpha=0.3, label='Raw Points')
        ax4.set_title('After Smoothing - Scatter', fontsize=12)
    else:
        ax4.scatter(init_poly[:, 0], init_poly[:, 1], c='yellow', s=10, alpha=0.6, label='Init Points')
        ax4.scatter(pred_poly[:, 0], pred_poly[:, 1], c='red', s=5, alpha=0.3, label='Pred Points')
        ax4.set_title('Initialization - Scatter', fontsize=12)
    ax4.legend()
    ax4.axis('equal')
    ax4.grid(True, alpha=0.3)

    # 5. 后处理 - 连线图
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if smoothed_poly is not None:
        ax5.plot(np.append(smoothed_poly[:, 0], smoothed_poly[0, 0]),
                 np.append(smoothed_poly[:, 1], smoothed_poly[0, 1]),
                 'b-', linewidth=2, label='Smoothed')
        ax5.plot(np.append(pred_poly[:, 0], pred_poly[0, 0]),
                 np.append(pred_poly[:, 1], pred_poly[0, 1]),
                 'r-', linewidth=1, alpha=0.5, label='Raw')
        ax5.set_title('After Smoothing - Connected', fontsize=12)
    else:
        ax5.plot(np.append(init_poly[:, 0], init_poly[0, 0]),
                 np.append(init_poly[:, 1], init_poly[0, 1]),
                 'y-', linewidth=1, alpha=0.7, label='Init')
        ax5.plot(np.append(pred_poly[:, 0], pred_poly[0, 0]),
                 np.append(pred_poly[:, 1], pred_poly[0, 1]),
                 'r-', linewidth=2, label='Pred')
        ax5.set_title('Prediction vs Init - Connected', fontsize=12)
    ax5.legend()
    ax5.axis('off')

    # 6. 统计指标文本
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    metrics_text = "=== Raw Prediction Metrics ===\n\n"
    metrics_text += f"Point Distance:\n"
    metrics_text += f"  Mean: {pred_metrics['dist_mean']:.2f}, Std: {pred_metrics['dist_std']:.2f}\n"
    metrics_text += f"  CV: {pred_metrics['dist_cv']:.3f} (变异系数)\n\n"

    metrics_text += f"Curvature:\n"
    metrics_text += f"  Mean: {pred_metrics['curv_mean']:.2f}, Max: {pred_metrics['curv_max']:.2f}\n"
    metrics_text += f"  Outliers: {pred_metrics['curv_outliers']} points\n\n"

    metrics_text += f"Sharp Angles (>120°): {pred_metrics['sharp_angles']}\n\n"

    metrics_text += f"High-Freq Oscillation:\n"
    metrics_text += f"  Deviation Mean: {pred_metrics['deviation_mean']:.2f}\n"
    metrics_text += f"  High-Freq Points: {pred_metrics['high_freq_points']}\n\n"

    if smoothed_metrics:
        metrics_text += "=== After Smoothing ===\n\n"
        metrics_text += f"Curvature Max: {smoothed_metrics['curv_max']:.2f} "
        metrics_text += f"({(smoothed_metrics['curv_max']/pred_metrics['curv_max']-1)*100:+.1f}%)\n"
        metrics_text += f"Outliers: {smoothed_metrics['curv_outliers']} "
        metrics_text += f"({smoothed_metrics['curv_outliers']-pred_metrics['curv_outliers']:+d})\n"
        metrics_text += f"High-Freq Points: {smoothed_metrics['high_freq_points']} "
        metrics_text += f"({smoothed_metrics['high_freq_points']-pred_metrics['high_freq_points']:+d})\n"

    ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    return pred_metrics, smoothed_metrics


def load_v3_4_model():
    """加载V3.4模型"""
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 获取权重路径 - 优先使用single_overfit版本
    model_dir_name = cfg.model_dir.split('/')[-1]

    # 尝试多个可能的路径
    possible_paths = [
        os.path.join(_THIS_DIR, 'data/outputs', model_dir_name, 'checkpoints', 'latest.pt'),
        os.path.join(_THIS_DIR, 'data/outputs', f"{model_dir_name}_single_overfit", 'checkpoints', 'epoch_10000.pt'),
        os.path.join(_THIS_DIR, 'data/outputs', f"{model_dir_name}_single_overfit", 'checkpoints', 'latest.pt'),
    ]

    ckpt_path = None
    for path in possible_paths:
        if os.path.exists(path):
            ckpt_path = path
            break

    if ckpt_path is None:
        print(f"[!] Error: No checkpoint found! Tried:")
        for path in possible_paths:
            print(f"    - {path}")
        sys.exit(1)

    print(f"[*] Loading V3.4 Weights: {ckpt_path}")

    ckpt_obj = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt_obj.get('state_dict') or ckpt_obj.get('model') or ckpt_obj.get('net') or ckpt_obj

    from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
    sd = remap_legacy_state_dict(sd)
    wrapper = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network

    info = wrapper.load_state_dict(sd, strict=False)
    print(f"[✔] Success: {len(sd) - len(info.missing_keys)} layers matched.")

    return trainer.network.to(device).eval(), device


def run_inference_and_analyze(model, device, batch, save_dir, index):
    """运行推理并分析毛刺"""
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    dr = float(snake_config.down_ratio)
    core = model.net if hasattr(model, 'net') else model

    with torch.no_grad():
        # 1. 网络前向
        yolo_out = core.yolo(batch['inp'])
        if isinstance(yolo_out, (list, tuple)):
            feat_p2 = yolo_out[1][0] if len(yolo_out) > 1 else yolo_out[0]
        else:
            feat_p2 = yolo_out
        cnn_feature = core.cnn_proj(feat_p2)

        # 2. 获取初始化和GT
        gt_all = batch['i_gt_py']
        if gt_all.numel() == 0:
            print("  [!] No GT polygons, skipping...")
            return

        B, M, P, _ = gt_all.shape

        if 'i_it_py' in batch and batch['i_it_py'].numel() > 0:
            i_it_py = batch['i_it_py'].view(-1, P, 2)
        else:
            print("  [!] No init poly, skipping...")
            return

        c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)

        if B == 1:
            py_ind = torch.zeros(i_it_py.size(0), dtype=torch.long, device=device)
        else:
            py_ind = torch.cat([torch.full((M,), i, dtype=torch.long, device=device) for i in range(B)])

        # 3. 原始预测（不加后处理）
        disp = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind, steps=50)
        pred_polys = (i_it_py + disp).cpu().numpy() * dr

        # 4. 简单平滑后处理（用于对比）
        from edge_smoothing import smooth_contours_numpy
        smoothed_polys = []
        for poly in pred_polys:
            smoothed = smooth_contours_numpy(poly, curvature_threshold=5.0, iterations=2)
            smoothed_polys.append(smoothed)
        smoothed_polys = np.array(smoothed_polys)

        init_np = i_it_py.cpu().numpy() * dr
        gt_np = gt_all.cpu().numpy() * dr

    # 5. 获取图像
    if 'orig_img' in batch:
        img_raw = batch['orig_img'][0]
        img = img_raw.detach().cpu().numpy() if torch.is_tensor(img_raw) else img_raw
        img = img.astype(np.uint8)
    else:
        img = np.zeros((512, 512, 3), dtype=np.uint8)

    # 6. 可视化和分析
    save_prefix = os.path.join(save_dir, f"burr_idx{index}")

    # 分析第一个轮廓
    pred_metrics, smoothed_metrics = visualize_scatter_and_metrics(
        img.copy(),
        gt_np[0, 0],
        init_np[0],
        pred_polys[0],
        smoothed_polys[0],
        save_prefix,
        index
    )

    # 7. 点序分析
    print("\n[*] 开始点序分析...")
    order_results, reordered_poly = analyze_point_order_issue(
        pred_polys[0],
        img.copy(),
        save_prefix
    )

    # 保存指标到JSON
    metrics_data = {
        'index': index,
        'raw_prediction': pred_metrics,
        'after_smoothing': smoothed_metrics,
        'point_order_analysis': order_results
    }

    with open(f"{save_prefix}_metrics.json", 'w') as f:
        json.dump(metrics_data, f, indent=2)

    print(f"[✔] Analysis saved: {save_prefix}_analysis.png")
    print(f"    Metrics: {save_prefix}_metrics.json")
    print(f"    Point order: {save_prefix}_original_order.png")
    print(f"    Reordered: {save_prefix}_reordered.png")

    return metrics_data


def main():
    index = int(os.environ.get('INDEX', 0))

    print("=" * 60)
    print("毛刺问题分析 - V3.4模型")
    print(f"分析样本: index={index}")
    print("=" * 60)

    model, device = load_v3_4_model()
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)

    save_dir = os.path.join(_THIS_DIR, 'visual/burr_analysis')
    os.makedirs(save_dir, exist_ok=True)

    batch = collator([dataset[index]])
    metrics = run_inference_and_analyze(model, device, batch, save_dir, index)

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
