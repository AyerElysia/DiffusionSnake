"""
毛刺问题完整分析 - V3.4单样本过拟合版本

关键改进：
1. 预测整张图像的所有轮廓
2. 对比原始预测、平滑后的散点图
3. 针对V3.4训练的样本（样本0）
"""

import sys, os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CFG = os.path.join(_THIS_DIR, 'configs', 'btcv_diffusion_dit_v3_4_single_overfit.yaml')
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
from edge_smoothing import smooth_contours_numpy
from analyze_point_order import detect_order_jumps


def compute_burr_metrics(contour):
    """计算毛刺统计指标"""
    P = len(contour)

    # 点间距离
    dists = np.linalg.norm(np.diff(contour, axis=0, append=contour[:1]), axis=1)

    # 曲率
    prev_pts = np.roll(contour, 1, axis=0)
    next_pts = np.roll(contour, -1, axis=0)
    d2 = (next_pts - contour) - (contour - prev_pts)
    curvatures = np.linalg.norm(d2, axis=1)

    # 转角
    v1 = contour - prev_pts
    v2 = next_pts - contour
    v1_norm = np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8
    v2_norm = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8
    cos_angles = np.sum(v1 * v2, axis=1) / (v1_norm.squeeze() * v2_norm.squeeze())
    cos_angles = np.clip(cos_angles, -1, 1)
    angles = np.arccos(cos_angles) * 180 / np.pi

    # 高频振荡
    line_vecs = next_pts - prev_pts
    point_vecs = contour - prev_pts
    cross = point_vecs[:, 0] * line_vecs[:, 1] - point_vecs[:, 1] * line_vecs[:, 0]
    line_lens = np.linalg.norm(line_vecs, axis=1) + 1e-8
    deviations = np.abs(cross) / line_lens

    return {
        'dist_mean': float(np.mean(dists)),
        'dist_std': float(np.std(dists)),
        'dist_cv': float(np.std(dists) / (np.mean(dists) + 1e-8)),
        'curv_mean': float(np.mean(curvatures)),
        'curv_max': float(np.max(curvatures)),
        'sharp_angles': int(np.sum(angles > 120)),
        'high_freq_points': int(np.sum(deviations > np.mean(deviations) + 2*np.std(deviations))),
    }


def visualize_all_contours_scatter(img, gt_polys, pred_polys, smoothed_polys, save_path, sample_idx):
    """
    可视化所有轮廓的散点图对比

    关键：对比原始预测、平滑后的散点图
    """
    num_contours = len(pred_polys)

    # 创建大图：每个轮廓一行，4列（GT连线、原始散点、平滑散点、对比）
    fig = plt.figure(figsize=(24, 6 * num_contours))

    for contour_idx in range(num_contours):
        gt = gt_polys[contour_idx]
        pred = pred_polys[contour_idx]
        smoothed = smoothed_polys[contour_idx]

        # 计算指标
        pred_metrics = compute_burr_metrics(pred)
        smoothed_metrics = compute_burr_metrics(smoothed)

        # 点序跳跃检测
        pred_jumps, _, _ = detect_order_jumps(pred)
        smoothed_jumps, _, _ = detect_order_jumps(smoothed)

        base_idx = contour_idx * 4

        # 1. GT连线图（参考）
        ax1 = plt.subplot(num_contours, 4, base_idx + 1)
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0.3)
        ax1.plot(np.append(gt[:, 0], gt[0, 0]),
                np.append(gt[:, 1], gt[0, 1]),
                'g-', linewidth=2, label='GT')
        ax1.scatter(gt[:, 0], gt[:, 1], c='green', s=20, alpha=0.6)
        ax1.set_title(f'Contour {contour_idx}: GT (Reference)', fontsize=12)
        ax1.legend()
        ax1.axis('off')

        # 2. 原始预测 - 散点图（关键！）
        ax2 = plt.subplot(num_contours, 4, base_idx + 2)
        ax2.scatter(pred[:, 0], pred[:, 1], c='red', s=30, alpha=0.7, label='Raw Pred')
        ax2.scatter(gt[:, 0], gt[:, 1], c='green', s=10, alpha=0.3, label='GT')
        # 标注起点
        ax2.scatter(pred[0, 0], pred[0, 1], c='yellow', s=100, marker='*',
                   edgecolors='black', linewidths=2, zorder=10)
        ax2.set_title(f'Raw Prediction - SCATTER\n'
                     f'Jumps={len(pred_jumps)}, Curv={pred_metrics["curv_max"]:.1f}, '
                     f'Sharp∠={pred_metrics["sharp_angles"]}', fontsize=11)
        ax2.legend()
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)

        # 3. 平滑后 - 散点图（关键对比！）
        ax3 = plt.subplot(num_contours, 4, base_idx + 3)
        ax3.scatter(smoothed[:, 0], smoothed[:, 1], c='blue', s=30, alpha=0.7, label='Smoothed')
        ax3.scatter(pred[:, 0], pred[:, 1], c='red', s=10, alpha=0.3, label='Raw')
        ax3.scatter(smoothed[0, 0], smoothed[0, 1], c='yellow', s=100, marker='*',
                   edgecolors='black', linewidths=2, zorder=10)
        ax3.set_title(f'After Smoothing - SCATTER\n'
                     f'Jumps={len(smoothed_jumps)}, Curv={smoothed_metrics["curv_max"]:.1f}, '
                     f'Sharp∠={smoothed_metrics["sharp_angles"]}', fontsize=11)
        ax3.legend()
        ax3.axis('equal')
        ax3.grid(True, alpha=0.3)

        # 4. 连线对比图
        ax4 = plt.subplot(num_contours, 4, base_idx + 4)
        ax4.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0.3)
        # GT
        ax4.plot(np.append(gt[:, 0], gt[0, 0]),
                np.append(gt[:, 1], gt[0, 1]),
                'g-', linewidth=1, alpha=0.5, label='GT')
        # 原始预测
        ax4.plot(np.append(pred[:, 0], pred[0, 0]),
                np.append(pred[:, 1], pred[0, 1]),
                'r-', linewidth=2, alpha=0.7, label='Raw')
        # 平滑后
        ax4.plot(np.append(smoothed[:, 0], smoothed[0, 0]),
                np.append(smoothed[:, 1], smoothed[0, 1]),
                'b--', linewidth=1.5, alpha=0.7, label='Smoothed')
        ax4.set_title(f'Connected Comparison\n'
                     f'Curv: {pred_metrics["curv_max"]:.1f}→{smoothed_metrics["curv_max"]:.1f} '
                     f'({(smoothed_metrics["curv_max"]/pred_metrics["curv_max"]-1)*100:+.0f}%)',
                     fontsize=11)
        ax4.legend()
        ax4.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[✔] 保存完整散点图对比: {save_path}")


def load_v3_4_model():
    """加载V3.4单样本过拟合模型"""
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 查找checkpoint
    possible_paths = [
        os.path.join(_THIS_DIR, 'data/outputs/btcv_diffusion_dit_v3_4_single_overfit/checkpoints/epoch_10000.pt'),
        os.path.join(_THIS_DIR, 'data/outputs/btcv_diffusion_dit_v3_4_single_overfit/checkpoints/latest.pt'),
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


def run_full_image_inference(model, device, batch, save_dir):
    """对整张图像的所有轮廓进行推理和分析"""

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
            return

        B, M, P, _ = gt_all.shape
        print(f"  [*] 检测到 {M} 个轮廓")

        if 'i_it_py' not in batch or batch['i_it_py'].numel() == 0:
            print("  [!] 没有初始化轮廓")
            return

        i_it_py = batch['i_it_py'].view(-1, P, 2)
        c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)

        if B == 1:
            py_ind = torch.zeros(i_it_py.size(0), dtype=torch.long, device=device)
        else:
            py_ind = torch.cat([torch.full((M,), i, dtype=torch.long, device=device) for i in range(B)])

        # 原始预测（不加后处理）
        disp = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind, steps=50)
        pred_polys = (i_it_py + disp).cpu().numpy() * dr

        # 保存原始预测轮廓（用于后续分析）
        np.save(os.path.join(save_dir, 'pred_contours_raw.npy'), pred_polys)
        print(f"  [✔] 保存原始预测: {os.path.join(save_dir, 'pred_contours_raw.npy')}")

        # 平滑后处理
        smoothed_polys = []
        for poly in pred_polys:
            smoothed = smooth_contours_numpy(poly, curvature_threshold=5.0, iterations=2)
            smoothed_polys.append(smoothed)
        smoothed_polys = np.array(smoothed_polys)

        gt_np = gt_all.cpu().numpy() * dr

    # 获取图像
    if 'orig_img' in batch:
        img_raw = batch['orig_img'][0]
        img = img_raw.detach().cpu().numpy() if torch.is_tensor(img_raw) else img_raw
        img = img.astype(np.uint8)
    else:
        img = np.zeros((512, 512, 3), dtype=np.uint8)

    # 可视化所有轮廓的散点图对比
    save_path = os.path.join(save_dir, f"full_image_scatter_comparison.png")
    visualize_all_contours_scatter(img, gt_np[0], pred_polys, smoothed_polys, save_path, 0)

    # 统计所有轮廓的指标
    all_metrics = []
    for i in range(len(pred_polys)):
        pred_metrics = compute_burr_metrics(pred_polys[i])
        smoothed_metrics = compute_burr_metrics(smoothed_polys[i])
        pred_jumps, _, _ = detect_order_jumps(pred_polys[i])
        smoothed_jumps, _, _ = detect_order_jumps(smoothed_polys[i])

        all_metrics.append({
            'contour_id': i,
            'raw': pred_metrics,
            'smoothed': smoothed_metrics,
            'raw_jumps': len(pred_jumps),
            'smoothed_jumps': len(smoothed_jumps),
        })

    # 保存指标
    metrics_path = os.path.join(save_dir, "full_image_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"[✔] 指标已保存: {metrics_path}")

    # 打印汇总
    print("\n" + "="*80)
    print("所有轮廓统计汇总")
    print("="*80)
    print(f"{'轮廓':<8} {'原始跳跃':<12} {'平滑跳跃':<12} {'原始曲率':<12} {'平滑曲率':<12} {'曲率降低':<12}")
    print("-"*80)

    for m in all_metrics:
        curv_reduction = (m['raw']['curv_max'] - m['smoothed']['curv_max']) / m['raw']['curv_max'] * 100
        print(f"{m['contour_id']:<8} {m['raw_jumps']:<12} {m['smoothed_jumps']:<12} "
              f"{m['raw']['curv_max']:<12.2f} {m['smoothed']['curv_max']:<12.2f} {curv_reduction:<12.1f}%")

    print("="*80)


def main():
    print("="*80)
    print("毛刺问题完整分析 - V3.4单样本过拟合")
    print("="*80)

    model, device = load_v3_4_model()
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)

    save_dir = os.path.join(_THIS_DIR, 'visual/burr_v3_4_full')
    os.makedirs(save_dir, exist_ok=True)

    # V3.4是在样本0上训练的，所以分析样本0
    print(f"\n[*] 分析V3.4训练的样本（样本0）...")
    batch = collator([dataset[0]])
    run_full_image_inference(model, device, batch, save_dir)

    print("\n" + "="*80)
    print("分析完成！")
    print(f"结果保存在: {save_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
