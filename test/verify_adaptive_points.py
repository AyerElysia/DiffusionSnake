"""
快速验证：自适应点数后处理

目标：不修改训练，只在推理后调整点数，验证假设
"""

import sys, os

# 在最开始设置GPU
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 默认使用GPU 1

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from scipy import interpolate

_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
sys.path.insert(0, _THIS_DIR)

os.environ['CFG_FILE'] = 'configs/btcv_diffusion_dit_v3_4_single_overfit.yaml'

from lib.config import cfg
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.datasets.make_dataset import make_dataset
from lib.datasets.collate_batch import make_collator
from lib.datasets.transforms import make_transforms
from lib.utils.snake import snake_config, snake_decode, snake_gcn_utils


def compute_perimeter(contour):
    """计算轮廓周长"""
    dists = np.linalg.norm(np.diff(contour, axis=0, append=contour[:1]), axis=1)
    return np.sum(dists)


def compute_curvature(contour):
    """计算曲率"""
    prev_pts = np.roll(contour, 1, axis=0)
    next_pts = np.roll(contour, -1, axis=0)
    d2 = (next_pts - contour) - (contour - prev_pts)
    curvatures = np.linalg.norm(d2, axis=1)
    return curvatures


def uniform_resample(contour, target_points):
    """均匀重采样到目标点数"""
    # 计算累积弧长
    dists = np.linalg.norm(np.diff(contour, axis=0, append=contour[:1]), axis=1)
    cumsum = np.concatenate([[0], np.cumsum(dists)])
    cumsum_norm = cumsum / cumsum[-1]

    # 插值
    fx = interpolate.interp1d(cumsum_norm, contour[:, 0], kind='linear')
    fy = interpolate.interp1d(cumsum_norm, contour[:, 1], kind='linear')

    # 均匀采样
    t = np.linspace(0, 1, target_points, endpoint=False)
    x_new = fx(t)
    y_new = fy(t)

    return np.stack([x_new, y_new], axis=1)


def adaptive_resample(contour, target_density=2.5):
    """
    自适应重采样

    Args:
        contour: (N, 2) numpy array
        target_density: 目标点密度（像素/点）

    Returns:
        resampled: 重采样后的轮廓（回到128点用于对比）
        num_points: 实际使用的点数
    """
    # 计算周长
    perimeter = compute_perimeter(contour)

    # 计算目标点数
    target_points = int(perimeter / target_density)
    target_points = max(32, min(target_points, 256))
    target_points = (target_points // 4) * 4

    # 下采样到目标点数
    downsampled = uniform_resample(contour, target_points)

    # 上采样回128点（用于公平对比）
    upsampled = uniform_resample(downsampled, 128)

    return upsampled, target_points


def load_model():
    """加载V3.4模型"""
    # 强制使用环境变量指定的GPU
    import os
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        device = torch.device('cuda:0')  # CUDA_VISIBLE_DEVICES会重新映射，所以用0
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network = make_network(cfg)
    trainer = make_trainer(cfg, network)

    ckpt_path = os.path.join(_THIS_DIR, 'data/outputs/btcv_diffusion_dit_v3_4_single_overfit/checkpoints/epoch_10000.pt')

    print(f"[*] 加载模型: {ckpt_path}")
    ckpt_obj = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt_obj.get('state_dict') or ckpt_obj.get('model') or ckpt_obj.get('net') or ckpt_obj

    from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
    sd = remap_legacy_state_dict(sd)
    wrapper = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    wrapper.load_state_dict(sd, strict=False)

    return trainer.network.to(device).eval(), device


def run_inference_and_verify(model, device, batch, save_dir):
    """运行推理并验证自适应点数效果"""

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

        # 获取初始化
        gt_all = batch['i_gt_py']
        B, M, P, _ = gt_all.shape
        i_it_py = batch['i_it_py'].view(-1, P, 2)
        c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)

        if B == 1:
            py_ind = torch.zeros(i_it_py.size(0), dtype=torch.long, device=device)
        else:
            py_ind = torch.cat([torch.full((M,), i, dtype=torch.long, device=device) for i in range(B)])

        # 原始预测
        disp = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind, steps=50)
        pred_polys = (i_it_py + disp).cpu().numpy() * dr
        gt_np = gt_all.cpu().numpy() * dr

    # 获取图像
    if 'orig_img' in batch:
        img_raw = batch['orig_img'][0]
        img = img_raw.detach().cpu().numpy() if torch.is_tensor(img_raw) else img_raw
        img = img.astype(np.uint8)
    else:
        img = np.zeros((512, 512, 3), dtype=np.uint8)

    # 对每个轮廓进行自适应重采样
    results = []
    adaptive_polys = []

    print("\n" + "="*100)
    print("自适应点数验证结果")
    print("="*100)
    print(f"{'轮廓':<8} {'周长':<10} {'原始点数':<12} {'建议点数':<12} {'原始曲率':<12} {'自适应曲率':<15} {'改善':<12}")
    print("-"*100)

    for i, pred in enumerate(pred_polys):
        # 计算周长
        perimeter = compute_perimeter(pred)

        # 原始曲率
        curv_original = compute_curvature(pred)

        # 自适应重采样
        pred_adaptive, num_points = adaptive_resample(pred, target_density=2.5)
        curv_adaptive = compute_curvature(pred_adaptive)

        # 统计
        improvement = (curv_original.max() - curv_adaptive.max()) / curv_original.max() * 100

        results.append({
            'contour_id': i,
            'perimeter': float(perimeter),
            'original_points': 128,
            'adaptive_points': int(num_points),
            'curv_original': float(curv_original.max()),
            'curv_adaptive': float(curv_adaptive.max()),
            'improvement': float(improvement)
        })

        adaptive_polys.append(pred_adaptive)

        print(f"{i:<8} {perimeter:<10.1f} {128:<12} {num_points:<12} "
              f"{curv_original.max():<12.2f} {curv_adaptive.max():<15.2f} {improvement:<12.1f}%")

    adaptive_polys = np.array(adaptive_polys)

    # 可视化对比
    visualize_comparison(img, gt_np[0], pred_polys, adaptive_polys, results, save_dir)

    # 保存结果
    with open(os.path.join(save_dir, 'adaptive_verification_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # 统计汇总
    print("\n" + "="*100)
    print("统计汇总")
    print("="*100)

    small_contours = [r for r in results if r['perimeter'] < 150]
    large_contours = [r for r in results if r['perimeter'] >= 150]

    if small_contours:
        avg_improvement_small = np.mean([r['improvement'] for r in small_contours])
        print(f"小轮廓（周长<150）平均改善: {avg_improvement_small:.1f}%")

    if large_contours:
        avg_improvement_large = np.mean([r['improvement'] for r in large_contours])
        print(f"大轮廓（周长≥150）平均改善: {avg_improvement_large:.1f}%")

    avg_improvement_all = np.mean([r['improvement'] for r in results])
    print(f"整体平均改善: {avg_improvement_all:.1f}%")

    print("\n" + "="*100)

    return results


def visualize_comparison(img, gt_polys, pred_polys, adaptive_polys, results, save_dir):
    """可视化对比"""
    num_contours = len(pred_polys)

    fig = plt.figure(figsize=(20, 6 * num_contours))

    for i in range(num_contours):
        gt = gt_polys[i]
        pred = pred_polys[i]
        adaptive = adaptive_polys[i]
        result = results[i]

        base_idx = i * 3

        # 1. 原始预测
        ax1 = plt.subplot(num_contours, 3, base_idx + 1)
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0.3)
        ax1.plot(np.append(gt[:, 0], gt[0, 0]), np.append(gt[:, 1], gt[0, 1]),
                'g-', linewidth=1, alpha=0.5, label='GT')
        ax1.plot(np.append(pred[:, 0], pred[0, 0]), np.append(pred[:, 1], pred[0, 1]),
                'r-', linewidth=2, label='Original (128pts)')
        ax1.set_title(f'C{i}: Original\nCurv={result["curv_original"]:.2f}', fontsize=12)
        ax1.legend()
        ax1.axis('off')

        # 2. 自适应点数
        ax2 = plt.subplot(num_contours, 3, base_idx + 2)
        ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0.3)
        ax2.plot(np.append(gt[:, 0], gt[0, 0]), np.append(gt[:, 1], gt[0, 1]),
                'g-', linewidth=1, alpha=0.5, label='GT')
        ax2.plot(np.append(adaptive[:, 0], adaptive[0, 0]), np.append(adaptive[:, 1], adaptive[0, 1]),
                'b-', linewidth=2, label=f'Adaptive ({result["adaptive_points"]}pts)')
        ax2.set_title(f'C{i}: Adaptive\nCurv={result["curv_adaptive"]:.2f}', fontsize=12)
        ax2.legend()
        ax2.axis('off')

        # 3. 叠加对比
        ax3 = plt.subplot(num_contours, 3, base_idx + 3)
        ax3.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0.3)
        ax3.plot(np.append(pred[:, 0], pred[0, 0]), np.append(pred[:, 1], pred[0, 1]),
                'r-', linewidth=1, alpha=0.7, label='Original')
        ax3.plot(np.append(adaptive[:, 0], adaptive[0, 0]), np.append(adaptive[:, 1], adaptive[0, 1]),
                'b-', linewidth=2, alpha=0.7, label='Adaptive')
        ax3.set_title(f'C{i}: Comparison\nImprovement={result["improvement"]:.1f}%', fontsize=12)
        ax3.legend()
        ax3.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'adaptive_verification_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[✔] 可视化已保存: {save_dir}/adaptive_verification_comparison.png")


def main():
    print("="*100)
    print("自适应点数快速验证")
    print("="*100)

    model, device = load_model()
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)

    save_dir = os.path.join(_THIS_DIR, 'test/adaptive_verification')
    os.makedirs(save_dir, exist_ok=True)

    batch = collator([dataset[0]])
    results = run_inference_and_verify(model, device, batch, save_dir)

    print(f"\n结果已保存到: {save_dir}")
    print("="*100)


if __name__ == '__main__':
    main()
