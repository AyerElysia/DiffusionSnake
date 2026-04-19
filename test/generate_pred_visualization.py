"""
生成清晰的GT vs 预测对比可视化
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys, os

_PROJ_DIR = '/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30'
sys.path.insert(0, _PROJ_DIR)

os.environ['CFG_FILE'] = os.path.join(_PROJ_DIR, 'configs/btcv_diffusion_dit_v3_4_single_overfit.yaml')

from lib.datasets.make_dataset import make_dataset
from lib.datasets.collate_batch import make_collator
from lib.datasets.transforms import make_transforms
from lib.config import cfg
from lib.utils.snake import snake_config

# 加载数据
dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
collator = make_collator(cfg)
batch = collator([dataset[0]])

dr = float(snake_config.down_ratio)
gt_all = batch['i_gt_py']
gt_contours = gt_all.cpu().numpy()[0] * dr

# 加载预测
pred_contours = np.load(os.path.join(_PROJ_DIR, 'visual/burr_v3_4_full/pred_contours_raw.npy'))

# 获取图像
if 'orig_img' in batch:
    img_raw = batch['orig_img'][0]
    img = img_raw.detach().cpu().numpy() if hasattr(img_raw, 'detach') else img_raw
    img = img.astype(np.uint8)
else:
    img = np.zeros((512, 512, 3), dtype=np.uint8)

# 计算IoU
def compute_iou(c1, c2, size=(512, 512)):
    m1 = np.zeros(size, dtype=np.uint8)
    m2 = np.zeros(size, dtype=np.uint8)
    cv2.fillPoly(m1, [c1.astype(np.int32)], 1)
    cv2.fillPoly(m2, [c2.astype(np.int32)], 1)
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter) / float(union) if union > 0 else 0.0

# 创建可视化
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

for i in range(6):
    ax = axes[i]

    # 显示图像
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0.5)

    gt_pts = gt_contours[i]
    pred_pts = pred_contours[i]

    # 绘制GT（绿色粗线）
    ax.plot(np.append(gt_pts[:, 0], gt_pts[0, 0]),
            np.append(gt_pts[:, 1], gt_pts[0, 1]),
            'g-', linewidth=3, label='GT', alpha=0.8)

    # 绘制预测（红色细线）
    ax.plot(np.append(pred_pts[:, 0], pred_pts[0, 0]),
            np.append(pred_pts[:, 1], pred_pts[0, 1]),
            'r-', linewidth=2, label='Prediction', alpha=0.8)

    # 标记起点
    ax.scatter(gt_pts[0, 0], gt_pts[0, 1], c='green', s=100, marker='*',
               edgecolors='black', linewidths=2, zorder=10, label='GT Start')
    ax.scatter(pred_pts[0, 0], pred_pts[0, 1], c='red', s=100, marker='*',
               edgecolors='black', linewidths=2, zorder=10, label='Pred Start')

    # 计算指标
    iou = compute_iou(pred_pts, gt_pts)
    area_gt = 0.5 * np.abs(np.dot(gt_pts[:, 0], np.roll(gt_pts[:, 1], 1)) -
                            np.dot(gt_pts[:, 1], np.roll(gt_pts[:, 0], 1)))
    area_pred = 0.5 * np.abs(np.dot(pred_pts[:, 0], np.roll(pred_pts[:, 1], 1)) -
                              np.dot(pred_pts[:, 1], np.roll(pred_pts[:, 0], 1)))

    # 计算曲率
    def compute_curv(c):
        prev = np.roll(c, 1, axis=0)
        next = np.roll(c, -1, axis=0)
        d2 = (next - c) - (c - prev)
        return np.linalg.norm(d2, axis=1)

    curv_pred = compute_curv(pred_pts)

    ax.set_title(f'Contour {i}\n'
                 f'IoU={iou:.3f}, Area(GT)={area_gt:.0f}, Area(Pred)={area_pred:.0f}\n'
                 f'Max Curvature={np.max(curv_pred):.2f}',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.axis('off')

plt.suptitle('V3.4 Prediction vs Ground Truth', fontsize=16, fontweight='bold')
plt.tight_layout()

output_path = os.path.join(_PROJ_DIR, 'visual/burr_v3_4_full/gt_vs_pred_comparison.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"预测对比可视化已保存到: {output_path}")

# 再生成一个放大的局部对比
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

for i in range(6):
    ax = axes[i]

    gt_pts = gt_contours[i]
    pred_pts = pred_contours[i]

    # 只绘制轮廓，不显示图像
    ax.plot(np.append(gt_pts[:, 0], gt_pts[0, 0]),
            np.append(gt_pts[:, 1], gt_pts[0, 1]),
            'g-', linewidth=3, label='GT', alpha=0.8)

    ax.plot(np.append(pred_pts[:, 0], pred_pts[0, 0]),
            np.append(pred_pts[:, 1], pred_pts[0, 1]),
            'r--', linewidth=2, label='Prediction', alpha=0.8)

    # 散点显示采样点
    ax.scatter(gt_pts[:, 0], gt_pts[:, 1], c='green', s=20, alpha=0.5, zorder=5)
    ax.scatter(pred_pts[:, 0], pred_pts[:, 1], c='red', s=20, alpha=0.5, zorder=5)

    iou = compute_iou(pred_pts, gt_pts)
    area_gt = 0.5 * np.abs(np.dot(gt_pts[:, 0], np.roll(gt_pts[:, 1], 1)) -
                            np.dot(gt_pts[:, 1], np.roll(gt_pts[:, 0], 1)))

    ax.set_title(f'Contour {i}: IoU={iou:.3f}, Area={area_gt:.0f}',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

plt.suptitle('V3.4 Contour Comparison (Points Visible)', fontsize=16, fontweight='bold')
plt.tight_layout()

output_path2 = os.path.join(_PROJ_DIR, 'visual/burr_v3_4_full/contour_points_comparison.png')
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"轮廓点对比可视化已保存到: {output_path2}")

print("\n可视化文件位置:")
print(f"1. GT vs 预测叠加图: {output_path}")
print(f"2. 轮廓点对比图: {output_path2}")
print(f"3. 散点对比图（已存在）: visual/burr_v3_4_full/full_image_scatter_comparison.png")
