import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.interpolate import interp1d
import sys
sys.path.insert(0, '/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30')

pred_contours = np.load('/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30/visual/burr_v3_4_full/pred_contours_raw.npy')

with open('/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30/visual/strategy_exploration/strategy_comparison.json', 'r') as f:
    strategies = json.load(f)

best_strategy = strategies[0]

def resample_contour(contour, num_points):
    if num_points == len(contour):
        return contour
    dists = np.linalg.norm(np.diff(contour, axis=0, append=contour[:1]), axis=1)
    cumsum = np.concatenate([[0], np.cumsum(dists)])
    total_length = cumsum[-1]
    target_lengths = np.linspace(0, total_length, num_points, endpoint=False)
    interp_x = interp1d(cumsum, np.concatenate([contour[:, 0], [contour[0, 0]]]), kind='linear')
    interp_y = interp1d(cumsum, np.concatenate([contour[:, 1], [contour[0, 1]]]), kind='linear')
    new_x = interp_x(target_lengths)
    new_y = interp_y(target_lengths)
    return np.stack([new_x, new_y], axis=1)

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

for i in range(6):
    ax = axes[i]
    result = best_strategy['results'][i]
    orig_contour = pred_contours[i]
    new_points = result['new_points']
    new_contour = resample_contour(orig_contour, new_points)

    ax.plot(orig_contour[:, 0], orig_contour[:, 1], 'r-', linewidth=2, label=f'Original (128 pts)', alpha=0.7)
    ax.scatter(orig_contour[:, 0], orig_contour[:, 1], c='red', s=15, alpha=0.4)
    ax.plot(new_contour[:, 0], new_contour[:, 1], 'b-', linewidth=2, label=f'Adaptive ({new_points} pts)', alpha=0.7)
    ax.scatter(new_contour[:, 0], new_contour[:, 1], c='blue', s=30, alpha=0.6, zorder=10)

    ax.set_title(f'Contour {i}: {result["orig_points"]}→{result["new_points"]} points\n'
                 f'IoU: {result["orig_iou"]:.3f}→{result["new_iou"]:.3f} ({result["iou_improvement"]:+.2f}%)\n'
                 f'Curv: {result["orig_curv"]:.2f}→{result["new_curv"]:.2f} ({result["curv_improvement"]:+.1f}%)',
                 fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Best Strategy: {best_strategy["strategy_name"]}\nAvg IoU: {best_strategy["avg_iou_orig"]:.3f}→{best_strategy["avg_iou_new"]:.3f} ({best_strategy["avg_iou_improvement"]:+.2f}%)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30/visual/strategy_exploration/adaptive_points_visualization.png', dpi=150, bbox_inches='tight')
print("自适应点数可视化已保存到: visual/strategy_exploration/adaptive_points_visualization.png")
