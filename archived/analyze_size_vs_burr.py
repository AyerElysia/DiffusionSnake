"""
分析轮廓大小与毛刺程度的关系
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 读取指标
with open('visual/burr_v3_4_full/full_image_metrics.json', 'r') as f:
    metrics = json.load(f)

# 读取实际的轮廓数据来计算真实的大小
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

os.environ['CFG_FILE'] = 'configs/btcv_diffusion_dit_v3_4_single_overfit.yaml'

from lib.config import cfg
from lib.datasets.make_dataset import make_dataset
from lib.datasets.collate_batch import make_collator
from lib.datasets.transforms import make_transforms
from lib.utils.snake import snake_config

# 加载数据
dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
collator = make_collator(cfg)
batch = collator([dataset[0]])

# 获取GT轮廓
gt_all = batch['i_gt_py']
dr = float(snake_config.down_ratio)
gt_np = gt_all.cpu().numpy() * dr

print('='*100)
print('轮廓大小 vs 毛刺程度深度分析')
print('='*100)

# 计算每个轮廓的真实大小
contour_data = []

for i in range(len(gt_np[0])):
    gt_contour = gt_np[0][i]

    # 计算周长
    perimeter = np.sum(np.linalg.norm(np.diff(gt_contour, axis=0, append=gt_contour[:1]), axis=1))

    # 计算面积（使用Shoelace公式）
    x = gt_contour[:, 0]
    y = gt_contour[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # 计算等效半径
    radius = np.sqrt(area / np.pi)

    # 获取毛刺指标
    m = metrics[i]

    contour_data.append({
        'id': i,
        'perimeter': perimeter,
        'area': area,
        'radius': radius,
        'curv_max': m['raw']['curv_max'],
        'sharp_angles': m['raw']['sharp_angles'],
        'high_freq_points': m['raw']['high_freq_points'],
        'dist_cv': m['raw']['dist_cv'],
    })

# 排序并显示
print(f"\n{'轮廓':<8} {'周长':<12} {'面积':<12} {'半径':<12} {'最大曲率':<12} {'尖锐角':<10} {'毛刺等级':<12}")
print('-'*100)

for data in sorted(contour_data, key=lambda x: x['area']):
    cid = data['id']
    perimeter = data['perimeter']
    area = data['area']
    radius = data['radius']
    curv = data['curv_max']
    angles = data['sharp_angles']

    if curv > 20:
        level = 'A (严重)'
    elif curv > 10:
        level = 'B (中等)'
    else:
        level = 'C (轻微)'

    print(f"{cid:<8} {perimeter:<12.1f} {area:<12.1f} {radius:<12.1f} {curv:<12.2f} {angles:<10} {level:<12}")

# 分析相关性
print('\n' + '='*100)
print('相关性分析')
print('='*100)

areas = [d['area'] for d in contour_data]
curvs = [d['curv_max'] for d in contour_data]
angles = [d['sharp_angles'] for d in contour_data]
radii = [d['radius'] for d in contour_data]

# 计算相关系数
corr_area_curv = np.corrcoef(areas, curvs)[0, 1]
corr_area_angles = np.corrcoef(areas, angles)[0, 1]
corr_radius_curv = np.corrcoef(radii, curvs)[0, 1]

print(f"\n面积 vs 最大曲率: 相关系数 = {corr_area_curv:.3f}")
print(f"面积 vs 尖锐角数量: 相关系数 = {corr_area_angles:.3f}")
print(f"半径 vs 最大曲率: 相关系数 = {corr_radius_curv:.3f}")

if corr_area_curv < -0.3:
    print("\n✓ 发现负相关：轮廓越大，曲率越小（毛刺越少）")
elif corr_area_curv > 0.3:
    print("\n✗ 发现正相关：轮廓越大，曲率越大（毛刺越多）")
else:
    print("\n○ 相关性较弱：轮廓大小与毛刺程度关系不明显")

# 分析点密度
print('\n' + '='*100)
print('点密度分析（关键！）')
print('='*100)

print(f"\n{'轮廓':<8} {'周长':<12} {'点数':<10} {'点密度':<15} {'最大曲率':<12} {'观察':<30}")
print('-'*100)

num_points = 128  # 所有轮廓都是128个点

for data in sorted(contour_data, key=lambda x: x['perimeter']):
    cid = data['id']
    perimeter = data['perimeter']
    density = perimeter / num_points  # 每个点对应的周长
    curv = data['curv_max']

    if density < 2:
        obs = "点密集（小轮廓）"
    elif density < 4:
        obs = "点适中"
    else:
        obs = "点稀疏（大轮廓）"

    print(f"{cid:<8} {perimeter:<12.1f} {num_points:<10} {density:<15.2f} {curv:<12.2f} {obs:<30}")

# 分析点密度与曲率的关系
densities = [d['perimeter'] / 128 for d in contour_data]
corr_density_curv = np.corrcoef(densities, curvs)[0, 1]

print(f"\n点密度 vs 最大曲率: 相关系数 = {corr_density_curv:.3f}")

if corr_density_curv < -0.3:
    print("\n✓✓✓ 关键发现：点越密集（小轮廓），曲率越大（毛刺越严重）！")
    print("这支持了你的假设：小轮廓点太密集，容易产生毛刺")
elif corr_density_curv > 0.3:
    print("\n✓✓✓ 关键发现：点越稀疏（大轮廓），曲率越大（毛刺越严重）！")
    print("这说明大轮廓点太稀疏，无法准确表示形状")
else:
    print("\n○ 点密度与曲率关系不明显")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. 面积 vs 曲率
ax = axes[0, 0]
ax.scatter(areas, curvs, s=100, alpha=0.7, c=curvs, cmap='hot')
for i, data in enumerate(contour_data):
    ax.annotate(f"C{data['id']}", (areas[i], curvs[i]), fontsize=10)
ax.set_xlabel('Area (pixels²)')
ax.set_ylabel('Max Curvature')
ax.set_title(f'Area vs Curvature (corr={corr_area_curv:.3f})')
ax.grid(True, alpha=0.3)

# 2. 点密度 vs 曲率
ax = axes[0, 1]
ax.scatter(densities, curvs, s=100, alpha=0.7, c=curvs, cmap='hot')
for i, data in enumerate(contour_data):
    ax.annotate(f"C{data['id']}", (densities[i], curvs[i]), fontsize=10)
ax.set_xlabel('Point Density (perimeter/128)')
ax.set_ylabel('Max Curvature')
ax.set_title(f'Point Density vs Curvature (corr={corr_density_curv:.3f})')
ax.grid(True, alpha=0.3)

# 3. 周长 vs 尖锐角
ax = axes[1, 0]
perimeters = [d['perimeter'] for d in contour_data]
ax.scatter(perimeters, angles, s=100, alpha=0.7, c=curvs, cmap='hot')
for i, data in enumerate(contour_data):
    ax.annotate(f"C{data['id']}", (perimeters[i], angles[i]), fontsize=10)
ax.set_xlabel('Perimeter (pixels)')
ax.set_ylabel('Number of Sharp Angles')
ax.set_title('Perimeter vs Sharp Angles')
ax.grid(True, alpha=0.3)

# 4. 轮廓大小分布
ax = axes[1, 1]
sorted_data = sorted(contour_data, key=lambda x: x['area'])
x = range(len(sorted_data))
areas_sorted = [d['area'] for d in sorted_data]
curvs_sorted = [d['curv_max'] for d in sorted_data]
labels = [f"C{d['id']}" for d in sorted_data]

ax2 = ax.twinx()
bars = ax.bar(x, areas_sorted, alpha=0.5, label='Area', color='blue')
line = ax2.plot(x, curvs_sorted, 'ro-', linewidth=2, markersize=8, label='Curvature')

ax.set_xlabel('Contour (sorted by area)')
ax.set_ylabel('Area (pixels²)', color='blue')
ax2.set_ylabel('Max Curvature', color='red')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title('Contour Size vs Curvature')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visual/burr_v3_4_full/size_vs_burr_analysis.png', dpi=150, bbox_inches='tight')
print("\n可视化已保存: visual/burr_v3_4_full/size_vs_burr_analysis.png")

# 总结
print('\n' + '='*100)
print('核心结论')
print('='*100)

print("\n基于数据分析：")
print(f"1. 面积与曲率相关系数: {corr_area_curv:.3f}")
print(f"2. 点密度与曲率相关系数: {corr_density_curv:.3f}")

print("\n具体观察：")
for data in sorted(contour_data, key=lambda x: x['area']):
    density = data['perimeter'] / 128
    print(f"  轮廓{data['id']}: 面积={data['area']:.0f}, 点密度={density:.2f}, 曲率={data['curv_max']:.2f}")

print("\n你的假设验证：")
if corr_density_curv < -0.3:
    print("✓ 支持假设：小轮廓（点密集）确实更容易产生毛刺")
    print("  原因可能是：点太密集，相邻点距离太小，微小的预测误差就会导致大的曲率变化")
elif corr_density_curv > 0.3:
    print("✗ 不支持假设：大轮廓（点稀疏）反而更容易产生毛刺")
    print("  原因可能是：点太稀疏，无法准确表示复杂的形状")
else:
    print("○ 数据不明确：需要更多样本或更深入的分析")
    print("  可能的原因：")
    print("  - 样本数量太少（只有6个轮廓）")
    print("  - 其他因素（如器官类型、形状复杂度）的影响更大")
