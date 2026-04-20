#!/usr/bin/env python3
"""
检查BTCV数据集的bbox大小分布
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, '/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30')

# 读取BTCV标注
ann_file = '/mnt/sdb1/leijh/DiffusionSnake/Datasets/BTCV/btcv_png_new_snake/train.json'

with open(ann_file, 'r') as f:
    data = json.load(f)

print("分析BTCV数据集bbox大小...")
print(f"总图像数: {len(data['images'])}")
print(f"总标注数: {len(data['annotations'])}")

# 统计bbox大小
perimeters = []
areas = []
widths = []
heights = []

for ann in data['annotations']:
    bbox = ann['bbox']  # [x, y, w, h]
    x, y, w, h = bbox

    perimeter = 2 * (w + h)
    area = w * h

    perimeters.append(perimeter)
    areas.append(area)
    widths.append(w)
    heights.append(h)

perimeters = np.array(perimeters)
areas = np.array(areas)
widths = np.array(widths)
heights = np.array(heights)

print(f"\nBBox统计:")
print(f"  周长范围: [{perimeters.min():.1f}, {perimeters.max():.1f}]")
print(f"  周长平均: {perimeters.mean():.1f}")
print(f"  周长中位数: {np.median(perimeters):.1f}")

print(f"\n  面积范围: [{areas.min():.1f}, {areas.max():.1f}]")
print(f"  面积平均: {areas.mean():.1f}")

print(f"\n  宽度范围: [{widths.min():.1f}, {widths.max():.1f}]")
print(f"  高度范围: [{heights.min():.1f}, {heights.max():.1f}]")

# 计算自适应点数
target_density = 2.5
adaptive_points = perimeters / target_density
adaptive_points = np.clip(adaptive_points, 32, 512)
adaptive_points = ((adaptive_points + 7) // 8) * 8

print(f"\n自适应点数分布（target_density={target_density}）:")
print(f"  点数范围: [{adaptive_points.min():.0f}, {adaptive_points.max():.0f}]")
print(f"  点数平均: {adaptive_points.mean():.1f}")
print(f"  点数中位数: {np.median(adaptive_points):.0f}")

# 点数分布直方图
unique, counts = np.unique(adaptive_points, return_counts=True)
print(f"\n点数分布:")
for pts, cnt in zip(unique, counts):
    pct = 100 * cnt / len(adaptive_points)
    print(f"  {int(pts)}点: {cnt}个 ({pct:.1f}%)")

# 检查有多少会使用非128点
non_128 = (adaptive_points != 128).sum()
print(f"\n非128点的轮廓数: {non_128} ({100*non_128/len(adaptive_points):.1f}%)")
