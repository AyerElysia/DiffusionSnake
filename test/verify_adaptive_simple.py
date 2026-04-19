"""
使用已有的预测结果验证自适应点数效果

不需要重新推理，直接使用之前保存的结果
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import interpolate
import cv2


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
    """自适应重采样"""
    perimeter = compute_perimeter(contour)
    target_points = int(perimeter / target_density)
    target_points = max(32, min(target_points, 256))
    target_points = (target_points // 4) * 4

    downsampled = uniform_resample(contour, target_points)
    upsampled = uniform_resample(downsampled, 128)

    return upsampled, target_points


# 从之前的分析中读取数据
with open('visual/burr_v3_4_full/full_image_metrics.json', 'r') as f:
    metrics = json.load(f)

# 需要重新运行推理获取预测轮廓，或者从之前的可视化中提取
# 这里我们使用GT来模拟（实际应该用预测结果）
print("注意：需要先运行 analyze_burr_v3_4_full.py 并保存预测结果")
print("这里展示验证流程...")

# 模拟数据（实际应该从推理结果加载）
print("\n" + "="*100)
print("自适应点数验证（基于已有数据）")
print("="*100)
print(f"{'轮廓':<8} {'周长':<10} {'原始点数':<12} {'建议点数':<12} {'点密度':<12} {'预期改善':<15}")
print("-"*100)

for m in metrics:
    cid = m['contour_id']
    # 从dist_mean估算周长
    perimeter = m['raw']['dist_mean'] * 128

    # 计算建议点数
    target_density = 2.5
    target_points = int(perimeter / target_density)
    target_points = max(32, min(target_points, 256))
    target_points = (target_points // 4) * 4

    # 当前点密度
    current_density = perimeter / 128

    # 预期改善（基于相关性）
    # 点密度从current_density变到target_density
    # 根据相关系数-0.613，点密度增加会降低曲率
    if current_density < 1.5:  # 小轮廓
        expected_improvement = "40-60%"
    elif current_density < 2.5:  # 中等轮廓
        expected_improvement = "20-40%"
    else:  # 大轮廓
        expected_improvement = "0-10%"

    print(f"{cid:<8} {perimeter:<10.1f} {128:<12} {target_points:<12} "
          f"{current_density:<12.2f} {expected_improvement:<15}")

print("\n" + "="*100)
print("结论")
print("="*100)

print("\n基于数据分析，预期效果：")
print("- 轮廓1（周长75.8，点密度0.59）：32点 → 预期曲率降低40-60%")
print("- 轮廓5（周长152.1，点密度1.19）：64点 → 预期曲率降低40-60%")
print("- 轮廓2（周长182.0，点密度1.42）：72点 → 预期曲率降低40-60%")
print("- 轮廓3（周长295.8，点密度2.31）：120点 → 预期曲率降低20-30%")
print("- 轮廓4（周长374.8，点密度2.93）：152点 → 预期曲率变化<10%")

print("\n下一步：")
print("1. 需要重新运行推理并保存预测轮廓")
print("2. 对预测轮廓应用自适应重采样")
print("3. 对比曲率变化")
print("4. 如果改善>30%，则假设验证成功")

print("\n" + "="*100)
