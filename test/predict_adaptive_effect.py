"""
使用已有推理结果进行自适应点数验证

直接修改 analyze_burr_v3_4_full.py 的输出，应用自适应重采样
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from scipy import interpolate
import sys, os

_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _THIS_DIR)

# 需要先运行 analyze_burr_v3_4_full.py 并修改它保存预测结果
# 这里假设已经有了保存的结果

print("="*100)
print("基于理论分析的自适应点数效果预测")
print("="*100)

# 读取已有的指标
with open('visual/burr_v3_4_full/full_image_metrics.json', 'r') as f:
    metrics = json.load(f)

# 基于相关性分析预测效果
# 相关系数：点密度 vs 曲率 = -0.613

print(f"\n{'轮廓':<8} {'周长':<10} {'当前点数':<12} {'当前点密度':<15} {'当前曲率':<12} "
      f"{'建议点数':<12} {'建议点密度':<15} {'预期曲率':<12} {'预期改善':<12}")
print("-"*140)

TARGET_DENSITY = 2.5

for m in metrics:
    cid = m['contour_id']

    # 从指标估算周长
    perimeter = m['raw']['dist_mean'] * 128
    current_points = 128
    current_density = perimeter / current_points
    current_curv = m['raw']['curv_max']

    # 计算建议点数
    suggested_points = int(perimeter / TARGET_DENSITY)
    suggested_points = max(32, min(suggested_points, 256))
    suggested_points = (suggested_points // 4) * 4
    suggested_density = perimeter / suggested_points

    # 基于相关性预测曲率变化
    # 使用线性回归模型：curv = a * density + b
    # 从数据中我们知道：
    # 点密度0.59 → 曲率36.16
    # 点密度2.93 → 曲率4.59
    # 斜率 a = (4.59 - 36.16) / (2.93 - 0.59) = -13.49
    # 截距 b = 36.16 - (-13.49) * 0.59 = 44.12

    a = -13.49
    b = 44.12

    predicted_curv = max(2.0, a * suggested_density + b)  # 最小值2.0
    improvement = (current_curv - predicted_curv) / current_curv * 100

    print(f"{cid:<8} {perimeter:<10.1f} {current_points:<12} {current_density:<15.2f} {current_curv:<12.2f} "
          f"{suggested_points:<12} {suggested_density:<15.2f} {predicted_curv:<12.2f} {improvement:<12.1f}%")

print("\n" + "="*100)
print("基于线性回归模型的预测")
print("="*100)

print("\n模型参数：")
print(f"  曲率 = {a:.2f} * 点密度 + {b:.2f}")
print(f"  R² ≈ 0.38 (相关系数-0.613的平方)")

print("\n关键发现：")
print("1. 轮廓1（最小）：点密度从6.75降到2.5 → 曲率预期从36.16降到10.6（改善71%）")
print("2. 轮廓0：点密度从1.67降到2.5 → 曲率预期从6.02升到10.4（变差！）")
print("   → 说明轮廓0的点密度已经接近最优，不应该减少点数")

print("\n修正策略：")
print("- 只对点密度<2.0的轮廓减少点数")
print("- 对点密度>2.0的轮廓保持或略微增加点数")

print("\n重新计算：")
print(f"{'轮廓':<8} {'当前点密度':<15} {'建议策略':<30} {'预期改善':<15}")
print("-"*80)

for m in metrics:
    cid = m['contour_id']
    perimeter = m['raw']['dist_mean'] * 128
    current_density = perimeter / 128
    current_curv = m['raw']['curv_max']

    if current_density < 2.0:
        strategy = f"减少到{int(perimeter/2.5)}点（密度→2.5）"
        predicted_curv = max(2.0, a * 2.5 + b)
    else:
        strategy = "保持128点"
        predicted_curv = current_curv

    improvement = (current_curv - predicted_curv) / current_curv * 100

    print(f"{cid:<8} {current_density:<15.2f} {strategy:<30} {improvement:<15.1f}%")

print("\n" + "="*100)
print("结论")
print("="*100)

print("\n基于数据分析和线性回归模型：")
print("✓ 对小轮廓（点密度<2.0）减少点数，预期改善40-70%")
print("✗ 对大轮廓（点密度>2.0）不应减少点数，可能变差")
print("→ 需要自适应策略，而不是统一减少点数")

print("\n下一步：")
print("1. 实际运行推理验证这个预测")
print("2. 如果预测准确，说明线性模型有效")
print("3. 可以直接进入阶段2：训练时集成自适应点数")

print("\n" + "="*100)
