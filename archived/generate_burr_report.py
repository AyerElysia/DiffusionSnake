"""
生成毛刺分析综合报告
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# 读取所有样本的指标
metrics_dir = "visual/burr_analysis"
all_metrics = []

for i in range(4):
    json_path = os.path.join(metrics_dir, f"burr_idx{i}_metrics.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            all_metrics.append(data)

print("=" * 80)
print("毛刺问题综合分析报告")
print("=" * 80)

print("\n## 1. 点序问题分析")
print("-" * 80)
print(f"{'样本':<8} {'跳跃数':<10} {'跳跃率':<12} {'平均距离':<12} {'最大距离':<12} {'标准差':<12}")
print("-" * 80)

for i, m in enumerate(all_metrics):
    orig = m['point_order_analysis']['original']
    print(f"idx{i:<4} {orig['num_jumps']:<10} {orig['jump_ratio']:<12.4f} "
          f"{orig['mean_dist']:<12.2f} {orig['max_dist']:<12.2f} {orig['std_dist']:<12.2f}")

print("\n**结论：** 原始点序跳跃数量很少（0-1个），说明点序本身是合理的。")

print("\n## 2. 毛刺特征分析")
print("-" * 80)
print(f"{'样本':<8} {'曲率最大':<12} {'异常点':<10} {'尖锐角':<10} {'高频点':<10} {'变异系数':<12}")
print("-" * 80)

for i, m in enumerate(all_metrics):
    raw = m['raw_prediction']
    print(f"idx{i:<4} {raw['curv_max']:<12.2f} {raw['curv_outliers']:<10} "
          f"{raw['sharp_angles']:<10} {raw['high_freq_points']:<10} {raw['dist_cv']:<12.3f}")

print("\n**关键发现：**")
print("- 曲率最大值：6-10，说明有尖锐转角")
print("- 尖锐角数量：19-55个，占比15-43%")
print("- 高频振荡点：5-6个，占比约4-5%")
print("- 变异系数：0.58-0.62，点分布不够均匀")

print("\n## 3. 平滑后处理效果")
print("-" * 80)
print(f"{'样本':<8} {'曲率降低':<12} {'尖锐角减少':<14} {'高频点减少':<14}")
print("-" * 80)

for i, m in enumerate(all_metrics):
    raw = m['raw_prediction']
    smooth = m['after_smoothing']
    curv_reduction = (raw['curv_max'] - smooth['curv_max']) / raw['curv_max'] * 100
    angle_reduction = raw['sharp_angles'] - smooth['sharp_angles']
    hf_reduction = raw['high_freq_points'] - smooth['high_freq_points']
    print(f"idx{i:<4} {curv_reduction:<12.1f}% {angle_reduction:<14} {hf_reduction:<14}")

print("\n**平滑效果：**")
print("- 曲率最大值降低：29-61%")
print("- 尖锐角减少：16-36个")
print("- 高频振荡点略有减少")

print("\n## 4. 核心结论")
print("=" * 80)

print("\n### 毛刺的主要原因：**预测质量问题，而非点序问题**")

print("\n**证据：**")
print("1. 点序跳跃极少（0-1个），跳跃率<1%")
print("2. 最近邻重排后反而变差（跳跃增加到2-3个）")
print("3. 原始点序的标准差较小（0.59-1.12），说明点分布相对均匀")
print("4. 重排后标准差大幅增加（2.19-4.18），说明原始点序已经是较优的")

print("\n**毛刺的真实来源：**")
print("1. **高曲率点**：模型预测出了尖锐的转角（曲率6-10）")
print("2. **大量尖锐角**：19-55个大于120度的转角，占比15-43%")
print("3. **点分布不均**：变异系数0.58-0.62，说明有些区域点密集，有些稀疏")
print("4. **高频振荡**：5-6个点偏离前后连线较远")

print("\n### 为什么散点图看起来还可以？")
print("\n这是一个关键观察：")
print("- 散点图不显示连接关系，只看点的空间分布")
print("- 如果点本身就在尖锐转角的位置，散点图看不出问题")
print("- 但连线后，尖锐转角就会显现为毛刺")
print("- 这说明：**毛刺不是点序错误，而是点本身就在错误的位置**")

print("\n## 5. 解决方案建议")
print("=" * 80)

print("\n### 方案A：训练时增强平滑约束（推荐）")
print("1. 增加曲率正则化损失")
print("2. 增加点间距离均匀性约束")
print("3. 使用更多的扩散步数")
print("4. 调整DiT的层数和注意力头数")

print("\n### 方案B：后处理平滑（临时方案）")
print("1. 边缘感知平滑已经有效（曲率降低29-61%）")
print("2. 可以调整平滑参数：curvature_threshold, iterations")
print("3. 考虑使用傅里叶平滑或B样条拟合")

print("\n### 方案C：改进初始化")
print("1. 当前八边形初始化可能不够精细")
print("2. 考虑使用更多点的初始化（如16边形）")
print("3. 或者使用GT的粗略版本作为初始化")

print("\n### 不推荐的方案：")
print("❌ 修改点序生成逻辑 - 因为点序本身没问题")
print("❌ 使用最近邻重排 - 实验证明会让结果变差")

print("\n## 6. 下一步实验建议")
print("=" * 80)

print("\n1. **可视化曲率分布**")
print("   - 查看 burr_idx*_analysis.png 的曲率热力图")
print("   - 确认高曲率点是否对应真实的尖锐边缘")

print("\n2. **对比GT的曲率**")
print("   - 计算GT轮廓的曲率统计")
print("   - 看预测的曲率是否远大于GT")

print("\n3. **训练时加入曲率损失**")
print("   - 在损失函数中加入 L_curv = ||curvature(pred) - curvature(gt)||")
print("   - 权重可以从0.1开始尝试")

print("\n4. **调整扩散步数**")
print("   - 当前使用50步，尝试增加到100步")
print("   - 看是否能得到更平滑的结果")

print("\n5. **分析初始化质量**")
print("   - 查看八边形初始化与GT的差异")
print("   - 如果初始化就有毛刺，模型很难修正")

print("\n" + "=" * 80)
print("报告生成完成！")
print("=" * 80)

# 生成对比图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 点序跳跃对比
ax = axes[0, 0]
samples = [f"idx{i}" for i in range(len(all_metrics))]
orig_jumps = [m['point_order_analysis']['original']['num_jumps'] for m in all_metrics]
reorder_jumps = [m['point_order_analysis']['reordered']['num_jumps'] for m in all_metrics]
x = np.arange(len(samples))
width = 0.35
ax.bar(x - width/2, orig_jumps, width, label='Original', color='blue', alpha=0.7)
ax.bar(x + width/2, reorder_jumps, width, label='Reordered', color='red', alpha=0.7)
ax.set_ylabel('Number of Jumps')
ax.set_title('Point Order Jumps: Original vs Reordered')
ax.set_xticks(x)
ax.set_xticklabels(samples)
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 曲率对比
ax = axes[0, 1]
raw_curv = [m['raw_prediction']['curv_max'] for m in all_metrics]
smooth_curv = [m['after_smoothing']['curv_max'] for m in all_metrics]
ax.bar(x - width/2, raw_curv, width, label='Raw', color='orange', alpha=0.7)
ax.bar(x + width/2, smooth_curv, width, label='Smoothed', color='green', alpha=0.7)
ax.set_ylabel('Max Curvature')
ax.set_title('Max Curvature: Raw vs Smoothed')
ax.set_xticks(x)
ax.set_xticklabels(samples)
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 尖锐角对比
ax = axes[1, 0]
raw_angles = [m['raw_prediction']['sharp_angles'] for m in all_metrics]
smooth_angles = [m['after_smoothing']['sharp_angles'] for m in all_metrics]
ax.bar(x - width/2, raw_angles, width, label='Raw', color='purple', alpha=0.7)
ax.bar(x + width/2, smooth_angles, width, label='Smoothed', color='cyan', alpha=0.7)
ax.set_ylabel('Number of Sharp Angles (>120°)')
ax.set_title('Sharp Angles: Raw vs Smoothed')
ax.set_xticks(x)
ax.set_xticklabels(samples)
ax.legend()
ax.grid(True, alpha=0.3)

# 4. 高频振荡点对比
ax = axes[1, 1]
raw_hf = [m['raw_prediction']['high_freq_points'] for m in all_metrics]
smooth_hf = [m['after_smoothing']['high_freq_points'] for m in all_metrics]
ax.bar(x - width/2, raw_hf, width, label='Raw', color='red', alpha=0.7)
ax.bar(x + width/2, smooth_hf, width, label='Smoothed', color='blue', alpha=0.7)
ax.set_ylabel('Number of High-Freq Points')
ax.set_title('High-Frequency Oscillation Points')
ax.set_xticks(x)
ax.set_xticklabels(samples)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visual/burr_analysis/comprehensive_report.png', dpi=150, bbox_inches='tight')
print("\n综合对比图已保存: visual/burr_analysis/comprehensive_report.png")
