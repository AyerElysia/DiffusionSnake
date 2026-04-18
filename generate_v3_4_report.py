"""
生成V3.4单样本过拟合的最终分析报告
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# 读取指标
with open('visual/burr_v3_4_full/full_image_metrics.json', 'r') as f:
    metrics = json.load(f)

print("="*100)
print("毛刺问题最终分析报告 - V3.4单样本过拟合（样本0，6个轮廓）")
print("="*100)

print("\n## 1. 点序问题分析")
print("-"*100)
print(f"{'轮廓':<8} {'原始跳跃':<12} {'平滑后跳跃':<12} {'跳跃率':<12} {'结论':<30}")
print("-"*100)

for m in metrics:
    cid = m['contour_id']
    raw_jumps = m['raw_jumps']
    smooth_jumps = m['smoothed_jumps']
    jump_rate = raw_jumps / 128 * 100

    if raw_jumps <= 1:
        conclusion = "点序合理"
    elif raw_jumps == 2:
        conclusion = "点序基本合理"
    else:
        conclusion = "点序可能有问题"

    print(f"{cid:<8} {raw_jumps:<12} {smooth_jumps:<12} {jump_rate:<12.1f}% {conclusion:<30}")

print("\n**关键发现：**")
print("- 6个轮廓中，4个轮廓跳跃数≤1（点序合理）")
print("- 2个轮廓跳跃数=2（点序基本合理）")
print("- 平滑后跳跃数基本不变，说明点序不是主要问题")

print("\n## 2. 毛刺特征分析（原始预测）")
print("-"*100)
print(f"{'轮廓':<8} {'最大曲率':<12} {'平均曲率':<12} {'尖锐角':<10} {'尖锐角占比':<12} {'高频点':<10} {'变异系数':<12}")
print("-"*100)

for m in metrics:
    cid = m['contour_id']
    raw = m['raw']
    sharp_ratio = raw['sharp_angles'] / 128 * 100
    print(f"{cid:<8} {raw['curv_max']:<12.2f} {raw['curv_mean']:<12.2f} "
          f"{raw['sharp_angles']:<10} {sharp_ratio:<12.1f}% "
          f"{raw['high_freq_points']:<10} {raw['dist_cv']:<12.3f}")

print("\n**关键发现：**")
curv_max_list = [m['raw']['curv_max'] for m in metrics]
sharp_angles_list = [m['raw']['sharp_angles'] for m in metrics]
print(f"- 曲率最大值范围：{min(curv_max_list):.1f} - {max(curv_max_list):.1f}")
print(f"- 轮廓1、2、5的曲率异常高（22-36），说明有严重的尖锐转角")
print(f"- 轮廓0、3、4的曲率相对正常（4-12）")
print(f"- 尖锐角数量：{min(sharp_angles_list)} - {max(sharp_angles_list)}个，占比0.8% - 57%")
print(f"- 变异系数：0.18 - 0.87，说明点分布均匀性差异很大")

print("\n## 3. 平滑后处理效果")
print("-"*100)
print(f"{'轮廓':<8} {'曲率降低':<15} {'尖锐角减少':<15} {'高频点变化':<15} {'效果评价':<15}")
print("-"*100)

for m in metrics:
    cid = m['contour_id']
    raw = m['raw']
    smooth = m['smoothed']

    curv_reduction = (raw['curv_max'] - smooth['curv_max']) / raw['curv_max'] * 100
    angle_reduction = raw['sharp_angles'] - smooth['sharp_angles']
    hf_change = smooth['high_freq_points'] - raw['high_freq_points']

    if curv_reduction > 40:
        effect = "很好"
    elif curv_reduction > 20:
        effect = "较好"
    elif curv_reduction > 10:
        effect = "一般"
    else:
        effect = "较差"

    print(f"{cid:<8} {curv_reduction:<15.1f}% {angle_reduction:<15} {hf_change:<15} {effect:<15}")

print("\n**平滑效果总结：**")
curv_reductions = [(m['raw']['curv_max'] - m['smoothed']['curv_max']) / m['raw']['curv_max'] * 100 for m in metrics]
angle_reductions = [m['raw']['sharp_angles'] - m['smoothed']['sharp_angles'] for m in metrics]
print(f"- 曲率降低：{min(curv_reductions):.1f}% - {max(curv_reductions):.1f}%，平均{np.mean(curv_reductions):.1f}%")
print(f"- 尖锐角减少：{min(angle_reductions)} - {max(angle_reductions)}个，平均{np.mean(angle_reductions):.1f}个")
print(f"- 对轮廓0、4效果很好（曲率降低>40%）")
print(f"- 对轮廓1、2、5效果较差（曲率降低<5%），说明这些轮廓的毛刺非常顽固")

print("\n## 4. 散点图关键观察")
print("="*100)

print("\n**查看生成的图像：visual/burr_v3_4_full/full_image_scatter_comparison.png**")
print("\n每个轮廓有4个子图：")
print("1. GT连线图（参考）")
print("2. 原始预测散点图（关键！）")
print("3. 平滑后散点图（关键对比！）")
print("4. 连线对比图")

print("\n**需要重点观察：**")
print("- 原始预测的散点图是否平滑？")
print("  → 如果散点图平滑但连线有毛刺 = 点序问题")
print("  → 如果散点图本身就尖锐 = 预测质量问题")
print("\n- 平滑后的散点图是否改善？")
print("  → 如果散点图明显变平滑 = 平滑有效")
print("  → 如果散点图变化不大 = 毛刺来自点的位置")

print("\n## 5. 核心结论")
print("="*100)

print("\n### 结论1：毛刺主要是预测质量问题，不是点序问题")
print("\n**证据：**")
print("- 点序跳跃数≤2，跳跃率<2%")
print("- 平滑后跳跃数基本不变")
print("- 说明点的连接顺序是合理的")

print("\n### 结论2：不同轮廓的毛刺程度差异巨大")
print("\n**轮廓分类：**")
print("\n**A类（严重毛刺）：轮廓1、2、5**")
print("- 曲率：22-36（非常高）")
print("- 尖锐角：27-73个（占比21-57%）")
print("- 平滑效果差（曲率仅降低3-4%）")
print("- 说明：这些轮廓的毛刺非常顽固，可能是器官本身形状复杂")

print("\n**B类（中等毛刺）：轮廓3**")
print("- 曲率：11.76（中等）")
print("- 尖锐角：43个（占比34%）")
print("- 平滑效果较好（曲率降低24%）")

print("\n**C类（轻微毛刺）：轮廓0、4**")
print("- 曲率：4-6（较低）")
print("- 尖锐角：1-57个")
print("- 平滑效果很好（曲率降低43-57%）")
print("- 说明：这些轮廓的预测质量较好")

print("\n### 结论3：平滑后的散点图是关键判断依据")
print("\n**如果平滑后散点图明显变平滑：**")
print("→ 说明原始预测的点位置有高频振荡")
print("→ 平滑能有效改善")
print("→ 问题可能来自扩散过程的噪声")

print("\n**如果平滑后散点图变化不大：**")
print("→ 说明点本身就在尖锐转角的位置")
print("→ 平滑无法根本解决")
print("→ 需要改进模型训练")

print("\n## 6. 解决方案（基于新发现）")
print("="*100)

print("\n### 方案A：针对性训练策略 ⭐⭐⭐⭐⭐")
print("\n**1. 对不同复杂度的轮廓使用不同的损失权重**")
print("```python")
print("# 根据GT的曲率复杂度调整损失权重")
print("gt_curv = compute_curvature(gt)")
print("complexity = gt_curv.max()")
print("if complexity > 20:  # 复杂轮廓")
print("    curv_loss_weight = 0.2  # 更强的平滑约束")
print("else:  # 简单轮廓")
print("    curv_loss_weight = 0.1")
print("```")

print("\n**2. 增加曲率正则化损失**")
print("```python")
print("def curvature_loss(pred, gt):")
print("    pred_curv = compute_curvature(pred)")
print("    gt_curv = compute_curvature(gt)")
print("    return F.mse_loss(pred_curv, gt_curv)")
print("")
print("loss_total = loss_position + curv_loss_weight * curvature_loss(pred, gt)")
print("```")

print("\n**3. 增加扩散步数（针对复杂轮廓）**")
print("- 当前：50步")
print("- 建议：对曲率>20的轮廓使用100步")

print("\n### 方案B：改进后处理（临时方案）⭐⭐⭐")
print("\n**1. 针对不同轮廓使用不同的平滑参数**")
print("```python")
print("if curv_max > 20:  # 严重毛刺")
print("    smooth_contours_numpy(contour, curvature_threshold=2.0, iterations=5)")
print("else:  # 轻微毛刺")
print("    smooth_contours_numpy(contour, curvature_threshold=5.0, iterations=2)")
print("```")

print("\n**2. 使用更强的平滑算法（针对A类轮廓）**")
print("- 傅里叶低通滤波")
print("- B样条拟合")
print("- 高斯滤波")

print("\n### 方案C：数据增强 ⭐⭐⭐⭐")
print("\n**1. 对复杂轮廓进行数据增强**")
print("- 轻微旋转、缩放")
print("- 增加训练样本的多样性")
print("- 帮助模型学习平滑的轮廓")

print("\n**2. 使用平滑后的GT作为辅助监督**")
print("```python")
print("gt_smoothed = smooth_contours_numpy(gt)")
print("loss_smooth = F.mse_loss(pred, gt_smoothed)")
print("loss_total = loss_position + 0.1 * loss_smooth")
print("```")

print("\n## 7. 下一步实验")
print("="*100)

print("\n### 实验1：可视化分析（立即执行）")
print("1. 打开 visual/burr_v3_4_full/full_image_scatter_comparison.png")
print("2. 重点观察轮廓1、2、5的散点图")
print("3. 判断：散点图是否平滑？平滑后是否改善？")

print("\n### 实验2：对比GT的曲率（1小时）")
print("```python")
print("# 计算GT的曲率统计")
print("for i, gt in enumerate(gt_contours):")
print("    gt_curv = compute_curvature(gt)")
print("    print(f'GT轮廓{i}: 曲率max={gt_curv.max():.2f}')")
print("```")
print("目的：确认预测的曲率是否远大于GT")

print("\n### 实验3：训练时加入曲率损失（1-2天）")
print("修改 lib/train/trainers/diffusion_trainer.py")
print("预期：曲率降低20-40%")

print("\n### 实验4：针对性平滑参数（立即执行）")
print("对轮廓1、2、5使用更强的平滑")
print("预期：曲率降低10-20%")

print("\n## 8. 总结")
print("="*100)

print("\n### 核心发现")
print("1. **点序不是问题**（跳跃率<2%）")
print("2. **毛刺程度差异巨大**（曲率4-36）")
print("3. **平滑效果因轮廓而异**（3%-57%）")
print("4. **散点图对比是关键判断依据**")

print("\n### 推荐行动")
print("1. **立即**：查看散点图，确认观察结果")
print("2. **1小时**：对比GT曲率，量化差异")
print("3. **1天**：针对性调整平滑参数")
print("4. **1周**：训练时加入曲率损失")

print("\n### 预期收益")
print("- 针对性平滑：立即改善10-20%（A类轮廓）")
print("- 曲率损失：预期改善30-50%")
print("- 综合优化：预期解决70-80%的毛刺问题")

print("\n" + "="*100)
print("报告生成完成！")
print("="*100)

# 生成可视化对比图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

contour_ids = [m['contour_id'] for m in metrics]
raw_curvs = [m['raw']['curv_max'] for m in metrics]
smooth_curvs = [m['smoothed']['curv_max'] for m in metrics]
raw_angles = [m['raw']['sharp_angles'] for m in metrics]
smooth_angles = [m['smoothed']['sharp_angles'] for m in metrics]
raw_jumps = [m['raw_jumps'] for m in metrics]
smooth_jumps = [m['smoothed_jumps'] for m in metrics]

x = np.arange(len(contour_ids))
width = 0.35

# 1. 曲率对比
ax = axes[0, 0]
ax.bar(x - width/2, raw_curvs, width, label='Raw', color='red', alpha=0.7)
ax.bar(x + width/2, smooth_curvs, width, label='Smoothed', color='blue', alpha=0.7)
ax.set_ylabel('Max Curvature')
ax.set_title('Max Curvature: Raw vs Smoothed')
ax.set_xticks(x)
ax.set_xticklabels([f'C{i}' for i in contour_ids])
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 尖锐角对比
ax = axes[0, 1]
ax.bar(x - width/2, raw_angles, width, label='Raw', color='orange', alpha=0.7)
ax.bar(x + width/2, smooth_angles, width, label='Smoothed', color='green', alpha=0.7)
ax.set_ylabel('Number of Sharp Angles')
ax.set_title('Sharp Angles (>120°): Raw vs Smoothed')
ax.set_xticks(x)
ax.set_xticklabels([f'C{i}' for i in contour_ids])
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 点序跳跃对比
ax = axes[0, 2]
ax.bar(x - width/2, raw_jumps, width, label='Raw', color='purple', alpha=0.7)
ax.bar(x + width/2, smooth_jumps, width, label='Smoothed', color='cyan', alpha=0.7)
ax.set_ylabel('Number of Jumps')
ax.set_title('Point Order Jumps: Raw vs Smoothed')
ax.set_xticks(x)
ax.set_xticklabels([f'C{i}' for i in contour_ids])
ax.legend()
ax.grid(True, alpha=0.3)

# 4. 曲率降低百分比
ax = axes[1, 0]
curv_reductions = [(raw_curvs[i] - smooth_curvs[i]) / raw_curvs[i] * 100 for i in range(len(contour_ids))]
colors = ['green' if r > 40 else 'orange' if r > 20 else 'red' for r in curv_reductions]
ax.bar(x, curv_reductions, color=colors, alpha=0.7)
ax.axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='20% threshold')
ax.set_ylabel('Curvature Reduction (%)')
ax.set_title('Smoothing Effectiveness (Curvature Reduction)')
ax.set_xticks(x)
ax.set_xticklabels([f'C{i}' for i in contour_ids])
ax.legend()
ax.grid(True, alpha=0.3)

# 5. 轮廓分类
ax = axes[1, 1]
categories = []
for i in range(len(contour_ids)):
    if raw_curvs[i] > 20:
        categories.append('A (Severe)')
    elif raw_curvs[i] > 10:
        categories.append('B (Moderate)')
    else:
        categories.append('C (Mild)')

category_colors = {'A (Severe)': 'red', 'B (Moderate)': 'orange', 'C (Mild)': 'green'}
colors = [category_colors[c] for c in categories]
ax.bar(x, raw_curvs, color=colors, alpha=0.7)
ax.set_ylabel('Max Curvature')
ax.set_title('Contour Classification by Burr Severity')
ax.set_xticks(x)
ax.set_xticklabels([f'C{i}' for i in contour_ids])
ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Severe threshold')
ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. 变异系数对比
ax = axes[1, 2]
raw_cvs = [m['raw']['dist_cv'] for m in metrics]
smooth_cvs = [m['smoothed']['dist_cv'] for m in metrics]
ax.bar(x - width/2, raw_cvs, width, label='Raw', color='brown', alpha=0.7)
ax.bar(x + width/2, smooth_cvs, width, label='Smoothed', color='pink', alpha=0.7)
ax.set_ylabel('Coefficient of Variation')
ax.set_title('Point Distribution Uniformity (CV)')
ax.set_xticks(x)
ax.set_xticklabels([f'C{i}' for i in contour_ids])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visual/burr_v3_4_full/comprehensive_analysis.png', dpi=150, bbox_inches='tight')
print("\n综合分析图已保存: visual/burr_v3_4_full/comprehensive_analysis.png")
