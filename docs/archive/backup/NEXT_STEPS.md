# 接下来要做什么 - 行动计划

## 📋 当前状态

✅ **已完成：**
1. 深度分析了毛刺问题
2. 发现小轮廓比大轮廓更容易毛刺
3. 数据证明：点密度 vs 曲率相关系数 = -0.613
4. 确认原因：固定128点导致小轮廓点太密集
5. 提出解决方案：自适应点数

## 🎯 接下来的三个阶段

### 阶段1：快速验证（1-2天）⏰ **← 当前阶段**

**目标：** 验证"减少小轮廓点数"是否真的能改善毛刺

**具体步骤：**

#### 步骤1：保存预测结果（30分钟）

修改 `analyze_burr_v3_4_full.py`，在推理后保存预测轮廓：

```python
# 在 run_full_image_inference 函数末尾添加
np.save(os.path.join(save_dir, 'pred_polys.npy'), pred_polys)
np.save(os.path.join(save_dir, 'gt_polys.npy'), gt_np[0])
print(f"[✔] 预测结果已保存")
```

然后重新运行：
```bash
python analyze_burr_v3_4_full.py
```

#### 步骤2：运行自适应验证（30分钟）

等GPU空闲后运行：
```bash
# 等待GPU 0空闲
nvidia-smi

# 运行验证
CUDA_VISIBLE_DEVICES=0 python test/verify_adaptive_points.py
```

或者修改脚本直接加载已保存的预测结果（不需要GPU）。

#### 步骤3：查看结果（10分钟）

查看生成的文件：
- `test/adaptive_verification/adaptive_verification_comparison.png` - 可视化对比
- `test/adaptive_verification/adaptive_verification_results.json` - 数值结果

**成功标准：**
- 小轮廓（1、5、2）曲率降低 > 30%
- 大轮廓（3、4）曲率变化 < 10%

**如果成功 → 进入阶段2**  
**如果失败 → 重新评估假设**

---

### 阶段2：训练时集成（1周）⏰

**目标：** 在训练流程中集成自适应点数

**具体步骤：**

#### 步骤1：修改数据准备（2天）

修改 `lib/datasets/btcv/snake.py`:
- 添加 `adaptive_points` 开关
- 根据周长计算目标点数
- 重采样到目标点数

#### 步骤2：修改模型支持可变点数（2天）

修改 `lib/datasets/collate_batch.py`:
- 支持batch中不同点数
- 使用padding + mask

修改 `lib/networks/snake/ct_snake.py`:
- 在损失计算时使用mask

#### 步骤3：单样本训练验证（2天）

```bash
# 训练V3.4自适应版本
python diffusion_train.py --cfg configs/btcv_diffusion_dit_v3_4_adaptive.yaml
```

#### 步骤4：评估改善（1天）

```bash
python analyze_burr_v3_4_full.py --cfg configs/btcv_diffusion_dit_v3_4_adaptive.yaml
```

**成功标准：**
- 小轮廓平均曲率 < 15（当前28.1）
- 训练收敛稳定
- 推理速度下降 < 20%

---

### 阶段3：全面评估（2周）⏰

**目标：** 在完整数据集上验证效果

**具体步骤：**

#### 步骤1：完整数据集训练（1周）
#### 步骤2：全面评估（3天）
#### 步骤3：消融实验（2天）
#### 步骤4：最终报告（2天）

---

## 📝 立即可以做的事情

### 选项A：等GPU空闲后运行完整验证

```bash
# 1. 检查GPU状态
nvidia-smi

# 2. 等GPU 0空闲后运行
CUDA_VISIBLE_DEVICES=0 python test/verify_adaptive_points.py
```

### 选项B：修改脚本使用已保存的结果（推荐）

修改 `test/verify_adaptive_points.py`，让它直接加载之前保存的预测结果，不需要重新推理。

### 选项C：先做理论分析和文档

继续完善：
- 验证方案的细节
- 实现代码的准备
- 预期效果的量化分析

---

## 🎯 推荐的下一步行动

### 今天（立即执行）

1. **修改 analyze_burr_v3_4_full.py 保存预测结果**
   ```bash
   # 在文件末尾添加保存代码
   # 然后重新运行（如果GPU空闲）
   ```

2. **或者：修改验证脚本直接使用已有数据**
   - 从 `visual/burr_v3_4_full/` 提取预测轮廓
   - 应用自适应重采样
   - 计算改善效果

3. **查看和理解验证方案**
   - 阅读 `docs/archive/plan/archived/adaptive_points_verification_plan.md`
   - 确认实施细节

### 明天

1. **完成阶段1验证**
   - 运行验证脚本
   - 分析结果
   - 决定是否继续

2. **如果验证成功（改善>30%）**
   - 开始准备阶段2的代码修改
   - 设计自适应点数的实现细节

### 本周内

1. **完成阶段2的代码修改**
2. **单样本训练验证**
3. **评估改善效果**

---

## 📊 预期时间线

| 阶段 | 时间 | 关键里程碑 |
|------|------|-----------|
| 阶段1 | 1-2天 | 验证假设，改善>30% |
| 阶段2 | 1周 | 训练集成，小轮廓曲率<15 |
| 阶段3 | 2周 | 全面评估，整体改善>30% |
| **总计** | **3-4周** | **完整解决方案** |

---

## 🔧 需要的资源

### 计算资源
- GPU：1张（训练时）
- 内存：24GB GPU内存
- 时间：单样本训练约2-4小时

### 代码修改
- 数据准备：`lib/datasets/btcv/snake.py`
- Collate函数：`lib/datasets/collate_batch.py`
- 模型：`lib/networks/snake/ct_snake.py`
- 配置：新增 `configs/btcv_diffusion_dit_v3_4_adaptive.yaml`

---

## ✅ 成功标准

### 阶段1（必须达成）
- ✓ 小轮廓曲率降低 > 30%
- ✓ 可视化效果明显改善

### 阶段2（必须达成）
- ✓ 小轮廓平均曲率 < 15
- ✓ 训练收敛稳定
- ✓ 推理速度下降 < 30%

### 阶段3（期望达成）
- ✓ 整体毛刺改善 > 30%
- ✓ 所有轮廓曲率标准差 < 8
- ✓ 在完整数据集上验证有效

---

## 🚨 风险提示

### 风险1：验证可能失败
**如果阶段1验证改善<10%：**
- 重新评估假设
- 考虑其他因素（如器官类型、形状复杂度）
- 尝试其他解决方案（尺度归一化、多尺度训练）

### 风险2：实现复杂度高
**如果可变点数实现困难：**
- 使用简化版本（固定几档点数）
- 使用padding + mask方案
- 考虑备选方案（尺度归一化）

### 风险3：训练不稳定
**如果训练出现问题：**
- 调整学习率
- 减小点数范围
- 逐步增加点数范围

---

## 📞 需要帮助时

如果遇到问题，可以：
1. 查看详细文档：`docs/archive/plan/archived/adaptive_points_verification_plan.md`
2. 查看分析报告：`docs/archive/notion/archived/contour_size_vs_burr_analysis.md`
3. 查看完整总结：`BURR_ANALYSIS_COMPLETE_SUMMARY.md`

---

**创建时间：** 2026-04-18  
**当前阶段：** 阶段1 - 快速验证  
**下一步：** 运行验证脚本或修改脚本使用已有数据  
**预计完成：** 2026-05-02（3-4周）
