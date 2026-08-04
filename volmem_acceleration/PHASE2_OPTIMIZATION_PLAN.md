# Phase 2: 轻量级加速实验计划

## Baseline 性能指标 (v0.7 balanced_memory)

### 关键数据
- **平均推理时间**: 36.4 秒/step
- **每体积时间**: 3.03 秒 (12 volumes/step)
- **峰值显存**: 23.4 GB
- **平均显存**: 23.1 GB
- **内存贡献度**: 0.00019 (极低!)

### 分析
1. **36.4s 是什么**：包括12个volume的完整forward+backward+loss计算
2. **3.03s 每体积**：意味着单个volume需要 ~3 秒
3. **显存充足**：还有剩余空间可以增加batch size或chunks
4. **内存机制未利用**：delta极小说明memory对预测帮助有限

---

## 实验方案 A1: 增加 chunks_per_step

### 策略
逐步增加每step处理的chunks数量，减少总体forward循环数

```
当前: chunks_per_step=12, chunk_length=8 → 96 slices/step
目标: chunks_per_step=16/20/24 → 128/160/192 slices/step
```

### 实验配置

#### A1-1: chunks=16 (保守)
```yaml
基础配置: v0.7 balanced_memory
修改:
  chunks_per_step: 12 → 16
  chunk_length: 8 (不变)
预期:
  - 推理时间: -10% ~ -15%
  - 显存: +5% ~ +10%
风险: 低
```

#### A1-2: chunks=20 (中等)
```yaml
基础配置: v0.7 balanced_memory
修改:
  chunks_per_step: 12 → 20
  chunk_length: 8 (不变)
预期:
  - 推理时间: -20% ~ -30%
  - 显存: +15% ~ +20%
风险: 中 (显存可能达到26-27GB)
```

#### A1-3: chunks=24 (激进)
```yaml
基础配置: v0.7 balanced_memory
修改:
  chunks_per_step: 12 → 24
  chunk_length: 8 (不变)
预期:
  - 推理时间: -35% ~ -45%
  - 显存: +25% ~ +35%
风险: 高 (显存可能超过28GB，接近A100上限30GB)
```

### 实现步骤
1. 复制 v0.7 配置到新文件
2. 修改 chunks_per_step 参数
3. 启动训练脚本，监控OOM风险
4. 记录前100个steps的性能数据
5. 对比baseline

---

## 实验方案 D3: 稀疏内存更新

### 想法
根据 memory_read_delta 的大小决定是否更新内存

当前观察：delta ≈ 0.00019，说明大多数slice的内存贡献都很小

### 实现

```python
# volmem/models/slice_memory.py 修改

class DynamicMemoryWrite:
    def __init__(self, threshold=0.0001):
        self.threshold = threshold
        self.write_count = 0
        self.skip_count = 0
    
    def should_write(self, delta: float) -> bool:
        if delta > self.threshold:
            self.write_count += 1
            return True
        else:
            self.skip_count += 1
            return False
```

### 实验配置

#### D3-1: threshold=0.00015
```yaml
修改:
  memory_write_threshold: 0.00015
预期:
  - 内存访问: -10% ~ -20%
  - 推理时间: -5% ~ -10%
  - 精度影响: 极小
风险: 低
```

#### D3-2: threshold=0.0001
```yaml
修改:
  memory_write_threshold: 0.0001
预期:
  - 内存访问: -20% ~ -30%
  - 推理时间: -8% ~ -15%
  - 精度影响: 小
风险: 低-中
```

### 实现步骤
1. 修改 VolMemSnake.write_step() 添加阈值过滤
2. 记录写入skip率
3. 监控内存访问模式变化
4. 对比baseline

---

## 实验方案 A2: 轻量级模型裁剪

### 想法
减少模型内层参数（DiT特征维度），直接降低计算量

### 配置方案

#### A2-1: 特征维度 2304 → 1728 (75%)
```python
# volmem/models/memflow_dit_light.py

class MemFlowDiTLight(MemFlowDiT):
    def __init__(self, ...):
        # 继承架构，只改特征维度
        self.feature_dim = 1728  # -25%
        # 所有后续的dense层都会按比例缩小
```

实验参数:
```yaml
model:
  type: MemFlowDiTLight
  feature_dim: 1728
  
预期:
  - 推理时间: -25% ~ -30%
  - 显存: -25% ~ -35%
  - 精度损失: -2% ~ -5%
训练时间: 3-5 天 (从scratch)
风险: 中 (需要重新训练)
```

#### A2-2: 特征维度 2304 → 1536 (67%)
```yaml
预期:
  - 推理时间: -30% ~ -40%
  - 显存: -35% ~ -45%
  - 精度损失: -3% ~ -8%
风险: 中-高
```

### 实现步骤
1. 创建 memflow_dit_light.py
2. 修改特征维度参数
3. 复制baseline权重初始化（迁移学习）
4. 启动训练，运行5天
5. 对比baseline精度和速度

---

## 综合对比表

| 方案 | 预期加速 | 显存变化 | 精度影响 | 风险 | 实现时间 |
|------|---------|---------|---------|------|---------|
| A1-1 (chunks=16) | 1.12-1.15x | +5-10% | 无 | 低 | 1天 |
| A1-2 (chunks=20) | 1.2-1.3x | +15-20% | 无 | 中 | 1天 |
| A1-3 (chunks=24) | 1.35-1.45x | +25-35% | 无 | 高 | 1天 |
| D3-1 (sparse 0.00015) | 1.05-1.1x | -5-10% | 极小 | 低 | 1天 |
| D3-2 (sparse 0.0001) | 1.08-1.15x | -10-15% | 小 | 低 | 1天 |
| A2-1 (feature 75%) | 1.25-1.3x | -25-35% | -2-5% | 中 | 5天 |
| **组合** | **1.5-1.8x** | **-10-15%** | **-2-5%** | **中** | **6-7天** |

---

## 推荐优化组合

### 快速验证路线 (2-3天)
1. **并行启动**：
   - A1-1 (chunks=16)
   - D3-1 (sparse threshold)
2. 预期综合加速：1.2-1.3x
3. 精度无损

### 中等投入路线 (5-6天)
1. A1-2 (chunks=20)
2. D3-2 (sparse)
3. A2-1 (轻量模型，并行训练)
4. 预期综合加速：1.5-1.8x
5. 精度损失 <5%

### 激进路线 (7-8天)
1. A1-3 (chunks=24)
2. D3-2 (sparse)
3. A2-2 (更轻的模型)
4. 预期综合加速：2.0-2.5x
5. 需要仔细监控OOM和精度

---

## 实验记录标准

每个实验需记录：
```json
{
  "experiment_id": "A1-2",
  "date": "2026-07-XX",
  "config": { "chunks_per_step": 20, ... },
  "metrics": {
    "avg_time_ms": 30000,
    "peak_memory_gb": 25.5,
    "speedup_vs_baseline": 1.21,
    "steps_completed": 50,
    "stability": "stable"
  },
  "notes": "..."
}
```

---

## 下一阶段入口

完成Phase 2后，我们将有：
- ✓ 快速参数优化的性能数据
- ✓ 轻量模型的baseline
- ✓ 累计 1.5-2x 的加速
- ✓ 为Phase 3 (蒸馏) 的基础

Phase 3 将在此基础上探索：
- 多步预测的自蒸馏
- 多尺度推理架构

