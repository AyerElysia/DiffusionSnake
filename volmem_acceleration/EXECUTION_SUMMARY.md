# VolMem 推理加速 - 执行总结

## 任务概要

**负责人**: AI Agent  
**基础线路**: verse_memflowdit_v0_7_balanced_memory_gpu1  
**总目标**: 推理时间减少 ≥50%，精度损失 <5%  
**预期周期**: 4-5 周分阶段实现  

---

## 现状快照

### Baseline 性能（v0.7 balanced_memory）
| 指标 | 数值 | 说明 |
|------|------|------|
| **推理时间** | 36.4 sec/step | 包含12个volume的forward+backward |
| **单体积耗时** | 3.03 sec | 128 slices + 梯度计算 |
| **峰值显存** | 23.4 GB | A100 还有 6.6 GB 空间 |
| **平均显存** | 23.1 GB | - |
| **内存贡献度** | 0.00019 | 极低!意味着内存机制未充分利用 |
| **训练数据** | 29 steps | 新架构探索优势 |

### 架构分析
- ✓ **强点**：稳定的基线，模块化架构，配置灵活
- ✗ **弱点**：完全序列化推理，内存机制利用率极低
- ✓ **机会**：显存空间充足，无过度拟合的预训练模型束缚
- ✗ **风险**：并行化改造需要重新训练

---

## 分阶段计划

### Phase 1: 基线与工具 (2-3 天) ✓ 完成

**目标**: 建立性能度量框架和实验基础设施

**交付物**:
- [x] Baseline 性能提取：36.4s/step，23.4GB
- [x] Benchmark 脚本框架
- [x] 实验记录系统
- [x] 文档框架和实验日志

**关键发现**:
- 内存贡献度低（0.00019）→ 稀疏优化空间
- 显存充足（还有6.6GB）→ 可增加并行度
- 架构成熟 → 可大胆优化

---

### Phase 2: 快速加速 (5-7 天) → **当前焦点**

**目标**: 获得 1.3-1.8x 加速，验证优化基础设施

**并行实验**（可同时启动）:

#### A系列：步数减少
| 实验 | 改动 | 预期 | 风险 | 时间 |
|------|------|------|------|------|
| A1-1 | chunks: 12→16 | 1.12-1.15x | 低 | 1天 |
| A1-2 | chunks: 12→20 | 1.20-1.30x | 中 | 1天 |
| A1-3 | chunks: 12→24 | 1.35-1.45x | 高 | 1天 |
| A2-1 | feat_dim: 2304→1728 | 1.25-1.30x | 中 | 5天 |

#### D系列：稀疏内存
| 实验 | 改动 | 预期 | 风险 | 时间 |
|------|------|------|------|------|
| D3-1 | threshold=0.00015 | 1.05-1.10x | 低 | 1天 |
| D3-2 | threshold=0.0001 | 1.08-1.15x | 低 | 1天 |

**推荐组合**:
1. **快速验证** (2-3天): A1-1 + D3-1 → 1.2-1.3x
2. **中等投入** (5-6天): A1-2 + D3-2 + A2-1 → 1.5-1.8x
3. **激进方案** (7-8天): A1-3 + D3-2 + A2-2 → 2.0-2.5x

**交付物**:
- 配置文件和实验脚本
- 性能对比数据
- 精度验证报告
- 为Phase 3的稳定基线

**实验启动**:
```bash
# A1-1 示例
python train_net.py --cfg configs/volmem/phase2_experiments/a1_1_chunks16_gpu0.yaml

# 监控（另一终端）
tail -f data/outputs/volmem/phase2_a1_1_chunks16/train.jsonl | python -m json.tool
```

---

### Phase 3: 中等加速 (2-3 周)

**目标**: 2-3x 加速，通过蒸馏和多尺度推理

**两条并行路线**:

1. **B1: 自蒸馏** (Temporal Distillation)
   - 多步预测头（预测 k=2/3/4 步未来）
   - Loss = L_pred + α·L_distill
   - 推理时步数减少 k 倍
   - 预期加速：2-3x

2. **C2: 多尺度推理** (Hierarchical)
   - 粗层：间隔3取1
   - 中层：间隔2取1
   - 精层：全处理
   - 早期粗层预测指导细层
   - 预期加速：2x

**交付物**:
- 自蒸馏框架代码
- 多尺度推理框架
- 融合性能数据
- 为Phase 4的架构验证

---

### Phase 4: 极限加速 (3-4 周)

**目标**: 5-10x 加速，完全并行推理架构

**核心改造**: C1 并行推理

从序列化改并行化：
```
原来: S1 -M-> S2 -M-> S3 -M-> ...
      ↓ predict ↓
      ...

改为: [S1, S2, S3, S4] (window)
      ↓ parallel
      predict_all
      ↓ memory_fusion
```

**关键技术**:
- WindowedAttention（窗口内并行）
- CrossSliceAttention（跨slice距离编码）
- 灵活的依赖管理

**风险**: 需要完整重训练，architecture 变化大

**预期收益**: 5-10x 推理加速

---

## 成功标准

### 必须满足
- ✓ 推理时间 < 18.2 sec/step (50%加速)
- ✓ 精度 ≥ baseline × 95%
- ✓ 显存 ≤ 24 GB（无爆炸）
- ✓ 完整文档和可复现结果

### 理想目标
- ✓ 推理时间 < 12 sec/step (66%加速)
- ✓ 精度 ≥ baseline × 98%
- ✓ 显存 ≤ 20 GB
- ✓ 高度并行化架构

---

## 关键文档位置

```
/home/medteam/Zhrch/DiffusionSnake-12-30/volmem_acceleration/

├── ACCELERATION_PLAN.md              # 总体规划
├── DETAILED_PLAN.md                  # 详细技术方案
├── PHASE2_OPTIMIZATION_PLAN.md       # Phase 2具体实验
├── EXPERIMENT_LOG.md                 # 逐日记录
│
├── scripts/
│   └── inference_benchmark.py        # Benchmark工具
│
├── configs/phase2_experiments/       # Phase 2配置
│   ├── a1_1_chunks16_gpu0.yaml
│   ├── a1_2_chunks20_gpu1.yaml
│   └── ...
│
└── results/
    ├── baseline_v0_7.json            # Baseline数据
    ├── phase2_results.json           # Phase 2结果
    └── ...
```

---

## 实验跟踪模板

每个实验需记录为 JSON：
```json
{
  "experiment_id": "A1-2",
  "date": "2026-08-02",
  "phase": 2,
  "config": {
    "chunks_per_step": 20,
    "chunk_length": 8
  },
  "metrics": {
    "avg_time_ms": 28800,
    "peak_memory_gb": 25.2,
    "speedup_vs_baseline": 1.26,
    "steps_completed": 100,
    "training_stable": true
  },
  "accuracy": {
    "f1_score": 0.92,
    "loss": 0.0045
  },
  "notes": "Stable training, no OOM risk"
}
```

---

## 下一步行动

### 立即执行 (今天/明天)
1. ✓ 基础设施和文档已就位
2. ✓ Baseline 指标已提取
3. **Next**: 启动 Phase 2 实验

### Phase 2 首个实验
```bash
# 启动 A1-1 (保守方案，低风险验证)
cd /home/medteam/Zhrch/DiffusionSnake-12-30
python train_net.py --cfg configs/volmem/phase2_experiments/a1_1_chunks16_gpu0.yaml

# 预期运行 100 steps (~2 小时)
# 监控性能变化
```

### 并行准备
- [ ] 分析 A2-1 轻量模型初始化策略
- [ ] 设计 D3 稀疏内存阈值实验
- [ ] 准备 Phase 3 蒸馏框架代码

---

## 风险管理

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| OOM 爆炸 | 中 | 高 | 从保守→激进逐步扩展 |
| 精度下降 | 中 | 中 | 快速回滚到baseline权重 |
| 收敛困难 | 低 | 中 | 详细log监控，调整lr |
| 架构bug | 低 | 高 | 完整单测和集成测试 |

---

## 预期时间表

| Phase | 周数 | 累计加速 | 关键门槛 |
|-------|------|---------|---------|
| 1 | 2-3天 | 1.0x | 基线建立 |
| 2 | 1周 | 1.5-1.8x | 快速参数优化验证 |
| 3 | 2-3周 | 2-3x | 蒸馏/多尺度有效性 |
| 4 | 3-4周 | 5-10x | 并行架构稳定性 |

**总周期**: 4-5 周内达到 1.5x 加速  
**理想目标**: 2-3 个月内达到 5x 加速

---

## 资源需求

### GPU
- Phase 1-2: 可用 5-6 个 GPU 并行实验
- Phase 3: 需要 3-4 个 GPU (训练 + 对照)
- Phase 4: 需要 2 个 GPU (baseline 对照)

### 存储
- 配置和脚本: ~5 MB
- 实验结果: ~50-100 MB
- 检查点: 自动管理（复用baseline）

---

## 成功指标和奖励

### 如果达成:
✓ **1.5x 加速** (Phase 2 中期) → 可立即用于生产
✓ **2-3x 加速** (Phase 3 完成) → 显著性能提升
✓ **5x 加速** (Phase 4 完成) → 技术突破论文

### 期中检查点 (第2周)
- [ ] Phase 2 至少 3 个实验完成
- [ ] 最好情况下获得 1.3x 加速
- [ ] 无内存溢出或精度崩溃

---

## 联系与讨论

### 关键决策点
1. Phase 2 中激进度选择 (A1-1 vs A1-2 vs A1-3)
2. Phase 3 蒸馏 vs 多尺度 优先级
3. Phase 4 时机 (基于Phase 3结果)

### 每周回顾点
- 实验结果对比
- 精度趋势分析  
- 架构改进建议
- 下周实验计划调整

---

**更新时间**: 2026-07-31  
**版本**: 1.0 - 初始执行计划  
**状态**: ✓ 准备就绪 → 开始 Phase 2

