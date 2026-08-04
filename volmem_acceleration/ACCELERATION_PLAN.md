# VolMem 推理加速探索方案

## 执行概要
- **负责人**：AI Agent
- **基础线路**：verse_memflowdit_v0_7_balanced_memory_gpu1
- **目标**：推理加速 ≥50%，精度损失 <5%
- **时间表**：4-5 周分阶段实现

## 核心发现
1. **当前架构特性**：
   - MemFlowDiT with slice-sequential autogressive inference
   - Memory bank capacity: 4 slices, 256-dim
   - Peak memory: 23.4GB (batch=128)
   - Memory contribution low (delta≈0.0001)

2. **关键瓶颈**：
   - 完全序列化推理（S1→S2→...→SN）
   - 每step处理96个slices（12 chunks × 8 length）
   - 内存机制未充分利用（贡献度极低）

3. **优势**：
   - 无成熟预训练模型约束 → 探索自由度大
   - 稳定的训练基线已建立
   - 模块化架构便于改造

## 加速方向（优先级排序）

### Phase 1: Baseline & Instrumentation (2-3 days)
- [ ] 建立推理benchmark脚本
- [ ] 记录baseline性能指标
- [ ] 设置实验跟踪框架

### Phase 2: Quick Wins (1 week)
- [ ] A1: chunks_per_step 12→16/20
- [ ] D3: Sparse memory writes
- [ ] A2: Lightweight model pruning (feature_dim reduction)
- Expected: 1.3-1.5x speedup

### Phase 3: Medium Risk (2 weeks)
- [ ] B1: Temporal self-distillation
- [ ] C2: Hierarchical multi-scale inference
- Expected: 2-3x speedup

### Phase 4: High Impact (3+ weeks)
- [ ] C1: Parallel n-way inference architecture
- Expected: 5-10x speedup

## 详见下列文档
- IMPLEMENTATION_GUIDE.md - 技术实现细节
- EXPERIMENT_LOG.md - 逐日实验记录
