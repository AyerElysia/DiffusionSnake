# VolMem 推理加速探索方案

## 1. 现状分析

### 架构现状
- **模型框架**：MemFlowDiT (Flow Diffusion Transformer with Memory)
- **核心特性**：
  - 串行自回归推理（slice-by-slice）
  - 内存机制：SliceMemoryBank（容量=4个slice）
  - 内存容量：256维，8个注意力头
  - 内存池大小：8x8（用于mask编码）
  - Mask通道数：26（多类分割）

### 推理流程（当前）
```
Slice 1 → Memory Write → Slice 2 → Memory Write → Slice 3 → ... → Slice N
  ↓
  Memory Read（前几个slice的特征）
  ↓
  Contour Adapter Forward
  ↓
  生成预测
```

### 性能指标（从最新训练日志 step 23）
- 平均损失：~0.005-0.006
- 内存大小：4.0（满容）
- 内存读取delta：~0.00019（很小，说明内存贡献度低）
- 峰值显存：23.4GB（batch=128, 12个volume/step）
- 推理步长：~50ms/step（包括梯度计算）
- 活跃状态数：48（8个volume × 6个contours？）

### 关键观察
1. **内存贡献度低**：memory_read_delta极小（0.0001-0.0002）
   - 说明目前的内存特征对预测帮助有限
   - 可能是缺乏足够的预训练数据
   
2. **序列化瓶颈**：每个slice必须等待前一个slice的结果
   - 当前配置：chunks_per_step=12, chunk_length=8
   - 意味着每一"步"处理96个slice（12 × 8）
   
3. **内存容量不足**：capacity=4（只能存4个slice的状态）
   - 相对于总的体积深度，这个容量有限
   
4. **预训练资源**：当前模型从2D基础检查点初始化
   - 没有专门的volmem预训练模型
   - 这给了我们自由探索的空间

## 2. 可行的加速方向

### 方向 A：步数减少（IMMEDIATE，低风险）

#### A1. 减少外层推理步数
- **目标**：每个体积的推理步数从当前的 `总slice数/12` 减少
- **方法**：
  - 增加 `chunks_per_step`（12 → 24/32）
  - 增加 `chunk_length`（8 → 16）
  - 组合优化：24 chunks × 16 length = 384 slice/step
  
- **权衡**：
  - ✓ 简单改配置即可
  - ✗ 可能OOM（显存已经23.4GB）
  - ✗ 长梯度反传可能不稳定

#### A2. 减少内层推理步数（模型层面）
- **目标**：降低 DiT 块数或自注意力分辨率
- **方法**：
  - 减少 flow_dit_blocks 数量
  - 降低特征维度（2304 → 1152）
  - 缩小时间戳编码维度
  
- **权衡**：
  - ✓ 显存占用大幅下降
  - ✗ 需要重新训练
  - ✗ 精度损失未知

### 方向 B：蒸馏（MEDIUM，中风险）

#### B1. 自蒸馏（Temporal Distillation）
- **想法**：用当前模型作为教师，学生模型预测多个future slice
- **方法**：
  ```
  Teacher: 逐slice预测
  Student: 一次预测 k 个slice（如5步）
  Loss = L_pred + α * L_distill
  ```
  
- **优势**：
  - 推理时步数减少 k 倍
  - 用同一个模型蒸馏，无需两个checkpoints
  - 可以逐步增加 k 值
  
- **实现**：
  - 修改 forward_step 添加多步预测头
  - 添加 multi-step contour loss
  - 添加知识蒸馏loss

#### B2. 模型压缩蒸馏
- **想法**：训练小模型学习大模型的行为
- **方法**：
  - 小模型：DiT块数 ↓50%，特征维 ↓30%
  - 损失：L_task + α*L_teacher_forcing + β*L_attention_align
  
- **权衡**：
  - ✓ 最快推理速度
  - ✗ 需要并行维护两个模型
  - ✗ 蒸馏收敛可能较慢

### 方向 C：并行推理架构（HIGH IMPACT，高风险）

#### C1. Parallel N-way Inference
- **想法**：同时处理 N 个slice，而不是序列化
- **架构变化**：
  ```
  原来：S1 -M-> S2 -M-> S3 -M-> S4 ...
  
  改为：[S1, S2, S3, S4]
        ↓
        Parallel 
        Contour Pred
        ↓
        Memory Fusion
        (后向依赖)
        ↓
        Loss & Update
  ```

- **内存机制**：
  - 允许跨slice注意力（距离编码）
  - 窗口内自注意力
  - 全局内存聚合

- **困难**：
  - ✗ 需要重新设计 MemFlowDiT
  - ✗ 需要新的注意力掩码机制
  - ✗ 需要重新训练
  - ✗ 依赖关系处理复杂

#### C2. Multi-scale Hierarchical Inference
- **想法**：分层推理，粗到细
- **流程**：
  1. 粗层：间隔采样slice（每3个取1个）
  2. 中层：细化间隔部分
  3. 精层：局部细化
  
- **优势**：
  - ✓ 并行性较好
  - ✓ 早期粗略预测可指导细层
  - ✓ 保持存储高效性

### 方向 D：内存优化（BASELINE，低风险）

#### D1. 动态内存分配
- 根据slice特性动态调整 memory_capacity
- 重要区域（器官边界）保持高容量
- 单调区域（均质组织）降低容量

#### D2. 内存压缩
- 用自编码器压缩内存状态（256 → 64）
- 减少跨slice传递的数据量

#### D3. 稀疏内存更新
- 只更新"显著"变化的slice
- 基于 memory_read_delta 阈值决定写入

## 3. 推荐的实验路线

### Phase 1: 基础测试与度量（Week 1）
**目标**：建立性能基线和评估框架

1. **选择基础实验线**：
   - ✓ 推荐：`verse_memflowdit_v0_7_balanced_memory_gpu1`
   - 理由：最新、配置稳定、已产生收敛的训练曲线
   
2. **建立评估框架**：
   - 推理速度（总时间、每slice时间）
   - 显存峰值
   - 精度指标（F1 score等）
   - 内存贡献度分析

3. **创建基准实验**：
   - 冻结当前模型，在验证集上测试
   - 记录baseline推理时间

### Phase 2: 轻量级加速（Week 2）
**目标**：快速获得10-20%加速，验证基础设施

1. **A1: 增加 chunks_per_step**
   - 配置修改：12 → 16（需要显存检查）
   - 风险低，改配置即可
   - 预期加速：~15%

2. **D3: 稀疏内存更新**
   - 基于 memory_read_delta 阈值过滤
   - 预期加速：~8%

3. **A2: 轻量级模型裁剪**
   - 特征维 2304 → 1728（25%裁剪）
   - 需要重新训练（但只需2-3天）
   - 预期加速：~25%，精度损失 <5%

### Phase 3: 中等风险加速（Week 3-4）
**目标**：30-50%加速，探索蒸馏和并行初步

1. **B1: 自蒸馏框架**
   - 添加多步预测头（2/3/4 steps ahead）
   - 修改loss函数
   - 从 k=2 开始逐步扩展
   - 预期加速：2x→3x（取决于 k）

2. **C2: 多尺度推理探索**
   - 设计粗中精三层
   - 实现层间通信机制
   - 预期加速：2x（50%步数减少）

### Phase 4: 高风险高收益（Week 5+）
**目标**：5-10x加速，完全并行推理

1. **C1: 完全并行架构**
   - 重设计 MemFlowDiT
   - 实现窗口内注意力
   - 预期加速：3-5x

## 4. 预训练资源问题的思考

### 当前状态
- 从 v4.6c 2D基础模型初始化
- 仅 ~20步训练数据（step 23）
- 内存贡献度极低（delta=0.0001）

### 策略
✓ **这其实是优势**：
  - 没有低效内存机制被过度拟合
  - 新架构探索自由度大
  - 可以用小型预训练数据验证策略

✓ **建议**：
  - 先用现有数据（或合成数据）验证加速效果
  - 不需要等待完整预训练
  - 一旦架构确定，再做长期训练

## 5. 详细实现计划

### Phase 1 具体步骤

#### 1.1 建立推理评估脚本
```python
# volmem/eval/inference_benchmark.py
- 加载 checkpoint
- 测试单个volume推理
- 记录：总时间、每slice时间、显存峰值
- 输出：inference_metrics.json
```

#### 1.2 获取性能基线
- 在验证集上运行当前 v0.7 模型
- 记录baseline数值

#### 1.3 代码文档框架
- 创建 VOLMEM_ACCELERATION.md（总体方案）
- 创建 EXPERIMENT_LOG.md（实验记录）
- 每个优化分支创建对应文档

### Phase 2 具体步骤

#### 2.1 配置参数扫描
```yaml
# configs/volmem/baseline_sweep.yaml
chunks_per_step: [12, 16, 20, 24]
chunk_length: [8, 12, 16]
memory_capacity: [4, 8, 12]
```

#### 2.2 轻量模型变种
```python
# volmem/models/memflow_dit_light.py
- feature_dim: 1728（75%）
- num_denoiser_layers: 12（从18）
```

#### 2.3 稀疏内存更新
```python
# volmem/models/slice_memory.py 修改
- memory_write_threshold: 0.0001
- 只写入 delta > threshold 的state
```

### Phase 3 具体步骤

#### 3.1 自蒸馏框架
```python
# volmem/models/memflow_dit_distilled.py
- MultiStepHead: 预测 k 步后的结果
- DistillationLoss: 知识蒸馏
- 支持可变 k（2/3/4）
```

#### 3.2 多尺度推理
```python
# volmem/models/hierarchical_inference.py
- HierarchicalMemFlowDiT
- 三层：coarse(step=3)/medium(step=1.5)/fine(step=1)
- 层间跳跃连接
```

### Phase 4 具体步骤

#### 4.1 重新设计注意力机制
```python
# volmem/models/parallel_memflow.py
- ParallelMemFlowDiT
- WindowedAttention（窗口 = chunk_length）
- CrossSliceAttention（相邻slice间）
```

## 6. 预期时间表

| Phase | 任务 | 时间 | 预期加速 | 风险 |
|-------|------|------|---------|------|
| 1 | 基线建立 | 2-3天 | 1x | 低 |
| 2 | 轻量化 | 1周 | 1.3-1.5x | 低 |
| 3 | 蒸馏+多尺度 | 2周 | 2-3x | 中 |
| 4 | 并行架构 | 3周+ | 5-10x | 高 |

## 7. 文档记录策略

### 核心文档
1. **VOLMEM_ACCELERATION.md**（本方案）
2. **EXPERIMENT_LOG.md**（逐日记录）
3. **IMPLEMENTATION_GUIDE.md**（技术细节）

### 代码文档
- 每个新模块：类级和函数级docstring
- 配置文件：参数注释说明
- 实验脚本：运行指南

### 结果跟踪
- 每个实验：CSV结果表
- 性能曲线图表
- 对比分析

## 8. 风险管理

| 风险 | 缓解策略 |
|-----|---------|
| 显存爆炸 | 从小参数开始，逐步扩展 |
| 精度下降 | 保持loss监控，快速回滚 |
| 复杂架构难实现 | 渐进式设计，阶段评估 |
| 缺少基础数据 | 用现有数据验证想法 |

## 9. 成功指标

- ✓ 推理时间减少 ≥ 50%
- ✓ 精度维持在 baseline 的 95% 以上
- ✓ 显存占用 ≤ baseline
- ✓ 完整文档和可复现实验
- ✓ 代码干净可维护

