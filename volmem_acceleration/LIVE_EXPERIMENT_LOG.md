# V0.5 推理加速 - 实时实验日志

## 第一轮实验 (22:03 启动)

### 配置对比

| 实验 | GPU | 改动1 | 改动2 | 改动3 | 预期加速 | 预期显存 |
|------|-----|-------|-------|-------|---------|---------|
| **Exp_A** | 0 | feat_dim: 1152→576 | snake_dim: 256→128 | replace_hidden: 512→256 | 1.25x | -25% |
| **Exp_B** | 1 | chunks: 12→20 | - | - | 1.3x | -10% |
| **Exp_C** | 2 | feat_dim: 1152→384 | snake_dim: 256→64 | replace_hidden: 512→128 | 1.4x | -40% |
| **Exp_D** | 3 | chunks: 12→16 | feat_dim: 1152→864 | snake_dim: 256→192 | 1.2x | -15% |

### V0.5 Baseline (对标)
- 推理时间: 55.3 sec/step
- 显存占用: 30.1 GB
- Loss: 0.005

### 实时进度

#### Exp_A (特征50%压缩)
- 启动: 22:03:??
- 状态: 运行中 🟢
- 更新时间: 待数据
- 预期: 44.2s/step, 22.6GB

#### Exp_B (Chunks+67%)
- 启动: 22:03:??
- 状态: 运行中 🟢
- 更新时间: 待数据
- 预期: 42.6s/step, 27.1GB

#### Exp_C (特征75%压缩)
- 启动: 22:03:??
- 状态: 运行中 🟢
- 更新时间: 待数据
- 预期: 39.2s/step, 18.1GB
- 风险: 精度可能下降3-5%

#### Exp_D (混合优化)
- 启动: 22:03:??
- 状态: 运行中 🟢
- 更新时间: 待数据
- 预期: 45.6s/step, 25.6GB

## 第二轮实验计划 (待触发)

如果任一实验达到预期加速，将启动：
- **Exp_E**: Chunks增加33% (12→16) + 特征压缩50%
- **Exp_F**: 更激进的Chunks (12→24)
- **Exp_G**: 特征压缩+内存容量调整

## 决策树

```
如果 Exp_A 或 Exp_B 达到 1.2x+
  → 启动 Exp_E (组合)
  
如果 Exp_C 达到 1.4x+ 且精度损失 <5%
  → 启动 Exp_F (更激进)
  
如果所有 Exp 都失败或低于 1.1x
  → 启动代码级改造 (MoE 剪枝)
  
如果任一 Exp 达到 1.5x+
  → 立即启动第三轮 (多尺度/蒸馏)
```

## 实验监控指令

```bash
# 实时查看各实验进度
watch -n 10 'for i in a b c d; do echo "=== Exp_$i ==="; tail -2 logs/exp_$i.log; done'

# 提取性能数据
for i in a b c d; do 
  echo "Exp_$i:"; 
  tail -100 data/outputs/volmem/exp_${i}_*/train.jsonl | tail -1 | python -m json.tool | grep -E "time_ms|peak_memory_gb|loss"; 
done

# 显存监控
nvidia-smi -q -d memory | grep -E "Process|gpu|memory"
```

## 关键时间点

- T+0: 22:03 - 4个实验启动
- T+2h: 预期第一批数据（取决于实验大小）
- T+4h: 数据对比 + 决策第二轮
- T+8h: 第二轮结果
- T+24h: 最优方案初步确定

