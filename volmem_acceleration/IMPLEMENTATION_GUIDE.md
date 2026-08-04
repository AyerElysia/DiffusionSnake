# 实现指南

## 1. 推理Benchmark脚本

位置：`volmem/eval/inference_benchmark.py`

```python
class VolMemInferenceBenchmark:
    - load_model(checkpoint_path)
    - benchmark_volume(volume_id, num_runs=3)
    - report_metrics()
```

关键指标：
- Total inference time
- Per-slice inference time
- Peak GPU memory
- Memory bandwidth usage

## 2. 配置参数优化空间

```yaml
# 外层参数
chunks_per_step: 12 → 16/20/24
chunk_length: 8 → 12/16

# 内层模型参数
feature_dim: 2304 → 1728/1152
memory_dim: 256 → 128/192
memory_capacity: 4 → 8/2
```

## 3. 蒸馏框架

Base class: `MultiStepMemFlowDiT`

```python
class DistillationHead(nn.Module):
    def __init__(self, steps_ahead: int):
        # Predict k steps into the future
        pass
    
    def forward(self, current_state):
        return [pred_slice_1, pred_slice_2, ..., pred_slice_k]
```

## 4. 并行推理设计

Window-based parallel processing:
- 窗口大小 = chunk_length
- 窗口内全并行（自注意力）
- 窗口间序列依赖（通过attention mask）

