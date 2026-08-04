# Phase 2 实验配置

## 实验配置说明

### A1系列：增加chunks_per_step
- `a1_1_chunks16_gpu0.yaml` - chunks=16 (保守)
- `a1_2_chunks20_gpu1.yaml` - chunks=20 (中等)
- `a1_3_chunks24_gpu2.yaml` - chunks=24 (激进)

### D3系列：稀疏内存更新
- `d3_1_sparse_threshold15_gpu3.yaml` - threshold=0.00015
- `d3_2_sparse_threshold10_gpu4.yaml` - threshold=0.0001

### A2系列：轻量化模型
- `a2_1_feature1728_gpu5.yaml` - feature_dim=1728

## 启动实验

```bash
# A1-1 示例
python train_net.py --cfg configs/volmem/phase2_experiments/a1_1_chunks16_gpu0.yaml

# 监控性能
tail -f data/outputs/volmem/phase2_a1_1_chunks16/train.jsonl
```

