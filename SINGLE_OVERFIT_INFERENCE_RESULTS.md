# 单样本过拟合推理结果汇总

## 推理时间
2026年4月15日 23:51 - 23:56

## 推理结果

所有6个模型版本都已成功完成推理，结果保存在 `visual/v3_clean_eval/` 目录。

### 推理文件列表

| 模型版本 | 推理结果文件 | 文件大小 | 生成时间 |
|---------|------------|---------|---------|
| V2.0 | CLEAN_v3_idx0_235213.png | 155K | 23:52:13 |
| V2.1 | CLEAN_v3_idx0_235307.png | 201K | 23:53:07 |
| V2.2 | CLEAN_v3_idx0_235358.png | 168K | 23:53:58 |
| V2.3 | CLEAN_v3_idx0_235433.png | 177K | 23:54:33 |
| V3.1 | CLEAN_v3_idx0_235521.png | 187K | 23:55:21 |
| V3.2 | CLEAN_v3_idx0_235601.png | 197K | 23:56:01 |

## 推理配置

所有模型使用相同的推理配置：
- 测试数据：单样本（来自 `/mnt/sdb1/leijh/DiffusionSnake/Datasets/BTCV/btcv_png_single_overfit`）
- 初始化方式：八边形初始化（与训练一致）
- 权重：各模型的 latest.pt checkpoint

## 模型训练进度（推理时）

| 模型版本 | 训练Epoch | 架构特点 |
|---------|----------|---------|
| V2.0 | 600+ | DiT V2 (RMSNorm + QK-Norm + SwiGLU + CyclicRoPE) |
| V2.1 | 600+ | DiT V2.1 (Anchor Pool) |
| V2.2 | 200+ | DiT V2.2 (MM-DiT Patchify) |
| V2.3 | 200+ | DiT V2.3 (MM-DiT Joint + Flow Matching) |
| V3.1 | 600+ | DiT V3.1 (Patchify + Self->Cross Flow) |
| V3.2 | 700+ | DiT V3.2 (Flow Matching) |

## 推理日志

完整推理日志保存在：`logs/infer_all_single_overfit.log`

## 位移统计

各模型推理时的位移统计：

- **V2.0**: Min: -849.359, Max: 562.290, Mean: 8.501
- **V2.1**: Min: -2185.783, Max: 2664.116, Mean: 11.584
- **V2.2**: Min: -14194.524, Max: 4597.630, Mean: 55.006
- **V2.3**: Min: -71.292, Max: 89.022, Mean: 3.420
- **V3.1**: Min: -3082.375, Max: 3301.341, Mean: 13.084
- **V3.2**: 数据待补充

## 查看结果

```bash
cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30
ls -lh visual/v3_clean_eval/CLEAN_v3_idx0_*.png
```

## 备注

- 所有模型都使用了与训练时一致的八边形初始化
- 推理在单样本上进行，用于验证模型的过拟合能力
- V2.3和V3.2使用了Flow Matching而非传统Diffusion
