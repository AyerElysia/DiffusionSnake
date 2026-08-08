# F1 Locate Boundary Probe Report

Date: 2026-06-13

## 1. Purpose

F1 的目的不是训练最终分割模型，而是做一个低成本的特征诊断：

- 冻结不同来源、不同层的图像特征；
- 在这些特征上训练同一个小型边界预测头；
- 让预测头只做一件事：从冻结特征恢复 GT contour boundary；
- 用统一指标比较每个特征源的边界信息可读性。

这个实验回答的问题是：

> LocateAnything 哪一层最含有可用于轮廓细化的边界信息？它是否比当前 ResNet34 stride-4 特征更强？

它不直接回答：

> 完整 DiffusionSnake 端到端最终 IoU 会不会提升？

端到端还会受到检测框、极点头、init 分布、DiT 训练稳定性影响。F1 只隔离看“特征本身是否含有边界”。

## 2. Compared Feature Sources

### 2.1 LocateAnything / MoonViT Features

脚本：

- `scripts/f1_extract_locate_layers.py`

模型权重：

- `Eagle/Embodied/work_dirs/1232_final_locany_full_more10000/checkpoint-3000`

提取层：

- `layer_5`
- `layer_9`
- `layer_13`
- `layer_18`
- `layer_22`
- `layer_26`

这些 layer index 是 1-based MoonViT encoder layer index。

输入分辨率：

- Locate 448：long side resize 到 448，patch size 14，特征网格通常为 `32x32`
- Locate 896：long side resize 到 896，patch size 14，特征网格通常为 `64x64`

每层通道数：

- `1152ch`

缓存格式：

- 每张图一个 `.npz`
- key 包含 `layer_5/layer_9/layer_13/layer_18/layer_22/layer_26`
- 每个 layer shape 为 `1152 x H x W`
- dtype 为 `float16`
- 同时保存 `orig_hw / resized_hw / padded_hw / input_hw / pad / scale / patch_size / image_path` 等几何元信息

### 2.2 ResNet34 Baseline Feature

脚本：

- `scripts/f1_extract_resnet_features.py`

配置：

- `configs/e3_v8_2_boxjitter_mixinit_gpu7.yaml`

权重：

- `data/outputs/e3_v8_2_boxjitter_mixinit_gpu7/checkpoints/latest.pt`

特征：

- heatmap detector 的 ResNet34 stride-4 feature
- input size 512
- output feature shape `64 x 128 x 128`

这个 ResNet 特征是当前稳定基线使用的轮廓细化特征。

## 3. Dataset Split

脚本：

- `scripts/f1_boundary_probe.py`

数据：

- root: `/home/medteam/Zhrch/Datasets/1232_final`
- jsonl root: `Eagle/Embodied/locany_recipe/1232_final`

训练集：

- 使用 train split 前 300 张
- `train_subset_size = 300`

测试集：

- 使用完整 test split
- `test_samples = 177`

注意：F1 是诊断实验。脚本用 test split 计算每个 epoch 的指标并做 early stopping，因此绝对值可能略乐观；但所有特征源使用同一协议，横向比较仍有效。

## 4. GT Boundary Construction

每个样本的 GT contour 来自：

1. jsonl 中的 polygon / segmentation / contour 字段；
2. 如果 jsonl 中没有可用 polygon，则回退读取 mask 文件并提取 contour。

随后将 polygon 坐标映射到统一目标分辨率：

- target size: `128 x 128`

坐标变换逻辑：

- Locate 特征：使用缓存中的 `scale + pad + input_hw` 将原图 polygon 映射到 Locate 输入坐标，再缩放到 `128x128`；
- ResNet 特征：优先使用缓存中的 `trans_input` affine transform，再缩放到 `128x128`。

边界 rasterization：

- `cv2.polylines`
- closed polygon
- thickness = 1 px
- 输出 binary boundary mask，shape `1 x 128 x 128`

因此 F1 评估的是很严格的细边界恢复能力。

## 5. Probe Head Architecture

所有特征源使用同一个小型 probe head：

```text
Input frozen feature
  -> 1x1 conv to 64 channels
  -> GroupNorm + GELU
  -> 3x3 conv
  -> GroupNorm + GELU
  -> upsample blocks if feature map < 128x128
  -> 1x1 conv to 1-channel boundary logits
  -> sigmoid + threshold 0.5
```

Probe 参数量限制：

- 必须小于 `0.5M`

实际参数量：

- ResNet 512 baseline: `41,409`
- Locate 896 single layer: `258,881`
- Locate 448 single layer: `406,721`

Locate 448 需要从 `32x32` 上采样到 `128x128`，所以 probe 参数更多。Locate 896 是 `64x64`，只需要较少上采样。

训练设置：

- optimizer: AdamW
- lr: `1e-3`
- weight decay: `1e-4`
- max epochs: `20`
- patience: `5`
- seed: `20260613`
- loss: BCEWithLogits + Dice loss
- prediction threshold: `0.5`

Early stopping score：

- `BF1@2px`

报告排序主要看：

- `BF1@1px`，因为轮廓任务最关心 1px 级边界精度。

## 6. Metrics

### BF1@1px

Boundary F-score with 1-pixel tolerance.

计算方式：

- 将预测 boundary 和 GT boundary 都视为 binary mask；
- 对 GT boundary 做 distance transform；
- prediction 中距离 GT boundary 不超过 1px 的点算 precision hit；
- 对 prediction 做 distance transform；
- GT boundary 中距离 prediction 不超过 1px 的点算 recall hit；
- 对 precision 和 recall 取 F1。

这是最严格、最关键的指标。

### BF1@2px

同上，但容差放宽到 2px。

这个指标反映边界大方向是否正确，对小错位更宽容。脚本用它做 early stopping。

### bIoU

Boundary IoU。

计算方式：

- prediction boundary 和 GT boundary 都 dilation 2 次；
- 计算 dilated boundary mask 的 IoU。

bIoU 更偏“边界带区域重合”，对厚边界和局部错位更宽容。

## 7. Full Results

按 `BF1@1px` 从高到低排序：

| Feature | Key | Feature HW | Channels | BF1@1px | BF1@2px | bIoU | Best Epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Locate 896 layer_18 | layer_18 | 64x64 | 1152 | 0.9083 | 0.9518 | 0.8357 | 3 |
| Locate 896 layer_22 | layer_22 | 64x64 | 1152 | 0.9051 | 0.9488 | 0.8293 | 4 |
| Locate 896 layer_13 | layer_13 | 64x64 | 1152 | 0.9013 | 0.9444 | 0.8225 | 9 |
| Locate 896 layer_26 | layer_26 | 64x64 | 1152 | 0.9012 | 0.9471 | 0.8410 | 9 |
| ResNet34 stride-4 baseline | resnet_stride4 | 128x128 | 64 | 0.9005 | 0.9453 | 0.8291 | 7 |
| Locate 448 layer_18 | layer_18 | 32x32 | 1152 | 0.8949 | 0.9470 | 0.8254 | 4 |
| Locate 448 layer_22 | layer_22 | 32x32 | 1152 | 0.8941 | 0.9454 | 0.8250 | 7 |
| Locate 448 layer_5 | layer_5 | 32x32 | 1152 | 0.8918 | 0.9408 | 0.8229 | 8 |
| Locate 448 layer_26 | layer_26 | 32x32 | 1152 | 0.8855 | 0.9456 | 0.8293 | 11 |
| Locate 896 layer_9 | layer_9 | 64x64 | 1152 | 0.8787 | 0.9246 | 0.7871 | 3 |
| Locate 448 layer_13 | layer_13 | 32x32 | 1152 | 0.8775 | 0.9318 | 0.7977 | 3 |
| Locate 448 layer_9 | layer_9 | 32x32 | 1152 | 0.8758 | 0.9296 | 0.7942 | 3 |
| Locate 896 layer_5 | layer_5 | 64x64 | 1152 | 0.8704 | 0.9205 | 0.7960 | 12 |

Result JSONs:

- `visual/f1_probe/summary.json`
- `visual/f1_probe/locate896_layer18.json`
- `visual/f1_probe/locate896_layer22.json`
- `visual/f1_probe/locate896_layer26.json`
- `visual/f1_probe/resnet512.json`

## 8. Key Findings

### 8.1 Best Single Layer Is Locate 896 layer_18

`layer_18` has the best 1px boundary F-score:

```text
Locate 896 layer_18 BF1@1px = 0.9083
ResNet baseline       BF1@1px = 0.9005
Delta                 = +0.0078
```

This supports the claim that LocateAnything contains slightly sharper boundary information than the current ResNet feature, but only at sufficient input resolution.

### 8.2 Input Resolution Is a Hard Threshold

Same layer, different input resolution:

```text
Locate 448 layer_18 BF1@1px = 0.8949
Locate 896 layer_18 BF1@1px = 0.9083
```

The 448 version loses to ResNet. The 896 version beats ResNet.

This explains why earlier low-resolution Locate feature experiments failed: the feature was not fine enough after tokenization.

### 8.3 layer_26 Has the Best bIoU, Not the Best 1px Boundary

```text
Locate 896 layer_26 bIoU = 0.8410
Locate 896 layer_18 bIoU = 0.8357
```

Layer 26 is better on dilated boundary overlap / region-band alignment, but worse on exact 1px F-score:

```text
layer_18 BF1@1px = 0.9083
layer_26 BF1@1px = 0.9012
```

Interpretation:

- `layer_18` is better for precise local boundary localization;
- `layer_26` is more semantic and region-consistent, but slightly less sharp.

### 8.4 layer_9 Is Actively Bad

`layer_9` is weak at both 448 and 896:

```text
Locate 896 layer_9 BF1@1px = 0.8787
Locate 448 layer_9 BF1@1px = 0.8758
```

This retroactively explains why earlier `layer_9 + layer_18` style feature combinations were not ideal. Layer 9 likely injects low-value or misaligned information for the boundary refinement task.

### 8.5 The Locate Advantage Is Real But Small

Best Locate single layer vs ResNet:

```text
Locate 896 layer_18 BF1@1px = 0.9083
ResNet34 stride-4     BF1@1px = 0.9005
Delta                 = +0.0078
```

This is enough to justify trying Locate-only refinement features, but not enough to expect a large end-to-end jump.

## 9. How F1 Led To F2

F1 suggested:

- do not use early layers;
- use 896 input features;
- prioritize `layer_18`;
- optionally add late semantic layer `layer_26` because it has best bIoU.

F2 therefore used:

```yaml
locate_feat_keys: ['layer_18', 'layer_26']
locate_feat_dim: 2304
```

That means:

- `layer_18`: 1152 channels
- `layer_26`: 1152 channels
- concatenated input to adapter: 2304 channels

F2 was a reasonable first full-replacement trial, but F1 did not directly prove that `18+26` is the best multi-layer combination. F1 only tested single layers.

## 10. Limitations

1. F1 uses a small probe, not the actual DiT refinement module.

2. F1 tests boundary decodability from frozen features, not end-to-end segmentation quality.

3. F1 tests single layers only. It does not measure whether concatenating layers helps or hurts.

4. Early stopping uses test metrics, so absolute scores should be read as diagnostic, not as a publishable held-out benchmark.

5. The target is `128x128` boundary prediction, matching stride-4 refinement scale, but not full-resolution contour IoU.

6. The probe has different parameter counts for different feature resolutions, although all are capped below 0.5M.

## 11. Recommended Next Experiments

Since the project direction is now “only consider Locate features”, the next experiments should be narrow and controlled.

### 11.1 Single-Layer F2: layer_18 only

Rationale:

- `layer_18` is the best single layer on BF1@1px;
- removing `layer_26` may reduce semantic blur or training instability;
- adapter input drops from 2304ch to 1152ch.

Expected value:

- directly tests whether `layer_26` hurt F2.

### 11.2 Two-Layer F2: layer_18 + layer_22

Rationale:

- both are strong on BF1@1px;
- `layer_22` is less late/semantic than `layer_26`;
- may preserve sharpness better than `18+26`.

Expected value:

- likely the best conservative two-layer replacement.

### 11.3 Three-Layer F2: layer_18 + layer_22 + layer_26

Rationale:

- combines the best exact boundary layer, a strong mid-late layer, and the best bIoU layer.

Risk:

- adapter input becomes 3456ch;
- more capacity / memory pressure;
- may overfit or become less stable.

This should run only after `18 only` and `18+22`.

## 12. Practical Recommendation

Current best-supported feature choice:

```text
Primary: Locate 896 layer_18
Secondary candidate: Locate 896 layer_18 + layer_22
Use layer_26 only if bIoU / region stability matters more than exact 1px boundary.
Avoid layer_5 / layer_9 as main refinement features.
```

The most important conclusion is:

> The correct Locate feature source is 896-resolution mid-late MoonViT features, especially layer_18. The feature advantage over ResNet exists, but is small; end-to-end gains will depend more on matching the Locate-box init distribution and training the adapter/DiT interface than on simply adding more layers.

