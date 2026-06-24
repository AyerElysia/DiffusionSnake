# F1 Boundary Probe Method Explanation

Date: 2026-06-15

## 1. Executive Summary

F1 边界探针的目标不是训练最终模型，而是用一个低成本、可控的诊断实验回答一个更基础的问题：

> 冻结的 LocateAnything 特征里，哪一层最容易读出 GT 轮廓边界？

核心做法是：

1. 先离线提取不同特征源的 frozen feature。
2. 冻结 feature，不更新 LocateAnything / ResNet backbone。
3. 在每一种 feature 上训练同一个小型 boundary probe head。
4. probe head 的任务只有一个：从 frozen feature 预测 GT contour boundary。
5. 用 `BF1@1px / BF1@2px / bIoU` 比较不同层的边界信息可读性。

F1 的结论是：

- Locate 896 的中后层明显比早层更适合边界细化。
- 单层最强的 1px 边界层是 `layer_18`。
- `layer_26` 的精确 1px 边界不如 `layer_18`，但边界带重合 `bIoU` 最高，说明它更偏语义/区域一致性。
- 448 分辨率不够，Locate 的优势只有在 896 输入下才出现。

后续 F2/F3 下游消融证明：

```text
F2 layer_18 + layer_26 是当前最优 Locate feature 配置。
```

GT-init full test 结果：

| Feature | sample IoU | contour IoU |
|---|---:|---:|
| layer_18 + layer_26 | 0.83914 | 0.83794 |
| layer_26 only | 0.83065 | 0.82855 |
| layer_18 only | 0.82216 | 0.81929 |
| layer_18 + layer_22 | 0.80068 | 0.79643 |
| layer_22 only | 0.79985 | 0.79606 |

所以 F1 的作用是“选候选层、解释为什么要试 18/26”，最终配置是否真的好，靠 F2/F3 的下游 GT-init IoU 判定。

## 2. Why We Need This Probe

直接把 LocateAnything 特征塞进 DiffusionSnake 训练有几个问题：

1. 成本高。
   每个完整训练要跑数千到上万 step，占 GPU 时间。

2. 干扰因素多。
   最终 IoU 会同时受检测框、极点初始化、Snake/DiT 训练稳定性、adapter 学习能力、init 分布影响。

3. 不容易解释失败原因。
   如果直接端到端失败，不知道是 feature 本身没边界信息，还是下游训练没适配好。

F1 probe 的设计就是把问题拆开：

> 先不训练完整模型，只问 feature 本身是否包含可线性/浅层解码出来的边界信息。

这相当于一个 representation diagnostic。它不证明最终模型一定提升，但能快速排除明显差的层，并缩小后续完整训练的搜索空间。

## 3. Feature Sources

### 3.1 LocateAnything Features

提取脚本：

```text
scripts/f1_extract_locate_layers.py
```

Locate 模型权重：

```text
Eagle/Embodied/work_dirs/1232_final_locany_full_more10000/checkpoint-3000
```

提取层：

```text
layer_5
layer_9
layer_13
layer_18
layer_22
layer_26
```

这里的 layer index 是 MoonViT encoder 的 1-based 层号。

输入分辨率：

| Feature cache | Locate input | Patch size | Feature grid |
|---|---:|---:|---:|
| Locate 448 | long side 448 | 14 | about 32x32 |
| Locate 896 | long side 896 | 14 | about 64x64 |

每层特征：

```text
shape = 1152 x H x W
dtype = float16 in cache, float32 when loaded for probe training
```

缓存中还保存几何信息：

```text
orig_hw
resized_hw
padded_hw
input_hw
grid_hw
pad
scale
patch_size
image_path
```

这些元信息用于把原图 polygon 精确映射到 probe 的统一目标坐标系。

### 3.2 ResNet34 Baseline Feature

ResNet baseline 是当前稳定 Snake/DiT 使用的 heatmap detector feature。

配置：

```text
configs/e3_v8_2_boxjitter_mixinit_gpu7.yaml
```

权重：

```text
data/outputs/e3_v8_2_boxjitter_mixinit_gpu7/checkpoints/latest.pt
```

特征：

```text
ResNet34 stride-4 feature
shape = 64 x 128 x 128
```

这个特征作为当前基线，F1 的问题之一就是：

> Locate 特征在边界可读性上是否能超过这个 ResNet stride-4 特征？

## 4. Dataset Protocol

F1 使用：

```text
data root:  /home/medteam/Zhrch/Datasets/1232_final
jsonl root: Eagle/Embodied/locany_recipe/1232_final
```

训练集：

```text
train split 前 300 张
```

测试集：

```text
完整 test split，177 张
```

为什么只用 300 张训练？

因为 F1 是诊断实验，不是要训练一个强分割模型。少量训练集有两个好处：

1. 成本低，可以快速比较很多层。
2. 更能反映 feature 本身的信息可读性。

如果一个 feature 需要大量数据和复杂模型才能读出边界，它对当前 Snake/DiT adapter 的直接价值也会更可疑。

注意：

F1 在 test split 上做每个 epoch 的指标和 early stopping，所以绝对值不能当成严格论文 benchmark。它适合做横向比较，因为所有 feature 都遵循同一协议。

## 5. GT Boundary Construction

F1 的监督目标不是 mask 区域，而是 contour boundary。

每个样本先读取 GT polygon：

1. 优先从 jsonl 中读取 `polygon / polygons / segmentation / contour` 等字段。
2. 如果 jsonl 没有可用 polygon，则读取 mask 文件，用 `cv2.findContours` 提取轮廓。

随后将 polygon 映射到统一目标坐标：

```text
target_size = 128 x 128
```

### 5.1 Locate Feature Geometry

Locate cache 通常记录了 resize + pad 过程。

对原图 polygon 点 `(x, y)`：

```text
x' = x * scale + pad_left
y' = y * scale + pad_top
```

然后缩放到 128x128：

```text
x_target = x' * 128 / input_w
y_target = y' * 128 / input_h
```

### 5.2 ResNet Feature Geometry

ResNet feature cache 如果有 `trans_input` affine transform，则使用：

```text
p' = trans_input * p
```

再映射到 128x128。

### 5.3 Boundary Rasterization

映射后的 polygon 用 OpenCV 画成 1px 二值边界：

```text
cv2.polylines(..., isClosed=True, thickness=1)
```

输出 target：

```text
shape = 1 x 128 x 128
value = 0 or 1
```

这让 F1 专门测试很严格的细边界定位能力，而不是宽松的区域分割能力。

## 6. Probe Head Architecture

所有 feature 使用同一个 probe head 模板。

结构：

```text
Input frozen feature
  -> 1x1 Conv to 64 channels
  -> GroupNorm(8) + GELU
  -> 3x3 Conv
  -> GroupNorm(8) + GELU
  -> PixelShuffle upsample blocks if needed
  -> bilinear resize to 128x128 if needed
  -> 1x1 Conv to 1-channel boundary logits
```

对应代码：

```text
scripts/f1_boundary_probe.py
class BoundaryProbe
```

参数限制：

```text
probe_params < 0.5M
```

实际参数量：

| Feature | Params |
|---|---:|
| ResNet 512 baseline | 41,409 |
| Locate 896 single layer | 258,881 |
| Locate 448 single layer | 406,721 |

为什么要限制 probe 很小？

因为 probe 太强会把比较污染掉。F1 想测的是 feature 中边界信息是否容易读出来，而不是让一个大模型自己学边界检测。如果 probe 容量很大，它可能从弱特征里“补”出边界，导致层间差异变小。

## 7. Training Objective

训练时 frozen feature 不更新，只更新 probe head。

优化器：

```text
AdamW
lr = 1e-3
weight_decay = 1e-4
max_epochs = 20
patience = 5
seed = 20260613
threshold = 0.5
```

loss：

```text
loss = BCEWithLogitsLoss + DiceLoss
```

为什么用 BCE + Dice？

边界 mask 极度稀疏。128x128 图里，正样本边界像素只占很小比例。

- BCE 提供逐像素分类监督。
- Dice 更关注预测边界和 GT 边界的整体重合，缓解正负样本不平衡。

如果只用 BCE，模型容易受大量背景像素主导。如果只用 Dice，训练初期可能不够稳定。两者组合更稳。

## 8. Metrics

F1 使用三个指标：

```text
BF1@1px
BF1@2px
bIoU
```

### 8.1 BF1@1px

Boundary F-score with 1 pixel tolerance。

计算逻辑：

1. 将预测 boundary 和 GT boundary 都视作二值 mask。
2. 对 GT boundary 做 distance transform。
3. 每个 predicted boundary pixel 如果距离 GT boundary 不超过 1px，就算 precision hit。
4. 对 prediction boundary 做 distance transform。
5. 每个 GT boundary pixel 如果距离 prediction 不超过 1px，就算 recall hit。
6. 计算 precision、recall 和 F1。

公式：

```text
precision = matched_pred_pixels / pred_boundary_pixels
recall    = matched_gt_pixels / gt_boundary_pixels
BF1       = 2 * precision * recall / (precision + recall)
```

为什么 BF1@1px 最重要？

因为 Snake/DiT 的任务是轮廓细化，最终误差往往体现在边界偏移。1px 容差非常严格，能测试 feature 是否包含精细定位信息。

### 8.2 BF1@2px

BF1@2px 和 BF1@1px 一样，只是容差放宽到 2px。

它更关注边界大方向是否对，而不是 1px 级别是否完全对齐。

F1 用 `BF1@2px` 做 early stopping，原因是：

- 训练早期 1px 指标波动更大。
- 2px 指标更稳定，更适合作为 stopping score。
- 最终排序仍主要看 BF1@1px。

### 8.3 bIoU

Boundary IoU。

计算逻辑：

1. 将 prediction boundary 和 GT boundary 各自 dilation 2 次。
2. 计算两个 dilated boundary band 的 IoU。

公式：

```text
bIoU = area(pred_boundary_band intersection gt_boundary_band)
       / area(pred_boundary_band union gt_boundary_band)
```

bIoU 比 BF1@1px 更宽松，更偏区域带重合。

它能反映：

- 边界整体覆盖是否稳定；
- 语义区域是否一致；
- 局部 1px 错位之外的轮廓带质量。

这也是为什么 `layer_26` 虽然 BF1@1px 不是最高，但仍值得进入 F2：它的 bIoU 最高。

## 9. Main F1 Results

关键结果：

| Feature | BF1@1px | BF1@2px | bIoU |
|---|---:|---:|---:|
| Locate 896 layer_18 | 0.9083 | 0.9518 | 0.8357 |
| Locate 896 layer_22 | 0.9051 | 0.9488 | 0.8293 |
| Locate 896 layer_26 | 0.9012 | 0.9471 | 0.8410 |
| ResNet34 stride-4 baseline | 0.9005 | 0.9453 | 0.8291 |
| Locate 448 layer_18 | 0.8949 | 0.9470 | 0.8254 |

解释：

1. `layer_18` 是最强 1px 精细边界层。

```text
Locate 896 layer_18 BF1@1px = 0.9083
ResNet baseline       BF1@1px = 0.9005
delta                 = +0.0078
```

2. `layer_26` 是最强边界带/区域一致性层。

```text
Locate 896 layer_26 bIoU = 0.8410
Locate 896 layer_18 bIoU = 0.8357
```

3. 分辨率是硬门槛。

```text
Locate 448 layer_18 BF1@1px = 0.8949
Locate 896 layer_18 BF1@1px = 0.9083
```

同一层从 448 到 896，边界可读性明显提升。Locate 的 patch size 是 14，输入过低时 token grid 太粗，边界细节会丢。

4. 早层不适合。

`layer_5 / layer_9` 明显弱，说明早层 token 还没有形成对医学器官边界足够稳定的表达。早期 `layer_9 + layer_18` 组合失败，这个结果能解释一部分原因。

## 10. Principle: What The Probe Measures

F1 probe 本质上测的是：

```text
boundary information decodability from frozen feature
```

也就是：

> 在不改 backbone 的情况下，一个很小的 decoder 能不能从该层 feature 里恢复 GT 边界？

如果一个 feature 的 probe 分数高，说明该 feature 中包含的边界信息满足两个条件：

1. 信息存在。
   GT 边界相关信号确实编码在 feature 里。

2. 信息容易读取。
   不需要很复杂的下游网络就能读出来。

这对 Snake/DiT 很重要，因为我们不是要从零训练一个超大 decoder，而是要把 feature 接到已有细化分支里。越容易被小 probe 读出来的边界信息，越可能被 adapter + DiT 有效利用。

## 11. Why F1 Does Not Directly Select The Final Config

F1 只测试单层 feature 的边界可读性，不等价于完整模型最终 IoU。

完整模型还会受这些因素影响：

1. 多层拼接后是否互补。
2. adapter 是否能学好 2304ch 到 64ch 的映射。
3. feature 是否适配当前 Snake/DiT 的训练分布。
4. GT-init 和 predicted-init 分布是否一致。
5. 检测框和极点头是否引入额外误差。
6. 长训是否稳定，是否过拟合或退化。

所以 F1 的正确角色是：

```text
F1 = layer candidate screening
F2/F3 = downstream validation
```

F1 说明 `layer_18` 和 `layer_26` 值得试；但 F2/F3 的 GT-init full test 才说明 `18+26` 是当前最优配置。

## 12. How F1 Led To F2

F1 给出的线索：

| Finding | Implication |
|---|---|
| 896 明显好于 448 | 后续只用 Locate 896 cache |
| layer_18 BF1@1px 最高 | 必须试 layer_18 |
| layer_26 bIoU 最高 | 值得作为语义/区域补充层 |
| layer_9 很弱 | 不再优先用 layer_9 |

因此 F2 配置是：

```yaml
locate_feat_keys: ['layer_18', 'layer_26']
locate_feat_dim: 2304
```

拼接方式：

```text
layer_18: 1152 channels
layer_26: 1152 channels
concat:   2304 channels
```

然后通过 `LocateFeatReplacer` 映射为 Snake/DiT 使用的 stride-4 64ch feature。

## 13. F2/F3 Downstream Validation

后续在同一 GT-init full test 口径下比较：

| Experiment | Feature | sample IoU | contour IoU |
|---|---|---:|---:|
| F2 | layer_18 + layer_26 | 0.83914 | 0.83794 |
| F3d | layer_26 only | 0.83065 | 0.82855 |
| F3a | layer_18 only | 0.82216 | 0.81929 |
| F3b | layer_18 + layer_22 | 0.80068 | 0.79643 |
| F3c | layer_22 only | 0.79985 | 0.79606 |

这个结果说明：

1. `layer_18` 的 probe 分数最高，但单独作为下游 feature 不够。
2. `layer_26` 单独比 `layer_18` 单独更强，说明它的语义/区域一致性对 Snake/DiT 很关键。
3. `layer_18 + layer_26` 最强，说明两者确实互补。
4. `layer_22` 在 F1 中看起来不错，但下游表现很差，说明 F1 不能代替完整验证。

最终推荐：

```text
Locate 896 layer_18 + layer_26
```

## 14. Limitations

F1 的限制必须明确：

1. 它不是最终分割模型。
   Probe head 只预测 128x128 boundary，不输出完整 polygon。

2. 它不评估检测和初始化。
   GT-init 下游表现仍要单独验证。

3. 它主要测试单层。
   多层组合是否互补，必须靠 F2/F3。

4. 绝对分数略乐观。
   因为 early stopping 使用 test split 指标。

5. Probe 参数量不同。
   ResNet / Locate 896 / Locate 448 的 probe 参数量不同，但都被限制在 0.5M 以下。

6. 目标分辨率是 128x128。
   这匹配当前 stride-4 refinement feature，但不是原图全分辨率。

## 15. Practical Takeaways

1. F1 的设计是合理的，因为它隔离了 feature 本身的边界信息，避免直接端到端训练时被太多因素干扰。

2. F1 的主指标是 `BF1@1px`，因为轮廓细化最关心精确边界定位。

3. `bIoU` 不能忽略，因为它能发现更语义、更区域一致的层，比如 `layer_26`。

4. F1 不能单独决定最终配置，只能决定候选。

5. 当前最终配置不是“F1 直接选出来的”，而是：

```text
F1 发现 layer_18 精细边界强、layer_26 区域一致性强
  -> F2 尝试 layer_18 + layer_26
  -> F3 消融验证 18+26 明确优于 18 only / 26 only / 18+22 / 22 only
  -> 最终推荐 layer_18 + layer_26
```

## 16. One-Sentence Explanation

F1 探针就是用一个受限的小边界解码器，在冻结特征上预测 128x128 GT 轮廓边界；它通过 `BF1@1px / BF1@2px / bIoU` 衡量每层特征的边界信息是否“存在且容易被读出”，从而低成本筛出值得进入完整 Snake/DiT 训练的 Locate 层。
