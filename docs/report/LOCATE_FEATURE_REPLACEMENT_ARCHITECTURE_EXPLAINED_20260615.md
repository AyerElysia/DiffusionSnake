# Locate Feature Replacement Architecture Explained

Date: 2026-06-15

## 1. Executive Summary

当前 F2/F4 使用的 LocateAnything 特征方案，严格来说不是“额外 feature injection”，而是：

> 用 Locate 896 的 `layer_18 + layer_26` 特征，替换 Snake/DiT 细化分支原来使用的 ResNet stride-4 feature map。

当前配置：

```yaml
locate_feat_inject: false
locate_feat_replace: true
locate_feat_keys: ['layer_18', 'layer_26']
locate_feat_dim: 2304
locate_feat_cache_root: data/f1_cache/locate896_full/locate_896
```

整体数据流：

```text
Image
  -> Heatmap ResNet detector
       -> detection heatmap / wh / mask / extreme init
       -> det_cnn_feature, still ResNet

Cached Locate 896 layer_18 + layer_26
  -> concat: 1152 + 1152 = 2304 channels, 64x64
  -> LocateFeatReplacer
       2304x64x64 -> 64x128x128
  -> geometry alignment with current input transform
  -> snake_feature
  -> FlowMatchingEvolution / DiT refinement
       -> global context from whole feature map
       -> local point context from contour-point sampling
       -> detail normal-band context from nearby point sampling
```

最重要的一点：

> 当前方案没有把 Locate 特征分成两条显式分支分别注入“全局语义”和“点局部语义”。它先把 Locate 特征转换成一张统一的 64ch stride-4 feature map，然后沿用原有 DiT 架构，在 DiT 内部从这张 feature map 中分别构造 global context 和 local point context。

也就是说，全局语义和局部语义的分工还在 DiT 里完成，只是底层 feature source 从 ResNet 换成了 Locate。

## 2. Current Config

当前最优 Locate 配置是 F2：

```text
configs/f2_locate_feat_replace_gpu0.yaml
```

关键开关：

```yaml
detector_backend: 'heatmap_resnet'

locate_feat_inject: false
locate_feat_replace: true
locate_feat_cache_dir: "data/f1_cache/locate896_full/locate_896"
locate_feat_cache_root: "data/f1_cache/locate896_full/locate_896"
locate_feat_keys: ['layer_18', 'layer_26']
locate_feat_dim: 2304

use_diffusion_evolution: true
use_flow_matching: true
use_dit_v4_1: true
v4_1_use_detail_context: true
v4_1_detail_context_mode: 'normal'
v4_1_final_head_type: 'moe'
```

含义：

- detection / heatmap / mask / extreme-point head 仍由 `heatmap_resnet` 跑；
- Snake/DiT 细化分支的 `cnn_feature` 被 Locate feature replacement 替换；
- DiT 使用 V4.1 Flow Matching；
- detail context 开启，模式是 normal direction sampling；
- final displacement head 是 MoE head。

## 3. What Is Being Replaced

原始 heatmap path：

```python
cnn_feature, ct_hm, wh, mask_logits = self.heatmap_detector(x)
```

这里的 `cnn_feature` 是 ResNet/FPN 风格的 stride-4 64ch feature，尺寸通常是：

```text
64 x 128 x 128
```

在普通 ResNet baseline 中，这个 feature 同时服务于：

1. detection heatmap；
2. extreme-point prediction；
3. Snake/DiT contour refinement。

当前 Locate replacement 只替换第 3 项。

代码逻辑：

```python
det_cnn_feature = cnn_feature
snake_feature, locate_feat_stats = self.apply_locate_feature_injection(det_cnn_feature, batch)
snake_feature, replace_stats = self.apply_locate_feature_replacement(snake_feature, batch)
...
output = self.attach_extreme_prediction(output, det_cnn_feature, batch)
...
output = self.gcn(output, snake_feature, batch)
```

关键区别：

| Module | Feature source |
|---|---|
| detection heatmap | ResNet feature |
| wh head | ResNet feature |
| mask head | ResNet feature |
| extreme-point head | ResNet `det_cnn_feature` |
| Snake/DiT refinement | Locate-replaced `snake_feature` |

所以当前不是全模型 Locate-only。当前是：

```text
ResNet handles detection/init side.
Locate handles contour refinement feature side.
```

在 GT-init 评测里，检测影响被隔离，主要考察 Locate feature 对 Snake/DiT refinement 的作用。

## 4. Dataset Loading Of Locate Features

Dataset 在 `locate_feat_replace=True` 时，会为每张图读取缓存：

```text
data/f1_cache/locate896_full/locate_896/{train|test}/{image_stem}.npz
```

F2 使用：

```yaml
locate_feat_keys: ['layer_18', 'layer_26']
```

读取后：

```python
arrays = [npz['layer_18'], npz['layer_26']]
feat = np.concatenate(arrays, axis=0)
```

shape：

```text
layer_18: 1152 x 64 x 64
layer_26: 1152 x 64 x 64
concat:   2304 x 64 x 64
```

同时 batch 里还带上几何元信息：

```text
locate_feat_grid_hw
locate_feat_orig_hw
locate_feat_resized_hw
locate_feat_padded_hw
locate_feat_pad
locate_feat_scale
locate_feat_patch_size
locate_feat_path
```

这些信息用于把 Locate feature grid 和当前训练输入坐标对齐。

## 5. LocateFeatReplacer

当前替换头定义在：

```text
lib/networks/snake/ct_snake.py
class LocateFeatReplacer
```

结构：

```text
Input:  2304 x 64 x 64
  -> 1x1 Conv: 2304 -> 256
  -> GroupNorm(16) + GELU
  -> 3x3 Conv: 256 -> 256
  -> GroupNorm(16) + GELU
  -> PixelShuffle(2): 256 x 64 x 64 -> 64 x 128 x 128
  -> 3x3 Conv: 64 -> 64
Output: 64 x 128 x 128
```

参数量约：

```text
1.218M
```

为什么要这样做？

1. Locate 896 token grid 是 64x64。
2. Snake/DiT 细化分支原来吃的是 stride-4 128x128 feature。
3. 所以需要把 64x64 上采样到 128x128。
4. 同时把 2304ch 压缩到原架构期望的 64ch。

PixelShuffle 的作用是：

```text
spatial upsample x2, channel reduce / rearrange
```

相比简单 bilinear upsample，它给 adapter 一个可学习的空间重排能力。

## 6. Geometry Alignment

Locate feature 不是直接 resize 后就用。它需要对齐当前 Snake 输入坐标。

替换流程：

```python
replaced = self.locate_feat_replacer(feat)
grid = self._build_locate_feature_grid(..., source_scale=2.0)
replaced = F.grid_sample(replaced, grid, ...)
return replaced
```

为什么还要 `grid_sample`？

因为 Locate cache 的特征坐标来自 Locate 的 resize + pad 过程，而当前训练 batch 里的 image input 也可能经过 affine transform、flip、resize 等处理。两者不是天然一一对应。

`_build_locate_feature_grid` 做的事：

1. 先枚举目标 `128x128` feature map 上每个位置。
2. 用 `meta.inv_trans_input` 映射回原图坐标。
3. 如果训练时 flip，则修正 x 坐标。
4. 用 Locate cache 的 `scale + pad` 映射到 Locate 输入坐标。
5. 再除以 patch size，得到 Locate token/upsampled-feature 坐标。
6. 归一化到 `[-1, 1]`，供 `F.grid_sample` 使用。

因为 `LocateFeatReplacer` 已经把 64x64 上采样到 128x128，所以调用时：

```python
source_scale = 2.0
```

直观理解：

```text
original image coordinate
  -> current Snake input coordinate
  -> Locate resized+padded coordinate
  -> Locate upsampled feature coordinate
```

这一步保证 `snake_feature[y, x]` 对应的是同一张图中同一解剖位置附近的 Locate 表达。

## 7. Current "Injection" Is Actually Replacement

代码里有两个不同机制：

### 7.1 locate_feat_inject

旧的 residual injection：

```text
LocateFeatAdapter
Locate feature -> 64ch residual -> add to ResNet cnn_feature
```

形式：

```python
return cnn_feature + aligned
```

这个模式当前关闭：

```yaml
locate_feat_inject: false
```

### 7.2 locate_feat_replace

当前使用的 replacement：

```text
LocateFeatReplacer
Locate feature -> 64ch feature map -> replace Snake/DiT cnn_feature
```

形式：

```python
return replaced
```

这个模式当前开启：

```yaml
locate_feat_replace: true
```

所以准确说法是：

```text
当前不是 ResNet feature + Locate residual。
当前是 Snake refinement branch 的 feature map 整体换成 Locate-derived feature map。
```

## 8. How Global And Local Semantics Work Now

你之前的架构里，本来就有两个层次：

1. 全局语义：让每个 contour 知道整张图/整个器官上下文。
2. 点轮廓局部语义：让每个 contour point 知道自己附近的局部边界/纹理。

当前 Locate replacement 保留了这个设计，只是把底层 feature source 换成了 Locate。

### 8.1 Unified Feature Map

进入 DiT 前，只有一张统一的 feature map：

```text
snake_feature: B x 64 x 128 x 128
```

这张图已经是 Locate-derived。

它同时承担：

- global context source；
- local point context source；
- detail normal context source。

### 8.2 Global Semantic Context

DiT 里有：

```python
global_ctx = self.global_compressor(cnn_feature)
```

`global_compressor` 是 Perceiver-style compressor：

```text
input:  B x 64 x 128 x 128
output: B x num_queries x state_dim
```

原始 V3 代码注释里写的是：

```text
Global Context: Perceiver IO, 256 learnable queries
```

含义：

1. 把整张 2D feature map 当作全局 token memory。
2. 用一组 learnable queries 从整图 feature 中压缩出 global tokens。
3. 对每条 contour，用 `py_ind` 取出对应图像的 global context。

这个 global context 负责提供：

- 当前器官处于图像中的大致区域；
- 周围结构；
- 全局形状先验；
- 类别/语义稳定性；
- 避免只看局部边界导致轮廓漂移。

在当前 F2/F4 下，这些 global tokens 已经来自 Locate-derived feature map，而不是 ResNet feature map。

### 8.3 Point Local Context

每条 contour 有 128 个点。FlowMatchingEvolution 会在当前 contour 点位置采样 feature：

```python
sampled_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_py, py_ind, h, w)
```

shape：

```text
sampled_feat: N_contours x 64 x P
P = 128 points
```

DiT 里再投影成 local context：

```python
local_ctx = self.local_proj(sampled_feat.transpose(1, 2))
```

shape：

```text
local_ctx: N_contours x P x state_dim
```

这个 local context 是点级别的：

```text
point i gets feature sampled at contour point i
```

它负责告诉模型：

- 这个点附近是不是边界；
- 往内/往外哪边更像器官；
- 这个点是否在局部凹凸、尖角、细长结构附近；
- 当前点应该如何移动。

当前 F2/F4 下，`sampled_feat` 也是从 Locate-derived 64ch map 上采样得到。

### 8.4 Detail Normal Context

当前配置还开启：

```yaml
v4_1_use_detail_context: true
v4_1_detail_context_mode: 'normal'
```

FlowMatchingEvolution 会沿 contour 法线方向采样更多点：

```text
plus_1  = feature(point + normal * radius_1)
minus_1 = feature(point - normal * radius_1)
plus_2  = feature(point + normal * radius_2)
minus_2 = feature(point - normal * radius_2)
```

然后构造 detail terms：

```text
plus_1 - minus_1
plus_2 - minus_2
0.5 * (plus_1 + minus_1) - sampled_feat
```

这些 detail terms 会进入：

```python
local_ctx = local_ctx + self.detail_local_proj(detail_ctx)
x = x + self.detail_point_proj(detail_ctx)
```

含义：

- `plus - minus` 提供内外侧差异；
- 多尺度 normal sampling 提供边界穿越方向的梯度信息；
- `mid - center` 提供局部中心点和邻域的关系。

这部分很接近“点轮廓的局部语义/局部几何”。

当前它同样基于 Locate-derived feature map，因此 Locate 的边界表达不只是点上采样一次，还会被法线方向局部比较使用。

## 9. How DiT Combines Global And Local Context

DiT V4.1 forward 里每层交替使用 global context 和 local context：

```python
for i, dit_layer in enumerate(self.dit_layers):
    context = global_ctx if (i % 2 == 0) else local_ctx
    x = dit_layer(x, context, t_emb)
```

也就是说：

```text
DiT layer 0 uses global context
DiT layer 1 uses local context
DiT layer 2 uses global context
DiT layer 3 uses local context
...
```

这个设计保留了原架构的语义分工：

| Context | Source now | Role |
|---|---|---|
| global_ctx | whole Locate-derived feature map | 全局器官/区域/形状语义 |
| local_ctx | feature sampled at contour points | 点级局部边界语义 |
| detail_ctx | feature sampled along normal offsets | 局部内外侧差异和边界梯度 |

区别只是底层 source：

```text
Before: ResNet feature map
Now: Locate layer_18+26 -> replacer -> 64ch feature map
```

## 10. What Happens To The Point Embedding

DiT 的 point token 初始化是：

```python
x = self.point_embed(x_t, sampled_feat)
```

其中：

- `x_t` 是当前 flow matching state，也就是当前要预测的 normalized displacement state；
- `sampled_feat` 是每个 contour point 位置采到的 64ch feature。

当前 `sampled_feat` 已经来自 Locate-derived map。

所以每个 point token 一开始就带有：

1. 当前点的几何状态；
2. 当前点位置的 Locate 局部视觉/边界特征。

之后再通过 DiT cross-attention 交替吸收 global context 和 local context。

## 11. Why F2 Uses layer_18 + layer_26

F1 探针给出的解释：

| Layer | Strength |
|---|---|
| layer_18 | 最强 1px 边界定位，BF1@1px 最高 |
| layer_26 | 最强 boundary-band overlap，bIoU 最高 |

所以 `layer_18 + layer_26` 的设计意图是：

```text
layer_18 -> local sharp boundary cue
layer_26 -> higher-level semantic / region consistency cue
```

拼接后通过 `LocateFeatReplacer` 学习如何把二者压成 64ch feature map。

后续 F3 消融验证了这一点：

| Feature | sample IoU | contour IoU |
|---|---:|---:|
| layer_18 + layer_26 | 0.83914 | 0.83794 |
| layer_26 only | 0.83065 | 0.82855 |
| layer_18 only | 0.82216 | 0.81929 |
| layer_18 + layer_22 | 0.80068 | 0.79643 |
| layer_22 only | 0.79985 | 0.79606 |

结论：

```text
layer_18 and layer_26 are complementary for downstream Snake/DiT.
```

`layer_18` 单独精细但不够稳定；`layer_26` 单独更稳但不如组合；组合最好。

## 12. What Is Not Happening In Current F2/F4

### 12.1 Not Using V11 Locate Token Path

代码里还有一套 `use_locate_token_dit` 机制。它会构造：

```text
locate_point_ctx
locate_global_ctx
locate_only
```

并把 raw Locate token 作为 separate point/global context 传给 DiT。

但当前 F2/F4 没有启用：

```yaml
use_locate_token_dit: false
```

所以当前不是：

```text
raw Locate point tokens -> DiT point context
raw Locate global tokens -> DiT global context
```

当前是：

```text
raw Locate layers -> 64ch replacement feature map -> original DiT global/local mechanisms
```

### 12.2 Not Adding Locate Residual On Top Of ResNet

`locate_feat_inject` 也没有启用。

所以当前不是：

```text
ResNet feature + Locate residual
```

而是：

```text
Snake/DiT feature = Locate-derived replacement feature
```

### 12.3 Not Replacing Detector Feature

当前 detection side 仍是 heatmap ResNet。

这意味着：

- F2/F4 主要验证 Locate feature 对 contour refinement 是否有用；
- LocateAnything detection boxes 是另一条链路；
- end-to-end 使用 LocateAnything 检测框时，还需要解决 init distribution mismatch。

## 13. Why This Design Is Conservative

当前设计的优点：

1. 对原 DiT 架构侵入小。
   不改 DiT 的 global/local context 结构，只换 feature source。

2. 保留 checkpoint 复用能力。
   输出仍是 64ch 128x128，和原 ResNet feature 接口一致。

3. 保留原有全局/局部分工。
   Global compressor 和 point sampling 都还能工作。

4. 避免 raw Locate token 直接改 DiT 接口。
   先通过小 replacer 学一个适配层，更稳定。

5. 能做清晰消融。
   只改 `locate_feat_keys` 就能比较不同层组合。

代价：

1. Locate 的 raw token 结构被压缩到 64ch，可能丢信息。
2. 全局语义和局部语义不是显式分支，都是从同一张 64ch map 里再分离。
3. Replacer 需要自己学会如何融合 `layer_18` 和 `layer_26`。
4. Detection/init side 仍不是 Locate-only。

## 14. Conceptual View

可以把当前方案理解成三层适配：

### Level 1: Feature Source Adaptation

```text
Locate layer_18 + layer_26
  -> 2304ch semantic-boundary feature
  -> LocateFeatReplacer
  -> 64ch Snake-compatible feature
```

作用：

```text
把 Locate 的 token feature 翻译成旧 Snake/DiT 能读的 feature map。
```

### Level 2: Context Decomposition

```text
64ch feature map
  -> global_compressor -> global_ctx
  -> contour point sampling -> local_ctx
  -> normal offset sampling -> detail_ctx
```

作用：

```text
把统一 feature map 分解成全局语义、点局部语义、法线局部细节。
```

### Level 3: Flow Matching Refinement

```text
current contour state x_t
  + point embedding
  + alternating global/local DiT context
  -> velocity / displacement prediction
  -> refined contour
```

作用：

```text
根据全局语义和局部边界信息，预测每个点应该往哪里移动。
```

## 15. Relation To The Old Architecture

旧架构的核心思想：

```text
global semantic context + point-level local context
```

当前 Locate replacement 没有废掉这个思想。

它只是把：

```text
ResNet feature map
```

换成：

```text
Locate-derived feature map
```

然后旧的全局/局部语义拆分仍然存在：

```text
global semantic = Perceiver compressor over full Locate-derived map
point local semantic = sampled Locate-derived feature at contour points
detail local semantic = normal-direction sampled Locate-derived feature differences
```

所以你可以这样理解：

> 以前是 ResNet 负责提供全局和局部语义素材；现在是 Locate 负责提供素材，但“全局怎么提取、局部怎么采样、DiT 怎么交替使用”的机制基本沿用原架构。

## 16. Why This May Work

F1 表明 Locate 896 的中后层确实含有更强边界信息：

```text
layer_18: sharp boundary cue
layer_26: semantic / region-consistent cue
```

Snake/DiT 恰好需要这两类信息：

- 点局部移动需要 sharp boundary cue；
- 整体轮廓稳定需要 semantic / region context。

当前 `layer_18 + layer_26` 通过同一张 replacement feature map 同时服务这两件事，因此比单层更好。

F3 消融也验证：

```text
18+26 > 26 only > 18 only > 18+22 ~= 22 only
```

这说明 `layer_26` 对下游不是拖后腿，而是提供了 `layer_18` 没有的稳定语义。

## 17. Remaining Weaknesses

当前方案还有几个弱点：

1. 全局和局部没有显式使用不同 Locate 层。
   现在是先融合 18+26，再由 DiT 从同一张 64ch map 里拆全局/局部。

2. Replacer 没有监督它如何分配 18/26。
   它只通过最终 diffusion loss 学习。

3. `layer_18` 和 `layer_26` 的角色没有硬编码。
   理论上更理想的是：

```text
layer_18 -> point/local/detail context
layer_26 -> global context
```

当前不是这样显式分配。

4. Detection side 还没有完全 Locate 化。
   LocateAnything boxes 进来后，init distribution mismatch 仍要处理。

5. 长训可能退化。
   原 F2 在 `step_58000` 最好，后面一些 checkpoint 有下降，所以现在 F4 用低 LR 长训并做 checkpoint sweep。

## 18. Possible Next Architecture

如果要进一步升级，可以考虑显式分路：

```text
layer_26
  -> global compressor
  -> global_ctx

layer_18
  -> point sampler + normal detail sampler
  -> local_ctx / detail_ctx
```

这会更符合 F1/F3 的观察：

- `layer_18` 更适合精确边界；
- `layer_26` 更适合区域一致性和语义。

但这需要改 DiT 接口，不再只是替换 64ch feature map。工程风险更高，因此当前 F2/F4 先采用保守 replacement。

## 19. One-Sentence Explanation

当前 Locate 特征方案是：把 `layer_18 + layer_26` 的 2304ch Locate 896 token feature 通过 `LocateFeatReplacer` 翻译成一张 64ch stride-4 feature map，替换 Snake/DiT 的细化特征；随后原 DiT 架构从这张 Locate-derived map 中分别构造全局语义 context、点局部 context 和法线 detail context，用于预测轮廓点的 flow displacement。
