# VolMem 体数据主线命名与代码边界

## 1. 方法定位

- 论文任务：3D volumetric medical image segmentation。
- 代码机制：slice-sequential volumetric segmentation with streaming memory。
- 工作名：VolMemSnake。
- 代码前缀：`volmem`。

VolMemSnake 逐张处理二维切片，并通过一套可学习的 Slice Memory 在切片之间传递信息。它不是 3D CNN、3D Transformer、三帧通道堆叠或 mesh/surface 网络。

## 2. 强制命名

| 对象 | 统一命名 |
|---|---|
| 主网络 | `VolMemSnake` |
| 网络任务 ID | `volmem` |
| 数据包 | `volmem_sequence` |
| 数据集类 | `VolMemDataset` |
| 序列样本 | `SliceSequenceSample` |
| Memory 编码器 | `SliceMemoryEncoder` |
| Memory 读取模块 | `SliceMemoryAttention` |
| Memory 状态库 | `SliceMemoryBank` |
| 单个状态 | `SliceMemoryState` |
| 当前切片特征 | `slice_features` |
| Memory 增强特征 | `memory_conditioned_features` |
| 体数据标识 | `volume_id` |
| 切片序号 | `slice_index` |
| 物理位置 | `slice_position_mm` |

不得使用 `frame`、`video`、`temporal` 表示医学切片；它们只能在引用 SAM 2 原论文时出现。

## 3. 目录边界

```text
volmem/models/                  VolMemSnake 和 Slice Memory
volmem/data/                    按 volume 组织的切片序列数据
volmem/engine/                  新主线训练逻辑，后续创建
configs/volmem/                 新主线配置
tools/volmem/                   新主线训练、评估与检查入口
tests/volmem/                   新主线测试
data/outputs/volmem/            新主线输出，后续使用
```

新主线使用顶层独立包 `volmem/`，不挂载到旧 `lib.datasets` 的导入链或 `lib.networks.snake` 的动态工厂中。旧目录 `lib/networks/snake/` 和 `lib/datasets/sagittal_2d_fixed/` 不再接收新的跨切片条件分支。V4.6c 后续通过显式 adapter 作为冻结的 2D 基座接入，但旧目录不得继续承载 VolMem 逻辑。

## 4. 禁止词与禁用机制

新主线的文件名、类名、配置键和输出目录禁止出现：

```text
pseudo3d
three_frame
three_slice_input
neighbor_mean
neighbor_fusion
prev_contour
previous_contour
video_frame
temporal_memory
true_3d
final
```

其中 `final` 被禁止，是因为未经完整验证的实验不能通过名称宣称成熟。

新主线首版明确排除：

- 三帧或多帧通道堆叠；
- 邻层特征均值或拼接；
- previous-contour 初始化旁路；
- 独立 contour memory；
- 与 Slice Memory 并列的跨层融合机制；
- V11 LocateToken-DiT 旁路；
- 3D convolution、3D attention 和 mesh/surface refinement。

## 5. 成熟度规范

每个配置必须显式声明 `maturity`：

| maturity | 含义 | 是否允许正式训练 |
|---|---|---|
| `scaffold` | 只有接口、命名和契约 | 否 |
| `smoke` | 通过合成数据前向/反向 | 否 |
| `prototype` | 可运行真实短序列实验 | 仅实验 |
| `baseline` | 完成固定协议评估 | 是 |
| `validated` | 完成主数据集和消融复核 | 是 |

版本命名示例：

```text
verse_volmem_v0_1_scaffold.schema.yaml
verse_volmem_v0_2_smoke.yaml
verse_volmem_v0_3_prototype.yaml
verse_volmem_v1_0_baseline.yaml
```

只有达到 `baseline` 后才允许成为后续实验的默认父配置。

## 6. 首版唯一数据流

```text
slice_image
  -> MoonViT(layer_18, layer_26)
  -> per-layer normalization + LocateFeatReplacer
  -> half-pixel aligned slice_features
  -> SliceMemoryAttention(slice_features, memory_bank)
  -> V4.6c contour flow
  -> contour_prediction
  -> SliceMemoryEncoder(slice_features, contour_prediction)
  -> memory_bank.append(state)
```

跨切片信息只能通过 `SliceMemoryAttention / SliceMemoryEncoder / SliceMemoryBank` 三类接口流动。任何其他读取邻层图像、轮廓或特征的代码都违反主线契约。

## 7. 数据契约

数据加载器按 `volume_id` 分组，并按真实物理位置排序。最小字段：

```text
volume_id
slice_index
slice_position_mm
slice_image
mask_target
contour_target
moonvit_layer_18
moonvit_layer_26
is_first_slice
sequence_direction
```

`sequence_direction` 只允许 `ascending` 或 `descending`；不得命名为视频传播方向。训练集划分必须以 `volume_id` 为单位，禁止切片级随机拆分造成泄漏。

## 8. 与旧代码的关系

- 旧 2D V4.6c：保留，不改名，不改变行为。
- 旧三帧 pseudo-3D：归档实验，不作为 VolMemSnake 的前身或基线。
- V11 LocateToken-DiT：归档探索，不进入 VolMemSnake 首版。
- VolMemSnake：新建、独立版本、独立输出、独立评估。

任何从旧代码复制的实现都必须在新命名空间中显式注明来源，并删除旧的跨切片启发式分支。