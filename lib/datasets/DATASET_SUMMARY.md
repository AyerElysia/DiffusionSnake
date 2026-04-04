# DiffusionSnake Dataset 模块详解

## 目录结构与入口
- `__init__.py` 仅导出 `make_data_loader` 作为数据加载入口。
- `dataset_catalog.py` 维护数据集名字到路径/标注/划分的映射。
- `make_dataset.py` 负责根据 `cfg` 动态创建 Dataset、Sampler、DataLoader。
- `transforms.py` 提供图像归一化流程。
- `collate_batch.py` 定义 Snake/DSnake 的 batch 拼接逻辑。
- `samplers.py` 提供迭代控制与可变尺寸采样器。
- `voc/snake.py` 是主要训练数据集实现。
- `sbd/snake.py` 直接复用 `voc/snake.Dataset`。
- `voc_test/snake.py` 是推理/验证用的 COCO 风格 dataset。
- `coco/` 目录为空（当前无实现）。

## DatasetCatalog：数据集注册
- `DatasetCatalog.dataset_attrs` 通过名字映射到：`id`（数据源目录名）、`data_root`、`ann_file`、`split`。
- `DatasetCatalog.get(name)` 返回字典副本，供 `make_dataset` 使用。
- 例如：`SbdTrain` → `id=sbd`，`split=train`；`SbdMedicalTrain` 用自定义 `data_root` + `dummy_annotations.json`。

## make_dataset.py：构建数据加载流程
1. **动态加载 Dataset 类**
   - `_dataset_factory(data_source, task)` 通过 `imp.load_source` 从 `lib/datasets/{data_source}/{task}.py` 加载模块。
   - 返回模块内的 `Dataset` 类（不是实例）。
2. **创建 Dataset 实例**
   - `make_dataset(cfg, dataset_name, transforms, is_train)` 读取 `DatasetCatalog` 返回的参数。
   - 删除 `id` 字段后，以关键字参数实例化 `Dataset(**args)`。
3. **Sampler / BatchSampler**
   - `make_data_sampler`：训练用 `RandomSampler`，测试用 `SequentialSampler`。
   - `make_batch_data_sampler`：基于 `BatchSampler`，可选 `IterationBasedBatchSampler` 控制最大迭代数。
4. **DataLoader**
   - `make_data_loader` 根据 `cfg.train/test` 设置 batch size、workers、shuffle。
   - 使用 `make_collator(cfg)` 决定 batch 拼接方式。

## transforms.py：图像预处理
- `Compose` 顺序执行变换，支持 `(img, kpts)` 或只传 `img`。
- `ToTensor`：仅执行 `img / 255`，不转为 torch tensor。
- `Normalize`：按 ImageNet 均值/方差做归一化。
- 训练/测试使用同一组变换。

## samplers.py：采样器
- `ImageSizeBatchSampler`：为 batch 随机生成 `(h, w)`，用于多尺度训练（当前未在 `make_data_loader` 中使用）。
- `IterationBasedBatchSampler`：限制最大迭代次数，用于训练收敛控制。

## collate_batch.py：Batch 拼接细节
### snake_collator（核心）
- 汇总字段：`inp`、`img_path`、`meta` 走 `default_collate`；`orig_img` 保留为 list（避免尺寸不一致）。
- **`max_ct_num` 截断**：读取 `cfg.train.max_ct_num`，对每张图只保留最多 `n` 个 contour。
- **检测分支**：
  - `ct_hm`：按 batch 直接堆叠。
  - `wh/ct_cls/ct_ind`：对齐到 `ct_num=max(meta['ct_num'])`，用 `ct_01` 标记有效条目。
- **Snake 初始化/进化分支**：
  - 使用 `snake_config` 的 `init_poly_num/poly_num/gt_poly_num` 创建目标张量。
  - 通过 `ct_01` 把各样本的多边形列表填充到固定形状。
- **YOLO 目标拼接**：
  - 如果样本含 `bboxes/cls/batch_idx`，在 batch 维度拼接。
  - 重新生成 `batch_idx` 为当前 batch 内的样本索引。

### dsnake_collator
- 在 `snake_collator` 基础上增加 `act_hm/awh/act_ind/act_01`（动作检测分支）。

### 其它 collator
- `rcnn_snake_collator` 与 `ext_snake_collator` 目前被三引号注释，不参与实际调用。
- `_collators` 仅注册 `snake/ct`，默认回退到 `default_collate`。

## voc/snake.py：主要 Dataset 实现（训练）
### 初始化（`__init__`）
- 先按 COCO 方式加载 `ann_file`，建立 `json_category_id_to_contiguous_id`（保留兼容性）。
- **自定义数据路径**：
  - 读取 `cfg.train.data_path/train_list.txt`。
  - 每行是图像文件名，拼成 `train_images_path`。
  - 同时生成 `train_masks_path`（用文件名前缀 + `_mask` 作为 mask 根路径）。
- **`per_contour` 模式**：
  - 读取 `cfg.train.per_contour`（或 `cfg.per_contour`）。
  - 若为 `True`，预先遍历所有 mask → polygon，生成 `self.samples`：
    `(img_idx, mask_path, cls_id, poly_idx)`。
  - Dataset 长度随单个 polygon 数量变化。

### 关键辅助函数
- `transform_original_data`：处理翻转与仿射变换，输出到网络输出尺寸。
- `get_valid_polys`：去掉长度 < 4 的多边形、裁剪边界、过滤小多边形、确保顺时针、去重。
- `get_extreme_points`：为每个 polygon 生成 4 个极值点。
- `prepare_detection`：
  - 将 bbox 中心画到 `ct_hm`（Gaussian heatmap）。
  - 生成 `wh` 和 `ct_ind` 供检测分支使用。
- `prepare_init`：用 bbox 极值点生成初始 4 点轮廓（img/can）。
- `prepare_evolution`：
  - 用 extreme points 构造 octagon，并 uniformsample 到 `poly_num`。
  - 生成 `img_gt_poly`，对齐到 `img_init_poly` 的起点。

### `__getitem__` 核心流程
1. **读取图像 & mask**
   - `ours=True` 固定使用自定义数据。
   - 若 `per_contour=True`：只选定一个 polygon 作为样本。
   - 否则：对一张图的所有 mask 转换成 polygons，并记录每个 mask 的 polygon 数量（`cla_mask_num`）。
   - 图像读取失败会尝试 jpg/png 互换。
2. **数据增强**
   - `snake_voc_utils.augment` 返回 `orig_img`、网络输入 `inp`、变换参数等。
   - 之后 `transform_original_data` → `get_valid_polys` → `get_extreme_points`。
3. **构建检测 + Snake 目标**
   - 初始化 `ct_hm`（形状：`cfg.heads.ct_hm x H x W`）。
   - 遍历 polygon：生成 bbox → `prepare_detection` → `prepare_init` → `prepare_evolution`。
4. **YOLO 目标（额外分支）**
   - 按输入尺寸计算 `xywh`，归一化到 `[0,1]`。
   - YOLO 类别使用 `cls_id-1`（0 基）；Snake 分支仍用原 `cls_id`。
5. **返回结构**
   - `ret` 包含：`inp`、`orig_img`、`img_path`、检测/初始化/进化多边形目标、YOLO 目标。
   - `meta` 包含 `center/scale/ct_num`。
   - 如果 `cfg.vis_zrc` 为真，调用 `visualize_utils` 可视化中间结果。

### `__len__`
- `per_contour` 模式下返回 `len(self.samples)`，否则返回图片数。

## sbd/snake.py
- 仅 `from ..voc.snake import Dataset`，SBD 与 VOC 共享同一逻辑。

## voc_test/snake.py：推理/验证 Dataset
- 继承 `torchvision.datasets.coco.CocoDetection`。
- `split=val` 时过滤没有标注的图片。
- `__getitem__`：
  - 读取图像并走 `snake_voc_utils.augment`。
  - 返回 `inp` + `meta(center, scale, img_id, vis_GT)`。
  - 代码中 `ret.updata` 有拼写错误，`orig_img` 实际没有正确写入。
- `vis_GT` 存在时，`snake_collator` 会直接返回，不做检测/蛇形目标拼接。

## 关键配置依赖汇总
- `cfg.task`：决定加载 `lib/datasets/{id}/{task}.py`，以及 collator 类型。
- `cfg.train.dataset / cfg.test.dataset`：选择 `DatasetCatalog` 条目。
- `cfg.train.data_path`：`train_list.txt`、图像、mask 根路径。
- `cfg.train.batch_size / num_workers / max_ct_num / per_contour`：训练加载参数。
- `cfg.heads.ct_hm`：决定 `ct_hm` 通道数。
- `cfg.vis_zrc`：触发可视化输出。
