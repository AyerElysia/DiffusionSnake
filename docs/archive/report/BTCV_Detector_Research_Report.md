# BTCV Detector Research Report

## 概述

本报告总结了本轮 **BTCV 检测分支** 的独立探索工作，目标是：

- 暂时不碰 diffusion 主线
- 只把检测部分单独跑通并做强
- 检测结果尽量服务于后续 **极值点 / 八边形初始化**

本轮最终结论是：

1. **仓库内嵌 YOLO 检测链本身是可用的**
2. 最早出现的“训练后仍然 0 框”是 **评估脚本加载 checkpoint 错误**，不是 detector 没学会
3. 当前最强可用基线是：
   - **official YOLOv8n pretrained 初始化**
   - **BTCV prepared Snake-format 数据**
   - **全量训练 bbox detector**
4. 同时已经把 BTCV 导出成：
   - **YOLO detect**
   - **YOLO pose（4 个极值点）**

也就是说，后面既可以继续沿 bbox baseline 优化，也可以直接切到更贴近八边形目标的 extreme-point / pose 路线。

---

## 环境与约束

- 代码目录：`/home/medteam/Zhrch/DiffusionSnake-12-30`
- 环境：`conda activate snake1`
- GPU：`CUDA_VISIBLE_DEVICES=3`
- 当前目标：只做 detector，不做 diffusion 训练

额外限制：

- `snake1` 是 **Python 3.7.12**
- 外部 `yoloe/ultralytics` 新版在这个环境下并不友好，很多分支要求 Python 3.8+

---

## 数据结论

### 1. 原始 BTCV 数据不是当前最优入口

仓库/文档里有不少旧路径和旧格式，但当前机器上真正适合 detector 实验的数据是：

- 训练：
  `/home/medteam/Zhrch/Datasets/BTCV/btcv_png_new_snake`
- 验证：
  `/home/medteam/Zhrch/Datasets/BTCV/btcv_png_test_new_snake`

### 2. 这套 prepared 数据和 Snake 当前代码是对齐的

格式为：

- `<id>_image.png`
- `<id>_mask_<class>.png`

例如：

- `0_image.png`
- `0_mask_1.png`
- `0_mask_2.png`

这套数据可以直接产出：

- bbox supervision
- polygon supervision
- extreme points

因此它非常适合同时支撑：

- bbox detector
- extreme-point / pose detector
- mask -> extreme-point 后处理

---

## 关键问题与修复

## 1. 最重要的误判来源：smoke 脚本没有正确加载训练 checkpoint

### 问题

`yolo_train.py` 保存的是 trainer wrapper checkpoint，键名带有：

- `net.xxx`
- 或 `module.net.xxx`

但最早的 `test/test_btcv_detector_smoke.py` 直接把它加载到 bare `Network` 上，导致真正训练出来的权重没有正确进入推理网络。

### 影响

表面现象会变成：

- loss 在下降
- 但 eval 端一直 0 detections
- class score 像常数一样不动

这会让人误以为 detector 完全没学会。

### 修复

在 `test/test_btcv_detector_smoke.py` 中增加了 checkpoint 解包逻辑，自动剥掉：

- `net.`
- `module.net.`

修复后，训练后的 detector checkpoint 可以被正确评估。

---

## 2. 官方 YOLOv8 预训练权重原本无法在当前仓库直接使用

### 问题

官方 `yolov8n.pt` 在 `torch.load()` 时会引用顶层模块名：

- `ultralytics.*`

而当前仓库使用的是 vendored 版本：

- `lib.networks.YOLOV8.*`

同时 `snake1` 环境又没有可直接使用的新版本外部 `ultralytics` 包。

### 修复

在：

- `lib/networks/YOLOV8/nn/tasks.py`

里给 `torch_safe_load()` 加了模块别名映射，把：

- `ultralytics`
- `ultralytics.nn`
- `ultralytics.nn.tasks`
- `ultralytics.nn.modules`
- `ultralytics.utils`
- `ultralytics.data`

映射到仓库自带的：

- `lib.networks.YOLOV8.*`

### 结果

现在可以直接加载官方：

- `data/pretrained/yolov8n.pt`

并成功把它作为当前内嵌 detector 的预训练初始化。

---

## 3. 单样本 overfit 需要显式关闭随机增强

### 问题

这个仓库的数据增强并不主要走 `make_transforms()`，而是 dataset 内部基于 `split == 'train'` 走随机 crop / flip / color aug。

所以最早的单样本 smoke，虽然只用了 1 个样本，但实际上每一步看到的几何分布并不固定。

### 修复

新增：

- `YOLO_DISABLE_AUG`
- `SNAKE_DISABLE_AUG`
- smoke 脚本里的 `--disable-aug`

这样单样本 overfit 就变成了真正可判定的 no-aug 过拟合测试。

---

## 新增脚本与产物

## 1. 新增检测 smoke 脚本

- `test/test_btcv_detector_smoke.py`

作用：

- 加载 detector checkpoint
- 跑单样本 train / val
- 输出 GT vs pred 对比
- 存图

## 2. 新增批量验证脚本

- `test/test_btcv_detector_eval.py`

作用：

- 跑完整 `train` 或 `val`
- 做 class-aware one-to-one greedy matching
- 输出：
  - precision_like
  - recall@0.5
  - recall@0.75
  - mean best IoU
  - matched mean IoU
  - per-class summary

## 3. 新增 BTCV 导出脚本

- `scripts/export_btcv_yolo_labels.py`

已导出到：

- `data/exports/btcv_yolo/`

内容包括：

- `btcv_detect.yaml`
- `btcv_pose.yaml`
- `labels/detect/{train,val}`
- `labels/pose/{train,val}`
- `images/{train,val}`

### 导出规模

- train: `720 images / 4746 objects`
- val: `150 images / 1020 objects`

### pose 标签定义

极值点顺序为：

1. top
2. left
3. bottom
4. right

这和当前 Snake 里的 `get_extreme_points()` 一致，适合后面做 octagon 初始化相关实验。

---

## 单样本 smoke 结果

## 1. no-aug 单样本 full-scope overfit

checkpoint：

- `data/outputs/btcv_yolo_detect_only_smoke_full_500step_noaug/checkpoints/step_369.pt`

结果：

- `6 / 6` GT 均匹配到正确预测
- IoU 约 `0.985 ~ 0.997`

### 结论

说明当前 detector 训练链是 **真的可学** 的。

## 2. official pretrained + no-aug 短程 smoke

checkpoint：

- `data/outputs/btcv_yolo_detect_only_smoke_pretrained_noaug_200/checkpoints/latest.pt`

结果：

- `6 / 6` 匹配
- IoU 约 `0.94 ~ 0.99`

### 结论

官方预训练初始化明显是更好的起点。

## 3. 带增强的单样本 smoke

checkpoint：

- `data/outputs/btcv_yolo_detect_only_smoke_full_500step_lr3e4/checkpoints/latest.pt`

结果：

- 能出框
- 但比 no-aug overfit 明显更粗
- IoU 约 `0.80 ~ 0.92`

### 结论

单样本 smoke 时应优先用 **no-aug**，否则不容易判断 detector 本身是否能过拟合。

---

## 全量训练配置

当前主线 baseline：

- 输出目录：`data/outputs/btcv_yolo_detect_only_full_pretrained_v1`
- 初始化：`official yolov8n.pt`
- 训练方式：`full YOLO scope`
- batch size：`8`
- 数据：
  - train: `btcv_png_new_snake`
  - val: `btcv_png_test_new_snake`

最终训练日志显示：

- 训练到 `epoch 99`
- 最终 step 约 `9000`
- loss 从早期高值下降到约 `15~19`

---

## 最终验证集结果

评估脚本：

- `test/test_btcv_detector_eval.py`

最终 checkpoint：

- `data/outputs/btcv_yolo_detect_only_full_pretrained_v1/checkpoints/latest.pt`

最终 val 指标：

| 指标 | 数值 |
|------|------|
| num_gt | 988 |
| num_pred | 1035 |
| num_matched | 855 |
| precision_like | 0.826 |
| recall@0.5 | 0.800 |
| recall@0.75 | 0.739 |
| mean best IoU | 0.734 |
| matched mean IoU | 0.848 |

---

## 分类别结果

| cls | 器官 | recall@0.5 | recall@0.75 | matched mean IoU |
|-----|------|------------|-------------|------------------|
| 0 | spleen | 1.000 | 0.993 | 0.886 |
| 1 | right kidney | 0.587 | 0.326 | 0.698 |
| 2 | left kidney | 0.804 | 0.790 | 0.936 |
| 3 | gallbladder | 0.924 | 0.914 | 0.919 |
| 4 | esophagus | 0.791 | 0.721 | 0.876 |
| 5 | liver | 0.551 | 0.370 | 0.644 |
| 6 | stomach | 0.914 | 0.914 | 0.934 |
| 7 | aorta | 0.707 | 0.650 | 0.757 |

### 当前最强类别

- spleen
- gallbladder
- stomach
- left kidney

### 当前短板类别

- **right kidney**
- **liver**

这两个类别是下一轮优化最值得优先针对的对象。

---

## 本轮最终判断

## 1. bbox baseline 已经成立

这条线现在不是“是否能跑通”的问题，而是“还能做多强”的问题。

已经可以认为：

- 当前内嵌 YOLO 检测链可以作为 BTCV detector baseline
- official YOLOv8n pretrained 初始化是当前最佳起点

## 2. 但这还不是终局

用户真正更关心的是：

- 极值点
- 八边形初始化

而不是仅仅 bbox mAP / bbox recall。

因此 bbox baseline 的价值主要是：

- 给下游 diffusion / snake 一个稳定候选框
- 提供当前机器和代码库里最快速可用的 detector 起点

---

## 下一步建议

### 建议 1：继续沿当前 bbox baseline 做定向优化

优先方向：

- 针对 `right kidney / liver` 做 class-specific error analysis
- 调整：
  - conf / iou 阈值
  - batch size / lr / schedule
  - 是否需要 per-class sampling

### 建议 2：把重点切到 pose / extreme-point detector

因为当前已经准备好了：

- `data/exports/btcv_yolo/btcv_pose.yaml`

所以下一轮可以直接基于这份数据训练：

- 4-point pose detector

这条线比 bbox-only 更贴近最终八边形初始化目标。

### 建议 3：保留 bbox detector 作为 fallback / ensemble 候选

即便后面 pose detector 更强，当前 bbox detector 仍有价值：

- 可作为候选框 proposal
- 可和极值点 detector 形成组合式初始化

---

## 相关文件清单

### 关键脚本

- `yolo_train.py`
- `test/test_btcv_detector_smoke.py`
- `test/test_btcv_detector_eval.py`
- `scripts/export_btcv_yolo_labels.py`

### 关键配置

- `configs/btcv_yolo_detect_only.yaml`

### 关键输出

- `data/pretrained/yolov8n.pt`
- `data/exports/btcv_yolo/`
- `data/outputs/btcv_yolo_detect_only_smoke_full_500step_noaug/`
- `data/outputs/btcv_yolo_detect_only_smoke_pretrained_noaug_200/`
- `data/outputs/btcv_yolo_detect_only_full_pretrained_v1/`

---

## 总结

本轮检测探索已经完成了三件关键事：

1. **把 BTCV detector 真正跑通**
2. **把“0 框假死”问题定位并修复**
3. **把后续 extreme-point / pose 路线需要的数据提前准备好**

当前最强可用结论是：

> **official YOLOv8n pretrained + BTCV full-data bbox training**
> 已经形成一个稳定可用的 detector baseline，
> 最终 val 表现约为：
> `precision_like 0.826 / recall@0.5 0.800 / recall@0.75 0.739 / matched_mean_iou 0.848`

如果后续目标是“继续把八边形初始化做到更低误差”，那么下一轮最值得投入的是：

> **在已导出的 BTCV pose 数据上继续训练 extreme-point detector**

