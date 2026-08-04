# Detector Stage A：目标契约统一与端到端损失因子隔离

更新时间：2026-08-04（Asia/Shanghai）  
负责人任务：`019fb3d5-abc9-7662-8731-8b8cb0c44755`  
状态：**执行中；未训练新检测器；未进入阶段 B/C**

## 1. 冻结接口

本阶段只接受 Flow interface manifest v1.1：

- 路径：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/flow_interface_freeze_20260804/FLOW_INTERFACE_FREEZE_MANIFEST_V1_1_20260804.json`
- SHA256：`c9ad7b8ffba2f3e2c5698a35a77f4b0b3c9fab23cd735d2241ae23aac2f55698`
- `label_id=1..25`：原始解剖 mask 标签，进入 `instance_id`
- `flow_class_id=label_id-1=0..24`：实际进入 Flow detection row 与 external cache
- 含混字段 `canonical_class_id` 已废弃；v1.0 只保留历史审计，不进入新产物。

冻结 Flow：

- Dense-6 + H1 checkpoint：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/output_head_h0_h1_h2_20260803/distilled/h1_distilled_full.pt`
- SHA256：`5e28f12df357ec4d18fc9f0baf67b5a57655932a585b4ae1a0254d8449ecfc72`
- 基础配置 SHA256：`d57ff1f9e0b620022e173e767bb03cad20c9ee126779b491cd323f4fb78755ae`
- 评估：8-NFE AB2，2 outer × 4 inner，seed `20260731`，memory-off，病例 `010/011/013`。

冻结 detector：LocateAnything `base1500`。正式 full-38 预测 JSONL SHA256 为 `bc2dab9f2e580e7eea4145b00b447e022eca92eb859bf14677431f817316a5db`；补充病例 011 JSONL SHA256 为 `6e93176377e0fee547d7ac2d7c4a7389e9069c932d1e8b9e34d4b2eb7c0288b3`。

## 2. 条件定义

- A：detector coverage + GT geometry + GT class。
- B：detector coverage + detector geometry + **oracle GT class**；只用于几何隔离，不是部署性能。
- C：detector coverage + detector geometry + predicted class。
- D：完整 GT geometry + GT class 控制。

条件 C 当前正式标记为 `blocked_no_registered_class_provider`。原因是 base1500 只输出 generic `vertebra`，没有已登记的非 oracle C1–L6/`label_id` 分类器。不得把匹配到的 GT class 冒充 predicted class；否则只是重复条件 B。

## 3. 目标计数口径

- `1796`：冻结 H1 `_mask_to_instances()` 下 significant components（每类 top-4、raw contour area ≥2、全局 cap32）再经过输出网格多边形有效性检查后的 Flow 初始化实例。
- `1178`：largest-only、无 detector bbox-area 200 门槛。
- `743`：largest-only + detector bbox area ≥200。

`1178/743` 不是在 `1796` 上继续过滤，不能用于阶段 A 的 Flow 目标替代。

## 4. 当前执行与严格失败记录

缓存构建器会逐切片调用冻结 H1 dataset，导出 Flow 实际消费的 original-image bbox、`label_id`、`flow_class_id`、类内 component rank、raw contour area 和 Flow row order。任何字段、计数或 SHA 不一致均停止。

已出现并保留两次“按设计停止”，均未生成缓存：

1. `sub-verse010/x0051`：一个两像素级 component 在中心取整与逆仿射后，Flow bbox 对 raw bbox 的 IoU 仅 0.1446。结论：不能用 bbox IoU 阈值猜 component rank；改为利用 `_mask_to_instances()` 的同类面积顺序绑定。
2. `sub-verse011/x0034`：raw label 9 虽通过 top-4/area≥2，但在输出网格 `min_poly_area_output=0.5` 检查中被丢弃。结论：1796 不能由 raw significant count 直接代替；构建器现已复现冻结仿射、裁剪、多边形面积和退化框门槛后再绑定。

第三次导出在 1000/6247 时由实现审计主动暂停：native GT packing 会用 `clip_to_image` 把 `x2/y2` 裁到 input size−1，原导出器在逆仿射前尚未复现该边界步骤。虽然它只影响贴边框，但会破坏 D-cache 与 native-D 的逐行等价。该次同样未落缓存；第四次导出已补齐完全相同的裁剪语义后重启。

当前第四次严格导出正在 CPU 后台运行。目标门槛：

- 选定三病例必须严格为 `333 slices / 1796 instances`；
- full-38 正式审计必须严格为 `38 volumes / 6160 slices`；
- 条件 D cache 与 native GT detection row 必须在 1796 行上保持行数、顺序、`flow_class_id` 完全一致，bbox 最大误差不超过 `1e-3` input pixel。

## 5. 计划产物

机器结果根目录：

`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_20260804`

预计包含：

- `flow_gt_instances_selected3.json`
- `condition_a_detector_coverage_gtgeom_gtclass_iou01.json`
- `condition_b_detector_coverage_detgeom_gtclass_iou01.json`
- `condition_c_blocked.json`
- `condition_d_full_gt_control.json`
- `coverage_audit.json`
- `cache_contract_validation.json`
- `stage_a_manifest.json`

冻结 Flow 联评将另外产生 native-D、cache-D、A、B 四组结果与逐病例 mask stack，并计算 class-agnostic foreground 3D Dice、NSD@2 mm、HD95(mm)。损失定义为：

- coverage：D → A；
- geometry：A → B；
- class：B → C，当前 blocked；
- combined detector-box：D → B。

GPU 当前均有其他任务占用。本任务不会中断或共享他人 GPU；只有显存使用低于 2 GiB 的空闲卡才会启动联评。
