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

## 6. CPU 阶段正式结果（已通过）

缓存与 full-38 覆盖审计已完成。release：

- `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_20260804/STAGE_A_CACHE_RELEASE_20260804.json`
- SHA256：`23789d234bab10706c7aee0caee2f3268d897add73df41defa7b1d8e637b8fb3`

三病例 exact H1 population：

- 333 slices，1796 GT Flow instances，894 detector boxes；
- IoU≥0.1 一对一匹配 779，GT coverage recall `0.4337416481`；
- IoU≥0.3 匹配 664，recall `0.3697104677`；
- IoU≥0.5 匹配 375，recall `0.2087973274`；
- 匹配框平均 IoU `0.4909929340`。

分层结果直接揭示主要问题是 **significant-component coverage**：

- component rank 0：735/1157，recall `0.6353`；
- rank 1：41/513，recall `0.0799`；
- rank 2：3/113，recall `0.0265`；
- rank 3：0/13；
- raw contour area 2–9：0/46；10–49：9/319 (`0.0282`)；50–199：208/620 (`0.3355`)；200+：562/811 (`0.6930`)；
- foreground edge slices：120/412 (`0.2913`)；center：260/555 (`0.4685`)；transition：399/829 (`0.4813`)；
- 连续漏检段 327 个，均值 3.11 slices，最大 36；未匹配预测轨迹 79 条，其中长度≥3 为 7 条，最长 8。

正式 full-38 exact H1 population：

- 38 volumes / 6160 slices；
- 31,772 GT significant Flow instances，15,317 detector boxes；
- IoU≥0.1 匹配 13,233，recall `0.4164988040`；
- IoU≥0.3 匹配 11,042，recall `0.3475387133`；
- IoU≥0.5 匹配 6,215，recall `0.1956124890`；
- 匹配框平均 IoU `0.4919628028`；
- 连续漏检段 4,822 个，均值 3.84 slices，最大 41；FP 轨迹 1,518 条，其中长度≥3 为 117，最长 13。

以上 full-38 数字针对 Flow 的 31,772 个 significant-component 实例，不可与旧 detector largest-only+area200 的 14,331 个 GT 框直接混用。

## 7. 缓存 release 与逐行合同

- A：779 rows，SHA `5f996fb77829c0c0cd8a7297b3d726f5f6f9a2769b94c8a5087a9a4ff6f604bb`
- B：779 rows，SHA `3041c334be91e43d5b0fbe2f69f041ffbbacdc2d656ee245745c80e2213137c3`
- D：1796 rows，SHA `af798bf2c3302102eede1257783d5a4a98c039dc2bfaa9e6d982eba4dcc47336`
- C：`blocked_no_registered_class_provider`，无缓存、禁止评估。

D-cache 对 native GT detection row 的合同验证：

- 路径：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_20260804/cache_contract_validation.json`
- SHA256：`3d1cd18e5dc848af0f13247ce0759410890f20eb3d6ae0e4d700cefe5eb67b45`
- 333 slices / 1796 rows；bbox mismatch 0，class mismatch 0，order mismatch 0；
- 最大 bbox 绝对误差 `3.0517578125e-05` input pixel，门槛 `1e-3`；status `pass`。

## 8. 与旧 detector 正式协议的关系

旧 largest-only + bbox area≥200 正式 full-38 协议仍是 detector 自身可引用指标：

- IoU 0.3：micro F1 `0.6988667`，macro-volume F1 `0.7144087`；
- IoU 0.5：micro F1 `0.4116298`，macro-volume F1 `0.4297367`。

count-prefix50 的 full-38 结果没有晋级：

- IoU 0.3 micro F1 `0.6977763`、macro F1 `0.7137506`，均略低于 base1500；
- IoU 0.5 micro F1 `0.4087601`、macro F1 `0.4267432`，也低于 base1500。

因此 count-prefix 是正式负结果；不能用 quick200 的局部改善替换 full-38 门槛。

## 9. Gate-0 sidecar 限制与 GPU 状态

当前 retained-only sidecar：

- `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_20260804/flow_gt_instances_selected3.json`
- SHA256：`5065635845776a6d22576ac79f24a7489c6801dc0d1e156b5815e1aec3f4ec48`

它含 1796 retained rows，含 `label_id`、`flow_class_id`、0-based component rank（字段名 `component_rank_within_class_by_raw_contour_area`）、raw area、original bbox、两个 retained flag 和 `flow_instance_order`。它不含显式 `rank_index_zero_based`/`rank_ordinal_one_based` 别名，也不保留 discarded rows；因此若加速 Gate-0 要求“同表保留舍弃行且 flow_instance_order=null”，需由 Flow 侧另发扩展 sidecar，不能把当前文件宣称为满足该 schema。

GPU 联评尚未启动：当前非服务 GPU 均被其他任务占用。安全守护只在 cache validation=`pass` 且显存占用<2 GiB 时启动 native-D/cache-D/A/B；否则持续等待。

## 10. Flow 签名 retained-only Gate-0 sidecar（独立复核通过）

Flow 侧未改写阶段 A 源表，而是在其上签发了 retained-only Gate-0 适配版：

- sidecar：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/flow_interface_freeze_20260804/FLOW_GT_INSTANCE_SIDECAR_RETAINED_V1_20260804.json`
- SHA256：`764f9f80e459a9a0272ba43c3cf90682aacf970fe37ab6ef988528d8932dd901`
- schema：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/flow_interface_freeze_20260804/FLOW_GT_INSTANCE_SIDECAR_RETAINED_V1_SCHEMA_20260804.json`
- schema SHA256：`8e4f8c253088e1b78ef94233674fd65e20354faca7fe35231a7a4efa05ca0c53`

检测器阶段 A 于 2026-08-04 完成逐行只读独立复核，结论 `PASS`：源表与签名表均为 1796 行；`source_row_number_zero_based` 精确保留源行顺序；除权威 `instance_id` 的 1-based ordinal 改写外，源字段逐值不变；旧 0-based ID 由 `source_instance_id` 精确保留；`rank_index_zero_based` 与原 component rank 相同，`rank_ordinal_one_based=rank+1`；172 个有实例切片内 `flow_instance_order` 均连续为 `0..N-1`；两种 retained flag 均为 true。

该 sidecar 的 scope 只允许 Gate-0 retained 行对齐，不包含 discarded rows，也不能用于重建 1178/743 口径。Flow 权威边界报告：`/home/medteam/Zhrch/DiffusionSnake-12-30/docs/report/FLOW_GT_ORACLE_AND_INTERFACE_STATUS_20260804.md`，SHA256 `687d07149ccca200d4ca8553ed5fab0de7dd0a4c984551d093d2d43fc225dc76`。

## 11. 阶段 A 正式联评协议升级：smoke 与 full-38 分离

论文统筹于 2026-08-04/05 明确要求：`sub-verse010/011/013` 只作为协议、缓存与 zero-control smoke，不得单独晋级；正式结论必须来自 detector 正式完整 38-volume cohort。验证池实际包含 40 例/6247 slices，其中 `sub-verse011` 与 `sub-verse252` 不在 base1500 正式 full-38 predictions 中。因此禁止使用简单的 `max-volumes=38`，否则会错误纳入 011/252 并丢弃两个排序靠后的正式病例。

当前执行链已改为：

1. smoke 使用显式 case-list 跑 native-D、cache-D、A、B；D-cache/native-D 的 Dice、NSD、HD95 和前景覆盖 zero-control 必须在预设数值容差内通过。smoke 状态固定为 `quick_not_eligible_for_promotion`。
2. 正在独立生成 formal full-38 exact-H1 A/B/D caches，目标为 38 volumes / 6160 slices / 31,772 D rows / 13,233 A/B matched rows；原三病例 release 不覆盖、不修改。
3. full-38 D-cache 必须再与 native H1 rows 做独立 bbox/class/order 合同验证；只有 full-38 validation=`pass` 且 smoke zero-control=`pass`，正式 Flow 联评才允许启动。
4. 正式 full-38 运行固定 Dense-6+H1、AB2、2 outer×4 inner=8 NFE、memory-off、seed 20260731、同病例/噪声/实例顺序。正式条件为 cache-D、A、B，并从 D 输出按 A instance ID 派生 common-noise coverage 控制。
5. 指标机器 JSON 将报告 physical 3D Dice、NSD@2 mm、HD95(mm)、前景覆盖/召回、coverage 与 geometry 因子、每病例损失和最差病例。B 始终是 oracle-class geometry isolation；C 继续 `blocked_no_registered_class_provider`。

A/B 只包含 IoU≥0.1 的 detector-to-GT matched rows；未匹配 detector FP 不进入 Flow 演化。因此阶段 A 能严格量化漏检覆盖和 matched-box geometry 的损失，但不包含 FP 轮廓对结果的破坏。该限制必须与正式数字同时报告。

## 12. full-38 original-image D-cache 失败与 input-space 修复

formal full-38 original-image cache release 已完成：

- manifest SHA256：`e6e11076716187b3fb19c097b58b82ac6f70b91a2665e94f87e3f04e71e1880a`
- case-list SHA256：`4a4a69e2a4bf1086f39ac42bd1b03f5bb076d29a03f3dd65a13ca18c4cac4513`
- A：13,233 rows，SHA `c88ccbc544bdef7e987a136c5c2cf8290ffc67121514d62a39c79cfb9ef74ca0`
- B：13,233 rows，SHA `ca97a58bebc677b5bedfb366b30ac6d0a7036b8430ba482659a9970f3afb5e56`
- D：31,772 rows，SHA `068f784adc06532c92fe9c314f3b43b48826623682d670ad81dcb63b2e40c87a`

但独立 D-cache/native 合同验证按门槛失败，正式 Flow 未启动：

- 失败 JSON：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_full38_cache_20260804/cache_contract_validation_full38.json`
- SHA256：`04ce44e2cdd43b614f41b467c1bc0339fdb0dab801e9d887842b371fee71e246`
- 6160 slices / 31,772 rows、shape failure 0、class mismatch 0、row-order mismatch 0；但 bbox mismatch 22，最大 input-space 绝对误差 `1.96075439453125` pixels。
- 明细诊断：`cache_contract_validation_full38_diagnostic_v2.json`，SHA256 `2318cd903f3d3ff125e956fa0322abfa4c6a7612f0507f92f98e04264a607b01`。

22 行只来自两个病例：`sub-verse071` 12 行（slice 186–197），`sub-verse150` 10 行。它们全部是贴住原图底边的 GT 实例；旧路径先把 native 512-input GT box 逆仿射到 original image，并把 `y2` 裁到 `height-1`，随后再正仿射回 input，因而无法恢复 native 在 padded input 中超出原图有效边界的 1.49–1.96 pixels。该误差不是类别、实例顺序或 split-volume spacing 导致。

禁止通过放宽 bbox tolerance 使该失败“通过”，因为那会让 A 的 GT geometry 带入非 coverage 损失。修复采用显式 `coordinate_space=network_input_pixels` 的版本化缓存：

- D/A 的 GT geometry 直接复制冻结 native H1 detection rows；
- B 将同一批 matched detector original-image boxes 只执行一次冻结 val affine，继续使用 oracle GT class；
- evaluator 只读复用输入空间 cache，不再做 original→input 二次变换；
- 原 `original_image_pixels` cache 格式保持兼容，未覆盖失败文件。

cache adapter 的专用回归测试为 41/41 PASS。另一个既有 `test_external_detection_contract` mock 参数测试仍失败，但已在未修改的主工作树上独立复现，确认不是本次 coordinate-space 变更引入。input-space full-38 A/B/D 正在生成；完成后必须重新跑 31,772-row D/native 严格验证，未通过前 formal Flow 仍保持关闭。

## 13. network_input_pixels V2 硬合同与当前放行状态（2026-08-05）

Flow 与论文统筹已只读签认新的消费边界：external detection 必须是当前 512×512 网络输入帧中的 `[x1,y1,x2,y2,score,flow_class_id]`。`network_input_pixels` provider 只允许直接打包，不读取 `trans_input`、`flipped` 或 `orig_hw`，不执行 `transform_detection`、flip、clip、排序坐标或 original↔input 往返，也不得在缓存侧预先除以 4。provider 在入 Flow 前严格拒绝非有限值、非正面积、越出 `0..512` 或缓存 shape 与当前输入帧不一致的记录。

Flow 边界代码绑定：

- `lib/networks/snake/ct_snake.py` SHA256 `c78cc615b9a5d9cfe8bcea66ce2e91922f97892f36fa0c0a815c150301974138`：`apply_external_detection` 直接消费 external rows，不做 affine/scale/flip/clip。
- `lib/utils/snake/snake_gcn_utils.py` SHA256 `237af652afe83f0b5d4f011978b6c717773db6ea6cabcb0b96e26c8c285433e6`：`prepare_testing_init` 构造初始化轮廓后恰好一次 `/4`。
- 主工作树旧 `locany_cache.py` SHA256 `8db12c46a8f95fd45f9957e8162c6a6ebd1b2ffe5682d3924bf83ddb04bb2dec` 只具备 original-image 语义，不能原样用于新缓存。
- Stage-A 新直通 adapter：`/home/medteam/Zhrch/DiffusionSnake-12-30-detector-stage-a-20260804/volmem/detection/locany_cache.py`，SHA256 `b36b762a550558781072e9a314f4290733c3d9509609c344087a9a5897c4a25b`；专用回归测试 44/44 PASS。

full-38 input-space 缓存已经生成，但在 V2 严格合同完成前只是 staging，不代表执行放行：

- staging manifest：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_full38_inputspace_cache_20260805/inputspace_cache_manifest.json`，SHA256 `d8c04ca42327710651f1d1d72793cd03e630c37750d3d83af378b30824eabfd0`。
- A：6160 slices / 13,233 rows，SHA256 `7120f154360c5d35403c06878496f3ce8f09f652d286cd3faf42fd429b559ca2`。
- B：6160 slices / 13,233 rows，SHA256 `b87dd9b951340846cf9aedaf9d360cbe2df91424a1e69b9aabc1ccca61b14455`。
- D：6160 slices / 31,772 rows，SHA256 `3c41077608ed78e0466830307972ff99c8d0e135c6c0ce154560d96e711c7d2c`。

V2 验证器 SHA256 为 `b7f20ab0a7f674d5c25765a7543b12c0a14f967ea03622560c69702d296c4033`。它要求 full-38 detection rows 的 bbox、score、class、shape、order 全部 exact，bbox max_abs=`0`；单列复核旧诊断中的 22 个 bottom-edge rows；并逐切片比较 `i_it_4py/c_it_4py/ind/i_it_py/c_it_py/py_ind` 六组初始化张量，全部必须 `torch.equal`。

当前状态：严格 CPU 只读验证正在运行，机器 JSON 尚未写出，因此 **31,772-row 合同尚未声明 PASS**；D-cache vs native GT 的 GPU/Flow zero-control 尚未启动；本 Stage-A 当前没有 GPU、Flow evaluator 或训练进程。放行顺序保持为：严格 V2 合同 PASS → 签发版本化 release manifest → selected3 smoke zero-control PASS（quick 不晋级）→ full38 D-cache/native final Flow output exact → 才允许 A/B 正式联评。任何不 exact 立即停止。

## 14. V2 严格合同正式 PASS（2026-08-05，取代上一节在途状态）

full-38 network-input D-cache/native 合同已经完成并正式通过：

- 机器结果：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_full38_inputspace_cache_20260805/cache_contract_validation_inputspace_full38_strict_v2.json`
- SHA256：`f7ae5d25a6986a991adf137cac6eefbd3a5c3df81aae195c8a129af8ed9135e2`
- 38 volumes / 6160 slices / 31,772 rows；bbox、score、class、shape、order mismatch 均为 0；`max_bbox_abs=0.0`，`max_detection_row_abs=0.0`。
- 六组初始化张量在 6160 张切片上全部 exact，tensor mismatch 0，`max_abs_diff=0.0`。
- `sub-verse071` slices 186–197 的 12 行和 `sub-verse150` 的 10 行定点审计共 22/22 exact，`max_abs_diff=0.0`。

签发的版本化执行 manifest：

- 路径：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_full38_inputspace_cache_20260805/INPUTSPACE_CACHE_RELEASE_MANIFEST_V2_20260805.json`
- SHA256：`f2c691ff3fdcaf2517e8d237ec9339617c290725f0ad8a383fc63e17bd0bbe60`
- 状态：`released_for_smoke_zero_control_then_formal_execution`。

该 PASS 只证明坐标、检测行和初始化轮廓合同修复成功，不代表 detector 性能改善，也不代表 D-cache final Flow output zero-control 已完成。当前没有启动 D-cache GPU/Flow zero-control，没有 Stage-A GPU/Flow/训练进程，A/B 正式联评继续关闭；下一门仍是 selected3 smoke 中的 native-D/cache-D final Flow output exact。
