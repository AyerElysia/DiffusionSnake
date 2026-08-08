# Detector / Flow 初始化轮廓审计与 Rectangle 消融（2026-08-07）

## 状态边界

- 任务只改变 bbox 到初始轮廓的几何形状；detector、类别、缓存、Flow 权重、NFE、solver、阈值、Memory 和 RL 均冻结。
- 不训练任何模型；010/011/013 未进入病例列表，也没有用于选择。
- 最终执行版本为 `v3`。早期4角矩形设计和启动失败均保留审计，但不进入最终比较。
- 本实验是非 locked 的5病例开发集消融，不是 full38 正式结果，也不是论文主指标。
- 按统筹最新边界，本任务仅比较 **2D 单切片 rectangle 与 octagon 初始化**；不接入邻片、Memory、2.5D 或 3D 模块。逐实例2D指标是主要判断依据，逐卷体积指标只作为结果一致性检查，不作为方法设计、调参目标或论文贡献。

## 正式 GT-box / detector-box 链路

### GT-box

1. `lib/evaluators/sagittal_2d_fixed/snake.py:177-185`：`configure_box_mode(..., 'gt')` 显式启用 `use_gt_det`。
2. `lib/networks/snake/ct_snake.py:1449-1504`：`use_gt_detection` 从冻结 dataset 的 `ct_ind/wh/ct_cls` 生成 `[x1,y1,x2,y2,score,class]`；bbox 从 feature grid 乘 `down_ratio=4` 回到当前512网络输入坐标，score=1，类别仅做一次 offset。
3. 输出写入 `output['detection']`，随后与 detector-box 共用同一初始化入口。

### detector-box（Stage-A B）

1. 签名 B cache 是 `coordinate_space=network_input_pixels` 的512输入帧 `[x1,y1,x2,y2,score,flow_class_id]`。
2. `volmem/detection/locany_cache.py:425-490`：network-input 分支直接打包，不再执行 original↔input affine、flip 或 clip。
3. `volmem/adapters/v4_6c.py:60-65`：写入 `batch['external_detection']`。
4. `lib/networks/snake/ct_snake.py:1506-1555`：只验证有限值、正面积、padding 和 class 0..24，随后原样写入 `output['detection']`；没有第二次坐标变换。

### bbox 到 Flow 128点初始轮廓

1. `lib/utils/snake/snake_config.py:45-64`：V3/V4 历史默认解析为 `octagon`；新 `box_init_shape` 仅在显式设置时覆盖，合法值为 quadrangle/octagon/rectangle。
2. `lib/utils/snake/snake_decode.py:154-241`：
   - quadrangle：上/左/下/右四个边中点；
   - octagon：从四个伪 extreme 点按宽高的 `1/8` 生成 **12个** DeepSnake 控制点，名称虽为 octagon，但不是严格8顶点多边形；
   - rectangle：最终版本使用矩形边界8个控制点，起点为上边中点，依次经过左上、左中、左下、下中、右下、右中、右上，与 octagon 共享循环相位和方向。
3. `lib/utils/snake/snake_gcn_utils.py:138-174`：去除 score≤1e-4 padding，控制点均匀重采样到 `init_poly_num=40`，随后恰好一次 `/4` 进入 Flow feature grid。
4. `lib/utils/snake/snake_gcn_utils.py:36-61`：40点轮廓再重采样为 `poly_num=128`；这128点即 frozen Flow 的 `i_it_py`。
5. `lib/networks/diffusion/flow_matching_evolution.py:3009-3045`：无 predicted-extreme/SAM 特殊入口时直接消费这128点；Flow 不再调用 `get_octagon`。

## 隐藏二次 octagon 路径

确实存在一个潜在隐藏路径：当 `use_pred_extreme_init_for_inference=True` 且 `output['ex']` 存在时，`flow_matching_evolution.py:3022-3026` 会把4个 refined extreme points 再传给 `_octagon_init_from_extreme`。如果只在 bbox 入口换矩形，rectangle 会在这里被偷偷改回 octagon。

本消融采取两层防护：

- rectangle config 显式固定 `use_extreme_refine=false` 和 `use_pred_extreme_init_for_inference=false`；
- `snake_gcn_utils.assert_box_init_route_is_not_bypassed` 与 Flow 入口硬联动；若 rectangle 仍遇到 predicted-extreme octagon 路径，立即报错而不是静默执行。

CPU 合成测试已验证该失败关闭逻辑。

## 数值例子

输入 bbox（512网络输入像素）：`[64,96,192,224]`。

Baseline octagon 12控制点：

`[(128,96),(112,96),(64,144),(64,160),(64,176),(112,224),(128,224),(144,224),(192,176),(192,160),(192,144),(144,96)]`

Rectangle 8控制点：

`[(128,96),(64,96),(64,160),(64,224),(128,224),(192,224),(192,160),(192,96)]`

两者：

- 起点相同：上边中点 `(128,96)`；
- 方向相同：上→左→下→右；
- bbox min/max 完全相同；
- 均先变40点，再 `/4`，再变128点；rectangle 的 feature-grid 起点为 `(32,24)`；
- 最终合成测试的 bbox min/max 浮点误差为0。

## 训练/推理是否一致

- 冻结 H1 训练配置 `configs/volmem/verse_memflowdit_v0_5_minimal_gpu6.yaml` 启用 V3/V4，因此训练数据默认 shape family 为 octagon，Flow 轮廓点数为128。
- `lib/datasets/voc/snake.py:446-478` 显示训练的40点 bbox init 也走 `snake_voc_utils.get_init`；Flow训练的128点 evolution init 走 `get_evolution_init(extreme_point,bbox)`，默认同样为 octagon。
- 因而 baseline 在“octagon family + 128点”上训练/推理一致，但几何来源并非完全相同：训练 evolution init 使用真实GT轮廓 extreme points，推理 baseline 使用 bbox 边中点构造的伪 extreme points。
- rectangle 是冻结 octagon 预训练模型上的推理期 OOD 初始化消融；本任务禁止训练，因此不能声称 rectangle 已训练匹配。若 rectangle 仅在冻结模型上表现更好，仍需 full38 验证；若要成为长期训练合同，需要另做训练侧 rectangle 对齐实验。

## 实现与静态测试

- 开关：`cfg.box_init_shape`，空值保持历史默认，`rectangle` 显式开启矩形。
- rectangle config：`configs/volmem/stage_a_h1_external_cache_rectangle.yaml`。
- CPU 合同测试：`tools/volmem/validate_rectangle_init_contract.py`。
- 最终静态结果：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/rectangle_init_ablation_static_v2_20260807/rectangle_init_contract.json`，SHA256 `8148fb5290342610984ddc572e0188b6b7bf5c33b9dd23560829ee33285092c5`。
- PASS：8点矩形控制点、同bbox、同起点/方向、40→128点、CPU张量、rectangle≠octagon、隐藏 get_octagon 路径被阻断。

## GPU 消融冻结协议

- 病例：sub-verse022、024、071、150、264；共5 volumes / 1248 sagittal slices；病例在 rectangle 推理前预注册。
- locked 010/011/013：访问数0。
- checkpoint：Dense-6+H1 `h1_distilled_full.pt`，SHA256 `5e28f12df357ec4d18fc9f0baf67b5a57655932a585b4ae1a0254d8449ecfc72`。
- GT-box pair：native GT detection，octagon vs rectangle。
- Pred-box pair：同一签名 B-cache detector geometry + oracle GT class，octagon vs rectangle；unmatched FP 不进入 Flow。
- 两对均为 memory-off、RL-off、2 outer × 4 inner、AB2=8 NFE、seed 20260731、batch1、同case/slice/instance顺序。
- observer 对每个 Flow call、每个 outer 的 CPU/CUDA RNG digest、`py_ind`、stage shape 做逐项核对；不一致即实验无效。
- 主要质量：stage_init、outer1、outer2/final 的逐实例2D Dice/IoU、失败计数和最差/最好病例。
- 次要结果检查：逐卷物理3D Dice/IoU、NSD@2mm、HD95；这些指标不用于方法扩展或调参。
- 阶段行为：feature-grid 点位移动，仅用于确认 rectangle 分支真实生效及演化过程正常。
- 参数量、调用数和时序只作同窗口诊断，不作为部署速度。

## 最终结果（v3，非 locked dev5）

### 合同与运行有效性

- 四臂均为 `PASS`；每臂均处理5 volumes / 1248 slices。GT-box 每臂4340个实例、422次非空 Flow call；detector-box 每臂1707个 matched 实例、358次非空 Flow call。
- GT 与 detector 两个配对的切片调用身份、`py_ind`、stage shape、每个 outer 前 CPU/CUDA RNG digest mismatch 均为0；共同噪声、顺序和 seed 合同通过。
- 四臂 checkpoint、配置、缓存、模型参数量、NFE、solver、病例列表和源码 SHA 一致；locked病例访问为空。逐阶段坐标经冻结 validation inverse affine 恢复后，与 evaluator 已保存 final polygon 的逐点 `max_abs=0`。
- Memory read delta 全程为0；没有接入邻片、Memory、2.5D 或 3D。评估结束后无 GPU/Flow 进程。

### 主要口径：逐实例等权2D Dice/IoU

| box条件 | init形状 | N | init mDice | outer1 mDice | final mDice | final mIoU |
|---|---:|---:|---:|---:|---:|---:|
| GT-box | octagon | 4340 | 0.734943 | 0.747407 | 0.750132 | 0.613461 |
| GT-box | rectangle | 4340 | 0.711806 | 0.742261 | 0.751330 | 0.613101 |
| detector-box + oracle class | octagon | 1707 | 0.582022 | 0.604295 | 0.609896 | 0.469057 |
| detector-box + oracle class | rectangle | 1707 | 0.563285 | 0.601419 | 0.612774 | 0.469609 |

Rectangle 的初始重叠更低：GT-box init mDice `-0.023138`，detector-box init mDice `-0.018737`。冻结 Flow 两轮演化后差距被追回，但最终收益很小：

- GT-box：final mDice `+0.001197`，final mIoU `-0.000360`（rectangle−octagon）。
- detector-box + oracle class：final mDice `+0.002878`，final mIoU `+0.000552`。

逐病例 final mDice 并不一致。GT-box 的5例中 rectangle 在3例更好、2例更差；最差为 `sub-verse022 -0.009916`，最好为 `sub-verse264 +0.006952`。detector-box 的5例中3例更好、2例更差；最差为 `sub-verse024 -0.005853`，最好为 `sub-verse264 +0.007814`。因此主口径只支持“几乎打平、轻微正向”，不支持稳定优势。

阶段演化确认 rectangle 分支真实生效。GT-box 的 init→final 平均点位移动为0.7215 feature px（octagon 0.6136）；detector-box 为1.3549 feature px（octagon 1.2096）。rectangle 需要 Flow 做更大修正，最终才达到近似相同质量。

### 失败与覆盖

- 空预测volume和surface-empty volume：四臂均为0。
- GT-box 前景切片无预测：两形状均7；detector-box为71。形状切换没有改变覆盖或失败计数。
- 逐实例 zero-Dice：GT两臂 init 各1，outer1/final均0；detector两臂所有阶段均0。
- detector arm 只包含签名 Stage-A B cache 的 matched rows，并使用 oracle GT class；unmatched FP不进Flow，因此这不是部署口径。

### 次要结果检查：逐卷物理体积指标

| box条件 | init形状 | mean-volume Dice | NSD@2mm | HD95 mm | foreground precision | foreground recall |
|---|---:|---:|---:|---:|---:|---:|
| GT-box | octagon | 0.795856 | 0.813689 | 3.5935 | 0.7794 | 0.8136 |
| GT-box | rectangle | 0.808579 | 0.802225 | 4.1818 | 0.7201 | 0.9226 |
| detector-box + oracle class | octagon | 0.600703 | 0.594574 | 9.6902 | 0.6887 | 0.5333 |
| detector-box + oracle class | rectangle | 0.631959 | 0.606879 | 9.6137 | 0.6416 | 0.6239 |

逐卷 Dice 在两对的5例中都上升（GT `+0.012723`；detector `+0.031256`），但 rectangle 同时明显扩大预测前景。GT-box 的 recall 从0.8136升至0.9226，而 precision从0.7794降至0.7201，NSD下降0.01146且HD95恶化0.5883 mm。detector-box 的 recall从0.5333升至0.6239，precision从0.6887降至0.6416，NSD升0.01231、HD95改善0.0765 mm。

这与逐实例等权2D结果的微小、病例不一致收益形成明显反差，说明体积 Dice 的上升主要伴随更大的前景覆盖和体素加权效应，不能单独解释为轮廓初始化更准确。按最新项目边界，这些体积值只作结果检查，不升级为3D方法结论。

### 判断与处置

本次没有预注册 rectangle 晋级阈值，且仅为5例探索性开发消融。基于主要逐实例2D口径，**不把 rectangle 替换为默认初始化，也不据此启动 full38、训练或新结构实验**。保留可开关实现与完整证据：它证明冻结 Flow 能把较差的矩形初态追回到与 octagon 基本相当，但尚未证明稳定优于训练匹配的 octagon。

当前默认仍应保持 octagon。若项目负责人以后单独授权 full38 2D确认，必须冻结本次实现与配对合同，并以逐实例 final mDice/mIoU和病例一致性为主门，体积指标只作辅助；本任务不会自行启动该工作。

### 机器结果

- 机器汇总：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/rectangle_init_ablation_dev5_v3_20260807/rectangle_init_ablation_metrics.json`，SHA256 `495b48cbf89813acdabea98705df1afc5bd766bf38f7152b4d7f24d73457ed8f`。
- CPU汇总代码：`tools/volmem/compute_rectangle_init_ablation_metrics.py`，SHA256 `e840c1d4471d10093f9016ff6bac72f8a21aab1640e9b4c8a767bfc97ded7f52`。
- 静态合同：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/rectangle_init_ablation_static_v2_20260807/rectangle_init_contract.json`，SHA256 `8148fb5290342610984ddc572e0188b6b7bf5c33b9dd23560829ee33285092c5`。

## 审计备注

- `rectangle_init_ablation_dev5_v1_20260807`：首次在模型加载前因相对 stats 路径错误停止；未读病例、无模型输出，保留不覆盖。
- `rectangle_init_ablation_dev5_v2_20260807`：完成一次旧代码 octagon 基线后，发现4角rectangle与octagon起点相位不同；旧rectangle进程被主动停止，v2不进入结论。
- `rectangle_init_ablation_dev5_v3_20260807`：从最终8控制点、同相位代码重新运行全部四臂，是唯一执行结论来源。
