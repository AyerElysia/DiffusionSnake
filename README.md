# DiffusionSnake: Flow Matching 轮廓演化框架

端到端医学图像实例轮廓分割框架：**检测 → 框初始化 → Flow Matching 轮廓演化**。
用 Flow Matching (FM) 重构 DeepSnake 式轮廓演化：内层为 FM 的 ODE 连续速度场积分，外层保留 Snake "边爬边采特征" 的迭代精修，两层 ODE 融合是本工作的核心方法贡献。

当前主战场已从早期 BTCV 腹部 CT（V1–V3.5 扩散原型）迁移到 **VerSe 矢状位椎体分割**（25 类，C1–C7 / T1–T12 / L1–L6）。

---

## 论文贡献层级

| 层级 | 内容 | 状态 |
|------|------|------|
| 第一贡献 | **Flow Matching 轮廓演化**：FM 速度场建模 `init → GT` 位移，推理约 8 NFE；外层按 fractions 多轮推进并在当前轮廓位置重新采样图像特征 | ✅ 已成立 |
| 第二贡献 | **Contour RL**：GRPO 后训练，以不可微几何质量（IoU / 曲率 / 毛刺 / NSD）为奖励 | 🔶 有实现，瓶颈已诊断 |
| 当前边界 | **纯 2D**；Memory / 2.5D / 3D 不再进入网络与论文贡献 | ✅ 已收敛 |
| 系统支撑 | 检测器（LocateAnything 接入）与推理加速 | 性能支撑，非核心创新 |

贡献详述见 `docs/report/INNOVATION_SUMMARY.md`（写于 2026-07-05，**战场与超参描述已落后于当前主线，阅读时以本 README 为准**）。

---

## 当前主线状态（2026-08-09）

### 冻结主线

- **结构**：纯 2D Dense-6 DiT + H1 Dense Residual 输出头（由 E8 Top-2 输出 MoE 蒸馏而来，函数相对误差 0.48%）。当前网络**不再实例化 Memory 模块，也不携带任何 Memory 参数**。
- **Checkpoint**：`data/outputs/volmem/output_head_h0_h1_h2_20260803/distilled/h1_distilled_full.pt`（SHA256 `5e28f12d…`）。
- **推理调度**：AB2，2 outer × 4 inner = **8 NFE**，outer fractions `[0.6667, 1.0]`，每个 outer 在更新后的轮廓位置重采特征。
- **特征**：冻结 MoonViT layer-18（center-only），离线缓存读取。
- **接口契约**：Flow interface manifest v1.1（`label_id 1..25 → flow_class_id 0..24`）。
- 冻结细节见 `docs/report/FLOW_MAIN_HANDOVER_STATUS_20260804.md` 与 `docs/report/FLOW_GT_ORACLE_AND_INTERFACE_STATUS_20260804.md`。
- **初始化形状**：**Route B（矩形框构造八边形）**。训练侧取 GT 轮廓的轴对齐外接矩形框，推理侧取检测器矩形框；两侧统一调用 `get_octagon(get_quadrangle(box))`，由框四边中点构造12点八边形，再直接均匀采样为128点（`evolve_init: bbox_octagon` / `init: octagon`）。训练不再使用 GT 极值点。主线确认 2026-08-09，详见下方§训推初始化统一。
- **外层 frac 采样**：**v4_10 MoG 连续采样**，centers `[0, 0.3333, 0.5, 0.80, 0.97]`，σ=0.05，15% 均匀底噪，开关 `v4_10_use_continuous_sampling: true`。主线确认 2026-08-09，详见下方§外层状态采样连续化。
- **模型边界**：Flow 模型组件为 **14,373,444 参数**，Memory 参数 **0**，内置 heatmap detector 参数 **0**；`flow_box_only` 后端在训练时接收 GT 框、部署时接收经校验的外部检测框。检测器是单独系统组件，不计入 Flow 模型参数。历史 39,865,760 参数模型只作证据来源，不再作为训练/推理主线。
- **训练入口**：`diffusion_train.py` + `configs/volmem/depth_sweep/pure2d_mainline_l6_f256_routeb_v410_48h.yaml`。训练配置中不再存在 `memory_capacity`、`memory_dim`、Memory reader/controller 或 memory learning-rate 参数。

### 关键数字（历史筛选均为 GT box、Memory-off 隔离评估、seed 20260731；新主线为物理无 Memory 网络）

| 实验 | 协议 | Volume Dice |
|------|------|---:|
| GT-oracle 三病例隔离上界 | 3 例 / 333 slices / 8-NFE AB2 | 0.7940（0.7960 / 0.8168 / 0.7691） |
| full-38 D 条件（完整 GT） | 38 例 / 6160 slices / 31,772 实例 | 0.7940（mean-volume）；NSD@2 0.8094 |
| v0.5 step2300 / H1 历史锚点 | 3 例 / Batch-8 | 0.7967 |

> ⚠️ **口径非标警告**：上表 Dice 为 volume-level 前景池化二值 Dice、NSD@2 为 voxel 单位，均**不符合** VerSe 论文与相关工作标准协议。在按标准协议重跑前，这些数字**不得**直接对标 VerSe 榜单；对外引用须加注"非标准口径"。完整规范见 `docs/report/eval_protocol_standard_20260809.md` 与下方「评估协议标准」章节。

### 评估协议标准（必须与 VerSe 论文及相关工作对齐）

**硬要求（Ayer 2026-08-09 定）**：本项目评估口径必须与 VerSe 官方协议（Sekuboyina et al. 2021；官方评测 `anjany/verse`）及相关工作惯例（Metrics Reloaded 2024、nnU-Net、SpineNet/VerTeBra）**完全一致**。当前主线评估在 5 个维度偏离，待按规范实现。

标准协议要点（完整规范见 `docs/report/eval_protocol_standard_20260809.md`）：
1. **单位**：从 manifest 读 spacing，重采样到 1mm 各向同性；所有表面/Hausdorff 指标用 **mm**，禁止 voxel。
2. **Dice**：**逐椎体平均**（aggregate + single 两种），禁止 volume-level 前景池化二值 Dice。
3. **NSD**：报告 **NSD@1mm / NSD@2mm（mm）**（Nikolov 2021，经 Metrics Reloaded 成为领域标准）。
4. **Hausdorff**：报告 **HD95(mm)**，可选 full HD(mm)。
5. **识别/分割分离**：报告 **ID rate** + **MLD 20mm 门控**；仅正确识别椎体计入分割指标，错标不冒充成功；另报 FP/FN。

当前偏离：Dice 池化 ❌ / NSD voxel ❌ / HD95 voxel ❌ / 无识别分离 ❌ / 无 spacing(mm) ❌。**任何 AI 改评估代码须先对齐本规范**，实现结果写入 `docs/report/` 后再提交（AGENTS §7/§8）。

**实施状态（2026-08-09 更新）**：旧 `eval_memflowdit_v03.py` 的 volume-level 前景 Dice 仍不能直接对标 VerSe leaderboard；它没有 per-vertebra ID matching 和毫米物理距离口径。深度试验现已通过独立的原空间 3D exporter/evaluator 补齐 scan-equal per-vertebra Dice、identification rate、dmean/maximum-HD（mm），并把 NSD@2mm/HD95 明确保留为项目诊断指标。实现与边界见 `docs/report/EVAL_PROTOCOL_IMPLEMENTATION_20260809.md`。

评估基础设施已纳入主线：`tools/volmem/verse_eval/verse_metrics.py` 实现 per-vertebra Dice、ID rate @ 20mm、HD95(mm) 和 NSD@1/2mm；9 种已知扰动的 calibration 为 **8/9 PASS**。Identity 指标全部达到理想值，`label_shift_+1` 下 pooled binary Dice 仍为 1.0 而 VerSe Dice 降为 0.0，证明新口径能识别标签身份错误。唯一未过阈值的 PNG↔NIfTI 检查 Dice=0.937，已披露为重采样差异；完整结果与来源追溯见同一实施报告。

**深度扩容结论（P0 已关闭）**：相同 Train72/2000 steps/随机流/GT-box/Memory-off/Dev5/8-NFE 下，P1 L8/F256 相对 P0 L6/F256 多 3.843M 参数和 0.201 GiB 峰值推理显存，但 5/5 病例 native volume Dice 均下降；均值 −0.000651，VerSe scan-equal Dice −0.000404，NSD@2mm −0.000980。主线保留 **L6/F256**，不采用 L8/F256。完整机器结果、坐标修复说明与 SHA 见 **`docs/report/DEPTH_WIDTH_DEV5_FINAL_20260809.md`** / `.json`。最初三模型全零的结果来自检测框 512 坐标误采样 128 特征图，已作废。

**NFE 筛选结论（2026-08-09）**：固定 P0 L6/F256 step2000、同一 Dev5/GT-box/Memory-off/B16/seed 后，4/8/12 NFE 的 native volume Dice 分别为 0.789226 / **0.801049** / 0.800487。4 NFE 明显退化；12 NFE 比 8 NFE 多 50% denoiser 调用，却没有 volume Dice 增益，且 4/5 病例下降。因此主线固定为 **AB2 2 outer × 4 inner = 8 NFE**。共享存储使本次墙钟不稳定，速度不按 E2E 秒数排序，而按确定的 NFE/denoiser 调用预算解释。详见 `docs/report/DEPTH_WIDTH_NFE_SWEEP_FINAL_20260809.md` / `.json`。

### Pure-2D 物理精简（2026-08-09，当前执行主线）

这里的“去掉 Memory”不是把开关设为 off，而是从模型结构和 checkpoint 中删除相应参数：

| 组成 | 参数量 | 当前主线 |
|------|------:|----------|
| 历史完整 H1（内置 detector + Memory + Flow） | 39,865,760 | 仅历史证据 |
| 内置 heatmap detector | 23,308,124 | 删除；由外部检测器负责 |
| Memory encoder/controller | 2,184,192 | **删除** |
| 纯 2D L6/F256 Flow + MoonViT replacer + H1 | **14,373,444** | **当前模型** |

P0 step2000 的纯 2D 基网已机械导出为 156 个 state keys；与已签纯 2D L6/F256 参考网络的 key 集合和 tensor shape 全部一致。主训练 checkpoint 为
`data/outputs/depth_sweep/pure2d_mainline_l6_f256_routeb_v410_48h/source/p0_step2000_memory_free.pt`
（SHA256 `737409fad5e60f8e72447a8f1079f4f4cfef5ac819647c30db57a5171c34ae32`）。GPU 严格加载结果为
missing/unexpected=0；构造后 Memory 参数=0、内置 detector 参数=0、总参数=14,373,444。

删除前后同输入、同噪声的 GT-box 训练协议逐阶段对照已经通过：`stage_init`、`outer1`、最终返回
`ret.py` 与 `py_ind` 均 `torch.equal=true`、`max_abs=0`。机器证据位于
`data/outputs/depth_sweep/pure2d_mainline_l6_f256_routeb_v410_48h/validation/p0_memory_free_exact_v5/`
（JSON SHA256 `f61a8bbcd4a513534383e35146782f789bd5102d04b307f9b2113b19078c9920`）。
外部检测器的参数与耗时必须单独报告，不能把 14.37M 写成完整部署系统总参数。


### 纯 2D 无内置检测器推理（2026-08-10 已验证）

标准入口为 `tools/volmem/depth_sweep_tools/run_pure2d_detector_free_inference.py`。它只构造
14,373,444 参数的 L6/F256 Flow 网络，逐键严格加载纯 2D checkpoint；不会构造 Memory wrapper，
也不会构造 heatmap/YOLO 检测器。科研评估固定使用 GT 矩形框与 GT 类别 oracle，矩形框按 Route B
执行 `get_octagon(get_quadrangle(box))`，再直接均匀采样到 128 点；推理轨迹固定 AB2、2 outer ×
4 inner = 8 NFE。部署时把 GT 框替换为外部检测器输出的 `[B,N,6]`，其 512 输入坐标在进入
128 Flow 网格前除以 `down_ratio=4`，初始化几何不变。

已验证 checkpoint 为续训 `step_8000.pt`（SHA256
`340e4a4734c003ac948e65a012cc11c8829b7589c7db68987c2be4e26fe2a7f7`）。在非锁定 Dev8
（8 cases / 1123 slices，seed 20260731）上：VerSe-2021 scan-equal per-vertebra Dice
**0.841594**，识别率 **1.000000**，maximum HD 均值 **7.0007 mm**，命中椎体 dmean
**12.2042 mm**；前景切片 Dice **0.795535**、IoU **0.673552**，缺失椎体率 0。
机器结果与可视化位于
`data/outputs/depth_sweep/pure2d_mainline_l6_f256_routeb_v410_48h/inference/pure2d_detector_free_step8000_dev8_v1/`。
历史同 Dev8/GT-box/8-NFE 的 A0 step2000 scan-equal Dice 为 0.767668；继续训练到 step8000
后提高 0.073927。该对比支持“生成式演化不能因 loss 平台提前停止”的长训原则。

### Detector Stage A 端到端损失归因（full-38，2026-08-05）

**结论：当前端到端损失主要来自检测器覆盖不足，其次是 matched box 定位几何，不能归因于 Flow。**

| 因子 | mean-volume Dice 损失 | NSD@2 损失 |
|------|---:|---:|
| D→A coverage（检测覆盖，recall 0.4165） | 0.1293 | 0.1767 |
| A→B geometry（框定位，oracle class） | 0.0894 | 0.0636 |
| D→B 合计 | 0.2190 | 0.2406 |

38/38 病例方向一致为下降，10,000 次 paired bootstrap 95% CI 均不跨 0。条件 C（predicted class）因无已登记分类器 blocked。详见 `docs/report/DETECTOR_STAGE_A_AB_ATTRIBUTION_STATUS_20260805.md`。

### 训推初始化统一（2026-08-09，A/B 已判定：主线走 Route B）

**问题**：训练用 GT 轮廓极值点构造八边形，推理只有检测框、构造不出同一个八边形
（LocateAnything 不输出极值点）。同一个 `get_octagon()`，两侧输入分布不同，
合同测试实测控制点逐点最大偏差 **102.4 px**。

**决定的主线（其他 AI 以此为准）**：训练侧使用 GT 轮廓的轴对齐外接矩形框，推理侧使用
检测器矩形框；两侧都从矩形框四边中点构造12点八边形，即
`get_octagon(get_quadrangle(box))`。开关 `evolve_init: bbox_octagon`（训练侧）+
`init: octagon`（推理侧），代码在 `lib/utils/snake/snake_voc_utils.py: get_evolution_init()`。
**训练初始化不再使用 GT 极值点。**

**证据**（dev5 = sub-verse022/024/071/150/264，1248 slices，GT box、Memory-off、step 600、
seed 20260731，三臂唯一差异就是 init 开关）：

| 臂 | 训练 init | 推理 init | 前景切片 mDice | 逐卷胜负 |
|----|----------|----------|---:|---|
| baseline（现状，训推不一致） | GT 极值点八边形 | 框中点伪八边形 | 0.760831 | — |
| route_A（统一 8 点矩形） | box 矩形 | box 矩形 | 0.788292 | 5W/0L vs baseline |
| **route_B（统一 bbox 八边形）** | 框中点八边形 | 框中点八边形 | **0.790968** | **5W/0L vs baseline** |

结论：**形状本身几乎不重要，训推一致才重要**（+0.027~0.030）。这与 2026-08-07 的
Rectangle 消融不矛盾——那次在冻结权重上只换推理侧形状（两臂都仍不一致），测的是
"冻结 Flow 能否吸收形状变化"（能，+0.0012 打平）；本次测的是"消除不一致值多少"。
选 B 不选 A：A 的索引对齐优势（16.35 vs 42.67 px）**全部来自重采样链**
（`p128_maxabs == resample_only_maxabs` 精确相等），修掉重采样链后归零，
而 B 的初始形状质量优势仍在；B 也拿下主口径（4/5 卷、10/17 类）且改动最小。

详见 `docs/report/INIT_TRAIN_INFER_UNIFICATION_20260808.md` 与
`data/outputs/init_unify/eval_dev5_gtbox_step600/COMPARISON.md`。

### 外层状态采样连续化（2026-08-09，**v4_10 已确认为主线训练采样配置**）

`frac` 语义（源码确证，勿反着读）：`i_init_train += full_disp * frac`、
`x1_raw = full_disp * (1 - frac)`，所以 `frac` = **已走完的 GT 位移比例** = 外层进度。
`frac=0` 就是推理第 1 步永远面对的原始初始轮廓。

**现状实测缺陷**（2M 采样，`tools/volmem/analyze_outer_state_sampling.py`）：

- `frac≈0`（±0.05）只占 **1.25%**，但 100% 的推理轨迹从这里出发，且这一步位移最多；
- `v4_9_discrete_fractions: [0.3333, 0.5, 1.0]` 被当**绝对进度**消费，`1.0` 又被
  `clamp_(0, 0.999)`，13.3% 样本浪费在近退化状态；根因是单位混用
  （`iterative_fractions` 是**残差**比例，`v4_9_infer_target_fractions` 是**绝对进度**）；
- 28.3% 样本落在 ≥0.95 进度，中位数 0.823——分布严重偏向"快到终点"。

**新设计**（`tools/volmem/design_continuous_sampling.py`，连续 + 按工作量加权）：
中心取 `{0} ∪ infer_target_fractions[:-1]` = `[0, 0.3333, 0.5, 0.80, 0.97]`，
权重 ∝ 该外层步要走的**绝对进度份额** `[33.3, 16.7, 30.0, 17.0, 3.0]%`，
再与均匀分布按 λ=0.30 混合防过窄；每个中心是折叠/反射高斯 σ=0.05，
另加 15% 均匀底噪覆盖 [0, 0.999]。

| 状态 | 现状 | 新设计 |
|------|---:|---:|
| 0.0 | 1.25% | **17.78%** |
| 0.3333 | 15.24% | 11.99% |
| 0.50 | 15.21% | 17.33% |
| 0.80 | 10.22% | 11.93% |
| 0.97 | 33.31% | 7.01% |
| ≥0.95 | 28.31% | 4.87% |
| 中位数 | 0.823 | 0.449 |

概率可视化：`data/outputs/init_unify/quantification/outer_state_sampling_design.html`。

**实现状态（2026-08-09 已完成）**：
- v4_10 branch 已插入 `lib/networks/diffusion/flow_matching_evolution.py`，
  门控开关 `v4_10_use_continuous_sampling: true`（优先级高于 v4_9 block，v4_9 推理调度不变）。
  **主线配置**：`configs/volmem/init_unify_route_B_v410.yaml`（Route B + v4_10，2026-08-09 确认为主线训练配置）。
  commit: `01f5304`。

**坐标与重采样链统一（2026-08-09 已修）**：训练侧的 `i_gt_py` 与 Flow 特征都位于
128×128 stride-4 网格；检测器框则位于 512×512 输入坐标。推理侧先将有效框除以
`snake_config.down_ratio=4`，再调用 `_box_to_octagon_init(box, poly_num)`，由框四边中点
构造12点八边形并直接均匀上采样为128点。训练侧
`build_box_octagon_from_poly(i_gt_py)` 使用 GT 轮廓的矩形外接框调用同一函数，因此训推的
**坐标尺度、八边形控制点和128点重采样完全一致**；训练不读取 GT 极值点作为初始化。
合同测试确认 train/infer 逐点 `max_abs=0`，并明确检测框到 Flow 网格的缩放比为4。

### 已淘汰路线（有严格证据，不再回退）

- **输出 MoE**：H1 蒸馏严格支配（质量保持、头参数 -63.6%、Batch-8 吞吐 +27.7%）。
- **3D Memory v0.7 / v0.8 / v0.9 及后续 Memory reader**：严格门控下均无稳定净收益，已从当前网络结构、checkpoint 与论文贡献中删除。历史实验只保留作审计证据。
- **数据管线 largest-only**：旧管线只保留 81.88% 前景，是早期演化失败的根因；修复后（每类 top-4 significant components、面积 ≥2、cap 32）前景保留 99.5%+。

---

## 方法流水线

```
矢状位切片
   │
   ├─▶ MoonViT (冻结, layer-18) ──▶ 视觉特征（离线缓存）
   │
   ├─▶ 检测：LocateAnything 离线预测 ──▶ external_detection [B,N,6]
   │        (x1,y1,x2,y2,score,class_id)；隔离实验用 GT box
   │
   ├─▶ 框初始化：512输入坐标检测框 → ÷down_ratio → 128×128 Flow网格框
   │        → 4边中点 → 12点八边形 → 直接均匀上采样为128点轮廓
   │        （训练/推理调用同一 _box_to_octagon_init；训练不再用GT极值点）
   │
   └─▶ Flow Matching 演化（两层 ODE）
        ├─ 内层：FM 速度场 v(x_t, t)，AB2 积分 4 NFE（≈推理 8 步内）
        ├─ 外层：fractions [0.6667, 1.0] 多轮推进，每轮在当前轮廓位置重采特征
        └─ 输出头 H1：linear(x + residual_mlp(x)) → 位移 → 最终轮廓 [N, 128, 2]
```

训练目标：归一化位移 `x1 = (GT − init) / contour_scale`，线性插值 `x_t = (1−t)·x0 + t·x1`，速度目标 `v = x1 − x0`，损失 `MSE(v_pred, v)`。
几何先验：闭环拓扑位置编码（CyclicRoPE）、轮廓点序、法向/切向局部上下文采样。

### 数据契约（sagittal_2d_fixed）

- 每类最多 4 个显著连通域，raw contour area ≥2，单切片全局 cap 32；
- 输出网格多边形面积 > `min_poly_area_output: 0.5`，退化框剔除；
- `label_id 1..25`（解剖 mask 标签）→ `instance_id`；`flow_class_id = label_id − 1`；
- 特征采样统一 `border` padding（修复旧 zero-padding 边界断点）。

---

## 五任务分工（2026-08-04 起）

| 任务 | 职责 |
|------|------|
| 轮廓演化 / Flow 主线 | FM 方法、训练目标、采样路径、8-NFE 冻结调度、inner/outer 归因 |
| 推理加速 / 批量并行 | 真实 pass / DiT calls / 吞吐 / 显存；不再包含 Memory 路线 |
| 检测器 / 初始化与覆盖 | LocateAnything、检测缓存、coverage/geometry/class 隔离归因 |
| 强化学习 / Contour RL | 在冻结 Dense-6 + H1 主线上单独验证 GRPO 收益 |
| 论文统筹 | 证据审计、数字机器可读落盘、中英文写作 |

归因铁律：GT box 隔离轮廓演化能力，predicted box 评估部署链路，两者不混用；oracle class 不得冒充 predicted class；任何部署质量下降先做 detector/evolution isolation。

---

## 关键文件索引

### 核心模块

| 文件 | 说明 |
|------|------|
| `lib/networks/diffusion/flow_matching_evolution.py` | FM 演化核心：训练目标、geom bridge 推理、外层迭代精修、位移归一化 |
| `lib/networks/diffusion/dit_denoiser_v4.py` | Dense DiT + H1 Dense Residual 输出头（含 SharedDenseSparseResidualHead） |
| `lib/networks/diffusion/prototype_phi_moe.py` | DiT FFN-MoE（E4K1 prototype 路由 + φ-balancing，研究候选） |
| `lib/networks/snake/ct_snake.py` | 主网络接线（检测 → 初始化 → 演化） |
| `lib/datasets/sagittal_2d_fixed/snake.py` | 矢状位数据契约（significant components、border 采样） |

### 训练 / 评估 / 工具

| 文件 | 说明 |
|------|------|
| `diffusion_train.py` | 当前纯 2D Flow 训练入口；构造时硬断言 Memory=0、内置 detector=0、参数量=14,373,444 |
| `tools/volmem/eval_memflowdit_parallel.py` | 整卷并行评估、蒸馏轨迹缓存 |
| `tools/volmem/depth_sweep_tools/run_pure2d_detector_free_inference.py` | 当前主线无 Memory、无内置检测器的直接推理、Dev8 3D 指标与 Route-B 可视化 |
| `tools/volmem/distill_output_head.py` | H1/H2 输出头蒸馏、权重移植、参数统计 |
| `tools/volmem/compute_stage_a_metrics.py` | Stage A 冻结指标（NSD@2 / HD95） |
| `configs/volmem/depth_sweep/pure2d_mainline_l6_f256_routeb_v410_48h.yaml` | 当前 L6/F256、Route B、v4.10、8-NFE、无 Memory 长训配置 |
| `scripts/extract_sagittal_moonvit_features.py` | MoonViT 特征离线提取 |

### 工作留档（docs/report/）

| 报告 | 内容 |
|------|------|
| `INNOVATION_SUMMARY.md` | 三大创新点总结（论文 Contributions 形式） |
| `FLOW_MAIN_HANDOVER_STATUS_20260804.md` | Flow 主线职责、Dense-6 + H1 冻结、归因规则 |
| `FLOW_GT_ORACLE_AND_INTERFACE_STATUS_20260804.md` | GT-oracle 隔离结果与接口冻结契约 |
| `DETECTOR_STAGE_A_STATUS_20260804.md` | 检测器 Stage A：契约冻结与四条件定义 |
| `DETECTOR_STAGE_A_AB_ATTRIBUTION_STATUS_20260805.md` | full-38 A→B 质量归因（coverage/geometry 损失分解） |
| `DETECTOR_STAGE_A_D_ZERO_CONTROL_STATUS_20260805.md` | D 条件 zero-control 验证 |
| `OUTPUT_HEAD_DISTILLATION_H0_H1_H2_20260803.md` | 输出头蒸馏实验（H1 胜出） |
| `MEMFLOWDIT_RECENT_WORK_REPORT_20260731.md` | 数据根因修复、MoE 消融、Memory v0.7–v0.9 全程记录 |
| `FLOW_MEMORY_3D_READONLY_REVIEW_20260804.md` | Memory / 3D 只读复核与淘汰结论 |
| `LOCATEANYTHING_DIFFUSIONSNAKE_INTEGRATION_REPORT_2026-07-31.md` | LocateAnything 检测接入契约 |
| `MEMFLOWDIT_NEXT_STAGE_EXECUTION_20260803.md` | DiT FFN 对照、Memory 因果审计执行链 |
| `DETECTOR_STAGE_A_INSTANCE2D_RECALC_20260807.md` | 逐实例 2D 指标探索性重算（只读；**不替代**正式 3D mean-volume Dice） |
| `DETECTOR_RECTANGLE_INIT_ABLATION_20260807.md` | bbox→初始轮廓几何消融（5 病例开发集，非 full-38 正式结果） |
| `INIT_TRAIN_INFER_UNIFICATION_20260808.md` | 训推初始化统一：A/B与baseline已完成，Route B bbox-octagon成为主线；坐标与128点重采样合同 `max_abs=0` |
| `FLOW_PURE2D_DIT4_TOTAL10K_BASELINE_AND_SLIM_B_GATE_20260807.md` | Pure-2D DiT-4 10k baseline 未达 H1 参考，slim-B 判为 NO-GO |
| `HISTORICAL_BEST_AUDIT_20260803.md` | 175 份 summary.json 审计，纠正“0.773345 是历史最佳”的误述 |
| `DETECTOR_EVOLUTION_ISOLATION_20260803.md` | 检测器与演化的隔离规则（含 2026-08-04 活契约条款） |
| `DISP_NORMALIZATION_REPORT.md` | disp 归一化统计规范——`flow_matching_evolution.py` 中 `_load_disp_stats/normalize_disp` 的唯一书面依据 |
| `GEOM_BRIDGE_PARADIGM_DESIGN_AND_RESAMPLE_DEFERRAL_20260620.md` | Geom Bridge 几何位置桥范式设计与重采样延后决策 |
| `LOCATE_FEATURE_REPLACEMENT_ARCHITECTURE_EXPLAINED_20260615.md` | `ct_snake.py` 中 `LocateFeatReplacer` 的架构说明（代码仍在用） |
| `locate_integration_analysis_20260611.md` | Locate 接入分析 + 2026-06-12 结果更新与推荐配置 |
| `RL_WORK_SUMMARY_20260708.md` | RL 阶段总结（§5 仍有未完成 TODO） |
| `RL/POLICY_GRADIENT_GRPO_EXPLANATION.md`、`RL/V5_GEOM_POLICY_GRADIENT_DETAILED.md` | 现行 RL 入口 `grpo_train_v5_geom_action.py` / `grpo_train_v7_seedflow_grpo.py` 的原理说明 |
| `CREDIT_DIAGNOSIS_FINDINGS_20260625.md` | RL 反馈信号离线诊断（reward 计算 vs 轨迹问责两条假说） |
| `inference_acceleration/ODE_ACCELERATION_REPORT_20260706.md` | 推理加速：AB2 积分器与 KV cache 基准 |
| `2D_FM_CURVE_FAILURE_THEORETICAL_ANALYSIS_20260626.html` | 2D FM 曲线失效的理论分析 |

### 设计文档（docs/design/）

| 文档 | 内容 |
|------|------|
| `VOLMEM_NAMING_AND_CODE_BOUNDARY.md` | VolMem 体数据主线的命名约定与代码边界（`volmem` 前缀、工作名 VolMemSnake） |

### 历史留档（docs/archive/）

已完成使命的文档与实验产物统一收于 `docs/archive/`，目录结构镜像整理前的原始路径。
索引见 `docs/archive/README.md`。**该目录不指导当前工作**，仅供追溯 BTCV 时代原型、
已淘汰路线（MoE / 3D Memory / per-point FM 尺度策略）的完整证据链，以及历史评测产物。

---

## 遗留版本说明（2026-03 ~ 2026-04，BTCV 时代）

早期基于 DDPM/DDIM 的扩散轮廓演化原型（V1–V3.5）在 BTCV 腹部 CT 上开发，包括：
V1 基础 Cross-Attention、V2 CyclicRoPE 奇偶交替、V2.2 MM-DiT Patchify、V2.3/V3.2 Flow Matching、V3 八边形初始化、V3.3 Circular Conv1d、V3.5 傅里叶空间扩散。
其后 V4 系列引入 MoE 输出头与 geom bridge，GRPO v5 引入几何动作 RL。

这些版本已被当前 FM 主线取代，仅作历史参考：

- V2 系列封存于 `archive/v2_legacy_2026-04-19/`；
- 早期文档（八边形初始化、DDIM 采样、边缘平滑、单样本过拟合流程等）见 git 历史中的旧版 README；
- 早期验证脚本（`verify_octagon_v3.py`、`edge_smoothing.py`、`compute_disp_stats.py` 等）已于 2026-08-08 清理，需要时从 git 历史找回；
- BTCV 时代 RL 训练脚本（grpo_train v1/v2/v4 系列）同样已清理，当前 RL 入口为 `grpo_train_v5_geom_action.py` 与 `grpo_train_v7_seedflow_grpo.py`。

---

## 快速开始

```bash
conda create -n snake1 python=3.10
conda activate snake1
pip install torch torchvision diffusers opencv-python numpy pyyaml tqdm

# 纯 2D 主线训练（无 Memory 参数、无内置 detector 参数）
python diffusion_train.py \
  --cfg_file configs/volmem/depth_sweep/pure2d_mainline_l6_f256_routeb_v410_48h.yaml

# 纯 2D 直接推理（科研评估：GT box + GT class；无 Memory / 无内置 detector）
CUDA_VISIBLE_DEVICES=5 python tools/volmem/depth_sweep_tools/run_pure2d_detector_free_inference.py \
  --project-root "$PWD" \
  --config "$PWD/configs/volmem/depth_sweep/pure2d_mainline_l6_f256_routeb_v410_resume6000.yaml" \
  --checkpoint "$PWD/data/outputs/depth_sweep/pure2d_mainline_l6_f256_routeb_v410_48h/training/pure2d_l6_f256_48h_v1_resume6000/checkpoints/step_8000.pt" \
  --result-dir "$PWD/data/outputs/depth_sweep/pure2d_mainline_l6_f256_routeb_v410_48h/inference/pure2d_detector_free_step8000_dev8_v2" \
  --metric-module /home/medteam/Zhrch/DiffusionSnake-12-30-pure2d-verse3d-eval-20260808/tools/verse2021_3d/verse2021_3d.py \
  --slice-manifest /home/medteam/Zhrch/detect_3D_lgz2/datasets/sagittal_2d_fixed/manifests/slice_manifest.csv \
  --case-metadata /home/medteam/Zhrch/detect_3D_lgz2/datasets/sagittal_2d_fixed/manifests/case_metadata.csv \
  --locate-feat-cache-root /home/medteam/Zhrch/DiffusionSnake-12-30/data/sagittal_moonvit_cache

# MoonViT 特征离线提取（训练/评估前必需）
python scripts/extract_sagittal_moonvit_features.py
```

---

## 更新日志

- **2026-08-10**：新增并验证纯 2D 直接推理入口：物理网络 14,373,444 参数、Memory=0、内置 detector=0，GT 矩形框按 Route B 构造 12 点八边形，AB2 8 NFE。续训 step8000 在非锁定 Dev8 的 VerSe-2021 scan-equal Dice=0.841594、识别率=1.0、maximum HD=7.0007mm；可视化与完整机器结果已落盘。
- **2026-08-09**：主线物理删除 Memory 参数：不再使用“完整模型 + Memory-off”的形式；当前 Flow 模型为 L6/F256、14,373,444 参数，Memory=0、内置 detector=0。外部检测器作为独立系统组件计数。新增纯 2D 长训配置和构造期硬门，P0 step2000 只提取纯 2D 基网权重作为初始化。
- **2026-08-09**: 训推初始化统一 A/B 判定完成——三臂 dev5 GT-box 对照（1248 slices、step 600、唯一差异是 init 开关）：baseline 0.760831 → route_A 0.788292 → route_B 0.790968 前景切片 mDice，两条统一路线均 5W/0L；**主线定为 Route B**（训练取 GT 轮廓外接矩形框、推理取检测器矩形框，再以同一框中点规则构造12点八边形；训练不再用 GT 极值点）。量化外层状态采样缺陷（frac≈0 仅 1.25%、28.3% 样本 ≥0.95 进度、discrete/infer_target 单位混用）并给出连续化重设计（frac≈0 → 17.78%、中位数 0.823 → 0.449）。**v4_10 连续采样已实现**（commit `01f5304`，门控 `v4_10_use_continuous_sampling`，四臂配置 `init_unify_route_B_v410.yaml`）。**重采样链不一致已修**（commit `67158bb`，合同测试 max |delta|=0.000000）。
  **Route B（bbox_octagon 初始化）与 v4_10 MoG 连续采样均已确认为主线默认训练配置**（2026-08-09 用户批准）。
- **2026-08-09**：评估协议定为项目硬要求——必须与 VerSe 论文（Sekuboyina 2021）+ 相关工作（Metrics Reloaded / nnU-Net / SpineNet）对齐；当前主线 5 处偏离待修（Dice 池化 / NSD·HD95 voxel / 无识别分离 / 无 spacing-mm）。规范见 `docs/report/eval_protocol_standard_20260809.md`，README 新增「评估协议标准」章节，AGENTS §5 加评估红线、§10 加 backlog。
- **2026-08-08**: 训推初始化统一实验启动——定位并量化 init 不一致（控制点 102.4 px），两条统一路线合同测试通过（逐点精确相同），发现重采样链为第三个不一致源；训练臂进行中。docs 整理——历史留档迁入 `docs/archive/`（镜像原路径），`docs/report/` 只保留活文档；删除 636 个无唯一内容的文件（浏览器 profile 缓存、渲染自检截图、可重生成的 pptx），其余一律归档不删
- **2026-08-07**: Pure-2D DiT-4 10k baseline 未保持 H1 质量，slim-B 判 NO-GO；bbox→初始轮廓 Rectangle 消融（开发集）；逐实例 2D 指标探索性重算（不替代正式指标）
- **2026-08-05**: Detector Stage A full-38 A→B 归因完成（coverage Dice -0.1293、geometry -0.0894，38/38 一致，bootstrap CI 不跨 0）；D zero-control 通过；README 按当前主线重写
- **2026-08-04**: Flow 主线接管与五任务分工；Flow interface manifest v1.1 与 H1 checkpoint 冻结；GT-oracle 三病例隔离上界 0.7940；Memory/3D 只读复核结论
- **2026-08-03**: 输出头 H0/H1/H2 蒸馏：H1 质量保持且吞吐 +27.7%，成为主线输出头；DiT FFN 四组结构对照启动
- **2026-08-02**: MoE 成本审计；3D Memory v0.8/v0.9 严格门控失败淘汰
- **2026-07-31**: 数据工程根因修复（前景保留 81.88% → 99.5%+）；MoE 重要性消融（-0.129 Dice）；DiT FFN-MoE E4K1；输出头 hard-φ 去退化；LocateAnything 检测接入
- **2026-07-29**: VolMem v0.2/v0.3 记忆条件 Flow DiT 原型
- **2026-07-21**: MoonViT 冻结特征伪 3D 矢状位正式训练
- **2026-07-10**: RL 修复 sampled_feat 转置，移除法向策略，新增 NSD 奖励
- **2026-07-07**: per-point FM 尺度策略（tanh 有界）
- **2026-06-24**: Geom Bridge 几何位置桥范式（单桥 0.98 IoU sanity）
- **2026-06-13**: GRPO v5 几何动作与信用分配诊断；Locate 集成 E 系列评估
- **2026-04 及更早**: BTCV 时代 V1–V3.5 扩散原型迭代（傅里叶空间扩散、Circular Conv1d、八边形初始化等）

---

## 参考文献

- DeepSnake: [Peng et al., CVPR 2020]
- Flow Matching / Rectified Flow: [Lipman et al., ICLR 2023], [Liu et al., 2022]
- DiT: [Peebles & Xie, ICCV 2023]
- DDPM: [Ho et al., NeurIPS 2020] / DDIM: [Song et al., ICLR 2021]
- GRPO: [Shao et al., 2024] (DeepSeekMath)
- MoonViT / LocateAnything / SAM 2 / XMem / RMem（Memory 对照依据）

---

## License

MIT License
