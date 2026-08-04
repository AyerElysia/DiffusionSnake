# Flow 主线对 Memory / 3D 的只读复核（2026-08-04）

任务身份：轮廓演化 / Flow 主线 `019fb3e3-3c35-72d2-ab0d-418a302def49`。

本复核没有启动训练、评估或 GPU smoke，没有修改共同论文文件，也没有把 Physical
Volume Memory 的代码或 CPU 单测当作有效实验结果。

## 一、结论摘要

### 已有证据

当前严格证据不支持“现有 Memory 已产生可验证净收益”：

- 旧单遍 `frozen-feature-bidirectional` 相对同 batch、同 refinement seed 的
  `parallel-off`，四个 batch 中最好仅 `+0.000065` Volume Dice；GT / predicted
  Batch-8 分别为 `-0.000094 / -0.000093`，未达到预设 `+0.001` 保留门。
- Jacobi 双向 predicted-mask Memory 是真实 2 pass。Batch-8 端到端相对 AR 仅
  `1.18×`（GT）/ `1.14×`（predicted）；其质量也没有稳定优于同 batch
  parallel-off。
- 当前 H1 checkpoint 的 Memory 平均读出残差约为 `1.06e-4`（AR）与 `1.09e-4`
  （feature/Jacobi）。这说明 Memory 在当前 Flow 轨迹中的数值作用很小，但该数值本身
  不能单独证明“所有 Memory 都无效”。
- 只有无 3D 上下文的 parallel-off Batch-16 同时在 GT / predicted box 达到端到端
  `2.25× / 2.01×`，证明整卷并行吞吐可实现，但不能支撑 3D 有效性叙事。

### 当前判断

问题不是单一的“bank=4 太小”，而是表征、注入时机、训练制度、上下文方向、物理几何、
误差传播共同造成的弱作用。当前最可信的描述是：**已有 Memory 机制既没有学习出足够大的
Flow 修正，也没有在严格质量—速度门下提供净收益。**

### 不能声称

- 不能声称新 Physical Volume Memory 已经跑通整模型。
- 不能声称物理位置/spacing 已提高 Dice 或速度。
- 不能声称 23 项 CPU/模块测试等价于 GPU 推理结果。
- 不能用 AR 与 parallel-off 的绝对差值直接做 Memory 纯因果归因：AR 使用
  `global_sequential` 噪声流，frozen/off 使用 `per_volume_common_noise`，且 batch 组织
  不同。严格因果证据应优先使用同 frozen 协议、同 batch 的 feature/Jacobi 与 off 配对。

## 二、为什么当前 Memory 没有净收益

| 维度 | 已有证据 | Flow 侧判断 |
|---|---|---|
| 表征 | MoonViT layer-18 center-only；每层特征 adaptive-average-pool 到 8×8；与 26 通道 mask concat 后投影到 256；bank=4、无 global pool | 密集相邻层高度冗余；平均池化削弱边界细节；predicted mask 很大程度重复当前 2D 输出，新增 3D 信息有限 |
| 注入位置 | 六个 Dense DiT block 后均挂一个 Memory cross-attention；输出投影零初始化；同一 Memory 在 2 outer × 4 AB2 的 8 次 denoiser 调用中重复使用 | 没有显式区分 inner NFE、outer stage 或 Flow time 的 Memory 门控；可能在不需要修正的阶段重复注入，或被后续 block 覆盖。当前仅约 1e-4 的读出残差显示实际利用很弱 |
| 训练制度 | checkpoint step=2300；chunk length=8；TBPTT=2；predicted-evidence 从 step500 开始、4500 step 才完成 ramp、最大概率0.5 | 训练远未覆盖完整 predicted-evidence ramp，且短 TBPTT 不鼓励学习长程体积依赖；训练时与部署时的预测误差暴露仍不匹配 |
| 因果上下文 | AR 只读历史；bank=4；signed index distance | 只能传播前向局部信息，无法利用未来解剖结构；密集切片下四层物理跨度很短，不足以形成稳定长程 3D 约束 |
| 双向上下文 | Jacobi 先 coarse、再 refine，是真实双遍；feature-only 可单遍双向但无 mask | Jacobi 的双向信息以 2× Flow 调用为代价；feature-only 单遍路径未带来质量收益，说明旧 encoder/read 不能仅靠“看见未来 feature”解决问题 |
| 物理 spacing | 严格矩阵使用旧 manifest、`position_unit=index`、distance scale=4；不读取真实毫米 spacing | 非均匀间距、缺片和不同病例的层厚被当作等距索引；相同 index 距离不代表相同解剖距离 |
| 误差传播 | AR/Jacobi 的 Memory evidence 来自预测轮廓；训练 predicted-evidence ramp 未完成 | 检测框偏差、漏实例和当前层轮廓误差会写入后续 Memory；Memory 可能放大局部错误，而不是提供独立纠错信号 |
| 评估门槛 | feature 相对 off 需至少 +0.001 Dice；相对 AR 平均下降≤0.003、最差≤0.01；E2E≥2× | 旧 feature 最高仅 +0.000065，Jacobi 速度不足；因此按预先门槛应删除旧候选。三病例规模限制了统计外推，但不影响“当前模块未过门”的结论 |

## 三、真实进展分层

### A. 已形成机器证据

1. H1 Dense checkpoint 的 26/26 零训练矩阵完成，覆盖 GT/predicted box、Batch
   1/4/8/16、AR/off/feature/Jacobi、8-NFE AB2、真实 pass/DiT calls/E2E/显存/质量。
2. 旧 feature-context 未过 +0.001 保留门，应删除。
3. Jacobi 质量基本保持，但 2 pass 导致速度门失败。
4. parallel-off Batch-16 证明无 3D 的整卷并行速度下界可达约 2×。
5. H1 Dense checkpoint 固定为：
   `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/output_head_h0_h1_h2_20260803/distilled/h1_distilled_full.pt`
   ，SHA256：`5e28f12df357ec4d18fc9f0baf67b5a57655932a585b4ae1a0254d8449ecfc72`。

### B. 已实现、但只有结构/模块验证

独立加速工作区已实现一个 Physical Volume Memory 原型：

- 整卷 MoonViT feature 一次构建；
- normalized physical position + slice spacing；
- 非均匀 spacing 和 missing-slot valid mask；
- 单遍、无 coarse mask、无第二 DiT pass；
- 第一 DiT block 后一次轻量 read，zero gate；
- 新增参数解析值 134,849；
- 远端编译和 23 项模块测试通过。

这些只能证明代码结构与局部契约成立，不能证明整模型质量、速度、显存或 3D 有效性。

### C. 明确未完成

- Physical Volume Memory 的整模型 step-0 zero-gate vs off 复现；
- 成功的 GPU `summary.json`；
- state-build 开销、峰值显存、真实 DiT calls 和端到端速度；
- 冻结 H1 后的 50/200/500 step 短训；
- GT/predicted box 的 +0.001 Dice 保留门；
- 是否保留 Physical Volume Memory 的最终决定。

首次 1-volume GPU smoke 因 SSH 输出通道中断而结束，结果目录为空，没有
`summary.json`，不能作为任何实验结果。

## 四、下一步最值得做的三个动作

以下只是待项目负责人批准后的计划，本次没有执行。

### 动作 1：冻结并签名 Flow 基线契约

由 Flow 主线维护一份不可变 manifest：

- checkpoint 路径与 SHA256 固定为上述 H1；
- Dense-6 + H1 结构固定；
- solver=`ab2`，outer stage=2，inner steps=4，总 NFE=8，fractions=`[0.6667,1.0]`；
- cases=`sub-verse010/011/013`，seed=`20260731`；
- GT/predicted box 分开，predicted box 固定检测 cache/阈值/版本；
- Flow 输入、输出、噪声张量、contour ordering 和计数接口冻结。

Memory 实验不得修改 Flow 权重、输出头、NFE/stage、检测器或数据顺序。

### 动作 2：先做 step-0 配对因果验证

由加速任务执行 GPU run，Flow 主线提供/审核轨迹探针：

- 同 batch、同 volume、同 box、同初始噪声比较 off 与 zero-gate Physical Memory；
- 要求最终 contour、每个 outer stage contour、denoiser calls 完全对齐；
- 再打开非零 gate，记录每个 DiT block、inner NFE、outer stage 的 Memory residual 与轮廓
  位移变化；
- state-build≤15%，总 pass=1，E2E≥AR 2×，否则不进入训练。

### 动作 3：受限短训与轮廓失败分析

仅在动作 2 通过后：

- 冻结完整 Dense-6 + H1，只训练小 context encoder/gate；
- 依次 50/200/500 step，GT box 先证伪，predicted box 后复核；
- 同 batch off 配对，要求 Volume Dice 至少 +0.001，最差病例改善且 E2E≥2×；
- Flow 主线分析断裂、粘连、漏实例、边界偏移，并把收益归因到具体 outer stage，而非只看
  最终平均 Dice；
- 任一门失败即删除候选，不进入大规模训练。

## 五、3D 叙事是否还能成立

### 当前判断

目前还不能把 3D 写成已成立的贡献。仍可保留一个条件性机会：若单遍 Physical Volume
Memory 在冻结 Flow 下，相对 parallel-off 提供可复现质量增益、最差病例改善且保持至少
2× E2E，则可以表述为 **volume-aware sequential context / 顺序体感知扩展**。

即便通过，也不能称为 native voxel 3D、3D segmentation backbone 或与原生 3D 网络竞争。

### 若 Memory 最终仍无收益

- 从核心方法与贡献列表中删除 Memory/3D 有效性主张；
- 第一贡献保留 Flow Matching 轮廓演化，第二贡献保留 Contour RL；
- parallel-off 只作为系统推理加速模式；
- 数据仍可按病例/切片组织，但表述为 slice-wise processing of sequential volumes；
- Memory 负结果放入消融、附录或 limitation：当前体积上下文未在质量—速度门下带来净收益；
- 不使用“利用 3D 上下文提高分割”或“3D-aware performance gain”等文字。

## 六、职责划分

| 事项 | 加速 / 3D `019fc203...` | Flow 主线 `019fb3e3...` | 论文统筹 `019fc08c...` |
|---|---|---|---|
| Physical Memory 代码、GPU 集成、state build | 执行与维护 | 审核 Flow 接口不变 | 不直接实现 |
| step-0、短训、B1/4/8/16、GT/predicted | 执行并落盘机器结果 | 审核 paired noise/contour 协议 | 审核能否进入论文 |
| pass、DiT calls、吞吐、显存、纯评估/E2E | 负责 | 提供 NFE/stage 定义 | 决定报告口径 |
| Dense-6 + H1、8-NFE AB2、Flow 接口冻结 | 不得修改 | 负责签名与核验 | 监督跨模块一致性 |
| inner NFE / outer stage、轮廓轨迹与失败分析 | 提供日志 | 负责归因 | 决定论文表述 |
| 3D 是否保留、贡献层级与负结果位置 | 提供证据建议 | 提供 Flow 侧判断 | 负责方案，项目负责人最终决策 |

检测器任务还需固定 predicted-box cache、阈值和覆盖版本；否则部署质量不能归因于 Memory。

## 七、证据路径

- 机器矩阵：
  `/home/medteam/Zhrch/DiffusionSnake-12-30-par3d-h1-outputs-20260803/comparison.json`
- 运行 manifest：
  `/home/medteam/Zhrch/DiffusionSnake-12-30-par3d-h1-outputs-20260803/matrix_manifest.json`
- 完整结果表：
  `/home/medteam/Zhrch/DiffusionSnake-12-30-par3d-h1-outputs-20260803/reports/PARALLEL3D_H1_8NFE_RESULTS_20260803.md`
- Physical 原型与未完成边界：
  `/home/medteam/Zhrch/DiffusionSnake-12-30-par3d-h1-outputs-20260803/reports/PARALLEL3D_H1_STATUS_20260803.md`
- Physical 原型独立工作区：
  `/home/medteam/Zhrch/DiffusionSnake-12-30-par3d-h1-20260803`，代码提交 `ce505ea`
- Flow 职责与接口记录：
  `/home/medteam/Zhrch/DiffusionSnake-12-30/docs/report/FLOW_MAIN_HANDOVER_STATUS_20260804.md`
