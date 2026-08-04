# MemFlowDiT 三维 Memory 与非自回归并行推理实验记录

**日期**：2026-07-31  
**负责人**：Codex  
**项目路径**：`/home/medteam/Zhrch/DiffusionSnake-12-30`  
**状态**：实现与首轮诊断进行中

## 1. 研究目标

本轮研究不停止 GPU 6/7 上正在运行的 v0.5、v0.6 主线训练，使用空闲 GPU 0、1、5 回答三个问题：

1. 当前 Memory 路径是否真正包含可利用的三维信息，而不只是一个接近恒等映射的弱残差；
2. bank 容量、单状态空间 token 数和状态选择方式分别如何影响精度、速度与显存；
3. 离线 CT 体积推理能否解除“当前预测写入 bank 后下一张才能继续”的自回归依赖，改成切片批量并行。

所有正式精度实验使用固定 seed `20260731` 和 GT box，以排除检测器误差。检测器结果只在最佳方案确定后作为独立端到端实验加入。

## 2. 审计结论

### 2.1 当前实现

- `SliceMemoryBank` 是容量为 4 的逐体积 FIFO，只保留最近 4 张切片，并非全部历史；
- 每张切片经 `8×8` 自适应池化后形成 64 个 memory token，因此满 bank 为 256 token；
- 每个 DiT block 后都执行一次 contour-to-memory cross-attention；
- 当前 state 选择没有置信度、特征相关性或长期压缩策略；
- 相对位置使用 slice index 的有符号差值，manifest 尚未记录物理层距；
- 自回归评估先预测当前切片，再把预测 mask 与 MoonViT 特征写入 bank，错误会沿切片方向传播。

### 2.2 训练与 Memory 使用强度

审计时状态：

| 训练 | GPU | step | Memory read delta | 说明 |
|---|---:|---:|---:|---|
| v0.5 minimal | 6 | 2325 | 约 `1.9e-4` | prediction evidence 已约 0.20 |
| v0.6 MoE combined | 7 | 550 | 约 `4.7e-4` | prediction evidence 刚开始爬坡 |

此前 v0.5 step 1600、3 volumes、GT-box、固定 seed 的结果为：

| 模式 | Volume Dice | Volume IoU | Foreground slice Dice |
|---|---:|---:|---:|
| Memory off | 0.792168 | 0.656264 | 0.765609 |
| Autoregressive | 0.792125 | 0.656225 | 0.765794 |

两者几乎相同，说明在扩大 bank 前必须先测 GT oracle 和错误 Memory 负对照。如果 oracle 也没有收益，问题是 Memory 注入/训练不足，而不是 bank 太小。

### 2.3 新发现：mask evidence 在拼接投影中被 MoonViT 特征数值淹没

当前 encoder 将 1152 个 MoonViT 通道与 26 个 class-aware mask 通道直接拼接，再经过同一个 `1×1 Conv`。对 sub-verse010 全部 103 张前景切片的 checkpoint 激活审计表明：

- mask 分支相对 feature 分支的 key 输出 RMS 比值，平均仅 `0.00441`；
- 中位数仅 `0.00312`；
- 90 分位数仍只有 `0.00992`；
- 最大值也只有 `0.01143`；
- 仅 1/103 张前景切片在缩到 `8×8` 后完全消失，因此主要问题不是个别小目标被池化掉，而是通道数和稀疏度共同造成的整体幅值失衡。

换言之，即使输入 GT mask，进入 key projection 的有效贡献通常也不到 MoonViT feature 的 1%。这解释了为何同一体积上 GT-mask 双向 Memory 与零-mask feature-only Memory 的 Dice 只相差 `0.000035`。

同时，六层 DiT Memory adapter 的 `output_proj.weight` RMS 约为：

`1.78e-4, 2.06e-4, 2.26e-4, 2.29e-4, 1.62e-4, 4.51e-5`

因此存在两个串联瓶颈：mask evidence 在 encoder 入口被淹没，Memory residual 又在 DiT 出口被极小的 projection 二次压低。仅扩大 bank 会重复更多弱 token，不能解决这两个问题。

## 3. 本轮实现

新增：`tools/volmem/eval_memflowdit_parallel.py`

支持以下模式：

| 模式 | Memory 来源 | 选择范围 | 轮廓网络次数 | 切片依赖 |
|---|---|---|---:|---|
| `off` | 无 | 无 | 1 | 独立，但旧评估逐张执行 |
| `autoregressive` | 当前模型预测 | 最近历史 | 1 | 严格串行 |
| `oracle` | GT mask | 最近历史 | 1 | 严格串行；仅上界诊断 |
| `frozen-causal` | 第一遍预测 | 过去最近 K 张 | 2 | 第二遍独立并行 |
| `frozen-bidirectional` | 第一遍预测 | 前后最近 K 张 | 2 | 第二遍独立并行 |
| `frozen-oracle-bidirectional` | GT mask | 前后最近 K 张 | 1 | 独立并行；仅上界诊断 |
| `frozen-feature-bidirectional` | MoonViT 特征、零 mask | 前后最近 K 张 | 1 | 完全非自回归并行 |
| `frozen-shuffled` | 第一遍预测 | 同体积随机 K 张 | 2 | 负对照 |

实现约束：

- target slice 永远不能读取自身 state；
- 不允许跨 volume 混合 Memory；
- 双向模式采用绝对层距编码，避免未来切片使用训练中未出现的负距离；
- frozen state 在第二遍保持不变，所有 target slice 可按 batch 同时预测；
- summary 记录容量、pool size、满 bank token 数、选择规则、证据来源、总耗时、吞吐和峰值显存。

同时新增了无参数状态选择函数：

- `causal-nearest`：只取距离最近的过去状态；
- `bidirectional-nearest`：取 target 前后距离最近的状态；
- `shuffled`：同体积确定性随机选择，用作错误 Memory 对照。

新增单元测试覆盖因果选择、双向选择、自身排除以及 absolute distance 对未来 Memory 的支持。本轮新增 9 项测试已全部通过。

## 4. Bank / token 预算实验

不直接把所有变量做全组合，先使用下列配置分离“历史宽度”和“单张空间细节”：

| 名称 | K | Pool | 总 token | 用途 |
|---|---:|---:|---:|---|
| K1-P8 | 1 | 8×8 | 64 | 短历史、高空间细节 |
| K4-P4 | 4 | 4×4 | 64 | 同成本下扩大历史范围 |
| K4-P8 | 4 | 8×8 | 256 | 当前基线 |
| K8-P4 | 8 | 4×4 | 128 | 较宽历史、适中成本 |
| K8-P8 | 8 | 8×8 | 512 | 仅当前四项证明有收益后测试 |

首轮不引入可学习 Memory selector。只有 K8 明显优于 K4，才考虑使用当前切片 MoonViT 全局特征与物理距离组成的简单无参数分数。

## 5. 48 小时调度

GPU 6/7 保持原训练不动。空闲卡调度如下：

| 时间 | GPU 0 | GPU 1 | GPU 5 |
|---|---|---|---|
| 0–6 h | off / oracle / shuffled 诊断 | K/P token 预算扫描 | parallel batch smoke 与速度测试 |
| 6–24 h | feature-only 一遍方案 | frozen-causal 两遍方案 | frozen-bidirectional 两遍方案 |
| 24–48 h | 淘汰最差方案后补充消融 | 前两名延长训练/评估 | 前两名全量可视化与速度复核 |

实验先从同一个 v0.5 checkpoint 分叉，避免与 v0.6 的 MoE 改动混杂。最佳 Memory 方案确认后再整合入 v0.6。

## 6. 决策规则

- 若 `causal GT oracle - off < 0.2` Dice 百分点：暂停扩大 bank，先解决 Memory 注入过弱；
- 若 oracle 明显优于 off，但 autoregressive 接近 off：主要瓶颈是预测证据噪声或自回归误差累积；
- 若正确邻居与 shuffled Memory 接近：模型没有利用真实三维关系，不能据此扩大 bank；
- 新方案进入主线需要满足以下至少一项：
  - 全量 Volume Dice 提升约 0.5 个百分点，且最差病例不明显退化；
  - 精度基本持平，但整卷推理显著加速；
- 所有晋级结论必须同时报告 Memory-off 消融、吞吐、峰值显存、前景体积比例和全量可视化。

## 7. 输出目录约束

- 指标与日志：`data/outputs/volmem/diagnostics/parallel_memory_*`
- 全量可视化：`visual/memflowdit/<experiment_name>/`

可视化文件不得写入项目根目录或 `data/outputs` 的零散位置。

## 8. 实验运行记录

| 时间 | 实验 | 状态 | 结果目录 | 备注 |
|---|---|---|---|---|
| 2026-07-31 | v0.5 step2300, feature-only bidirectional, K4-P8, 1 volume | 完成 | `data/outputs/volmem/diagnostics/parallel_memory_smoke_feature_k4p8_v05_step2300` | Dice 0.794881；0.921 slice/s；峰值 2.44GB |
| 2026-07-31 | Stage-0 off / causal oracle / bidirectional oracle, 3 volumes | 完成 | `data/outputs/volmem/diagnostics/parallel_memory_stage0_v05_step2300` | 旧 Memory 无正收益；parallel-off 提速 4.39× |
| 2026-07-31 | Stage-1 autoregressive / frozen feature / frozen causal / frozen bidirectional | 完成 | `data/outputs/volmem/diagnostics/parallel_memory_stage1_v05_step2300` | 旧自回归相对顺序 off：Dice -0.000276，吞吐 -17.7% |
| 2026-07-31 | 严格同噪声 off / feature / GT / predicted 配对 | 完成 | `data/outputs/volmem/diagnostics/parallel_memory_common_noise_v05_step2300` | 三种 Memory 均无收益且几乎不可区分 |
| 2026-07-31 | v0.7 balanced Memory 1-step smoke | 完成 | `/dev/shm/memflowdit_v07_balanced_smoke` | checkpoint 迁移 461/462；step1 正常反传 |
| 2026-07-31 | v0.7 balanced Memory 首次启动 | 已归档 | `data/outputs/volmem/verse_memflowdit_v0_7_balanced_memory_gpu1_superseded_long_ramp_step9` | step9 主动停止；发现 4500-step 预测证据爬升与 2000-step 上限不匹配 |
| 2026-07-31 | v0.7 balanced Memory 修正日程后正式训练 | 运行中 | `data/outputs/volmem/verse_memflowdit_v0_7_balanced_memory_gpu1` | GPU1；最多 2000 steps；最晚 2026-08-02 20:50 停止 |
| 2026-07-31 | v0.7 step100 causal gate 守护 | 失败、未产生评估 | `data/outputs/volmem/diagnostics/memflowdit_v07_step000100_gt_causal_gate` | 守护运行中脚本曾被覆盖，bash 读取偏移导致 `break` 被截断；训练未受影响 |
| 2026-08-01 | v0.7 step500 causal GT gate | 完成、未通过 | `data/outputs/volmem/diagnostics/memflowdit_v07_step000500_gt_causal_gate` | GT-oracle 相对 off：Volume Dice -0.000750；三个 volume 均未转正 |

后续每次启动、停止、失败原因、checkpoint、seed、评估范围和核心指标均追加到本节，不以聊天记录代替实验记录。

### 8.1 首个并行 smoke 结果

`frozen-feature-bidirectional` 在 sub-verse010 上完成 186 张切片的单遍批量推理：

- Volume Dice：`0.7948806654`
- Volume IoU：`0.6595866837`
- Foreground slice Dice：`0.7823523426`
- 平均 Memory read delta：`1.2453e-4`
- 总评估时间：`201.98 s`，其中 state build `86.43 s`、轮廓推理 `115.54 s`
- 吞吐：`0.9209 slice/s`
- 峰值显存：`2.435 GB`

这证明了 batch=8 的多切片推理、每个 target 独立 bank、future state 和绝对层距编码在工程上可以正常工作。该结果尚不能证明精度收益，因为需要同一 step2300、同一体积的 Memory-off 配对结果。Memory read delta 仍在 `1e-4` 量级，因此保留“现有 Memory 注入过弱”的主要怀疑。

### 8.2 Stage-0 上界诊断结论

同一 v0.5 step2300、相同 3 volumes、GT-box、seed 20260731：

| Run | Volume Dice | 配对差值 | Read delta | slice/s | Peak GB |
|---|---:|---:|---:|---:|---:|
| sequential off | 0.795673 | 基线 | 0 | 0.398 | 0.517 |
| causal GT oracle | 0.795430 | -0.000243 vs sequential off | 0.000106 | 0.326 | 0.521 |
| parallel off | 0.796269 | 批量基线 | 0 | 1.747 | 2.546 |
| frozen bidirectional GT oracle | 0.795842 | -0.000426 vs parallel off | 0.000109 | 1.160 | 2.579 |
| frozen bidirectional GT oracle, read scale 4 | 0.790712 | -0.005556 vs parallel off | 0.000436 | 1.390 | 2.579 |

结论：

1. 当前旧 Memory 即使读取 GT 证据也没有正收益，因此不能把问题归因于 autoregressive prediction error，也不能通过扩大 bank 解决；
2. 强制放大 residual 会稳定恶化，说明旧 Memory 学到的方向不只是幅值过小，而是包含错误或无效更新；
3. 多切片 batch 本身将吞吐从 0.398 提升到 1.747 slice/s，约 4.39 倍，且显存仅约 2.55GB；
4. 因此保留“非自回归并行推理”作为主线收益，但旧 Memory 结构停止晋级，不运行 K8-P8 等无意义的大 bank 实验；
5. 下一步改为修复数值上被淹没的 mask entrance，再重新训练 Memory，之后才重新讨论 bank 容量。

### 8.3 v0.7 balanced Memory 修复与启动

为保持结构简洁，v0.7 没有增加额外 Memory router、长期 bank 或法向细节分支，只修改现有 Memory encoder 的融合方式：

旧设计：

`Conv1x1(concat[1152-channel MoonViT, 26-channel sparse mask])`

新设计：

1. MoonViT feature 与 class-aware mask 分别执行 `1×1 projection`；
2. mask 分支单独执行无 affine 的 GroupNorm，避免稀疏 26 通道被 1152 个 feature 通道在数值上淹没；
3. mask 使用 adaptive max pooling 保留小目标是否出现，不增加新的采样网络；
4. 两个分支以固定 `0.25` 比例相加，不引入额外门控器；
5. 空 mask 显式保持为零，不让归一化产生伪证据。

旧 checkpoint 迁移规则也是确定性的：原 `key/value projection` 的前 1152 通道加载到 feature projection，后 26 通道加载到 mask projection。1-step smoke 的兼容结果为 `461/462`，唯一缺失项与旧 v0.5 加载时相同；训练 step1 loss `0.000743`、read delta `0.000252`、峰值显存 `1.89GB`，反向传播和 checkpoint 保存均正常。

正式训练命令由 `tools/volmem/launch_memflowdit_v07_balanced_2day_gpu1.sh` 固化。训练从 v0.5 step2300 完整权重分叉，GPU1 运行，最多 2000 steps，并由 watchdog 保证不超过两天。GPU6/7 上原 v0.5/v0.6 训练不受影响。

首次正式启动后，在 step9 复核配置时发现，继承自长训练的 `prediction_evidence_start_step=500`、`ramp_steps=4500` 与本轮 2000-step 上限不匹配：结束时预测证据概率只能到约 16.7%，不能充分覆盖实际推理所使用的 predicted Memory。该 run 未删除，完整归档到 `data/outputs/volmem/verse_memflowdit_v0_7_balanced_memory_gpu1_superseded_long_ramp_step9`。正式 run 将日程修正为：step250 开始引入预测证据、750 steps 完成爬升、step1000 后保持 50% 混合。这样前 250 步用于适配新的 mask entrance，后半程用于稳定训练—推理一致性，不增加任何网络模块。

修正后 run 已通过前 10 步稳定性检查：平均 loss `0.004368`，平均 read delta `1.936e-4`，全部数值有限；step10 的 MoE normalized entropy `0.9874`、hard CV `0.7753`、dead experts `0`。预测证据日程的 start/ramp/max 三个值也已写入 `run_manifest.json`，不依赖 YAML 或聊天记录才能复现。

### 8.4 严格同噪声配对结果

Stage-0 的微小差值最初受到 Flow 随机噪声消费顺序影响。评估器现已在每个 volume 的最终 refinement 前按 `seed + volume_number × 1,000,003` 重置全部随机源，使 parallel-off、feature-only、GT-oracle 和 predicted-memory 使用完全相同的最终噪声。GT-box 同时排除了检测器影响。

v0.5 step2300、3 个固定 validation volumes、K4-P8、batch=8 的结果如下：

| Run | Volume Dice | 差值 vs parallel-off | Foreground slice Dice | Read delta | slice/s | Peak GB |
|---|---:|---:|---:|---:|---:|---:|
| parallel-off | 0.796574 | 0 | 0.770027 | 0 | 1.789 | 2.546 |
| frozen feature-only bidirectional | 0.796354 | -0.000219 | 0.769847 | 0.000109 | 1.362 | 2.580 |
| frozen GT-oracle bidirectional | 0.796343 | -0.000231 | 0.769835 | 0.000109 | 1.401 | 2.579 |
| frozen predicted bidirectional | 0.796363 | -0.000210 | 0.769906 | 0.000109 | 0.859 | 2.625 |

另一个严格顺序单遍对照中，旧 autoregressive 从 off 的 `0.795673` 降至 `0.795397`，差值 `-0.000276`；吞吐从 `0.398` 降至 `0.327 slice/s`，下降 17.7%。

这一组结果给出了比 Stage-0 更可靠的结论：

1. GT、预测和零 mask 的三条 Memory 路径，Dice 最大差异只有约 `2.0e-5`，平均 read delta 在记录精度内相同；旧网络实际上没有分辨三种证据；
2. 即使 GT-oracle 也不优于 off，当前主要矛盾不是自回归误差累积，也不是 bank 太小，而是 Memory encoder 与 DiT read path 没有学会利用跨层信息；
3. predicted two-pass 吞吐仅 `0.859 slice/s`，为 parallel-off 的 48.0%，在没有精度收益时不能进入主线；
4. feature-only 是一遍非自回归方案，但旧权重下同样无收益；它保留为未来独立训练方向，而不是直接把未训练的 future Memory 塞入当前模型；
5. 在 v0.7 重新证明 oracle Memory 有正收益前，停止 K8/K16、P4/P16 和复杂 Memory 选择器扫描，避免用更多 GPU 重复无效 token。

### 8.5 下一阶段门槛

- v0.7 每 100 steps 保存 checkpoint；step100/200/500 先做 3-volume GT-box 的 off 与 causal-oracle 配对；
- step100 已由 `tools/volmem/watch_eval_memflowdit_v07_step100.sh` 自动守候；checkpoint 写入稳定且 GPU4 显存低于 1GB 后才启动；
- 只有当 causal-oracle 至少稳定转正，才补 predicted、自回归和 bank K2/K4/K8；
- 只有当 feature-only 经过专门训练后仍接近 oracle，才推进真正的一遍双向并行 Memory；
- 任一候选进入主线前，必须再跑完整 validation、最差病例检查和全量可视化；可视化仅写入 `visual/memflowdit/<experiment_name>/`。

### 8.6 v0.7 step500 因果门槛结果（2026-08-01）

step100 自动守护没有实际完成评估。原因不是训练或模型报错，而是守护仍在运行时曾更新同一个 shell 文件，bash 从已改变的文件偏移继续读取时把 `break` 截断成 `eak`，随后退出。该故障已记录；以后后台任务启动后不再原位覆盖其脚本。由于训练已经超过 step500，本轮不再补价值较低的 step100，而是直接评估最新稳定 checkpoint `step_000500.pt`。

评估条件：同一 step500 checkpoint、GT-box、seed 20260731、固定 3 个 validation volumes、顺序单遍推理。off 与 causal GT-oracle 分别在 GPU0/GPU4 独立运行；checkpoint `462/462` 全量兼容，无缺失参数或 shape mismatch。

| Run | Volume Dice | Foreground slice Dice | Class mean Dice | Read delta | slice/s | Peak GB |
|---|---:|---:|---:|---:|---:|---:|
| off | 0.790857 | 0.764841 | 0.747477 | 0 | 0.334 | 0.517 |
| causal GT-oracle | 0.790106 | 0.764453 | 0.747050 | 0.000136 | 0.308 | 0.521 |
| oracle - off | **-0.000750** | -0.000388 | -0.000427 | +0.000136 | -7.9% | +0.004 |

逐 volume 的 Dice 差值也全部非正：

- sub-verse010：`-0.000022`
- sub-verse011：`-0.000699`
- sub-verse013：`-0.001529`

结论：

1. balanced entrance 确实增强了 Memory 对 DiT 的影响；活跃前景区间的日志 read delta 约为 `2.7e-4` 至 `3.1e-4`，明显高于旧模型约 `1.1e-4`；
2. 但增强后的更新方向仍然有害。严格配对下 oracle 在 Volume、slice、class 三个聚合层面全部退化，所以 v0.7 step500 不能称为“有效 Memory”；
3. step500 的 off 本身比初始化来源 v0.5 step2300 的顺序 off `0.795673` 低 `0.004816`，说明早期继续训练也改变并恶化了基础 2D 路径；这不是 Memory read 的配对差值，但属于独立风险信号；
4. 不启动 K2/K4/K8、pool 或 Memory selector 扫描，也不生成全量可视化；当前候选没有达到这些工作的精度门槛；
5. v0.7 继续到 step1000，因为预测证据混合届时才达到设定的 50%。step1000 再做同一 causal gate；若 oracle 仍不转正，停止 v0.7 并把该 entrance 从主线候选中淘汰。

检查时正式训练已到 step618，最近 20 步平均 loss `0.004964`，全部有限；当前 predicted-evidence 目标概率约 `0.245`。原两天上限 watchdog 意外退出但没有触发停止，已于 `2026-08-01T05:55:38+08:00` 重新挂载到同一训练 PID，step2000 和 `2026-08-02T20:50:00+08:00` 两个上限保持不变。

## 9. 研究依据

- SAM 2：使用流式 Memory 处理视频分割，说明受限 working memory 是可行基础；
- XMem：将 sensory、working 和 compact long-term memory 分开，提示历史范围和 token 预算应独立研究；
- Cutie：指出底层像素匹配会受到噪声干扰，对象级摘要可作为后续方案；
- SAM2Long：指出贪心自回归 Memory 会产生误差累积，支持本轮 frozen two-pass 设计。

本轮暂不复制复杂的多层 Memory 或搜索树。当前首要任务是证明现有 Memory 路径确实能够利用正确的三维上下文。

## 10. Bank 覆盖范围与物理坐标复核（2026-08-01）

### 10.1 对“bank=4 太短”的判断

这个怀疑成立了一半，而且进一步暴露了一个比容量更基础的数据契约问题。当前 v0.7 为 `memory_capacity=4`、`memory_pool_size=8`，每层产生 64 个 token；训练和推理均把 `slice_index` 直接作为位置，未读取原始 NIfTI 的毫米间距。对本轮 3 个 validation volumes 直接审计原始头信息后得到：

| Volume | 矢状位层间距 | 总层数 | 最近 K4 的物理跨度 | 全历史最大跨度 |
|---|---:|---:|---:|---:|
| sub-verse010 | 1.000 mm | 186 | 3.000 mm | 185.000 mm |
| sub-verse011 | 1.99993 mm | 38 | 6.000 mm | 73.998 mm |
| sub-verse013 | 3.000 mm | 109 | 9.000 mm | 324.000 mm |

因此 K4 确实只覆盖很窄的局部范围，容易仍表现为“带少量邻层的 2D”。但当前网络还把相同的 index 距离在 1/2/3 mm 数据中编码成同一种几何关系；也就是说，即使盲目增大 bank，网络看到的三维距离仍不一致。这是必须优先修正的真实问题。

训练端还有第二个约束：`chunk_length=8` 且只用最近 K4，模型从未在训练时见过几十到上百层的原始历史。若只在推理时切换为无上限，会产生明显的训练—推理 Memory 长度错配。以 sub-verse010 最后一层为例，K4 为 256 个 Memory token，全历史为 11,840 个，增加约 46.25 倍，并在 6 个 DiT block 中重复参与注意力；它同时改变了历史跨度、token 数、冗余度和计算量，不能作为单变量修复直接并入主线。

### 10.2 论文结论与本项目决策

- SAM 2 默认保存 7 个空间 Memory（当前帧输入加 6 个过去帧），并支持“最近帧 + 间隔采样旧帧”的评估策略；它没有把所有高分辨率历史平铺进注意力。
- RMem 的直接结论是：不断扩张的历史 bank 会引入重复和混淆特征，固定容量、兼顾相关性与新鲜度的 restricted bank 可以同时改善精度并减小训练—推理长度差异。
- XMem 使用 sensory / working / compressed long-term 三层记忆，通过压缩进入长期记忆来避免容量爆炸。
- Cutie 指出逐像素历史匹配容易受噪声与冗余干扰，因而额外使用对象级查询；这支持未来研究紧凑类别/对象摘要，但不支持现在直接叠加复杂 router。

据此，当前不把“无上限原始 bank”直接设为新默认。最简主线候选是：固定约 6–8 个状态的预算，始终保留最近层，再按真实毫米距离从较久历史取锚点；只有在固定预算仍不足时，才增加一个压缩的长期类别摘要，而不是保留全部原始 token。

### 10.3 已实现的鉴别实验

为避免把容量与覆盖跨度混为一谈，评估器新增 `causal-strided` 选择，保持 K4/256 token 不变，选择最近一层以及距离约为 5、9、13 个 index 的旧层；对应单测已通过。该改动只作用于新启动的诊断评估，不会改变正在运行的 v0.7 训练进程。

同一 v0.7 step500、GT-box、seed 20260731、同一 volume 的鉴别顺序为：

1. `parallel-off`：同噪声基线；
2. recent K4：现有短范围基线；
3. strided K4：token 数相同，只扩大历史覆盖；
4. recent K8 / K16：判断容量是否不足；
5. all-history：先限 1 个 volume 做上界与资源压力测试，不把它默认视为可部署方案。

判据也预先固定：strided K4 转正说明主要是覆盖问题；recent K8/K16 转正说明主要是容量问题；all-history 反而下降说明冗余和长度错配占主导；所有 oracle 方案均不转正，则根因仍在 Memory 表征/读取方向，而不是 bank 数量。

### 10.4 同 checkpoint 鉴别结果

固定 v0.7 step500、sub-verse010、GT-box、seed 20260731、batch=1，并在每个 volume 的最终 refinement 前重置同一噪声。完整结果位于：

`data/outputs/volmem/diagnostics/memflowdit_v07_step000500_history_span_ablation/`

| Run | Volume Dice | Δ vs off | FG slice Dice | Class Dice | Read Δ | slice/s | Peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 0.791592 | 0 | 0.781193 | 0.762536 | 0 | 0.362 | 0.493 |
| feature-only recent K4 | 0.790041 | -0.001551 | 0.778466 | 0.760705 | 0.000158 | 0.281 | 0.522 |
| GT recent K4 | 0.791451 | -0.000140 | 0.779875 | 0.761433 | 0.000152 | 0.308 | 0.522 |
| GT recent K8 | 0.789947 | -0.001644 | 0.777435 | 0.758927 | 0.000417 | 0.313 | 0.524 |
| GT recent K16 | 0.788471 | -0.003120 | 0.771305 | 0.754505 | 0.000696 | 0.295 | 0.527 |
| GT strided K4, stride=4 | 0.789143 | -0.002449 | 0.774379 | 0.756548 | 0.000573 | 0.301 | 0.522 |
| GT all-history | 0.790383 | -0.001209 | 0.771831 | 0.756328 | 0.000685 | 0.258 | 1.331 |

all-history 在 sub-verse010 最后一层实际最多使用 185 个状态、11,840 个 token。它虽然未 OOM，但相对 off 慢 28.6%，前景切片 Dice 下降 `0.009363`。逐切片配对还显示其伤害随历史长度增大：前景 Dice 差值与 slice index 的相关系数为 `-0.459`；index 100–149 的平均差值为 `-0.016106`，明显差于 index 50–99 的 `-0.003414`。这不是“全历史还不够多”，而是历史越长，错误/冗余内容越明显。

容量趋势同样明确：recent K4/K8/K16 的 Volume Dice 差值依次为 `-0.000140/-0.001644/-0.003120`，read delta 依次为 `0.000152/0.000417/0.000696`。Memory 作用越强，结果越差，因此不能再把失败解释为单纯“read 太弱”或“bank 太小”。

最有信息量的是 evidence 配对：feature-only K4 相对 off 为 `-0.001551`，加入 GT mask 后恢复 `+0.001411`，最终只剩 `-0.000140`。这证明 v0.7 的 balanced mask evidence 已经真正被利用，而且方向是有益的；当前主要有害项是写入/读出的历史 MoonViT feature 与距离内容，而不是 GT mask 没进入 Memory。

### 10.5 位置写入 value 的诊断边界

代码审计发现旧实现把相对层距编码同时加到 key 和 value。SAM 2 将 memory feature 与 positional embedding 分开交给注意力，而不是把位置直接混成 memory value 内容。为验证旧权重是否可以直接改为 key-only，加入了不新增参数的推理开关。结果 recent/strided K4 都约降至 `0.7277`，read delta 放大到约 `1.90e-3`。这不是 key-only 设计本身的反证，而是说明旧 checkpoint 已与“位置进入 value”的分布强耦合，不能在推理时硬切；若采用标准的 key-only 位置语义，必须重新训练。

### 10.6 v0.8 最小修复与正式训练

v0.8 不引入双 bank、学习 selector、额外 router 或法向细节分支，只做四项有直接证据的修改：

1. MoonViT 只形成 key，负责检索相关历史；
2. class-aware GT/预测 mask 单独形成 value，只有任务证据被写回 DiT；空 mask 的 value 严格为零；
3. 相对层距只进入 key，不再污染 value；
4. 使用真实毫米位置和有界 K7 bank。K7 取自 SAM 2 默认 7 个空间 Memory 的量级，不使用无限原始历史。

物理清单由 `tools/volmem/build_physical_slice_manifest.py` 从 160 个原始 NIfTI 头生成，共 26,589 行：

`data/outputs/volmem/manifests/slice_manifest_physical_mm.csv`

SHA256 为 `05001f9a813bfea782e294a81439525ce189d0629f39b7a16e7e8ae93ea0e2e4`。旧 legacy dataset 曾主动丢弃附加 CSV 列，首次 smoke 因而被数据契约检查提前拦截；现已只在 VolMem adapter 边界补回 `slice_position_mm/slice_spacing_mm`，未修改通用 2D dataset 行为。

直接继承 v0.7 输出投影的第二次 smoke 虽可训练，但初始 read delta 达 `0.002375`，有立即破坏 2D 路径的风险，因此未用于正式 run。最终方案从 2D 表现更好的 v0.5 step2300 初始化，确定性迁移 feature-key、mask-key 和 mask-value 权重，并将 6 个 Memory read 输出投影清零，使启动时严格等价于 2D。14 项单测通过；最终 1-step smoke 为：兼容 `460/461`、loss `0.001086`、read delta `0`、峰值 `5.04GB`，前反向和 checkpoint 均正常。

正式训练已在不停止 v0.7 的前提下启动：

- config：`configs/volmem/verse_memflowdit_v0_8_evidence_value_mm_gpu5.yaml`
- output：`data/outputs/volmem/verse_memflowdit_v0_8_evidence_value_mm_gpu5/`
- GPU：5
- train PID：`2745777`
- watchdog PID：`2745778`
- 上限：step2000 或 `2026-08-03T07:00:00+08:00`，先到即停
- 启动后 step1–8 全部有限；read delta 从严格 0 平滑打开到约 `5e-5`–`6e-5`，未出现 v0.7 迁移时的突跳

v0.7 同时仍在 GPU1 运行，检查时已到 step800，原 watchdog 与两天上限保持不变。v0.8 的早期门槛仍使用 GT-box、同噪声 off/oracle 配对；只有 oracle 明确转正后，才讨论 strided K7 或紧凑 global volume token，不能先增加长期模块。

不可变的 step100 评估守护脚本 `tools/volmem/watch_eval_memflowdit_v08_step0100_v1.sh` 已启动（PID `2748662`）。它只读稳定落盘的 `step_000100.pt`，在 GPU0/GPU4 空闲后并行运行 1-volume `parallel-off` 与 `frozen-oracle-causal K7`；结果写入 `data/outputs/volmem/diagnostics/memflowdit_v08_step000100_gt_gate/`。该运行期间不再原位修改脚本，避免重现 v0.7 step100 的 shell 偏移故障。

### 10.7 新增论文依据

- 2026-07 的 SAMRI-3D 使用持久化 Global Volume Tokens，并通过 TSDF 辅助目标训练体积上下文；重点是紧凑全局 token，而非无限原始切片。
- 2025 的 Short-Long Memory SAM 2 指出单一 memory/attention 在体数据边界容易过传播，并将短期与长期记忆分开。
- SAMed-2 使用 confidence-driven memory，只保存高确定性特征。
- 2026 Spatial-SAM 用 SDF memory 强化三维空间一致性，说明长期信息应当是结构化几何摘要。

这些工作共同支持后续“有界局部状态 + 少量结构化全局摘要”的方向。但为保持主线简洁，本轮 v0.8 先验证 evidence-only value 是否能让最基本的 oracle Memory 转正；未通过前不实现 Global Volume Token、SDF 分支或双注意力 bank。

### 10.8 v0.8 最终双门控与 v0.9 compact global tokens（2026-08-02）

v0.7 和 v0.8 均已按既定上限完成 step2000，训练进程自然结束，没有被新实验中断。为避免只用早期 step100 判断 v0.8，固定 sub-verse010、GT-box、seed 20260731、同 checkpoint 同最终噪声，补做最新两个 checkpoint 的 `parallel-off` 与 `frozen-oracle-causal K7`：

| Step | 模式 | Volume Dice | FG slice Dice | Class Dice | Read Δ | slice/s |
|---:|---|---:|---:|---:|---:|---:|
| 1900 | off | 0.793209 | 0.781976 | 0.763925 | 0 | 0.376 |
| 1900 | oracle K7 | 0.793611 | 0.782827 | 0.764229 | 0.000147 | 0.309 |
| 1900 | oracle - off | **+0.000402** | +0.000851 | +0.000304 | — | -17.9% |
| 2000 | off | 0.795369 | 0.782222 | 0.763760 | 0 | 0.378 |
| 2000 | oracle K7 | 0.795481 | 0.779988 | 0.762734 | 0.000143 | 0.305 |
| 2000 | oracle - off | **+0.000111** | **-0.002234** | **-0.001027** | — | **-19.4%** |

结果目录为：

`data/outputs/volmem/diagnostics/memflowdit_v08_latest_dual_gt_gate/`

预先约定的有效门槛是：Volume Dice 至少 `+0.001`、前景 Dice 同向、吞吐损失不超过 10%。两个 checkpoint 均未通过；step2000 虽有极小体积正差值，前景与类别指标反向，且速度损失接近 20%。因此 v0.8 不能声明 Memory 已有效，不再恢复或续训该 run。

随后对 v0.8 step2000 做零训练 compact 探针，用于选容量而非证明最终性能：

| 模式 | Token 上限 | Volume Dice | Δ vs off | FG Δ vs off | Class Δ vs off | slice/s |
|---|---:|---:|---:|---:|---:|---:|
| off | 0 | 0.795369 | 0 | 0 | 0 | 0.378 |
| local K4 + global 4×4 | 272 | 0.795550 | +0.000180 | -0.002261 | -0.001329 | 0.304 |
| local K7 + global 4×4 | 464 | 0.795530 | +0.000160 | -0.002129 | -0.000994 | 0.295 |

探针位于：

`data/outputs/volmem/diagnostics/memflowdit_v09_compact_probe_step002000/`

K4+G16 略优且更省，因此 v0.9 采用更简单的 K4，不保留 K7。实现只有一个直接路径：

1. 最近 4 层仍保留 8×8 evidence state，共 256 tokens；
2. 被挤出的全部旧状态在线平均为一个 4×4 全局摘要，只增加 16 tokens，总预算恒定为 272；
3. 只有非空 mask evidence 参与均值；空历史通过 token valid mask 屏蔽，更新过程不做逐状态 CPU/GPU 同步；
4. 全局摘要覆盖一段物理范围，因此不伪造单一 slice distance；毫米距离仍只加给局部 key；
5. 不增加参数、学习 selector、第二个 attention bank、router、法向细节或 SDF 分支；
6. 训练 chunk 从 8 调为 12、chunks/step 从 12 调为 8，在保持每步 96 个 micro-slices 不变的同时，让模型在训练中实际读到 global token。

代码编译和 17 项 Memory 单测通过。2-step smoke 从 v0.8 step2000 加载 `461/461` 参数；step2 loss `0.001032`、read delta `0.000206`、active states `5`，峰值显存 `4.25GB`，前向、反向和 checkpoint 均正常。

正式 v0.9 已启动：

- config：`configs/volmem/verse_memflowdit_v0_9_compact_global_gpu6.yaml`
- output：`data/outputs/volmem/verse_memflowdit_v0_9_compact_global_gpu6/`
- GPU：6
- train PID：`3403812`
- watchdog PID：`3403813`
- 上限：step2000 或 `2026-08-03T23:50:00+08:00`，先到即停
- 首个正式 step：loss `0.003758`、read delta `0.000191`、active states `40`（8 个并行 bank × 5 states）、峰值显存 `12.98GB`

step100 三路门控守护 PID 为 `3404945`，固定比较 off、oracle local K4、oracle local K4+G16；结果写入：

`data/outputs/volmem/diagnostics/memflowdit_v09_step000100_compact_gate/`

该三路配对会同时回答“局部 Memory 是否有用”和“16 个全局 token 是否提供额外净收益”，不会把 global 与 local 的作用混在一个总差值中。

### 10.9 v0.9 step100 三路门控结果与最终止损点（2026-08-02）

step100 的三路评估已完整结束。三路固定使用同一个暂存 checkpoint、GT-box、sub-verse010、seed 20260731 和同一 MoonViT cache；checkpoint 参数兼容数均为 `461/461`，oracle 两路均构建了 `100/186` 个非空历史状态。结果为：

| 模式 | Volume Dice | Δ vs off | FG slice Dice | Class Dice | Read Δ | slice/s | 速度变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 0.794863 | 0 | 0.783747 | 0.765392 | 0 | 0.366 | 0 |
| oracle local K4 | 0.793886 | -0.000977 | 0.782053 | 0.764136 | 0.000127 | 0.304 | -17.0% |
| oracle local K4 + global16 | 0.793874 | -0.000989 | 0.781835 | 0.764034 | 0.000130 | 0.299 | -18.2% |

结果目录：

`data/outputs/volmem/diagnostics/memflowdit_v09_step000100_compact_gate/`

compact 相对 local K4 的 Volume Dice 额外差值仅为 `-0.0000119`，近似零且方向为负；相对 off 的前景差值为 `-0.001912`。因此 step100 明确未通过 `Volume +0.001、前景同向、减速不超过 10%` 的预设门槛。当前不能声称 Memory 已经真实有效，也不能把 compact global token 描述为已带来三维收益。

该结果还把问题进一步拆开：首先，local K4 本身仍轻微伤害预测；其次，在 local 路径上增加 16 个全局摘要 token 没有产生可测的额外收益。后者不是 token 未进入网络：compact 的 read delta 从 `0.000127` 变为 `0.000130`，并有独立 global state 和 valid mask；更可能的解释是简单在线均值压缩后缺少可判别的结构，或现有 Memory controller 仍学不到何时应利用历史。继续增大原始 bank 不会解决这两个问题。

考虑到这是新增 global 路径从 v0.8 权重继续适配后的前 100 steps，尚不足以作为唯一终止点；但不允许无条件跑满两天。训练仅保留到 step500，并已启动不可变三卷复核 watcher（PID `3422258`）：

- checkpoint：`step_000500.pt`；
- 三路：off、oracle local K4、oracle local K4+G16；
- validation volumes：固定前 3 个；
- 输出：`data/outputs/volmem/diagnostics/memflowdit_v09_step000500_compact_gate_3vol/`；
- 判据不变：Volume 至少 `+0.001`、前景同向、减速不超过 10%。

若 step500 三卷门控仍失败，则停止 v0.9，compact 在线均值方案淘汰，不继续增加 global token 数量、bank 容量或额外长期模块。下一轮应优先检查 controller 的条件利用能力和全局摘要目标，而不是继续扩充 Memory 容量。

### 10.10 v0.9 step500 最终三卷门控与停止（2026-08-02）

step500 固定 3 个 validation volumes 的 off / oracle local K4 / oracle local K4+G16 配对已经完成：

| 模式 | Volume Dice | Δ vs off | FG slice Dice | Class Dice | slice/s | 速度变化 |
|---|---:|---:|---:|---:|---:|---:|
| off | 0.789150 | 0 | 0.762940 | 0.745665 | 0.386 | 0 |
| oracle local K4 | 0.789112 | -0.000038 | 0.764109 | 0.745272 | 0.317 | -17.8% |
| oracle local K4+G16 | 0.789166 | +0.000016 | 0.764250 | 0.745414 | 0.316 | -18.1% |

compact 相对 local K4 只有 `+0.000054`，仍近似零；相对 off 的 Volume 增益只有 `+0.000016`，远低于 `+0.001` 门槛。前景指标虽同向，但不能抵消体积指标无收益和 18.09% 的速度损失。`passes_gate=false` 后，决策守护先核对训练 PID 与配置，再于 `2026-08-02T02:02:07+08:00` 发送 SIGTERM；训练在 step608 停止。

结果目录：`data/outputs/volmem/diagnostics/memflowdit_v09_step000500_compact_gate_3vol/`

至此 v0.9 compact 在线均值方案正式淘汰，不再续训，也不增加 global token 或 raw bank 容量。Memory 路径已真实运行，但 v0.7、v0.8、v0.9 的多轮严格门控均未证明净收益，因此当前主线不能宣称 3D Memory 有效。
