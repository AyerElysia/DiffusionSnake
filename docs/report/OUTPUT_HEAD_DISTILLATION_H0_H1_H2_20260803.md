# MemFlowDiT 输出头 H0/H1/H2 蒸馏实验（2026-08-03）

## 1. 实验目的

本实验回答一个单独而明确的问题：在不改变 v0.5 主干、数据、检测框、Memory
策略和推理噪声的前提下，能否用更简单、更快、路由更稳定的输出头，保留历史最佳
输出 MoE 的质量。

历史质量锚点为 v0.5 step2300。在严格三验证体积、GT box、Memory-off、seed
20260731、333 slices、Batch-8 协议下，其 Dice 为 **0.7965737849**，FG Dice 为
**0.7700270444**。该结果来自成熟 E8 Top-2 输出 MoE，因此不能用此前从
`shared_base_only.pt` 重新训练 1000 step 的 D1/L0 结果直接否定输出 MoE；后者移除了
15 个成熟输出专家/路由张量，比较的训练起点不同。

## 2. 受控变量与候选

三组均固定：

- v0.5 step2300 的非输出头全部权重；
- MoonViT layer18、center-only 特征；
- Dense6 DiT，不引入 DiT-FFN MoE；
- GT box、Memory-off / parallel-off；
- ODE steps、噪声种子、验证体积和 Batch 大小。

仅替换 `final_layer`：

| 组别 | 输出头 | 作用 |
|---|---|---|
| H0 | 成熟旧 E8 Top-2 MLP MoE | 质量上界与现网锚点 |
| H1 | 单个 hidden=1024 Dense Residual MLP | 无路由效率对照 |
| H2 | H1 共享头 + E4 Top-1 hidden=128 稀疏残差适配器 | 验证少量专门化是否值得 |

H2 按整条轮廓路由，不按点路由；先选专家再执行，未被选中的专家不计算。H2 不再
引入 point embedding、环形路由卷积、类别路由或多级共享专家。设计有意保持最小化。

## 3. 功能保持初始化

H1 不是随机重训输出头。旧头共享路径为：

`linear(x + shared_mlp(x))`

初始化时，将旧 `norm / adaLN / linear` 原样复制，并把
`linear(shared_mlp(x))` 等价折叠到 H1 残差 MLP 的前 256 个隐藏通道。因而 H1
初始化已经精确保留旧头的共享函数，蒸馏只需拟合路由专家产生的剩余项。

H2 从蒸馏完成的 H1 载入同名共享权重。四个专家的输出层为全零，因此 H2 step0 与
H1 输出逐元素完全相同。路由采用轮廓级 Top-1 和 straight-through softmax 梯度；
负载控制采用 hard-load EMA 更新的非训练 routing bias，不添加与位移目标竞争的辅助
loss。

## 4. 蒸馏数据与防泄漏

- 教师：`data/outputs/volmem/rl3d/ckpt_backup/v05_step_002300.pt`；
- 数据：training split 的 8 个体积；
- 轨迹：GT box、parallel-off、真实 5-step ODE 推理轨迹；
- 缓存内容：最终输出头输入 `x`、时间条件 `t_emb` 和 H0 输出；
- 抽样：每 8 次输出头调用保存一次。由于 8 与 5 个 ODE 调用互质，抽样会轮转覆盖
  不同扩散时刻，同时把预计约 46GB 的高度重复缓存压到约 6GB；
- 严格验证仍只使用 val split 的三个体积，训练缓存不包含验证样本。

曾试运行全密度缓存，101 个切片已产生约 8.8GB；服务器磁盘占用已达 94%，因此停止
了本实验自己启动的缓存进程，删除未完成临时缓存后改为上述均匀抽样。没有停止或修改
任何既有训练和评估进程。

## 5. 训练与验证协议

1. H1：冻结 `norm / adaLN / linear` 和完整主干，仅训练 H1 `residual_mlp`；
2. H2：冻结 H1 共享路径，仅训练路由器和四个小专家；
3. 头部离线目标：H0 输出的逐元素 MSE；
4. 保存验证 RMSE 最低的头，并移植回 H0 完整 checkpoint；
5. H1/H2 分别运行 Batch-1、Batch-8 的三体积严格评估；
6. 同时报告 Dice、FG Dice、参数量、端到端秒数、峰值显存和 H2 专家负载。

判定原则：

- H1 若质量基本保持且明显更快/更小，优先作为效率候选；
- H2 只有在严格 Dice 明确优于 H1，或恢复 H0 质量且专家无退化时才保留；
- H2 若提升小于测量波动，则删除 H2，选择更简洁的 H1；
- 任一蒸馏头若明显低于 H0，则保持 H0，不以参数量为由牺牲主结果。

## 6. 实现与产物

核心实现：

- `lib/networks/diffusion/dit_denoiser_v4.py`：`SharedDenseSparseResidualHead`；
- `tools/volmem/eval_memflowdit_parallel.py`：训练集推理轨迹缓存；
- `tools/volmem/distill_output_head.py`：H1/H2 蒸馏、权重移植、参数统计；
- `tools/volmem/watch_output_head_h0_h1_h2.sh`：缓存、蒸馏、四路严格评估；
- `configs/volmem/verse_memflowdit_output_head_h1_distilled_dense_gpu0.yaml`；
- `configs/volmem/verse_memflowdit_output_head_h2_shared_sparse_gpu1.yaml`。

实验根目录：

`data/outputs/volmem/output_head_h0_h1_h2_20260803/`

## 7. 蒸馏拟合结果

最终缓存覆盖 8 个训练体积、611 个切片、5700 次输出头调用，共保留 43,544 条轮廓，
分成 139 个 shard。实际缓存约 2GB。

| 头 | 初始相对 RMSE | 最佳相对 RMSE | 最佳 step | 验证路由 |
|---|---:|---:|---:|---|
| H1 Dense | 27.6278% | **0.4776%** | 3800 | 无路由 |
| H2 E4 pilot | 0.4776% | 0.4609% | 2700 | 31.2% / 0 / 0 / 68.8%，2 个死专家 |
| H2 E2 anti-collapse | 0.4776% | 0.4619% | 2500 | 68.6% / 31.4%，无死专家 |

E4 只把 H1 的拟合误差改善约 3.5%，却退化为两个有效专家。因此增加了一个不扩展设计
复杂度的修订：删掉两个冗余专家，将路由 logits 改为有界 cosine routing，并把动态
load-bias 的范围提高。E2 成功消除死专家，且拟合误差与 E4 基本相同，说明 E4 的两个
闲置专家确实没有提供有效容量。

## 8. 严格 Batch-8 结果

所有结果均为同 3 个 val 体积、333 slices、GT box、Memory-off / parallel-off、seed
20260731。H0 在当前环境重新评估，以避免拿不同机器负载下的历史耗时直接比较。

| 组别 | 总参数 | 头参数 | Volume Dice | FG Dice | 时间(s) | slices/s | 死专家 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 旧 E8 Top-2 | 40.560M | 1.092M | 0.796574 | 0.770027 | 179.070 | 1.8596 | 0（但 Top-1 高度偏置） |
| **H1 Dense** | **39.866M** | **0.398M** | **0.796703** | **0.770191** | **140.281** | **2.3738** | 不适用 |
| H2 E4 pilot | 40.000M | 0.531M | 0.796636 | 0.770006 | 150.795 | 2.2083 | 2 |
| H2 E2 anti-collapse | 39.933M | 0.465M | 0.796658 | 0.770264 | 159.681 | 2.0854 | 0 |

相对 H0，H1：

- 总参数减少 694,294（约 1.71%）；
- 输出头参数减少约 63.6%；
- 端到端时间减少约 21.7%，吞吐提高约 27.7%；
- Volume Dice 增加 0.000130，属于完全保持质量，不能解读为统计显著提升。

H2-E2 虽然解决了专家退化，但相对 H1 多 67,076 个参数、慢约 13.8%，Dice 还低
0.000045；H2-E4 同样没有质量收益，并存在两个死专家。因此两个 H2 均被 H1 严格
支配。

## 9. 严格 Batch-1 复核

| 组别 | Volume Dice | 时间(s) | slices/s | 峰值显存(GB) |
|---|---:|---:|---:|---:|
| H0 旧 E8 Top-2 | 0.796106 | **792.462** | **0.42021** | 0.5138 |
| **H1 Dense** | **0.796296** | 796.955 | 0.41784 | **0.5102** |
| H2 E4 pilot | 0.796163 | 821.230 | 0.40549 | 0.5109 |
| H2 E2 anti-collapse | 0.796200 | 822.746 | 0.40474 | 0.5106 |

Batch-1 下 H1 与 H0 的速度差为 -0.56%，属于运行波动范围，应表述为持平，不能宣称
H1 在小 batch 更快；但 H1 Dice 仍高 0.000190，且比两个 H2 快约 3%。结合 Batch-8，
H1 的效率结论是：单切片延迟不变，批量吞吐显著更高。

## 10. 最终结论

当前输出头的推荐架构是 **H1 Dense Residual**，而不是继续保留输出 MoE。这里的结论
不是“MoE 对所有位置都无效”，而是更具体的：在成熟 v0.5 表征之后，输出位移函数可以
被一个共享 MLP 以小于 0.5% 的相对函数误差蒸馏；额外输出专家无论是否解决负载退化，
都没有转化为严格 Dice 收益。

H1 checkpoint：

`data/outputs/volmem/output_head_h0_h1_h2_20260803/distilled/h1_distilled_full.pt`

因此不建议为了“保留 MoE 形式”继续增加专家、共享专家或路由支路。下一步只需完成
额外微调应另设受控实验，不能覆盖当前纯蒸馏 checkpoint；当前 H1 已通过 Batch-1 与
Batch-8 复核，可以作为主线输出头候选，H0 保留为教师和可回退 checkpoint。

统一机器可读汇总：

`data/outputs/volmem/output_head_h0_h1_h2_20260803/comparison.json`
