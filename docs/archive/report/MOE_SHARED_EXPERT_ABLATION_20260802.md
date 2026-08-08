# MemFlowDiT DiT-MoE 共享专家严格消融（2026-08-02）

## 1. 问题与结论边界

当前 DiT 内部 `PrototypePhiMoE` 使用奇数三层 E4 Top-1，完全替换对应 dense FFN，没有共享专家。此前 E4/E8、Top-1/Top-2 消融没有回答“共享专家是否必要”，因此不能依据 ProMoE 或短程指标直接把无共享结构定为主线。

本实验只改变 DiT-MoE 的共享路径，输出头、MoonViT、数据、训练顺序、随机种子和评估协议保持一致。实验完成前不修改正式主线结论。

## 2. 对照结构

### Control：routed-only E4 Top-1

- 奇数三个 DiT block 使用 4 个全宽 SwiGLU 专家；
- 每个轮廓 Top-1，仅执行一个专家；
- 每个专家隐藏宽度 704；
- 不含共享 FFN。

### ProMoE-style：shared-half + routed-half E4 Top-1

- 奇数三个 DiT block 均包含一个始终执行的共享 SwiGLU；
- 另有 4 个路由专家，每个轮廓 Top-1；
- 共享和路由分支隐藏宽度均为 384；
- 每次执行一个共享分支和一个路由分支。

384 是在本项目 64 对齐约束下最接近 704/2 的硬件友好宽度。其激活 FFN 隐藏宽度为 768，相对 control 的 704 增加 9.09%，但完整模型每条路由的条件参数只增加约 0.36%。最终效率以 Batch 1/8 实测为准，不用理论 FLOPs 代替。

## 3. 公平初始化

两组均从同一个已验证 dense 2D checkpoint 开始。共享方案没有随机破坏已有 FFN：

1. dense FFN 前 384 个隐藏通道复制到共享分支；
2. 后 320 个通道复制到每个路由专家；
3. 路由专家剩余 64 个通道置零；
4. 初始化时 `shared(x) + routed_i(x)` 与原 dense FFN 输出严格相同，且与路由选择无关。

合成测试已覆盖 Top-1 输出等价和参数量关系；真实训练 1-step 烟测通过。

## 4. 参数审计（训练前）

| 指标 | Control | Shared | Shared 相对变化 |
|---|---:|---:|---:|
| 完整模型总参数 | 45.429174M | 43.364790M | -4.54% |
| 可训练参数 | 22.121050M | 20.056666M | -9.33% |
| 每条路由条件参数 | 40.563126M | 40.710582M | +0.36% |
| FP32 参数存储 | 173.299 MiB | 165.424 MiB | -4.54% |
| 单个 DiT-MoE 总参数 | 2.163712M | 1.475584M | -31.81% |

共享方案并未增加总参数，反而因为专家半宽而减少约 2.064M。它是否值得保留，主要取决于质量是否有明确改善，以及两个激活分支是否带来不可接受的真实延迟。

## 5. 训练与评估协议

- 两组训练：1000 step、chunks-per-step=4、chunk length=8；
- 训练 seed：`20260802`；
- 输出头统一为 8 专家 Top-2、共享专家、hard-phi；
- DiT 层位统一为奇数三层，专家数统一 E4，激活统一 Top-1；
- 质量评估：3 个固定 validation volumes、GT box、Memory-off、seed `20260731`；
- 速度评估：Batch 1 和 Batch 8；
- 使用两轮交叉 GPU 测量，第二轮交换 control/shared 的物理 GPU，降低卡间差异；
- 同时报告 Dice、前景 Dice、类别均值、路由负载、显存、总参数、激活参数和吞吐。

## 6. 综合决策原则

不会因为“来自 ProMoE”直接保留共享专家，也不会因为参数略有变化直接否决。按以下顺序判断：

1. 明确质量回退：淘汰；
2. 质量变化处于噪声级且速度无改善：不引入；
3. 质量基本持平但参数或速度明显改善：作为效率方案考虑；
4. 质量明确提升且 Batch 1/8 额外减速均在 10% 内：进入主线候选；
5. 质量提升很大时可以容忍一定效率代价，但必须明确披露总参数、激活参数和吞吐，不能只报 Dice。

## 7. 运行位置与当前状态

- Control 配置：`configs/volmem/verse_memflowdit_moe_shared_ablation_control_gpu0.yaml`
- Shared 配置：`configs/volmem/verse_memflowdit_moe_shared_ablation_promoe_gpu1.yaml`
- Control 输出：`data/outputs/volmem/verse_memflowdit_moe_shared_ablation_control_gpu0`
- Shared 输出：`data/outputs/volmem/verse_memflowdit_moe_shared_ablation_promoe_gpu1`
- 汇总目录：`data/outputs/volmem/diagnostics/moe_shared_ablation_20260802`
- 自动评估：`tools/volmem/watch_and_eval_moe_shared_ablation.sh`

## 8. 最终结果

两组均完成 1000 step。质量使用相同的三个验证体积、GT box、Memory-off 和固定评估 seed。

| 质量指标 | Control | Shared | Shared - Control |
|---|---:|---:|---:|
| volume mean Dice | **0.789760** | 0.788594 | **-0.001167** |
| volume mean IoU | **0.652978** | 0.651412 | **-0.001566** |
| foreground slice Dice | **0.762322** | 0.761818 | -0.000503 |
| all-slice Dice | **0.875094** | 0.874829 | -0.000265 |
| class mean Dice | **0.742783** | 0.741850 | -0.000933 |

三个体积均下降，分别为 `-0.000912`、`-0.000525` 和 `-0.002062`，不是由单个异常体积造成。17 个已观察类别中 10 类下降、7 类上升；类别 13 下降最大（`-0.011076`），没有形成稳定的类别收益。

两轮交叉 GPU 吞吐：

| 推理设置 | Control | Shared | Shared 相对变化 |
|---|---:|---:|---:|
| Batch 1 | **0.361940 slice/s** | 0.337521 slice/s | **-6.75%** |
| Batch 8 | **1.656853 slice/s** | 1.629311 slice/s | **-1.66%** |

训练最后 100 step 的平均单步时间从 `4571.81 ms` 增至 `4585.64 ms`（+0.30%）；峰值显存从 `16.610 GB` 增至 `16.667 GB`（+0.34%）。

共享结构也没有改善路由退化。三层训练末 hard CV：

| DiT-MoE 层 | Control | Shared |
|---|---:|---:|
| 第 1 个 MoE 层 | 0.3352 | 0.3371 |
| 第 2 个 MoE 层 | 0.5879 | 0.6004 |
| 第 3 个 MoE 层 | 1.0165 | 1.0419 |

最后一层最活跃专家占比由 66.43% 增至 67.67%。没有死亡专家，但共享分支没有使路由更均匀，反而轻微加重集中。

## 9. 决策与解释

**淘汰当前半宽共享专家方案，不合入主线。**

综合原因：

1. 总参数减少 4.54%，这是唯一明确优点；
2. 五个主要质量指标全部下降，且三个体积方向一致；
3. Batch 1/8 都变慢，没有把参数减少转化为实际速度收益；
4. 路由 hard CV 没有改善，因此没有验证“共享专家缓解退化”的预期；
5. 本任务中半宽 routed branch 把每个专家的可专门化隐藏容量从 704 压缩到 384（有效继承通道 320），共享分支承担公共变换，但没有补偿专家容量损失；
6. 当前 DiT 只有三层 MoE，另外三层仍是 dense FFN，且每个 block 已有残差路径；在这种浅层、混合 dense/MoE 结构中，额外共享 FFN 的边际价值可能低于大规模全 MoE 扩散模型。

因此不能因为 ProMoE 已验证共享专家，就直接迁移到当前轮廓演化网络。论文机制是研究依据，不是任务适配证据。本轮直接消融已经给出否定结果。

后续不再扩大共享专家宽度，因为全宽共享会进一步增加激活计算和延迟；在当前结果已经质量回退的情况下，不符合“提升不大不增加复杂度”的原则。主线继续使用 routed-only E4 Top-1 候选，但其层位和真实稀疏调度仍需分别解决。

状态：训练、GT/Memory-off 质量评估、两轮交叉 GPU Batch 1/8 测速和参数审计全部完成。机器可读结果见 `data/outputs/volmem/diagnostics/moe_shared_ablation_20260802/comparison.json`。
