# MemFlowDiT 下一阶段执行记录：Memory 因果、DiT-MoE 与并行推理

日期：2026-08-03  
状态：实验已启动，自动训练与评估链运行中

## 1. 本阶段目标

本阶段不再增加没有直接证据的复杂模块，集中回答三个问题：

1. 当前 Memory 是否依赖正确历史内容，而不只是 Memory 分支带来的普通残差偏置；
2. 固定 D1 dense residual 输出头后，Dense-6、Odd-3 MoE、All-6 MoE 和共享专家哪个在质量、参数量和真实速度上最合理；
3. 两遍 frozen volume 推理能否在不自回归的情况下利用前后切片，并保持可接受的速度。

所有新增训练最多 1000 step，统一使用 seed `20260802`、chunks-per-step `4`、同一 `shared_base_only.pt`，不运行 10 万 step。

## 2. 输出头 1000-step 复核

D1 dense residual 与最佳 MoE 控制 L0 正在从同一公平 checkpoint 重新训练：

- D1：GPU5，PID `164831`；
- L0：GPU4，PID `164834`；
- 检查时均到 step361，loss 有限，无 NaN/Inf；
- 完成后由 PID `164837` 在 GPU6 自动执行 GT memory-off、autoregressive 和两轮反向顺序测速。

结果目录：

`data/outputs/volmem/diagnostics/output_head_moe_2026_confirm_1000_20260803/`

## 3. 固定 D1 的四组 DiT FFN 对照

Odd-3 D1 直接复用上面的 D1 1000-step checkpoint；新增三组：

| 结构 | 总参数 | 每条路由条件参数 | 训练安排 |
|---|---:|---:|---|
| Dense-6 | 39.865760M | 39.865760M | GPU7，PID `195186`，运行中 |
| Odd-3 E4 Top-1 | 44.734880M | 39.868832M | 复用 D1 confirm |
| All-6 E4 Top-1 | 49.604000M | 39.871904M | 输出头质量与速度复核结束后在 GPU4 自动启动 |
| Odd-3 shared-half + routed-half | 42.670496M | 40.016288M | 输出头质量与速度复核结束后在 GPU5 自动启动 |

这里区分总参数和条件激活参数。MoE 的理论激活参数接近 Dense 并不等于真实速度相同，最终必须用同一物理 GPU 的 Batch1/Batch8 双轮反序测试判断。

自动链：

- 启动器：`tools/volmem/run_dit_ffn_d1_ablation_1000.sh`；
- 延迟启动 All-6/Shared：`tools/volmem/wait_and_launch_dit_ffn_d1_pending_v2.sh`，PID `205784`；该版本等待输出头最终 `comparison.json`，避免并发训练污染 GPU6 的速度复核；
- 自动评估：`tools/volmem/watch_and_eval_dit_ffn_d1_ablation.sh`，PID `195194`；
- 汇总：`tools/volmem/summarize_dit_ffn_d1_ablation.py`；
- 结果：`data/outputs/volmem/diagnostics/dit_ffn_d1_ablation_20260803/`。

四组均完成后，在 GPU6 串行执行两轮反向顺序的 Batch1/Batch8、三验证体积、GT-box、Memory-off 评估。汇总同时报告 Dice、前景 Dice、类别 Dice、参数、显存、吞吐、每层 hard CV 和死亡专家。

## 4. D1 step1000 Memory 内容因果审计

审计等待 D1 step1000、Dense-6 训练和输出头速度评估结束后在 GPU7 自动运行，避免并发任务污染速度结果。守护 PID：`198831`。

### 4.1 单体积机制矩阵

- `parallel-off`；
- autoregressive K=`1/2/4/8/16`；
- frozen predicted causal K=`4/8/16`；
- frozen predicted bidirectional K=`4/8/16`；
- frozen shuffled K=`4/8/16`；
- frozen predicted causal all-history K=`256`；
- frozen feature-only K4；
- frozen GT-oracle K4。

AR K1 也等价于“只重复最近状态”的非冗余版本：完全相同的 K/V token 重复进入 softmax 不增加信息，因此不单独实现重复 token 分支。

### 4.2 三体积因果确认

复用输出头复核中的 full off 和 autoregressive K4，再新增：

- frozen causal K4；
- frozen bidirectional K4；
- frozen shuffled K4；
- frozen causal all-history K256；
- frozen GT-oracle K4。

预注册判据：

1. normal causal 至少比 shuffled 高 `0.001 Dice`，才认为存在可检测的历史内容依赖；
2. normal causal 至少比 off 高 `0.001`，才认为两遍 Memory 有净质量收益；
3. all-history 至少比 bounded K4 高 `0.001` 且速度可接受，才考虑扩大历史；
4. oracle 有效而 predicted 无效，根因指向 evidence 质量或误差累积；
5. oracle 也无效，根因仍在 Memory 表征或 controller，而不是 bank 数量。

自动链：

- 评估：`tools/volmem/watch_and_run_memory_causal_d1_1000.sh`；
- 汇总：`tools/volmem/summarize_memory_causal_d1_1000.py`；
- 结果：`data/outputs/volmem/diagnostics/memory_causal_d1_step1000_20260803/`。

## 5. 两遍并行推理

仓库现有 `eval_memflowdit_parallel.py` 已实现所需原型，不再新建重复模块：

1. 第一遍对全体切片独立并行生成 coarse prediction 和 Memory state；
2. 固定整卷 state table；
3. 第二遍按 causal 或 bidirectional 策略为每个目标切片选状态；
4. 所有目标切片可分 batch 独立细化，不再依赖上一张最终输出。

本阶段直接通过 `frozen-causal` 和 `frozen-bidirectional` 与 autoregressive、off、shuffled 做因果比较。只有两遍方案通过质量和速度门槛，才进入主线；否则保留为否定实验，不包装成核心 3D 贡献。

## 6. 后续实现门槛

MoonViT 相似度历史选择已作为纯评估策略实现，但不会先于因果结果合入主线。策略固定保留最近状态，其余 K-1 个位置按当前 coarse state 与历史 Memory key 的余弦相似度选择；Memory key 由冻结 MoonViT 特征的同一 8×8 池化路径产生，不新增参数、router、loss 或训练分支。

实现位置：

- `volmem/models/memory_bank.py`：`causal-recent-key-similar`；
- `tools/volmem/eval_memflowdit_parallel.py`：`frozen-key-similar`；
- `tests/volmem/test_memflowdit.py`：最近状态保留、相似历史命中和错误 target 防护测试；
- 19 项 Memory 测试全部通过；
- versioned watcher：`tools/volmem/watch_and_run_memory_key_selection_d1_1000_v1.sh`，PID `204133`。

该 watcher 等待基础因果矩阵结束，再顺序评估 quick K4/K8/K16 和三体积 K8/K16，不与基础评估争抢 GPU7。若 shuffled 与 normal 无差异，或 key-similar 不超过简单 causal-nearest，则删除该选择策略，不继续优化 selector。

最终主线保持一个输出头、一个 DiT FFN 方案和一个 Memory 策略。任何新增结构如果提升低于噪声、造成死亡专家或带来不成比例的速度代价，均淘汰。

## 7. 输出头 1000-step 最终结果（2026-08-03 06:54）

| 指标 | D1 dense residual | L0 legacy E8K2 | D1 - L0 |
|---|---:|---:|---:|
| Memory-off Volume Dice | 0.773345 | **0.774311** | -0.000966 |
| Autoregressive Volume Dice | 0.772983 | **0.773881** | -0.000898 |
| Foreground slice Dice | **0.726761** | 0.724233 | +0.002528 |
| Batch1 slice/s | **0.385203** | 0.343674 | +12.08% |
| Batch8 slice/s | **2.175116** | 1.718635 | +26.56% |

三个验证体积的 Memory-off 差值为 `-0.004917/-0.000731/+0.002749`，方向并不一致；autoregressive 同样为两负一正。因此 L0 的约 `0.001` 体积均值优势不足以证明稳定质量提升。D1 同时具有更高前景切片 Dice、更少约 0.694M 总参数、更小输出头和显著更高吞吐。

L0 step1000 的输出路由仍有 1 个低于 1% 的死亡专家。D1 是该 fair-head 组内更合理的效率候选；但随后完成的历史审计确认，该组剥离了 epoch130 中 15 个已经训练成熟的输出 specialist/router tensor，绝对能力明显下降。历史三体积最好仍是 v0.5 step2300 的 `0.796574`。因此撤回“D1 已成为绝对主线”的表述，在函数保持迁移或蒸馏完成前保留 v0.5 成熟输出头。详见 `docs/report/HISTORICAL_BEST_AUDIT_20260803.md`。

机器可读结果：

`data/outputs/volmem/diagnostics/output_head_moe_2026_confirm_1000_20260803/comparison.json`

## 8. Memory 因果审计最终结果

| 三体积模式 | Volume Dice | Δ vs off | FG Dice | slice/s | Peak GB |
|---|---:|---:|---:|---:|---:|
| Off | **0.773345** | 0 | 0.726761 | **2.175** | 2.57 |
| Autoregressive K4 | 0.772983 | -0.000362 | **0.733616** | 0.344 | 0.54 |
| Frozen causal K4 | 0.773953 | +0.000608 | 0.726771 | 0.950 | 2.65 |
| Frozen bidirectional K4 | 0.773620 | +0.000275 | 0.724696 | 0.919 | 2.65 |
| Frozen shuffled K4 | 0.765031 | -0.008314 | 0.712234 | 0.979 | 2.65 |
| GT-oracle causal K4 | 0.773986 | +0.000641 | 0.726778 | 1.631 | 2.60 |
| Frozen all-history K256 | 0.767267 | -0.006078 | 0.716846 | 0.575 | 9.41 |

这组结果把“Memory 是否真实运行”和“Memory 是否值得启用”分开了：

1. 正常 causal 历史比 shuffled 高 `+0.008922`，证明模型确实读取并依赖历史内容，Memory 不是恒等或完全失效；
2. 但正常 causal 只比 off 高 `+0.000608`，未达到预注册 `+0.001` 门槛；autoregressive 反而低 `-0.000362`，因此当前仍不能声明净收益；
3. oracle 与 predicted causal 几乎相同（差 `+0.000032`），主要瓶颈不是预测 mask evidence，而是 controller/Memory 表征的收益上限；
4. bidirectional 没有超过 causal，说明加入未来邻层没有自动形成互补；
5. all-history 比 bounded K4 低约 `-0.006686`，相对 off 慢 73.6%、峰值显存增至 9.41GB。无限 bank 明确淘汰；
6. 单体积 K1/K2/K4/K8/K16 也没有随容量增长转正，较大的 K8/K16 更差。

因此当前推理主线默认关闭 Memory；保留 bounded K4 和 frozen two-pass 作为研究/视频能力代码，但不把 3D Memory 写成已验证贡献。下一轮若继续研究，应修改 controller 的条件利用方式，而不是继续扩 bank。

机器可读结果：

`data/outputs/volmem/diagnostics/memory_causal_d1_step1000_20260803/comparison.json`

## 9. 相似检索与 DiT 四组当前进度

Memory-key quick 结果为：K4 `0.772664`、K8 `0.772043`、K16 `0.771619`。相对相同 K 的 causal-nearest 差值只有 `-0.000075/+0.000087/+0.000233`，全部低于 off，暂时没有保留依据。三体积 K8/K16 正在完成最终确认。

四组 DiT 均已训练到 step1000。最后 100 step：Dense-6 平均 loss `0.007025`、训练步耗时 `4481 ms`；Odd-3 为 `0.008573/4995 ms`；All-6 为 `0.010080/4974 ms`；共享 Odd-3 为 `0.008536/4729 ms`。第一轮 Batch1 已完成的三组质量为：

| 结构 | Volume Dice | FG Dice | slice/s |
|---|---:|---:|---:|
| Dense-6 | **0.774402** | **0.734874** | **0.4407** |
| Odd-3 | 0.772522 | 0.731859 | 0.4023 |
| All-6 | 0.770356 | 0.729325 | 0.3503 |

Dense-6 当前同时领先质量和速度，All-6 明显不占优；共享 Odd-3 正在评估。最终决策仍等待共享组、第二轮反序和 Batch8 全部结束后落盘。需要强调的是，这组使用降质后的 fair-head 起点，只作为重复确认；保留成熟输出头的旧严格实验已经得到 Dense-6 `0.790473` 高于 Odd-3 `0.789760`，并且 Dense-6 更快、更小。

## 10. 历史最好结果复核与纠正

对 175 个历史 `summary.json` 的统一审计确认：在同三个验证体积、GT box、Memory-off/parallel-off、seed `20260731`、333 张切片协议下，最高结果不是本报告的 D1 `0.773345`，而是 v0.5 step2300 的 **0.796574**。

本报告中 D1/L0 的低绝对值来自有意剥离成熟输出专家后的公平起点；该设计适合比较新头的相对效率，却不适合选择绝对主线。当前正确状态为：v0.5 step2300 是质量 anchor，Dense-6 是 DiT 选择，成熟旧输出头暂时保留，D1 等待函数保持迁移验证。

完整审计：`docs/report/HISTORICAL_BEST_AUDIT_20260803.md`
