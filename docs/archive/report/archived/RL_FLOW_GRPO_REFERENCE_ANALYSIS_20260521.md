# Flow-GRPO 参考方案分析

日期：2026-05-21

对象：
- 参考实现：`flow_grpo-main`
- 当前实现：`grpo_train_v2.py`

## 结论

`flow_grpo-main` 给出的关键启发不是“换一个奖励函数”，而是严格保证：

1. 采样时产生的每个随机动作都有 old logprob。
2. reward 只评价这些随机动作最终造成的结果。
3. 更新时重新计算同一个动作的新 logprob。
4. 用 `new_logprob - old_logprob` 做 clipped GRPO 更新。
5. KL、clipfrac、approx_kl 必须持续监控。

我们当前 RL 崩溃的核心问题，是很多正收益来自 `x0/search/best-of-k/distillation`，但真正有概率约束、能被 GRPO 正确更新的动作并不总是这些收益来源。也就是说，reward source 和 policy action 没有稳定对齐。

## 参考实现的关键做法

`flow_grpo-main/scripts/train_flux_kontext.py` 的流程是：

1. 采样一批结果，同时保存每个 flow step 的 `latents`、`next_latents`、`log_probs`。
2. 对最终结果算 reward。
3. 按同一 prompt/group 做 advantage 标准化。
4. 训练时逐 timestep 重新计算当前模型对当时 `next_latent` 的 logprob。
5. 用 `ratio = exp(new_logprob - old_logprob)` 做 clipped GRPO。
6. 可选加入 reference KL，防止模型漂移。

它没有把“搜索出来的最好图”直接硬蒸馏回主模型作为主训练路径。

## 与我们当前实现的差异

| 项目 | Flow-GRPO | 我们当前实现 |
|---|---|---|
| action | 每个 denoise/flow step 的随机状态转移 | step noise、initial latent、best-of-k、distill 混在一起 |
| old logprob | 采样时完整保存 | step action 有，部分 x0/search 来源原来没有 |
| reward 对齐 | reward 评价同一条 sampled trajectory | 正收益常来自 search 或最终候选选择 |
| update | 只更新有 logprob 的 sampled action | PPO、latent PPO、ranker、distill 多路径并存 |
| 风险控制 | KL、clipfrac、advantage clipping | 有监控，但蒸馏路径可绕过 GRPO 约束 |
| 训练窗口 | 支持只训练部分 timestep/window | 我们也有 window，但收益来源未完全对齐 |

## 对失败原因的修正判断

之前不能简单说“PPO 失败”。更准确的说法是：

当前主线是 GRPO，但代码里有 PPO-style ratio/clip 分支。问题不是用了 PPO 这个名字，而是 GRPO 更新只对有 logprob 的动作成立。此前很多实验的正收益来自 x0/search/best-of-k，更新却落在 velocity field 或 distillation 上，因此长程会漂移。

## 下一版应该怎么改

推荐做 `V13 Flow-GRPO aligned`，只验证一个问题：改权重式 GRPO 在 action/reward 对齐后是否还会崩。

必须保留：

- 从 V3.4-FM 最新权重继承，能加载多少就加载多少。
- 不随机重训主模型。
- 仍然是改权重式 RL。
- 使用完整测试或大固定验证集判断，不再只看小验证。

必须去掉或默认关闭：

- best-of-k hard distillation。
- advantage distillation。
- ranker 作为主更新路径。
- 没有 logprob 覆盖的 search 结果作为训练目标。

推荐训练设计：

| 模块 | 设置 |
|---|---|
| policy action | late window 的 flow step noise |
| group | 同一 contour、同一 init、同一 GT，采 K 条随机 trajectory |
| reward | 相对 deterministic baseline 的提升 |
| reward 组成 | IoU + Dice + mBoundF + 双向边界距离 |
| update | 只用 step logprob 做 GRPO ratio/clip |
| KL | 对 frozen reference flow 加 KL |
| window | 只训练后 20%-40% ODE step |
| eval | 每 5-10 step 大固定验证，定期完整测试 |
| stop | 连续下降立即停止，不做长程硬跑 |

## 为什么这个方案更合理

这个方案只让模型学习它真正“做过并且有概率记录”的动作。reward 如果变好，优势就推高这条 sampled trajectory 的概率；reward 如果变差，就压低它的概率。这样才是合格的 GRPO。

以前的 search/distill 路径会把“幸运候选”当成监督标签，模型参数一变，下一轮候选分布也变，错误会累积。这就是长程后期下降的主要原因。

## 第一轮实验标准

不要直接长训。先做 50-100 step 的严密验证：

1. `policy_loss` 不为 0。
2. `ratio_mean` 接近 1，但不是恒等于 1。
3. `clipfrac` 不长期为 0，也不能过高。
4. `approx_kl` 小于目标阈值。
5. `reward_best - det_score` 有正样本，但 `reward_mean` 不能长期为负。
6. fixed-val IoU 至少不低于 baseline。
7. full-test 不能低于当前最佳推理噪声基线。

如果这些不过，不能继续长程训练。

## 当前最重要的判断

奖励函数确实要改，但不是第一优先级。第一优先级是 action/reward/logprob 对齐。

轮廓到 GT 的距离奖励可以保留，因为它对细节更敏感；但它只能作为 reward 的一个分量，不能再配合 hard distillation 使用。
