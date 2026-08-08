# RL 后训练失败诊断报告

日期：2026-05-19  
对象：V3.4-FM / YOLO-M 轮廓后训练  
结论：当前失败不是单一奖励项问题，而是“奖励来源、可学习动作、权重更新路径、验证方式”四者不匹配。

## 1. 外部方法参考

RL 大佬们排查 PPO / diffusion RL 时通常先看这些问题：

- PPO 是否真在做 policy update：ratio、KL、clipfrac、advantage、old logprob 是否正常。参考 PPO 原论文：https://arxiv.org/abs/1707.06347
- RL 实验方差是否被误当作提升：必须用足够评估样本、重复种子、显著性判断。参考 Deep RL that Matters：https://arxiv.org/abs/1709.06560
- PPO 实现细节是否改变算法行为：code-level optimization 会显著影响结果。参考 Implementation Matters：https://arxiv.org/abs/2005.12729
- diffusion RL 必须把每一步去噪/采样看成有 logprob 的 action，奖励要能回传到真实采样动作。参考 DDPO：https://rl-diffusion.github.io/
- 奖励被过度优化会导致真实指标下降。参考 reward overoptimization：https://arxiv.org/abs/2210.10760

这些标准套到我们的任务上，核心检查项是：

| 检查项 | 我们要看什么 |
|---|---|
| policy 是否真的更新 | `policy_loss`、`ratio`、`KL`、`grad_norm` |
| 正奖励是否来自可学习 action | 好候选到底来自 step action、latent x0，还是随机 seed |
| 更新是否安全 | eval 是否随训练单调改善，是否后期崩 |
| 奖励是否泛化 | 小验证集提升是否能过完整 150 测试集 |
| 是否发生过优化 | 训练 reward 上升但完整测试下降 |

## 2. 我们已有 RL run 的统一汇总

### 2.1 强 distillation 类：一有强更新就崩

| run | 最好小验证 | 最后小验证 | 现象 |
|---|---:|---:|---|
| V8 adv | -0.0090 | -0.2028 | 快速崩 |
| V8b fast | -0.0050 | -0.7072 | 灾难性崩 |
| V8c detgate | -0.0059 | -0.5471 | 灾难性崩 |
| V9 searchdistill | -0.0041 | -0.1432 | 搜索蒸馏崩 |
| V12 boundary-late | +0.0026 @ step25 | -0.0059 @ step60 | 先涨后跌 |

判断：只要更新强到能明显改变主模型，长程就不稳定。  
这不是“奖励不够强”，而是更新方向本身会污染主模型。

### 2.2 弱 distillation / 多起点类：稳定但几乎不学

| run | 完整测试结果 |
|---|---:|
| V5b step20 | median 0.8925 |
| V5b step40 | median 0.8925 |
| V5b step60 | median 0.8924 |
| V5b step80 | median 0.8925 |

判断：弱更新能避免崩，但没有实际收益。

### 2.3 latent x0 / ranker 类：小验证会涨，但不稳定

| run | 最好小验证 | 最后小验证 |
|---|---:|---:|
| V10a latent PPO | +0.0036 @ step10 | +0.0011 |
| V10b latent PPO | +0.0034 @ step10 | +0.0011 |
| V10c det-adv latent PPO | +0.0037 @ step25 | -0.0018 |
| V11c ranker | +0.0036 @ step10 | +0.0007 |

判断：latent / ranker 能在小验证集上看到信号，但后续不稳定。  
这类结果必须过完整测试，不能只看 31 个 contour 的固定小验证。

## 3. V12 关键实验结论

V12 是按“近 GT 状态 + 边界距离奖励”做的针对性实验：

| 项目 | 结果 |
|---|---:|
| 基础权重加载 | 100% |
| 新奖励 | IoU + Dice + mBoundF + 双向轮廓距离 |
| 训练分布 | 85% late-t 状态 |
| 小验证最好 | step25，+0.002604 |
| 小验证最后 | step60，-0.005896 |
| 完整测试 step25 | mean IoU 0.892015，median 0.891210 |
| 完整测试是否有效 | 否 |

对比：

| 方法 | median IoU |
|---|---:|
| V12 step25 RL 权重 | 0.891210 |
| V3.4-FM baseline 附近 | 约 0.8925 |
| V3.4-FM + noise=0.5 | 0.897344 |
| V3.4-FM + anneal noise 0.6/0.5/0.5 | 0.897490 |

判断：边界距离奖励有局部信号，但没有泛化成完整测试收益。

## 4. 严格根因判断

### 根因 1：很多 run 不是合格的 policy-gradient RL

大量日志里：

- `policy_loss = 0`
- `grad_norm = 0`
- `ratio_mean = 0` 或 `1`
- `clipfrac = 0`

这说明 PPO 主路径要么没有有效 action logprob，要么 action 与奖励来源断开。

实际有效更新常常来自 distillation，而不是 PPO。  
所以很多实验名字叫 RL，但训练机制更像“best-of-k 自训练”。

### 根因 2：正收益来源和可训练动作不一致

我们看到的正收益主要来自：

- 不同 x0 seed
- best-of-k 搜索
- 推理噪声 scale
- 短期 ranker 选择

但直接更新的是：

- velocity field
- latent policy 小头
- ranker
- 或 distillation 目标

正收益来自“采样/搜索找到好候选”，不等于“主模型参数应该朝这个方向更新”。  
这解释了为什么 best-of-k 有 oracle gain，但蒸馏回权重就失败。

### 根因 3：在线蒸馏会自我污染

流程是：

1. 当前模型生成 K 个候选。
2. 用奖励挑最好候选。
3. 把最好候选教回当前模型。
4. 模型变一点，下一轮候选分布也变。
5. 错误方向积累，后期退化。

V12 证明了这一点：step25 小验证涨，step50/55/60 连续下跌。

### 根因 4：小验证集不足以判断提升

V12 小验证：

- step25：+0.002604

完整测试：

- median IoU：0.891210
- 不如 baseline，也不如噪声推理方案

这说明固定 31 个 contour 的 eval 很容易把局部波动误判成进步。  
后续所有 RL 判断必须以完整测试或更大的固定验证集为准。

### 根因 5：奖励函数有问题，但不是最大问题

边界距离奖励确实让早期信号变清楚。  
但是完整测试没涨，说明：

- 奖励不是完全没用；
- 但仅换奖励不能解决权重更新方向污染；
- 奖励需要配合真正可学习的 action 和安全约束。

## 5. 当前最可能的失败链条

```text
模型已经很强
→ 可提升空间很小
→ best-of-k 找到的提升多是局部/采样幸运
→ 小验证集容易误判为提升
→ 在线蒸馏把幸运样本当成训练目标
→ 主模型分布漂移
→ 细节或整体开始退化
→ 完整测试不涨甚至下降
```

## 6. 后续必须做的排查实验

### 实验 A：动作归因

目的：确认奖励到底来自哪个 action。

固定同一批样本，分别只打开：

| 条件 | 目的 |
|---|---|
| 只变 x0 seed | 看随机起点贡献 |
| 只变 ODE step noise | 看 step action 是否有正收益 |
| 只变 noise scale | 看推理噪声贡献 |
| 只训练 latent policy | 看是否可学习 |
| 只 distill | 看蒸馏是否污染 |

如果只有 x0/noise scale 有收益，而 PPO step action 没收益，就不能继续用普通 PPO 更新主模型。

### 实验 B：更新方向验证

目的：判断一次更新是不是朝正确方向。

做法：

1. 固定 100 个验证样本。
2. 每次只训练 1 step。
3. 训练后立刻评估同一批样本。
4. 记录“每一步更新后，改善样本数 vs 退化样本数”。

如果退化样本数长期大于改善样本数，说明更新方向错，不是训练不够久。

### 实验 C：奖励泛化验证

目的：判断 reward 是否过拟合。

每个 checkpoint 同时看：

| 指标 | 作用 |
|---|---|
| train reward | 是否被优化 |
| fixed-val reward | 是否泛化 |
| full-test IoU | 最终真实指标 |
| boundary distance | 是否真修细节 |

如果 train reward 涨而 full-test IoU 跌，就是过优化。

### 实验 D：安全约束验证

目的：确认能不能做“改权重但不崩”的 RL。

约束：

- 每次更新必须比原模型同 batch 好。
- 加 anchor loss：新模型输出不能偏离原模型太多。
- 加 early-stop：大验证集连续两次下降直接停止。
- 只允许 LoRA / 小 adapter / late head 更新，暂不动全主干。

这是仍然“改权重”的 RL，但不会让整网被少数幸运候选拖走。

## 7. 当前建议

不建议再跑无门控长程 RL。

下一步应该先做三个诊断：

1. 动作归因：到底哪个 action 产生正收益。
2. 单步更新方向：一次 RL 更新到底让多少样本变好/变差。
3. 奖励泛化：训练 reward 是否和完整测试 IoU 同向。

只有这三个过了，才继续长程改权重 RL。

如果必须坚持“改权重 RL”，推荐从下面这个安全版本开始：

```text
冻结主干
只训练小 adapter / LoRA / latent policy
奖励 = 相对原模型提升
奖励项 = IoU + Dice + mBoundF + 双向边界距离
每 5 step 大验证
连续下降立即回滚
```

核心原则：先证明“每一步更新方向是正的”，再谈长程训练。
