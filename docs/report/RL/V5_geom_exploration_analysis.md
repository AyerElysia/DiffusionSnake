# V5 Geom RL 探索机制分析报告

日期：2026-06-04

## 1. 结论先说

当前正式运行的强化学习版本是 **V5 geom action**：

- 基础权重：`epoch_3500.pt`
- 策略：三步外层 contour refinement
- 每一步 action：模型给出 deterministic flow 位移均值，然后叠加一个低频法向随机扰动
- K：每个训练样本采样 `8` 条 rollout
- 更新目标：让模型更倾向于产生那些比 deterministic baseline 更好的随机轨迹

它的“探索”不是来自扩散模型的随机初始化噪声，也不是来自一个显式 learned stochastic policy head。  
它的探索来自：

```text
deterministic flow mean + 手工定义的低频几何随机扰动
```

也就是说，当前 V5 更像是：

```text
在当前模型预测附近做局部随机搜索，然后把搜索到的好方向蒸馏回 flow 模型
```

这解释了为什么它能涨一点，也解释了为什么它涨得慢：探索空间是人为限制的，主要覆盖轮廓法向上的低频形变，不能自由探索所有 diffusion latent / seed 空间。

## 2. 当前 V5 的 action 到底是什么

当前配置是：

```yaml
rl_v4_k: 8
rl_v4_outer_steps: 3
rl_v4_fractions: [0.3333, 0.5, 1.0]
rl_v4_geom_lowfreq_modes: 8
rl_v4_geom_sigma_px: [1.2, 0.8, 0.5]
rl_v4_ppo_inner_epochs: 2
rl_v4_ppo_clip: 0.05
rl_v4_kl_beta: 0.01
rl_v4_lr: 5.0e-8
```

一次 rollout 有三步 action。第 `s` 步：

```text
state_s = 当前 contour
mean_s = 当前 flow 模型给出的 deterministic 位移
z_s ~ N(0, I)，维度是 geom_lowfreq_modes=8
delta_s = lowfreq_basis(z_s) * contour_normal * sigma_s
action_s = mean_s + delta_s
state_{s+1} = state_s + action_s
```

其中 `sigma_px=[1.2, 0.8, 0.5]`，换到 feature 尺度后是 `[0.3, 0.2, 0.125]`。所以越到后面的 refinement，随机扰动越小。

这个设计的含义是：  
它不会让点随便乱飞，而是沿着轮廓法向做平滑、低频的整体形变，比如局部向外鼓一点、向内收一点、某一段轻微偏移。这比较符合轮廓修正的直觉。

## 3. deterministic baseline 是什么

baseline 不是一个 value network，也不是移动平均 reward。

当前代码里的 deterministic baseline 是：

```text
同一个模型，同一个输入，不加随机扰动，连续跑三步 deterministic flow
```

也就是：

```text
baseline_action_s = flow_mean_s
baseline_final = 三步 flow_mean 后的 contour
baseline_score = score(baseline_final, GT)
```

随机 rollout 的质量用相对 baseline 的提升衡量：

```text
quality = rollout_score - burr_penalty * weight - baseline_score
```

所以 RL 学的不是绝对分数，而是：

```text
这个随机动作相对于当前 deterministic 输出有没有更好
```

这点很重要。因为它让训练变成“局部改进搜索”，而不是从零学一个 segmentation policy。

## 4. reward 怎么定义

当前 reward 是 contour 质量分数，主要由四部分组成：

```text
0.30 * boundary score
+ 0.10 * Dice
+ 0.25 * IoU
+ 0.35 * boundary distance score
- 0.06 * burr penalty
```

其中 burr penalty 是为了惩罚轮廓尖刺/毛刺，避免 RL 为了局部贴边界而学出不自然的轮廓。

当前 detail reward 没开：

```text
reward_detail_weight = 0.0
```

所以现在的 RL 主要优化全局区域和边界贴合，不额外强调角点、曲率局部细节。

## 5. K=8 如何形成探索

每个训练 batch 里，对同一个 deterministic baseline，采样 `K=8` 条随机轨迹：

```text
rollout_1, rollout_2, ..., rollout_8
```

每条 rollout 都有三步随机 action。  
然后分别计算：

```text
quality_k = reward_k - baseline_reward
```

日志里经常能看到这种现象：

```text
reward_mean < 0
quality_best_mean > 0
```

这说明大部分随机轨迹比 baseline 差，但 K=8 里面偶尔有一条或几条比 baseline 好。  
RL 更新的核心就是抓住这些“偶然更好的扰动方向”，让模型的 deterministic mean 往这些方向靠。

所以当前探索不是“随机直接带来最终答案”，而是：

```text
随机产生候选方向 -> 用 GT reward 选出相对更好的方向 -> 通过 PPO logprob 更新模型
```

## 6. logprob 在这里起什么作用

### 6.1 rollout 时的 old logprob

在 geom action 中，随机变量本质是：

```text
z ~ N(0, I)
```

采样时保存：

```text
old_log = log N(z; 0, I)
```

注意：这里的 action 是：

```text
action = mean_old + geom_delta(z)
```

`old_log` 记录的是这个随机 action 在旧策略下的概率。

### 6.2 更新时的 current logprob

参数更新时，模型参数已经参与计算新的：

```text
mean_cur = flow_model_current(state)
```

但 action 是之前采样出来的固定 action。  
为了计算这个 action 在当前模型下的概率，需要反推：

```text
z_cur = project(action - mean_cur)
lp_cur = log N(z_cur; 0, I)
```

如果当前模型的 mean 更接近一个好 action，那么：

```text
z_cur 变小
lp_cur 变大
ratio = exp(lp_cur - old_log) 变大
```

PPO 会鼓励这个变化。  
如果某个 action 很差，advantage 是负的，PPO 会让模型降低这个 action 的概率。

所以 logprob 的作用是：

```text
把“哪个随机动作好”转化成“模型均值应该往哪里移动”
```

它不是为了反传 reward 本身，因为 mask IoU / Dice / boundary score 都是不可微的。  
reward 只给方向标签，logprob 才提供可微的 policy gradient。

## 7. PPO 更新如何控制探索不会跑飞

当前用了几层约束。

第一层是 action 空间约束：

```text
只允许低频法向扰动
```

这让探索天然偏向平滑轮廓变化。

第二层是 PPO clip：

```text
ppo_clip = 0.05
ratio 被限制在约 [0.95, 1.05]
```

这意味着一次更新不能让新策略相对旧策略变化太大。

第三层是 KL 到 frozen reference flow：

```text
kl_beta = 0.01
ref_flow = 训练开始时复制并冻结的 gcn
```

它惩罚当前 flow mean 偏离初始 flow mean 太远。

第四层是 KL early stop：

```text
ppo_kl_target = 0.002
```

如果一次 inner PPO 更新里近似 KL 超过阈值，就提前停止当前 batch 的 PPO inner epoch。

第五层是 gate：

```text
gate = 1 if max(quality) > gate_margin else 0
```

如果这一组 K=8 没有任何 rollout 比 baseline 好，就把 advantage 关掉，不强行从一组全差样本里学习。

这些约束共同导致当前训练比较稳，但也导致提升慢。

## 8. 当前 V5 和 Flow-GRPO 式探索的区别

代码里保留了一个可选分支：

```text
action_policy = df_inner_step / flow_inner_step
```

如果使用这个分支，action 就不是三步几何扰动，而是 diffusion/flow 的内部 ODE/SDE transition：

```text
x_t -> x_{t+1}
```

它的 logprob 来自每个 diffusion inner step 的高斯/SDE transition probability。

区别如下：

| 项目 | 当前 V5 geom | df_inner_step / Flow-GRPO 风格 |
|---|---|---|
| 随机变量 | 低频几何 z | diffusion step noise |
| action 数量 | 3 个外层 action | outer_steps * ode_steps 个 inner action |
| action 形态 | contour 法向位移 | latent / displacement transition |
| logprob | project(action - mean) 得到 z 后算高斯概率 | 对每个 x_t -> x_{t+1} transition 算高斯/SDE logprob |
| 探索范围 | 局部、低频、几何可控 | 更接近扩散过程，空间更大 |
| 风险 | 探索能力有限 | 更难训、更容易噪声大 |

当前正式跑的是第一种，也就是 V5 geom。

## 9. 当前 V5 的探索是不是来自随机初始化噪声？

不是。

当前 V5 的 deterministic flow 从：

```text
latent = zeros_like(i_state)
```

开始算 mean。随机性是在 mean 之后额外加的低频几何扰动。

所以当前 V5 的随机性发生在：

```text
每个外层 contour refinement step 的 action 采样时
```

不是发生在：

```text
扩散模型初始 x0 / seed noise
```

这也是为什么我们后来讨论 V7 时会觉得“让 seed 提供随机性”更符合扩散模型流程。V7 的直觉是：每个 seed 对应一条完整 diffusion trajectory，RL 学会让模型把好 seed/好轨迹附近的行为吸收进去，而不是手工在 contour 法向加扰动。

## 10. 为什么这种探索能带来提升

它能提升的原因是：

1. 预训练模型已经有一个不错的 deterministic 解。
2. 在这个解附近，某些小幅几何扰动会让边界更贴 GT。
3. K=8 随机 rollout 能偶尔采到这些更好的扰动。
4. PPO 用 logprob ratio 把这些“好扰动”变成模型参数更新。
5. 更新后 deterministic mean 更接近这些好扰动，评估时即使不采样也可能更好。

用一句话说：

```text
RL 不是直接依赖随机采样做最终预测，而是在训练时用随机采样发现局部更优方向，再把这些方向蒸馏进 deterministic flow。
```

这和我们最近的 full-test 结果是一致的：

```text
pretrain epoch3500: 0.847218
RL V5 geom latest: 0.851252
```

说明这个局部搜索确实挖到了一点预训练模型没有直接输出的潜力。

## 11. 为什么它涨得慢

当前 V5 的探索慢，主要有四个原因。

第一，随机扰动空间比较小：

```text
8 个低频模式 + 法向扰动 + sigma 很小
```

这让它稳定，但探索不到复杂形变。

第二，大部分 rollout 比 baseline 差：

日志里常见：

```text
reward_mean 为负
quality_best_mean 偶尔为正
```

这说明训练信号主要来自少数正样本。

第三，PPO 和 KL 很保守：

```text
lr = 5e-8
ppo_clip = 0.05
kl_beta = 0.01
grad_clip = 0.3
```

这些设置能防止崩，但会让模型改得非常慢。

第四，当前 reward 是外部不可微指标：

IoU、Dice、boundary distance 都不是直接反传的 loss。  
它们只能通过 policy gradient 间接影响模型，所以样本效率天然比监督学习低。

## 12. 当前设计的优点和缺点

优点：

- 稳定，不容易把 contour 搞崩。
- action 有几何意义，容易解释。
- 三步 action 和 V5/V4.6c 的 iterative refinement 对齐。
- deterministic eval 能受益，因为更新的是 flow mean。
- 不依赖检测训练，能直接从 `epoch_3500.pt` 做后训练。

缺点：

- 探索不是扩散模型原生的 seed noise。
- 搜索空间被限制在低频法向扰动，表达力有限。
- 没有 learned std，探索强度是手工 sigma。
- K=8 采样成本高，但有效正样本比例不高。
- reward 对每个 batch 的 GT 依赖很强，泛化需要靠 KL 和小 LR 保守控制。

## 13. 如果继续改，我建议的方向

### 方向 A：保留 V5 geom，但提高有效探索

可以尝试：

```text
K: 8 -> 12 或 16
geom_modes: 8 -> 12
sigma_px: [1.2, 0.8, 0.5] -> [1.5, 1.0, 0.6]
reward_detail_weight: 0.0 -> 小值，比如 0.1
```

风险是毛刺和不稳定会增加，需要 burr penalty 或 KL 更强。

### 方向 B：切到 df_inner_step

这会让探索更接近 Flow-GRPO：

```text
随机性来自 diffusion/flow 内部 transition noise
```

优点是更符合扩散轨迹；缺点是 action 数量变成 `3 * ode_steps`，训练更慢、更难稳定。

### 方向 C：做 V7 seed-flow

这是更符合我们直觉的版本：

```text
每个 rollout 从不同初始化噪声 seed 开始
同一个 seed 决定完整 diffusion trajectory
reward 评价最终 contour
logprob 记录 seed/trajectory probability
PPO 或 GRPO 把好 seed 诱导出的轨迹吸收到 flow 参数里
```

这比当前 V5 更像真正的扩散模型探索。  
当前 V5 是“在 contour action 上探索”；V7 应该是“在 diffusion seed / trajectory 上探索”。

## 14. 最终理解

当前强化学习的探索可以这样理解：

```text
预训练模型给出一个 deterministic 解；
RL 在这个解附近采样 K=8 个低频几何扰动；
用 GT 指标判断哪些扰动比 deterministic baseline 好；
用 PPO logprob 把好扰动方向写回模型参数；
因此最终 deterministic 输出可能超过原始预训练模型。
```

所以它不是神秘地“凭空探索”，也不是直接搜索 seed。  
它本质上是一个带 PPO 约束的局部随机几何搜索和策略蒸馏过程。

