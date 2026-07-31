# RL V4 三段外层策略报告（V4.6c / three-iter policy）

日期：2026-05-26

## 先讲结论

你这次要的 **强化学习 V4**，和之前 V3.4 / GRPO-V2 那套确实**不是一回事**。

**V3.4 / V2 那套**，是把 **内层 ODE 的每一个小 step** 当成随机动作来做 PPO；
**V4 这套**，是把 **外层 iterative refinement 的 3 次大步** 当成策略轨迹。

也就是说，V4 的 policy 不是“20 个 ODE 小步的 step-action policy”，而是：

1. 当前 contour 状态 `s_1`
2. 做第 1 次 outer refinement 动作 `a_1`
3. 更新 contour，得到 `s_2`
4. 做第 2 次 outer refinement 动作 `a_2`
5. 更新 contour，得到 `s_3`
6. 做第 3 次 outer refinement 动作 `a_3`
7. 最终 contour 拿去算 reward

所以 **V4 RL 的核心差异** 是：

- **随机性从“内层 ODE 每一步噪声”移到了“每个 outer step 的 latent 采样”**；
- **策略梯度也不再绑在 ODE 小步上，而是绑在 3 个 outer action 上**；
- **log-prob 也不是内层 SDE 的精确 step logprob，而是 outer action 的高斯 surrogate logprob**。

这个差异非常大，不能按我上一份 V3.4 报告的思路去理解。

---

## 1. 这份 V4 RL 具体是哪条线

当前对应代码和配置是：

- 训练脚本：`grpo_train_v4_three_iter.py`
- RL 配置：`configs/btcv_v4_6c_rl_v4_three_iter_gpu5.yaml`
- warm start 基座：`configs/btcv_diffusion_dit_v4_6c_mlp_shared_moe_newdist_long_gpu5.yaml`

配置里最关键的几行：

- `use_dit_v4_1: true`
- `v4_1_final_head_type: 'moe'`
- `v4_1_use_detail_context: true`
- `use_grpo: true`
- `rl_v4_outer_steps: 3`
- `rl_v4_fractions: [0.3333, 0.5, 1.0]`
- `rl_v4_ode_steps: 20`

代码锚点：

- `configs/btcv_v4_6c_rl_v4_three_iter_gpu5.yaml:64-160`
- `grpo_train_v4_three_iter.py:293-329`

先把名字讲清楚：

### 1.1 它虽然走 `use_dit_v4_1: true`，但本质上是 V4.6c 结构

这里“V4.6c”不是一个单独 Python 类名，而是：

- 用 `DiTFlowMatchingV4_1` 这条代码路径；
- 打开了 P3 feature、detail context；
- 关闭 per-point delta；
- 把 final head 改成了 **MoE final head**；
- 再叠加 richer state sampling 分布。

所以你可以把它理解成：

> **V4.6c = V4.1 trunk + V4.6 MoE final head + V4.9 richer state distribution**

代码锚点：

- `lib/networks/diffusion/flow_matching_evolution.py:174-269`
- `lib/networks/diffusion/dit_denoiser_v4_1.py:11-173`
- `lib/networks/diffusion/dit_denoiser_v4.py:138-260`
- `configs/btcv_diffusion_dit_v4_6c_mlp_shared_moe_newdist_long_gpu5.yaml:64-123`

---

## 2. 这个 V4 基座模型本身在做什么

先不谈 RL，先看 V4.6c 本体。

### 2.1 主干仍然是 Flow Matching

这一点和 V3.4 一样：

- 仍然是 `use_flow_matching: true`
- 仍然学速度场 `v_theta(x_t, t)`
- 仍然通过内层 ODE rollout 从 latent 位移生成 contour displacement

所以 **V4 不是放弃 FM**，而是在 FM backbone 上换了更强的结构和更适合 outer policy 的 RL 包装。

### 2.2 和 V3.4 主干相比，V4.6c 基座最关键的结构变化

#### 变化 1：P3 feature 融合

V4.6c 打开了：

- `v4_1_use_p3_features: true`

表示除了 P2，还会把更高层 feature 上采样后融合进 contour refinement feature。

这意味着 V4 不是只靠局部边缘纹理，它的 contour 决策里还吃到更多语义上下文。

#### 变化 2：detail context

V4.6c 打开：

- `v4_1_use_detail_context: true`

`FlowMatchingEvolution.sample_detail_features(...)` 会沿 contour normal / tangent 周围再采 feature，形成 detail feature，然后送入 denoiser。

这会让模型不仅知道“当前点在哪里”，还知道“边界法线两边长什么样”。

代码锚点：

- `lib/networks/diffusion/flow_matching_evolution.py:642-713`

#### 变化 3：final head 不是普通 linear head，而是 MoE head

V4.6c 配置里：

- `v4_1_final_head_type: 'moe'`
- `v4_6_moe_num_experts: 8`
- `v4_6_moe_top_k: 2`
- `v4_6_moe_use_shared_expert: true`
- `v4_6_moe_expert_type: 'mlp'`

这意味着最终位移头不再是统一一套线性投影，而是：

- 有 shared base predictor；
- 再由 router 决定每个点/局部状态该走哪些 expert；
- 用 routed expert 在 shared base 上做偏移修正。

也就是说，V4.6c 的 final displacement head 是 **按状态自适应选择专家** 的，这一点和 V3.4 很不一样。

代码锚点：

- `lib/networks/diffusion/dit_denoiser_v4_1.py:44-69, 163-173`
- `lib/networks/diffusion/dit_denoiser_v4.py:138-260`

#### 变化 4：rich-state / newdist 训练分布

V4.6c base config 打开了：

- `v4_9_use_rich_state_sampling: true`
- `v4_9_use_rich_infer_schedule: true`

它在监督训练时，不只喂一种“从很远的 init 走向 GT”的状态，而是混合：

- continuous 中间态
- discrete fraction 态
- small residual 态
- hard far 态
- near zero 态

这会让模型对“离终点很近”与“离终点很远”的轮廓状态都更稳。

这点对 RL V4 非常关键，因为 RL V4 直接把 outer refinement 的几次中间态当 policy state。

---

## 3. V4 RL 为什么和之前不一样

这是最关键的一节。

### 3.1 之前那套（V2 / V3.4 GRPO）的 action 是什么

之前在 `grpo_train_v2.py` 那类设计里，action 更像是：

> **内层 ODE rollout 中某一个 step 的下一步 latent sample**

即：

- state = `x_t`、`t`、当前 contour feature
- action = `x_{t+dt}`

因此之前的随机性是在 **inner ODE step** 上加的，策略梯度也是对这些小步做的。

### 3.2 V4 这里的 action 是什么

V4 脚本 `grpo_train_v4_three_iter.py` 明确写了：

> **three outer inference refinements as the policy trajectory**

也就是把 3 次外层 refinement 直接当成 policy trajectory。

具体地：

- `outer_steps = 3`
- `fractions = [0.3333, 0.5, 1.0]`

每一个 outer step 都是一个动作：

\[
a_i = \text{当前 contour 状态下，本轮 refinement 应该加上的位移}
\]

然后：

\[
contour_{i+1} = contour_i + a_i
\]

代码锚点：

- `grpo_train_v4_three_iter.py:293-300`
- `grpo_train_v4_three_iter.py:493-504`
- `grpo_train_v4_three_iter.py:507-533`

所以 V4 的动作不是 ODE 小步，而是 **outer refinement displacement**。

这就是它和之前最根本的区别。

---

## 4. V4 里的 deterministic policy 是怎么定义的

在 V4 里，每个 outer step 的 deterministic action mean 不是手写的，而是由 FM backbone 计算出来的。

### 4.1 `outer_action_mean(...)`

代码：

```python
latent = torch.zeros_like(i_state)
return _flow_disp_from_latent(flow, ..., latent, steps) * frac
```

代码锚点：

- `grpo_train_v4_three_iter.py:260-263`

也就是说：

1. 对当前 contour state，先把 latent 初始化成 **零 latent**；
2. 用 inner flow ODE rollout 跑 `ode_steps=20` 步；
3. 得到本轮 raw displacement；
4. 再乘上这一轮 outer fraction（比如 1/3、1/2、1）。

于是 deterministic policy mean 是：

> **“在当前 outer state 下，从零 latent 出发，inner ODE 认为该走的标准 refinement displacement”**

### 4.2 三段 deterministic rollout

V4 baseline 不是一步结束，而是：

```python
for frac in fractions:
    action = outer_action_mean(..., frac)
    current = current + action
```

代码锚点：

- `grpo_train_v4_three_iter.py:493-504`

所以 deterministic baseline 其实是 3 个 outer action 串起来的结果，而不是单步结果。

---

## 5. V4 的 stochastic rollout 是怎么做的

### 5.1 随机性不是 inner-step Gaussian，而是 outer-step latent sample

V4 这里的 stochastic rollout 写法是：

```python
mean = _outer_action_mean(...)
latent = torch.randn_like(current) * noise_scale
raw_disp = _flow_disp_from_latent(..., latent, ode_steps)
action = raw_disp * frac
old_log = _action_logprob(action, mean, action_std)
```

代码锚点：

- `grpo_train_v4_three_iter.py:517-529`

这里一定要看明白：

### V4 不是：

\[
a = mean + \sigma \epsilon
\]

### V4 实际是：

1. 先采样一个 outer-step latent `z ~ N(0, noise_scale^2 I)`；
2. 再通过 inner flow ODE 这个非线性映射：
   \[
   raw\_disp = f_\theta(state, z)
   \]
3. 最后定义：
   \[
   a = frac \cdot raw\_disp
   \]

也就是说，**随机性来自 latent -> displacement 的隐式诱导分布**，不是像 V2 那样直接在每个 ODE 小步上显式加高斯噪声。

这是 V4 最大的建模变化。

---

## 6. 那 V4 的 log-prob 又是怎么来的

这也是 V4 最不一样、最值得单独讲的地方。

### 6.1 V4 用的是 outer-action surrogate Gaussian

虽然 action 真实来源是：

\[
a = f_\theta(s, z)
\]

但 PPO 更新时，它没有去算这个 induced distribution 的精确密度，而是**定义了一个 surrogate policy**：

\[
\pi_\theta(a|s) = \mathcal N(a; \mu_\theta(s), \sigma^2 I)
\]

其中：

- `mu_theta(s)` = `outer_action_mean(...)`
- `sigma` = 固定的 `action_std`

代码锚点：

- `_action_logprob(...)`：`grpo_train_v4_three_iter.py:222-227`
- 调用点：`grpo_train_v4_three_iter.py:523, 636-637`

### 6.2 这是什么意思

V4 的 policy gradient 不是在说：

> “这个 action 真的是高斯采出来的”

而是在说：

> “我们把 outer refinement displacement 视作一个围绕 deterministic mean 的连续动作，并用固定方差高斯来近似它的 policy density。”

所以严格说：

- **V2/V3.4 inner-step policy** 更接近“精确写出来的 step Gaussian policy”；
- **V4 outer-step policy** 是“对 outer action 的 surrogate Gaussian policy”。

这也是为什么你说“V4 和之前不一样吧”——是的，**不只是实现细节不一样，连策略建模层级都不一样**。

---

## 7. V4 里 ODE、随机性、策略梯度三者的关系

这一节把你最关心的逻辑重新整理成 V4 版。

### 7.1 ODE 在 V4 里仍然存在，但它退居到“action generator”

inner FM ODE 仍然在做：

\[
\frac{dx}{dt} = v_\theta(x,t)
\]

并通过 `step_with_logprob(..., action_std=0.0)` 这种确定性方式滚 20 步，把 latent 变成 displacement。

代码锚点：

- `grpo_train_v4_three_iter.py:230-257`
- 其中 inner rollout 调的是 `flow.step_with_logprob(..., action_std=0.0)`

也就是说，在 V4 里，**内层 ODE 本身不是 policy**，而是 outer policy 的动力学生成器。

### 7.2 随机性在 V4 里进入的位置

随机性在每个 outer step 只发生一次：

- 采一个 latent `z`
- 用 ODE 把它映射成 displacement action

所以 V4 更像：

\[
a_i = g_\theta(s_i, z_i)
\quad,
\quad z_i \sim \mathcal N(0, \sigma_z^2 I)
\]

然后环境转移是：

\[
s_{i+1} = s_i + a_i
\]

### 7.3 策略梯度为什么还能成立

因为 PPO 更新时，代码显式定义了 surrogate logprob：

\[
\log \pi_\theta(a_i|s_i)
= \log \mathcal N(a_i; \mu_\theta(s_i), \sigma_a^2 I)
\]

于是就可以做：

\[
ratio = \exp(\log \pi_\theta - \log \pi_{old})
\]

代码锚点：

- `grpo_train_v4_three_iter.py:633-640`

然后标准 clipped PPO：

\[
L = -\min(ratio \cdot A, clip(ratio) \cdot A)
\]

因此，V4 的策略梯度成立依赖的是：

1. 先把 outer refinement displacement 视作连续动作；
2. 给它一个 surrogate Gaussian density；
3. 用这个 density 写 `log_prob`；
4. 再按 PPO 更新。

所以 V4 不是“ODE 自动变 SDE”，而是：

> **inner ODE 负责生成 outer action，outer action 再被一个高斯策略 surrogate 包起来用于 PPO。**

---

## 8. V4 里的 reward 是怎么定义的

V4 的 reward 也和之前不一样。

### 8.1 V4 不再混 absolute reward 和 delta reward

在 `grpo_train_v4_three_iter.py` 里，先算 deterministic baseline：

```python
baseline_score = quality_score(det_three_step_output, gt)
```

再对每个 stochastic rollout 算最终质量：

```python
final_score = quality_score(stochastic_output, gt)
quality = final_score - baseline_score
```

代码锚点：

- `grpo_train_v4_three_iter.py:596-612`

所以 V4 的 reward 核心是：

> **相对 deterministic three-step baseline 的提升量**

不是：

- 对 init 的提升；
- 也不是绝对 score 与 delta 的加权混合；
- 更不是 best-of-k 蒸馏那种间接目标。

### 8.2 quality score 的组成

`compute_region_score(...)` 里组合了：

- boundary score (`w_boundary`)
- Dice (`w_dice`)
- IoU (`w_iou`)
- boundary distance (`w_dist`)

当前 V4 config 权重是：

- region: `0.30`
- dice: `0.10`
- iou: `0.25`
- dist: `0.35`

代码锚点：

- `grpo_train_v4_three_iter.py:322-328`
- `grpo_train_v4_three_iter.py:477-491`
- `lib/train/rewards/region_reward.py:84-122`

因此 V4 奖励的本质就是：

> **最终三段 refinement 后的 contour，是否比 deterministic 基线更贴近 GT。**

---

## 9. V4 的 advantage、gate、KL 是怎么做的

### 9.1 advantage

先对 K 个 rollout 得到：

\[
quality_{k,b} = score_{k,b} - baseline_b
\]

再做标准差归一化：

\[
adv = quality / std(quality)
\]

并 clip 到 `[-adv_clip_max, adv_clip_max]`。

代码锚点：

- `grpo_train_v4_three_iter.py:608-612`

### 9.2 gate

只要某个 batch item 上，最好的 rollout 仍然没超过 margin，就直接把该样本 advantage 压掉：

```python
gate = (quality.max(dim=0) > gate_margin).float()
adv = adv * gate
```

代码锚点：

- `grpo_train_v4_three_iter.py:609-612`

这一步的作用是：

> 防止模型在“全都不如 deterministic baseline”的一组 rollout 中，去强化那个“最不坏”的动作。

### 9.3 KL 到 frozen ref flow

V4 里还会把当前 outer mean action 和 frozen reference flow 的 outer mean action 拉近：

```python
mean_ref = _outer_action_mean(ref_flow, ...)
kl_loss = ((mean_cur - mean_ref)^2 / (2 * var)).mean()
```

代码锚点：

- `grpo_train_v4_three_iter.py:641-648`

这个 KL 不是对整个 latent trajectory 的精确 KL，而是对 **outer action mean** 的二次惩罚近似。

它的作用很直接：

- 不让 RL 把已经很强的 V4.6c 基座带崩；
- 限制 PPO 更新只做小幅偏移。

---

## 10. V4 的 PPO 实际在更新什么

V4 的每条 trajectory 只有 3 个 action：

- `a_1`
- `a_2`
- `a_3`

于是 PPO 实际在更新的是：

> **在每一个 outer refinement state 下，deterministic outer action mean 应该如何偏移，才能让整条三步 contour refinement 轨迹最终比 baseline 更好。**

梯度链路是：

1. `policy_loss`
2. `lp_cur = log N(action ; mean_cur, sigma^2)`
3. `mean_cur = outer_action_mean(...)`
4. `outer_action_mean` 内部调用 `_flow_disp_from_latent(..., latent=0)`
5. `_flow_disp_from_latent` 内部跑 20 步 deterministic inner flow ODE
6. inner flow ODE 依赖 V4.6c 的 FM denoiser / MoE final head / detail context 参数

所以最后被更新的仍然是 V4 FM backbone，但更新信号来自 outer action 的 PPO，而不是 inner step 的 PPO。

---

## 11. V4 和之前那套最本质的不同

这里给你一个最直接的对照表。

| 项 | V3.4 / V2 GRPO | V4 RL three-iter |
|---|---|---|
| policy 层级 | inner ODE step | outer refinement step |
| 轨迹长度 | 通常是若干 ODE 小步窗口 | 固定 3 个 outer action |
| 随机性来源 | 每个 ODE step 的显式噪声 | 每个 outer step 的 latent sample |
| log-prob | inner-step Gaussian 的精确写法 | outer-action Gaussian surrogate |
| baseline | 常是 init / det / group 混合 | deterministic three-step baseline |
| reward 对象 | rollout 最终 contour 或 mixed reward | 最终 contour 相对 three-step deterministic 的提升 |
| 核心目标 | 优化 inner rollout policy | 优化 3 段 coarse-to-fine outer policy |

所以如果只用一句话概括：

> **V3.4/V2 是“在 ODE 里做 RL”，V4 是“把 iterative refinement 本身当 RL”。**

---

## 12. 你问的“ODE 转 SDE”在 V4 里该怎么理解

这一点要特别纠正。

### 对 V4 来说，不应该再简单说“ODE 加噪声就变 SDE”

那种说法更适合之前 V2/V3.4 inner-step policy。

在 V4 里，更准确的理解是：

### 12.1 inner 层

inner FM ODE 仍然是 deterministic solver：

\[
latent \xrightarrow{20\ step\ ODE} raw\_disp
\]

### 12.2 outer 层

outer policy 通过随机 latent 诱导出随机 action：

\[
z_i \sim N(0, \sigma_z^2 I), \quad a_i = frac_i \cdot f_\theta(s_i, z_i)
\]

### 12.3 PPO 层

再把这个 induced action 用 Gaussian surrogate 包起来：

\[
\pi_\theta(a_i|s_i) \approx N(\mu_\theta(s_i), \sigma_a^2 I)
\]

所以 V4 更像：

> **“deterministic inner ODE + stochastic latent-induced outer action + Gaussian PPO surrogate”**

而不是“inner ODE 直接离散成 SDE”。

这就是 V4 和你前面那条线真正不一样的地方。

---

## 13. 为什么 V4 要这样设计

因为它在优化目标上更贴近模型真实推理方式。

### 13.1 真实推理本来就是 outer refinement

V4 的 contour refinement 本来就是 coarse-to-fine 多段式的。

所以与其把 20 个 inner ODE 小步都当 action，不如直接把：

- 第 1 次大修
- 第 2 次细修
- 第 3 次终修

当成真实控制变量。

### 13.2 降低 credit assignment 难度

3 步 outer trajectory 比 20 步 inner trajectory 更短、更容易分配 credit。

### 13.3 更贴近“策略应该改什么”

inner ODE 小步太底层，噪声很难解释；
outer refinement displacement 则更接近“本轮轮廓该怎么改”。

所以 V4 从 RL 建模角度是更高层、更结构化的 policy 设计。

---

## 14. 这套 V4 RL 的风险点

虽然它更结构化，但也有明确代价。

### 14.1 log-prob 是 surrogate，不是精确诱导密度

真实 action 来自：

\[
a = f_\theta(s,z)
\]

但 PPO 用的是：

\[
\log N(a; \mu_\theta(s), \sigma^2I)
\]

所以它并不是对真实 induced distribution 的精确 logprob，而是一个近似。

这意味着：

- 梯度方向有启发意义；
- 但它不等于严格的 exact policy gradient。

### 14.2 stochastic action 和 surrogate mean 可能不完全匹配

如果 latent-induced action 分布很偏、很非高斯，Gaussian surrogate 会有偏差。

### 14.3 只优化 outer 3 步，inner 20 步只作为生成器

这样更稳定，但也可能错过 inner dynamics 里的更细粒度可优化结构。

---

## 15. 最后给你的最简洁版本

### 15.1 V4 强化学习到底在学什么

V4 学的是：

> **在三段 outer contour refinement 中，每一段应该给当前 contour 加什么 displacement，才能让最终轮廓比 deterministic baseline 更好。**

### 15.2 它和之前最大的区别是什么

之前是在 **ODE 小步上做 policy**；
现在是在 **outer refinement 大步上做 policy**。

### 15.3 ODE 在 V4 里是什么角色

不是直接 policy，而是 **outer action 的生成器**。

### 15.4 随机性从哪里来

来自每个 outer step 的 latent sample，而不是每个 inner ODE 小步的显式噪声。

### 15.5 策略梯度怎么成立

通过 outer action 的 Gaussian surrogate logprob + PPO ratio + quality reward。

---

## 16. 关键代码索引

### V4 RL 主脚本

- `grpo_train_v4_three_iter.py`

重点位置：

- action surrogate logprob：`222-227`
- inner ODE from latent：`230-257`
- deterministic outer mean：`260-263`
- deterministic three-step baseline：`493-504`
- stochastic rollout：`507-533`
- advantage / gate：`608-612`
- PPO ratio / KL：`633-648`

### V4 RL 配置

- `configs/btcv_v4_6c_rl_v4_three_iter_gpu5.yaml`

### V4.6c 基座配置

- `configs/btcv_diffusion_dit_v4_6c_mlp_shared_moe_newdist_long_gpu5.yaml`

### V4 架构实现

- `lib/networks/diffusion/dit_denoiser_v4.py`
- `lib/networks/diffusion/dit_denoiser_v4_1.py`
- `lib/networks/diffusion/flow_matching_evolution.py`

### reward

- `lib/train/rewards/region_reward.py`
