# V4d latent-noise policy 计划：把随机初始化噪声作为真正策略动作

日期：2026-05-26

## 1. 问题背景

当前 RL V4 的实现是近似方案：

- 用随机 latent 初始化提供探索；
- 但训练时没有直接对 latent noise 建模概率；
- 而是把最终 outer displacement 近似成 Gaussian action：

```text
action = outer displacement
logprob = Gaussian(action | deterministic_mean, fixed_std)
```

这个方案能跑，但它不是你现在提出的最自然设计。

你的新想法更准确：

```text
随机初始化噪声 z 本身提供随机性；
三步 outer refinement 还是策略轨迹；
每一步的动作应该可以定义为该步的初始噪声 z_t。
```

所以新的核心问题是：

> 能不能把 `z_t` 当作 action，并直接用 `p(z_t | state_t)` 作为动作概率函数？

答案是：**可以，而且这比当前 displacement Gaussian 更干净。**

但有一个关键前提：  
如果 `z_t ~ N(0, I)` 是完全固定分布，且不依赖可学习参数，那么 `log p(z_t)` 对模型没有梯度。  
这种情况下它只能提供探索，不能形成真正 policy gradient。

因此正确设计应该是：

```text
把随机初始化噪声 z_t 的分布做成可学习 policy：

z_t ~ π_phi(z | state_t)

然后 Flow/FM 只作为 deterministic transition：

disp_t = Flow_theta(state_t, z_t)
state_{t+1} = state_t + fraction_t * disp_t
```

这样，动作概率就是 `log π_phi(z_t | state_t)`，不再需要用 displacement Gaussian 近似。

---

## 2. 推荐方案：V4d latent-noise policy

### 2.1 策略轨迹

仍然保持你的核心想法：

```text
trajectory = 三次 outer refinement
```

每个 outer step 是一个策略决策：

```text
s_t = 当前轮廓 + canonical 轮廓 + 图像特征 + t/fraction
a_t = 初始噪声 z_t
transition = 用 z_t 经过 FM rollout 得到 displacement
```

也就是：

```text
s_0 --z_0--> s_1 --z_1--> s_2 --z_2--> s_3
```

reward 仍然只看最终轮廓 `s_3` 和 GT 的质量。

### 2.2 动作定义

旧 V4：

```text
action = final displacement
```

新 V4d：

```text
action = initial latent noise z
```

每一步的动作 shape 和轮廓一致：

```text
z_t: [num_contours, num_points, 2]
```

这和当前 `torch.randn_like(current)` 完全对齐。

---

## 3. 动作概率函数设计

### 3.1 policy distribution

对每个 outer step，定义：

```text
π_phi(z_t | s_t) = Normal(mu_phi(s_t), sigma_phi(s_t))
```

最简单版本：

```text
mu_phi(s_t): [N, P, 2]
logstd_phi(s_t): [N, P, 2] 或全局 scalar
```

采样：

```text
eps ~ N(0, I)
z_t = mu_phi(s_t) + exp(logstd_phi(s_t)) * eps
```

动作 logprob：

```text
logprob_t = Normal(mu_phi(s_t), sigma_phi(s_t)).log_prob(z_t)
logprob_t = mean over points and xy dims
```

这时 PPO ratio 是严格对应 latent action 的：

```text
ratio = exp(logprob_new(z_t | s_t) - logprob_old(z_t | s_t))
```

不再需要：

```text
Gaussian(displacement | deterministic_mean, fixed_std)
```

### 3.2 初始化方式

为了不改变当前推理分布，policy head 应该初始化为：

```text
mu_phi(s) = 0
sigma_phi(s) = rl_v4_noise_scale
```

这样训练开始时：

```text
z_t ~ N(0, noise_scale^2 I)
```

和现在的随机初始化噪声一致。

这点很重要：  
V4d 一开始不改变采样行为，只是让这个采样行为有了可学习 logprob。

---

## 4. 为什么固定 N(0, I) 不够

如果保持：

```text
z_t ~ N(0, I)
```

并且这个分布不依赖任何参数，那么：

```text
log p(z_t)
```

是常数分布的 logprob。

它可以用于记录概率，但：

```text
∇ log p(z_t) = 0
```

所以它不能通过 policy gradient 更新模型。

因此必须至少让下面某个东西可学习：

1. `mu_phi(s)`
2. `sigma_phi(s)`
3. 或更复杂的 latent distribution

否则“噪声提供随机性”只成立于探索层面，不成立于策略学习层面。

---

## 5. 需要新增的模块

### 5.1 LatentNoisePolicyHead

建议新增一个轻量 head：

```text
LatentNoisePolicyHead(
    state features -> mu_z, logstd_z
)
```

输入可以复用：

- `cnn_feature`
- 当前 contour state
- canonical contour
- `fraction`
- 可选：flow 的 sampled point feature

最小实现有两个选择。

#### 方案 A：全局 per-step scalar policy

最简单：

```text
mu_z = learned scalar/bias broadcast 到 [N, P, 2]
logstd_z = learned scalar
```

优点：

- 实现简单；
- 稳定；
- 可快速验证“latent logprob 是否有效”。

缺点：

- 表达力弱；
- 只能整体调节噪声偏置和噪声强度。

#### 方案 B：per-point policy head

更合理：

```text
point_feat = sample image feature at contour points + point embedding + fraction embedding
mu_z, logstd_z = MLP(point_feat)
```

优点：

- 能根据局部图像和轮廓状态决定噪声方向；
- 更接近真正策略。

缺点：

- 多一个 head；
- 需要更严格 KL / entropy 控制，避免过早塌缩。

推荐路线：

```text
先做 A 版 smoke；
如果 logprob / ratio / reward 信号正常，再做 B 版。
```

---

## 6. 训练流程改造

### 6.1 rollout collection

旧逻辑：

```python
latent = torch.randn_like(current) * noise_scale
raw_disp = flow(current, latent)
action = raw_disp * fraction
old_log = gaussian_logprob(action, mean_disp, action_std)
```

新逻辑：

```python
mu_z, logstd_z = noise_policy(state, c_state, cnn_feature, fraction)
eps = torch.randn_like(mu_z)
z = mu_z + exp(logstd_z) * eps
old_log = gaussian_logprob(z, mu_z, exp(logstd_z))

raw_disp = flow(current, z)
action_disp = raw_disp * fraction
next_state = current + action_disp
```

trajectory 中保存：

```text
states
c_states
latents z_t
old_latent_logprob
fractions
disp/action_disp
```

### 6.2 PPO update

旧逻辑：

```python
mean_cur = outer_action_mean(...)
lp_cur = logprob(action_disp | mean_cur)
ratio = exp(lp_cur - old_log)
```

新逻辑：

```python
mu_cur, logstd_cur = noise_policy(state, c_state, cnn_feature, fraction)
lp_cur = logprob(z_t | mu_cur, exp(logstd_cur))
ratio = exp(lp_cur - old_log_z)
```

PPO loss 不变：

```text
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

只是 action 概率从 displacement 空间换成 latent noise 空间。

---

## 7. 是否还训练 Flow/FM 主体

这是 V4d 的关键设计选择。

### 7.1 第一阶段：只训练 noise policy head

推荐先这样做：

```text
freeze flow denoiser
train latent noise policy head only
```

原因：

- 概率定义干净；
- PPO 梯度明确；
- 不会破坏已经训练好的 FM；
- 可以验证“选择什么噪声”是否足以提升结果。

这相当于学习一个：

```text
state-conditioned noise sampler
```

而不是直接改 FM 生成器。

### 7.2 第二阶段：再考虑联合微调 Flow

如果第一阶段有效，再考虑：

```text
train noise policy head + small LR fine-tune flow
```

但这时需要额外约束：

- reference KL
- displacement drift penalty
- 保持 deterministic baseline 不退化
- full validation 检查

因为如果同时改 Flow，`z -> disp` 的 transition 也变了，训练更容易不稳定。

---

## 8. 和“诱导 action density”的区别

理论上也可以把 final displacement 的真实密度写成：

```text
p_theta(disp | s) = p(z) * |det dz / ddisp|
```

这叫 change-of-variables / induced density。

但在当前项目里不推荐作为第一版：

1. flow rollout 不是专门实现为可逆 normalizing flow；
2. `clamp_pred_disp` 会破坏严格可逆性；
3. 高维 Jacobian determinant 代价很高；
4. 数值上很难稳定；
5. 实现复杂度远高于收益。

所以更实用的方案是：

```text
把 z 直接定义成 action；
直接学习 p(z | state)；
不要试图反推 displacement 的精确密度。
```

---

## 9. 新实验命名建议

建议不要改当前 V4 长训，另开隔离实验：

```text
grpo_train_v4d_latent_noise_policy.py
configs/btcv_select_v4_6c_rl_v4d_latent_noise_policy_gpu*.yaml
scripts/run_v4d_latent_noise_policy_gpu*.sh
```

输出目录：

```text
data/outputs/btcv_select_v4_6c_rl_v4d_latent_noise_policy_gpu*
```

可视化目录：

```text
visual/rl_v4d_latent_noise_policy_*
```

---

## 10. 实施步骤

### Step 1：保留当前 V4，不干扰

当前 `grpo_train_v4_three_iter.py` 已经在长训，不建议直接改坏。  
新方案另开文件。

### Step 2：复制 V4 训练入口

从：

```text
grpo_train_v4_three_iter.py
```

复制为：

```text
grpo_train_v4d_latent_noise_policy.py
```

### Step 3：新增 latent noise policy head

先做最小 A 版：

```text
global/per-step mu_z + logstd_z
```

确认：

- `logprob_z` 非空；
- `ratio` 不恒等于 1；
- `entropy` 正常；
- `outer_log_count_mean = 3.0`。

### Step 4：替换 rollout logprob

把：

```text
displacement Gaussian logprob
```

替换为：

```text
latent z Gaussian logprob
```

保留三步 outer trajectory。

### Step 5：先 freeze Flow

第一版只训练 policy head：

```text
freeze_yolo = true
freeze_flow = true
train_noise_policy = true
```

这样能最干净地验证“学会选择噪声”有没有价值。

### Step 6：smoke

最小验证：

```text
RL_V4D_STEPS=1
K=2
ODE_STEPS=2
```

必须看到：

- 三步都有 latent logprob；
- loss 有梯度；
- policy head 参数发生变化；
- checkpoint 正常保存。

### Step 7：短训

建议：

```text
50 step
100 step
200 step
```

每个 checkpoint 做 full validation。

### Step 8：如果 A 版有效，再做 B 版

B 版改为 per-point policy head：

```text
sampled_feat + contour_embed + fraction_embed -> mu_z/logstd_z
```

---

## 11. 判断标准

V4d 是否有效，不看小 batch reward，主要看 full validation：

```text
baseline: V4.6c 原始 checkpoint
V4: 当前 displacement Gaussian 近似版
V4d: latent-noise policy 版
```

关键指标：

- `mean_iou_sample_avg`
- `mean_iou_contour_avg`
- `mean_dice_sample_avg`
- `mean_mboundf_sample_avg`
- `median_iou_sample_avg`
- worst-case / low-percentile samples

如果 V4d 在 100-200 step 后仍然只提升 `1e-4`，说明仅学习 noise sampler 可能不够。  
如果能稳定超过当前 V4 的 `step100` 增益，并且 boundary 指标同步提升，就说明你的这个“随机初始化噪声作为策略动作”的想法是更正确的。

---

## 12. 预期风险

### 风险 1：policy 只学会减小噪声

如果 logstd 很快变小，策略会退化成 deterministic。  
需要加：

```text
entropy bonus
logstd clamp
min_std
```

### 风险 2：高维 latent logprob 太尖锐

`[N, P, 2]` 维度很高，直接 sum logprob 会导致 ratio 爆。  
建议和当前一样先用 mean logprob：

```text
mean over points and xy dims
```

### 风险 3：只训练 noise policy 不够强

如果 Flow 本身对噪声不敏感，noise policy head 学不到东西。  
解决：

1. 先诊断不同 z 的输出差异；
2. 再考虑联合微调 Flow；
3. 或改成低维 structured latent，让噪声影响更集中。

---

## 13. 最终建议

推荐下一版叫：

```text
RL V4d latent-noise policy
```

核心原则：

```text
动作 = 三个 outer step 的初始噪声 z_t
动作概率 = 可学习 Gaussian π_phi(z_t | state_t)
Flow/FM = deterministic transition from z_t to contour displacement
Reward = final 3-step contour quality relative to deterministic baseline
```

这比当前 displacement Gaussian 近似更符合你的新想法。

