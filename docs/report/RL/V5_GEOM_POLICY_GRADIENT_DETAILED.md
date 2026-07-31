# V5 Geom RL 策略梯度详细解释

日期：2026-06-13

配套图：

- `report/RL/policy_gradient_grpo_explainer_imagegen.png`

核心代码：

- `grpo_train_v5_geom_action.py`
- `configs/1232_final_v7_8_antiregression_gpu0.yaml`

## 1. 这份文档讲的是哪个版本

这里讲的不是最早的 `grpo_train.py`，也不是 `grpo_train_v7_seedflow_grpo.py` 的 seed latent policy，而是当前更值得看的 **V5 geom action** 线：

```text
grpo_train_v5_geom_action.py
```

当前 `1232_final_v7_8_antiregression_gpu0.yaml` 虽然文件名里有 `v7_8`，但它的 RL 训练超参仍然是 `rl_v4_*` 前缀，并且对应的是 V5 geom 脚本中的几何动作策略：

```yaml
rl_v4_k: 8
rl_v4_outer_steps: 3
rl_v4_fractions: [0.3333, 0.5, 1.0]
rl_v4_action_policy: 'band_detail'
rl_v4_adaptive_explorer: true
```

所以本文中的 “V5 geom” 指的是：

```text
三步 contour refinement
+ 每个训练样本 K=8 条 rollout
+ 每步 action = Flow deterministic mean + Fourier 法向扰动
+ 用 reward - deterministic baseline 得到 advantage
+ 用 PPO clipped objective 产生策略梯度
```

## 2. 先给结论

V5 geom 的策略梯度不是从 Dice、IoU、边界距离这些 reward 函数直接反传出来的。

它真正的梯度路径是：

```text
PPO loss
  -> 当前策略下旧 action 的 log-prob
  -> Flow mean / Explorer 分布参数
  -> 模型参数
```

公式上是：

```text
ratio = exp(log_pi_current - log_pi_old)
loss_policy = - advantage * ratio
```

PPO clip 后代码写成：

```python
unclipped = -adv_ri * ratio
clipped = -adv_ri * torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
policy_loss = torch.maximum(unclipped, clipped).mean()
```

反向传播时：

```text
advantage detach
old_log detach
old action detach
old state detach
reward detach
```

只有：

```text
log_pi_current(action_old | state_old)
```

对当前参数可导。

因此，策略梯度本质是：

```text
grad ~= - A * grad log pi_theta(a_old | s_old)
```

如果 `A > 0`，提高这个旧 action 的概率。  
如果 `A < 0`，降低这个旧 action 的概率。

这就是 V5 geom 强化学习真正发生的事情。

## 3. V5 geom 里的 state 是什么

普通 RL 里，state 可能是游戏画面或环境状态。这里的 state 更具体：

```text
state = 当前 contour / polygon 点集
```

代码中 rollout 开始时：

```python
current = output['i_it_py'].detach()
```

`current` 的形状可以理解为：

```text
[num_contours, num_points, 2]
```

每个 outer step 都会把当前轮廓 `current` 转成 canonical polygon：

```python
c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
```

然后用当前轮廓、canonical 轮廓、CNN feature、polygon index 一起喂给 Flow/GCN。

所以 V5 geom 的策略不是直接在整张图上选动作，而是在已经初始化好的 contour 上做局部 refinement。

## 4. deterministic mean 是什么

每一步 action 都先有一个确定性均值：

```python
mean = _outer_action_mean(
    gcn,
    output['cnn_feature'],
    current,
    c_cur,
    output['py_ind'],
    frac,
    ode_steps,
)
```

`_outer_action_mean()` 里面做的是：

```python
latent = torch.zeros_like(i_state)
return _flow_disp_from_latent(..., latent, steps) * frac
```

也就是说：

```text
mean = Flow 在 zero latent 下预测出的 deterministic displacement
```

这个 `mean` 是当前模型“不探索时”会采取的轮廓修正动作。

如果没有 RL 随机扰动，三步 refinement 就是：

```text
state_1 = state_0 + mean_0
state_2 = state_1 + mean_1
state_3 = state_2 + mean_2
```

这条轨迹就是后面用来做 baseline 的 deterministic baseline。

## 5. V5 geom 的 action 分布

### 5.1 geom action

最基础的 geom action 是：

```text
action = mean + low_delta
```

其中：

```text
z_low ~ N(0, I)
low_delta = FourierLowFreq(z_low) * contour_normal * sigma
```

代码对应：

```python
z = torch.randn((current.size(0), geom_modes), ...)
low_delta = _geom_delta_from_z(current, z, sigma)
action = mean + low_delta
```

`_geom_delta_from_z()` 做了三件事：

1. 构造低频 Fourier basis。
2. 把低维随机向量 `z` 展开成每个 contour point 上的标量扰动。
3. 沿 contour 法向移动点。

也就是说它不是对每个点独立加噪声，而是生成平滑的轮廓变形。

### 5.2 band_detail action

当前推荐配置用的是：

```yaml
rl_v4_action_policy: 'band_detail'
```

所以 action 实际是：

```text
action = mean + low_delta + detail_delta
```

其中：

```text
low_delta    = 低频 Fourier 法向扰动
detail_delta = k_min 到 k_max 的 detail band 法向扰动
```

配置里：

```yaml
rl_v4_geom_lowfreq_modes: 8
rl_v4_geom_sigma_px: [0.0, 0.0, 0.45]
rl_v4_detail_k_min: 9
rl_v4_detail_k_max: 28
rl_v4_detail_sigma_px: [0.0, 0.0, 0.32]
rl_v4_detail_gate: [0.0, 0.0, 0.35]
```

这说明当前版本主要在第三步 refinement 上加探索：

```text
step 0: sigma = 0
step 1: sigma = 0
step 2: sigma > 0
```

这样设计很保守。前两步让模型先稳定靠近目标，第三步再尝试细节修正，避免早期大幅扰动把轮廓带偏。

### 5.3 adaptive explorer

当前配置还开启了：

```yaml
rl_v4_adaptive_explorer: true
```

这会创建 `FourierExplorer`：

```python
explorer = FourierExplorer(
    low_modes=geom_modes,
    detail_modes=detail_modes,
    hidden_dim=explorer_hidden_dim,
    mu_max=explorer_mu_max,
    logstd_min=explorer_logstd_min,
    logstd_max=explorer_logstd_max,
)
```

Explorer 根据当前 polygon 的几何特征输出：

```text
low_mu, low_logstd, detail_mu, detail_logstd
```

于是采样不再是固定标准正态，而是：

```text
z_low    = low_mu    + exp(low_logstd)    * eps_low
z_detail = detail_mu + exp(detail_logstd) * eps_detail
```

这让策略可以学习：

```text
什么形状的 contour 应该往哪个 Fourier 方向探索，
探索方差应该大还是小。
```

## 6. 采样阶段：old_log 怎么产生

rollout 采样函数是 `_sample_rollout()`。它被 `@torch.no_grad()` 包住，所以采样阶段不反传，只收集轨迹。

每一步会保存：

```python
traj['states'].append(current.detach())
traj['c_states'].append(c_cur.detach())
traj['actions'].append(action.detach())
traj['old_logs'].append(old_log.detach())
traj['fractions'].append(float(frac))
traj['sigmas'].append(sigma)
traj['detail_sigmas'].append(float(d_sigma))
```

这里最关键的是 `old_log`。

### 6.1 没有 explorer 时

如果没有 adaptive explorer，`z` 来自标准正态：

```text
z ~ N(0, I)
```

log-prob 是：

```python
def _z_logprob(z):
    lp = -0.5 * z.pow(2) - 0.5 * log(2*pi)
    return lp.mean(...)
```

所以：

```text
old_log = log N(z_old; 0, I)
```

### 6.2 有 explorer 时

有 explorer 时：

```python
low_mu, low_logstd, detail_mu, detail_logstd = explorer(current, frac)
z = low_mu + torch.exp(low_logstd) * eps
old_log = _normal_z_logprob(z, low_mu, low_logstd)
```

`_normal_z_logprob()` 是标准 Gaussian log-prob：

```python
lp = -0.5 * (((z - mu) ** 2) / var + 2.0 * logstd + log(2*pi))
```

detail band 也会有自己的 log-prob：

```python
detail_log = detail_logprob_weight * _normal_z_logprob(
    z_detail, detail_mu, detail_logstd
)
old_log = old_log + detail_log
```

注意：这里的 `old_log` 是采样时旧策略的概率。后面 PPO 更新时，它只作为常数出现。

## 7. K 条 rollout 如何变成 advantage

每个训练 step，对同一个 batch 采样：

```yaml
rl_v4_k: 8
```

也就是 K=8 条随机轨迹。

每条轨迹都从同一个初始 contour 出发，但因为采样的 Fourier 噪声不同，最终 contour 不同：

```text
rollout_1 -> final_poly_1
rollout_2 -> final_poly_2
...
rollout_8 -> final_poly_8
```

每条 final contour 都会算 reward：

```python
score, score_comps = _score_with_components(final_poly, i_gt, output['image_hw'])
```

当前配置下 reward 包括：

```yaml
rl_v4_reward_w_region: 0.25
rl_v4_reward_w_dice: 0.10
rl_v4_reward_w_iou: 0.22
rl_v4_reward_w_dist: 0.33
rl_v4_reward_burr_weight: 0.08
rl_v4_reward_regression_weight: 0.5
rl_v4_reward_detail_weight: 0.25
```

这几个项的含义：

- `region / dice / iou / dist`：全局区域和边界贴合。
- `burr`：惩罚毛刺和尖刺。
- `regression`：防止本来不错的初始轮廓被改坏。
- `detail`：鼓励角点、曲率、局部边界细节。

但是 reward 计算出来后会 detach：

```python
score = score.detach()
final_scores_reward.append(... .detach())
```

所以 reward 不提供直接梯度。它只决定动作好坏。

## 8. deterministic baseline 的作用

V5 geom 没有训练 value network，也没有 critic。它用 deterministic 三步结果当 baseline：

```python
det_ret = _deterministic_three_step(output, gcn)
baseline_score, baseline_comps = _score_with_components(
    output['i_it_py'] + det_ret['disp'], i_gt, output['image_hw']
)
```

deterministic baseline 的意思是：

```text
同一个模型、同一个输入、同样三步 refinement，
但不加随机 Fourier 扰动，只走 mean action。
```

然后每条 rollout 的相对质量是：

```python
quality = final_scores_reward_t - baseline_score_adj.unsqueeze(0)
```

也就是：

```text
quality = 随机 rollout reward - 不随机 baseline reward
```

这一步非常重要。它让训练目标从：

```text
绝对 reward 高不高
```

变成：

```text
这次随机探索有没有比当前 deterministic 输出更好
```

因此 V5 geom 更像是一种“局部随机搜索 + 把好方向蒸馏回模型”的后训练方法。

## 9. advantage 的计算

有了 `quality` 后，代码做标准化和裁剪：

```python
adv = quality / quality.std(dim=0, unbiased=False, keepdim=True).clamp_min(0.1)
adv = adv.clamp(-adv_clip_max, adv_clip_max) * gate
```

配置：

```yaml
rl_v4_adv_clip_max: 2.0
rl_v4_gate_margin: 0.0
```

含义：

```text
adv > 0：这条 rollout 比 deterministic baseline 好
adv < 0：这条 rollout 比 deterministic baseline 差
adv = 0：不更新，或者被 gate 过滤
```

`gate` 的逻辑是：

```python
gate = (quality.max(dim=0, keepdim=True).values > gate_margin).float()
```

也就是说，如果某个 contour 的 K 条 rollout 没有任何一条超过 baseline，就可以不给这个 contour 更新，避免全是坏探索时乱推模型。

## 10. 更新阶段：重新计算当前 log-prob

这是策略梯度产生的核心。

采样时保存的是：

```text
old_log = log pi_old(action_old | state_old)
```

更新时不会重新采样动作，而是拿旧 action、旧 state，用当前参数重新算：

```text
lp_cur = log pi_theta(action_old | state_old)
```

代码里：

```python
mean_cur = _outer_action_mean(
    gcn, output['cnn_feature'], state, c_state, output['py_ind'], frac, ode_steps
)
```

然后根据 action policy 计算 `lp_cur`。

### 10.1 不带 explorer 的 geom log-prob

```python
lp_cur = _geom_action_logprob(action, mean_cur, state, sigma, geom_modes)
```

这里会做：

```python
z = _project_geom_z(state, action.detach() - mean_cur, sigma, geom_modes)
lp_cur = _z_logprob(z)
```

注意：

```text
action.detach() 是旧动作，不动；
mean_cur 是当前模型输出，可导。
```

如果 `mean_cur` 更接近这个旧 action，则投影出来的 `z` 更接近高概率区域，`lp_cur` 更高。  
如果 `mean_cur` 远离这个旧 action，则 `lp_cur` 更低。

### 10.2 带 explorer 的 band_detail log-prob

当前配置是 adaptive explorer + band_detail。更新时会：

```python
residual = action.detach() - mean_cur
low_mu, low_logstd, detail_mu, detail_logstd = explorer(state, frac)
z_low = _project_geom_z(state, residual, sigma, geom_modes)
lp_low = _normal_z_logprob(z_low, low_mu, low_logstd)
```

detail 部分类似：

```python
z_detail = _project_band_detail_z(...)
lp_detail = detail_logprob_weight * _normal_z_logprob(
    z_detail, detail_mu, detail_logstd
)
lp_cur = lp_low + lp_detail
```

这里有两条可导路径：

```text
lp_cur -> mean_cur -> Flow / GCN 参数
lp_cur -> low_mu/logstd/detail_mu/logstd -> Explorer 参数
```

这就是为什么策略梯度既能让 Flow mean 向好 action 靠近，也能让 explorer 学会更合理的探索分布。

## 11. PPO ratio 和 clipped loss

更新阶段有：

```python
ratio = torch.exp(lp_cur - old_log)
```

含义：

```text
ratio = 当前策略给旧 action 的概率 / 采样时旧策略给旧 action 的概率
```

如果 `ratio > 1`，说明当前策略比旧策略更喜欢这个 action。  
如果 `ratio < 1`，说明当前策略比旧策略更不喜欢这个 action。

然后构造 PPO loss：

```python
unclipped = -adv_ri * ratio
clipped = -adv_ri * torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
policy_loss = torch.maximum(unclipped, clipped).mean() / max(total_actions, 1)
```

配置：

```yaml
rl_v4_ppo_clip: 0.05
rl_v4_ppo_inner_epochs: 2
rl_v4_ppo_kl_target: 0.002
```

这说明更新非常保守：

```text
每次只允许策略概率相对旧策略小幅变化。
```

医学轮廓 refinement 对稳定性要求高，所以这个 clip 很小。

## 12. 策略梯度到底怎么从 loss 出来

先看没有 clip 的近似形式：

```text
loss = - A * exp(log_pi_current - old_log)
```

因为：

```text
old_log 是常数
A 是常数
```

所以：

```text
grad loss
= - A * exp(log_pi_current - old_log) * grad log_pi_current
= - A * ratio * grad log_pi_current
```

如果更新很小，`ratio` 接近 1，就接近经典 REINFORCE：

```text
grad loss ~= - A * grad log_pi_theta(action | state)
```

这就是策略梯度。

PPO clip 只是把 `ratio` 限制在一个安全范围里，避免一次更新太大。它没有改变“梯度来自 log-prob”的本质。

## 13. 为什么 reward detach 仍然能训练

这点最容易误解。

监督学习通常是：

```text
prediction -> differentiable loss -> parameters
```

V5 geom 的 RL 不是这条路。它是：

```text
sample action -> get non-differentiable reward -> weight log-prob gradient
```

数学上：

```text
J(theta) = E_{a ~ pi_theta}[R(a)]
grad J(theta) = E[R(a) * grad log pi_theta(a)]
```

因此 reward 不需要可导。它只需要告诉我们 action 是好是坏。

在代码中：

```python
score = score.detach()
final_scores_reward.append(... .detach())
adv_ri = adv[ri].detach()
```

这是正确的。否则如果试图让梯度穿过 IoU/Dice/边界距离，不但很多地方不可导，而且会把问题变成另一个不稳定的 surrogate loss。

V5 geom 选择的是更标准的 policy gradient 路径：

```text
reward 作为权重
log_prob 作为可导项
```

## 14. 正 advantage 和负 advantage 分别做什么

假设某个旧 action 的 advantage 是正的：

```text
A > 0
```

loss 是：

```text
loss = - A * ratio
```

为了减小 loss，优化器会增大 `ratio`，也就是增大 `lp_cur`：

```text
提高 pi_theta(action_old | state_old)
```

直观上：

```text
这条随机轮廓比 baseline 好，以后更应该采这种动作。
```

如果 advantage 是负的：

```text
A < 0
```

loss 等价于惩罚高 ratio。为了减小 loss，优化器会降低 `lp_cur`：

```text
降低 pi_theta(action_old | state_old)
```

直观上：

```text
这条随机轮廓比 baseline 差，以后少采这种动作。
```

这就是策略学习。

## 15. Flow 参数为什么会被更新

很多人会以为：

```text
action 是采样出来的，detach 了，那 Flow 怎么更新？
```

关键在于更新时重新算的是：

```python
residual = action.detach() - mean_cur
z = project(residual)
lp_cur = log_prob(z)
```

`action` 确实 detach，但 `mean_cur` 不 detach。`mean_cur` 来自当前 Flow：

```python
mean_cur = _outer_action_mean(gcn, ...)
```

所以：

```text
lp_cur 对 mean_cur 可导
mean_cur 对 Flow 参数可导
```

如果某个 action 是好 action，优化会让 `mean_cur` 往 `action` 靠近。  
如果某个 action 是坏 action，优化会让 `mean_cur` 远离 `action`。

这就是 V5 geom 的一个核心效果：

```text
把随机探索中发现的好几何修正方向，蒸馏进 deterministic Flow mean。
```

## 16. Explorer 参数为什么会被更新

开启 adaptive explorer 后：

```python
low_mu, low_logstd, detail_mu, detail_logstd = explorer(state, frac)
lp_cur = Normal(mu, std).log_prob(z_old)
```

所以：

```text
lp_cur 对 mu/logstd 可导
mu/logstd 对 explorer 参数可导
```

正 advantage 会让 explorer 提高好 `z` 的概率：

```text
mu 往好 z 靠近，或者 std 调整到更容易覆盖好 z
```

负 advantage 会让 explorer 降低坏 `z` 的概率。

这比固定标准正态更灵活，因为不同形状的 contour 可能需要不同探索方向。

## 17. KL 和 prefix distill 不是策略梯度主来源

V5 geom 里还有两个稳定项。

### 17.1 KL regularization

代码里：

```python
loss = policy_loss + kl_beta * kl_loss
```

配置：

```yaml
rl_v4_kl_beta: 0.012
```

它的作用是约束当前模型不要偏离冻结的 reference flow 太远。

这不是策略梯度本身，而是防止 RL 更新把模型带偏。

### 17.2 prefix distill

配置：

```yaml
rl_v4_prefix_distill_weight: 0.06
rl_v4_prefix_distill_steps: [0, 1]
```

它让前两个 step 的 mean 继续贴近 reference flow：

```python
prefix_loss = smooth_l1_loss(mean_cur, mean_ref)
```

因为当前探索主要放在第三步，所以前两步做 prefix distill 可以减少早期 contour 漂移。

它也不是策略梯度主来源，而是稳定训练的辅助约束。

## 18. 最小伪代码

下面是 V5 geom 去掉工程细节后的策略梯度流程：

```python
for batch in loader:
    output = model(batch)
    state0 = output["i_it_py"].detach()

    # deterministic baseline
    baseline_final = deterministic_three_step(model, state0)
    baseline_reward = reward_fn(baseline_final).detach()

    rollouts = []
    for k in range(K):
        state = state0
        traj = []

        for step in range(3):
            mean_old = flow_mean(model, state)

            mu, logstd = explorer(state)          # optional
            z = mu + exp(logstd) * randn()
            delta = fourier_normal_delta(state, z)

            action = mean_old + delta
            old_log = normal_log_prob(z, mu, logstd).detach()

            traj.append({
                "state": state.detach(),
                "action": action.detach(),
                "old_log": old_log,
            })

            state = (state + action).detach()

        final_reward = reward_fn(state).detach()
        rollouts.append((traj, final_reward))

    advantage = normalize(final_reward - baseline_reward).detach()

    for traj, A in rollouts:
        for item in traj:
            state = item["state"]
            action = item["action"]
            old_log = item["old_log"]

            mean_cur = flow_mean(model, state)
            z_cur = project_action_to_fourier_z(action - mean_cur)
            log_cur = normal_log_prob(z_cur, explorer(state))

            ratio = exp(log_cur - old_log)
            loss = -A * clip_ratio(ratio)
            loss.backward()

    optimizer.step()
```

这个伪代码里最关键的一行是：

```python
log_cur = log_prob_current_policy(old_action | old_state)
```

只要这行里的当前策略参数可导，策略梯度就产生了。

## 19. 用一句话概括 V5 geom 的 RL

V5 geom 的 RL 可以理解为：

```text
围绕当前 Flow 的 deterministic contour 修正，采样一批平滑的 Fourier 法向几何扰动；
用 Dice/IoU/边界/毛刺/细节 reward 判断哪些扰动比 deterministic baseline 好；
再通过 PPO 的 log-prob 梯度，让模型以后更倾向于产生好扰动、远离坏扰动。
```

所以它既不是“reward 直接反传”，也不是“单纯随机增强”。  
它是一个真正的 policy-gradient 后训练方法，只是 action 空间被工程上限制在稳定、可解释的几何扰动里。

## 20. 代码索引

| 主题 | 文件位置 | 作用 |
| --- | --- | --- |
| V5 训练入口 | `grpo_train_v5_geom_action.py:441` | 设置 RL/Flow-GRPO 训练流程 |
| outer mean | `grpo_train_v5_geom_action.py:436` | zero latent 下的 deterministic Flow 位移 |
| 低频 basis | `grpo_train_v5_geom_action.py:262` | 构造低频 Fourier 探索空间 |
| detail band basis | `grpo_train_v5_geom_action.py:276` | 构造中高频 detail 探索空间 |
| Explorer | `grpo_train_v5_geom_action.py:307` | 输出 `mu/logstd`，控制探索分布 |
| Gaussian log-prob | `grpo_train_v5_geom_action.py:344` | 计算 adaptive explorer 的 log-prob |
| action log-prob | `grpo_train_v5_geom_action.py:379` | 把旧 action 投影回 z 并算概率 |
| rollout 采样 | `grpo_train_v5_geom_action.py:966` | 采样 K 条轨迹并保存 old_log |
| baseline | `grpo_train_v5_geom_action.py:1437` | deterministic 三步 baseline |
| advantage | `grpo_train_v5_geom_action.py:1509` | `reward - baseline` 并标准化 |
| PPO loss | `grpo_train_v5_geom_action.py:1607` | ratio、clip、policy_loss |
| 参数更新 | `grpo_train_v5_geom_action.py:1677` | clip grad 后 `optimizer.step()` |
| 当前配置 | `configs/1232_final_v7_8_antiregression_gpu0.yaml:132` | K、三步、band_detail、reward、PPO 超参 |

