# DiffusionSnake 强化学习策略梯度解释

日期：2026-06-13

配套图：`report/RL/policy_gradient_grpo_explainer_imagegen.png`

## 1. 结论先说

当前更值得看的强化学习实现是：

- 训练脚本：`grpo_train_v5_geom_action.py`
- 推荐配置参考：`configs/1232_final_v7_8_antiregression_gpu0.yaml`
- 主要方法：三步 contour refinement + K 条随机 rollout + GRPO/PPO 风格更新
- 当前 action 形式：deterministic Flow 位移均值 + Fourier 法向随机扰动

它的策略梯度不是从 IoU、Dice、边界距离这些 reward 函数直接反传出来的。

真正产生策略梯度的是这一项：

```text
policy_loss = - advantage * ratio
ratio = exp(log_pi_current - log_pi_old)
```

更准确地说，梯度来自：

```text
grad_theta loss ~= - A * grad_theta log pi_theta(action | state)
```

其中：

- `A` 是 rollout 相对 deterministic baseline 的优势。
- `action` 是之前采样出来并 detach 保存的动作。
- `log pi_theta(action | state)` 是用当前参数重新计算的动作概率。
- reward 本身是 detach 的，只作为权重，不走梯度。

所以这套 RL 的本质是：

```text
先随机试几个轮廓修正方向，看哪些比 deterministic baseline 好，
再提高模型以后产生这些好动作的概率，降低坏动作的概率。
```

## 2. 一次训练迭代的总流程

一次训练 step 可以拆成六段：

```text
1. 从当前模型得到初始轮廓 state
2. 对同一个样本采样 K 条 rollout
3. 每条 rollout 记录动作 action 和 old_log
4. 算每条 rollout 的最终 reward
5. 用 reward - deterministic baseline 得到 advantage
6. 用 PPO clipped objective 反传更新策略参数
```

图里对应的是：

```text
当前轮廓 s
  -> 策略采样 a ~ pi_theta(a|s)，记录 old_log
  -> K 次 rollout
  -> Reward - Baseline，得到 A
  -> 重新计算 log pi_theta(a|s)
  -> PPO Loss
  -> 梯度回到 Flow / Explorer 参数
```

## 3. state、action、policy 分别是什么

### 3.1 state

这里的 state 是当前 contour：

```python
current = output['i_it_py'].detach()
```

它不是普通 RL 里的图像状态，而是已经被检测/初始化后的 polygon 点集。每个 outer step 都把当前 polygon 当作状态。

### 3.2 deterministic mean

每一步先由 Flow 模型给一个确定性位移均值：

```text
mean = Flow(cnn_feature, current contour, canonical contour, step fraction)
```

代码位置在 `_sample_rollout()` 中：

```python
mean = _outer_action_mean(
    gcn, output['cnn_feature'], current, c_cur, output['py_ind'], frac, ode_steps
)
```

这个 `mean` 可以理解为当前模型如果不随机探索，会采取的默认修正动作。

### 3.3 action

V5 最新配置里主要使用 `band_detail` action policy：

```yaml
rl_v4_action_policy: 'band_detail'
rl_v4_geom_lowfreq_modes: 8
rl_v4_geom_sigma_px: [0.0, 0.0, 0.45]
rl_v4_detail_k_min: 9
rl_v4_detail_k_max: 28
rl_v4_detail_sigma_px: [0.0, 0.0, 0.32]
rl_v4_detail_gate: [0.0, 0.0, 0.35]
rl_v4_adaptive_explorer: true
```

也就是说，主要在第三个 refinement step 加探索。探索由两部分组成：

```text
low_delta    = 低频 Fourier 法向扰动
detail_delta = 中高频 detail band 法向扰动
action       = mean + low_delta + detail_delta
```

代码里对应：

```python
z = torch.randn((current.size(0), geom_modes), ...)
low_delta = _geom_delta_from_z(current, z, sigma)

z_detail = torch.randn((current.size(0), detail_modes), ...)
detail_delta = _band_detail_delta_from_z(...)

action = mean + low_delta + detail_delta
current = (current + action).detach()
```

这里的随机性不是随便给每个点加噪声，而是先采样低维 `z`，再投影到 Fourier basis，再沿轮廓法向移动。这样探索更平滑，不容易把 contour 点打乱。

## 4. old_log 是怎么来的

策略梯度必须知道“当时采样这个动作的概率是多少”。所以采样时要保存 `old_log`。

如果没有 adaptive explorer，噪声是标准正态：

```python
def _z_logprob(z):
    lp = -0.5 * z.pow(2) - 0.5 * log(2*pi)
    return lp.mean(...)
```

如果开了 adaptive explorer，则 explorer 会根据 polygon 特征输出 `mu` 和 `logstd`：

```python
low_mu, low_logstd, detail_mu, detail_logstd = explorer(current, frac)
z = low_mu + exp(low_logstd) * eps
old_log = Normal(low_mu, exp(low_logstd)).log_prob(z)
```

代码对应：

```python
old_log = (
    _normal_z_logprob(z, low_mu, low_logstd)
    if low_mu is not None else _z_logprob(z)
)
```

对于 detail band，也会额外加上 detail 的 log-prob：

```python
detail_log = detail_logprob_weight * _normal_z_logprob(
    z_detail, detail_mu, detail_logstd
)
old_log = old_log + detail_log
```

最后保存：

```python
traj['states'].append(current.detach())
traj['actions'].append(action.detach())
traj['old_logs'].append(old_log.detach())
```

注意这里全部 detach。采样阶段不反传，采样阶段只收集训练数据。

## 5. reward 怎么变成 advantage

每个训练样本会采样 K 条 rollout：

```yaml
rl_v4_k: 8
```

每条 rollout 最终得到一个 final contour：

```python
final_poly = output['i_it_py'] + ret['disp']
score, score_comps = _score_with_components(final_poly, i_gt, image_hw)
```

reward 由多个轮廓质量指标组合而成。当前配置里包含：

```yaml
rl_v4_reward_w_region: 0.25
rl_v4_reward_w_dice: 0.10
rl_v4_reward_w_iou: 0.22
rl_v4_reward_w_dist: 0.33
rl_v4_reward_burr_weight: 0.08
rl_v4_reward_regression_weight: 0.5
rl_v4_reward_detail_weight: 0.25
```

直观上：

- region / Dice / IoU：区域重合。
- dist：边界距离。
- burr penalty：惩罚毛刺。
- regression penalty：避免高质量初始轮廓被 RL 改坏。
- detail reward：强调局部角点、曲率、局部边界质量。

但是这些 reward 只是数值评价，不直接提供梯度。

为了降低方差，代码不是直接用绝对 reward，而是减掉 deterministic baseline：

```python
det_ret = _deterministic_three_step(output, gcn)
baseline_score = score(output['i_it_py'] + det_ret['disp'], gt)

quality = final_scores_reward_t - baseline_score_adj.unsqueeze(0)
```

这里的 `quality` 就是“这条随机 rollout 比不随机的 deterministic 三步好多少”。

然后归一化成 advantage：

```python
adv = quality / quality.std(dim=0, unbiased=False, keepdim=True).clamp_min(0.1)
adv = adv.clamp(-adv_clip_max, adv_clip_max) * gate
```

所以：

```text
adv > 0：这条随机轨迹比 baseline 好，应该提高概率
adv < 0：这条随机轨迹比 baseline 差，应该降低概率
adv = 0：不更新或被 gate 关掉
```

这里没有 value network，也没有 critic。它是 GRPO 风格：用同组 K 条 rollout 的相对表现来构造优势。

## 6. PPO loss 如何产生策略梯度

采样阶段记录的是旧策略概率：

```text
old_log = log pi_old(action | state)
```

更新阶段，拿同一个旧 action，在当前参数下重新算：

```python
lp_cur = log pi_current(action | state)
ratio = exp(lp_cur - old_log)
```

代码里：

```python
ratio = torch.exp(lp_cur - old_log)
unclipped = -adv_ri * ratio
clipped = -adv_ri * torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
policy_loss = torch.maximum(unclipped, clipped).mean() / total_actions
loss.backward()
```

这就是 PPO clipped objective 的最小化写法。

为什么这里会有梯度？

因为 `old_log`、`action`、`state` 都是 detach 的旧数据，但 `lp_cur` 是用当前参数算出来的：

```text
lp_cur = log pi_theta(action_old | state_old)
```

所以反传时：

```text
action_old 不动
reward / advantage 不动
old_log 不动
只有 log pi_theta 对 theta 求导
```

最终得到：

```text
grad_theta policy_loss
  ~= - advantage * grad_theta log pi_theta(action_old | state_old)
```

这就是策略梯度。

## 7. 为什么 reward detach 还能训练

这是最容易误解的点。

监督学习里常见形式是：

```text
loss = f(prediction, target)
```

然后梯度来自：

```text
prediction -> loss -> parameters
```

但这里不是这种路径。这里 reward 不需要可导，因为用的是 score-function estimator：

```text
maximize E_a~pi_theta [ R(a) ]
grad = E[ R(a) * grad log pi_theta(a) ]
```

也就是说：

```text
reward 负责告诉你这个 action 好不好；
log_prob 负责告诉你如何改变参数，让这个 action 以后更常见或更少见。
```

所以代码里 reward 都 detach 是合理的：

```python
score = score.detach()
final_scores_reward.append(... .detach())
adv_ri = adv[ri].detach()
```

如果 `adv > 0`，优化器会增加当前 action 的 log-prob。

如果 `adv < 0`，优化器会降低当前 action 的 log-prob。

梯度不需要穿过 IoU/Dice，也不需要穿过 rasterization 或边界距离计算。

## 8. 梯度到底更新哪些参数

当前 V5 有几类可能被更新的参数：

### 8.1 Flow / GCN 参数

`mean_cur = _outer_action_mean(...)` 依赖当前 Flow 模型参数。

当重新计算某个旧 action 的 log-prob 时：

```python
lp_cur = _geom_action_logprob(action, mean_cur, state, sigma, geom_modes)
```

`action` 是旧动作，`state` 是旧状态，`sigma` 是固定超参；但 `mean_cur` 来自当前模型，所以 `lp_cur` 对 Flow 参数可导。

直观解释：

```text
如果某个旧 action 的 advantage 是正的，
模型会把自己的 deterministic mean 往这个好 action 靠近；
如果 advantage 是负的，
模型会让 mean 远离这个坏 action。
```

这就是“把好随机探索蒸馏回 deterministic Flow”的含义。

### 8.2 Adaptive Explorer 参数

如果配置里：

```yaml
rl_v4_adaptive_explorer: true
```

则 explorer 会输出 `mu/logstd`，控制探索分布：

```python
low_mu, low_logstd, detail_mu, detail_logstd = explorer(state, frac)
lp_cur = Normal(mu, std).log_prob(z_old)
```

这时策略梯度也会更新 explorer，让它在不同形状、不同 refinement step 上学会更合适的探索均值和方差。

### 8.3 KL 和 prefix distill

除了 policy loss，代码里还可能有：

- `kl_beta`：限制当前模型不要偏离参考 Flow 太多。
- `prefix_distill_weight`：前几步继续贴近 reference，避免早期 contour 被 RL 改坏。

这些是稳定项，不是策略梯度的主来源。

## 9. PPO clip 在这里的作用

如果不用 PPO clip，更新目标近似是：

```text
loss = - A * exp(log_pi_current - log_pi_old)
```

问题是 ratio 可能变得很大，导致一次更新把策略推太远。

所以代码用：

```python
torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip)
```

当前推荐配置：

```yaml
rl_v4_ppo_clip: 0.05
rl_v4_ppo_kl_target: 0.002
rl_v4_ppo_inner_epochs: 2
```

这非常保守。含义是：

```text
每次只允许当前策略相对采样策略小幅变化。
```

对医学轮廓任务这是合理的，因为探索动作稍微大一点就可能把本来不错的边界改坏。

## 10. 和 V7 SeedFlow-GRPO 的区别

目录里还有 `grpo_train_v7_seedflow_grpo.py`。它的思想相同，也是：

```text
采样 latent -> 记录 old_log -> reward-baseline -> advantage -> PPO loss
```

区别在 action 空间：

- V7 是对 Flow 初始化 latent 采样。
- V5 是对 contour action 加 Fourier 几何扰动。

V7 的策略是：

```text
latent ~ Normal(mean_z, std_z)
raw_disp = Flow(latent)
action = raw_disp * frac
```

V5 的策略是：

```text
action = Flow_mean + Fourier normal delta
```

当前更推荐解释和分析 V5，是因为它更接近现在这批 `v7_7/v7_8` 配置实际在做的事情：动作空间更受控，并且包含 anti-regression、detail reward、prefix distill 等稳定设计。

## 11. 一句话对应代码

策略梯度链路可以按下面这张表查代码：

| 环节 | 代码位置 | 含义 |
| --- | --- | --- |
| 定义 log-prob | `grpo_train_v5_geom_action.py:251` | 标准正态噪声 log-prob |
| adaptive explorer | `grpo_train_v5_geom_action.py:307` | 根据 polygon 输出探索分布 |
| action log-prob | `grpo_train_v5_geom_action.py:379` | 把 action 投影回 z 后算概率 |
| rollout 采样 | `grpo_train_v5_geom_action.py:966` | 采样 K 条轨迹并记录 old_log |
| deterministic baseline | `grpo_train_v5_geom_action.py:1437` | 不加随机扰动的三步结果 |
| reward/advantage | `grpo_train_v5_geom_action.py:1475` | 计算 rollout 分数和相对优势 |
| PPO 更新 | `grpo_train_v5_geom_action.py:1522` | 重新算 log-prob，构造 policy loss |
| optimizer step | `grpo_train_v5_geom_action.py:1684` | 裁剪梯度并更新参数 |

## 12. 最小伪代码

下面是去掉工程细节后的最小版本：

```python
for batch in loader:
    output = model(batch)
    state0 = output["i_it_py"].detach()

    baseline = deterministic_three_step(model, state0)
    baseline_reward = reward_fn(baseline).detach()

    rollouts = []
    for k in range(K):
        state = state0
        traj = []
        for step in range(3):
            mean = model.flow_mean(state)
            z = sample_normal()
            action = mean + fourier_delta(state, z)
            old_log = log_prob(z).detach()

            traj.append((state.detach(), action.detach(), old_log))
            state = (state + action).detach()

        final_reward = reward_fn(state).detach()
        rollouts.append((traj, final_reward))

    quality = final_rewards - baseline_reward
    adv = normalize_and_clip(quality).detach()

    for traj, A in rollouts:
        for state, action, old_log in traj:
            mean_cur = model.flow_mean(state)
            log_cur = log_prob_of_old_action(action, mean_cur, state)
            ratio = exp(log_cur - old_log)

            loss = -A * clipped_ratio(ratio)
            loss.backward()

    optimizer.step()
```

这段伪代码里最关键的是：

```python
log_cur = log_prob_of_old_action(action, mean_cur, state)
```

只要 `mean_cur` 依赖模型参数，`loss.backward()` 就能把策略梯度传回模型。

## 13. 读这套代码时的判断标准

判断一个 RL 实现是不是“真的有策略梯度”，看三件事：

1. 采样动作时是否保存了 `old_log`。
2. 更新时是否用当前参数重新计算 `log_prob(action_old | state_old)`。
3. loss 是否包含 `advantage * log_prob` 或 PPO ratio。

当前 V5 满足这三点。

所以它不是“reward 直接反传”，也不是“只是随机增强”。它确实有策略梯度，只是策略梯度走的是 log-prob 路径，而且 action 空间被设计成 Fourier 法向几何扰动，更新非常保守。

