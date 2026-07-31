# RL V4d 随机初始化噪声策略失败分析报告

日期：2026-05-26  
实验目录：`/home/medteam/Zhrch/DiffusionSnake-12-30`  
主数据集：`/home/medteam/Zhrch/Datasets/BTCV/btcv_select`

## 1. 结论

这轮实验验证了一个核心问题：**把 FM 推理时的随机初始化噪声 `z` 作为 RL action，在理论上比“位移高斯近似”更干净，但在当前 V4.6c frozen-flow 设置下，验证集效果稳定下降。**

当前最重要结论：

1. 固定随机噪声 `z ~ N(0,I)` 本身不能形成可学习策略；必须引入可学习的 `pi_phi(z | state)`，否则 `log p(z)` 对模型参数没有策略梯度。
2. 我实现了这个可学习噪声策略，并测试了 global、per-point、structured、structured sum-logprob 四种版本。
3. 四个 V4d 版本都在 `btcv_select` full val 上稳定低于 corrected V4.6c baseline，约为：
   - `mean_iou_sample_avg: -0.0010`
   - `mean_mboundf_sample_avg: -0.0014`
4. V4d-D 已经把动作概率改成更正确的联合 Gaussian logprob，PPO ratio/gradient 信号明显增强，但 full validation 仍然负增益。
5. 所以目前不建议继续按 **frozen-flow + learned latent noise initializer + deterministic mean eval** 这个方向长训。
6. 目前更有价值的仍是原来的 V4 displacement-action 三步轨迹版本，它在 corrected `btcv_select` 上至少有很小正增益。

## 2. 我具体做了什么

### 2.1 V4d 的核心定义

原始 V4 的近似做法是：

- 三个 outer refinement step 作为策略轨迹；
- 每一步的 action 是最终外层 displacement；
- logprob 用“当前位移 action 相对于 deterministic zero-latent mean 的 Gaussian”近似。

用户指出的更合理设想是：

- 不把 displacement 当 action；
- 把每个 outer step 的 FM 初始噪声 `z_t` 当 action；
- FM ODE 去噪过程只作为 deterministic transition：

```text
state_t = 当前轮廓
action_t = z_t
disp_t = FlowODE(state_t, z_t)
state_{t+1} = state_t + frac_t * disp_t
```

于是 V4d 的策略概率是：

```text
pi_phi(z_t | state_t) = Normal(mu_phi(state_t, t), sigma_phi(state_t, t))
logprob_t = log pi_phi(z_t | state_t)
```

这样随机初始化噪声确实成为了 policy action。

### 2.2 训练流程

实现入口主要在：

- `grpo_train_v4d_latent_noise_policy.py`
- `grpo_train_v4d_point_noise_policy.py`

训练流程如下：

1. 加载 V4.6c baseline checkpoint：
   - `data/outputs/btcv_diffusion_dit_v4_6c_mlp_shared_moe_newdist_long_gpu5/checkpoints/latest.pt`
2. 冻结 YOLO / feature projection。
3. 在 V4d-A/B/C/D 中默认冻结 Flow，即 `rl_v4d_freeze_flow: true`。
4. 对每个 batch 构造手工推理上下文：
   - `cnn_feature`
   - 初始轮廓 `i_it_py`
   - canonical polygon `c_it_py`
   - GT polygon `i_gt_py`
   - `py_ind`
5. 每个样本采样 `K=8` 条 rollout。
6. 每条 rollout 走 3 个 outer step：
   - fractions: `[0.3333, 0.5, 1.0]`
   - 每步从 policy 采样 latent noise；
   - 用 frozen Flow ODE 得到 displacement；
   - 更新当前轮廓。
7. 用 existing region reward 计算 final contour 分数。
8. baseline 使用 deterministic zero-latent / policy mean 三步推理。
9. advantage 使用 rollout score 相对 baseline 的差值，并有 gate：
   - 如果该组中 best rollout 没超过 baseline，则部分样本 gate 为 0。
10. 用 PPO clipped objective 更新 noise policy。

关键代码位置：

- policy 定义：`grpo_train_v4d_point_noise_policy.py`
  - `PointLatentNoisePolicy`
  - `StructuredLatentNoisePolicy`
- latent logprob / entropy / KL：
  - `_latent_logprob`
  - `_latent_entropy`
  - `_latent_kl_to_base`
- latent 到 Flow displacement：
  - `_flow_disp_from_latent`
- 三步 rollout：
  - `_sample_rollout`
- deterministic eval：
  - `_policy_three_step`
- full validation：
  - `_compute_full_eval`

### 2.3 四个 V4d 版本

| 版本 | 文件/配置 | action policy | logprob | 结果 |
| --- | --- | --- | --- | --- |
| V4d-A | `grpo_train_v4d_latent_noise_policy.py`, `configs/btcv_select_v4_6c_rl_v4d_latent_noise_policy_gpu0.yaml` | global/per-step latent Gaussian | mean-reduced | 负增益 |
| V4d-B | `grpo_train_v4d_point_noise_policy.py`, `configs/btcv_select_v4_6c_rl_v4d_point_noise_policy_gpu0.yaml` | per-point/state-conditioned Gaussian | mean-reduced | 负增益 |
| V4d-C | `configs/btcv_select_v4_6c_rl_v4d_structured_noise_policy_gpu0.yaml` | 低维 structured latent coefficients | mean-reduced | 负增益 |
| V4d-D | `configs/btcv_select_v4_6c_rl_v4d_structured_sumlogprob_gpu0.yaml` | 同 V4d-C | summed joint logprob | 负增益 |

V4d-D 是最接近正确 policy-gradient 形式的版本。它把低维系数的 logprob 从 mean 改成 sum：

```text
log pi(z | state) = sum_i log Normal(z_i; mu_i, sigma_i)
```

这样 PPO ratio 不再被维度平均压小。smoke 时观察到：

- `grad_norm ~= 0.184`
- `ratio_min ~= 0.991`
- `ratio_max ~= 1.007`

这说明 V4d-D 的学习信号比 V4d-B/C 更明显，但最终 validation 仍没有变好。

## 3. 结果汇总

Corrected baseline：

`visual/rl_v4_select_baseline_full_corrected/v3_7_full_test_iou_20260525_223938.json`

| metric | baseline |
| --- | ---: |
| mean_iou_sample_avg | 0.931131768 |
| mean_iou_contour_avg | 0.929323452 |
| mean_dice_sample_avg | 0.963887090 |
| mean_mboundf_sample_avg | 0.825072971 |
| median_iou_sample_avg | 0.934811573 |

各 V4d `step200` full val 对比：

| variant | summary | mean IoU delta | mBoundF delta | median IoU delta |
| --- | --- | ---: | ---: | ---: |
| V4d-A global | `visual/rl_v4d_latent_noise_policy_step200/v4d_full_eval_20260526_064632.json` | -0.001025 | -0.001465 | -0.000179 |
| V4d-B per-point | `visual/rl_v4db_point_noise_policy_step200/v4d_full_eval_20260526_165549.json` | -0.001006 | -0.001438 | -0.000141 |
| V4d-C structured mean-log | `visual/rl_v4dc_structured_noise_policy_step200/v4d_full_eval_20260526_180720.json` | -0.001088 | -0.001359 | -0.000489 |
| V4d-D structured sum-log | `visual/rl_v4dd_structured_sumlogprob_step200/v4d_full_eval_20260526_192307.json` | -0.001044 | -0.001470 | -0.000661 |

对比早期 V4 displacement-action：

- `visual/rl_v4_select_auto_eval_step100/v3_7_full_test_iou_20260526_021049.json`
- step100 相对 corrected baseline：
  - `mean_iou_sample_avg: +0.000191`
  - `mean_mboundf_sample_avg: +0.000185`
  - `median_iou_sample_avg: +0.000799`

所以目前经验结论是：

```text
V4 displacement-action 三步轨迹 > V4d learned latent-noise policy
```

## 4. 为什么会失效

### 4.1 随机噪声提供探索，但不等于可优化策略

用户的直觉是对的：随机初始化噪声能提供探索。

但 RL 需要的是：

```text
gradient of log pi(action | state)
```

如果 `z ~ N(0,I)` 是固定先验，不依赖参数，那么：

```text
log p(z)
```

对模型没有可学习梯度。它只能产生多样性，不能告诉模型“以后更倾向采哪类 z”。

所以 V4d 必须把固定噪声改成：

```text
z ~ pi_phi(z | state)
```

这一步我已经做了。但新的问题变成：**学出来的 latent initializer 是否真的能改善 frozen Flow 的输出？** 实验答案目前是否定的。

### 4.2 Frozen Flow 把 action 的有效空间压得很窄

V4d-A/B/C/D 都默认：

```yaml
rl_v4d_freeze_flow: true
```

也就是只训练 noise policy，不更新 Flow。

这会造成一个结构性限制：

- Flow 已经在 V4.6c 中学会“从合理 latent 到合理 displacement”的映射；
- zero-latent / 原始推理策略本身已经很强；
- noise policy 只能在 Flow 固定映射的输入端做偏置；
- 这种偏置很容易变成“离开 Flow 训练分布”的扰动，而不是稳定改进。

换句话说，V4d 并不是直接学习几何 correction，而是在学习“怎样输入一个噪声，让 frozen Flow 间接产生更好位移”。这个 credit assignment 更难。

### 4.3 Deterministic mean eval 和 stochastic training 不匹配

训练时 V4d 采样：

```text
z = mu + sigma * epsilon
```

但 full validation 为了稳定比较，用的是 deterministic policy mean：

```text
z = mu
```

这带来一个偏差：

- 训练奖励来自 stochastic sampled rollout；
- eval 衡量的是 policy mean rollout；
- 如果好的结果来自少数 sampled noise，而不是 mean 本身，deterministic eval 就看不到收益；
- policy 可能学会“增加某些随机方向的概率”，但 mean 输出仍然不是最佳几何修正。

这可能是 V4d 的一个核心失配点。当前 full eval 的结论更准确地说是：

```text
learned deterministic latent mean initializer 不如 baseline
```

它还没有完全否定 “stochastic best-of-k learned noise sampler”。

### 4.4 Reward 是 final contour reward，action 是 latent noise，信用分配太间接

每个 rollout 有三步：

```text
z1 -> disp1 -> contour1
z2 -> disp2 -> contour2
z3 -> disp3 -> final contour
```

reward 只在 final contour 上计算。PPO 更新时三个 latent action 共享这个 final advantage。

问题：

- `z1` 的影响会被后两步覆盖或放大；
- `z2/z3` 的局部作用难以从 final score 中分离；
- latent noise 到 displacement 经过 20-step ODE；
- advantage 信号在 high-variance rollout 中很容易变噪。

相比之下，V4 displacement-action 直接对“外层位移”建模，credit assignment 更近，因此 empirically 更稳。

### 4.5 Baseline 太强，增益空间太小

Corrected V4.6c baseline 已经很高：

- mean IoU: `0.931131768`
- mean Dice: `0.963887090`
- mBoundF: `0.825072971`

这意味着：

- 很多随机扰动都会伤害已有好轮廓；
- RL 改进空间可能只有 `1e-4 ~ 1e-3`；
- 如果 policy 有一点系统性偏置，就会表现为稳定负增益。

V4d 的负增益规模约 `-0.0010` mean IoU，和 V4 原始正增益 `+0.000191` 是同一数量级，说明不是崩溃，而是方向性偏差。

### 4.6 Mean-reduced logprob 版本确实学得太弱

V4d-B/C 最初用了 mean-reduced logprob：

```text
logprob = mean_i log Normal(z_i)
```

这会把高维 action 的 logprob 差异压小，导致 PPO ratio 非常接近 1。实际日志中 per-point 版本后期 ratio 基本是：

```text
ratio_min ~= 0.99996
ratio_max ~= 1.00004
grad_norm 很小
```

这解释了 V4d-B/C 为什么几乎学不动。

V4d-D 改成 sum/joint logprob 后，学习信号明显增强。但 full val 仍负，说明失败不只是 logprob 缩放问题。

### 4.7 Structured latent 仍可能不是 Flow 的“有效控制基”

V4d-C/D 用了低维 structured basis：

- 平移 x/y
- radial
- tangent
- normal
- sin/cos harmonic variants

它比 per-point 自由噪声更平滑、更像几何控制。但它仍然作用在 Flow latent 输入上，而不是直接作用在轮廓几何位移上。

如果 Flow 对这些 latent basis 的响应不是线性的、稳定的、可解释的，structured noise 仍可能产生不稳定或不必要的 displacement。

### 4.8 Best rollout 经常为正，但 policy mean 不一定变好

训练日志经常出现：

```text
reward_mean < 0
quality_best_mean > 0
```

这说明：

- 在 K 个随机 rollout 中，确实有某些噪声能让结果更好；
- 但平均 rollout 往往更差；
- PPO 更新试图把概率推向这些少数好样本；
- 但是 deterministic mean eval 没有继承这些 best sampled rollout 的收益。

这支持一个判断：**噪声更适合做 candidate generator / best-of-k search，而不一定适合压成一个 deterministic mean policy。**

## 5. 失败不是哪些原因

这次失败基本不是以下问题：

1. **不是数据路径错误**  
   已使用 corrected `btcv_select`：
   - train: `/home/medteam/Zhrch/Datasets/BTCV/btcv_select/train`
   - val: `/home/medteam/Zhrch/Datasets/BTCV/btcv_select/test`

2. **不是 checkpoint 没加载上**  
   V4d 各版本 checkpoint load ratio 都是 100%。

3. **不是三步轨迹没记录**  
   smoke 和训练日志均显示：
   - `outer_log_count_mean = 3.0`

4. **不是 NaN / crash**  
   B/C/D 都能稳定训练到 step200 并完成 full val。

5. **不是 logprob 完全没修正**  
   V4d-D 已经做了 sum/joint logprob，学习信号显著增强，但 validation 仍负。

## 6. 当前代码改动说明

### 6.1 新增 policy 类型

在 `grpo_train_v4d_point_noise_policy.py` 中现在支持：

```yaml
rl_v4d_policy_type: 'point'
rl_v4d_policy_type: 'structured'
```

`point`：

- 每个轮廓点输出 `mu_z/logstd_z`
- action shape 约等于原始 latent field

`structured`：

- 每个 contour 输出少量 coefficient
- 再映射成平滑 latent field
- 目前默认 `rl_v4d_structured_coeffs: 8`

### 6.2 新增 logprob reduction

现在支持：

```yaml
rl_v4d_logprob_reduction: 'mean'
rl_v4d_logprob_reduction: 'sum'
```

`mean` 是 V4d-B/C 的弱信号版本。  
`sum` 是 V4d-D，更接近真正联合 action probability。

### 6.3 新增配置和 runner

新增配置：

- `configs/btcv_select_v4_6c_rl_v4d_structured_noise_policy_gpu0.yaml`
- `configs/btcv_select_v4_6c_rl_v4d_structured_sumlogprob_gpu0.yaml`

新增 runner：

- `scripts/run_v4d_structured_noise_policy_gpu0.sh`
- `scripts/run_v4d_structured_sumlogprob_gpu0.sh`

## 7. 后续建议

### 7.1 不建议继续当前 V4d-A/B/C/D

这四个版本已经覆盖了：

- global policy
- per-point policy
- structured low-dimensional policy
- corrected sum/joint logprob

结果都稳定负，所以继续单纯加长训练意义不大。

### 7.2 如果坚持“随机噪声提供探索”，更建议改成 sampler/selector

更符合实验现象的方向是：

```text
随机噪声负责产生候选
selector/value model 负责选出好候选
```

也就是：

- 不把 policy mean 当最终输出；
- 训练一个 score/value/selector 判断哪个 sampled rollout 好；
- eval 用 stochastic best-of-k 或 learned selector。

这和训练日志中 `quality_best_mean > 0` 的现象更匹配。

### 7.3 或者允许 Flow 轻微共同更新

当前 frozen Flow 限制太强。可以尝试：

- Flow 使用极小 LR；
- policy 使用正常 LR；
- 加强 KL / displacement drift regularization；
- 只开放 final head 或 small adapter。

这可能让 Flow 适应 learned latent distribution。

### 7.4 实用优先级：回到 V4 displacement-action

如果目标是短期拿到有效 RL 增益，目前最稳的是：

- 保留三步 outer trajectory；
- action 仍定义为外层 displacement；
- 或者用几何 structured displacement action；
- 而不是 latent noise action。

因为 displacement action 和 final contour reward 的 credit assignment 更直接，已有 full val 微正信号。

## 8. 最终判断

V4d 的理论建模是更“正统”的：action 是随机初始化噪声，logprob 来自可学习 `pi_phi(z | state)`。

但在当前模型和评估方式下，它失败的主要原因是：

```text
随机噪声确实能探索到更好候选，
但 learned deterministic latent mean 不能稳定复现这些好候选，
而 frozen Flow 又让 latent-space control 太间接。
```

因此，V4d 不是代码失败，而是实验假设在当前条件下不成立。下一步若继续利用随机噪声，应把它作为 **candidate generator**，配合 selector/best-of-k，而不是把它压成 deterministic policy mean。
