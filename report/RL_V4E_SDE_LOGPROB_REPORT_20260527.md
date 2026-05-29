# RL V4e SDE 动作概率函数实验报告

## 结论

V4e 已按 Flow-GRPO 的 ODE 转 SDE 思路实现：动作概率不再用外层位移的均值高斯近似，而是直接使用内层随机 SDE latent transition 的真实 Gaussian logprob。代码可以正常训练，logprob/ratio/梯度都有效；但在 BTCV select 全量验证上，step50 和 step100 都稳定低于 corrected V4.6c baseline，因此本轮已早停。

当前判断：这个版本的“真实 SDE transition logprob”比 V4d 的噪声 policy 更贴近你的设想，但直接把每个内层 ODE step 都变成随机 SDE step，会引入持续的轻微质量损失；它没有超过 baseline，也没有超过早期 V4 displacement-action 的小幅正收益。

## 新增文件

- `grpo_train_v4e_sde_three_iter.py`
- `configs/btcv_select_v4_6c_rl_v4e_sde_three_iter_gpu1.yaml`
- `scripts/run_v4e_sde_three_iter_gpu1.sh`

输出目录：

- 训练：`data/outputs/btcv_select_v4_6c_rl_v4e_sde_three_iter_gpu1/`
- step50 全量评估：`visual/rl_v4e_sde_step50_full/`
- step100 全量评估：`visual/rl_v4e_sde_step100_full/`

## 参考的 Flow-GRPO 做法

参考文件：

- `flow_grpo-main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`
- `flow_grpo-main/flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py`
- `flow_grpo-main/flow_grpo/diffusers_patch/flux_pipeline_with_logprob.py`

核心思想是：

1. 先由 ODE/flow 预测当前 step 的 deterministic mean。
2. 再加 SDE noise 得到真实采样的 `prev_sample`。
3. logprob 不是对最终 displacement 建模，而是计算真实 `prev_sample` 在 `N(prev_sample_mean, transition_std)` 下的概率。

本项目里已有接近实现：

- `lib/networks/diffusion/flow_matching_evolution.py`
  - `_gaussian_step_with_logprob()`
  - `_flow_grpo_step_with_logprob()`
  - `step_with_logprob()`
  - `sample_with_logprob()`

V4e 直接复用这些接口。

## V4e 具体怎么训练

V4e 仍保留你的三步 outer refinement 作为高层策略轨迹：

1. outer step 1：fraction `0.3333`
2. outer step 2：fraction `0.5`
3. outer step 3：fraction `1.0`

区别是动作概率函数改成内层 SDE transition logprob：

- 每个 outer step 内部跑 `20` 个 flow/SDE latent step。
- 三个 outer step 一共记录 `3 * 20 = 60` 个 logprob。
- rollout 时保存真实采样到的：
  - `x_t`
  - `x_prev`
  - timestep
  - step index
  - self condition
  - sampled feature/context
  - old logprob
- PPO 更新时，用当前模型重新计算同一个 `x_prev` 在当前 transition distribution 下的 logprob：
  - `ratio = exp(logprob_current - logprob_old)`
- KL 用当前 transition mean 与 frozen reference transition mean 的差异估计。

这比 V4/V4d 更接近“随机初始化/随机 SDE 噪声提供探索，动作概率来自实际随机过程”的设想。

## 噪声调参

最初直接用 Flow-GRPO 风格 SDE 噪声会破坏轮廓：

- `flow_grpo` step mode：严重崩溃，smoke eval IoU 约 `0.017`
- `gaussian` step mode + `std=0.10`：仍明显破坏，smoke eval IoU 约 `0.635`

随后扫了更小的 SDE action std：

| std | smoke reward_mean | gate_active_frac | smoke eval_iou | 结论 |
| --- | ---: | ---: | ---: | --- |
| 0.02 | -0.03276 | 0.00 | 0.8668 | 仍太强 |
| 0.01 | -0.01378 | 0.25 | 0.8905 | 仍偏强 |
| 0.005 | -0.00704 | 0.50 | 0.8970 | 可训练但有损 |
| 0.002 | -0.00292 | 0.75 | 0.8987 | 最稳，进入长训 |

最终长训配置固定：

- `rl_v4_sde_step_mode: 'gaussian'`
- `rl_v4_sde_action_std: 0.002`
- `rl_v4_sde_noise_level: 0.25`
- `rl_v4_k: 8`
- `rl_v4_ode_steps: 20`
- `rl_v4_outer_steps: 3`
- `rl_v4_lr: 3.0e-8`

## 长训与全量评估

corrected baseline：

- 文件：`visual/rl_v4_select_baseline_full_corrected/v3_7_full_test_iou_20260525_223938.json`
- `mean_iou_sample_avg = 0.931131768`
- `mean_mboundf_sample_avg = 0.825072971`
- `median_iou_sample_avg = 0.934811573`

V4e step50：

- 文件：`visual/rl_v4e_sde_step50_full/v4e_sde_full_eval_20260527_004741.json`
- `mean_iou_sample_avg = 0.930144307`，delta `-0.000987461`
- `mean_mboundf_sample_avg = 0.823694320`，delta `-0.001378652`
- `median_iou_sample_avg = 0.933415860`，delta `-0.001395714`
- failed samples：`0`

V4e step100：

- 文件：`visual/rl_v4e_sde_step100_full/v4e_sde_full_eval_20260527_023259.json`
- `mean_iou_sample_avg = 0.930148671`，delta `-0.000983097`
- `mean_mboundf_sample_avg = 0.823657926`，delta `-0.001415045`
- `median_iou_sample_avg = 0.933434218`，delta `-0.001377355`
- failed samples：`0`

训练过程观察：

- `sde_log_count_mean = 60.0`，说明每个 rollout 都在记录三个 outer step 内的全部 SDE transition logprob。
- ratio 范围稳定，没有 PPO ratio 爆炸。
- KL 很小，没有明显发散。
- reward EMA 接近 0，但全量评估始终低于 baseline。

## 为什么仍然失效

1. **内层每步加噪会累积质量损失**

   即使 `std=0.002` 已经很小，三次 outer refinement 共 `60` 个 stochastic transition，噪声扰动仍会累计。它没有像 best-of-k 那样只保留好样本，而是在训练/eval 中持续改变 latent trajectory。

2. **真实 SDE logprob 正确，但优化目标不等于最终轮廓最优**

   logprob 的数学定义更正确，但 PPO 优化的是“这条随机 latent path 的概率”，最终 reward 又来自三步后的轮廓质量。中间 credit assignment 很长，单步 SDE transition 的微小概率调整不一定能稳定提升最终 IoU/Boundary。

3. **baseline 已经很强，随机 SDE 更容易带来负扰动**

   corrected V4.6c baseline 的 mean IoU 已经约 `0.9311`。当前随机扰动更像在强模型附近做微小采样，而不是发现新的高质量 mode。

4. **SDE 适合作为候选生成器，不一定适合作为直接部署策略**

   本轮结果继续支持之前的判断：随机噪声更可能适合做 best-of-k / seed selector / value selector，而不是直接把 stochastic trajectory 作为最终 inference 策略。

## 当前建议

不要继续 V4e 当前版本的长训。step50 和 step100 都稳定负于 baseline，继续到 step200 大概率只是重复 `-0.001` 左右的差距。

下一步更值得尝试的方向：

1. 保留真实 SDE logprob，但只在最后少数 inner steps 加噪，例如每个 outer step 只对最后 4/8 个 latent step 采样，前面保持 deterministic。
2. 使用 stochastic SDE 产生 K 个候选，然后训练 selector/value model 选择最优 seed，而不是把 SDE 轨迹本身作为最终策略。
3. 使用 V4 displacement-action 作为主线，因为它在 corrected `btcv_select` 上目前仍是唯一略正的 RL 版本。
4. 如果继续 SDE RL，评价时应比较 sampled best-of-k 或固定 seed ensemble，而不是单条随机轨迹。

