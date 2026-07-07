# 逐点 FM Scale 探索路线设计报告

## 1. 背景与动机

现有 geom 探索（低频谐波采样）为所有轮廓点共享同一组低频方向系数，本质上是全局刚性扰动，探索空间受限于谐波阶数，难以表达局部（如单点凹陷、局部粘连区域）的精细修正需求。

逐点探索的目标是让 policy 独立学习每个轮廓点上 FM velocity 的幅度调整，使 RL 能够针对局部误差做精修，而不是整体平移/缩放式的粗调。

## 2. 方案演进

**v1 失败（`per_point_fm_velocity`）**：直接学习每点的加性偏移。训练全程 kl≈0，即 policy 完全没有更新。根因排查为 `init_logstd` 过小（-2.5 ~ -3.5，对应 std≈0.03~0.08）叠加学习率过小（`fmv_lr=4e-8`），两者共同导致梯度信号在数值上被截断，policy 参数几乎不动。

**v2 改进（`per_point_fm_scale`）**：改为乘法参数化，`action = fm_vel * (1 + scale_pp)`，并将 `fmv_lr` 提升到 1e-6、`init_logstd` 提升到 -0.5（std≈0.6）。梯度诊断确认 `fmv_policy_grad_norm=0.019`，非零，说明学习信号已经打通。但代价是 burr（轮廓不光滑度）爆炸到 0.7~1.8，原因是 `scale ~ N(0, 0.6)` 在约 5% 的采样下会使乘子 `(1+scale)` 反向（<0），导致该点位移方向反转、轮廓局部翻折。

**v3 最终方案（tanh 有界）**：
- 参数化改为 `raw ~ N(mu, std)`，`scale = tanh(raw) * max_scale`，`action = fm_vel * (1 + scale)`，从结构上杜绝反向乘子。
- `max_scale=0.1`：乘子被限制在 `(0.9, 1.1)`，幅度与精修任务的定位相符（不需要大幅度探索）。
- 移除 burr 惩罚（`weight=0`）：在 tanh 已经约束幅度的前提下，burr 惩罚的信号噪声大于收益，去掉后给 policy 更干净的学习信号。
- log_prob 采用 tanh-corrected 形式（SAC 风格，含 tanh 雅可比修正项），避免截断高斯参数化下 PPO 重要性比估计产生的偏差。

## 3. 当前实验配置

- Config: `configs/1232_final_v5_perpoint_fmscale_gpu5.yaml`
- 关键参数：`init_logstd=-1.5`（std≈0.22）、`max_scale=0.1`、`fmv_lr=1e-6`、`ppo_inner_epochs=2`、`per_step_credit_mode=full_extrap`
- 运行在 GPU5，当前进度约 step=20/1000，burr≈0.05~0.25，reward 在 0 附近震荡，梯度诊断 `fmv_policy_grad_norm=0.0024`，非零。

## 4. 待验证的关键问题

- step=50 eval 时 `eval_iou` 是否能超过基线 0.8546。
- policy 是否真正在学习：kl 是否随训练步数缓慢、稳定地上升（而非维持 0 或剧烈震荡）。
- 探索幅度 `max_scale` 的最优取值：当前试验值为 0.1，需结合 burr 与 reward 曲线判断是否需要进一步收窄或放宽。
