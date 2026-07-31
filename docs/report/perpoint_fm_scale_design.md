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

## 5. 2026-07-10：空间归因修复与连续逐点 credit

### 5.1 旧逐点方案的正确全量复核

旧版评估脚本曾在 `eval_manual_gt_init=true` 时错误绕过逐点 policy，导致所谓 `v3_mean` 实际退化为 pure FM。修复 `scripts/eval_v37_full_iou.py` 后，在固定 177 个测试样本、相同 checkpoint、`ODE_STEPS=10`、`EVAL_SEED=20260504` 下得到：

| 路线 | IoU | Dice | mBoundF | NSD@2px |
|---|---:|---:|---:|---:|
| deterministic FM | 0.851894 | 0.918754 | 0.776860 | 0.865740 |
| old per-point v3 | 0.851612 | 0.918631 | 0.776564 | 0.867080 |
| delta-NSD RL | 0.854005 | 0.920005 | 0.778915 | 0.868963 |

old v3 相对 deterministic FM 的 IoU 差值为 `-0.000283`，配对 bootstrap 95% CI 为 `[-0.001012, 0.000462]`，W/T/L=`82/0/95`。因此旧逐点路线没有证明优于 pure FM；delta-NSD 仍是当前可靠最佳 RL 路线。

### 5.2 v7-v10：从跨区域归因到联合动作归因

- v7-v9 暴露了空间归因错误：同一 GRPO 组内不同 rollout 扰动不同轮廓区域，却用标量 reward 相互排序，reward 与动作不具可比性。
- 修复后，同一个训练 step 的 8 条 rollout 在每个 outer step 共享同一连续 8 点选区；mask 外点不进入 log-prob，也没有梯度。
- v10 将选中 8 点视为一个联合动作，masked log-prob 从 mean 改为 sum。合成梯度测试确认选中点有梯度、未选点梯度严格为零。
- 但 v10 在 step1010/1020/1030/1040 的方向诊断始终接近随机：相关性约 `-0.013~+0.009`，sign accuracy `0.409~0.454`，deterministic gain 没有持续为正。说明消除跨区域错误归因是必要条件，但 8 点共享一个 contour scalar credit 仍不足以学习局部方向。

### 5.3 v11：真正逐点 marginal credit 及栅格量化瓶颈

v11 对每个真实五步 rollout 的同一 state、同一步、同一选区，分别构造“仅第 j 点采用 sampled FM scale、其余点保持 pure FM”的 full-extrap endpoint，并将第 j 点 log-prob 只与第 j 点 marginal advantage 配对。由此消除了选区内部的共享 scalar credit。

实际训练发现 region rasterization 会吞掉单点微小动作：前 10 步的 `point_quality_std_mean` 主要落在 `5e-6~9e-5`，经常低于 `adv_std_floor=1e-4`。step1010 诊断为 `corr=-0.0076`、`sign_acc=0.426`、`gain=-0.000004`。这说明问题不在小网络容量，而在逐点动作通过 CPU 栅格 mask 后几乎没有连续 reward 分辨率。

### 5.4 v12：连续点到 GT 边界距离归因

v12 保持以下部分完全不变：

- 动作仍为 `scale=tanh(raw)*0.1`，FM velocity multiplier 始终在 `(0.9, 1.1)`，不能反向；
- 仍执行真实五步 sequential rollout，后一步 state 包含前一步带探索动作；
- terminal reward、full-dataset eval 和最终区域指标不变；
- 无熵正则化，`reward_burr_weight=0`。

唯一变化是逐点归因尺子：用 sampled 点和 pure-FM 点到闭合 GT polyline 线段的连续距离差，定义
`(pure_dist_px - sampled_dist_px) / 8px`，并截断到 `[-1, 1]`。正值表示 sampled 点比 pure FM 更接近 GT。

前 6 个训练 step 的初步统计：

- `point_quality_nonzero_frac` 均值 `0.9962`；
- `point_quality_std_mean` 均值 `0.00343`，范围 `0.00111~0.00948`；
- `point_adv_abs_mean` 均值 `0.7999`；
- 没有 NaN/OOM，PPO ratio 范围约 `0.99961~1.00072`。

与 v11 首步 `point_quality_std=4.53e-5` 相比，v12 首步为 `0.002016`，约提升 45 倍，并且 99.84% credit 非零，证明连续距离已经解除栅格量化瓶颈。但严格校正后的 step1010/1020 诊断均显示：

- balanced sign accuracy 均为 `0.500`；
- positive recall 为 `0`，negative recall 为 `1`；
- centered preference correlation 分别为 `-0.0071`、`-0.0110`；
- 原始 `sign_acc≈0.659` 只是因为约 66% 标签偏向“减速”，并非局部学习能力。

step1010 的 177 样本同 checkpoint 全量评估为：deterministic FM IoU `0.851882`，v12 mean-policy IoU `0.851600`，差值 `-0.000282`，95% CI `[-0.001014, 0.000475]`，W/T/L=`84/0/93`。v12 的 NSD 提升 `+0.001361`，95% CI `[0.000016, 0.002695]`，但 IoU/Dice/mBoundF 均无提升，且 mean-policy 与 old v3 几乎完全一致。因此 v12 解决了 reward 分辨率，却没有解决局部策略塌缩。

### 5.5 v13：零均值局部策略

v12 checkpoint 的输出层分析显示 global branch 与 point branch 都快速学成负偏置；多数“减速”标签被 global `mu_g` 吸收，最终所有点均预测减速。v13 新增 `rl_v4_fm_velocity_zero_mean_local=true`：

- 策略前向只使用逐点 local offset，并在每条轮廓内部减去均值，禁止学习整体加速/减速偏置；
- 去均值项在反向中 detach，避免一个选中点的 loss 给未选点引入梯度，继续满足 mask 外梯度为零；
- 动作范围、五步 sequential rollout、连续 point credit、terminal reward 和所有训练超参保持 v12 不变。

该版本的目的不是改变探索幅度，而是强制网络学习“同一轮廓上哪些点应相对加速、哪些点应相对减速”。配置为 `configs/1232_final_v5_perpoint_fmscale_v13_zeromean_contdist_gpu6.yaml`。

v14 是 v13 的单变量学习率探针：仅将 `rl_v4_fm_velocity_lr` 从 `1e-6` 提高到 `1e-4`，其余 action、reward、rollout、credit、seed 与模型结构均保持一致。其目的在较少训练 step 内区分“局部表征不可学”和“更新速度过慢”，配置为 `configs/1232_final_v5_perpoint_fmscale_v14_zeromean_lr1e4_gpu5.yaml`。两条路线首个 step 的 reward、point credit 和 grad norm 一致；v13 ratio 范围为 `[0.999936, 1.000053]`，v14 为 `[0.993662, 1.005342]`，approx KL 从 `1.33e-11` 放大到 `1.33e-7`，约符合 100× LR 对参数步长的预期，且仍远未触及 PPO clip `[0.8, 1.2]`。

step1010 校正诊断表明，零均值机制已经解除 v12 的单类塌缩：v13 正/负召回为 `0.494/0.491`，v14 为 `0.501/0.488`，不再是 `0/1`。但整体 balanced sign accuracy 仍只有 `0.492/0.494`，centered correlation 为 `0.0156/0.0100`，尚不能认定学会局部方向。两者最后一步的相关性相对最高（v13 `0.0814`、v14 `0.0789`）。

step1020 重复诊断确认该结论稳定：v13 的 preference correlation/centered correlation/balanced sign accuracy 为 `0.0392/0.0229/0.4915`，v14 为 `0.0294/0.0138/0.4926`；最后一步 correlation 分别为 `0.0840/0.0698`，balanced sign accuracy 为 `0.5026/0.5174`。v14 虽把 deterministic gain 放大到 `+8.48e-7`，仍未改善整体局部方向准确率。因此瓶颈不只是学习率太低；可学习信号主要集中在最终精修阶段。

### 5.6 v15：完整五步 rollout，仅末两步探索与更新

v15 保留真实五步 sequential rollout：前三步执行 deterministic FM，第四、第五步才启用逐点 FM scale 探索、连续 marginal credit 和 PPO 更新。后两步 state 仍包含前三步的确定性演化以及前一 active step 的探索动作，因此没有退化为单步 RL。

实现增加 `rl_v4_policy_train_last_n_steps=2`，并显式保存 `active_policy_step_indices=[3,4]`：

- inactive 步仍保留 `states/actions/mean_actions/polys`，保证五步状态链和张量步索引对齐；
- inactive 步不采样、不构造伪 mask、不进入 PPO，point credit 用零占位，loss 归一化只统计 active actions；
- 校正诊断只汇总第4、5步，但前三步仍实际推进 deterministic state；
- 全量评估读取 checkpoint 元数据，前三步只走 FM，后两步才应用 policy mean，避免训练/推理窗口不一致。

配置为 `configs/1232_final_v5_perpoint_fmscale_v15_last2_gpu5.yaml`。除末两步窗口外，它与 v14 保持相同起点、seed、动作参数化、连续 credit 和 `1e-4` policy LR。一步 GPU smoke test 中 `point_quality_nonzero_frac=0.3977`，符合五步张量中仅 `2/5` active；PPO ratio `[0.9959,1.0045]`、KL `1.75e-7`，未出现索引错误或数值异常。

step1010 首轮校正诊断显示：末两步聚合 preference correlation `0.0529`、balanced sign accuracy `0.4998`、deterministic gain `+6.67e-7`。其中第4步为 correlation/balanced `0.0324/0.4762`，最终步为 `0.0734/0.5234`。这说明最终步仍有弱信号，但第4步抵消了它；与 v14 相同末两步的 `0.0588/0.5087` 相比没有改善。

step1020 重复诊断为整体 correlation/centered correlation/balanced `0.0436/0.0321/0.5038`，gain `+1.77e-6`。分步看，第4步仍为 `0.0363/0.4749`，最终步增强到 `0.0713/0.5396`。因此最终步局部信号可重复且增强，但“末两步共同训练”整体仍接近随机，不满足预设的显著高于 `0.5` 验收标准。v15 按停止规则结束于 step1020；保留 checkpoint 并完成 177 样本 off/mean 全量对照，用最终区域指标判定它是否仍可能带来实用收益。

### 5.7 复现文件

- 核心训练：`grpo_train_v5_geom_action.py`
- 逐点策略：`lib/train/per_point_fm_policy.py`
- 逐点 PPO：`lib/train/per_point_ppo.py`
- 连续边界 credit：`lib/train/continuous_boundary_credit.py`
- v12 配置：`configs/1232_final_v5_perpoint_fmscale_v12_pointmarginal_contdist_gpu5.yaml`
- v13 配置：`configs/1232_final_v5_perpoint_fmscale_v13_zeromean_contdist_gpu6.yaml`
- v14 LR 探针配置：`configs/1232_final_v5_perpoint_fmscale_v14_zeromean_lr1e4_gpu5.yaml`
- v15 末两步配置：`configs/1232_final_v5_perpoint_fmscale_v15_last2_gpu5.yaml`
- v12 校正诊断：`report/perpoint_final_eval/v12_step1010_corrected_diag.json`、`report/perpoint_final_eval/v12_step1020_corrected_diag.json`
- v13/v14 分步诊断：`report/perpoint_final_eval/v13_step1020_corrected_by_step_diag.json`、`report/perpoint_final_eval/v14_step1020_corrected_by_step_diag.json`
- v15 分步诊断：`report/perpoint_final_eval/v15_step1010_corrected_by_step_diag.json`、`report/perpoint_final_eval/v15_step1020_corrected_by_step_diag.json`
- v12 同 checkpoint 全量结果：`report/perpoint_final_eval/v12_off`、`report/perpoint_final_eval/v12_mean`
- 单元式直接测试：`tests/test_per_point_ppo.py`、`tests/test_continuous_boundary_credit.py`
- 全量评估：`scripts/eval_v37_full_iou.py`
- 曲线与配对统计：`scripts/analyze_perpoint_grouped_runs.py`
- 当前配对结果：`report/perpoint_grouped_comparison/paired_full_eval_comparisons.csv`
