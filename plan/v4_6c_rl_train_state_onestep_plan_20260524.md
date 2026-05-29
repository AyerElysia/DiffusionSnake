# V4.6c RL V2：训练分布中间状态的一步强化学习方案

## 结论

当前 V4.6c RL V1 不应继续作为主线长跑。它从初始轮廓出发完整跑多阶段随机推理，再用最终轮廓打分；这和 V4.6c 的训练分布不一致，导致 group 探索大多只是毛边扰动，随机 rollout 经常不如无扰动 baseline。

V2 改成：按训练阶段分布采样中间轮廓，只做一步可控更新，用“是否超过无扰动一步”作为核心奖励。目标是让 RL 学局部一步修正，而不是随机版完整推理。

## 关键改动

- 新增 `train_state_onestep` RL 采样模式。
- 停止使用完整 iterative rollout 作为 RL group 主路径。
- 从训练分布采样中间轮廓，复用 V4.6c 的 rich state sampling。
- 每个 group 只做一步候选采样，并单独计算无扰动一步 baseline。
- reward 使用 `candidate_score - deterministic_baseline_score`。
- 整组没有超过 baseline 时，不更新。
- 可视化只显示当前轮廓、一步候选、GT，并明确标出 baseline / sampled rollout。

## 实现方案

- `grpo_train_v2.py` 增加：
  - `grpo_v2_rollout_source: train_state_onestep`
  - `grpo_v2_onestep: true`
  - `grpo_v2_onestep_steps: 1`
  - `grpo_v2_update_only_if_beats_det: true`
- `train_state_onestep` 使用训练 forward 产生的中间轮廓状态，而不是从初始轮廓完整推理。
- 一步 group 指“一次 refinement 更新”，不是单个 ODE step。
- 一次 refinement 内部使用 `sample_with_logprob(..., steps=20, window_size=4, window_range=[12, 20])`，只在后段记录 policy action。
- deterministic baseline 使用同一中间轮廓上的无扰动一步预测。
- advantage 使用 `candidate_score - deterministic_score`。
- 新建配置：
  - `configs/btcv_v4_6c_rl_v2_train_state_onestep_gpu5.yaml`
  - 继承 `data/outputs/btcv_diffusion_dit_v4_6c_mlp_shared_moe_newdist_long_gpu5/checkpoints/latest.pt`

## 奖励设置

质量分：

| 项 | 权重 |
|---|---:|
| IOU | 0.25 |
| Dice | 0.10 |
| mBoundF | 0.30 |
| boundary distance | 0.35 |

最终 reward：

```text
reward = candidate_quality - deterministic_baseline_quality
```

更新规则：

```text
best_reward <= 0：整组不更新
best_reward > 0：只学习正 advantage 候选
```

## 验证标准

- 20 step smoke test：
  - 权重加载 100%。
  - 无 OOM。
  - `step_log_count_mean > 0`。
  - `gate_active_frac` 能反映超过 baseline 的 group 比例。
- 300 step 观察：
  - 正 reward group 比例。
  - `best_candidate - baseline` 的均值和分位数。
  - 小验证 IOU / mBoundF 是否稳定高于 baseline。
- 3000 step 后完整测试：
  - 对比原始 V4.6c full BTCV baseline。
  - 对比 RL V1。
  - 报告 mean IOU、median IOU、mBoundF、失败样本数。

## 默认假设

- V4.6c RL V1 只保留为诊断，不作为最终方案。
- V4.6c RL V2 继承原始 V4.6c 权重，不随机初始化。
- 不使用 best-of-k distillation 伪装 RL。
- 不使用完整多步随机推理作为主训练信号。
- 默认使用 5 号卡；如果 5 号卡被占用，使用当前空闲卡。
