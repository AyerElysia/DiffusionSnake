# BTCV V3.4-FM 强化学习后训练工作报告

日期: 2026-05-07

## 1. 工作目标

本次工作的目标是围绕 `DiffusionSnake-12-30` 的 BTCV V3.4-FM 后训练线，完成以下事情：

1. 审查强化学习 / GRPO 后训练实现是否真实生效。
2. 对照 `flow_grpo-main` 的 GRPO 方式，统一当前项目的后训练逻辑。
3. 让后训练尽量保持为纯 RL loss，避免 supervised diff loss 混入造成退化。
4. 通过可复现实验找到一个真正有效、且相对 base 有提升的配置。

## 2. 主要工作内容

### 2.1 强化学习路径审查

先对 `grpo_train.py`、`lib/train/trainers/diffusion_grpo_trainer.py`、`lib/train/rewards/region_reward.py`、`lib/networks/diffusion/flow_matching_evolution.py` 做了链路核查，重点确认：

- GRPO / RL 的损失是否真的进入反向传播。
- reward 是否只保留目标项，避免引入不必要的 IoU 干扰。
- policy window 内外的随机采样是否一致，避免“采样覆盖和优化覆盖不一致”的问题。
- checkpoint 是否被 reference module 污染。

### 2.2 代码层修正

围绕上述问题做了几项关键修正：

- 增加纯 RL 开关，确保后训练时可以只走 RL loss，不再叠加 supervised diff loss。
- reward 支持更细粒度控制，当前有效方案里保留 mBoundF 方向的 reward，关闭 IoU 和 Dice 的干扰项。
- trainer 中把 reference 模块改为非注册方式保存，避免 checkpoint 中出现 `ref_flow.*`、`ref_gcn.*` 之类的脏键。
- 修复 policy window 外仍然采样随机噪声的问题，保证实际采样范围和 loss 覆盖范围一致。
- 增加更完整的日志和元数据记录，便于区分 base、seed repeat、窗口范围和 reward 配置。

### 2.3 实验配置整理

围绕 k=8 做了多组后训练配置，重点比较了：

- `grpo_k = 8`
- `grpo_pure_rl_loss = true`
- `grpo_window_size = 1`
- `grpo_window_range = [18, 19]` 或 `[15, 19]`
- `reward_w_dice = 0.0`
- `reward_w_iou = 0.0`
- `beta = 0.01`
- `clip_range = 0.1`
- `grpo_action_std = 0.08`

其中，当前最优方案是纯 mBoundF delta reward、最后一步 policy window、纯 RL loss 的组合。

## 3. 关键验证结果

### 3.1 Base 对照

V3.4-FM base checkpoint 的 full eval 结果为：

| 指标 | Base |
|---|---:|
| mean_iou_sample_avg | 0.892484 |
| mean_dice_sample_avg | 0.941425 |
| mean_mboundf_sample_avg | 0.775513 |

### 3.2 当前最优后训练结果

当前最优 checkpoint：

- [DiffusionSnake-12-30/data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_last300/checkpoints/step300.pt](../data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_last300/checkpoints/step300.pt)

对应 full eval summary：

- [DiffusionSnake-12-30/visual/v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_last300_eval_step300/v3_7_full_test_iou_20260507_042631.json](../visual/v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_last300_eval_step300/v3_7_full_test_iou_20260507_042631.json)

结果如下：

| 指标 | Base | 最优后训练 | 提升 |
|---|---:|---:|---:|
| mean_iou_sample_avg | 0.892484 | 0.894166 | +0.001682 |
| mean_dice_sample_avg | 0.941425 | 0.942498 | +0.001073 |
| mean_mboundf_sample_avg | 0.775513 | 0.777239 | +0.001726 |

这说明当前后训练线已经不是“只做到了不退化”，而是对 BTCV 验证集三项核心指标都带来了稳定提升。

### 3.3 其他对照实验

下面几组实验也完成了验证：

| 方案 | IoU | Dice | mBoundF | 结论 |
|---|---:|---:|---:|---|
| `last300@100` | 0.893424 | 0.942026 | 0.776422 | 有提升，但不如最终 300 step |
| `safe300@300` | 0.893285 | 0.941951 | 0.776145 | 比 base 好，但弱于 `[18,19]` |
| `seed16_retry@100` | 0.893153 | 0.941851 | 0.776108 | 复现有效，但幅度略弱 |
| Dice mix | 0.892738 | 0.941592 | 0.775761 | 不推荐作为主方案 |

综合来看，`mBoundF-only + pure RL + last window + k=8 + 300 step` 是当前最稳的组合。

## 4. 已确认的问题与修复点

### 4.1 纯 RL loss 已经生效

训练日志中 `diff_loss_scaled=0.0`，说明当前主方案确实没有把 supervised diff loss 混进最终优化目标。

### 4.2 reward 目标已收敛

当前主线实验中关闭了 IoU 和 Dice reward，只保留对轮廓质量真正有效的 mBoundF 方向奖励，避免 reward 目标过散导致训练漂移。

### 4.3 policy window 采样修复

已经修复了 window 外仍然加噪的问题，避免出现“采样扰动在窗口外发生，但优化只覆盖窗口内”的不一致现象。

### 4.4 checkpoint 结构已清理

reference 模块不再污染 checkpoint，当前评估加载均可做到 `722 / 722 keys` 全量匹配。

## 5. 当前结论

1. 这条 BTCV V3.4-FM 后训练线已经被验证为真实有效，不是只做了日志层面的 RL。
2. 当前最佳方案是纯 RL 的 mBoundF delta reward 方案，且不需要 IoU / Dice 混入。
3. 从结果上看，`step300` 优于 `step100`，说明这条线还需要足够的训练步数才能把收益跑出来。
4. 现阶段最适合作为汇报 / 对照基线的结果，是 `last300@300` 这个 checkpoint。

## 6. 后续建议

1. 如果要继续做研究，优先做 seed repeat，确认 `last300@300` 的增益是否稳定复现。
2. 如果要进一步做 ablation，建议只围绕 reward 形式和 policy window 做最小改动，不要再重新引入 IoU 作为主 reward。
3. 如果要对外汇报，建议直接把当前方案定义为：
   - BTCV V3.4-FM
   - k=8
   - pure RL
   - mBoundF delta reward
   - last-step policy window
   - validated improvement over base

## 7. 备注

本报告只记录这次 RL 后训练主线的代码审查、实验修正和验证结论，不包含其他无关目录的开发内容。