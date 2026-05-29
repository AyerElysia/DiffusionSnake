# V13 Flow-GRPO 对齐版 RL 报告

日期：2026-05-21

## 结论

当前 V13 RL 已经从“会假训练/会被蒸馏污染”改成了真正的改权重式 GRPO。

但截至 100 step，结果还不能证明它真实有效：

| 项目 | 结果 |
|---|---:|
| 固定验证 baseline | 0.884020 |
| step 75 eval IoU | 0.886372 |
| step 100 eval IoU | 0.884170 |
| step 100 相对 baseline | +0.000149 |
| 是否崩溃 | 否 |
| 是否证明有效 | 否，涨幅仍在噪声范围内 |

判断：这版 RL 的训练路径是健康的，但收益还没有被证明。它比之前安全，但还不是“已经有效”。

## 这次改了什么

新增配置：

`configs/btcv_v3_4_fm_rl_v13_flowgrpo_aligned_gpu5.yaml`

修改代码：

`grpo_train_v2.py`

核心变化：

- 增加 `grpo_v2_aligned_policy_only`
- 强制 `action_std > 0`
- 如果没有 step logprob，直接报错
- 强制关闭 distillation
- 强制关闭 latent ranker
- 强制关闭 latent policy 捷径
- 使用 deterministic baseline 做 advantage 比较
- 只允许有 logprob 的 step action 参与权重更新

当前 V13 不是 best-of-k 蒸馏，也不是 ranker 选择器，而是实际会改模型权重的 step-action GRPO。

## 之前踩过的坑

### 1. action 和 reward 不对齐

之前很多提升来自：

- 初始随机 `x0`
- best-of-k 搜索
- 推理噪声
- ranker 选择
- hard distillation

但真正被 GRPO 约束的 action 不是这些收益来源。

结果就是：reward 说这个候选好，但模型并没有通过同一个有 logprob 的动作去学习它。这样长程训练会漂移。

### 2. `action_std=0` 会造成假 RL

`action_std=0` 时没有有效随机动作，logprob 不产生真实 policy 梯度。

以前这种情况下训练可能还在跑，但主要不是 GRPO 在改权重，而是 distillation 或其他路径在动。

V13 已经禁止这种情况：对齐模式下 `action_std <= 0` 会直接报错。

### 3. distillation 会自我污染

之前的流程是：

1. 采 K 个候选。
2. 挑最好的。
3. 把它硬教回主模型。
4. 模型变了，下一轮候选分布也变。
5. 小错误逐步积累，后面退化。

V8、V9、V12 都有这个问题：前面可能小涨，后面跌。

V13 默认把这条路径关掉。

### 4. 小验证集会误导判断

V12 step25 小验证涨过，但完整测试不行。

V13 100 step 里也出现了 step1、step75 短暂上涨，但 step100 回到接近 baseline。

所以不能再用单个 step 的上涨判断“RL 有效”，必须看长程趋势和完整测试。

### 5. ratio 太接近 1 不一定是坏事

V13 的 ratio 基本在 `0.999-1.001`，说明更新很小。

这有两面：

- 好处：不容易崩。
- 坏处：可能学得太慢，收益不明显。

当前 100 step 的表现更像“安全但弱”。

## 100 step 诊断结果

训练目录：

`data/outputs/btcv_v3_4_fm_rl_v13_flowgrpo_aligned_run100_20260521`

关键检查：

| 检查项 | 结果 |
|---|---:|
| 权重加载率 | 100% |
| `step_log_count_mean` | 9 |
| `distill_loss` | 0 |
| `latent_policy` | 0 |
| `latent_ranker` | 0 |
| `grad_norm` | 正常非零 |
| 是否真 GRPO 更新 | 是 |

固定验证轨迹：

| step | eval IoU | 相对 baseline |
|---:|---:|---:|
| 1 | 0.886474 | +0.002454 |
| 20 | 0.879109 | -0.004911 |
| 40 | 0.885542 | +0.001522 |
| 75 | 0.886372 | +0.002351 |
| 100 | 0.884170 | +0.000149 |

判断：没有长程崩溃，但也没有稳定上升。

## 当前长跑安排

已经启动继续训练：

| 项目 | 内容 |
|---|---|
| 运行目录 | `data/outputs/btcv_v3_4_fm_rl_v13_flowgrpo_aligned_long500_continue_20260521` |
| 起点权重 | `run100/checkpoints/latest.pt` |
| 继续步数 | 400 |
| 总路径 | 约 500 step |
| GPU | 5 |
| tmux session | `v13_flowgrpo_long500_20260521` |
| 日志 | `data/outputs/btcv_v3_4_fm_rl_v13_flowgrpo_aligned_long500_continue_20260521/train.log` |

继续训练的固定验证 baseline 是 `0.883038`。这个 baseline 是 step100 权重下重新取固定验证集后的数值，不是原始 V3.4-FM baseline。

最终判断要同时看：

- 是否超过继续训练 baseline `0.883038`
- 是否超过原始 100-step baseline `0.884020`
- 是否能在完整 150 测试集上超过当前最佳推理方案

## 下一步判断标准

如果 500 step 后：

| 结果 | 判断 |
|---|---|
| eval IoU 长期高于 0.884020 且不回落 | V13 有继续价值 |
| eval IoU 只在 ±0.002 内震荡 | 当前 GRPO 信号太弱 |
| eval IoU 明显下跌 | 对齐后仍会损害主模型 |
| full-test 不涨 | 不能算有效 |

## 当前建议

现在不要急着说 RL 成功。

比较客观的判断是：

1. V13 修好了训练设计里的关键错误。
2. V13 至少没有像之前 distillation 方案那样快速崩。
3. 但 100 step 还没有证明收益。
4. 500 step 长跑和完整测试才是决定它是否值得继续的证据。

## 2026-05-22 完整测试结果

已评估 4 号卡续跑后的两个 checkpoint：

| checkpoint | mean IoU sample | median IoU sample | mean IoU contour | failed |
|---|---:|---:|---:|---:|
| `best_iou.pt` | 0.892199 | 0.892012 | 0.888295 | 0 |
| `latest.pt` | 0.891897 | 0.891298 | 0.887979 | 0 |

对应结果文件：

- `data/outputs/btcv_v3_4_fm_rl_v13_flowgrpo_aligned_gpu4_continue325_20260521/full_eval_best_20260522/v3_7_full_test_iou_20260522_023216.json`
- `data/outputs/btcv_v3_4_fm_rl_v13_flowgrpo_aligned_gpu4_continue325_20260521/full_eval_latest_20260522/v3_7_full_test_iou_20260522_023814.json`

结论：V13 在 4-batch 固定小验证集上曾达到 `0.906410`，但完整 150 测试集没有同步提升。小验证上涨没有泛化。

当前判断需要下调：V13 修好了 RL 训练路径，但还不是有效提升方案。
