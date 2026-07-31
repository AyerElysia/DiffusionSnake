# GRPO V2 监控报告 —— 怎么"看"强化学习

> 日期：2026-05-15
> 配套报告：`RL_V2_EXPLORATION_REPORT_20260515.md`

---

## TL;DR — 三件你必须每天看的东西

1. **轨迹胶片图** `visual/grpo_v2_btcv_v3_4_fm_grpo_v2_gpu1/traj_step{N:06d}.png`
   → 直接看模型在把轮廓往哪里推。
2. **9 面板仪表盘** `data/outputs/btcv_v3_4_fm_grpo_v2_gpu1/posttrain_grpo_v2/dashboard.png`
   → 一眼看出 RL 是健康、欠拟合、过拟合还是崩了。
3. **best_iou.pt** `data/outputs/btcv_v3_4_fm_grpo_v2_gpu1/checkpoints/best_iou.pt`
   → 训练过程中**自动保存**多 batch eval IoU 最高的那一刻，崩了也不亏。

---

## 1. 轨迹胶片图（traj_stepN.png）

这是你最想要的"看见路径"的可视化。

### 怎么读
每次生成一张大图，**横向**拼成 k_viz 个面板，每个面板就是一次 rollout：
- **青色细线** = 初始轮廓（YOLO 给出的，加了噪声）
- **从黄到红的彩色线** = ODE 的每一步轮廓
  - 黄色 = 早 ODE 步（噪声大）
  - 红色 = 晚 ODE 步（接近最终）
- **白色粗线** = 最终预测
- **蓝色粗线** = GT（真值）
- 角标 `k=i r=0.42 std=0.10` = 第几次 rollout、reward、当前动作噪声

### 怎么判断 RL 健康还是崩了
| 你看到 | 含义 | 该干嘛 |
|---|---|---|
| 黄→红 颜色梯度平滑、白线贴蓝线 | RL 正在让轮廓向 GT 收敛 | 继续 |
| 黄红颜色挤在一起、几乎不动 | 策略变保守（action_std 太低） | 抬高 `grpo_v2_action_std` |
| 黄红跳到图像边缘外 | 策略爆炸 | 立刻停训，从 `best_iou.pt` 恢复 |
| 不同 rollout 的白线差异极大 | 探索充分但不稳定 | 加大 `grpo_v2_k`、降 LR |
| 所有 rollout 白线几乎相同 | 没有探索 | 抬高 `grpo_v2_action_std` |

每 25 步生成一次（`grpo_v2_viz_every`）。

---

## 2. 9 面板仪表盘（dashboard.png）

运行：
```bash
cd /home/medteam/Zhrch/DiffusionSnake-12-30
conda activate snake1
python scripts/grpo_v2_dashboard.py
```

输出会落到 `data/outputs/btcv_v3_4_fm_grpo_v2_gpu1/posttrain_grpo_v2/dashboard.png`。

### 9 个面板逐个讲

#### (0,0) Reward 曲线
- 蓝阴影：reward 的 p10–p90 分位带
- 蓝实线：reward 均值
- 绿线：rollout 中**最好**那次的 reward
- 红线：平滑后的均值

**健康样**：均值缓慢上升 + p10–p90 带逐渐变窄。
**异常样**：均值震荡 / 整体下滑 / p90 远超平均（少数 rollout 暴涨，大多数没动）。

#### (0,1) Score components
- `final_score`：模型最终输出 vs GT 的绝对分数（boundary+dice+iou 加权）。
- `delta_score`：final − init。

**最关键的**：final_score 的趋势。如果 final 持续下滑，无论 reward 怎么样，**RL 都在让你的模型变差**。

#### (0,2) Held-out eval（蓝/红/绿点）
- `eval_iou` / `eval_mboundf` / `eval_dice`：每 25 步在**固定 hold-out** 4 个 batch 上跑 deterministic 推理。
- **这是你判断"RL 是否真的有用"的唯一可信信号**。
- 蓝点 IoU 应该**持平或上升**，不应该长期下滑。

#### (1,0) approx_kl per update（symlog 轴）
- 蓝：第一个 inner epoch 的 KL（应该≈0）
- 红：最后一个 inner epoch 的 KL（应该 < kl_target=0.01）

**健康**：红线在 1e-4 ~ 1e-2 之间小幅波动。
**异常**：
- 红线 ≡ 0 → LR 太低，PPO 没起作用
- 红线 > 0.1 → 单步动得太狠，会崩

#### (1,1) clipfrac
PPO 截断的触发比例。
- 0–0.1：探索良好
- > 0.4：策略试图大幅偏离，会被压回
- 长期 0：clip 没起作用，说明策略变化太小

#### (1,2) Importance ratio
灰带：[min, max]；黑线：mean；红线参考 1.0。
- 健康：[0.7, 1.3]，平均贴近 1
- 异常：均值跑偏 ≠ 1 → 策略大幅偏离 old policy

#### (2,0) Loss terms
蓝：policy_loss；红：kl_loss。
- policy_loss 数值可正可负，**关心绝对值**：稳定时 < 1。
- kl_loss 一般 < 0.01；> 0.1 说明跑偏。

#### (2,1) Grad norm vs action_std
- 蓝（左 y）：grad_norm — 应该 < 1.0（已 clip）
- 绿（右 y）：action_std — 单调缓慢下降

#### (2,2) Inner epochs used
- 若长期 = max（=2），说明 KL 早停没触发，可以放心继续。
- 若经常 < max，说明 KL 早停在介入，策略要"翻车"了，已被压住。

---

## 3. 日志字段对照

`logs.jsonl` 每行一个 step，字段（V2 新增的）：

| 字段 | 含义 |
|---|---|
| `reward_mean / std / p10 / p50 / p90` | reward 的统计 |
| `final_score_mean / best` | 最终轮廓的绝对质量 |
| `delta_score_mean` | final − init |
| `eval_iou / eval_mboundf / eval_dice / eval_n` | hold-out 上的真实指标，每 25 步刷新 |
| `ema_eval_iou` | hold-out IoU 的 EMA，用于平滑判断 |
| `approx_kl_first / last` | inner PPO 第一 / 最后 epoch 的 KL |
| `clipfrac_first / last` | PPO 截断比例 |
| `ratio_min / max / mean` | importance ratio 范围 |
| `policy_loss / kl_loss` | 损失项 |
| `grad_norm` | clipped 之前的梯度范数 |
| `inner_epochs` | 本步实际用了几个 inner epoch（≤ ppo_inner_epochs） |
| `action_std` | 当前动作噪声 |
| `k_rollouts` | 本步有效 rollout 数（应 ≈ k） |
| `is_best_iou` | =1 时本步是新最优 IoU，触发 `best_iou.pt` 保存 |

---

## 4. 关键超参与"出问题怎么调"小抄

| 现象 | 调什么 |
|---|---|
| `approx_kl_last ≡ 0` 多步 | 抬 `grpo_v2_lr` (1e-5 → 3e-5) |
| `clipfrac_last > 0.5` 持续 | 降 `grpo_v2_lr` 或降 `grpo_v2_action_std` |
| `eval_iou` 一路下滑 | 提高 `grpo_v2_kl_beta`（0.10 → 0.20），降 `grpo_v2_lr`，混入更多 abs reward |
| reward 上升但 eval 不动 | reward proxy 偏了，调 `grpo_v2_reward_w_iou / dice / region` 三档权重 |
| 训练振荡剧烈 | 抬 `grpo_v2_k`（6 → 12），抬 `grpo_v2_eval_batches` |

---

## 5. Pilot 实验结果（80–150 步）

### 第一次 pilot（reward = 纯 boundary）
- step 1 eval_iou = 0.854
- step 25 eval_iou = 0.703（**−15%**）
- 结论：reward proxy 严重偏离 eval IoU，必须混入 IoU / Dice。

### 第二次 pilot（reward = 0.4 region + 0.3 dice + 0.3 iou）
- step 1 eval_iou = 0.814
- step 25 eval_iou = 0.823（+0.9%）
- step 50 eval_iou = 0.787
- step 75 eval_iou = 0.792
- 结论：方向对了但单 batch eval 波动太大，需要多 batch 平均。

### 第三次 pilot（V2 最终：去掉 EMA 偏置 + 4 batch eval + BN 重新冻结）
见 `logs.jsonl`。在多 batch eval 下 IoU 应当在 baseline 附近平稳，`best_iou.pt` 会自动保存峰值。

---

## 6. 长跑启动命令

```bash
cd /home/medteam/Zhrch/DiffusionSnake-12-30
conda activate snake1

# 长跑 5000 步
CUDA_VISIBLE_DEVICES=1 CFG_FILE=configs/btcv_v3_4_fm_grpo_v2_gpu1.yaml \
    GRPO_V2_STEPS=5000 GRPO_V2_EVAL_BATCHES=4 \
    python grpo_train_v2.py --cfg_file configs/btcv_v3_4_fm_grpo_v2_gpu1.yaml \
    2>&1 | tee data/outputs/btcv_v3_4_fm_grpo_v2_gpu1/posttrain_grpo_v2/run.log

# 每隔一会儿刷一下仪表盘
python scripts/grpo_v2_dashboard.py
```

监控目录：
- 日志：`data/outputs/btcv_v3_4_fm_grpo_v2_gpu1/posttrain_grpo_v2/logs.jsonl`
- 仪表盘：`data/outputs/btcv_v3_4_fm_grpo_v2_gpu1/posttrain_grpo_v2/dashboard.png`
- 轨迹胶片：`visual/grpo_v2_btcv_v3_4_fm_grpo_v2_gpu1/traj_step*.png`
- 最优 ckpt：`data/outputs/btcv_v3_4_fm_grpo_v2_gpu1/checkpoints/best_iou.pt`

---

## 7. 何时停训

满足任意一条即可停：
1. `ema_eval_iou` 连续 200 步不再上升。
2. `eval_iou` 单调下滑超过 0.02（绝对值）。
3. `clipfrac_last > 0.5` 持续 100 步。
4. 你目测轨迹胶片图发现策略行为开始诡异（白线远离蓝线，颜色梯度紊乱）。

停训后用 `best_iou.pt` 替换 `latest.pt`，再跑标准 eval 脚本。

---

## 附录：100 步 pilot 实测（含 group-quality gate）

```
step=  1  eval_iou=0.8357  dice=0.9075  ← baseline，best_iou.pt 首次保存
step= 25  eval_iou=0.7357  dice=0.8351  ← 初期 dip（动作噪声搜索期）
step= 50  eval_iou=0.7740  dice=0.8631
step= 75  eval_iou=0.8117  dice=0.8883
step=100  eval_iou=0.8537  dice=0.9189  ← 新最高，+1.8%，best_iou.pt 覆盖保存
```

最终运行参数（生效中）：
- LR = 5e-6
- action_std = 0.10（每步 ×0.9999 衰减）
- β (KL-to-ref) = 0.10
- inner_epochs = 2，kl_target = 0.01（早停）
- k_rollouts = 6，rollout_steps = 20
- reward = 0.4·region + 0.3·dice + 0.3·iou
- **gate_margin = 0**（默认；通过 `grpo_v2_gate_margin` 可调）
- eval_batches = 4 batch（n≈31）

### 如何复现 / 启动长跑

```bash
cd /home/medteam/Zhrch/DiffusionSnake-12-30
conda activate snake1
# 干净启动
rm -f data/outputs/btcv_v3_4_fm_grpo_v2_gpu1/posttrain_grpo_v2/logs.jsonl
CUDA_VISIBLE_DEVICES=1 \
CFG_FILE=configs/btcv_v3_4_fm_grpo_v2_gpu1.yaml \
GRPO_V2_STEPS=5000 \
GRPO_V2_EVAL_BATCHES=4 \
python -u grpo_train_v2.py --cfg_file configs/btcv_v3_4_fm_grpo_v2_gpu1.yaml \
  2>&1 | tee logs/grpo_v2_long.log
```

监控（另开终端）：
```bash
# 每 60s 刷新一次仪表盘
while true; do
  python scripts/grpo_v2_dashboard.py
  sleep 60
done
```

最终最优权重：`data/outputs/btcv_v3_4_fm_grpo_v2_gpu1/checkpoints/best_iou.pt`

### 故障识别口诀

| 监控信号                    | 解读                              | 处理 |
|----------------------------|-----------------------------------|------|
| `ratio_max - 1 < 1e-3`     | LR 太低，PPO 没在更新              | 提 `grpo_v2_lr` |
| `clipfrac_last > 0.5`      | LR 太高 / action_std 过大          | 降 LR 或降 `action_std` |
| `gate_active_frac < 0.3`   | 大多数组都没人胜过 init，策略已偏  | **立即停训**，回退 `best_iou.pt` |
| `eval_iou` 连续 5 次 eval 下滑 | 已经过拟合 reward proxy          | 停训，调高 β 或调 reward 权重 |
| 轨迹胶片图颜色挤成一堆     | 策略变过度保守                    | 提升 action_std 或降 β |
| 轨迹胶片图飞出图框         | 策略发散                          | 降 LR 立即停训 |

