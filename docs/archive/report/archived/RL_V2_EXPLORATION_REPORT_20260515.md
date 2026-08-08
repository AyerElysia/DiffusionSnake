# GRPO V2 强化学习后训练 —— 探索与诊断报告

> 日期：2026-05-15
> 范围：BTCV V3.4-FM (flow-matching contour snake) 的 RL 后训练（V1 → V2）
> 主线脚本：`grpo_train.py`（V1，废弃路径）→ `grpo_train_v2.py`（V2，新主路径）

---

## 1. 现象与诊断

### 1.1 你看到的现象
- V1 用 10k 步 GRPO 之后，最终 boundary score 从 ~0.79 跌到 ~0.71，**越训越差**。
- 训练里 reward 数字看着没崩，但**实际质量在下滑**。
- 你的直觉：训练不稳定 / 局部震荡 / 整体下滑。

这个直觉是对的。

### 1.2 用 log 证据落地
读 V1 日志 `posttrain_grpo/logs.jsonl`（5761 行），关键指标：

| 指标 | V1 实际值 | 健康 RL 应该是 | 含义 |
|---|---|---|---|
| `approx_kl` | ≈ 0 | > 0 | 新旧策略**完全没有差异** |
| `clipfrac` | 0 | 0 ~ 0.3 之间 | PPO 截断**从未触发** |
| `grpo_policy_loss_raw` | ≈ 0 | 数十量级 | 策略损失实际**没有信号** |
| `final_score_mean` | 0.79 → 0.71 | 应该平或上升 | 真正的质量在退化 |
| `reward_mean` | ≈ 0.12（持平） | 应能爬升 | 看不见质量在掉 |

### 1.3 根因（V1 的 4 个独立 bug）

1. **没有真正的 old policy 快照**
   `_flow_grpo_loss` 里 `lp_cur` 是用**当前还未更新**的网络在同一个 `prev_sample` 上重新计算的，因此 `lp_cur == old_log`，ratio 恒等于 1，PPO surrogate `−adv·ratio` 退化为常数 ×0 均值的优势 → 实际损失约 0。
   *这是 GRPO 退化到"零信号"的直接原因。*

2. **只看 delta reward，没有绝对锚**
   `reward = R(final) − R(init)`。初始 contour 由 YOLO+augment 给出，本身噪声大，导致 reward 信号几乎被噪声主导。

3. **KL 锚定太弱**
   β=0.01，并且没有冻结参考策略；策略在低信号梯度下会向 reward 的偏置缓慢漂移。

4. **BN running stats 没冻结**
   RL 微调通常必须冻结 BN 的 running mean/var。V1 让 BN 在每个 forward 都更新，等价于"参数没动但策略输出在漂"。这点是论文里基本默认要做的事，V1 漏了。

### 1.4 还有一些次要因子

- `action_std=0.08` 太小，单步窗口下几乎没有探索。
- group baseline（k=8）在每张图上算，但每张图的 advantage 期望就是 0，**没有跨样本的全局基线**。结果优势的方差大、scale 不稳。
- 没有 held-out eval：你只能看 train reward；reward 麻木在 0.12 时你不会知道 final 在掉。
- 没有可视化：你只能看抽象数字，看不出"策略到底在把轮廓往哪里推"。

---

## 2. 论文参考

在线检索了和 flow / diffusion + GRPO 相关的几个主流方案：

- **PPO（Schulman 2017）**：标准做法——rollout 时冻结 old policy，多个 inner epoch 在同一批 rollout 上做更新，用 `approx_kl` 做早停。我们 V1 直接漏了 inner epoch 这一步。
- **DanceGRPO**：在视频 diffusion 上做 GRPO，**核心是**：(a) 多步 rollout 提取真正的 log-prob，(b) 冻结 reference policy 做 KL，(c) 训练时同步用 deterministic eval 做监控。
- **Flow-GRPO**：在 flow matching 上做 GRPO，提到 **冻结 BN / GroupNorm running stats**、**EMA reward baseline** 与 **KL 早停** 是稳定训练的必要条件。
- **DPO / RLOO**：提示用 group-relative advantage 时，**始终要混入绝对项**（abs reward），否则等价于鼓励 within-group 排名提升，但 group 自身可以整体下滑。

我们 V2 的设计基本对齐了这几条经验。

---

## 3. V2 设计 —— 一对一映射 V1 的问题

| V1 问题 | V2 修复 |
|---|---|
| `lp_cur == old_log`，PPO 退化 | **真 old-policy 快照**：rollout 时把 `log_probs` detach 存下；做 N (=2) 个 PPO inner epoch 在**同一批 rollout** 上重算 `lp_cur` |
| 没人监控 KL | 每个 inner epoch 计算 `approx_kl = 0.5·E[(lp_cur−old)²]`，`approx_kl > kl_target(=0.01)` 时**早停** |
| KL 锚定弱 | β=0.10，参考策略 = 冻结的 base ckpt（`freeze_ref_flow`），KL 用高斯均值闭式计算 |
| 只看 delta reward | `reward = 0.5·R(final) + 0.5·(R(final)−R(init))`，绝对锚 + delta 各占一半 |
| 单步探索 | window_size 默认 8，action_std=0.10 |
| 只 group baseline | group 基线 + **跨步 EMA 基线**双重去 bias |
| 无 grad-norm 截断 | `clip_grad_norm_(1.0)` 替换 V1 的 `clip_grad_value_(40)` |
| 看不见模型在 hold-out 上的真实质量 | 每 `eval_every=25` 步在**固定 eval batch**上跑 deterministic inference，记录 IoU/Dice/mBoundF |
| BN 漂 | **冻结所有 BN running stats**（参数仍可训），并在每个训练步开头重新冻结一次（防止 train/eval 切换偷偷打开） |
| 看不到策略在做什么 | 每 `viz_every=25` 步生成**轨迹胶片图**（黄→红是 ODE 步、白色是 final、青色是 init、蓝色是 GT）+ 每个 rollout 的 reward 柱状图 |

> 详细超参见 `configs/btcv_v3_4_fm_grpo_v2_gpu1.yaml`，全部 `grpo_v2_*` 键。

---

## 4. 文件与入口

| 文件 | 角色 |
|---|---|
| `grpo_train_v2.py` | V2 主入口，自己持有 PPO 内循环，**不再走** `_flow_grpo_loss` 的退化分支 |
| `lib/train/grpo_v2_utils.py` | `EMA` / `freeze_ref_flow` / `freeze_bn_running_stats` / `compute_eval_metrics` / `percentiles` |
| `configs/btcv_v3_4_fm_grpo_v2_gpu1.yaml` | GPU 1 实验配置 |
| `scripts/grpo_v2_dashboard.py` | 9 面板监控仪表盘 |
| `archive/report/archived/RL_V2_MONITORING_REPORT_20260515.md` | 监控面板使用说明 |

启动命令（GPU 1，conda `snake1`）：

```bash
cd /home/medteam/Zhrch/DiffusionSnake-12-30
conda activate snake1
CUDA_VISIBLE_DEVICES=1 CFG_FILE=configs/btcv_v3_4_fm_grpo_v2_gpu1.yaml \
    python grpo_train_v2.py --cfg_file configs/btcv_v3_4_fm_grpo_v2_gpu1.yaml
```

可选 env 变量覆盖：`GRPO_V2_STEPS`（总步数）、`GRPO_V2_LR`（学习率）、`GRPO_V2_MIN_LOAD_RATIO`（最小加载比）。

---

## 5. 探索过程（按调试时间序）

1. **第一轮 smoke**：`GRPO_V2_STEPS=10`，10 步全部跑通，但只打印了 step 1。结论：脚本本身能跑，print 节流到每 20 步。
2. **第一处 bug**：viz 函数里 `i_gt = out['i_gt_py']` KeyError。
   `inner.eval()` 模式下网络不会跑 GT 处理分支。修：改成 `inner.train()` 后再 `torch.no_grad()` 一次 forward，结束后恢复模式。
3. **第二处 bug**：30 步运行后看到 `ratio∈[1.000, 1.000]`，PPO 仍是退化的。
   读 log 发现 `policy_loss` 非零、`grad_norm` 非零，但参数变化太小 → 排查到 **LR=5e-7 是从监督训练继承的**。加 `grpo_v2_lr` 覆盖，默认 1e-5。
4. **40 步运行**：现在 `ratio∈[0.966,1.012]`，PPO 在动了！但 `final_score` 仍从 0.67 → 0.55 下滑。
   分析：奖励信号有了，但策略仍在**朝错误方向**走 → 高动作噪声（0.15）+ 弱 KL（β=0.04）+ BN running stats 在漂 三者叠加导致 over-shoot。
5. **第三处修复（关键）**：冻结 BN running stats，降低 action_std 到 0.10，提高 β 到 0.10，降 LR 到 5e-6，降 inner_epochs 到 2，KL 早停阈值 0.01。
6. **第四处修复**：eval 函数里也是 `eval()` 模式问题，导致 eval 静默失败 → 强制 train()-mode + no_grad 完成 forward，恢复模式。
7. **80 步运行**：验证三个关键指标
   - `approx_kl_last` 应当 > 0 且 < kl_target
   - `clipfrac_last` 应当偶尔非零
   - `eval_iou` 应当能被记录、不崩
   - `final_score_mean` 应当不再单调下滑

（80 步运行结果见 `RL_V2_MONITORING_REPORT_20260515.md` 与 `data/outputs/btcv_v3_4_fm_grpo_v2_gpu1/posttrain_grpo_v2/logs.jsonl`。）

---

## 6. 给你的可视化与监控（你最在意的部分）

- **轨迹胶片图（trajectory tape）**：`visual/grpo_v2_btcv_v3_4_fm_grpo_v2_gpu1/traj_step{N:06d}.png`
  - 每 25 步生成一次，针对**同一张固定 eval 图**做 k_viz=4 次 rollout。
  - 颜色编码：**黄→红 = ODE 早→晚步**（你可以直观看到轮廓从 init 向哪里"漂"过去）；青色 = init；白色 = final；蓝色 = GT。
  - 角标：每个 rollout 的 reward；面板上方说明栏。
  - **这就是你说的"看路径"**。轨迹颜色之间的间距能告诉你单步走得多大；如果训练后期颜色全挤在一起，说明策略变保守；如果颜色乱跳出图框，说明策略在崩。

- **9 面板监控仪表盘**：`scripts/grpo_v2_dashboard.py`
  - 见 `RL_V2_MONITORING_REPORT_20260515.md`。

---

## 7. 已知风险与后续

- 即使做了所有修复，RL 微调依然可能在长跑中 over-fit 到 reward proxy（boundary-based score 不完全等于真实临床指标），所以**必须**继续看 eval IoU / mBoundF，而不是 train reward。
- 如果 `approx_kl_last` 长期为 0：说明 LR 又太低；如果 `clipfrac_last` 长期 > 0.5：说明 LR 太高 / action_std 太大。
- 如果 `eval_iou` 长期低于初始：策略已经退化，立即停训并复用上一次 `latest.pt`。

最终长跑命令在 `RL_V2_MONITORING_REPORT_20260515.md` 末尾。

---

## 8. 关键突破：Group-Quality Gate（最终修复）

前 6 轮试运行都出现一个共同现象：**ratio 已经接近 1.0、approx_kl 已经接近 0**，参数几乎没动，但 `eval_iou` 还是在 step 25 附近大幅下滑。这意味着：**即使 PPO 更新极其微小，方向也是错的**。

定位到根因：组内相对优势 `(reward - reward.mean()) / std` 永远会奖励"组内最好"的 rollout，**哪怕全组都比初始解更差**。换句话说，在一个"全员翻车"的 batch 里，PPO 依然会把策略推向"少数派翻车幅度最小的那一支"，这正是震荡 → 整体下滑的根源。

**修复**（`grpo_train_v2.py` 第 433–446 行）：

```python
# Group-quality gate
delta_best = delta_scores.max(dim=0, keepdim=True).values  # (1, B)
gate_mask = (delta_best > gate_margin).float()             # 1=本张图组内至少有一支胜过 init
advantages = advantages * gate_mask
gate_active_frac = float(gate_mask.mean().item())
```

并在日志里加 `gate_active_frac` 字段，方便监控有多少 batch 被门控筛掉。

### 修复后 100 步 pilot 结果

| step | eval_iou | dice  | 备注 |
|------|----------|-------|------|
|   1  | 0.8357   | 0.9075 | baseline |
|  25  | 0.7357   | 0.8351 | 仍有初期 dip（早期 rollout 噪声大） |
|  50  | 0.7740   | 0.8631 | 开始恢复 |
|  75  | 0.8117   | 0.8883 | 接近 baseline |
| 100  | **0.8537** | **0.9189** | **新最高，+1.8% vs base，已保存 `best_iou.pt`** |

这是 V2 第一次出现"U 形稳定回升 + 超过 baseline"的曲线。  
最终 `best_iou.pt` 位于 `data/outputs/btcv_v3_4_fm_grpo_v2_gpu1/checkpoints/best_iou.pt`。

### 综合诊断清单（按重要性排序）

1. **Old-policy snapshot**（V1 致命缺陷，必须有）
2. **BN running stats 冻结**（不冻结 → eval 模式静悄悄漂移）
3. **奖励代理 ≈ eval 指标**（boundary-only 与 IoU 错配 → 越训越差）
4. **零均值组内优势**（不可减 EMA 项，否则注入方向偏置）
5. **Group-quality gate**（避免"全组翻车 → 训成翻车冠军"）★ 最易忽视
6. **多 batch eval + best ckpt**（单 batch IoU 抖动 ±5%，无法判断趋势）


---

## 9. 5000步完整长跑结果（2026-05-16）

### 核心指标

| 指标 | 值 |
|------|---|
| 总步数 | 5000 / 5000 ✅ |
| baseline eval_iou | 0.8357 |
| **历史最高 eval_iou** | **0.9012 @ step 1625** |
| **提升幅度** | **+6.55点 / +7.8%** |
| 末尾5000步 eval_iou | 0.8753（持续高于baseline） |
| 最终 action_std | 0.04（已衰减到最小值） |
| best_iou.pt 保存时间 | 2026-05-16 01:09 |

### 峰值进化轨迹

| 里程碑 | step | eval_iou | Δbaseline |
|--------|------|----------|-----------|
| 初始baseline | 1 | 0.8357 | ±0 |
| 第1峰 | 100 | 0.8537 | +1.8% |
| 第2峰 | 300 | 0.8758 | +4.0% |
| 第3峰 | 625 | 0.8875 | +5.2% |
| 第4峰 | 1475 | 0.8899 | +5.4% |
| **首破0.90 / 历史最高** | **1625** | **0.9012** | **+7.8%** |
| 长跑末尾维持 | 4725–5000 | 0.88–0.90 | +4%~+6% |

### 训练行为观察

1. **周期性 U 形**：每 ~100–200 步出现一次探索 dip（dip 最低 -10%），随后恢复并超过前一个峰值。这是正常的 RL 探索-利用交替，不是崩溃。
2. **峰值单调上涨**（前期）：s100→s300→s625→s1325→s1475→s1625，每轮峰值刷新历史最高。
3. **step 1625后进入稳态**：0.88–0.90 区间震荡，不再创历史最高，说明在当前超参下已接近可达上限。
4. **action_std 衰减到 0.04**（最小值）后策略仍保持正向，说明 KL 锚住了，没有退化。
5. **gate_active_frac 全程 0.5–1.0**，探索质量健康。

### 最佳权重使用

```bash
# 用历史最优权重做推理
cp data/outputs/btcv_v3_4_fm_grpo_v2_gpu1/checkpoints/best_iou.pt \
   data/outputs/btcv_v3_4_fm_grpo_v2_gpu1/checkpoints/best_for_inference.pt

# 或用 step1600.pt（最接近峰值的定期 ckpt）
```


---

## 10. 最终全量评估结果（Full 150-Sample Test Set）

**评估日期**：2026-05-16  
**评估集**：BTCV test set, 150 samples  
**配置**：`btcv_v3_4_fm_grpo_v2_gpu1.yaml`（YOLO-m）  
**ODE steps**：10  

### 10.1 指标对比

| 模型 | IoU | Dice | BoundF |
|---|---|---|---|
| Baseline (`latest.pt`) | **0.8925** | **0.9414** | **0.7755** |
| V2 Best (step 1625) | 0.8861 | 0.9377 | 0.7661 |
| Δ | −0.0064 (−0.71%) | −0.0037 (−0.39%) | −0.0094 (−1.21%) |

### 10.2 关键发现 ⚠️

**训练时监控指标 vs 全量评估存在严重偏差**：
- 训练中 eval_iou：0.8357（step=0）→ 0.9012（step=1625），表面提升 **+7.8%**  
- 全量测试：Baseline=0.8925，V2 Best=0.8861，实际下降 **−0.7%**

**根本原因**：训练时使用的 eval batch 太小、不具代表性。恰好那几个 batch 对应的 baseline IoU 偏低（0.8357 远低于全集 0.8925），RL 对这批样本的优化无法泛化到完整测试集。

**每样本分析**（150个样本）：
- 提升（ΔIoU > 0）：**15/150（10%）**  
- 下降（ΔIoU < 0）：**135/150（90%）**  
- 下降超过 1%：34/150  
- 最大单样本提升：+0.0061（样本137）  
- 最大单样本下降：−0.0210（样本63）

### 10.3 可视化输出

| 文件 | 说明 |
|---|---|
| `visual/eval_comparison/eval_comparison_summary.png` | 整体指标对比 + 每样本散点图 + ΔIoU 分布 + 箱线图 |
| `visual/eval_comparison/eval_comparison_gallery.png` | 最优改善 vs 最大退步 12 样本对比图 |
| `visual/eval_comparison/eval_comparison_report.txt` | 详细文本报告 |

### 10.4 结论与建议

**RL 训练未能在全量测试集上实现改善。** 真正的问题是评估信号不可靠：

1. **增大 eval batch 至 ≥50 样本**，才能得到稳定的训练监控信号  
2. **奖励函数与真实评估指标解耦**：当前 reward 是本地 delta（训练批次内相对提升），与全集绝对 IoU 关联弱  
3. **周期性跑全量评估**（每 500 步）作为 ground truth 来停止训练，不依赖代理指标  
4. **考虑改变 reward 设计**：用在固定 100 样本 eval set 上的绝对 IoU 做 reward，而不是 k 个 rollout 内的相对 delta  

