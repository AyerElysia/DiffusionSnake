# RL 反馈信号问题：离线诊断实验报告（首轮定位）

日期：2026-06-25
作者：Claude（编排 + 实验）
范围：复现 `1232_final_v5_geom8_baseline_bs6_gpu2` baseline（峰 0.8604，复现到 0.8555/step1）
目的：用实测数据证伪/证实"反馈信号问题可分为 (1) reward 计算 (2) 轨迹问责"两条假说，先证实问题、再谈改法。

诊断方式：在 `grpo_train_v5_geom_action.py` 的主训练循环里加了一个**默认关闭**的探针（env `RL_V4_CREDIT_DIAG`），在每条 K=8 rollout 每个 outer step 的截断点上算一次 partial score（复用同一份 `_quality_score`），把 deterministic 同截断点的 partial 也一起 dump，写为 `credit_diag_step{N}.json`。然后纯离线汇总。**不修改训练行为**。

跑了 20 个 training batch × K=8 rollout × 5 outer step = 800 条轨迹 × 5 步 = 4000 条 step-wise 评估。

诊断产物：
- 12 份 5-step 截断 partial 数据：`data/outputs/1232_final_v5_geom8_baseline_bs6_gpu2/credit_diag_step*.json`
- 汇总：`data/analysis/credit_diag/summary.json`
- 图：`data/analysis/credit_diag/credit_diag_evidence.png`
- 复现命令：`bash test/run_credit_diag_gpu2.sh`

---

## 1. 现状 TL;DR

**两个层面的反馈信号问题都成立，且耦合在一起**。下面给出 4 + 5 条**硬证据**。

---

## 2. 第一层：reward 计算层面（K=8 整体偏向"sampled < deterministic"）

| # | 指标 | 实测 | 含义 |
|---|---|---|---|
| 1.1 | `terminal_quality_mean` | **−0.0204** | 平均：sampled 比 deterministic 三步均值差 ~2%。**每个 batch 平均是负奖励**（与你训练日志 `reward` 字段全程为负一致）。 |
| 1.2 | `gate_active_frac_mean` | **0.6875** | 平均每批仅 69% 的 contour 有"K=8 中至少一条超 deterministic" → 31% 的 sample **整个 group 被 gate 全杀**、完全不进梯度。 |
| 1.3 | `gate < 50% 比例` | **5%** | 5% 的 batch 里 gate 失活的 contour 超过一半 → 这部分统计周期内 system-wide 学不动。 |
| 1.4 | 平均动作幅度（per-step px） | `[0.79, 0.87, 0.98, 1.09, 1.09]` | 末段动作幅度反而更大，但末段与 deterministic 偏离也在变大——**末段 sigma=0.4px 的小动作反而引发了更高的"降分"**（步5 sampled < det −0.0117）。 |

**含义**：reward 整体把 sampled rollout 评得比 deterministic 坏，**不是因为有奖励信号没收上来**，而是 K=8 探索大多落到了"被 behaved 的等价几何动作+随机低频法向扰动"的差区——已有的 reward 函数对这套 sample **系统性给负分**。这与 §13 "动作空间表达上限" 一致：随机低频法向扰动对已 strong 的 contour 平均是伤害性的（因为 baseline 已经在 det 路径上压住了全部正方向）。

但更值得注意——并不是说"reward 喊不出去"，因为本节证据 1.2 显示 **31% 的样本连 K=8 里挑不出任何一条>baseline**，意味着反馈在 *PPO 更新前* 就被 baseline+gate 联合灭掉了。

## 3. 第二层：轨迹问责层面（5 步共享同一个 advantage）

| # | 指标 | 实测 | 含义 |
|---|---|---|---|
| 2.1 | `per_step_deltas_vs_det_mean` | `[−0.0037, −0.0067, −0.0093, −0.0105, −0.0117]` | sampled 比 det 的 partial score **越往后越拉大**——早期 step 几乎与 det 平齐，后期被规模化甩开。**也就是说"伤害"主要发生在后面 step**，但 PPO 给所有 5 个 step **同样的一个数**。 |
| 2.2 | `per_step_frac_rollouts_better_than_det` | `[0.288, 0.225, 0.156, 0.150, 0.138]` | 早期 step 有 28.8% 的 rollout 优于 det、末期只有 13.8%。一条 rollout 整局质量 = 终局给 14% 概率的奥斯卡；但 step1 它有 28.8% 的命中。Terminal-only discount **扼杀掉这 28.8% 的早期正向**。 |
| 2.3 | `per_step_frac_sign_agree_per_step_mean` | `[0.75, 0.775, 0.806, 0.875, 1.0]` | **25% 的 step-1 动作所领 advantage 的符号与该 step 实际 partial 贡献的方向相反**——即 PPO 在改 step-1 的 mean 时，按终端负号把 step-1 推向**错误方向**。 |
| 2.4 | `spearman(step_partial vs terminal_partial)` | `[0.34, 0.37, 0.52, 0.65, 1.0]` | step-1 与 terminal 只 0.34 弱正相关。**用 terminal-only 算 advantage 给早期 step 的 learning signal 几乎随机** —— 这是经典 "credit-assignment lagging" 的硬指标。 |
| 2.5 | `per_step_std_K_dispersion` / `info_ratio_step / terminal_std` | `[0.0042, …, 0.0081]` / `[0.042, …, 0.081]` | K=8 在每步 partial 上的 std 仅占被 clamp 到 floor=0.1 的 terminal_std 的 4–8%。也就是 PPO 在算 advantage 时，**真正反映每步好坏的、直接可学的 dispersion 信息全被 terminal-only 平均吸收**。 |

Pictorial（详见 `credit_diag_evidence.png`）：
- 左上：partial_score 在每步都系统低于 det 的 truncation。
- 右上：大于 det 的概率随 step 单调上升地恶化。
- 左下：step1 sign-agree 只有 75%、spearman 只 0.34——terminal credit 与早期 step 几乎独立。
- 右下：在 20 batch 里，`terminal_q >= 0` 的 rollout 比例平均仅 **33%**——大部分时间所有 K 都负。

## 4. 证据是耦合的（不是"要么要么"）

四条线索合起来：

1. **reward 几乎全员打负 → K=8 探索的命中率低**（1.1-1.3）
2. **越往后越差 → 最后一步 disqualify 之前几乎所有探索**（2.1-2.2）
3. **terminal advantage 一个标量复制给 5 步 → 早期 25% 的更新方向是反的**（2.3）
4. **早期 step 与 terminal 相关性仅 0.34** → 早期 step 的细 dispersion 信息被全丢（2.4-2.5）

→ **这正是"reward 计算"和"轨迹问责"两层耦合的真根**：
- reward 几乎全员负 + 终端 squash → advantage 平均是脏的负号，再由共享 adv 复制给所有 step → 早期 step 的细 dispersion 信号被压平 → PPO 把早期 step 改向"反正更差"。explorer 的 net.4 absmax ≈ 0（worklog 已知）就是这个机制的果：explorer 想学，但梯度的有用差分信号在 step1 就被平均害死了，剩 25% 反向"训练"它往错处挪。

## 5. 我的不动训练就得到证据复用

我加的诊断 hook 默认 off，运行时 `set +e` 无副作用，仅写 JSON 文件。**不影响正常训练，checkpoint 不被 dirty**。

留下来的工具：
- 诊断 hook：`grpo_train_v5_geom_action.py` 内 `RL_V4_CREDIT_DIAG` 块
- 启动器：`test/run_credit_diag_gpu2.sh`
- 分析：`scripts/analyze_credit_diag.py`
- 出图：`scripts/plot_credit_diag.py`

下次如果改了 credit assignment（per-step reward / per-step gate / 任何 step-wise modification），把这套相同的诊断再跑一次、对比 `summary.json` 各 step 的 sign_agree / spearman / std_per_step 是否走到 1.0 / 1.0 / 真std 的同比上升，就能客观地评价改法是否真的让反馈蔓延到了早期 step——不靠最后的天花板数字（你已经 worklog 反复说"短跑必误判"）。

## 6. 下一步建议（先等你拍）

根据证据，可选 three directions，工作量和它们的针对性：

| 方案 | 改什么 | 直接对应证据 | 工程量 | 风险 |
|---|---|---|---|---|
| A. **per-step shaped reward**（先把 reward 改成每步 partial delta 的折现和，叠加 terminal bonus） | `grpo_train_v5_geom_action.py:2059+`，把 quality 从单值改成 [K, n_steps]（各步 partial−det_partial）→ adv 也 [K, n_steps]，PPO 里对每步用 `adv[ri, si]` 而不是 `adv_ri` | 2.1-2.5 全条证；直接给早期 step 真信号 | 中（动 reward + 优势张量化 + PPO inner loop 索引），但adv 链路本地可测 | 需要调 reward shaping 系数 |
| B. **per-step gate**（最低熵：reward 不变，只把 gate 也按 step 算，5 步各自放行） | 仅改 `gate` 张量化 | 2.2+部分1.2 | 小 | 不能解决早期 step 符号错配 |
| C. **value head/GAE** | 加 V(s)、按 δ 更新 | 全部 | 大 | 需新网络 |

**我的建议**：先做 A 的最小子集——即把"每步 partial delta vs det partial"作为 reward shaping 加到现有 terminal reward 上（叠加，不替换），让 PPO 的 `adv_ri` 改成 per-step 张量；然后用 `run_credit_diag_gpu2.sh` **在线诊断**对比改前改后的 sign_agree / spearman：期望改后 step1 的 spearman → 1.0、sign_agree → 100%，那即"早期 step 的 feedback 已修复"。再决定要不要长跑看天花板。

但这只是建议方案——**先等你判定这个证据是不是足够说明两个层面的问题**，再讨论怎么改。如果你认为还有更值得查的维度，我继续做探测。