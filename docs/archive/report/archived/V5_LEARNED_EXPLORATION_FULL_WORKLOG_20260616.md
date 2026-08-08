# V5 学习型探索分布探索：完整工作记录

日期：2026-06-16
作者：Claude（编排）+ Codex/ai02（执行）+ 多智能体评审
项目：DiffusionSnake / BTCV V5 geom post-training
起点文档：`archive/report/archived/V5_NEW_EXPLORATION_DISTRIBUTION_PLAN_20260616.md`（GPT 撰写的原计划）

---

## 0. TL;DR（最重要的结论先说）

> ⚠️ **本节初版有错，已校正。** 初版结论"探索分布撞墙 0.859"是因为对照跑太短（≤500步）看太早造成的误判。详见 §7。以下为校正后的结论。

1. **RL 后训练真实可达 ~0.861（慢爬，需 750+ 步），不是撞墙 0.859。** 已知最优 = 纯随机 geom 5-step @ step650 = **0.8609**（起步 0.8556，涨 +0.0053）。
2. **最大方法论教训**：RL 后训练增益是慢爬的，任何探索方式对照**必须跑到 step700+ 才能比天花板**；在 step100-200 比较会系统性误判。本轮一度据此误判"撞墙"。
3. **用户两个直觉的当前状态（尚未被长跑完全验证）**：
   - "会学习、不随机" → 短轨迹下学习型**加速收敛**但 step350-400 略低于纯随机；长跑未验证。
   - "探索空间更大" → 16模态短轨迹略低于纯随机，但用户判断值得赌一天长跑验证（§8 进行中）。
   - （Claude 加测）"喂图像特征" → 短轨迹无增益且慢一倍；长跑未验证。
4. **GPT 原计划的 prototype 方向**：被 reward confound + 残差谱诊断削弱（§2/§4），但"探索分布整体无效"这个强结论**已撤回**——因为从未做过公平长跑对照。
5. **路上修掉/否决的真问题**（见 §6）是本轮确凿的正向产出：reward confound、explorer lr 陷阱、高频是真信号、128点修正。

---

## 1. 背景与触发

GPT 写了一份计划 `V5_NEW_EXPLORATION_DISTRIBUTION_PLAN_20260616.md`，核心叙事：
- 当前 V5 geom/Fourier 探索能采到正收益候选（quality_best_mean ≈ +0.0088），但表达力有限。
- GPT 实现了一个 "structured mixture Laplace" 特殊分布（per-contour mode，M=4，per-point 法线 residual），跑 140 step 无正收益，判定"Laplace 分布失败"。
- 结论：应转向 "GT Residual Prototype Mixture"——离线统计训练集真实 GT residual，聚类出 prototype，作为探索方向。

用户要求评估这份计划，随后明确了两个真实诉求：
- **R1：探索空间更大**
- **R2：探索越来越会探索，不是每次随机**

---

## 2. 第一层发现：GPT 的"失败"是 reward 配置混淆（reward confound）

核查 `configs/1232_final_v5_structured_mixture_5step_from3500_gpu0.yaml` vs geom 基线 config，发现失败的 mixture run 头上压着 geom **完全没有**的三块惩罚：

| reward 项 | geom 基线 | 失败 mixture | 说明 |
|---|---|---|---|
| `reward_regression_weight` | 0（默认） | **0.5** | mixture 多扛 |
| `reward_global_weight` | 1.0（默认） | **0.85** | mixture 降权 |
| `reward_detail_weight` | 0（默认） | **0.25** | mixture 多扛整套 detail 惩罚 |

实时日志铁证：失败 mixture 每步 `detail=+0.42~0.57`、`burr=0.2~0.5`；geom `detail=0.000`、`burr~0.01`。

**做了公平对照**（`configs/1232_final_v5_structured_mixture_5step_fair_gpu3.yaml`，reward 完全对齐 geom + scale 0.12→0.08）：
- step1 `best=+0.0026`（正）、`detail=0`、`eval_iou=0.8562`
- step50 eval **0.8583**、step100 **0.8579**（涨 +0.002）

**结论：GPT "140 step 无正收益 = Laplace 不行"的判定下早了，负结果主要是 reward 配置不公平造成的。**

> 关键方法论：`best=`（quality_best_mean）是 best-of-K=8 的**向上偏置选择最大值**，不能用来判生死；必须看 mean-policy 的 eval IoU。

---

## 3. 第二层：13-agent 对抗式评审 Q1-Q6

对 GPT 计划第 9 节的 6 个问题做了多视角对抗评审（4 类立场 reviewer + skeptic 对抗 + lead 综合，13 agent，64 万 token）。agent 自行读了 `_align_gt`/`_burr_penalty`/`compute_disp_gate_target` 源码，结论（对抗后存活）：

- **Q1（prototype 是否合理）**：不合理。地基（逐点 `gt-pred` residual）被 `_align_gt` 的对应噪声污染。
- **Q3（PPO logprob）**：当前 mixture logprob 有真实缺陷（`.mean` over 点让 mode 项被放大、gate*mean 双重 drift）。
- **Q4（先监督预训练？）**：不要离线 clone GT residual（目标错+会 stale）。
- **Q5（少动/不动）**：mixture/point 的 gate 接错了量（gate 只缩放 baseline，噪声无条件加，表达不了 delta=0）。geom 路径无此 gate。
- **Q6（更好的分布）**：把 geom 的 Fourier 基换成数据驱动 PCA——但诊断后发现增益 marginal。

---

## 4. 第三层：离线诊断（决定方向的三个脚本）

### 4.1 残差谱诊断 `scripts/analyze_v5_gt_residual_distribution.py`
50 样本 / 1377 contours：
- 轮廓 **128 点**（不是之前以为的 512，修正了 CLAUDE.md 的假设）
- `|r_n|` 中位 0.69px / p95 3.89px
- PCA top-6 EVR=0.61，top-12=0.81，90% 能量需 20 模态
- 现有 8 模态 Fourier 覆盖 0.657；PCA top-8 ≈ 0.68（**数据驱动换基几乎打平，增益 marginal**）
- PCA mode1/3 是干净低频正弦（真信号），mode6 起出现局部抖动

### 4.2 重配准有效性诊断 `scripts/analyze_reregistration_effect.py`
对比三种 pred/GT 对应方式的高频能量：
| 对应方式 | 高频能量占比 |
|---|---|
| current（单次 roll） | 0.353 |
| optimal_roll | 0.347 |
| pointwise 重配准 | 0.341 |

**逐点重配准只把高频能量降 3.45% → 高频残差 96.5% 是真实边界信号，不是配准噪声。**
- 推翻了"高频段混噪声、不能放开高频"的顾虑
- 印证用户"高频细节探索有必要"的直觉
- **B 路线（修 `_align_gt`）NO-GO**，省一条工程线
- burr penalty 本身设计上支持真实高频（`relu(lap_final - lap_gt - margin)`，对标 GT 曲率）

### 4.3 explorer 学习监控 `scripts/inspect_geom_explorer_learning.py`
从 checkpoint 加载 FourierExplorer，在真实轮廓上前向，统计 per-mode logstd/mu 偏离 zero-init 的程度。用于判"真学 vs 假学"。
> 注意：此脚本是为旧 FourierExplorer 签名写的，Step 2 改签名后对图像特征版会报错，需更新。

---

## 5. 第四层：实验路线与结果

### 5.1 关键 bug：explorer lr 陷阱
geom 的 `FourierExplorer` lr 默认回退到主网 `rl_v4_lr=4e-8`（line 855），对从 zero-init 学习的探索头**小 ~1000 倍**，把它冻死：
- explorer_lr=4e-8：step1 后 net.4 absmax 仅 8e-8，21 步仍 ~0
- explorer_lr=5e-5：step1 后 net.4 absmax = 9.98e-5（精确放大 1250 倍）

**这很可能是"adaptive_explorer 一直没人用出效果"的真因。** 修复：config 显式加 `rl_v4_explorer_lr: 5.0e-5`。

### 5.2 学习型探索 vs 纯随机（Step 1 探针）
对照（唯一变量 = explorer 学不学）：
| | step50 | step100 |
|---|---|---|
| 纯随机基线 | 0.8562 | 0.8583 |
| 学习型 geom | **0.8583** | 0.8586 |

学习型 step50 就到了纯随机 step100 的水平——**加速收敛约一倍**，机制有效。

### 5.3 实现图像特征探索头（Step 2，委派 Codex）
给 `FourierExplorer` 加 `sampled_feat`（每个轮廓点的局部图像特征，64维）输入：
- sample/update 传同一份特征（traj['geom_sampled_feats']），保证 PPO logprob 匹配
- 支持更大模态数 + 可选高频阻尼
- 离线 sanity（`test/stage0b_image_explorer_sanity.py`）独立复跑 PASS：zero-init 身份、round-trip 1e-6、sample/update logprob 一致 4.7e-7

### 5.4 三臂对照（最终结果，跑到 step400-500）

| 探索方式 | step50 | step100 | step150 | step200 | step350 | step500 |
|---|---|---|---|---|---|---|
| 8模态几何学习型 | 0.8583 | 0.8586 | 0.8591 | 0.8586 | 0.8580 | 0.8591 |
| 8模态图像学习型 | 0.8583 | 0.8582 | 0.8586 | 0.8582 | 0.8592 | — |
| 16模态图像学习型 | 0.8565 | 0.8578 | 0.8577 | 0.8584 | 0.8590 | — |

**三者全部在 0.857-0.859 带震荡，差异 <0.001，无人突破 0.86。**

- 16 模态 step50 候选质量 `best=+0.0302`（其他的 2-20 倍）惊艳，但**是昙花**——最近 20 步均值掉回 +0.0017，eval 没转化。
- 图像特征版无增益，还慢一倍（90s/步 vs 45s/步，图像特征采样+处理开销）。

---

## 6. 确凿的正向产出（副产物）

| # | 产出 | 价值 |
|---|---|---|
| 1 | reward confound 发现 | 推翻 GPT "Laplace 失败"结论 |
| 2 | explorer lr 陷阱修复 | 真 bug，解释 adaptive_explorer 一直没效果 |
| 3 | 高频是真信号（重配准诊断） | 否决 B 路线，省工程；印证高频探索有必要 |
| 4 | 轮廓 128 点（非 512）修正 | 事实纠错 |
| 5 | 学习型探索加速收敛验证 | R2 部分成立的证据 |
| 6 | 3 个可复现诊断脚本 | 残差谱 / 重配准 / explorer 监控 |

---

## 7. ⚠️ 重要校正：原"撞墙 0.859"结论是错的（看太早了）

**2026-06-16 晚发现：我（Claude）之前所有探索对照都只跑到 step100-500，且多在 step100 附近就比较，把"跑得不够久"误判成了"撞墙 0.859"。这个结论错了，必须校正。**

**反例铁证**：隔壁的 geom 5-step run（`data/outputs/1232_final_v5_geom_action_5step_from3500_gpu2_1k/`，纯随机探索 + lr=4e-8，跑满 1000 步）完整轨迹：

| step | 100 | 200 | 400 | 650 | 1000 |
|---|---|---|---|---|---|
| eval | 0.8583 | 0.8589 | 0.8607 | **0.8609** | 0.8602 |

**它慢爬到 step650 峰值 0.8609（mBF 0.7949），从起点涨 +0.0053——是我以为的"+0.003 天花板"近两倍。** 后期在 0.859-0.861 震荡平台化。

**校正后的真实图景：**
1. **RL 后训练真实可达 ~0.861，但是"慢爬"出来的，需要 750+ 步。** 不是撞墙在 0.859。
2. **我之前的探索对照（学习型/16模态/图像）都跑太短，没资格比天花板。** 在可比步数（step350-400）下，纯随机（0.8598/0.8607）反而**领先**学习型（0.8580/0.8587）和 16模态（0.8590/0.8582）0.001-0.002。
3. **被连带证伪的派生假设**："主网 lr=4e-8 太小是瓶颈"——错。lr=4e-8 + 纯随机跑够步数就到 0.861。基于此起的 lr=5e-7 实验已停（kl 飙到 0.0008 制造不稳定，无收益）。

**关键教训（已写入记忆 [[v5_exploration_ceiling_858]]）：RL 后训练增益是慢爬的，任何对照必须跑到 step700+ 才能比天花板，在 step100-200 比较会系统性误判。** 这是本轮工作最大的方法论失误。

**当前已知最优：纯随机 geom 5-step @ step650 = 0.8609。**

**仍开放的问题（未被长跑验证）：** "更大空间"（16模态）和"会学习"在 step350-400 下略低于纯随机，但**没有任何探索方式跑满 1000 步和 0.8609 正面比过**。短轨迹下的劣势可能是"空间大学得慢"，长跑未必。→ 已起 16模态纯随机长跑对照（见 §8）赌一天验证。

---

## 8. 进行中：公平长跑对照（赌一天验证"更大空间"）

用户判断"更大空间"值得赌一天 GPU 长跑验证（不接受在短轨迹上收尾）。已起：

- **config**：`configs/1232_final_v5_geom16_random_gpu1.yaml`（GPU1）
- **唯一变量**：基于跑出 0.8609 的纯随机 8模态 config，仅把 `geom_lowfreq_modes: 8 → 16`。lr/policy/sigma/reward 全同。
- **纯随机**（无 adaptive_explorer），归因最干净：vs 0.8609 唯一差别就是模态数。
- step1：eval=0.8555（身份成立）、burr=0.007（16模态在 sigma 衰减下没爆）。
- **判据**：跑满 1000 步，best eval 能否超过纯随机 8模态的 0.8609。
  - 超过 → "探索空间更大"是真杠杆，往更大推。
  - 追平/低于 → 长跑下也证伪，"更大空间"对此任务无效，结论才扎实。

---

## 10. ✅ 定论（2026-06-17）：公平长跑证伪"更大空间"

§8 的赌注跑完了，**结论扎实：「更大空间」(R1) 对此任务不是杠杆，反而略拖累。**

为加速到 step650（RL 慢爬判据），把两个对照都改 `batch_size=6` 重启（先验证 PPO 是 per-contour 沿 K 归一化、batch>1 数学正确无需改代码；探显存定 bs=6 峰值 37.8GB/49GB 安全）。唯一变量 = `geom_lowfreq_modes` 8 vs 16，sigma 表/lr/policy/reward 全同。

| step | 8模态基线 | 16模态 | 差 |
|---|---|---|---|
| 100 | 0.8590 | 0.8572 | +0.0018 |
| 200 | 0.8593 | 0.8573 | +0.0020 |
| 400 | **0.8604**(峰) | 0.8584 | +0.0020 |
| 650 | 0.8592 | 0.8582 | +0.0010 |

峰值：8模态 **0.8604**（mbf 0.7950，step400）vs 16模态 0.8592（step350）。**13 个 eval 点，16模态无一超过同期 8模态**，差稳定 +0.0006~0.0020，后段两者各自平台震荡、8模态高一档。

**机制查实**（`grpo_train_v5_geom_action.py:648-656`）：`delta = normals × (z @ basisᵀ) × sigma`，basis 正交归一，z~N(0,I) 每模态独立，**sigma 不随模态数归一化**。所以 16模态是 √2 倍更大总探索能量 + 覆盖更高频，**不是"摊薄"** —— 真给了更大空间，它反而略输。多出的 8 个高频维度采的是噪声方向（GT 残差高频能量低，见 §4.1），增大方差拖慢收敛。**"更大空间无效"是根本结论，非配置问题。** 与残差谱诊断闭环。

**附带收获**：8模态 bs6 把丢失的 0.8609 基线复现回来（峰 0.8604/mbf 0.7950），受保护 ckpt 在 `data/outputs/1232_final_v5_geom8_baseline_bs6_gpu2/checkpoints/best_iou.pt`。且 batch=6 step100=0.8590 略超 batch=1 同期（≈0.8583），大 batch 平滑梯度略加速，没伤天花板。

## 12. ✅ 转向 reward：审 reward + detail 实验（2026-06-17→18）

探索分布走到头后，转攻 reward（用户选）。

**离线诊断**（`scripts/diagnose_detail_score_saturation.py`，复现的 0.8604 ckpt，全测试集 5411 轮廓）：高 IoU(>0.88) 轮廓 corner_dist 仅 0.72、curv_match 0.89（差 11-28%），corr(IoU,corner_dist)=0.31、corr(IoU,curv_match)=-0.05 → **detail 信号与区域 IoU 正交、确有空间**。量化证实用户"后期都是细节"的直觉。

**开训证伪**（`1232_final_v5_geom8_detail015_bs6_gpu3.yaml`，唯一变量 detail_weight 0→0.15，跑到 step700）：
- eval 全程与基线缠绕 0.858-0.860，峰 0.8600(step400) < 基线 0.8604，**没破 860**。
- **致命证据**：detail 实验 best ckpt 的 detail 诊断与基线**逐位相同**（region_iou 0.8479vs0.8480, corner_dist 0.7202vs0.7204, curv_match 0.8929vs0.8928, local_biou 0.6884vs0.6885）。**detail reward 连它本该优化的 detail 分数都没动 → 被 geom 轨无视。**

**根因（两个 null 拼成一个故事）：**
1. detail reward 大部分也栅格化（corner_dist w0.35 用 np.round 距离场、local_biou w0.10 用膨胀栅格；只有 curv_match w0.20 纯亚像素）。detail_weight=0.15 时纯亚像素有效权重仅 0.15×0.20=0.03，可忽略。
2. 更深层：geom 动作 = 法向×(z@低频basis)×sigma，8 个低频模态只能产生平滑形变；细节需高频局部点移，低频动作**结构上无法表达**。同时解释两个负结果：奖励细节→实现不了→不动；给高频模态(16模态)→采噪声→输。**低频 geom 动作参数化才是绑定约束，不是探索大小也不是 reward。**

**curv_match 专项验证（已出决定性结论）**：detail_weight 0.3 + w_curv_match 0.5（纯亚像素有效权重 0.03→0.15，放大 5x），训 650 步。**reward 端 detail 指标涨到 0.8（确在试图奖励），但 curv_match 分数纹丝不动 0.8928→0.8928，三方 checkpoint 诊断逐位相同。**

## 13. 🏁 V5 线最终统一定论：墙 = 低频 geom 动作空间表达上限

三层递进实验把整条 V5 线的负结果收束为一个统一根因：

| 实验 | 操作 | 结果 |
|---|---|---|
| 离线诊断 | 测高IoU轮廓的detail分项 | corner_dist 0.72/curv_match 0.89，与IoU正交 → detail未饱和、确有空间 |
| 16模态长跑 | 探索空间×√2 | 全程输8模态 → 高频维度采噪声 |
| detail reward 0.15 | 奖励亚像素细节 | policy逐位不动 → 没破860 |
| **curv_match 5x** | 纯亚像素信号加权放大5倍 | **curv_match 0.8928→0.8928纹丝不动** |

**统一根因**：geom 动作 = 法向×(z@低频Fourier basis)×sigma，8 低频模态只能产生平滑法向形变。三个负结果同一面墙：①探索更大→高频采噪声→输；②奖励细节→低频实现不了→不动；③学习型→只在低频空间加速。**reward 喊破喉咙，动作空间够不着。**

**0.860 天花板 = 低频 geom 动作参数化的表达上限**，对 reward shaping 和探索分布都鲁棒。

**破 860 唯一方向 = 给 policy 高频动作能力**：①开 geom_band_detail 的 detail 高频分支（代码已有，现 gate=0 关着）；②per-point 法向位移（point_explorer，代码已有）；③换更弱起点验证是否模型/数据上限而非动作上限。否则接受 0.860 收工（复现的好 ckpt 峰 0.8604）。

## 11. 后续方向（探索分布这条线已走到头）

探索分布的三个维度——**模态数（R1，已证伪）/ 学习型 explorer（R2，仅短轨迹验证加速收敛、天花板未变）/ 图像特征（无增益）**——都不是此任务杠杆。天花板 ~0.860 由模型/数据/reward 决定，不在探索头。建议转向（待用户定）：

1. **换更早/更弱起点**（epoch_3000 或更早）跑同样 RL —— 验证 0.860 是 RL 上限还是被强起点"锁住"。
2. **审 reward 设计** —— 当前 reward 主导 region IoU，可能已饱和；查 boundary/dice 权重是否压制了细节增益。
3. **R2 再赌**（学习型 explorer 长跑）—— 若仍想验证"会学习"，必须同样 bs6 跑到 step650 比峰值，不能看短轨迹。
4. **接受 0.860 收工** —— 用复现的好 checkpoint 做下游。

---

## 9. 复现清单

### 配置
- `configs/1232_final_v5_structured_mixture_5step_fair_gpu3.yaml` — reward 对齐的公平 mixture 对照
- `configs/1232_final_v5_geom_learned_probe_gpu0.yaml` — 8模态几何学习型探针（含 explorer_lr 修复）
- `configs/1232_final_v5_geom_image_explorer_verify_gpu2.yaml` — 8模态图像特征
- `configs/1232_final_v5_geom_image16_gpu6.yaml` — 16模态图像特征

### 脚本
- `scripts/analyze_v5_gt_residual_distribution.py` — 残差谱诊断
- `scripts/analyze_reregistration_effect.py` — 重配准有效性诊断
- `scripts/inspect_geom_explorer_learning.py` — explorer 学习监控（需更新以支持图像特征签名）
- `test/stage0_geom_rail_sanity.py` — geom_band_detail rail 数学 sanity
- `test/stage0b_image_explorer_sanity.py` — 图像特征 explorer sanity（含 16模态 round-trip）

### 输出目录
- `data/outputs/1232_final_v5_structured_mixture_5step_fair_gpu3/`
- `data/outputs/1232_final_v5_geom_learned_probe_v2_gpu0/`
- `data/outputs/1232_final_v5_geom_image_explorer_gpu2/`
- `data/outputs/1232_final_v5_geom_image16_gpu6/`
- `data/analysis/v5_residual_stats/`、`data/analysis/v5_reregistration/`

### 代码改动
- `grpo_train_v5_geom_action.py`：`FourierExplorer` 加 `sampled_feat` 输入（line ~314）；geom 路径存 `geom_sampled_feats` + update 复用；支持更大 `geom_lowfreq_modes` + 可选 `geom_lowfreq_damp_highfreq`。reward/PPO 逻辑未动。

### 关键超参纪律
- 用 adaptive_explorer 必须显式 `rl_v4_explorer_lr: 5.0e-5`（否则回退 4e-8 冻死）
- 判 explorer 学没学：看 net.4 偏移 / logstd 偏离 zero-init；注意 latest.pt 只在 step1 和每 save_every 步更新
