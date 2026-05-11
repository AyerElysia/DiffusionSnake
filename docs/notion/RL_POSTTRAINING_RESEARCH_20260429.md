# RL 后训练方案研究与辩论报告

**项目**: DiffusionSnake — V3.4-FM 后训练强化学习方案  
**日期**: 2026-04-29  
**作者**: Copilot 综合三方子代理辩论  
**状态**: 最终分析，待用户决策

---

## 背景与问题

### 当前最优模型

V3.4-FM（`DiTFlowMatchingV3_4`）是当前分割效果最好的架构：
- 配置：`configs/btcv_diffusion_dit_v3_4_fm_full_noleak.yaml`
- 使用 Flow Matching ODE（10步采样，3阶段迭代精化）
- 在 BTCV 单样本测试中 IoU 达到 97.38%（V4 FM）
- 泛化时大轮廓贴合好，但**高曲率"拐弯"部位细节差**

### 核心问题定位

在展开 RL 讨论之前，综合代理的最重要发现：

> **"smooth-but-imprecise" 失败的直接原因是有监督训练的三个未启用特性，而非缺乏 RL 信号。**

| 已有但未启用 | 位置 | 作用 |
|---|---|---|
| `v3_7_use_curvature_reweight` | `flow_matching_evolution.py` L297-304 | 对高曲率 GT 点上加权 MSE 损失 |
| `fourier_smooth_k` | 推理后处理 | 低通滤波器正在销毁高频边界细节 |
| `poly_resample_curvature_alpha` | `snake_gcn_utils.py` L343-367 | 在高曲率段增加 GT 采样点密度 |

这三个功能的代码都已存在、能运行，只是配置为 false/0。

---

## 三方辩论记录

### 阵营 A：支持 Online Flow-GRPO（激进派）

**核心论点**：
- 现有 GRPO 代码的核心错误是用 DDPM logprob 计算 FM 轨迹概率，这是"类型错误"
- 正确修复：用 ODE→SDE 转换（`flow_grpo-main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py` 已有实现）：
  ```
  dx = v_θ(x_t, t)dt + σ_t√(−dt)·dW
  ```
  每步采样来自条件高斯 N(mean, σ_t²)，log_prob 变成简单二次型
- 适配路径约 150 行代码：`grpo_sampler.py` 替换 DDPM 核心，`diffusion_grpo_trainer.py` 调整损失计算
- **对"拐弯"失败的解释**：FM 在 t≈0.5 处学到的是**平均速度方向**，而高曲率器官（食管、胰腺尾部）的位移方向在训练集中具有**高方差**。在线 GRPO 的 SDE 噪声扰动迫使模型在拐弯局部探索速度场，而不只学分布众数。
- 奖励设计：`R = α·mBoundF + β·IoU + γ·SmoothnessPenalty`，按 GT 曲率加权各样本
- 阶段分离奖励：`R = R_stage1 + 0.9·R_stage2 + 0.81·R_stage3`

**反驳 RWR 的论点**：
- RWR 在高维位移空间（128×2D）存在严重的分布偏移问题：3 个 epoch 后 `π_θ³/π_θ⁰` 的重要性权重可能爆炸或归零
- BoN+SFT 不惩罚错误预测，梯度在 128 个点均匀流动，稀释了高曲率信号

---

### 阵营 B：支持 Offline RWR（稳健派）

**核心论点**：
- BTCV 只有 30 个训练 CT 卷、~1800 个切片，在线 RL 的探索空间远小于模型参数规模
- RWR 实现极简：`forward()` 中约 25 行改动，使用已有的 `sample_train_x0()` + `_sample_disp_from_sampled_feat()`
- 算法：采样 M=4-8 个不同 x0 种子的轨迹，计算每条轨迹的 mBoundF，用 `softmax(β·r)` 加权 FM 损失
- 曲率感知奖励：`r = α·IoU + β_r·BoundaryF1 + γ·exp(-ΔK)` 其中 ΔK 是预测轮廓与 GT 的曲率偏差
- 只对 Stage 3 做 RWR，Stage 1-2 冻结
- 与 `_infer_avg_samples`（推理时多轨迹平均）是对称的训练对：推理已经在做多轨迹，训练也应该

**局限性承认**：
- 固定 buffer 会迅速收敛到原训练分布，没有真正的探索
- 高 β 时模式坍塌风险
- 全局标量奖励无法区分"整体 sloppy 的 0.85 mBoundF"和"一处拐弯失败的 0.85 mBoundF"

---

### 阵营 C：综合与批判（工程现实派）

**最关键发现**：

> 现有 GRPO 代码（`grpo_evolution.py` L181-182）将 `states[t]`（潜在状态张量）而非模型噪声预测输出传递给 `ddpm_step_with_logprob`——这是从未完成的占位符，任何 GRPO 运行都会静默计算垃圾策略损失。

**对问题的深层诊断**：
1. `fourier_smooth`（推理后处理）是低通滤波器，高曲率拐弯需要**高频模式**。这个后处理正在销毁 ODE 努力学到的边界细节。
2. 均匀采样 128 个点：高曲率段约占轮廓 15-20°，对应 10-15 个点，被 113-118 个平直边界点以 9:1 投票压制。
3. `compute_curvature_weights` 已实现，但 `v3_7_use_curvature_reweight = False`。

**最终裁决**：

> RL 后训练是"在没有打好监督训练地基的情况下建造二楼"。在 30 个 CT 卷上，带非可微光栅化奖励的 RL 先验成功概率低。有监督修复的先验成功概率高且能当天出结果。

**工程建议的优先级**：
1. **Stage 0（有监督修复）**：一行配置改动，当天出结果
2. **Stage 1（软 RL）**：若 Stage 0 后仍有方差增大，用 RWR
3. **Stage 2（在线 RL）**：仅当患者分层评估显示 RWR 过拟合训练分布时，才上 Flow-GRPO
4. **备选：测试时优化（TTO）**：在 V3.4-FM 输出基础上做 10-20 步可微活动轮廓能量下降

---

## 综合比较表

| 维度 | Online Flow-GRPO | Offline RWR | 有监督修复 (Stage 0) |
|---|---|---|---|
| **实现代价** | ~150 行（需重建 logprob 核心）| ~25 行 | 1 行配置 |
| **BTCV 规模下的稳健性** | 中（需精细 KL 调参）| 中高（无探索，保守）| 高（单调安全）|
| **能否解决高曲率问题** | 是（探索机制）| 部分（全局奖励无法区分局部失败）| 是（直接对高曲率点加权）|
| **理论正确性** | 高（ODE→SDE 数学成立）| 中（离线分布偏移风险）| 高（MSE 加权是标准做法）|
| **调试难度** | 高（SDE 噪声、KL、window 需调）| 中（β、M、α 需调）| 低（curvature_loss_weight 扫描）|
| **时间成本** | 数天-数周 | 数天 | 数小时 |
| **当前 GRPO 代码可用性** | 需重写核心（现有代码是垃圾）| 可部分复用 | 无关 |

---

## 三方共识

三个代理在以下几点**完全一致**：

1. **现有 GRPO 代码不可用**：`grpo_evolution.py` 的 DDPM logprob 对 FM 是类型错误，且 L181 是未完成的占位符。任何基于现有代码的运行会产生垃圾梯度。

2. **有监督基线必须先做**：启用 `v3_7_use_curvature_reweight` + 修复 `fourier_smooth_k` 是零风险的高优先级操作。

3. **曲率感知是关键**：无论监督还是 RL，对高曲率点的专门处理都是解决"拐弯"失败的核心。奖励设计必须包含曲率偏差项，不能只用全局 IoU。

4. **评估指标要升级**：IoU 和 mBoundF 都无法捕捉曲率系统性低估。必须增加 `Δκ = |κ_pred - κ_GT|`（在弧长参数化轮廓上的 `np.gradient` 计算）。

---

## 建议最终方向

综合三方辩论，建议的路径如下：

**Phase 1（立即可做，成本极低）**：
- 启用 `v3_7_use_curvature_reweight: true`，权重 2.0-3.0
- 关闭或提高 `fourier_smooth_k`（检查推理后处理是否在销毁高频成分）
- 开启 `poly_resample_curvature_alpha: 2.5`
- 在最差的 5 个测试切片上比较前后 Δκ

**Phase 2（如 Phase 1 效果不足）**：
- 用 RWR 做轻量 RL（Stage 3 only，M=4-6 轨迹，β=0.5-1.0）
- 奖励加入每点曲率偏差项

**Phase 3（如仍需进一步提升）**：
- 从零重建 FM-GRPO：用 `flow_grpo-main/` 的 SDE logprob 核心替换现有 DDPM 逻辑
- 留一交叉验证验证是否真正泛化

**备选 Phase（可与 Phase 2 并行）**：
- 测试时优化（TTO）：以 V3.4-FM 输出初始化，做 10-20 步可微活动轮廓，仅在 mBoundF 0.80+ 的样本上最有效

---

## 参考文献

1. Flow-GRPO: arXiv 2505.05470 — SD3/Flux/WAN2.1 的 Flow Matching GRPO
2. ORW-CFM-W2 (ICLR 2025) — 带 Wasserstein 正则的在线奖励加权 FM 微调
3. MARL-MambaContour (2025) — 每个点作为独立 agent 的多代理 RL 轮廓演化
4. Preference Flow Matching (NeurIPS 2024) — DPO 风格的流匹配偏好学习
