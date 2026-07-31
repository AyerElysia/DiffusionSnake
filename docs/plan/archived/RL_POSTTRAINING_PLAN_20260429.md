# RL 后训练实施方案

**项目**: DiffusionSnake — V3.4-FM 后训练  
**日期**: 2026-04-29  
**版本**: v1.0  
**配套研究文档**: `docs/notion/RL_POSTTRAINING_RESEARCH_20260429.md`

---

## 执行摘要

本方案将 V3.4-FM 的后训练分为三个递进阶段。每个阶段都有明确的验证标准，只有当前阶段效果不足时才升级到下一阶段。整个方案从成本最低、风险最小的有监督修复出发。

**关键原则**：
- 不引入新依赖，优先使用已有代码中被禁用的功能
- 每阶段必须完成留一交叉验证（LOO），不允许跳过
- GRPO 的现有实现**不可直接使用**，需重建 FM 兼容的 logprob 核心

---

## Phase 0：前置确认（30 分钟）

**目标**：确认"拐弯"失败的直接来源，为 Phase 1 锁定修复点。

### 步骤

**0.1 检查推理后处理是否启用了 Fourier 平滑**

```bash
cd /home/medteam/Zhrch/DiffusionSnake-12-30
grep -n "fourier_smooth" configs/btcv_diffusion_dit_v3_4_fm_full_noleak.yaml
grep -n "fourier_smooth" lib/networks/diffusion/flow_matching_evolution.py | head -20
```

**0.2 确认 curvature_reweight 未启用**

```bash
grep -n "v3_7_use_curvature_reweight\|curvature_loss_weight\|curvature_reweight_power" \
    configs/btcv_diffusion_dit_v3_4_fm_full_noleak.yaml
```

**0.3 在最差 5 个测试切片上运行推理，记录基线 Δκ**

```bash
# 选取 5 个已知的高曲率失败切片（食管、胰腺、胆囊）
export CFG_FILE=configs/btcv_diffusion_dit_v3_4_fm_full_noleak.yaml
python infer_v3_refinement.py --ckpt data/outputs/btcv_diffusion_dit_v3_4_fm_full_noleak/checkpoints/latest.pt \
    --vis_save visual/phase0_baseline/
```

计算基线曲率偏差（离线脚本）：

```python
import numpy as np

def compute_delta_kappa(pred_pts, gt_pts):
    """pred_pts, gt_pts: shape (128, 2)"""
    def curvature(pts):
        dx = np.gradient(pts[:, 0])
        dy = np.gradient(pts[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        k = np.abs(ddx * dy - dx * ddy) / (dx**2 + dy**2 + 1e-8)**1.5
        return k
    k_pred = curvature(pred_pts)
    k_gt = curvature(gt_pts)
    # 只在 GT 曲率最高的 20% 点上评估
    top_mask = k_gt > np.percentile(k_gt, 80)
    return np.mean(np.abs(k_pred[top_mask] - k_gt[top_mask]))
```

**验收标准**：记录 5 个切片的基线 Δκ，作为 Phase 1 的比较基准。

---

## Phase 1：有监督修复（1-2 天实验）

**目标**：启用已有但被禁用的高曲率感知机制，在不引入任何 RL 的情况下改善"拐弯"贴合。

**理论依据**：`compute_curvature_weights()` 在 `flow_matching_evolution.py` L297-304 已实现，Fourier 平滑正在销毁 ODE 输出的高频边界成分，均匀 128 点采样使高曲率段以 9:1 的比例被平直边界投票压制。

### 修改 1：启用曲率加权损失

**文件**：`configs/btcv_diffusion_dit_v3_4_fm_full_noleak.yaml`

```yaml
# 在配置文件中添加/修改：
v3_7_use_curvature_reweight: true
v3_7_curvature_loss_weight: 2.5      # 起点，扫描范围 [1.5, 4.0]
v3_7_curvature_reweight_power: 1.5   # 起点，扫描范围 [1.0, 2.5]
```

**验证**：确认 `flow_matching_evolution.py` 在 `v3_7_use_curvature_reweight=True` 时会在 FM MSE 损失中使用这些权重（检查 forward() 中 L297-304 的条件分支）。

### 修改 2：禁用或放宽 Fourier 平滑

**文件**：`configs/btcv_diffusion_dit_v3_4_fm_full_noleak.yaml`

```yaml
# 先完全关闭，看效果
fourier_smooth_k: 0      # 0 = 不做低通滤波

# 如果完全关闭导致抖动，逐步提高到 30-48
# fourier_smooth_k: 32
```

**目的**：低通滤波 `fourier_smooth_k` 在推理时截断高于 k 的 Fourier 模式。对高曲率轮廓，这等价于直接抹去边界细节。先完全关闭，确认效果后再根据需要重新引入。

### 修改 3：曲率感知 GT 点重采样

**文件**：`configs/btcv_diffusion_dit_v3_4_fm_full_noleak.yaml`

```yaml
poly_resample_curvature_alpha: 2.5   # 在高曲率段增加点密度
```

**代码检查**：先确认这个参数是否在数据预处理中被读取：

```bash
grep -n "poly_resample_curvature_alpha" lib/utils/snake/snake_gcn_utils.py
grep -n "poly_resample_curvature_alpha" lib/datasets/
```

### 训练命令

```bash
conda activate snake1
cd /home/medteam/Zhrch/DiffusionSnake-12-30

# 先检查 GPU 空闲情况
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader

# 从现有 V3.4-FM checkpoint 继续训练，2000-3000 步
export CFG_FILE=configs/btcv_diffusion_dit_v3_4_fm_full_noleak.yaml
CUDA_VISIBLE_DEVICES=<空闲卡> python diffusion_train.py \
    --resume data/outputs/btcv_diffusion_dit_v3_4_fm_full_noleak/checkpoints/latest.pt \
    --max_iters 3000
```

### Phase 1 验收标准

| 指标 | 当前基线 | Phase 1 目标 | 失败标准 |
|---|---|---|---|
| 5个高曲率切片平均 Δκ | (Phase 0 记录) | 降低 > 20% | 无变化或上升 |
| 整体 mBoundF (验证集) | (记录基线) | 不下降 | 下降 > 2% |
| 可视化最差切片 | (截图) | 拐弯处明显改善 | 无法用肉眼分辨 |

**如果 Phase 1 目标达成**：停止，不进行 RL，记录实验结果，更新配置文件。  
**如果 Phase 1 失败或改善不足**：进入 Phase 2。

---

## Phase 2：轻量 RWR（软强化学习，3-5 天实验）

**前置条件**：Phase 1 已完成，但曲率误差仍在目标以上，或患者分层评估显示 Phase 1 在训练患者上过拟合（指标提升但跨患者方差增大）。

**方法**：Reward-Weighted Regression（奖励加权回归），在 `flow_matching_evolution.py` 的 `forward()` 中做最小侵入式改动。

### 算法设计

```
对每个 batch:
  1. 对每个样本，用 M=4 个不同的 x0 随机种子运行 ODE 推理
     (利用已有的 sample_train_x0() 和 _sample_disp_from_sampled_feat())
  2. 对每条轨迹计算奖励：
     r_i = α·mBoundF(pred_i, GT) + β·IoU(pred_i, GT) + γ·exp(-Δκ_i)
     其中 α=0.5, β=0.3, γ=0.2，高曲率样本额外乘以 (1 + 0.5·κ̄_GT)
  3. 归一化权重：w_i = softmax(β_temp · r_i)，β_temp=2.0
  4. 加权 FM 损失：L_RWR = Σ_i w_i · L_FM(pred_i, GT)
  5. 只对 Stage 3 (iterative_num_steps=3 的最后一步) 应用 RWR
     Stage 1-2 保持标准 FM 损失（梯度冻结或使用低权重）
```

### 代码实现位置

**修改文件**：`lib/networks/diffusion/flow_matching_evolution.py`

- **添加位置**：`forward()` 方法中，约 L724-737 的 FM MSE 损失计算之后
- **使用已有接口**：
  - `self.sample_train_x0(state, n_samples=4)` — 生成多个 x0 起点（L288）
  - `self._sample_disp_from_sampled_feat(...)` — ODE 推理到终点（L420）
  - `self.compute_curvature_weights(gt_pts)` — GT 曲率权重（L297）
  - `compute_region_reward()` from `lib/train/rewards/region_reward.py`

**约 25-35 行新代码**，不需要新类或新文件。

### 超参数扫描方案

```
β_temp ∈ {0.5, 1.0, 2.0}   # softmax 温度
M ∈ {4, 6, 8}               # 每样本轨迹数
γ ∈ {0.1, 0.2, 0.5}         # 曲率项权重
```

优先扫描 `β_temp`（最敏感），固定 M=4 以控制计算成本。

### 计算成本估算

- M=4 时，训练成本约为标准 FM 的 4×（Stage 3 only）= 整体约 1.5-2×
- 建议设置 `rwr_start_iter: 1000`，前 1000 步做标准 FM 热身

### Phase 2 验收标准

| 指标 | Phase 1 结果 | Phase 2 目标 |
|---|---|---|
| 高曲率切片 Δκ | (记录) | 再降低 > 15% |
| 留一交叉验证 mBoundF | (记录) | 不下降 > 1% |
| 奖励曲线稳定性 | N/A | 无 reward spike (> 3σ) |

---

## Phase 3：Online Flow-GRPO（完整 RL，1-2 周开发）

**前置条件**：Phase 2 完成，且患者分层分析显示仍有系统性偏差（不是随机误差），或需要向用户提交的论文中需要 RL 方法。

### 架构重建概述

**现有 GRPO 代码的致命问题**：
- `grpo_evolution.py` 使用 `ddpm_step_with_logprob` — 这是 DDPM 核心，FM 的 ODE 没有此类型的轨迹 logprob
- `grpo_evolution.py` L181-182：`model_output=states[t]` 传入了状态张量而非速度预测——这是从未完成的占位符
- **任何基于现有 `grpo_train.py` 的 GRPO 运行都在计算垃圾梯度，静默失败**

### FM-GRPO 重建路径

**新文件**：`lib/train/trainers/fm_grpo_trainer.py`

**核心改动**：用 `flow_grpo-main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py` 中的 `sde_step_with_logprob` 替换 DDPM 核心：

```python
# 旧（错误）:
x_prev, logprob = ddpm_step_with_logprob(model_output=eps_pred, ...)

# 新（正确，从 flow_grpo-main 移植）:
# SDE 转换：dx = v_θ(x_t,t)dt + σ_t·sqrt(-dt)·dW
x_prev, logprob = fm_sde_step_with_logprob(
    velocity_pred=v_theta,  # 来自 predict_velocity()
    t=t, x=x_t,
    noise_level=0.7,  # 控制探索强度，需扫描 [0.3, 1.0]
)
```

**FM 时间步映射**：V3.4-FM 使用 `t ∈ [0,1]`，sigma = t，与 SD3 的约定相同。`flow_grpo-main` 的 SDE 核心可直接使用，只需确认 `sigma_prev = t + dt`。

**窗口机制**：保留现有 `window_range=(2,5)` 逻辑（只训练末尾 2-5 个 ODE 步骤），加上 CPS（从 step k=15 处注入噪声，在 20 步 ODE 的末端探索）。

**PPO-clip 损失**（保留现有 `diffusion_grpo_trainer.py` 中的结构）：

```python
ratio = torch.exp(logprob - ref_logprob)  # 策略比率
advantage = (reward - reward.mean()) / (reward.std() + 1e-8)  # 归一化优势
loss = -torch.min(
    ratio * advantage,
    torch.clamp(ratio, 1-clip_range, 1+clip_range) * advantage
).mean()
kl_loss = kl_beta * (logprob - ref_logprob).mean()  # KL 正则
total_loss = loss + kl_loss
```

**关键超参数**：

| 参数 | 建议初始值 | 范围 |
|---|---|---|
| `kl_beta` | 0.02 | [0.005, 0.1] |
| `clip_range` | 0.2 | [0.1, 0.3] |
| `noise_level` | 0.7 | [0.3, 1.0] |
| `group_size` | 8 | [4, 16] |
| `window_range` | (2, 4) | (2,3) to (3,6) |

### 器官门控（防止简单器官退化）

```python
# 只对 mBoundF < 0.80 的器官施加 GRPO 梯度
# mBoundF > 0.90 的器官只用标准 SFT 损失
organ_gate = (organ_mboundf < 0.80).float()
grpo_loss = organ_gate * grpo_loss + (1 - organ_gate) * sft_loss
```

### 开发优先级

1. `lib/train/trainers/fm_grpo_trainer.py` — 新文件，核心 FM-GRPO 逻辑
2. `lib/networks/diffusion/flow_matching_evolution.py` — 添加 `sample_with_fm_logprob()` 方法
3. `grpo_train.py` — 更新为调用新的 FM-GRPO trainer，删除"暂时不能用"注释
4. `configs/btcv_diffusion_dit_v3_4_fm_grpo.yaml` — 新配置文件

---

## 统一验证协议（所有 Phase 必须执行）

### V1：最差切片目视检查（无法自动化）

选取 5 个已知最差切片（食管截面、胆囊小截面、胰腺尾部），对比 Phase 前后可视化：

```bash
python infer_v3_refinement.py --ckpt <新checkpoint> --vis_indices 23,45,67,89,112
```

**如果失败类型从"平滑欠拟合"变为"锯齿状过拟合"，立即停止——修复了一个问题但引入了另一个。**

### V2：患者分层留一验证（LOO）

```python
# 伪代码：对 30 个 CT 卷的 LOO 评估
for patient_id in range(30):
    train_30 = all_patients - {patient_id}
    test_1 = patient_id
    # 用 train_30 的 checkpoint，在 test_1 的切片上评估
    mboundf_loo[patient_id] = evaluate(checkpoint, test_1_slices)

print(f"LOO mean mBoundF: {np.mean(mboundf_loo):.4f}")
print(f"LOO std: {np.std(mboundf_loo):.4f}")
```

LOO 标准差下降（或不变）才算真正泛化提升，否则判定为训练集过拟合。

### V3：曲率偏差指标 Δκ

使用 Phase 0 中定义的 `compute_delta_kappa()` 函数，报告：
- 高曲率点（GT 曲率 top-20%）的平均 Δκ
- 全部点的平均 Δκ（作为整体对照）

目标：Phase 1 后高曲率点 Δκ 降低 > 20%，全局 Δκ 不升高。

---

## 风险与应急预案

| 风险 | 概率 | 预案 |
|---|---|---|
| Phase 1 curvature_reweight 导致平直段过拟合 | 中 | 降低 loss_weight 到 1.5，或只在 Stage 3 启用 |
| Fourier smooth 关闭后输出抖动 | 中 | 提高到 fourier_smooth_k=24，不要完全关闭 |
| Phase 2 RWR β_temp 过高导致模式坍塌 | 低 | 降低 β_temp，增加熵正则 |
| Phase 3 GRPO kl_beta 调参困难 | 高 | 器官门控防止全局退化；先在 2 个器官上测试 |
| 小数据（30 卷）reward hacking | 高 | 强制 kl_beta ≥ 0.02；每 500 步做 LOO 快照 |

---

## 文件清单

### Phase 1 修改文件（仅配置）

- `configs/btcv_diffusion_dit_v3_4_fm_full_noleak.yaml`

### Phase 2 修改文件

- `lib/networks/diffusion/flow_matching_evolution.py`（在 forward() 添加 RWR block）
- `lib/train/rewards/region_reward.py`（添加曲率偏差项到奖励）

### Phase 3 新增文件

- `lib/train/trainers/fm_grpo_trainer.py`（新文件）
- `lib/networks/diffusion/fm_sde_logprob.py`（从 flow_grpo-main 移植的 SDE logprob 核心）
- `grpo_train.py`（更新，移除破损的 DDPM GRPO 调用）
- `configs/btcv_diffusion_dit_v3_4_fm_grpo.yaml`（新配置）

### 验证脚本（统一放到 test/）

- `test/eval_delta_kappa.py`
- `test/loo_cross_validation.py`

---

*本方案由三个子代理（Pro-GRPO、Pro-RWR、批判综合）辩论后综合生成。*  
*每个 Phase 都有独立的验收标准，不允许在未完成验证的情况下跳级到更复杂的 Phase。*
