# V3.7 系列网络整理（2026-04-21）

## 1. 快速结论

V3.7 现在不是一条线，而是 4 条并行路线：

1. **单样本主配置线**（`btcv_diffusion_dit_v3_7*.yaml`）  
目标是快速验证结构与训练策略，后期明显偏向“单样本可拟合性”。

2. **泛化实验脚本线**（`scripts/train_v37_gen_v2 ~ v6b.py` + `data/outputs/v37v*`）  
目标是找“更泛化”的训练机制（尺度条件、低方差采样、触发式学习率等）。

3. **精修线**（`scripts/finetune_denoiser.py` + `data/outputs/v3_7_ft_*`）  
目标是单样本精度上限，多个分支已达到 `mean_iou=1.0`。

4. **正式 no-leak 配置线**（`btcv_diffusion_dit_v3_7_gen_full_noleak.yaml`）  
目标是 Train/Val 分离的泛化训练框架，但目前没有统一导出的验证指标文档。

---

## 2. 路线总览

| 路线 | 代表文件/目录 | 核心特征 | 当前定位 |
|---|---|---|---|
| 单样本主配置线 | `configs/btcv_diffusion_dit_v3_7*.yaml` | 从多步 FM 演化到 1-step/zero-noise/direct regression | 历史主干，做过大量 ablation |
| 泛化实验脚本线 | `scripts/train_v37_gen_*.py` + `data/outputs/v37v*` | 引入 per-contour 归一化、scale conditioning、IoU 触发衰减等 | 方法探索密集区 |
| 精修线 | `scripts/finetune_denoiser.py` + `data/outputs/v3_7_ft_*` | RAW/F64/residual/per-point 局部优化 | 单样本上限最强 |
| no-leak 正式线 | `configs/btcv_diffusion_dit_v3_7_gen_full_noleak.yaml` | BtcvTrain/BtcvVal，非泄露设置 | 泛化主线候选 |

---

## 3. 单样本主配置线（V3.7.0 ~ V3.7.11）

### 3.1 关键演进逻辑

- **V3.7 / 7.1 / 7.2**：保留 V3.7 架构，多步 ODE，仍有迭代细化。
- **V3.7.3 / 7.4**：引入低噪声 + 多次采样平均（`infer_avg_samples=50`）。
- **V3.7.5**：开始往“近似直接回归”转（`flow_ode_steps=1`, `noise≈0`）。
- **V3.7.6**：进一步固定 `t=0`（`flow_fix_t0=true`）。
- **V3.7.7 / 7.8 / 7.9 / 7.10 / 7.11**：很多版本直接关闭 `use_dit_v3_7`，回到更朴素结构，继续走 deterministic 1-step 路线。
- **pt_id / f64**：进入 per-point 与高精度头方向（为精修线铺路）。

### 3.2 当前仓库中的最后状态（主配置线）

| 目录 | 最后 epoch | 最后 loss | 备注 |
|---|---:|---:|---|
| `btcv_diffusion_dit_v3_7_single_overfit` | 7769 | 4.1048e-02 | 早期主线，loss 较高 |
| `btcv_diffusion_dit_v3_7_1_single_overfit` | 2597 | 1.8942e-02 | 仍在多步阶段 |
| `btcv_diffusion_dit_v3_7_2_single_overfit` | 2596 | 2.1699e-03 | 明显下降 |
| `btcv_diffusion_dit_v3_7_3_single_overfit` | 2615 | 3.2962e-04 | 进一步下降 |
| `btcv_diffusion_dit_v3_7_4_single_overfit` | 2319 | 1.8956e-05 | 低噪声多采样 |
| `btcv_diffusion_dit_v3_7_5_single_overfit` | 3260 | 1.3803e-06 | 1-step 近零噪声 |
| `btcv_diffusion_dit_v3_7_6_single_overfit` | 1470 | 4.3927e-07 | fix_t0 |
| `btcv_diffusion_dit_v3_7_7_single_overfit` | 1842 | 3.2372e-07 | `use_dit_v3_7=false` |
| `btcv_diffusion_dit_v3_7_8_single_overfit` | 1933 | 2.3477e-04 | no norm + 高 lr |
| `btcv_diffusion_dit_v3_7_9_single_overfit` | 1795 | 3.4123e-07 | `use_dit_v3_7=false` |
| `btcv_diffusion_dit_v3_7_10_single_overfit` | 1079 | 1.3676e-03 | 2W 设定但未跑满 |
| `btcv_diffusion_dit_v3_7_11_single_overfit` | 1039 | 2.3887e-03 | 低 lr 版本 |

---

## 4. 泛化实验脚本线（v37v2 ~ v37v6b）

这条线的输出在 `data/outputs/v37v*`，日志文件是 `train_log.jsonl`。  
它们多数是**单样本评估**，但方法上在探索泛化机制。

### 4.1 最佳 IoU（按各目录历史 best）

| 目录 | best_mean_iou | epoch |
|---|---:|---:|
| `v37v6b_scale` | **0.9738** | 3899 |
| `v37v5h_multit` | **0.9583** | 3799 |
| `v37v6_balanced` | 0.9503 | 2999 |
| `v37v3d_zero_lowlr` | 0.9349 | 0 |
| `v37v2a_ns03` | 0.9306 | 24999 |
| `v37v3c_lowlr` | 0.9290 | 399 |
| `v37v2b_ns01` | 0.9219 | 6999 |
| `v37v2c_ft` | 0.9199 | 3299 |
| `v37v5f_pph_fm` | 0.9110 | 14799 |

### 4.2 这条线的主要新增思路

- `v5b/v5c`：per-contour normalization
- `v5h`：multi-t variance reduction
- `v6`：contour-balanced loss
- `v6b`：**scale conditioning + IoU-triggered LR decay**

---

## 5. 精修线（v3_7_ft_* / 3_7_12*）

这条线面向单样本上限，不代表泛化能力。

### 5.1 已达到/接近上限的分支

| 目录 | best_mean_iou | 特点 |
|---|---:|---|
| `v3_7_ft_f64_res` | **1.0** | F64 + residual |
| `v3_7_ft_raw_residual` | **1.0** | RAW + residual |
| `v3_7_ft_residual_best` | **1.0** | residual best 版本 |
| `v3_7_ft_per_pt` | 0.9974 | per-point 精修 |
| `v3_7_ft_f64` | 0.9860 | F64 精修 |

### 5.2 也有失败/退化样例

- `v3_7_ft_c_finallayer` 末尾 `loss=NaN`，`mean_iou=0.0`。

---

## 6. `gen_full_noleak` 与 `v6b` 的关系（重点）

结论：**`gen_full_noleak` 没有直接用上 `v6b` 的关键思路。**

`gen_full_noleak` 当前配置：
- 开 `v3_7_use_regularized_per_point=true`
- 未开 `v3_7_use_scale_conditioning`
- 常规训练调度（非 IoU 触发）

`v6b` 脚本关键点：
- 明确开启 `cfg.v3_7_use_scale_conditioning = True`
- 关闭 regularized-per-point（脚本里 `cfg.v3_7_use_regularized_per_point = False`）
- IoU 触发式学习率衰减（达到阈值再衰减）

所以两者不是“同一策略的两个配置”，而是**训练范式和开关组合都不同**。

---

## 7. 代码能力现状（V3.7 核心模块）

V3.7 网络代码已支持这些能力（并非都在同一配置中启用）：

- Per-point head
- Regularized per-point（shared + delta）
- Float64 head
- Point ID 注入（input/output）
- Laplacian regularization
- Delta regularization
- Scale conditioning（代码有，配置线未普遍启用）

对应文件：
- [lib/networks/diffusion/dit_denoiser_v3_7.py](/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30/lib/networks/diffusion/dit_denoiser_v3_7.py)
- [lib/networks/diffusion/flow_matching_evolution.py](/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30/lib/networks/diffusion/flow_matching_evolution.py)

---

## 8. 建议的整理策略（面向后续维护）

1. **保留 2 条主线**
- 泛化主线：`gen_full_noleak` + 吸收 `v6b`/`v5h` 的有效机制
- 单样本上限线：`ft_residual_best`（作为“精度天花板参考”）

2. **其余历史方案归档**
- `v3_7_1~11`、`v37v2~v6b` 作为实验库保留，不再并行扩散。

3. **统一评估出口**
- 对泛化主线固定输出：Val IoU / Dice / 每类或每轮廓指标。
- 避免只看训练 loss 或单样本 IoU。

---

## 9. 索引

- 主配置目录：`configs/btcv_diffusion_dit_v3_7*.yaml`
- 泛化脚本目录：`scripts/train_v37_gen*.py`
- 精修脚本：`scripts/finetune_denoiser.py`
- 结果目录：
  - 单样本主线：`data/outputs/btcv_diffusion_dit_v3_7*`
  - 泛化实验线：`data/outputs/v37v*`
  - 精修线：`data/outputs/v3_7_ft_*`, `data/outputs/btcv_diffusion_dit_v3_7_12*`

