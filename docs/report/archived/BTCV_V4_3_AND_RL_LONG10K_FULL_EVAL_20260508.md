# BTCV V4.3 与最新 RL 模型全量评估报告

本报告总结两条线：

1. **V4.3** 的全量评估与可视化。
2. **最新 RL 产生的 long10k 模型** 的全量评估与可视化。

## 1. 结论先行

- **V4.3 明显强于最新 RL 模型**，尤其在 mBoundF 上差距更大。
- V4.3 的可视化也更稳定，和 V4.1 非常接近。
- 最新 RL 模型目前还没有超过 V4.3，反而更像是“训练继续推进但泛化暂时回落”的状态。

---

## 2. V4.3 全量评估

### 2.1 评估口径

- 数据集：`BtcvVal`
- 样本数：150
- 推理步数：ODE=10
- 评估输出：
  - `visual/v4_compare_20260507/v4_3/v3_7_full_test_iou_20260507_014943.json`
  - `visual/v4_compare_20260507/v4_3/summary_rows_20260507_014943.json`

### 2.2 指标

| Metric | V4.3 |
| --- | ---: |
| IoU sample avg | 0.903157 |
| IoU contour avg | 0.900154 |
| Dice sample avg | 0.947951 |
| mBoundF sample avg | 0.793894 |
| IoU std | 0.020343 |
| failed samples | 0 |

### 2.3 可视化

V4.3 的对比图在：

- `visual/v4_compare_20260507/v4_compare_grid_49_64_71.png`

这张图里能直接看到：

- **V4.1 / V4.3** 基本处于同一档，边界贴合最好。
- **V4.2** 明显更弱。
- V4.3 相比 V4.0/V4.2 更稳，边界更紧。

---

## 3. 最新 RL 模型全量评估

这里的“最新 RL 模型”指当前 long10k 训练线里最新保存的 `latest.pt`。

### 3.1 评估口径

- checkpoint：`data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k/checkpoints/latest.pt`
- 评估配置：`configs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_last300_eval_gpu3.yaml`
- 数据集：`BtcvVal`
- 样本数：150
- 推理步数：ODE=10
- 评估输出：
  - `visual/v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k_eval_latest_vis_20260508/v3_7_full_test_iou_20260508_151156.json`
  - `visual/v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k_eval_latest_vis_20260508/summary_rows_20260508_151156.json`

### 3.2 指标

| Metric | RL latest |
| --- | ---: |
| IoU sample avg | 0.860709 |
| IoU contour avg | 0.857544 |
| Dice sample avg | 0.923268 |
| mBoundF sample avg | 0.718246 |
| IoU std | 0.027149 |
| failed samples | 0 |

### 3.3 可视化

这次全量评估已经生成逐样本可视化：

- `visual/v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k_eval_latest_vis_20260508/per_sample/idx_000/overlay.png`
- `visual/v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k_eval_latest_vis_20260508/per_sample/idx_001/overlay.png`
- ...

也就是说，这条线现在既有全量指标，也有全量可视化目录。

---

## 4. 对比

| Metric | V4.3 | RL latest | Delta |
| --- | ---: | ---: | ---: |
| IoU sample avg | 0.903157 | 0.860709 | +0.042449 |
| Dice sample avg | 0.947951 | 0.923268 | +0.024683 |
| mBoundF sample avg | 0.793894 | 0.718246 | +0.075648 |
| IoU std | 0.020343 | 0.027149 | -0.006806 |

## 5. 结论

1. **V4.3 现在仍是更强的版本**，尤其是边界指标优势明显。
2. **最新 RL 模型没有把 long10k 继续推高**，目前的 full eval 结果明显落后于 V4.3。
3. 如果后面继续推进 RL，应该优先检查：
   - reward 目标是否过散
   - long10k 后段是否出现过拟合
   - 当前 checkpoint 是否已经偏离最优区域

