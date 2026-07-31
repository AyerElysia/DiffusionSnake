# BTCV V4.1 最新模型全量评估报告

## 1. 结论先行

V4.1 最新 checkpoint（`latest.pt`）已经跑完 BTCV `BtcvVal` 全量评估和逐样本可视化。

本次结果：

- **IoU**: 0.903734
- **Dice**: 0.948228
- **mBoundF**: 0.796244
- **IoU Std**: 0.022099
- **failed samples**: 0

这版比之前记录的 V4.1 结果略有提升，尤其是 **mBoundF** 已经超过了此前 V4.3 的记录值。

---

## 2. 评估配置

- 配置：`configs/btcv_diffusion_dit_v4_1_fm_detail_curv_gpu4.yaml`
- checkpoint：`data/outputs/btcv_diffusion_dit_v4_1_fm_detail_curv_gpu4/checkpoints/latest.pt`
- 数据集：`BtcvVal`
- 样本数：150
- 推理步数：ODE=10
- 评估 seed：`20260508`

---

## 3. 评估结果

| Metric | Value |
| --- | ---: |
| IoU sample avg | 0.903734 |
| IoU contour avg | 0.900513 |
| Dice sample avg | 0.948228 |
| mBoundF sample avg | 0.796244 |
| IoU std | 0.022099 |
| failed samples | 0 |

评估输出文件：

- `visual/v4_1_fm_eval_latest_20260508/v3_7_full_test_iou_20260508_180320.json`
- `visual/v4_1_fm_eval_latest_20260508/summary_rows_20260508_180320.json`

---

## 4. 可视化

逐样本可视化目录：

- `visual/v4_1_fm_eval_latest_20260508/per_sample/`

该目录下每个样本都保存了 `overlay.png`，可直接查看初始化轮廓、GT 和预测轮廓的重叠效果。

---

## 5. 简短判断

V4.1 最新模型目前是 **稳定且很强** 的版本，边界质量已经非常好。

