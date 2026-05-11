# V4.1 机制消融实验记录

## 目的

验证 V4.1 相对 V3.4-FM 的新增机制到底哪些有效，而不是只看整体 V4.1 分数。

## 固定条件

- 数据：只用 BTCV train 训练，不合并 val/test。
- 初始权重：统一从 `data/outputs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolos_detail_gpu4_reusemax/checkpoints/latest.pt` 开始。
- Backbone/特征：统一使用 YOLO-s + P3/detail，作为已验证有效的基础。
- 训练长度：3000 epoch。
- Batch size：32。
- 推理评价：训练结束后统一跑 `scripts/eval_v37_full_iou.py`，统计 IoU、Dice、mBoundF。

## 实验组

| 组别 | 作用 |
|---|---|
| `control` | V3.4-FM + YOLO-s + P3/detail，作为对照 |
| `full` | V4.1 全机制 |
| `no_delta` | 去掉 per-point delta |
| `no_curv` | 去掉 curvature reweight |
| `no_small` | 去掉 small-displacement training |
| `delta_only` | 只加 per-point delta |
| `curv_only` | 只加 curvature reweight |
| `small_only` | 只加 small-displacement training |

## 当前启动状态

- 已启动：`control`、`full`、`no_delta`、`no_curv`。
- 等待启动：`no_small`、`delta_only`、`curv_only`、`small_only`，等待 0、1、3、5 号卡上的 V5.2 稳定性验证释放后自动启动。

## 输出位置

训练脚本实际按配置文件名保存 checkpoint：

- `data/outputs/btcv_ablate_v41_00_v34_detail_control`
- `data/outputs/btcv_ablate_v41_01_full`
- `data/outputs/btcv_ablate_v41_02_no_delta`
- `data/outputs/btcv_ablate_v41_03_no_curv`
- `data/outputs/btcv_ablate_v41_04_no_small_disp`
- `data/outputs/btcv_ablate_v41_05_delta_only`
- `data/outputs/btcv_ablate_v41_06_curv_only`
- `data/outputs/btcv_ablate_v41_07_small_only`

控制台日志另存于：

- `data/outputs/v4_1_mechanism_ablation/*/train_*.log`

## 汇总命令

```bash
source /home/medteam/miniconda3/etc/profile.d/conda.sh
conda activate snake1
cd /home/medteam/Zhrch/DiffusionSnake-12-30
python scripts/summarize_v41_ablation.py
```
