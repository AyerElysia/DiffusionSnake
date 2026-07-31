# BTCV V3.4-FM RL long10k run report

## Run purpose and config
- Purpose: monitor the in-progress BTCV V3.4-FM GRPO pure-RL long run to the true target of step **10000**.
- Config: `configs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k.yaml`
- Python env: `/home/medteam/miniconda3/envs/snake1/bin/python` (`snake1`)
- Output dir: `data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k`
- JSONL log: `data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k/posttrain_grpo/logs.jsonl`
- Health log: `data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k/posttrain_grpo/health_monitor_20260507.log`
- Resume log: `data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k/train_resume_20260507_1706.log`

## Final outcome
- Status: **incomplete**
- Final step reached: **5320 / 10000**
- Final checkpoint present: **no** (`data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k/checkpoints/final_step_10000.pt`)
- Latest log timestamp: **2026-05-09T14:09:10.617885**
- Parsed report generation time: **2026-05-09 14:09:11**

## Resume method
- Resume path: `data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k/checkpoints/latest.pt`
- Stateful resume flag: `GRPO_RESUME_STATE=1`
- Target override: `GRPO_TRAIN_STEPS=10000`
- The active recovery path preserved the already-added true stateful resume behavior in `grpo_train.py`; it was **not** reverted.
- The watchdog selected GPUs by physical GPU UUID, set `CFG_FILE` before project imports, and relaunched from the latest checkpoint when necessary.

## Replay / duplicate-step caveat
- Before this safe resume, `logs.jsonl` already extended to approximately **step 2158** while the valid resumable `latest.pt` checkpoint was still at **step 2000**.
- Because the resumed run intentionally restarted from checkpoint state rather than trusting unmatched JSONL tail rows, steps **2001+** were replayed.
- This means duplicate step numbers in the resumed region are **expected** and should not be mistaken for corruption.
- Parsed JSONL summary:
  - total rows: **5758**
  - unique steps: **5320**
  - duplicate rows: **438**
  - observed step range: **1 -> 5320**

## Restart and health history
- Progress events recorded in health log: **1259**
- Launch events recorded: **16**
- Restart events recorded: **8**
- Recent health/relaunch lines:
- [2026-05-07 21:03:11] Launching trainer on GPU 2 from checkpoint /home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k/checkpoints/latest.pt
- [2026-05-07 21:03:11] Spawned trainer PID 3114731
- [2026-05-07 21:09:42] Launching trainer on GPU 1 from checkpoint /home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k/checkpoints/latest.pt
- [2026-05-07 21:09:42] Spawned trainer PID 3135956
- [2026-05-07 20:23:06] Trainer process missing before target completion at step=2504. Restarting.
- [2026-05-07 20:29:37] Trainer process missing before target completion at step=2516. Restarting.
- [2026-05-07 20:32:07] Trainer process missing before target completion at step=2516. Restarting.
- [2026-05-07 20:36:26] Trainer process missing before target completion at step=2506. Restarting.
- [2026-05-07 20:40:09] Trainer process missing before target completion at step=2511. Restarting.
- [2026-05-07 20:56:40] Trainer process missing before target completion at step=2527. Restarting.
- [2026-05-07 21:03:11] Trainer process missing before target completion at step=2508. Restarting.
- [2026-05-07 21:09:42] Trainer process missing before target completion at step=2505. Restarting.

## GPU binding quirk
- During live monitoring, the trainer process environment reported `CUDA_VISIBLE_DEVICES=GPU-edd1898e-07b1-e51a-28fa-99cca5ef9f6a`.
- At the same time, `nvidia-smi` attributed the actual active memory to physical GPU UUID `GPU-00afd58f-4151-77d1-6bce-cb95c10ab985` (GPU index **2**).
- Training nevertheless remained healthy and continued making progress, so this was documented as an operational quirk rather than used as a restart trigger.

## Quantitative signals visible from logs
- Last logged step metrics (step **5320**):
  - `reward_mean`: **0.116080**
  - `reward_std`: **0.129816**
  - `final_score_mean`: **0.760248**
  - `kl_loss`: **0.001736**
  - `approx_kl`: **0.000000**
  - `loss`: **0.000017**
  - `diff_loss_scaled`: **0.000000**
- Last 100 unique-step averages:
  - mean `reward_mean`: **0.118149**
  - mean `final_score_mean`: **0.745355**
  - mean `kl_loss`: **0.001873**
  - mean `approx_kl`: **0.000000**
  - mean `loss`: **0.000019**
- `diff_loss_scaled` was exactly `0.0` on **100 / 100** of the last 100 unique steps, consistent with the intended pure-RL regime.
- Latest logged best-so-far values:
  - `reward_best`: **0.141066**
  - `final_score_best`: **0.785234**

## Checkpoints and timestamps
| File | Modified | Size (bytes) |
| --- | --- | ---: |
| `latest.pt` | 2026-05-09 13:03:31 | 228436237 |

## Full-eval gain check
- Latest available long10k full eval: `visual/v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k_eval_latest/v3_7_full_test_iou_20260509_130900.json`
- Evaluated checkpoint path: `data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k/checkpoints/latest.pt`
- Eval config: `configs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_last300_eval_gpu3.yaml` | ODE steps: `10`
- Evaluated saved-checkpoint step: `5250` | checkpoint mtime: `2026-05-09 13:03:31`
- Note: this section reflects the latest **saved** checkpoint that has been fully evaluated, which can lag the live JSONL step (`5320`) while training is still running.

| Metric | Base | `last300@300` | Current long10k saved ckpt | vs Base | vs `last300@300` |
| --- | ---: | ---: | ---: | ---: | ---: |
| IoU | 0.892484 | 0.894166 | 0.857320 | -0.035164 | -0.036847 |
| Dice | 0.941425 | 0.942498 | 0.921400 | -0.020025 | -0.021098 |
| mBoundF | 0.775513 | 0.777239 | 0.711979 | -0.063535 | -0.065260 |

### Saved-checkpoint eval trend
| Saved ckpt step | Eval JSON | IoU | Dice | mBoundF | vs Base IoU | vs Base mBoundF |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4000 | `v3_7_full_test_iou_20260508_182235.json` | 0.855642 | 0.920346 | 0.709253 | -0.036841 | -0.066260 |
| 4250 | `v3_7_full_test_iou_20260508_221010.json` | 0.856091 | 0.920641 | 0.709798 | -0.036393 | -0.065715 |
| 4500 | `v3_7_full_test_iou_20260509_015605.json` | 0.857958 | 0.921764 | 0.713041 | -0.034526 | -0.062473 |
| 4750 | `v3_7_full_test_iou_20260509_054206.json` | 0.851847 | 0.918069 | 0.703349 | -0.040637 | -0.072164 |
| 5000 | `v3_7_full_test_iou_20260509_092804.json` | 0.851792 | 0.918044 | 0.703339 | -0.040692 | -0.072174 |
| 5250 | `v3_7_full_test_iou_20260509_130900.json` | 0.857320 | 0.921400 | 0.711979 | -0.035164 | -0.063535 |

- Latest trend `5000 -> 5250`: IoU +0.005528, Dice +0.003356, mBoundF +0.008639.

## Resume-log tail
- [GRPO] 步骤 5120 总损失=0.0000 GRPO损失=0.0000 奖励均值=0.1371 奖励标准差=0.1264
- [GRPO] 步骤 5140 总损失=0.0000 GRPO损失=0.0000 奖励均值=0.0996 奖励标准差=0.1221
- [GRPO] 步骤 5160 总损失=0.0000 GRPO损失=0.0000 奖励均值=0.1095 奖励标准差=0.1189
- [GRPO] 步骤 5180 总损失=0.0000 GRPO损失=0.0000 奖励均值=0.1269 奖励标准差=0.1249
- [GRPO] 步骤 5200 总损失=0.0000 GRPO损失=0.0000 奖励均值=0.0882 奖励标准差=0.1318
- [GRPO] 步骤 5220 总损失=0.0000 GRPO损失=0.0000 奖励均值=0.1469 奖励标准差=0.1402
- [GRPO] 步骤 5240 总损失=0.0000 GRPO损失=0.0000 奖励均值=0.0845 奖励标准差=0.1147
- [GRPO] checkpoint saved: /home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k/checkpoints/latest.pt
- [GRPO] 步骤 5260 总损失=0.0000 GRPO损失=0.0000 奖励均值=0.1234 奖励标准差=0.1202
- [GRPO] 步骤 5280 总损失=0.0000 GRPO损失=0.0000 奖励均值=0.1213 奖励标准差=0.1146
- [GRPO] 步骤 5300 总损失=0.0000 GRPO损失=0.0000 奖励均值=0.1177 奖励标准差=0.1258
- [GRPO] 步骤 5320 总损失=0.0000 GRPO损失=0.0000 奖励均值=0.1161 奖励标准差=0.1298

## Key artifacts
- Final report: `/home/medteam/Zhrch/DiffusionSnake-12-30/report/BTCV_V3_4_FM_RL_LONG10K_RUN_REPORT_20260507.md`
- Final checkpoint directory: `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k/checkpoints`
- Health log: `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k/posttrain_grpo/health_monitor_20260507.log`
- JSONL log: `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k/posttrain_grpo/logs.jsonl`

## Recommendation
- Treat `final_step_10000.pt` (or the terminal completed checkpoint set at step 10000) as the canonical completed long-run artifact.
- Preserve the duplicate-step replay caveat whenever comparing JSONL counts against checkpoint step numbers or downstream evaluation summaries.

