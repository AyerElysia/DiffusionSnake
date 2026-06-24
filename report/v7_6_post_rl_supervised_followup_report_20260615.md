# V7.6 Post-RL Supervised Follow-up Report

Date: 2026-06-15  
Repo: `/home/medteam/Zhrch/DiffusionSnake-12-30`  
Base checkpoint: `data/outputs/1232_final_v7_6_conditional_residual_explorer_long_gpu3/checkpoints/best_iou.pt`

## 1. Background

The previous RL post-training campaign showed that additional reward engineering and policy-gradient variants were no longer the best path forward. The practical conclusion was to convert the useful signals discovered by RL into supervised or structural training signals:

- Ensemble expectation distillation: learn the 8-seed average output as a single-pass student.
- Displacement gate: learn when the model should move less or stay close to init.
- Supervised continue control: measure how much of any gain is just extra supervised training.
- Locate-feature continuation: continue the V10/F-series locate line as a separate tail-contour/init investigation.

The main working hypothesis was:

> Dense supervised signals should be used wherever GT or offline teacher targets are available; policy-gradient signals should not be used for these remaining pools unless no supervised target exists.

## 2. Code Changes

### 2.1 Displacement Gate in FlowMatchingEvolution

Touched file:

- `lib/networks/diffusion/flow_matching_evolution.py`

Implemented an optional supervised displacement gate inside the flow-matching evolution module.

Main behavior:

- Adds a lightweight `disp_gate_head`.
- Predicts a scalar gate per contour.
- In inference, applies:

```text
final_disp = predicted_disp * gate
```

- Gate values near `0` preserve the init contour.
- Gate values near `1` keep the original model behavior.
- The gate is initialized close to legacy behavior, so enabling the module does not start from an aggressive suppression policy.

Training-side gate target:

- Uses the predicted displacement and GT residual to estimate the ideal amount of movement.
- Optimizes a supervised Smooth L1 gate loss.
- Supports gate-only training by freezing non-gate parameters.

Relevant config switches:

```yaml
flow_use_disp_gate: false
flow_disp_gate_apply_inference: true
flow_disp_gate_apply_training_pred: false
flow_disp_gate_loss_weight: 0.0
flow_disp_gate_hidden_dim: 128
flow_disp_gate_init_bias: 4.0
flow_disp_gate_detach_input: true
```

### 2.2 Post-training Scripts

Added/used scripts:

- `scripts/posttrain_common.py`
- `scripts/export_ensemble_teacher.py`
- `scripts/train_ensemble_distill.py`
- `scripts/train_supervised_posttrain.py`
- `scripts/run_v7_6_export_ensemble_teacher_gpu3.sh`
- `scripts/run_v7_6_ensemble_distill_gpu3.sh`
- `scripts/run_v7_6_gate_posttrain_gpu3.sh`
- `scripts/run_v7_6_supervised_continue_gpu3.sh`

Script responsibilities:

| Script | Purpose |
|---|---|
| `posttrain_common.py` | Shared config/model/dataset/checkpoint helpers for post-training scripts. |
| `export_ensemble_teacher.py` | Offline teacher export from 8 stochastic inference seeds. |
| `train_ensemble_distill.py` | Student fine-tuning against ensemble teacher targets plus optional GT loss. |
| `train_supervised_posttrain.py` | Two modes: full supervised continue and gate-only supervised training. |
| `run_v7_6_*.sh` | Reproducible launch wrappers with conda `snake1` activation. |

Important operational note:

- Long jobs were run through `tmux`. Earlier `nohup ... &` style was avoided because it was less reliable in this environment.
- The run scripts activate conda env `snake1`.

## 3. Validation / Smoke Tests

Completed before full jobs:

- Python compile checks passed for the new scripts and touched files.
- `--help` checks passed in conda env `snake1`.
- Smoke runs passed:
  - teacher export with `MAX_SAMPLES=1`
  - gate training with `STEPS=1`
  - supervised continue with `STEPS=1`
  - distill training with `STEPS=1`

This established that the added scripts were importable, could load the V7.6 checkpoint, could construct datasets, and could execute at least one forward/backward step.

## 4. Experiment A: Ensemble Expectation Distillation

### 4.1 Teacher Export

Teacher export completed successfully.

Output:

```text
data/teachers/v7_6_train_noise05_seed8_teacher.json
```

Teacher generation settings:

| Item | Value |
|---|---|
| Base checkpoint | `1232_final_v7_6_conditional_residual_explorer_long_gpu3/checkpoints/best_iou.pt` |
| Dataset | train split, 1056 samples |
| Seeds | `101,202,303,404,505,606,707,808` |
| Inference noise | `0.5` |
| Teacher target | per-point average contour over 8 seeds |
| Output size | about 434 MB |
| Log | `logs/v7_6_export_ensemble_teacher_gpu7_20260614.log` |

Conclusion:

- The teacher-generation pipeline is usable.
- The expected offline teacher artifact exists and is no longer a blocker for distillation.

### 4.2 Distillation Training

Run:

```text
data/outputs/v7_6_ensemble_distill_noise05_seed8_gpu3
```

Checkpoint outputs:

```text
data/outputs/v7_6_ensemble_distill_noise05_seed8_gpu3/checkpoints/step200.pt
data/outputs/v7_6_ensemble_distill_noise05_seed8_gpu3/checkpoints/step400.pt
...
data/outputs/v7_6_ensemble_distill_noise05_seed8_gpu3/checkpoints/step2000.pt
data/outputs/v7_6_ensemble_distill_noise05_seed8_gpu3/checkpoints/latest.pt
```

Training settings:

| Item | Value |
|---|---|
| Steps | 2000 |
| LR | `1e-8` |
| Teacher weight | `1.0` |
| GT weight | `0.25` |
| Loss mode | low-memory flow-level loss |
| GPU | 3 |
| Log | `logs/v7_6_ensemble_distill_noise05_seed8_gpu3_20260615.log` |

Observed losses:

| Step | Total loss | Teacher loss | GT loss |
|---:|---:|---:|---:|
| 200 | 0.967209 | 0.773707 | 0.774008 |
| 1000 | 0.931082 | 0.744889 | 0.744774 |
| 1600 | 1.107309 | 0.885826 | 0.885933 |
| 2000 | 0.901634 | 0.721241 | 0.721575 |

Interpretation:

- Distillation ran to completion.
- The teacher-vs-GT loss values are almost identical throughout training.
- This is a warning sign: under the current flow-level target formulation, the ensemble teacher may not be sufficiently differentiated from the GT residual target, or the loss scale may hide the teacher's useful final-contour advantage.
- This does not prove the distillation failed, but it lowers confidence that the current loss is the right way to capture the ensemble expectation gain.

Current status:

- Full test evaluation for `step2000` has been launched but was still running when this report was written.

## 5. Experiment B: Supervised Displacement Gate

Run:

```text
data/outputs/v7_6_disp_gate_head_gpu2
```

Training settings:

| Item | Value |
|---|---|
| Mode | gate-only |
| Trainable params | 25,473 |
| Steps | 1000 |
| LR | `3e-5` |
| Gate loss weight | `1.0` |
| GPU | 2 |
| Log | `logs/v7_6_disp_gate_head_gpu2_20260615.log` |

Checkpoint outputs:

```text
data/outputs/v7_6_disp_gate_head_gpu2/checkpoints/step100.pt
...
data/outputs/v7_6_disp_gate_head_gpu2/checkpoints/step1000.pt
data/outputs/v7_6_disp_gate_head_gpu2/checkpoints/latest.pt
```

Observed losses:

| Step | Total loss | Gate loss |
|---:|---:|---:|
| 20 | 0.202588 | 0.201379 |
| 100 | 0.184334 | 0.183361 |
| 300 | 0.144578 | 0.142803 |
| 360 | 0.091333 | 0.089292 |
| 500 | 0.089821 | 0.087424 |
| 800 | 0.091470 | 0.086960 |
| 1000 | 0.090188 | 0.088807 |

Interpretation:

- The gate objective is learnable.
- Loss drops from about `0.20` to about `0.09`, then plateaus.
- This supports the hypothesis that "learn when to move less" is a real supervised signal rather than an RL-only policy choice.
- The plateau suggests the current lightweight scalar gate may capture only part of the needed behavior, but the signal is not random or degenerate.

Current status:

- Full test evaluation for `step1000` has been launched with:

```text
flow_use_disp_gate=True
flow_disp_gate_apply_inference=True
infer_noise_scale=0.5
infer_avg_samples=1
EVAL_ABLATION_MODE=gt_init
```

- The evaluation was still running when this report was written.

## 6. Experiment D: Supervised Continue Control

Run:

```text
data/outputs/v7_6_supervised_continue_control_gpu0
```

Training settings:

| Item | Value |
|---|---|
| Mode | full supervised continue |
| Steps | 1000 |
| LR | `1e-8` |
| GPU | 0 |
| Log | `logs/v7_6_supervised_continue_control_gpu0_20260615.log` |

Checkpoint outputs:

```text
data/outputs/v7_6_supervised_continue_control_gpu0/checkpoints/step100.pt
...
data/outputs/v7_6_supervised_continue_control_gpu0/checkpoints/step1000.pt
data/outputs/v7_6_supervised_continue_control_gpu0/checkpoints/latest.pt
```

Observed losses:

| Step | Loss |
|---:|---:|
| 100 | 0.001957 |
| 300 | 0.002059 |
| 500 | 0.000973 |
| 700 | 0.000790 |
| 900 | 0.001366 |
| 1000 | 0.003278 |

Interpretation:

- The supervised continuation run is stable.
- There was no loss explosion or obvious training failure.
- This run is important as a control: if it improves as much as distillation, then the distillation gain may mostly be extra training rather than teacher-specific information.

Current status:

- Full test evaluation for `step1000` has been launched but was still running when this report was written.

## 7. F4 Locate18+26 Continuation

Separate from the V7.6 post-training work, two F4 locate-feature continuation runs were monitored.

Runs:

```text
data/outputs/f4_locate18_26_from58000_lr1e5_gpu0
data/outputs/f4_locate18_26_from58000_lr5e6_gpu2
```

Settings:

| Item | Value |
|---|---|
| Resume checkpoint | `data/outputs/f2_locate_feat_replace_gpu0/checkpoints/step_58000.pt` |
| Locate feature keys | `layer_18`, `layer_26` |
| LR candidates | `1e-5`, `5e-6` |
| Planned end step | 70000 |
| Eval mode | `gt_init` |
| Eval script | `scripts/f4_eval_sweep_locate18_26.sh` |

### 7.1 Training Status

Both training runs reached about `63100` steps and then failed with CUDA OOM.

Observed failures:

- `lr1e-5` failed around `step 63100`.
- `lr5e-6` failed around `step 63150`.
- The eval sweep is currently waiting for `step_64000.pt`, which will not appear unless the training is resumed.

OOM examples:

```text
RuntimeError: CUDA out of memory. Tried to allocate 84.00 MiB ...
RuntimeError: CUDA out of memory. Tried to allocate 40.00 MiB ...
```

### 7.2 Available Evaluation Results

Completed F4 full-test evals:

| Run | Step | mean_iou_sample_avg | mean_iou_contour_avg | mean_dice_sample_avg | failed_samples |
|---|---:|---:|---:|---:|---:|
| lr1e-5 | 60000 | 0.830510 | 0.828503 | 0.905152 | 0 |
| lr1e-5 | 62000 | 0.836181 | 0.834480 | 0.908933 | 0 |
| lr5e-6 | 60000 | 0.835750 | 0.833990 | 0.908415 | 0 |
| lr5e-6 | 62000 | 0.838420 | 0.836570 | 0.910035 | 0 |

Interpretation:

- `5e-6` is better than `1e-5` at both evaluated steps.
- Both LRs improve from `60000` to `62000`.
- Absolute numbers are not competitive with the strongest V7.6 line, so this is not yet a breakthrough.
- The run cannot continue to `64000+` without addressing OOM or resuming with a safer memory configuration.

## 8. Evaluation Status for V7.6 Post-training

At the time of this report, V7.6 post-training full evaluations were launched but not finished.

### 8.1 Running Evaluation Jobs

Session:

```text
v7_6_posttrain_eval_gpu3
```

Runs sequentially:

1. Baseline V7.6 best checkpoint.
2. Supervised continue `step1000`.
3. Ensemble distill `step2000`.

Session:

```text
v7_6_gate_eval_gpu7
```

Runs:

1. Gate `step1000`.

Evaluation settings:

```text
EVAL_ABLATION_MODE=gt_init
SAVE_VISUALS=0
ODE_STEPS=10
infer_noise_scale=0.5
infer_avg_samples=1
```

Gate-specific overrides:

```text
flow_use_disp_gate=True
flow_disp_gate_apply_inference=True
flow_disp_gate_loss_weight=0.0
```

Planned output directories:

```text
visual/v7_6_baseline_best_noise05_single_gtinit_eval
visual/v7_6_supctrl_step1000_noise05_single_gtinit_eval
visual/v7_6_distill_step2000_noise05_single_gtinit_eval
visual/v7_6_gate_step1000_noise05_single_gtinit_eval
```

Planned logs:

```text
logs/v7_6_baseline_best_noise05_single_gtinit_eval_20260615.log
logs/v7_6_supctrl_step1000_noise05_single_gtinit_eval_20260615.log
logs/v7_6_distill_step2000_noise05_single_gtinit_eval_20260615.log
logs/v7_6_gate_step1000_noise05_single_gtinit_eval_20260615.log
```

Current progress when checked:

- Baseline and gate evals had loaded checkpoints successfully.
- Both were processing the 177-sample validation/test split.
- No JSON summary had been emitted yet.

## 9. Current Conclusions

### 9.1 Conclusions With Evidence

1. The offline ensemble teacher pipeline works.

   The full 1056-sample teacher file was generated successfully.

2. The displacement gate signal is learnable.

   Gate loss dropped by more than 50% and stabilized. This supports continuing the gate direction, pending full-test IoU.

3. Supervised continue is stable.

   It ran to `step1000` without exploding. It is a valid control.

4. Current flow-level distillation loss does not clearly separate teacher from GT.

   Teacher and GT losses were nearly identical throughout the distillation run. This weakens the case for the current loss formulation.

5. F4 locate18+26 is not a clear win yet.

   Best evaluated F4 result so far is `0.838420` mean IoU sample average at `lr5e-6 step62000`, under `gt_init` eval. It is improving locally but not enough to call it a breakthrough.

### 9.2 Conclusions Not Yet Supported

The following should not be claimed until eval JSONs finish:

- Gate improves final IoU.
- Distillation improves final IoU.
- Supervised continue improves final IoU.
- Distillation beats supervised continue.
- Gate and distillation are additive.

## 10. Risks and Issues

### 10.1 Distillation Objective Risk

The current distill loss is low-memory flow-level regression. It may be too close to ordinary GT residual supervision.

If full-test eval is flat, likely next fixes are:

- Try rollout-level distillation despite higher memory cost.
- Distill final contours rather than intermediate flow target.
- Compare per-sample teacher improvement against student target distance to verify the teacher file contains the intended useful delta.

### 10.2 Gate Capacity Risk

The scalar gate is intentionally cheap. It may be too coarse if different contour regions require different movement strength.

Possible next versions:

- Per-contour scalar gate plus confidence calibration.
- Per-point gate.
- Class/scale-aware gate.
- Gate trained together with a small supervised fine-tune of the final head.

### 10.3 Evaluation Consistency Risk

Historical V7.6 numbers include several inference configurations:

- noise `1.0`, single
- noise `0.5`, single
- noise `0.5`, avg4
- noise `0.5`, avg8
- seed oracle

The currently launched post-training evals use:

```text
noise0.5 + single + gt_init
```

This is a good first comparison, but final decision-making should also compare the most relevant deployment setting, especially `noise0.5 + avg8` if compute allows.

### 10.4 F4 OOM

Both F4 locate runs OOMed around `63100`.

The eval sweep is still waiting for `step_64000.pt`, so it should either be stopped or the training should be resumed with safer memory settings.

## 11. Recommended Next Steps

1. Let the four V7.6 post-training full evals finish.

   Required outputs:

   - baseline best
   - supervised continue `step1000`
   - gate `step1000`
   - distill `step2000`

2. Build a direct comparison table.

   The first decision table should include:

   | Model | mean_iou_sample_avg | delta vs baseline | failed_samples |
   |---|---:|---:|---:|
   | V7.6 baseline | pending | 0 | pending |
   | supervised continue | pending | pending | pending |
   | gate | pending | pending | pending |
   | distill | pending | pending | pending |

3. Interpret results using the control.

   - If supervised continue and distill improve similarly: the gain is probably extra supervised training.
   - If distill beats supervised continue: teacher expectation has useful information.
   - If gate improves or preserves mean IoU while improving bad cases: combine gate with distill or continue.
   - If gate drops mean IoU: inspect whether the gate is suppressing necessary movement.

4. Stop or fix the F4 eval sweep.

   It is currently waiting for `64000` checkpoints that do not exist because both F4 trainings OOMed.

5. Do not restart RL variants unless a new structural target is introduced.

   Current evidence still favors supervised/structural methods over additional policy-gradient reward variants.

## 12. Useful Commands

Check running jobs:

```bash
tmux list-sessions | rg 'v7_6|f4_locate'
ps -eo pid,ppid,etime,stat,pcpu,pmem,cmd | rg 'eval_v37_full_iou|train_ensemble_distill|train_supervised_posttrain|f4_eval'
```

Tail V7.6 post-training eval logs:

```bash
tail -80 logs/v7_6_baseline_best_noise05_single_gtinit_eval_20260615.log
tail -80 logs/v7_6_supctrl_step1000_noise05_single_gtinit_eval_20260615.log
tail -80 logs/v7_6_distill_step2000_noise05_single_gtinit_eval_20260615.log
tail -80 logs/v7_6_gate_step1000_noise05_single_gtinit_eval_20260615.log
```

Find V7.6 post-training eval JSONs:

```bash
find visual/v7_6_baseline_best_noise05_single_gtinit_eval \
     visual/v7_6_supctrl_step1000_noise05_single_gtinit_eval \
     visual/v7_6_distill_step2000_noise05_single_gtinit_eval \
     visual/v7_6_gate_step1000_noise05_single_gtinit_eval \
     -maxdepth 1 -type f -name 'v3_7_full_test_iou_*.json' | sort
```

Check F4 existing eval summaries:

```bash
rg -n 'mean_iou_sample_avg|mean_iou_contour_avg|mean_dice_sample_avg|failed_samples' \
  visual/f4_locate18_26_*_gtinit_eval/v3_7_full_test_iou_*.json
```

Check GPU usage:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

