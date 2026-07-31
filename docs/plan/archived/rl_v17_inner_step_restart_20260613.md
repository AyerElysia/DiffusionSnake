# RL V17 Inner-Step Flow-GRPO Restart

Date: 2026-06-13

## Goal

Restart the route where diffusion/flow inner steps are the RL trajectory, but avoid the historical failure modes:

- no best-of-k distillation path;
- no latent ranker;
- no unlogged x0 search as the main reward source;
- only actions with old logprob can update the model;
- compare sampled trajectories against the deterministic policy output.

## Code Fix

`FlowMatchingEvolution.sample_with_logprob()` now clamps the denormalized displacement with `clamp_pred_disp`, matching the normal `sample_disp()` inference path.

Without this, sampled RL rollouts and deterministic baselines were not in the same action space.

## Probe Results

All probes used V3.4-FM, `rollout_noise_scale=0.0`, no distillation, and K=16.

| Probe | Window | std | Result |
|---|---:|---:|---|
| late2 | `[8,10]` | `0.05` | damaged contours, `gate_active_frac=0.0`, no update |
| full10 | `[0,10]` | `0.01` | weak sparse positives, `gate_active_frac=0.143` |
| mid3 | `[4,7]` | `0.02` | usable signal, `gate_active_frac=1.0`, nonzero grad |

Selected configuration: middle 3 inner steps, Gaussian transition, std `0.02`.

## Active Run

Config:

`configs/btcv_v3_4_fm_rl_v17_mid3_stepgrpo_gpu2.yaml`

Run script:

`scripts/run_v3_4_rl_v17_mid3_stepgrpo_gpu2.sh`

Output:

`data/outputs/btcv_v3_4_fm_rl_v17_mid3_stepgrpo_gpu2/`

tmux session:

`v17_mid3_stepgrpo_gpu2`

Initial health:

- checkpoint load ratio: `100%`
- fixed eval baseline: IoU `0.884210`
- step 1: `step_log_count_mean=9`, `gate_active_frac=1.0`, `grad_norm=0.0156`
- steps 2-4: continued nonzero policy gradients

## Next Decision Points

1. Inspect step 10/50/100 fixed eval trend.
2. If stable, run full 150-sample evaluation at step 50 and 100.
3. If KL remains near zero while gradients are stable, try lr or PPO clip increase only after the first full-eval checkpoint.
