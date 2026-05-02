# Post-Training Diagnosis

Date: 2026-05-01

## Verdict

The current `btcv_diffusion_dit_v3_1_fm_posttrain` line is **not a valid GRPO / RL post-training result**.

What exists today is:

- two short test runs that each ended at `step = 1`
- two saved output models under `data/outputs/grpo_t/`
- one overwritten visualization file under `visual/grpo_train/`

What does **not** exist today is:

- a dedicated post-training checkpoint lineage under `data/outputs/btcv_diffusion_dit_v3_1_fm_posttrain/checkpoints/`
- evidence that GRPO policy loss was actually active
- evidence that the two saved models represent a completed or even minimally credible post-training experiment

Recommended decision: **stop treating the current post-training line as an active experiment result**. Keep the files only as debugging artifacts.

## Hard Facts

### 1. The post-training config has no local checkpoint lineage

Config:

- `configs/btcv_diffusion_dit_v3_1_fm_posttrain.yaml`

Observed outputs:

- `data/outputs/btcv_diffusion_dit_v3_1_fm_posttrain/posttrain_grpo/logs.jsonl`
- `data/outputs/grpo_t/model_grpo.pth`
- `data/outputs/grpo_t/model_grpo_20260430_165128.pth`

Missing:

- `data/outputs/btcv_diffusion_dit_v3_1_fm_posttrain/checkpoints/latest.pt`

This matters because `grpo_train.py` resolves its base checkpoint from the nearest existing candidate. In practice, that means it falls back to:

- `data/outputs/btcv_diffusion_dit_v3_1/checkpoints/latest.pt`

The base checkpoint exists and reports:

- `step = 9360`
- `epoch = 780`

So the current post-training line is bootstrapping from the old `v3_1` model, not from a dedicated post-training lineage.

### 2. Both recorded runs ended after one optimization step

`data/outputs/btcv_diffusion_dit_v3_1_fm_posttrain/posttrain_grpo/logs.jsonl` contains exactly two records:

- run 1: `step = 1`, `loss = 2.8975992`, `grpo_loss = 0.0`
- run 2: `step = 1`, `loss = 2.1959145`, `grpo_loss = 0.0`

Shared characteristics:

- `grpo_steps = 4`
- `grpo_k = 1`
- `grpo_loss = 0.0`

This is not a partial long run. It is two separate single-step runs.

### 3. The saved models are only tiny perturbations of the base checkpoint

Compared against `data/outputs/btcv_diffusion_dit_v3_1/checkpoints/latest.pt`:

- both saved post-train models changed `127` tensors
- `max_abs` parameter delta is about `5.049e-05`
- total L1 parameter delta is about `559.65` and `565.21`

Compared against each other:

- `126` tensors differ
- `max_abs` between the two saved models is about `1.0002e-04`

Interpretation:

- they are not identical copies
- they do reflect one optimizer step
- but they are still extremely close to the base model
- they should be treated as one-step debug artifacts, not trained variants

### 4. The visualization file is consistent with short test runs

Observed file:

- `visual/grpo_train/vis_affine_final.png`

Timestamp:

- `2026-04-30 16:51:28`

This matches the second run window and supports the conclusion that the script completed its final-save path, not that it crashed.

## Why GRPO Was Not Actually Active

Relevant files:

- `grpo_train.py`
- `lib/train/trainers/diffusion_grpo_trainer.py`

### Key behavior

`grpo_train.py` forces:

- `cfg.use_grpo = True`
- `cfg.use_flow_matching = True` for this V3.1-FM post-training path

However, inside `DiffusionGRPONetworkWrapper`:

- `flow_posttrain_mode = bool(cfg.use_flow_matching)`
- `allow_grpo = raw_use_grpo and (not disable_grpo) and (not flow_posttrain_mode)`

That means:

- when Flow Matching is enabled, GRPO is explicitly disabled

So the wrapper enters the Flow Matching branch, not the GRPO policy-gradient branch.

### Observable consequence

The wrapper still keeps reward logging on:

- `enable_reward_logging = raw_use_grpo and (not disable_grpo)`

Therefore the logs show:

- nonzero `reward_mean`
- nonzero `reward_std`
- but always `grpo_loss = 0.0`

This is the central failure mode:

- the run looks like "RL post-training" from the config and log names
- but the actual optimization path is still standard Flow Matching loss, with reward only being observed and logged

In other words:

- **the current run is named like GRPO post-training**
- **but it does not optimize a GRPO objective**

## Meaning of the Two Existing Output Models

These files do have narrow debugging value:

- they prove the script can load a base checkpoint
- they prove the wrapper can complete one step and save a model
- they prove the reward logging path executes in FM mode

They do **not** have experiment value:

- they do not prove RL post-training works
- they do not prove GRPO improves the model
- they do not represent a reproducible trained checkpoint line
- they are too close to the base model to justify being treated as a new result

Recommended label for these files:

- `debug artifacts`

Not recommended labels:

- `posttrained model`
- `GRPO result`
- `final checkpoint`

## Decision

### Recommended decision now

Stop the current line as an experiment track.

Use the current materials only for diagnosis, not for reporting performance or selecting a model.

### Recommended next move

Prioritize the main training line first.

If post-training is revisited later, do it only after choosing one of these two paths:

1. Minimal validation path
   - keep the current framework
   - rename it honestly as FM fine-tuning with reward logging
   - do not call it GRPO
   - run a small controlled validation only to test whether one-step / few-step FM fine-tuning helps at all

2. Correct rebuild path
   - implement a real FM-compatible GRPO / RL objective
   - remove the current "GRPO requested but disabled by FM mode" ambiguity
   - create a dedicated checkpoint lineage and evaluation protocol before spending more GPU time

### Recommended default

Choose path 1 only if the goal is quick diagnosis.

Choose path 2 only if RL post-training is strategically important.

Given the current state of the project, the default recommendation is:

- **do not spend more compute on the current post-training script as-is**
- **finish the main training line first**

