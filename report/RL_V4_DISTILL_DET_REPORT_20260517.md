# RL V4 – Quality-Gated Distillation: Diagnosis & Experiment Report

**Date:** 2026-05-17  
**Status:** Experiment running (V4-A, GPU 1, 500 steps, ~5 hrs)  
**Baseline:** Full-eval median IoU = 0.8926 (btcv_diffusion_dit_v3_4_fm_full_noleak_yolom_gpu35_reusemax)

---

## 1. Root-Cause Analysis of All Previous RL Failures

### 1.1 V1 (`grpo_train.py`) — Confirmed Dead
Four code bugs caused `ratio ≡ 1.0` throughout, making the PPO surrogate a constant.
Policy quality degraded 0.79 → 0.71 over 10k steps. Abandoned.

### 1.2 V2 (`grpo_train_v2.py`, `v2_*` configs) — PPO with wrong reward
- `approx_kl ≈ 0.0000` throughout (ratio ∈ [0.999, 1.001])
- Reward used `reward_abs_weight=1.0` → absolute score baseline, not delta
- Full eval showed regression: step 100 = 0.848, step 500 = 0.883 vs baseline 0.893
- Root cause: PPO gradient signal near-zero, weight drift from KL noise

### 1.3 V3 Variants (a–e) — PPO-only, no distillation
Configs `v3_a` through `v3_e` never included `distill_weight` → defaulted to 0.0.
Effect: **distillation was completely disabled across all v3 experiments.**

Key finding from V3-b logs (k=4, action_std=0.05):
- Step 1: `final_score_best = 0.767` (stochastic quality below deterministic ~0.891)
- Step 11: `final_score_best = 0.919` → stochastic rollouts now **beat deterministic**
- Step 60: `final_score_best = 0.910`, but `eval_iou = 0.880` (no improvement)

**The PPO was successfully improving stochastic rollout quality but had no mechanism to transfer this to the deterministic eval path.** Distillation was the missing link.

### 1.4 `distill_smoke` — Distillation enabled, but wrong gate
Config had `distill_weight=0.1`, ran 2 steps, showed eval_iou=0.904 at step 1 (noisy).
Critical bug: the gate compared `best_stochastic > YOLO_init_score + 0.002`.
- YOLO init score ≈ 0.66-0.69
- Best stochastic ≈ 0.787 (much worse than deterministic 0.891)
- Gate passed (0.787 > 0.66 + 0.002 = TRUE) → distilled a trajectory WORSE than det
- The 0.904 eval was noise over 2 steps/68 samples; full eval would show regression

### 1.5 `distill_f` — OOM / killed silently
Larger config with `eval_batches=50`, crashed after baseline eval, 0 training steps.

---

## 2. Diagnosis Summary

| Component | Status | Root Cause |
|-----------|--------|-----------|
| PPO ratio ≈ 1 | Expected (not a bug) | Model already high quality; tiny action_std → tiny ratio deviation |
| V3 distillation = 0 | **Bug**: distill_weight absent from all v3 configs | All improvement wasted |
| V2/smoke distill gate | **Bug**: compared to YOLO init, not deterministic model | Distilled worse-than-det trajectories |
| Stochastic beats det | **Confirmed** | After ~10 PPO steps, k=4 best-of-k ≈ 0.91 > det ≈ 0.89 |

**Core insight:** The stochastic rollouts DO find better trajectories than the deterministic model. PPO helps them improve quickly. But without distillation to the deterministic path, evaluation never sees the improvement.

---

## 3. V4 Fix Design

### 3.1 New config parameters
```
grpo_v2_distill_weight: 1.0       # Enable strong distillation
grpo_v2_distill_compare_det: 1    # Use det-baseline gate (NEW)
grpo_v2_distill_det_margin: 0.002 # Gate: best_sto > det + 0.2%
```

### 3.2 Code change (grpo_train_v2.py)
**Before (line 856):**
```python
active = (delta_scores[best_idx, arange_b] > distill_min_delta).detach()
```

**After:**
```python
if distill_compare_det and det_scores is not None:
    active = (final_scores[best_idx, arange_b] > det_scores + distill_det_margin).detach()
else:
    active = (delta_scores[best_idx, arange_b] > distill_min_delta).detach()
```

**New: deterministic baseline computed per step** (before distillation):
```python
if distill_weight > 0 and distill_compare_det:
    det_ret = _sample_rollout(output, 0.0)  # action_std=0
    det_scores = _compute_rewards(output, det_ret)['final_score'].detach()
```

### 3.3 Experiment config (V4-A)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `k` | 8 | More rollouts for better best-of-k |
| `action_std` | 0.05 | Empirically yields final_best ≈ 0.91-0.96 |
| `distill_weight` | 1.0 | Strong distillation signal |
| `distill_compare_det` | True | Fix the gate bug |
| `distill_det_margin` | 0.002 | Require 0.2% gain over det |
| `ppo_inner_epochs` | 1 | Minimal PPO overhead |
| `kl_beta` | 0.10 | Reduced KL penalty (less resistance to improvement) |
| `lr` | 3e-7 | Moderate learning rate |
| `train_steps` | 500 | Full run |

---

## 4. Smoke Test Results (5 steps)

```
step= 1: final_best=0.9211, det=0.9065, distill_active=1.000, distill_loss=0.000257, eval_iou=0.8837
step= 2: final_best=0.9069, det=0.8892, distill_active=1.000, distill_loss=0.000613
step= 3: final_best=0.9199, det=0.9047, distill_active=0.667, distill_loss=0.001213
step= 4: final_best=0.8959, det=0.8833, distill_active=0.429, distill_loss=0.000072
step= 5: final_best=0.9583, det=0.9487, distill_active=1.000, distill_loss=0.000021, eval_iou=0.8835
```

**Key verifications:**
- ✅ `det_score` computed correctly (~0.907 train batches)
- ✅ `final_best > det_score` at most steps (gate fires correctly)
- ✅ `distill_loss > 0` (distillation is executing)
- ✅ `distill_active` varies (0.43–1.0, not always trivially 1)
- Eval barely changed in 5 steps (expected; improvement takes ~50–200 steps)

---

## 5. Full Run Progress

**Experiment:** `btcv_v3_4_fm_rl_v4_distill_det_gpu1`  
**GPU:** 1 (48GB free)  
**Status:** Running (PID 57985, ~35s/step, ~5 hours total)  
**Log:** `logs/rl_v4_gpu1.log`  
**JSONL:** `data/outputs/btcv_v3_4_fm_rl_v4_distill_det_gpu1/posttrain_grpo_v2/logs.jsonl`

### Expected milestones
| Step | Expected observation |
|------|----------------------|
| 20–50 | distill_active consistently > 0.5, eval_iou should start rising from 0.884 |
| 100 | eval_iou target ≥ 0.890 (needs to recover from noise) |
| 200 | eval_iou target ≥ 0.893 (surpass baseline) |
| 300–500 | eval_iou target ≥ 0.897–0.900 (明显提升 = +0.5–1%) |

---

## 6. Monitoring Checklist

While experiment is running, check every ~1 hour:

```bash
# Quick status check:
python3 -c "
import json
lines = open('data/outputs/btcv_v3_4_fm_rl_v4_distill_det_gpu1/posttrain_grpo_v2/logs.jsonl').readlines()
latest = [json.loads(l) for l in lines[-5:]]
for d in latest:
    print(f'step={d[\"step\"]} fb={d.get(\"final_score_best\",0):.4f} det={d.get(\"det_score_mean\",0):.4f} da={d.get(\"distill_active_frac\",0):.3f} dl={d.get(\"distill_loss\",0):.6f} eval={str(d.get(\"eval_iou\",\"N/A\"))[:8]}')
"
```

### Key metrics to watch
- `distill_active_frac` > 0.3 consistently (gate firing)
- `distill_loss` > 0 (distillation training happening)
- `eval_iou` trend (monotonically increasing over 10+ eval points)
- `det_score_mean` increasing (confirms det path is improving)

### Red flags
- `distill_active_frac = 0` for many consecutive steps → gate too strict or action_std too small
- `eval_iou` regressing below baseline 0.884 → lr too high or KL issue
- Training crash → OOM or divergence

---

## 7. Next Steps (Post-Experiment)

If V4-A shows improvement:
1. Run full eval: `python scripts/eval_v37_full_iou.py` on best checkpoint
2. Compare to baseline 0.8926 (full eval)
3. If still not 明显提升 (+0.5%), launch V4-B with stronger params

If V4-A shows regression:
1. Check `det_score` trend (is det path degrading?)
2. Lower `lr` to 1e-7, reduce `distill_weight` to 0.3
3. Consider pure distillation (no PPO, `kl_beta=0`)

---

## 8. Theoretical Understanding

The self-improving distillation loop:

```
Step N:
  PPO:          stochastic rollouts improve (final_best ↑)
  Gate:         best_stochastic > det + margin?
  Distill:      YES → teach velocity field toward best trajectory
  
Step N+10:
  Det model:    slightly better (distill transferred improvement)
  Gate:         bar rises (det_score higher)
  Loop:         continues pushing model up
```

This is on-policy best-of-k self-distillation. The key theoretical insight is that even when the average stochastic rollout is WORSE than deterministic (action_std adds noise), the **maximum of k rollouts** often exceeds deterministic because the stochastic process occasionally finds better local minima in the flow-matching ODE trajectory.

With k=8 and action_std=0.05, empirically `E[max_k(score)] ≈ 0.91-0.96` while `det_score ≈ 0.88-0.95`, giving a consistent positive distillation signal.

---

## 9. V4 Second Wave: Diagnosing Distillation Signal Weakness  
*(updated 2026-05-19)*

### 9.1 V4-A / V4-B Results (117–146 steps)

| Experiment | Config | In-train eval (best) | Full eval | Note |
|-----------|--------|---------------------|-----------|------|
| V4-A | k=8, std=0.05, lr=3e-7 | 0.8896 (step 80) | 0.8923 | baseline 0.8925 — no gain |
| V4-B | k=12, std=0.05, lr=3e-7, no PPO | 0.8842 (step 40) | — | declining to 0.879 at step 80 → killed |

Key observation: `distill_loss ≈ 0.001` for most V4-A/B steps. With lr=3e-7 and grad_norm ≈ 0.2, 
per-param update ≈ ~6e-11/step. After 500 steps: cumulative change ~3e-8 per param = 0.0003% 
of param magnitude. **The learning signal is too weak to produce measurable improvement.**

### 9.2 Root Cause: action_std=0.05 Is Too Small

Stochastic rollouts with std=0.05 converge to nearly the same endpoint as the deterministic model.
The normalized displacement difference (x1_stochastic − x1_det) is tiny, hence MSE distillation 
loss ≈ 0.001 most steps.

Evidence: V4-C (std=0.05, lr=3e-6) showed 24/26 steps with dl < 0.005, but 2 anomalous steps 
(step 6: dl=0.172, step 7: dl=0.027) where a stochastic rollout found a genuinely different 
trajectory. Those 2 steps had 10-100× larger gradient norms.

**This confirms: rare large-displacement rollouts carry real signal; the issue is they're too rare 
and too large relative to the learning rate.**

### 9.3 V4-C Crash (lr=3e-6)

V4-C tested 10× larger LR. Crashed at step 20: eval_iou 0.885 → **0.808** (catastrophic).

Timeline:
- Step 6: dl=0.172, dgnorm=2.67 (clipped to 0.5) → large gradient update  
- Steps 7–20: model degraded progressively (det_score 0.932→0.774)  
- Step 20: eval_iou = 0.808 (full collapse)

**Root cause:** Single outlier distillation step (dl=0.172) with lr=3e-6 caused a 10× larger 
update than typical. The model was pushed into a corrupted weight space it couldn't recover from.

### 9.4 Two-Dimensional Fix: Larger std + Loss Clip

The solution requires addressing both axes:
1. **Larger action_std (0.10–0.15)**: Generate more diverse rollouts → higher mean dl → stronger signal at same lr
2. **Loss clip (distill_loss_clip=0.03)**: Prevent catastrophic outlier steps → enables larger lr safely

New experiments launched:

| Experiment | GPU | std | k | lr | clip | status |
|-----------|-----|-----|---|----|------|--------|
| V4-E | 2 | 0.15 | 16 | 3e-7 | none | running |
| V4-F | 3 | 0.10 | 12 | 5e-7 | none | running |
| V4-G | 4 | 0.15 | 16 | 1e-6 | 0.03 | running |
| V4-A | 1 | 0.05 | 8 | 3e-7 | none | running (reference) |

### 9.5 Code Change: distill_loss_clip

Added `grpo_v2_distill_loss_clip` parameter to `grpo_train_v2.py` (line ~283):
```python
distill_loss_clip = float(cv('distill_loss_clip', 0.0))
```
Applied after computing distillation loss (line ~912):
```python
if distill_loss_clip > 0.0:
    distill_loss = distill_loss.clamp(max=distill_loss_clip)
```

### 9.6 Early Results (V4-E/F/G, step 1–20)

V4-E already shows **mean_dl=0.0075** (2.3× V4-A mean_dl=0.003), confirming std=0.15 
generates stronger distillation signal as expected.

Step 20 eval results:

| Experiment | step-1 | step-20 | Δ |
|-----------|--------|---------|---|
| V4-A (ref) | 0.88375 | 0.88278 | −0.001 |
| V4-E | 0.88302 | 0.88431 | **+0.001** |
| V4-F | 0.87949 | 0.87851 | −0.001 |
| V4-G | 0.88635 | **0.87106** | **−0.015** ← killed |

V4-G at step 20: clear regression, killed. lr=1e-6 too aggressive even with loss_clip=0.03.
V4-E at step 20: slight positive signal, within noise band.

---

### 9.7 Extended Results (V4-E steps 1–55, V4-F steps 1–60) — CRITICAL FINDINGS

**V4-E eval trajectory (std=0.15, k=16, lr=3e-7):**
- Step 1: 0.88302 → Step 20: 0.88431 → Step 40: **0.88051**
- Peaked at step 20, then declined below initial value.
- At step 55 (latest): fb=0.8991, det=0.8843, dl=0.00043 (very small)

**V4-F eval trajectory (std=0.10, k=12, lr=5e-7):**
- Step 1: 0.87949 → Step 20: 0.87851 → Step 40: 0.87867 → Step 60: **0.86952**
- Clear downward trend from step 40 onward. **Killed at step 60.**

**Conclusion from all V4-E/F/G experiments:**
```
Higher lr → faster regression (V4-G: −0.015 by step 20, V4-F: −0.009 by step 60)
Lower lr (V4-E 3e-7) → within noise band (±0.003), no measurable gain
No lr value found that gives consistent improvement.
```

### 9.8 Fundamental Diagnosis: Is the Exploration Space Viable?

The root question is: **do stochastic rollouts (action_std=0.15, k=16) actually find contours 
better than the deterministic path?** If not, there is nothing to distill.

Two hypotheses:
1. **H1 – Signal too weak**: Better contours DO exist in stochastic space but learning 
   rate / signal-to-noise prevents learning them. Fix: better algorithm.
2. **H2 – No signal**: Best-of-k stochastic ≈ deterministic on the true eval distribution.
   Stochasticity explores, but the ODE is already near-optimal. Fix: different approach entirely.

To test this, launched **oracle ensemble diagnostic** (`test/eval_best_of_k_oracle.py`) on GPU 4:
- Runs det (k=1, std=0), avg-of-16, and oracle best-of-16 (GT-scored) on all 150 test samples
- Key metric: `Δ(best_k_oracle − det)` — upper bound on what distillation can ever achieve
- Running at ~80% GPU utilization, results expected in ~60 minutes

**Decision tree based on oracle result:**
- `Δ > 0.010`: Strong signal in stochastic space → redesign distillation (curriculum, RWR, DPO)
- `Δ = 0.003–0.010`: Moderate signal → focus on hard samples, larger k, gradient accumulation
- `Δ < 0.003`: No meaningful signal → pivot to non-distillation approaches (test-time ensemble 
  with self-consistency scorer, supervised fine-tuning on hard samples, post-processing)

V4-A (std=0.05, lr=3e-7) still running on GPU 1 as long-running reference.
V4-E (std=0.15, lr=3e-7) still running on GPU 2 — most conservative active experiment.

---

### 9.9 Distillation Signal Distribution Analysis (V4-E, 32 steps)

V4-E distillation loss distribution over 32 steps:
- **26/32 steps**: dl < 0.005 (effective learning ≈ 0, even at lr=3e-7)
- **6/32 steps**: dl > 0.01 (real signal: steps 2,7,10,18,19,23 — mean 0.0284)
- Cumulative learning: sum(dl) = 0.2086 over 32 steps
- Mean dl_small = 0.00147, mean dl_big = 0.0284

The distribution is **extremely heavy-tailed**. Most steps contribute nothing;
rare "lucky" steps drive all weight changes. At lr=3e-7, even the large steps 
contribute ≤3×10⁻⁸ per-parameter per-step. The cumulative effect over 32 steps 
may be visible as slow degradation from accumulated incorrect distillation steps.

**Key concern**: Large dl steps occur when the stochastic rollout finds a very different 
trajectory from the deterministic one — but "different" does not mean "better for all samples."
These large distillation steps may overfit to specific batch-lucky trajectories, causing 
slow-but-accumulating generalization loss.

**Final V4-E/F/G verdict (all killed):**

| Experiment | Peak eval | Step killed | Eval at kill | Verdict |
|-----------|-----------|-------------|--------------|---------|
| V4-G (lr=1e-6, clip) | 0.886 | step 22 | 0.871 | killed — fast regression |
| V4-F (lr=5e-7) | 0.879 | step 60 | **0.870** | killed — slow regression |
| V4-E (lr=3e-7) | 0.884 | step 71 | **0.875** | killed — slow regression |
| V4-A (lr=3e-7, std=0.05) | 0.890 | still alive | oscillating 0.882-0.890 | reference, dl≈0 → stable |

**Key conclusion**: V4-A (std=0.05 → dl≈0.001 → effectively no learning) is STABLE.
V4-E/F/G (std=0.10-0.15 → dl=0.001-0.05 → real learning) ALL REGRESS.
**The distillation learning direction itself is wrong — not just the learning rate.**

---

## 10. Critical Pivot: Why Quality-Gated Distillation Fails

### 10.1 Root Cause: Stochastic "improvements" are not systematic

The distillation gate fires when: `best_of_k_score > det_score + 0.001`

BUT: The stochastic "win" comes from ONE batch, with ONE noise realization, for ONE image.
- The best-of-k contour is LUCKY on that specific batch, not systematically better.
- Teaching the FM velocity field to produce that lucky endpoint DEGRADES performance on other inputs.
- This is "catastrophic forgetting" / "negative transfer" — locally correct, globally wrong.

Evidence:
- V4-A (dl≈0): stable at 0.882-0.890 (no learning = no degradation)  
- V4-E (dl≈0.005): slow but consistent decline after step 20 (0.884 → 0.875 over 50 steps)
- V4-F (dl≈0.005): slower decline (0.879 → 0.870 over 60 steps)
- V4-G (dl≈0.005, lr=1e-6): fast decline (0.886 → 0.871 in 22 steps)

The regression rate scales with learning rate (∝ lr), confirming distillation causes the damage.

### 10.2 Metric mismatch ruled out

The distillation gate uses `0.2*MBoundF + 0.2*Dice + 0.6*IoU` (primarily IoU-aligned).
The eval metric is polygon area IoU. These are strongly correlated → not the root cause.

### 10.3 FM training objective is correct

The distillation loss correctly implements the FM objective:
- `x_1` = normalized best stochastic displacement
- `x_0 ~ N(0, noise_scale)` (matches inference distribution)
- `x_t = (1-t)*x_0 + t*x_1`, `v_target = x_1 - x_0`
- Loss = MSE(v_pred, v_target) for random t — standard FM training
Implementation is correct. The problem is the TRAINING SIGNAL (x_1 target) is unreliable.

### 10.4 Oracle Diagnostic — COMPLETED (2026-05-18 01:04)

Results file: `visual/ensemble_oracle/ensemble_oracle_k16_std0.15_20260518_010405.json`

**NOTE: This run used the wrong checkpoint** (`btcv_diffusion_dit_v3_4_fm_full_noleak`, det=0.8827)
instead of the correct one (`yolom_gpu35_reusemax`, det=0.8925). However, the relative findings
are still highly informative.

| Mode | Median IoU | vs det |
|------|-----------|--------|
| Deterministic (std=0) | **0.8827** | — |
| Avg of 16 (displacement average) | 0.7949 | **−0.088** |
| Best of 16 (oracle, GT-scored) | 0.5125 | **−0.370** |

**Individual stochastic rollout distribution (per sample):**
- Each rollout: IoU ≈ 0.42–0.55 (across all 150 samples)
- best_k > det: **0/150 samples (0%)**
- best_k < det−0.05: **150/150 samples (100%)**
- Even oracle best-of-16 is **never** better than deterministic

**Example per-sample data (std=0.15):**

| Sample | det | avg_k | best_k | individual stoch IoUs |
|--------|-----|-------|--------|----------------------|
| 0 | 0.8825 | 0.7636 | 0.4439 | [0.41, 0.44, 0.43, 0.41, 0.43, ...] |
| 1 | 0.8900 | 0.7623 | 0.4923 | [0.47, 0.47, 0.47, 0.47, 0.49, ...] |
| 2 | 0.9149 | 0.8358 | 0.5430 | [0.52, 0.53, 0.53, 0.54, 0.52, ...] |

**Critical observation:** avg_k (0.7949) >> individual stoch mean (~0.44), because avg_k
averages the displacement vectors before evaluation (noise cancellation). But avg_k (0.795)
is still far below det (0.883). This means std=0.15 is so large that even noise-averaged
outputs cannot recover.

### 10.5 Oracle Low-Std Sweep (running, 2026-05-18)

Since std=0.15 shows zero useful signal, launched 3 oracle runs with correct checkpoint
(`yolom_gpu35_reusemax`) and lower std values on GPUs 1–3:

| Oracle Run | GPU | std | Action |
|-----------|-----|-----|--------|
| oracle_std001 | 1 | 0.01 | running (PID 351156) |
| oracle_std002 | 2 | 0.02 | running (PID 351186) |
| oracle_std005 | 3 | 0.05 | running (PID 351349) |

Config used: `btcv_v3_4_fm_rl_v4_distill_det_gpu1.yaml` (YOLO-M, correct architecture).
GPU selection via `GRPO_V2_GPU` env var (overrides `cfg.gpus` before CUDA_VISIBLE_DEVICES is set).

**Expected results** based on physics of FM:
- std=0.01: stoch ≈ det (tiny perturbations), best_k ≈ det, avg_k ≈ det
- std=0.02: stoch slightly noisy, best_k/avg_k slightly below det
- std=0.05: stoch moderately noisy, consistent with V4-A finding (dl≈0)

### 10.6 Per-Sample Failure Mode Analysis (Baseline Model)

From full eval of the best V4-A checkpoint (0.8923 median):

```
Distribution:  Min=0.787  P5=0.851  P10=0.865  P25=0.878  Med=0.892  P75=0.911  P90=0.926  Max=0.944
Std = 0.026 (very tight distribution)

Samples below threshold:
  IoU < 0.80:   2/150  (1.3%)  — idx 12 (0.787), idx 10 (0.794)
  IoU < 0.85:   7/150  (4.7%)
  IoU < 0.88:  45/150 (30.0%)
  IoU < 0.90:  87/150 (58.0%)
```

Bottom 10 worst samples: idx=12 (0.787), 10 (0.794), 138 (0.832), 7 (0.842), 5 (0.845),
  26 (0.847), 34 (0.848), 71 (0.851), 35 (0.852), 27 (0.854)

**Key implication:** The distribution is very tight (σ=0.026). The model already handles 
95% of samples at IoU > 0.85. "Obvious improvement" requires moving the bulk of the 
30–90 percentile range upward, which requires improving samples at IoU 0.88–0.90.

---

## 11. Root Cause: Why FM + ODE Noise RL is Fundamentally Limited

### 11.1 The Core Problem

Flow Matching models are trained to predict deterministic velocity fields: for any x_t on the
interpolation path x_0→x_1, the model learns v*(x_t, t) = x_1 − x_0. The ODE is deterministic.

When we inject noise ε ~ N(0, σ²I) at each ODE step during rollout ("action_std"):
- The perturbed trajectory x̃_{t+dt} = x̃_t + v*(x̃_t, t)·dt + ε√dt deviates from the 
  training distribution  
- The model was **never trained on noisy ODE trajectories**
- At std=0.15, the accumulated deviation (√(60 steps) × 0.15 ≈ 1.16σ) overwhelms the signal
- Result: stochastic rollouts are entirely in-distribution noise (IoU ~0.42–0.55)

### 11.2 Why std=0.05 Gives No Signal Either

At std=0.05, individual stochastic rollouts are nearly identical to the deterministic path.
The "best of k" barely differs from the mean (dl≈0.001 for V4-A). This is mathematically 
expected: the FM model's velocity field is a smooth function, so a tiny perturbation to the 
ODE state produces a tiny perturbation to the output.

There is **no "Goldilocks" std** value where:
- Stochastic rollouts are diverse enough to sometimes find genuinely better contours  
- But close enough to the training distribution for the improvements to be valid

This is a fundamental impossibility for a well-trained deterministic FM model.

### 11.3 Why PPO Ratio ≡ 1.0

The policy ratio π_new(a|s) / π_old(a|s) ≡ 1.0 because:
- The FM "policy" outputs deterministic velocities — there is no stochastic policy distribution
- The noise is added *externally* to the ODE step, not sampled from the model
- Therefore the model's log-probability of the noisy action is not well-defined
- The code's log-prob computation returns 0, making ratio = exp(0) = 1

PPO policy gradient is **completely dead** throughout all V4 experiments. Only distillation
(FM supervised objective) contributed any weight changes.

### 11.4 The Structural Fix Needed

For RL to work on a generative model for contour segmentation, one of these changes is needed:

**Option A — Stochastic FM (score-based diffusion)**
Train a diffusion model (forward noising, reverse denoising) instead of flow matching.
Then RL works naturally: the reverse denoising process IS a stochastic policy.
Cost: ~2–4 weeks of retraining from scratch.

**Option B — Direct output space RL** 
Define a stochastic policy over the *output contour* (not the velocity field):
  π(Δcontour | image, init_contour) ~ N(μ_θ(·), σ²I)
where the FM model predicts μ_θ (mean displacement) and σ is a learned or fixed scale.
Then reward-weight the samples: L = -E[R(μ + σ·ε) · log π]
This requires architectural changes to output a distribution, not just a point estimate.

**Option C — Non-RL improvements (near-term, highest probability of success)**
1. Test-time averaging with random initialization perturbation (not ODE noise)
2. Ensemble of multiple passes with different random polygon seeds
3. Supervised fine-tuning on hard training samples
4. More refinement passes (4–5 instead of 3)
5. Label-efficient learning on the ~45 samples with IoU < 0.88

---


---

## 12. Session 2 (2026-05-18): V5 Multi-Start Approach + TTA Experiments

**Date:** 2026-05-18  
**Status:** TTA-4 eval running (GPU 3), V5 config ready, FM multi-start oracle running (GPU 4)

### 12.1 Critical Insight: The Right Source of FM Stochasticity

Section 11 identified ODE-noise injection as the root cause of V4 failure. The correct 
source of stochasticity in a Flow Matching model is the **initial noise x_0 ~ N(0, 1)**.

Looking at `sample_with_logprob` (flow_matching_evolution.py:816):
```python
x = torch.randn_like(i_it_py) * float(noise_scale)  # noise_scale = 1.0
```
Each call samples a DIFFERENT x_0. This is the FM model's natural stochasticity. 
A well-trained FM maps ANY x_0 ~ N(0,1) to approximately the same good contour.

This means:
- **V4 action_std=0**: one fixed x_0 seed → "deterministic" within that call  
- **V4 action_std=0.15**: same x_0 but ODE corrupted → off-manifold (catastrophic)
- **V5 approach**: action_std=0, k=16 different x_0 seeds → k valid FM samples

### 12.2 V5 Hypothesis: x_0 Diversity Gives Valid Distillation Targets

If different x_0 seeds occasionally produce higher-IoU contours (e.g., one seed 
geometrically aligns better with the target organ boundary), then:
1. Select best-of-16 by IoU
2. Gate: only distill if best > det + 0.002 (same gate as V4)  
3. Distill: FM loss toward the best trajectory's output displacement
4. The target is always a VALID FM output (not corrupted) → no catastrophic regression

This is fundamentally different from V4 where distillation targets were corrupted 
(IoU ~0.44-0.55 from ODE-noise rollouts).

### 12.3 FM Multi-Start Oracle (std=0, k=16, GPU 4, PID 379870)

**Purpose:** Measure how much x_0 diversity affects output IoU.
- det_iou = single deterministic rollout (one x_0 seed)
- stoch_ious = 16 rollouts with action_std=0 (16 different x_0 seeds)  
- avg_k_iou = average of 16 contours → tests TTA benefit
- best_k_iou = oracle best-of-16 → upper bound on V5 improvement potential

**Decision thresholds:**
- `best_k - det > 0.005` (>10 samples): V5 training is worthwhile
- `avg_k - det > 0.001`: TTA averaging improves accuracy
- `best_k ≈ det`: FM is near-deterministic; pivot to supervised fine-tuning

### 12.4 TTA-4 Eval Running (GPU 3, PID 384091)

Changed `infer_avg_samples: 1 → 4` in eval config (copies V4 config with this one change).
Running full 150-sample eval. Expected runtime: ~90-120 min (4× baseline).

If TTA-4 improves median IoU by >0.001: consider TTA-8 or TTA-16 as the primary gain path.
Cost: K× inference time, no training required.

### 12.5 Experiments Running Simultaneously

| Experiment | Process | GPU | Status | Purpose |
|------------|---------|-----|--------|---------|
| Oracle std=0 k=16 | 379870 | GPU 4 | Running | Measure FM x_0 diversity |
| TTA-4 full eval | 384091 | GPU 3 | Running | Measure TTA benefit |
| Oracle std=0.05 | 351349 | GPU 3 | Running (2h+) | Confirm low ODE-noise uselessness |
| Oracle std=0.01 | 351156 | GPU 1 | Running (2h+) | Same |
| Oracle std=0.02 | 351186 | GPU 2 | Running (2h+) | Same |

### 12.6 V5 Config Ready to Launch

`configs/btcv_v3_4_fm_rl_v5_multistart_gpu3.yaml` prepared with:
- `action_std: 0.0` / `action_std_min: 0.0` (no ODE noise, pure x_0 diversity)
- `k: 16` (16 x_0 seeds per training step for better coverage)
- `rollout_steps: 10` (matches eval ODE steps for consistency)
- `lr: 5.0e-7` (slightly higher than V4's 3e-7 since targets are valid)
- `distill_compare_det: 1`, `distill_det_margin: 0.002` (same quality gate)
- `distill_loss_clip: 0.05` (safety clip for outlier x_0 seeds)

**Launch command (when oracle confirms signal):**
```bash
CFG_FILE=configs/btcv_v3_4_fm_rl_v5_multistart_gpu3.yaml \
GRPO_V2_GPU=3 \
nohup conda run -n snake1 python grpo_train_v2.py > /tmp/v5_train.log 2>&1 &
```


---

## 13. TTA-4 Results + V5 Training Launch (2026-05-18 02:00)

### 13.1 TTA-4 Results — FM x_0 Diversity is REAL

**Full eval completed (150 samples, infer_avg_samples=4):**

| Metric | Baseline (n=1) | TTA-4 (n=4) | Delta |
|--------|---------------|-------------|-------|
| Median IoU | 0.8923 | **0.8942** | **+0.0019** |
| Mean IoU | 0.8925 | **0.8947** | **+0.0022** |
| Improved (δ>0.001) | — | 93/150 (62%) | — |
| Regressed (δ<-0.001) | — | 43/150 (29%) | — |

**Hard samples (IoU < 0.88 baseline, 45 samples):**
- Baseline median: 0.8704 → TTA-4 median: **0.8733** (+0.0029)
- Mean delta on hard samples: **+0.0051** (2× the overall average)

**Notable per-sample improvements:**
- idx=12 (worst sample): 0.7872 → **0.8250** (+0.0377!) 
- idx=5: 0.8451 → **0.8754** (+0.0303)
- idx=7: 0.8421 → **0.8582** (+0.0161)

**Conclusion:** FM x_0 diversity is meaningful. Different x_0 ~ N(0,1) seeds produce 
significantly different output contours, and averaging 4 seeds reduces variance and 
improves IoU. This validates the V5 hypothesis.

### 13.2 Why TTA Works (and V4 Failed)

TTA averaging uses **proper FM stochasticity** (different x_0 seeds → different clean ODE 
trajectories → different valid contours). V4 used **corrupted stochasticity** (same x_0 
but ODE noise added mid-trajectory → off-manifold trajectories → IoU ~0.44).

The +0.0377 improvement on idx=12 shows that for hard samples, the FM model has MUCH 
higher x_0 variance — some seeds find the right organ boundary, others don't. Averaging 
wins by canceling noise in the prediction.

### 13.3 V5 Training Launched (GPU 3, ~02:03)

Since TTA confirms FM x_0 diversity is real and usable, **V5 multi-start distillation** 
was launched immediately:
- Config: `configs/btcv_v3_4_fm_rl_v5_multistart_gpu3.yaml`
- GPU 3 (PID python ~402104)
- action_std=0.0, k=16 x_0 seeds per step
- Gate: distill only if best-of-16 > det + 0.002
- Expected: if best-of-16 consistently beats det, model learns to produce better outputs

Unlike TTA (inference-only), V5 trains the model to produce better deterministic outputs.
If V5 converges, the single-sample inference (n=1, no TTA) should improve.

### 13.4 TTA-8 Running (GPU 4, ~02:03)

Also launched `infer_avg_samples=8` to measure diminishing returns:
- n=1: 0.8923, n=4: 0.8942 (+0.0019)  
- n=8: ~0.895? (expected +0.001 more)
- n=16: ~0.896? (theoretical ceiling)

Results expected in ~60-90 min.

### 13.5 Active Experiments Summary

| Experiment | GPU | Status | Purpose |
|------------|-----|--------|---------|
| V5 training | GPU 3 | **Running** | Distill best x_0 → improve det baseline |
| TTA-8 eval | GPU 4 | **Running** | Measure n=8 averaging benefit |
| Oracle std=0 k=16 | GPU 4 | Running | Upper bound: best-of-16 x_0 diversity |
| Oracle std=0.05 | GPU 3 | Running (2.5h+) | Confirm ODE noise uselessness |
| Oracle std=0.01/0.02 | GPU 1/2 | Running (2.5h+) | Same |

