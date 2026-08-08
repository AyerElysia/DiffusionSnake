# Pure-2D Flow 10k baseline and slim-B gate (2026-08-07)

## Decision

**STOP before slim-B.** The 10.858756M, 4-layer/state-256/replacer-512 baseline completed its required total-10k training and frozen dev8 evaluation, but it did not preserve the quality of the frozen H1 reference on the same eight development volumes. The single-variable `feature replacer hidden 512 -> 256` experiment is therefore **NO-GO**. Training a smaller model against this failed baseline could establish only non-inferiority to an unacceptable model.

No slim-B model was constructed, trained, or evaluated. Memory, RL, PCAA, detector training, and locked cases 010/011/013 remained out of scope.

## Frozen baseline

- Architecture: pure 2D Flow; 4 DiT layers; state dim 256; 8 heads; dense residual output hidden 256; MoonViT layer-18 center-only feature replacer hidden 512.
- Trainable parameters: **10,858,756**.
- Initialization: octagon.
- Training: H1-lineage weights-only transition from lineage step 2300, then 7,700 optimizer steps to lineage step 10,000.
- Outer-state sampler: Smooth Anchor-Biased Continuous, full support on `[0,1]`, no exact-anchor atoms; anchors `{1/3, 1/2, 1}`, uniform floor 0.4, beta-mixture mass 0.6, kappa 24.
- Inference: GT box + GT class, detector excluded, Memory/RL/PCAA absent, 2 outer x 4 inner AB2 = 8 NFE.
- Development cohort: fixed dev8 (1,123 slices); locked 010/011/013 access count = 0.

## Training audit

The training log contains exactly 7,700 contiguous continuation steps (lineage 2301--10000). There were no non-finite loss, gradient, LR, or timing records. Mean loss fell from 0.161581 over the first 100 continuation steps to 0.009011 over the final 100. The lineage-5000, lineage-8000, and lineage-10000 checkpoints have identical state schemas.

Primary checkpoint:

- `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/flow_main_h1_pure2d_dit4_train72_total10k_20260807/checkpoints/lineage_step_010000_continuation_007700.pt`
- SHA256: `0864f6b126d504d83370c02a8b73f26b1970c04be0c3e8ec532ecd4476c886fc`

Training audit:

- `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/flow_main_h1_pure2d_dit4_train72_total10k_audit_20260807/FLOW_H1_PURE2D_DIT4_TOTAL10K_TRAINING_AUDIT.json`
- SHA256: `1196236f1475a2d19f9ebd042b11b91ac2ef7fb0502da67d2944a50d7fbe1cde`

## Frozen dev8 result

| checkpoint | mean-volume Dice | median Dice | worst Dice | mean NSD@2 (voxel) | E2E evaluator loop | peak allocated VRAM |
|---|---:|---:|---:|---:|---:|---:|
| lineage 5000 (diagnostic) | 0.414814 | 0.428304 | 0.313183 | 0.264015 | 313.72 s | 401,501,696 B |
| lineage 8000 (diagnostic) | 0.431524 | 0.444522 | 0.330084 | 0.290896 | 308.32 s | 401,501,696 B |
| lineage 10000 (primary) | **0.426569** | **0.439355** | **0.325490** | **0.284474** | **332.72 s** | **401,501,696 B** |
| frozen H1/off reference, same dev8 | **0.794821** | **0.799031** | **0.760531** | **0.824198** | not used for speed comparison | not used for memory comparison |

Lineage 10000 minus the frozen H1/off reference:

- mean-volume Dice: **-0.368252**
- median-volume Dice: **-0.359677**
- worst-volume Dice: **-0.435041**
- mean NSD@2: **-0.539724**

All eight lineage-10000 per-volume Dice values lie between 0.325490 and 0.517334, whereas the same eight frozen H1/off values lie between 0.760531 and 0.816714. The 5k -> 8k trajectory improves, but 10k regresses slightly from 8k and remains far below the H1 reference.

Primary dev8 summary:

- `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/flow_main_h1_pure2d_dit4_train72_lineage10000_dev8_eval_20260807/summary.json`
- SHA256: `44fd13de4895d52e667b97aff07f18bfe0e58f94677407785254e6607c76f756`

Frozen H1/off reference:

- `/home/medteam/Zhrch/DiffusionSnake-12-30-oracle-memory-gate0-outputs-20260807/final_evaluation_v1/oracle_memory_gate0_results.json`
- SHA256: `f00e2e6e5eb483af4c3e7ce6c11055f85615f8882f01b08fdaf34833347e5c52`

## Timing scope

The recorded 332.72 seconds is the complete GPU evaluation loop for 1,123 batch-1 slices, including data transfer, model forward, contour-to-mask conversion, and slice evaluation (about 3.38 slices/s). It is **not** an isolated model-only kernel benchmark. No additional timing-only rerun was performed after the quality gate failed, because speed cannot rescue the failed scientific baseline.

## Gate interpretation

The requested slim-B non-inferiority test is conditional on an acceptable 10.858756M baseline. That condition failed before B registration. Consequently:

1. do not train replacer-256 against this baseline;
2. do not claim the 4-layer slim architecture preserves H1 quality;
3. do not use the low 10k loss as evidence of good inference quality;
4. first diagnose the large H1-to-slim loss (architecture transition, new output-head initialization, training objective/inference-state mismatch, and/or evaluator-path mismatch) under a separately preregistered recovery study;
5. only after the 10.858756M baseline recovers should the one-variable replacer-256 experiment be reconsidered.

