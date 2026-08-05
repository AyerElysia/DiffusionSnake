# Flow GT-oracle and interface status — 2026-08-04

## Executive result

The frozen Flow main line is now bound to Dense-6 DiT + H1 Dense Residual, GT-box/GT-class oracle initialization, memory-off, two outer refinements with four inner AB2 evaluations each (8 NFE), and seed 20260731. The three-case full replay processed 333 slices in fixed case/slice order and excluded detector and 3D-Memory effects.

Case Volume Dice:

| case | Volume Dice |
|---|---:|
| sub-verse010 | 0.796045 |
| sub-verse011 | 0.816807 |
| sub-verse013 | 0.769078 |
| mean over cases | 0.793977 |

The large directly reviewable example is sub-verse011 slice 0006 (median representative), Dice 0.818207. Its reconstructed prediction contains 1,872 foreground pixels versus 1,731 GT pixels, and CPU reconstruction from the signed stage-coordinate NPZ exactly reproduces the machine Dice and pixel count.

## Frozen protocol

- Architecture: Dense-6 DiT + H1 Dense Residual output head.
- Checkpoint: `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/output_head_h0_h1_h2_20260803/distilled/h1_distilled_full.pt`
- Checkpoint SHA256: `5e28f12df357ec4d18fc9f0baf67b5a57655932a585b4ae1a0254d8449ecfc72`
- Flow interface manifest v1.1: `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/flow_interface_freeze_20260804/FLOW_INTERFACE_FREEZE_MANIFEST_V1_1_20260804.json`
- Manifest SHA256: `c9ad7b8ffba2f3e2c5698a35a77f4b0b3c9fab23cd735d2241ae23aac2f55698`
- Schedule: AB2, two outer stages, four inner NFE per outer, fractions `[0.6667, 1.0]`, total 8 NFE.
- Initialization: GT box + GT anatomical label, mapped from `label_id=1..25` to `flow_class_id=label_id-1=0..24`.
- Execution: serial batch 1, memory-off, cases 010/011/013, ascending slice order.
- Feature path: frozen MoonViT layer 18, center-only replacement, cache read from `/dev/shm/memflowdit_moonvit_cache`.
- Not used: detector prediction, autoregressive Memory, parallel-off policy, Jacobi, Physical Volume Memory, coarse/refine second pass.

Each valid GT box becomes a four-point box initialization in the input coordinate system, is mapped to the quarter-resolution Flow grid, then uniformly upsampled to a 128-point contour. Outer stage 1 samples the current contour features, makes four AB2 denoiser calls, and applies fraction 0.6667. Outer stage 2 resamples at the updated contour, makes four more calls, and applies fraction 1.0. A non-empty slice therefore makes exactly eight DiT denoiser forwards, jointly over all instances in that slice. The retained sidecar has 172 non-empty slices, so the full 333-slice run corresponds to 1,376 denoiser forwards; empty slices make zero.

## Reproducible visualization selection

Within each case, selection first restricts to slices with nonzero GT foreground. The median representative is the smallest slice index minimizing absolute distance to the case median final per-slice Dice. The worst slice is minimum Dice, with the smallest slice index breaking ties.

| group | case | slice | Dice | GT pixels | predicted instances |
|---|---|---:|---:|---:|---:|
| median | sub-verse010 | 64 | 0.798828 | 1,012 | 10 |
| median | sub-verse011 | 6 | 0.818207 | 1,731 | 15 |
| median | sub-verse013 | 63 | 0.719896 | 1,516 | 18 |
| worst GT-positive | sub-verse010 | 147 | 0.000000 | 4 | 0 |
| worst GT-positive | sub-verse011 | 37 | 0.745763 | 52 | 1 |
| worst GT-positive | sub-verse013 | 35 | 0.000000 | 1 | 0 |

The zero-Dice end-cap examples in cases 010 and 013 are not evidence that a supplied GT box was evolved badly. Their GT foreground is only 4 and 1 original pixels, respectively, and downstream target validation supplies no Flow instance (`n_pred=0`). They expose a target/evaluation edge case. The worst slice with an actual Flow initialization is case 011 slice 37, Dice 0.745763.

Artifacts:

- Large single-case PNG: `/home/medteam/Zhrch/DiffusionSnake-12-30/visual/flow/gt_oracle_h1_8nfe_20260804_final_v2/gt_oracle_h1_8nfe_sub-verse011_slice0006_large_2x3.png`
- PNG SHA256: `987a5bd97d73059e53fccaac735b49e5da3a4e6f664c933db8f16d6a2a2f5c94`
- Large-figure machine sidecar: `/home/medteam/Zhrch/DiffusionSnake-12-30/visual/flow/gt_oracle_h1_8nfe_20260804_final_v2/single_case_sub-verse011_slice0006_manifest.json`
- Sidecar SHA256: `7026abfe4565124b1ce32c2dbb807819ecba7c1c61b1dae261f4137a9609688a`
- Six-row montage: `/home/medteam/Zhrch/DiffusionSnake-12-30/visual/flow/gt_oracle_h1_8nfe_20260804_final_v2/gt_oracle_h1_8nfe_median_and_worst_montage_labeled_r2.png`
- Montage SHA256: `c26ef5ddacdebccea1179dbf8c72ab146cca37fe7281db9ed526ea7618adc094`
- Visualization machine JSON: `/home/medteam/Zhrch/DiffusionSnake-12-30/visual/flow/gt_oracle_h1_8nfe_20260804_final_v2/visualization_manifest.json`
- Machine JSON SHA256: `ae169ec6fb8ba2ddb0354cfe5b547aa6c5629b0b6c0394b4c71ab0781f0b7a8a`
- Per-slice JSONL (333 rows): `/home/medteam/Zhrch/DiffusionSnake-12-30/visual/flow/gt_oracle_h1_8nfe_20260804_final_v2/slice_metrics.jsonl`
- JSONL SHA256: `fa348e410e8d3f0a6cf8a2048ce2b5e3f8f8d5c670a46b17b69827e71a4a5c03`

The visualization run used an observer and artifact I/O and is explicitly not a latency benchmark.

## GT instance population and count semantics

The frozen H1 dataset uses `sagittal_component_mode=significant`, top four raw contours per anatomical label, raw contour area at least 2, global cap 32 per slice, and output-grid polygon area greater than 0.5 with nondegenerate box validation.

- 1,796: components that survive the significant-component path and downstream output validation and actually initialize Flow in the three-case evaluation.
- 1,178: a different largest-only population without the detector area-200 eligibility gate.
- 743: the largest-only detector population after bbox area at least 200.

The 1,178 and 743 counts are not nested filters applied to the 1,796 Flow population. Raw significant components may still be discarded by output-grid polygon or degenerate-box validation; raw component counts must not replace the 1,796 retained Flow-row count.

## Flow-signed retained-only Gate-0 sidecar

The detector task produced a source sidecar that passed 1,796/1,796 native-GT row validation. Flow signed a non-destructive compatibility sidecar with explicit class and rank terms:

- Sidecar: `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/flow_interface_freeze_20260804/FLOW_GT_INSTANCE_SIDECAR_RETAINED_V1_20260804.json`
- SHA256: `764f9f80e459a9a0272ba43c3cf90682aacf970fe37ab6ef988528d8932dd901`
- Schema: `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/flow_interface_freeze_20260804/FLOW_GT_INSTANCE_SIDECAR_RETAINED_V1_SCHEMA_20260804.json`
- Schema SHA256: `8e4f8c253088e1b78ef94233674fd65e20354faca7fe35231a7a4efa05ca0c53`
- Report: `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/flow_interface_freeze_20260804/FLOW_GT_INSTANCE_SIDECAR_RETAINED_V1_REPORT_20260804.md`
- Report SHA256: `ff4b6d87bea9434edbfb9818ea4c7bf57d2c3fcc4258f5e6abd1e73ed773cb1a`

The sidecar covers 333 evaluated slices, 172 slices with retained instances, and exactly 1,796 retained rows. It explicitly records `label_id`, `flow_class_id`, zero-based rank index, one-based rank ordinal, raw contour area, original-image Flow bbox, both retained flags, and a dense zero-based `flow_instance_order` within each slice. The authoritative instance ID uses the one-based rank ordinal; the detector source ID is preserved separately.

This sidecar is intentionally retained-only. It is authorized for Gate-0 `py_ind`/contour-row alignment. It does not include discarded raw components and must not be used for raw retention auditing or to reconstruct 1,178/743.

## Box/class factorial status

| condition | status |
|---|---|
| GT box + GT anatomical class | available; authoritative Flow isolation path |
| predicted box + GT anatomical class | available only as explicitly oracle-class geometry isolation |
| GT box + predicted anatomical class | blocked; no registered non-oracle C1-L6 class provider |
| predicted box + predicted anatomical class | blocked; base1500 emits a generic vertebra class only |

Oracle class must never be presented as predicted class or deployment performance.

## Physical Memory interface boundary

Flow owns the frozen checkpoint, schedule, case/seed/noise/box/cache/contour ordering contract, stage-init/outer-1/outer-2 equality review, and gate-on contour failure attribution. The acceleration task owns Physical Volume Memory code, GPU Gate-0/1/2 execution, and pass/DiT-call/latency/throughput/peak-memory accounting.

For Gate-0, `stage_init`, `outer_1`, and `outer_2_final` must pass exact tensor equality (`max_abs=0`, mismatch count 0) together with exact instance sequence, `py_ind`, and pre-outer RNG digests. Numerical tolerances of `1e-6` in Flow coordinates and `1e-4` restored pixels are diagnostic only and cannot turn a non-exact run into a pass.
