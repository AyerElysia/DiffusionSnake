# Detector-to-evolution isolation experiment

Date: 2026-08-03

## Scope

The production detector remains frozen LocateAnything base1500 and the evolution implementation is not modified. This experiment measures how detector geometry affects the strongest currently audited evolution checkpoint. It does not promote a new production integration.

## Audited evolution anchor

- Model: MemFlowDiT v0.5 minimal
- Checkpoint: `data/outputs/volmem/rl3d/ckpt_backup/v05_step_002300.pt`
- Verified mirror: `/dev/shm/memflowdit_checkpoints/v05_step_002300.pt`
- SHA256: `b3158d5648bdc1c0469302af3c5a13f8d92cc073f93be600a6cb932796f964ee`
- Historical fixed protocol: `sub-verse010`, `sub-verse011`, `sub-verse013`; 333 slices; GT boxes; memory off; seed 20260731
- Historical Volume Dice: 0.796574
- Historical foreground-slice Dice: 0.770027

The 2026-08-03 historical audit ranks this checkpoint above v0.6 step6100 (0.794934) and v0.5 step8000 (0.793664) under the same three-volume protocol.

## Detector contract limitation

Frozen base1500 outputs generic `vertebra` geometry only. DiffusionSnake requires canonical `[x1,y1,x2,y2,score,class_id]` with C1-L6 class IDs. The production adapter correctly rejects generic labels and bare boxes. Class IDs will not be fabricated.

The safe diagnostic therefore uses detector geometry plus oracle one-to-one GT class assignment. Unmatched detector boxes are dropped and unmatched GT instances remain misses. This is an optimistic detector-geometry upper bound, not a production numbering result.

Two diagnostic gates are planned on exactly the historical three volumes:

1. lenient oracle match at detector/GT IoU >=0.1;
2. release-like oracle match at detector/GT IoU >=0.3.

Both use the frozen base1500 predictions, MemFlowDiT v0.5 step2300, memory off, and seed 20260731. The existing GT-box result is the evolution upper-bound control.

## Live status and detector geometry audit

- External-cache tests: 40/40 passed.
- External-detection contract tests: 10/10 passed.
- The formal base1500 validation output already contained all slices for `sub-verse010` and `sub-verse013` but not `sub-verse011`; the missing 38 slices were inferred with the exact same frozen model, prompt, and inference program. The completed run is `detect_3D_lgz2/outputs/eval_locany_v3a_tier_a_base_1500_subverse011_20260803`.
- Combined protocol: 333 slices, 743 eligible GT boxes, 894 raw detector boxes.

At oracle-match IoU >= 0.1:

- 648/743 GT boxes matched: recall 0.872140.
- 648/894 detector boxes retained: keep rate 0.724832.
- 95 GT boxes missed; 246 unmatched detector boxes dropped.
- Mean IoU among matched boxes: 0.524551.

At oracle-match IoU >= 0.3:

- 588/743 GT boxes matched: recall 0.791386.
- 588/894 detector boxes retained: keep rate 0.657718.
- 155 GT boxes missed; 306 unmatched detector boxes dropped.
- Mean IoU among matched boxes: 0.555056.

These counts already identify recall as a material risk, especially for `sub-verse011`. The completed end-to-end metrics are recorded below.

## Completed end-to-end result

All three runs use the same MemFlowDiT v0.5 step2300 checkpoint, 333 slices, memory off, and seed 20260731.

| Box input | Volume Dice | Foreground-slice Dice | All-slice Dice | Foreground slices with predictions | Evolution initializations |
|---|---:|---:|---:|---:|---:|
| GT control | 0.796574 | 0.770027 | 0.879143 | 172/175 | 1796 |
| Detector geometry + oracle class, IoU >= 0.1 | 0.599545 | 0.365286 | 0.666442 | 116/175 | 648 |
| Detector geometry + oracle class, IoU >= 0.3 | 0.589185 | 0.355527 | 0.661313 | 115/175 | 588 |

Relative to GT boxes, the lenient detector path loses 0.197029 Volume Dice (24.73% relative) and 0.404741 foreground-slice Dice (52.56% relative). The IoU >= 0.3 path loses 0.207389 Volume Dice (26.04% relative) and 0.414500 foreground-slice Dice (53.83% relative).

Per-volume Dice:

| Case | GT | IoU >= 0.1 | Drop | IoU >= 0.3 | Drop |
|---|---:|---:|---:|---:|---:|
| sub-verse010 | 0.796639 | 0.602266 | -0.194373 | 0.593924 | -0.202715 |
| sub-verse011 | 0.819729 | 0.583942 | -0.235787 | 0.567039 | -0.252690 |
| sub-verse013 | 0.773354 | 0.612427 | -0.160926 | 0.606592 | -0.166762 |

The detector-input runs are strongly under-segmented. Total predicted foreground voxels are 489051 at IoU >= 0.1 and 452833 at IoU >= 0.3, versus 686379 in the GT-box control and 638499 GT foreground voxels.

## Interpretation

1. The strongest audited evolution checkpoint is not the bottleneck under GT initialization; it reaches 0.796574 Volume Dice.
2. Tightening the oracle match from IoU 0.1 to 0.3 changes Volume Dice by only 0.010360. The dominant problem is missing instance/slice coverage, not merely moderate box jitter.
3. The official detector protocol itself is misaligned with the evolution target. It evaluates only the largest connected component per label with box area >= 200, whereas the GT-box evolution control initializes 1796 components. Under the detector protocol only 743 eligible GT boxes remain; this produces zero detections on many small edge/partial-vertebra slices and cuts foreground-slice coverage from 172 to 115-116 of 175.
4. `sub-verse011` is the most detector-sensitive case and should be retained as a hard regression case.
5. These are optimistic diagnostic numbers because anatomical class IDs are supplied by oracle matching. Frozen base1500 still outputs generic `vertebra`, so a fully automatic production path would not be better without a real 2D numbering solution.

For context, the frozen base1500 formal 38-volume validation result at IoU 0.3 and area >= 200 is micro precision 0.676373, recall 0.722908, and F1 0.698867. That detector metric is internally valid, but it is not sufficient for the segmentation contract demonstrated here.

## Proposed 2D-first next work

The 3D/volume route remains optional. The core release gate must use a single sagittal 2D slice and must not depend on cross-slice tracking.

1. Export the exact GT boxes consumed by DiffusionSnake and reconcile the 1796 evolution initializations against the 1178 largest-per-label components and the 743 area-filtered detector targets. This prevents training against the wrong instance population.
2. Retune or fine-tune the 2D proposal detector for recall on the exact evolution targets. Include small/edge components, sweep the area threshold, and select score/NMS/box expansion using end-to-end Dice rather than detector F1 alone.
3. Add a genuinely 2D class path: a full-slice global context head plus per-box crop classifier/order decoder. It must produce canonical C1-L6 IDs without a volume tracker. Cross-slice tracking may only be an optional refinement.
4. After detector coverage improves, fine-tune a copy of v0.5 step2300 with detector-like box shift/scale/expansion augmentation. This can improve box-jitter tolerance, but it must not be presented as a remedy for missing boxes.
5. Keep the GT-box checkpoint and protocol unchanged as the evolution upper bound. Proposed promotion gates are foreground-slice coverage >= 95%, Volume Dice >= 0.74 on the fixed three-volume diagnostic, and no case below 0.70 before expanding validation.

Recommendation: do not spend the next experiment budget on a newer evolution architecture or the optional 3D path. First repair the 2D detector/evolution target contract and recall; only then use detector-aware evolution fine-tuning.

## Collaboration registry and paper hierarchy (recorded 2026-08-04)

This detector-module status report records the project lead's authorized collaboration structure:

1. Contour evolution / Flow mainline: task `019fb3e3-3c35-72d2-ab0d-418a302def49`.
2. Inference acceleration / whole-volume parallelism: task `019fc203-8a9e-76f1-a244-bab23d8f9bd5`.
3. Detector / initialization and coverage: task `019fb3d5-abc9-7662-8731-8b8cb0c44755` (this module).
4. Reinforcement learning / Contour RL: task `019fc1f4-6777-7752-9b19-244f5651882a`.
5. Project coordination, evidence audit, and Chinese/English paper writing: task `019fc08c-78a7-78f2-a56a-aba921e423ef`.

The unified paper hierarchy is fixed as follows:

- Flow is the first contribution.
- Contour RL is the second contribution.
- 3D is an extension for ordered volume data and does not compete with native voxel 3D methods.
- Detection and inference acceleration are system-performance support, not core paper innovations.
- The project lead owns the complete pipeline and is the final decision-maker.

The detector task is responsible for 2D-first initialization geometry and coverage, the detector/evolution target contract, canonical C1-L6 class delivery, and evidence-backed detector-to-evolution diagnostics. It must not elevate detector work to a core paper contribution or make the optional 3D path a dependency of the core 2D result.

Cross-task rule: any detector conclusion, failure, configuration, checkpoint, cache, or report that changes another module or a paper claim must be proactively synchronized with the affected task and paper coordination. A number may enter the paper only after a persisted report and machine result have been cross-checked.

Paper-coordination contact: task `019fc08c-78a7-78f2-a56a-aba921e423ef`. Any matter affecting paper claims, contribution hierarchy, experimental numbers, evaluation protocol, negative conclusions, or wording may be synchronized directly to this task. The project lead remains responsible for the complete pipeline and is the final decision-maker.

Current highest-priority synchronization is with the Flow mainline task `019fb3e3-3c35-72d2-ab0d-418a302def49`: jointly export and reconcile the exact GT boxes consumed by evolution. The current evidence contains 1796 GT evolution initializations, 1178 largest-per-label components without the detector area gate, and 743 detector-protocol targets at area >= 200. Until this contract is unified, detector recall and end-to-end Dice cannot be attributed cleanly. The Flow owner also needs the explicit caveat that the 0.599545/0.589185 detector-input Dice results use oracle class assignment and are not fully automatic production results. This contract resolution must subsequently be synchronized with paper coordination before any detector-to-evolution number is promoted into the manuscript.

## Whole-volume parallel synchronization and detector versioning (recorded 2026-08-04)

The H1/8-NFE whole-volume matrix provides an independent confirmation that box source dominates absolute deployment quality. Under one H1 Dense checkpoint, seed 20260731, three-volume protocol, and 2x4 AB2 schedule, autoregressive Volume Dice is 0.793892 with GT boxes but 0.544103 with predicted boxes, a gap of 0.249789. This gap must not be attributed to 3D Memory. Memory or parallelism deltas are valid only inside a fixed box source/provider, threshold set, and coverage audit.

Authoritative parallel evidence:

- Status report: `/home/medteam/Zhrch/DiffusionSnake-12-30-par3d-h1-outputs-20260803/reports/PARALLEL3D_H1_STATUS_20260803.md`.
- Full result table: `/home/medteam/Zhrch/DiffusionSnake-12-30-par3d-h1-outputs-20260803/reports/PARALLEL3D_H1_8NFE_RESULTS_20260803.md`.
- Machine comparison: `/home/medteam/Zhrch/DiffusionSnake-12-30-par3d-h1-outputs-20260803/comparison.json`.
- Matrix manifest: `/home/medteam/Zhrch/DiffusionSnake-12-30-par3d-h1-outputs-20260803/matrix_manifest.json`.

The H1 matrix did not use either LocateAnything cache below. Its predicted-box provider was the frozen in-model `heatmap_resnet` with ResNet-34, `det_conf_thresh=0.10`, per-class NMS at IoU 0.45, and `det_max_det=100`, as declared by `configs/volmem/verse_memflowdit_output_head_h1_distilled_dense_gpu0.yaml`. Record this condition as provider ID `h1pred-heatmap-r34-conf010-pcnms045-max100-v1`; config SHA256 is `d57ff1f9e0b620022e173e767bb03cad20c9ee126779b491cd323f4fb78755ae`, H1 checkpoint SHA256 is `5e28f12df357ec4d18fc9f0baf67b5a57655932a585b4ae1a0254d8449ecfc72`, and matrix git revision is `899c1847bbca90206b86b2bd74334ef8222e7fe1`. There is no persisted external detection cache for this H1 provider in the audited matrix, so future H1 deployment comparisons must not claim cache identity until predictions are materialized and coverage-audited.

Reusable persisted LocateAnything diagnostic caches for the v0.5 step2300 isolation protocol are:

| Coverage/cache ID | Cache path | SHA256 | Matched/eligible GT | GT-box recall | Detector keep rate | Mean matched IoU | E2E foreground-slice coverage | E2E Volume Dice |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `detcov3v-base1500-a200-iou01-oracleclass-v1` | `data/outputs/volmem/diagnostics/detector_evolution_isolation_v05_step2300_20260803/cache_iou01.json` | `082b8a5a6c47e92e8d41edc50445263144557645fdbed4eb5198c3008dc78bd8` | 648/743 | 0.872140 | 0.724832 | 0.524551 | 116/175 | 0.599545 |
| `detcov3v-base1500-a200-iou03-oracleclass-v1` | `data/outputs/volmem/diagnostics/detector_evolution_isolation_v05_step2300_20260803/cache_iou03.json` | `3163c3a2a6e3370d1ec92926e183096468414c992aff6d0d8eacad432e6eea9d` | 588/743 | 0.791386 | 0.657718 | 0.555056 | 115/175 | 0.589185 |

Both cache paths are relative to `/home/medteam/Zhrch/DiffusionSnake-12-30`. They cover `sub-verse010`, `sub-verse011`, and `sub-verse013`, 333 slices, area gate 200, frozen LocateAnything base1500 geometry, and one-to-one oracle GT class assignment. They are reusable for exact diagnostic reproduction but are not deployment caches and cannot support a claim of automatic C1-L6 numbering.

Paper/evidence rule: every predicted-box result must cite either the H1 provider ID above or a persisted cache ID plus SHA256, score/NMS/match thresholds, eligible-target contract, slice/instance coverage, checkpoint, seed, and machine-result path. GT-box and predicted-box absolute Dice may illustrate initialization sensitivity, but only within-box-source deltas may be used to attribute effects to Flow, parallelism, or 3D Memory.

## Flow interface responsibility correction (recorded 2026-08-04)

Task `019fb3e3-3c35-72d2-ab0d-418a302def49` is the contour evolution / Flow mainline, not the inference-acceleration / whole-volume-parallel task. The actual acceleration task is `019fc203-8a9e-76f1-a244-bab23d8f9bd5`. The preceding H1 matrix section remains a valid reference to the acceleration module's persisted evidence; it must not be interpreted as assigning acceleration ownership to the Flow task.

Authoritative Flow handover: `/home/medteam/Zhrch/DiffusionSnake-12-30/docs/report/FLOW_MAIN_HANDOVER_STATUS_20260804.md`.

The detector-to-Flow interface is fixed as follows:

- GT-box and predicted-box definitions, detector provider/cache ID, score and NMS thresholds, target eligibility contract, and coverage version must be frozen before a Flow comparison.
- The detector task owns the initialization audit: missed instances and slices, false positives, class errors, box IoU/centroid/scale offset, per-case coverage, and the exact persisted cache or provider signature.
- The Flow task owns contour-evolution analysis under the same frozen initialization: boundary/foreground/volume quality, topology and failure morphology, sampling schedule, and Flow-vs-matched-baseline attribution.
- Any deployment-quality decline must first pass a detector/evolution isolation with identical initialization. It cannot be attributed directly to Flow, Memory, parallelism, or 3D context.
- GT-box results isolate contour-evolution capability; predicted-box results evaluate the deployment chain. They may not be mixed in a single causal delta.

For synchronization, the detector task will provide the Flow task and paper coordination with a reusable coverage record keyed by cache/provider ID. At minimum it must include TP/FP/FN or matched/missed counts, class source, box-offset statistics, foreground-slice coverage, cases/slices, thresholds, SHA256, and the machine-result paths. Flow will then report evolution quality using that unchanged initialization version.
