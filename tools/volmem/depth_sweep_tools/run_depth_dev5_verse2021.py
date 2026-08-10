#!/usr/bin/env python3
"""Evaluate one depth-sweep checkpoint on the frozen nonlocked Dev5 cohort.

The native evaluator performs the only model pass.  This wrapper observes the
already-produced multiclass masks, restores them to source NIfTI geometry, and
computes scan-equal VerSe-2021 metrics plus clearly separated NSD@2mm/HD95
diagnostics.  It does not add model, DiT, Flow, or checkpoint calls.
"""

from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys

import cv2
import numpy as np


DEV5_CASES = (
    "sub-verse022",
    "sub-verse024",
    "sub-verse071",
    "sub-verse150",
    "sub-verse264",
)
LOCKED_CASES = ("sub-verse010", "sub-verse011", "sub-verse013")
EXPECTED_SLICE_COUNT = 1248


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("P0", "P1"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--slice-manifest", required=True)
    parser.add_argument("--case-metadata", required=True)
    parser.add_argument("--native-entry", required=True)
    parser.add_argument("--metric-module", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parallel-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--locate-feat-cache-root", required=True)
    parser.add_argument(
        "--schedule-profile",
        choices=("fast4", "fast8", "fast12", "fast16", "native-rich100"),
        default="fast8",
    )
    parser.add_argument(
        "--native-only",
        action="store_true",
        help="Stop after the single native model pass; skip physical 3D exports/metrics.",
    )
    parser.add_argument("--geometry-only", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(os.fspath(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_file(path, label):
    value = pathlib.Path(path)
    if not value.is_absolute():
        raise ValueError("{} must be absolute".format(label))
    value = value.resolve()
    if not value.is_file():
        raise FileNotFoundError("{} is missing: {}".format(label, value))
    return value


def fresh_dir(path):
    value = pathlib.Path(path)
    if not value.is_absolute():
        raise ValueError("result directory must be absolute")
    value = value.resolve()
    if value.exists():
        raise FileExistsError("result directory must be fresh: {}".format(value))
    return value


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, os.fspath(path))
    if spec is None or spec.loader is None:
        raise ImportError("could not load {} from {}".format(name, path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def png_reader(path):
    image = cv2.imread(os.fspath(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError("failed to decode mask PNG: {}".format(path))
    return image


def prepare_geometry(metric, slice_manifest, case_metadata):
    rows = metric.read_csv(slice_manifest)
    metadata_rows = metric.read_csv(case_metadata)
    metadata = {row["case_id"]: row for row in metadata_rows}
    selected = {case_id: [] for case_id in DEV5_CASES}
    seen_locked = []
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in LOCKED_CASES:
            seen_locked.append(case_id)
        if case_id in selected:
            selected[case_id].append(row)
    if seen_locked:
        raise RuntimeError("subset manifest contains locked cases")
    if set(row["case_id"] for row in rows) != set(DEV5_CASES):
        raise ValueError("subset manifest must contain exactly Dev5")
    if sum(len(value) for value in selected.values()) != EXPECTED_SLICE_COUNT:
        raise ValueError("Dev5 slice count must be exactly {}".format(EXPECTED_SLICE_COUNT))
    result = {}
    for case_id in DEV5_CASES:
        if case_id not in metadata:
            raise ValueError("case metadata missing {}".format(case_id))
        result[case_id] = metric.build_scan_geometry(
            case_id,
            selected[case_id],
            metadata[case_id],
            validate_png_reader=png_reader,
        )
    return result


def configure_formal_runtime(cfg, slice_manifest, locate_feat_cache_root, schedule_profile):
    cfg.volmem.manifest_file = os.fspath(slice_manifest)
    cfg.locate_feat_cache_root = os.fspath(locate_feat_cache_root)
    cfg.sagittal_moonvit_cache_root = os.fspath(locate_feat_cache_root)
    if schedule_profile in ("fast4", "fast8", "fast12", "fast16"):
        inner_steps = {
            "fast4": 2,
            "fast8": 4,
            "fast12": 6,
            "fast16": 8,
        }[schedule_profile]
        cfg.flow_ode_steps = inner_steps
        cfg.iterative_num_steps = 2
        cfg.iterative_fractions = [0.6667, 1.0]
        cfg.iterative_ode_steps = inner_steps
        cfg.v3_7_ode_solver = "ab2"
        cfg.v4_9_use_rich_infer_schedule = False
        cfg.v4_9_infer_target_fractions = [0.6667, 1.0]
        nfe = 2 * inner_steps
    elif schedule_profile == "native-rich100":
        cfg.flow_ode_steps = 5
        cfg.iterative_num_steps = 5
        cfg.iterative_fractions = [0.3333, 0.5, 0.80, 0.97, 1.0]
        cfg.iterative_ode_steps = 20
        cfg.v3_7_ode_solver = "euler"
        cfg.v4_9_use_rich_infer_schedule = True
        cfg.v4_9_infer_target_fractions = [0.3333, 0.5, 0.80, 0.97, 1.0]
        nfe = 100
    else:
        raise ValueError("unsupported schedule profile {}".format(schedule_profile))
    cfg.use_grpo = False
    cfg.use_gt_det = True
    cfg.skip_heatmap_detector_when_gt = True
    return {
        "flow_ode_steps": int(cfg.flow_ode_steps),
        "iterative_num_steps": int(cfg.iterative_num_steps),
        "iterative_fractions": [float(x) for x in cfg.iterative_fractions],
        "iterative_ode_steps": int(cfg.iterative_ode_steps),
        "ode_solver": str(cfg.v3_7_ode_solver),
        "rich_infer_schedule": bool(cfg.v4_9_use_rich_infer_schedule),
        "memory_mode": "parallel-off",
        "box_mode": "gt",
        "effective_passes": 1,
        "profile": str(schedule_profile),
        "nfe_per_prediction": int(nfe),
    }


def symmetric_surface_metrics(metric, gt_mask, pred_mask, affine):
    gt_points = metric.surface_world(gt_mask, affine)
    pred_points = metric.surface_world(pred_mask, affine)
    if not len(gt_points):
        raise ValueError("GT surface is empty")
    if not len(pred_points):
        return {"nsd_at_2mm": 0.0, "hd95_mm": None, "surface_distance_count": 0}
    gt_tree = metric.cKDTree(gt_points)
    pred_tree = metric.cKDTree(pred_points)
    pred_to_gt = gt_tree.query(pred_points, k=1)[0]
    gt_to_pred = pred_tree.query(gt_points, k=1)[0]
    distances = np.concatenate([pred_to_gt, gt_to_pred], axis=0)
    return {
        "nsd_at_2mm": float(np.mean(distances <= 2.0)),
        "hd95_mm": float(np.percentile(distances, 95.0)),
        "surface_distance_count": int(distances.size),
    }


def diagnostic_scan(metric, case_id, gt_volume, pred_volume, affine):
    labels = sorted(int(x) for x in np.unique(gt_volume) if int(x) > 0)
    per_label = []
    for label in labels:
        gt_mask = gt_volume == label
        pred_mask = pred_volume == label
        surface = symmetric_surface_metrics(metric, gt_mask, pred_mask, affine)
        gt_count = int(gt_mask.sum())
        pred_count = int(pred_mask.sum())
        if pred_count <= 0:
            expansion = 0.0
            abs_log_expansion = None
        else:
            expansion = float(pred_count / float(gt_count))
            abs_log_expansion = float(abs(np.log(expansion)))
        per_label.append({
            "case_id": case_id,
            "label": label,
            "gt_voxels": gt_count,
            "pred_voxels": pred_count,
            "foreground_ratio": expansion,
            "abs_log_foreground_ratio": abs_log_expansion,
            **surface
        })
    hd95 = [row["hd95_mm"] for row in per_label if row["hd95_mm"] is not None]
    return {
        "case_id": case_id,
        "scan_equal_label_nsd_at_2mm": float(np.mean([row["nsd_at_2mm"] for row in per_label])),
        "scan_equal_label_hd95_mm_mean_finite": float(np.mean(hd95)) if hd95 else None,
        "scan_equal_label_hd95_mm_penalty_100": float(np.mean([
            row["hd95_mm"] if row["hd95_mm"] is not None else 100.0
            for row in per_label
        ])),
        "scan_equal_label_foreground_ratio": float(np.mean([row["foreground_ratio"] for row in per_label])),
        "missing_label_count": int(sum(row["pred_voxels"] == 0 for row in per_label)),
        "per_label": per_label,
    }


def write_json(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def export_results(metric, output_root, collector, identity, native_summary):
    export_root = output_root / "verse2021_3d_dev5"
    prediction_root = export_root / "predictions"
    scan_root = export_root / "per_scan"
    prediction_root.mkdir(parents=True)
    scan_root.mkdir(parents=True)
    formal_results = []
    diagnostic_results = []
    created = []
    for case_id in DEV5_CASES:
        info = collector.geometries[case_id]
        header = info["gt_header"]
        prediction = collector.source_volume(case_id)
        gt_volume = metric.load_nifti(info["gt_path"], header)
        prediction_path = prediction_root / "{}_pred-multiclass.nii.gz".format(case_id)
        metric.write_uint8_like(prediction_path, prediction, header)
        prediction_header = metric.NiftiHeader.read(prediction_path)
        roundtrip = metric.load_nifti(prediction_path, prediction_header)
        if prediction_header.shape != header.shape:
            raise AssertionError("prediction shape drift")
        if not np.array_equal(prediction_header.affine, header.affine):
            raise AssertionError("prediction affine drift")
        if prediction_header.axis_codes != header.axis_codes:
            raise AssertionError("prediction axis drift")
        if not np.array_equal(roundtrip, prediction):
            raise AssertionError("prediction payload drift")
        formal = metric.evaluate_scan(
            case_id, gt_volume, prediction, header.affine, info["centroids"]
        )
        diagnostic = diagnostic_scan(
            metric, case_id, gt_volume, prediction, header.affine
        )
        scan_payload = {
            "schema": "diffusionsnake.depth_sweep_dev5_per_scan.v1",
            "identity": identity,
            "case_id": case_id,
            "source": {
                "gt_nifti_path": info["gt_path"],
                "gt_nifti_sha256": info["gt_sha256"],
                "centroid_json_path": info["centroid_path"],
                "centroid_json_sha256": info["centroid_sha256"],
                "slice_mapping_digest": info["mapping_digest"],
            },
            "prediction": {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
                "shape": list(prediction_header.shape),
                "affine": prediction_header.affine.tolist(),
                "axis_codes": list(prediction_header.axis_codes),
                "spacing_mm": list(prediction_header.spacing),
            },
            "verse2021": formal,
            "surface_diagnostic": diagnostic,
        }
        scan_path = scan_root / "{}.json".format(case_id)
        write_json(scan_path, scan_payload)
        formal_results.append(formal)
        diagnostic_results.append(diagnostic)
        created.extend([prediction_path, scan_path])
    formal_cohort = metric.summarize_cohort(formal_results)
    finite_hd95 = [
        row["scan_equal_label_hd95_mm_mean_finite"]
        for row in diagnostic_results
        if row["scan_equal_label_hd95_mm_mean_finite"] is not None
    ]
    diagnostic_cohort = {
        "definition": {
            "nsd": "pooled_bidirectional_3d_surface_distance_leq_2mm_per_label_then_scan_equal",
            "hd95": "pooled_bidirectional_3d_surface_distance_p95_per_label_then_scan_equal",
            "missing_prediction": "NSD=0; HD95 absent from unpenalized and 100mm in sensitivity",
            "status": "project_diagnostic_not_VerSe2021_primary",
        },
        "scan_equal_nsd_at_2mm_mean": float(np.mean([
            row["scan_equal_label_nsd_at_2mm"] for row in diagnostic_results
        ])),
        "scan_equal_hd95_mm_mean_unpenalized": float(np.mean(finite_hd95)) if finite_hd95 else None,
        "scan_equal_hd95_mm_mean_penalty_100": float(np.mean([
            row["scan_equal_label_hd95_mm_penalty_100"] for row in diagnostic_results
        ])),
        "scan_equal_foreground_ratio_mean": float(np.mean([
            row["scan_equal_label_foreground_ratio"] for row in diagnostic_results
        ])),
        "missing_label_count": int(sum(row["missing_label_count"] for row in diagnostic_results)),
        "per_scan": diagnostic_results,
    }
    result = {
        "schema": "diffusionsnake.depth_width_dev5_verse2021.v1",
        "status": "PASS",
        "identity": identity,
        "cohort": {
            "case_ids": list(DEV5_CASES),
            "case_count": len(DEV5_CASES),
            "slice_count": EXPECTED_SLICE_COUNT,
            "locked_case_opens": 0,
        },
        "verse2021": formal_cohort,
        "surface_diagnostic": diagnostic_cohort,
        "native_summary": native_summary,
        "observer_additional_model_dit_flow_calls": 0,
    }
    result_path = export_root / "DEPTH_WIDTH_DEV5_VERSE2021_RESULTS.json"
    write_json(result_path, result)
    created.append(result_path)
    manifest = metric.write_sha256_manifest(export_root, created)
    return result_path, manifest, result


def run(args):
    config = existing_file(args.config, "config")
    checkpoint = existing_file(args.checkpoint, "checkpoint")
    slice_manifest = existing_file(args.slice_manifest, "slice manifest")
    case_metadata = existing_file(args.case_metadata, "case metadata")
    native_entry = existing_file(args.native_entry, "native evaluator")
    metric_path = existing_file(args.metric_module, "metric module")
    locate_root = pathlib.Path(args.locate_feat_cache_root).resolve()
    if not locate_root.is_dir():
        raise FileNotFoundError("MoonViT cache root missing: {}".format(locate_root))
    result_root = fresh_dir(args.result_dir)
    if args.device != "cuda":
        raise ValueError("formal depth/width evaluation must run on GPU")
    if args.parallel_batch_size != 16:
        raise ValueError("formal comparison uses B16")
    if args.seed != 20260731:
        raise ValueError("formal comparison seed drift")

    metric = load_module(metric_path, "depth_width_verse2021_metrics")
    geometries = prepare_geometry(metric, slice_manifest, case_metadata)
    if args.geometry_only:
        payload = {
            "status": "PASS_GEOMETRY_ONLY",
            "case_ids": list(geometries),
            "case_slice_counts": {
                case_id: len(info["rows"]) for case_id, info in geometries.items()
            },
            "slice_count": sum(len(info["rows"]) for info in geometries.values()),
            "locked_case_opens": 0,
            "model_imported": False,
            "model_constructed": False,
            "checkpoint_opened": False,
            "gpu_used": False,
            "slice_manifest_sha256": sha256_file(slice_manifest),
            "case_metadata_sha256": sha256_file(case_metadata),
        }
        print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)
        return payload

    old_argv = list(sys.argv)
    native_tokens = [
        str(native_entry),
        "--cfg_file", str(config),
        "--ckpt", str(checkpoint),
        "--split", "val",
        "--memory-mode", "parallel-off",
        "--box-mode", "gt",
        "--result-dir", str(result_root),
        "--device", "cuda",
        "--parallel-batch-size", str(args.parallel_batch_size),
        "--seed", str(args.seed),
        "--log-every", "100",
        "--locate-feat-cache-root", str(locate_root),
    ]
    sys.argv = native_tokens
    try:
        native = load_module(native_entry, "depth_width_native_eval_{}".format(args.arm.lower()))
    finally:
        sys.argv = old_argv
    schedule = configure_formal_runtime(
        native.cfg, slice_manifest, locate_root, args.schedule_profile
    )
    if os.path.abspath(str(native.cfg.volmem.manifest_file)) != str(slice_manifest):
        raise AssertionError("runtime manifest binding drift")

    collector = metric.PredictionVolumeCollector(geometries)
    original_prediction_masks = native.prediction_masks
    original_build_model = native.build_model
    counters = {"prediction_mask_calls": 0, "build_model_calls": 0}
    model_identity = {}

    def observed_prediction_masks(output, batch, evaluator):
        masks = original_prediction_masks(output, batch, evaluator)
        paths = batch["img_path"]
        if isinstance(paths, str):
            paths = [paths]
        if len(paths) != len(masks):
            raise RuntimeError("mask/batch length mismatch")
        for image_path, mask in zip(paths, masks):
            record = evaluator._record_for_path(image_path)
            collector.add(record["case_id"], record["slice_idx"], mask)
        counters["prediction_mask_calls"] += 1
        return masks

    def observed_build_model(device):
        counters["build_model_calls"] += 1
        if counters["build_model_calls"] != 1:
            raise RuntimeError("build_model called more than once")
        model, step = original_build_model(device)
        model_identity.update({
            "checkpoint_step": int(step),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "trainable_parameter_count": int(sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )),
            "dit_num_layers": int(native.cfg.dit_num_layers),
            "dit_state_dim": int(native.cfg.dit_state_dim),
            "dit_num_heads": int(native.cfg.dit_num_heads),
        })
        return model, step

    native.prediction_masks = observed_prediction_masks
    native.build_model = observed_build_model
    native_summary = native.main()
    if counters["build_model_calls"] != 1:
        raise RuntimeError("build_model count drift")
    if int(native_summary.get("processed_slices", -1)) != EXPECTED_SLICE_COUNT:
        raise RuntimeError("native processed slice count drift")
    if int(native_summary.get("num_volumes", -1)) != len(DEV5_CASES):
        raise RuntimeError("native volume count drift")
    if int(native_summary.get("effective_contour_passes", -1)) != 1:
        raise RuntimeError("native effective pass count drift")
    if collector.capture_calls != EXPECTED_SLICE_COUNT:
        raise RuntimeError("observer capture count drift")
    collector.assert_complete()
    identity = {
        "arm": args.arm,
        "config_path": str(config),
        "config_sha256": sha256_file(config),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "native_entry_path": str(native_entry),
        "native_entry_sha256": sha256_file(native_entry),
        "metric_module_path": str(metric_path),
        "metric_module_sha256": sha256_file(metric_path),
        "slice_manifest_path": str(slice_manifest),
        "slice_manifest_sha256": sha256_file(slice_manifest),
        "case_metadata_path": str(case_metadata),
        "case_metadata_sha256": sha256_file(case_metadata),
        "runner_path": str(pathlib.Path(__file__).resolve()),
        "runner_sha256": sha256_file(pathlib.Path(__file__).resolve()),
        "schedule": schedule,
        "model": model_identity,
        "seed": int(args.seed),
        "parallel_batch_size": int(args.parallel_batch_size),
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "command_argv": native_tokens,
        "native_only": bool(args.native_only),
    }
    if args.native_only:
        result = {
            "schema": "diffusionsnake.depth_width_dev5_native_nfe.v1",
            "status": "PASS_NATIVE_ONLY",
            "identity": identity,
            "cohort": {
                "case_ids": list(DEV5_CASES),
                "case_count": len(DEV5_CASES),
                "slice_count": EXPECTED_SLICE_COUNT,
                "locked_case_opens": 0,
            },
            "native_summary": native_summary,
            "observer_additional_model_dit_flow_calls": 0,
        }
        result_path = result_root / "DEPTH_WIDTH_DEV5_NATIVE_NFE_RESULTS.json"
        write_json(result_path, result)
        print(json.dumps({
            "status": result["status"],
            "arm": args.arm,
            "result_path": str(result_path),
            "result_sha256": sha256_file(result_path),
            "schedule": schedule,
            "native_summary": {
                "evaluation_seconds": native_summary["evaluation_seconds"],
                "slices_per_second": native_summary["slices_per_second"],
                "peak_cuda_memory_gb": native_summary["peak_cuda_memory_gb"],
                "volume_mean_dice": native_summary["volume_mean_dice"],
                "foreground_slice_mean_dice": native_summary["foreground_slice_mean_dice"],
                "class_mean_dice": native_summary["class_mean_dice"],
            },
        }, sort_keys=True, allow_nan=False), flush=True)
        return result
    result_path, manifest, result = export_results(
        metric, result_root, collector, identity, native_summary
    )
    print(json.dumps({
        "status": result["status"],
        "arm": args.arm,
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "artifact_manifest": str(manifest),
        "artifact_manifest_sha256": sha256_file(manifest),
        "verse2021_main": result["verse2021"]["main"],
        "surface_diagnostic": {
            key: value for key, value in result["surface_diagnostic"].items()
            if key != "per_scan"
        },
    }, sort_keys=True, allow_nan=False), flush=True)
    return result


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
