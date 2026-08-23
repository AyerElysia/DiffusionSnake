#!/usr/bin/env python3
"""Evaluate one native-resolution Dev8 prediction tree with shared metrics."""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time
import traceback

import cv2
import numpy as np


REQUIRED_METHOD_KEYS = {
    "schema", "method_id", "display_name", "comparison_regime", "source",
    "weights", "parameters", "prediction_command", "environment",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("cannot load {} from {}".format(name, path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve(project_root, value):
    path = pathlib.Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    with pathlib.Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def checked_mask(path, expected_hw, allowed_labels):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError("prediction missing or unreadable: {}".format(path))
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("prediction must be a single-channel integer PNG: {}".format(path))
    if tuple(mask.shape) != tuple(expected_hw):
        raise ValueError("prediction shape mismatch {}: {} != {}".format(path, mask.shape, expected_hw))
    labels = set(int(value) for value in np.unique(mask))
    invalid = labels.difference(allowed_labels)
    if invalid:
        raise ValueError("prediction has invalid labels {}: {}".format(sorted(invalid), path))
    return mask.astype(np.uint16, copy=False)


def result_files(root):
    return sorted(path for path in pathlib.Path(root).rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")


def write_hash_manifest(root):
    root = pathlib.Path(root).resolve()
    rows = ["{}  {}".format(sha256_file(path), path.relative_to(root).as_posix()) for path in result_files(root)]
    output = root / "SHA256SUMS.txt"
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output


def validate_method(method, protocol):
    missing = REQUIRED_METHOD_KEYS.difference(method)
    if missing:
        raise ValueError("method manifest missing keys {}".format(sorted(missing)))
    regime = method["comparison_regime"]
    if regime not in protocol["regimes"]:
        raise ValueError("unknown comparison regime {}".format(regime))
    source = method["source"]
    if not source.get("repository_url") or not source.get("commit"):
        raise ValueError("method source requires repository_url and commit")
    parameters = method["parameters"]
    for key in ("total", "trainable"):
        if key not in parameters or int(parameters[key]) < 0:
            raise ValueError("method parameters.{} is invalid".format(key))
    weights = method["weights"]
    if not weights.get("identity"):
        raise ValueError("method weights.identity is required")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-audit", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--method-manifest", required=True)
    args = parser.parse_args()

    project_root = pathlib.Path(args.project_root).resolve()
    protocol_path = resolve(project_root, args.protocol)
    audit_path = resolve(project_root, args.protocol_audit)
    prediction_root = resolve(project_root, args.prediction_root)
    result_root = resolve(project_root, args.result_root)
    method_path = resolve(project_root, args.method_manifest)
    if result_root.exists():
        raise FileExistsError("refusing to overwrite {}".format(result_root))
    result_root.mkdir(parents=True)
    started = time.time()

    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        method = json.loads(method_path.read_text(encoding="utf-8"))
        validate_method(method, protocol)
        if audit.get("status") != "PASS_COMPARISON_PROTOCOL":
            raise ValueError("protocol audit is not PASS")
        if audit.get("protocol_sha256") != sha256_file(protocol_path):
            raise ValueError("protocol changed after audit")
        dataset = protocol["dataset"]
        dev_ids = tuple(dataset["dev_case_ids"])
        locked_ids = tuple(dataset["locked_case_ids"])
        if set(dev_ids).intersection(locked_ids):
            raise ValueError("Dev8 intersects locked cases")

        verse_path = resolve(project_root, protocol["upstream_metric_modules"]["verse_3d"])
        teams_path = resolve(project_root, protocol["upstream_metric_modules"]["teams_style"])
        verse = load_module("comparison_verse2021_3d", verse_path)
        teams = load_module("comparison_teams_style_metrics", teams_path)
        if tuple(dev_ids) != tuple(verse.DEV8_CASES) or tuple(locked_ids) != tuple(verse.LOCKED_CASES):
            raise ValueError("protocol cases do not match audited 3D module")

        slice_path = resolve(project_root, dataset["slice_manifest"])
        case_path = resolve(project_root, dataset["case_metadata"])
        rows = [
            row for row in read_csv(slice_path)
            if row["split"] == dataset["manifest_split"] and row["case_id"] in set(dev_ids)
        ]
        if len(rows) != int(dataset["dev_row_count"]):
            raise ValueError("Dev8 slice count mismatch")
        rows.sort(key=lambda row: (dev_ids.index(row["case_id"]), int(row["slice_idx"])))
        png_reader = lambda path: cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        geometries = verse.prepare_dev8_geometry(slice_path, case_path, png_reader=png_reader)
        volume_collector = verse.PredictionVolumeCollector(geometries)
        teams_root = result_root / "teams_style_2d"
        upstream_teams = resolve(project_root, protocol["upstream_metric_modules"]["teams_repository"])
        teams_collector = teams.TEAMSStyleCollector(teams_root, upstream_root=upstream_teams)
        allowed = {int(protocol["prediction_contract"]["background_label"])}.union(
            int(value) for value in protocol["prediction_contract"]["allowed_foreground_labels"]
        )
        prediction_files = []
        template = protocol["prediction_contract"]["relative_path"]
        for row in rows:
            case_id = row["case_id"]
            slice_idx = int(row["slice_idx"])
            relative = template.format(case_id=case_id, slice_idx=slice_idx)
            pred_path = prediction_root / relative
            expected_hw = (int(row["image_height"]), int(row["image_width"]))
            pred = checked_mask(pred_path, expected_hw, allowed)
            gt = cv2.imread(row["mask_path"], cv2.IMREAD_UNCHANGED)
            if gt is None or gt.ndim != 2 or tuple(gt.shape) != expected_hw:
                raise ValueError("invalid GT PNG {}".format(row["mask_path"]))
            gt_labels = set(int(value) for value in np.unique(gt))
            if gt_labels.difference(allowed):
                raise ValueError("GT PNG has labels outside protocol")
            volume_collector.add(case_id, slice_idx, pred)
            teams_collector.add(case_id, slice_idx, row["image_path"], row["mask_path"], pred, gt)
            prediction_files.append(pred_path)
        volume_collector.assert_complete()

        teams_summary = teams_collector.finalize()
        volumes_root = result_root / "verse_3d"
        scan_results = []
        per_label_rows = []
        for case_id in dev_ids:
            info = geometries[case_id]
            pred_source = volume_collector.source_volume(case_id)
            gt_source = verse.load_nifti(info["gt_path"], info["gt_header"])
            scan_result = verse.evaluate_scan(
                case_id, gt_source, pred_source, info["gt_header"].affine, info["centroids"]
            )
            scan_results.append(scan_result)
            per_label_rows.extend(scan_result["per_label"])
            case_root = volumes_root / "per_scan" / case_id
            case_root.mkdir(parents=True, exist_ok=True)
            verse.write_uint8_like(case_root / "prediction_seg-vert_msk.nii.gz", pred_source, info["gt_header"])
            dump_json(case_root / "metrics.json", scan_result)
        cohort = verse.summarize_cohort(scan_results)
        dump_json(volumes_root / "cohort_summary.json", cohort)
        with (volumes_root / "per_label.jsonl").open("w", encoding="utf-8") as handle:
            for row in per_label_rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

        prediction_hashes = result_root / "PREDICTION_SHA256SUMS.txt"
        prediction_hashes.write_text(
            "\n".join(
                "{}  {}".format(sha256_file(path), path.relative_to(prediction_root).as_posix())
                for path in sorted(prediction_files)
            ) + "\n",
            encoding="utf-8",
        )
        identity = {
            "schema": "diffusionsnake.comparison_evaluation_identity.v1",
            "method": method,
            "method_manifest_path": str(method_path),
            "method_manifest_sha256": sha256_file(method_path),
            "protocol_path": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "protocol_audit_path": str(audit_path),
            "protocol_audit_sha256": sha256_file(audit_path),
            "metric_modules": {
                "verse_3d": {"path": str(verse_path), "sha256": sha256_file(verse_path)},
                "teams_style": {"path": str(teams_path), "sha256": sha256_file(teams_path)},
            },
            "prediction_root": str(prediction_root),
            "prediction_file_count": len(prediction_files),
            "dev_case_ids": list(dev_ids),
            "locked_case_ids": list(locked_ids),
            "locked_accessed": False,
        }
        dump_json(result_root / "EVALUATION_IDENTITY.json", identity)
        final = {
            "schema": "diffusionsnake.comparison_evaluation.v1",
            "status": "PASS_COMPARISON_EVALUATION",
            "method_id": method["method_id"],
            "comparison_regime": method["comparison_regime"],
            "prediction_file_count": len(prediction_files),
            "case_count": len(dev_ids),
            "elapsed_seconds": time.time() - started,
            "primary_3d": cohort,
            "supplementary_teams_style_2d": teams_summary,
        }
        dump_json(result_root / "COMPARISON_EVALUATION.json", final)
        write_hash_manifest(result_root)
        print(json.dumps({
            "status": final["status"],
            "method_id": method["method_id"],
            "result_root": str(result_root),
            "elapsed_seconds": final["elapsed_seconds"],
        }, sort_keys=True))
    except Exception as exc:
        failure = {
            "schema": "diffusionsnake.comparison_evaluation_failure.v1",
            "status": "FAILED_COMPARISON_EVALUATION",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
        dump_json(result_root / "COMPARISON_EVALUATION_FAILURE.json", failure)
        write_hash_manifest(result_root)
        raise


if __name__ == "__main__":
    main()
