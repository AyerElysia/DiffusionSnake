#!/usr/bin/env python3
"""Build an explicitly diagnostic detector-geometry + oracle-class cache."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def number(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def iou(a: dict[str, float], b: dict[str, float]) -> float:
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def greedy_match(
    predictions: list[dict[str, float]],
    ground_truth: list[dict[str, Any]],
    min_iou: float,
) -> list[tuple[int, int, float]]:
    candidates = []
    for pred_index, prediction in enumerate(predictions):
        for gt_index, gt in enumerate(ground_truth):
            overlap = iou(prediction, gt)
            if overlap >= min_iou:
                candidates.append((overlap, pred_index, gt_index))
    candidates.sort(reverse=True)
    used_predictions: set[int] = set()
    used_gt: set[int] = set()
    matches = []
    for overlap, pred_index, gt_index in candidates:
        if pred_index in used_predictions or gt_index in used_gt:
            continue
        used_predictions.add(pred_index)
        used_gt.add(gt_index)
        matches.append((pred_index, gt_index, overlap))
    return matches


def load_predictions(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                image_path = row["image_path"]
                if image_path in result:
                    raise RuntimeError(f"duplicate prediction image: {image_path}")
                result[image_path] = row
    return result


def load_ground_truth(
    path: Path,
    cases: set[str],
    min_area: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "validation" or row["case_id"] not in cases:
                continue
            grouped[(row["image_path"], number(row, "label_id"))].append(row)
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (image_path, label_id), rows in grouped.items():
        row = max(
            rows,
            key=lambda item: (
                number(item, "mask_pixel_count"),
                number(item, "bbox_width") * number(item, "bbox_height"),
                -number(item, "component_id"),
            ),
        )
        area = number(row, "bbox_width") * number(row, "bbox_height")
        if area < min_area:
            continue
        result[image_path].append(
            {
                "x1": float(row["x_min"]),
                "y1": float(row["y_min"]),
                "x2": float(row["x_max"]),
                "y2": float(row["y_max"]),
                "label_id": label_id,
                "class_id": label_id - 1,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--component-manifest", type=Path, required=True)
    parser.add_argument("--slice-manifest", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--min-iou", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not 0.0 <= args.min_iou <= 1.0:
        raise ValueError("--min-iou must be in [0, 1]")

    cases = set(args.cases)
    predictions = load_predictions(args.predictions)
    ground_truth = load_ground_truth(args.component_manifest, cases, args.min_area)
    selected_slices = []
    with args.slice_manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "validation" and row["case_id"] in cases:
                selected_slices.append(row)
    selected_slices.sort(key=lambda row: (row["case_id"], number(row, "slice_idx")))
    if not selected_slices:
        raise RuntimeError("no selected validation slices")

    records = []
    per_case: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "slices": 0,
            "gt_boxes": 0,
            "detector_boxes": 0,
            "oracle_matched_boxes": 0,
            "dropped_unmatched_detector_boxes": 0,
            "missed_gt_boxes": 0,
            "matched_ious": [],
        }
    )
    all_ious = []
    for slice_row in selected_slices:
        image_path = slice_row["image_path"]
        if image_path not in predictions:
            raise KeyError(f"missing detector predictions for {image_path}")
        prediction_row = predictions[image_path]
        pred_boxes = [
            {key: float(box[key]) for key in ("x1", "y1", "x2", "y2")}
            for box in prediction_row.get("predictions", [])
        ]
        gt_boxes = ground_truth.get(image_path, [])
        matches = greedy_match(pred_boxes, gt_boxes, args.min_iou)
        instances = []
        for pred_index, gt_index, overlap in matches:
            pred = pred_boxes[pred_index]
            gt = gt_boxes[gt_index]
            instances.append(
                {
                    "bbox": [pred["x1"], pred["y1"], pred["x2"], pred["y2"]],
                    "score": 1.0,
                    "class_id": gt["class_id"],
                    "label_id": gt["label_id"],
                    "source": "LocateAnything_base1500_geometry_oracle_class_diagnostic",
                    "oracle_match_iou": overlap,
                }
            )
            all_ious.append(overlap)
        instances.sort(key=lambda item: (item["class_id"], item["bbox"][1], item["bbox"][0]))
        case_stats = per_case[slice_row["case_id"]]
        case_stats["slices"] += 1
        case_stats["gt_boxes"] += len(gt_boxes)
        case_stats["detector_boxes"] += len(pred_boxes)
        case_stats["oracle_matched_boxes"] += len(matches)
        case_stats["dropped_unmatched_detector_boxes"] += len(pred_boxes) - len(matches)
        case_stats["missed_gt_boxes"] += len(gt_boxes) - len(matches)
        case_stats["matched_ious"].extend(item[2] for item in matches)
        records.append(
            {
                "id": Path(image_path).stem,
                "img_path": image_path,
                "image_path": image_path,
                "width": number(slice_row, "image_width"),
                "height": number(slice_row, "image_height"),
                "instances": instances,
            }
        )

    summarized_cases = {}
    for case_id, stats in sorted(per_case.items()):
        case_ious = stats.pop("matched_ious")
        summarized_cases[case_id] = {
            **stats,
            "mean_matched_iou": mean(case_ious) if case_ious else 0.0,
        }
    total_gt = sum(item["gt_boxes"] for item in summarized_cases.values())
    total_detector = sum(item["detector_boxes"] for item in summarized_cases.values())
    total_matched = sum(item["oracle_matched_boxes"] for item in summarized_cases.values())
    summary = {
        "diagnostic_only": True,
        "not_production_numbering": True,
        "cases": sorted(cases),
        "slices": len(records),
        "min_area": args.min_area,
        "min_iou": args.min_iou,
        "gt_boxes": total_gt,
        "detector_boxes": total_detector,
        "oracle_matched_boxes": total_matched,
        "dropped_unmatched_detector_boxes": total_detector - total_matched,
        "missed_gt_boxes": total_gt - total_matched,
        "gt_box_recall": total_matched / total_gt if total_gt else 0.0,
        "detector_box_keep_rate": total_matched / total_detector if total_detector else 0.0,
        "mean_matched_iou": mean(all_ious) if all_ious else 0.0,
        "per_case": summarized_cases,
    }
    payload = {
        "format": "volmem_locany_cache_v1",
        "coordinate_space": "original_image_pixels",
        "diagnostic_contract": (
            "LocateAnything base1500 geometry; one-to-one GT assignment supplies oracle class; "
            "unmatched detector boxes are dropped; never use as production numbering"
        ),
        "samples": records,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
