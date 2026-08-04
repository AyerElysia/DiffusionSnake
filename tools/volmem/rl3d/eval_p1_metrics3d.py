"""Evaluate a frozen P1 contour policy with read-only 3D volume metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--policy-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--apply-labels", default="",
        help="Comma-separated labels/ranges, e.g. 16-24. Empty applies all.")
    return parser.parse_args()


ARGS = parse_args()
os.environ["CFG_FILE"] = ARGS.cfg_file
sys.argv = [sys.argv[0], "--cfg_file", ARGS.cfg_file]

import cv2
import numpy as np
import torch

from lib.config import cfg
from lib.evaluators.sagittal_2d_fixed import Evaluator
from volmem.adapters import (
    configure_single_slice_compatibility,
    make_single_slice_dataset_class,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_core import FourierPolicy, apply_fourier_action
import refine_metrics3d as m3d


def parse_labels(spec):
    if not str(spec).strip():
        return None
    labels = set()
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            labels.update(range(int(left), int(right) + 1))
        else:
            labels.add(int(token))
    return sorted(labels)


def load_policy(npz_path, checkpoint_path, device):
    with np.load(npz_path, allow_pickle=False) as data:
        feature_dim = int(data["point_feat"].shape[-1])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    policy = FourierPolicy(feature_dim=feature_dim).to(device)
    policy.load_state_dict(checkpoint["policy"], strict=True)
    policy.eval()
    return policy, checkpoint


@torch.no_grad()
def refine_arrays(policy, arrays, device, batch_size, active_labels,
                  max_displacement):
    refined = arrays["poly"].astype(np.float32, copy=True)
    active = np.ones((refined.shape[0],), dtype=bool)
    if active_labels is not None:
        active = np.isin(arrays["label"], active_labels)
    active_indices = np.flatnonzero(active)
    fields = []
    for start in range(0, active_indices.size, int(batch_size)):
        indices = active_indices[start:start + int(batch_size)]
        point_feat = torch.as_tensor(
            arrays["point_feat"][indices], device=device, dtype=torch.float32)
        poly = torch.as_tensor(
            arrays["poly"][indices], device=device, dtype=torch.float32)
        normal = torch.as_tensor(
            arrays["normal"][indices], device=device, dtype=torch.float32)
        label = torch.as_tensor(
            arrays["label"][indices], device=device, dtype=torch.long)
        coefficients = policy(point_feat, poly, normal, label)
        output, field = apply_fourier_action(
            poly, normal, coefficients, policy.basis,
            max_displacement=max_displacement)
        refined[indices] = output.cpu().numpy()
        fields.append(field.cpu().numpy())
    flat_field = np.concatenate(fields, axis=0) if fields else np.zeros((0, 128))
    saturation = float(np.mean(
        np.abs(flat_field) >= 0.95 * float(max_displacement)
    )) if flat_field.size else 0.0
    return refined, active, saturation


def metric_projection(report):
    foreground = report["foreground"]
    return {
        key: foreground.get(key)
        for key in (
            "dice", "nsd@1", "nsd@2", "nsd@3", "asd", "hd95",
            "area_lap_excess", "area_lap_deficit", "centroid_drift",
        )
    }


def main():
    device = torch.device(ARGS.device)
    active_labels = parse_labels(ARGS.apply_labels)
    os.makedirs(ARGS.out_dir, exist_ok=True)
    with open(os.path.join(ARGS.cache_dir, "index.json")) as handle:
        cache_index = json.load(handle)
    if not cache_index.get("volumes"):
        raise RuntimeError("cache contains no volumes")

    first_npz = os.path.join(
        ARGS.cache_dir, cache_index["volumes"][0]["npz"])
    policy, policy_checkpoint = load_policy(
        first_npz, ARGS.policy_ckpt, device)
    max_displacement = float(
        policy_checkpoint.get("args", {}).get("max_displacement", 3.0))

    if ARGS.cache_root and os.path.isdir(ARGS.cache_root):
        cfg.locate_feat_cache_root = ARGS.cache_root
    configure_single_slice_compatibility(cfg)
    cfg.test.dataset = {
        "train": "VolMemTrain",
        "val": "VolMemVal",
        "test": "VolMemTest",
    }[ARGS.split]
    cfg.result_dir = os.path.join(ARGS.out_dir, "_evaluator")
    dataset = make_single_slice_dataset_class()(
        ann_file=str(cfg.volmem.manifest_file),
        data_root=str(cfg.volmem.data_root),
        split=ARGS.split,
    )
    by_volume = defaultdict(list)
    for dataset_index, record in enumerate(dataset.records):
        by_volume[str(record["case_id"])].append(
            (int(record["slice_idx"]), dataset_index))
    evaluator = Evaluator(cfg.result_dir)

    summary = {
        "cache_dir": ARGS.cache_dir,
        "base_checkpoint": cache_index.get("checkpoint"),
        "base_checkpoint_step": cache_index.get("checkpoint_step"),
        "policy_checkpoint": ARGS.policy_ckpt,
        "policy_step": int(policy_checkpoint.get("step", -1)),
        "policy_seed": policy_checkpoint.get("args", {}).get("seed"),
        "apply_labels": active_labels,
        "max_displacement": max_displacement,
        "volumes": {},
        "per_class_rows": [],
    }

    for volume_entry in cache_index["volumes"]:
        volume_id = volume_entry["volume_id"]
        npz_path = os.path.join(ARGS.cache_dir, volume_entry["npz"])
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        refined_poly, active, saturation = refine_arrays(
            policy, arrays, device, ARGS.batch_size, active_labels,
            max_displacement)

        items = sorted(by_volume[volume_id])
        slice_ids = [slice_index for slice_index, _ in items]
        cache_slice_ids = sorted(
            row["slice_idx"] for row in volume_entry["slices"])
        if cache_slice_ids != slice_ids:
            raise RuntimeError(volume_id + ": cache/dataset slice sets differ")
        cached_counts = {
            row["slice_idx"]: (row["pred_fg"], row["gt_fg"])
            for row in volume_entry["slices"]
        }

        gt_stack, base_stack, refined_stack = [], [], []
        for slice_index, dataset_index in items:
            gt = np.asarray(evaluator._read_mask(
                dataset.records[dataset_index]["mask_path"]))
            base = np.zeros(gt.shape, dtype=np.uint16)
            refined = np.zeros(gt.shape, dtype=np.uint16)
            for contour_index in np.flatnonzero(
                    arrays["slice_idx"] == slice_index):
                label = int(arrays["label"][contour_index])
                cv2.fillPoly(
                    base,
                    [np.rint(arrays["poly"][contour_index]).astype(np.int32)],
                    label,
                )
                cv2.fillPoly(
                    refined,
                    [np.rint(refined_poly[contour_index]).astype(np.int32)],
                    label,
                )
            expected_pred, expected_gt = cached_counts[slice_index]
            if int((base > 0).sum()) != expected_pred:
                raise RuntimeError(
                    "{} slice {}: base rasterization mismatch".format(
                        volume_id, slice_index))
            if int((gt > 0).sum()) != expected_gt:
                raise RuntimeError(
                    "{} slice {}: GT rasterization mismatch".format(
                        volume_id, slice_index))
            gt_stack.append(gt.astype(np.int32))
            base_stack.append(base.astype(np.int32))
            refined_stack.append(refined.astype(np.int32))

        gt_volume = np.stack(gt_stack)
        base_volume = np.stack(base_stack)
        refined_volume = np.stack(refined_stack)
        base_report = m3d.evaluate_volume(base_volume, gt_volume, axis=0)
        refined_report = m3d.evaluate_volume(refined_volume, gt_volume, axis=0)
        m3d.save_report(base_report, os.path.join(
            ARGS.out_dir, volume_id + ".base.metrics3d.json"))
        m3d.save_report(refined_report, os.path.join(
            ARGS.out_dir, volume_id + ".refined.metrics3d.json"))

        base_metrics = metric_projection(base_report)
        refined_metrics = metric_projection(refined_report)
        deltas = {
            key: (
                float(refined_metrics[key] - base_metrics[key])
                if base_metrics[key] is not None
                and refined_metrics[key] is not None
                and np.isfinite(base_metrics[key])
                and np.isfinite(refined_metrics[key])
                else None
            )
            for key in base_metrics
        }
        summary["volumes"][volume_id] = {
            "n_slices": len(slice_ids),
            "n_contours": int(arrays["poly"].shape[0]),
            "n_active_contours": int(active.sum()),
            "saturation_fraction": saturation,
            "base": base_metrics,
            "refined": refined_metrics,
            "delta": deltas,
            "base_slice_dice_by_role": base_report["slice_dice_by_role"],
            "refined_slice_dice_by_role": refined_report["slice_dice_by_role"],
        }
        for label, base_entry in base_report["per_class"].items():
            refined_entry = refined_report["per_class"][label]
            row = {
                "volume_id": volume_id,
                "label": int(label),
                "active": active_labels is None or int(label) in active_labels,
            }
            for key in ("dice", "nsd@1", "nsd@2", "nsd@3", "asd", "hd95"):
                row["base_" + key] = base_entry.get(key)
                row["refined_" + key] = refined_entry.get(key)
            summary["per_class_rows"].append(row)
        print(
            "[P1-3D] {} active={}/{} dNSD1={:+.5f} dNSD2={:+.5f} "
            "dDice={:+.5f} dASD={:+.5f} dHD95={:+.5f}".format(
                volume_id, int(active.sum()), int(active.size),
                deltas["nsd@1"], deltas["nsd@2"], deltas["dice"],
                deltas["asd"], deltas["hd95"]),
            flush=True,
        )

    metric_names = (
        "dice", "nsd@1", "nsd@2", "nsd@3", "asd", "hd95",
        "area_lap_excess", "area_lap_deficit", "centroid_drift",
    )
    pooled = {}
    for key in metric_names:
        base_values = np.asarray([
            row["base"][key] for row in summary["volumes"].values()
        ], dtype=np.float64)
        refined_values = np.asarray([
            row["refined"][key] for row in summary["volumes"].values()
        ], dtype=np.float64)
        finite = np.isfinite(base_values) & np.isfinite(refined_values)
        pooled[key] = {
            "base_mean": float(base_values[finite].mean()),
            "refined_mean": float(refined_values[finite].mean()),
            "mean_delta": float(
                (refined_values[finite] - base_values[finite]).mean()),
            "wins": int((refined_values[finite] > base_values[finite]).sum()),
            "ties": int((refined_values[finite] == base_values[finite]).sum()),
            "losses": int((refined_values[finite] < base_values[finite]).sum()),
        }
    summary["pooled"] = pooled
    pooled_roles = {}
    for role in ("interior", "end_cap", "transition"):
        base_values, refined_values = [], []
        for row in summary["volumes"].values():
            base_value = row["base_slice_dice_by_role"].get(role, {}).get("mean")
            refined_value = row["refined_slice_dice_by_role"].get(role, {}).get("mean")
            if (
                base_value is not None and refined_value is not None
                and np.isfinite(base_value) and np.isfinite(refined_value)
            ):
                base_values.append(base_value)
                refined_values.append(refined_value)
        base_values = np.asarray(base_values, dtype=np.float64)
        refined_values = np.asarray(refined_values, dtype=np.float64)
        pooled_roles[role] = {
            "n_volumes": int(base_values.size),
            "base_mean": float(base_values.mean()) if base_values.size else None,
            "refined_mean": (
                float(refined_values.mean()) if refined_values.size else None),
            "mean_delta": (
                float((refined_values - base_values).mean())
                if base_values.size else None),
        }
    summary["pooled_slice_dice_by_role"] = pooled_roles

    per_class_pooled = {}
    for active_name, active_value in (("active", True), ("inactive", False)):
        selected = [
            row for row in summary["per_class_rows"]
            if row["active"] is active_value
        ]
        per_class_pooled[active_name] = {"n": len(selected)}
        for key in ("dice", "nsd@1", "nsd@2", "nsd@3", "asd", "hd95"):
            base_values = np.asarray([
                row["base_" + key] for row in selected
            ], dtype=np.float64)
            refined_values = np.asarray([
                row["refined_" + key] for row in selected
            ], dtype=np.float64)
            finite = np.isfinite(base_values) & np.isfinite(refined_values)
            per_class_pooled[active_name][key] = {
                "base_mean": (
                    float(base_values[finite].mean()) if finite.any() else None),
                "refined_mean": (
                    float(refined_values[finite].mean()) if finite.any() else None),
                "mean_delta": (
                    float((refined_values[finite] - base_values[finite]).mean())
                    if finite.any() else None),
                "wins": int((refined_values[finite] > base_values[finite]).sum()),
                "ties": int((refined_values[finite] == base_values[finite]).sum()),
                "losses": int((refined_values[finite] < base_values[finite]).sum()),
            }
    summary["per_class_pooled"] = per_class_pooled
    m3d.save_report(summary, os.path.join(ARGS.out_dir, "summary.json"))
    print("[P1-3D][DONE] " + json.dumps(pooled, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
