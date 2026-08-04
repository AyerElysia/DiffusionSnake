#!/usr/bin/env python3
"""Evaluate a frozen P1 policy with paired contour-level 2D statistics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from p1_core import FourierPolicy, apply_fourier_action, score_contours
from refine_metrics3d import paired_bootstrap


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", action="append", required=True)
    parser.add_argument("--policy-ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--apply-labels", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    return parser.parse_args()


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


def load_caches(paths, active_labels):
    required = (
        "poly", "gt_poly", "gt_dist", "normal", "point_feat", "label",
        "orig_hw", "n_gt_boundary",
    )
    loaded = []
    loaded_paths = []
    volume_ids = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(set(required) - set(data.files))
            if missing:
                raise RuntimeError("cache {} missing {}".format(path, missing))
            arrays = {key: data[key] for key in required}
        valid = (
            np.isfinite(arrays["poly"]).all(axis=(1, 2))
            & np.isfinite(arrays["gt_poly"]).all(axis=(1, 2))
            & np.isfinite(arrays["point_feat"]).all(axis=(1, 2))
            & (arrays["n_gt_boundary"] > 0)
            & (np.nanmedian(arrays["gt_dist"], axis=1) <= 6.0)
        )
        if active_labels is not None:
            valid &= np.isin(arrays["label"], active_labels)
        arrays = {key: value[valid] for key, value in arrays.items()}
        if arrays["poly"].shape[0] == 0:
            continue
        loaded.append(arrays)
        loaded_paths.append(path)
        volume_ids.append(np.full(
            arrays["poly"].shape[0], Path(path).stem, dtype="U96"))
    if not loaded:
        raise RuntimeError("no valid in-scope contours")
    reference = loaded[0]
    for path, arrays in zip(loaded_paths[1:], loaded[1:]):
        for key in required:
            if arrays[key].shape[1:] != reference[key].shape[1:]:
                raise RuntimeError("incompatible {} shape in {}".format(key, path))
    combined = {
        key: np.concatenate([arrays[key] for arrays in loaded], axis=0)
        for key in required
    }
    combined["volume_id"] = np.concatenate(volume_ids)
    return combined


@torch.no_grad()
def refine(policy, arrays, device, batch_size, max_displacement):
    outputs = []
    fields = []
    count = int(arrays["poly"].shape[0])
    for start in range(0, count, int(batch_size)):
        stop = min(start + int(batch_size), count)
        indices = slice(start, stop)
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
        outputs.append(output.cpu().numpy())
        fields.append(field.cpu().numpy())
    return np.concatenate(outputs), np.concatenate(fields)


def summarize(base, treatment, field, max_displacement, n_boot, seed):
    nsd = paired_bootstrap(
        base["nsd"], treatment["nsd"], n_boot=n_boot, seed=seed)
    distance = paired_bootstrap(
        -base["mean_distance"], -treatment["mean_distance"],
        n_boot=n_boot, seed=seed)
    burr = paired_bootstrap(
        -base["burr"], -treatment["burr"], n_boot=n_boot, seed=seed)
    objective = (
        treatment["nsd"] - base["nsd"] - 0.06 * treatment["burr"])
    abs_field = np.abs(field)
    return {
        "count": int(field.shape[0]),
        "nsd2": nsd,
        "mean_distance_reduction": distance,
        "burr_reduction": burr,
        "exact_training_objective_mean": float(objective.mean()),
        "displacement_abs_mean": float(abs_field.mean()),
        "displacement_abs_p95": float(np.percentile(abs_field, 95.0)),
        "displacement_abs_max": float(abs_field.max()),
        "saturation_fraction": float(np.mean(
            abs_field >= 0.95 * float(max_displacement))),
    }


def main():
    args = parse_args()
    active_labels = parse_labels(args.apply_labels)
    arrays = load_caches(args.cache, active_labels)
    device = torch.device(args.device)
    checkpoint = torch.load(args.policy_ckpt, map_location="cpu")
    policy = FourierPolicy(feature_dim=int(arrays["point_feat"].shape[-1])).to(device)
    policy.load_state_dict(checkpoint["policy"], strict=True)
    policy.eval()
    max_displacement = float(
        checkpoint.get("args", {}).get("max_displacement", 3.0))
    refined, field = refine(
        policy, arrays, device, args.batch_size, max_displacement)
    base_scores = score_contours(
        arrays["poly"], arrays["gt_poly"], arrays["orig_hw"])
    refined_scores = score_contours(
        refined, arrays["gt_poly"], arrays["orig_hw"])

    result = {
        "policy_checkpoint": os.path.abspath(args.policy_ckpt),
        "policy_step": int(checkpoint.get("step", -1)),
        "policy_seed": checkpoint.get("args", {}).get("seed"),
        "caches": [os.path.abspath(path) for path in args.cache],
        "apply_labels": active_labels,
        "bootstrap": {"n_boot": args.n_boot, "seed": args.bootstrap_seed},
        "pooled": summarize(
            base_scores, refined_scores, field, max_displacement,
            args.n_boot, args.bootstrap_seed),
        "per_volume": {},
        "per_label": {},
    }
    for name, values in (
        ("per_volume", np.unique(arrays["volume_id"])),
        ("per_label", np.unique(arrays["label"])),
    ):
        source = arrays["volume_id"] if name == "per_volume" else arrays["label"]
        for value in values:
            selected = source == value
            key = str(value)
            result[name][key] = summarize(
                {metric: scores[selected] for metric, scores in base_scores.items()},
                {metric: scores[selected] for metric, scores in refined_scores.items()},
                field[selected], max_displacement,
                args.n_boot, args.bootstrap_seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("[P1-2D] " + json.dumps(result["pooled"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
