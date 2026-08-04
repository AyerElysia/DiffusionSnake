"""Phase0-E: freeze the RL evaluation protocol into content-hashed artifacts.

Emits three files under --out-dir:
  eval_slices_v1.csv   one row per val slice: volume, slice, role, gt stats
  refine_subset_v1.csv one row per predicted contour: in_scope flag + reason
  eval_spec_v1.json    the protocol itself + md5 of both CSVs

Deterministic given the probe dump + GT masks; rerunning must reproduce the
hashes bit-for-bit, which is the point: every future A/B uses literally this
slice list, not "the val split as currently configured".
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max-dist-in-scope", type=float, default=6.0)
    parser.add_argument("--cache-root", default=None)
    return parser.parse_args()


ARGS = parse_args()
os.environ["CFG_FILE"] = ARGS.cfg_file
sys.argv = [sys.argv[0], "--cfg_file", ARGS.cfg_file]

import numpy as np

from lib.config import cfg
from lib.evaluators.sagittal_2d_fixed import Evaluator
from volmem.adapters import (
    configure_single_slice_compatibility,
    make_single_slice_dataset_class,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refine_metrics3d as m3d


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if ARGS.cache_root and os.path.isdir(ARGS.cache_root):
        cfg.locate_feat_cache_root = ARGS.cache_root
    configure_single_slice_compatibility(cfg)
    cfg.test.dataset = "VolMemVal" if ARGS.split == "val" else "VolMemTest"
    os.makedirs(ARGS.out_dir, exist_ok=True)
    cfg.result_dir = os.path.join(ARGS.out_dir, "_evaluator")

    dataset = make_single_slice_dataset_class()(
        ann_file=str(cfg.volmem.manifest_file),
        data_root=str(cfg.volmem.data_root),
        split=ARGS.split,
    )
    by_volume = defaultdict(list)
    for ds_idx, record in enumerate(dataset.records):
        by_volume[str(record["case_id"])].append(
            (int(record["slice_idx"]), ds_idx))
    evaluator = Evaluator(cfg.result_dir)

    with open(os.path.join(ARGS.probe_dir, "index.json")) as handle:
        probe_index = json.load(handle)

    slice_rows = []
    contour_rows = []
    role_totals = defaultdict(int)
    scope_totals = defaultdict(int)

    for vol_entry in probe_index["volumes"]:
        volume_id = vol_entry["volume_id"]
        npz = np.load(os.path.join(ARGS.probe_dir, vol_entry["npz"]))
        gt_dist = npz["gt_dist"]      # [N, P]
        labels = npz["label"]
        slice_of = npz["slice_idx"]

        items = sorted(by_volume[volume_id])
        gt_stack = [np.asarray(
            evaluator._read_mask(dataset.records[ds]["mask_path"]))
            for _, ds in items]
        roles = m3d.slice_roles(np.stack(gt_stack).astype(np.int32), axis=0)

        probe_rows = {r["slice_idx"]: r for r in vol_entry["slices"]}
        for (slice_idx, _), role in zip(items, roles):
            row = probe_rows[slice_idx]
            slice_rows.append([volume_id, slice_idx, role,
                               len(row["gt_labels"]), row["gt_fg"],
                               row["pred_fg"], row["n_contours"]])
            role_totals[role] += 1

        for n in range(labels.shape[0]):
            dists = gt_dist[n]
            finite = dists[np.isfinite(dists)]
            med = float(np.median(finite)) if finite.size else float("nan")
            if not np.isfinite(med):
                reason = "no_gt_match"
            elif med > ARGS.max_dist_in_scope:
                reason = "detection_failure"
            else:
                reason = "in_scope"
            scope_totals[reason] += 1
            contour_rows.append([volume_id, int(slice_of[n]), int(labels[n]),
                                 round(med, 4) if np.isfinite(med) else "",
                                 1 if reason == "in_scope" else 0, reason])

    slices_csv = os.path.join(ARGS.out_dir, "eval_slices_v1.csv")
    with open(slices_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["volume_id", "slice_idx", "role", "n_gt_labels",
                         "gt_fg", "pred_fg", "n_contours"])
        writer.writerows(slice_rows)

    subset_csv = os.path.join(ARGS.out_dir, "refine_subset_v1.csv")
    with open(subset_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["volume_id", "slice_idx", "label",
                         "median_gt_dist", "in_scope", "reason"])
        writer.writerows(contour_rows)

    spec = {
        "version": "v1",
        "frozen_from": {
            "probe_dir": ARGS.probe_dir,
            "checkpoint": probe_index.get("checkpoint"),
            "checkpoint_step": probe_index.get("checkpoint_step"),
            "box_mode": probe_index.get("box_mode"),
            "split": ARGS.split,
        },
        "counts": {
            "volumes": len(probe_index["volumes"]),
            "slices": len(slice_rows),
            "slices_by_role": dict(role_totals),
            "contours": len(contour_rows),
            "contours_by_scope": dict(scope_totals),
        },
        "endpoints": {
            "primary": ["surface nsd@1", "surface nsd@2"],
            "primary_rationale": (
                "baseline 0.720/0.878 leaves headroom; nsd@0.5 is label-noise "
                "dominated, nsd@3 saturated at 0.958"),
            "secondary": ["asd", "hd95", "dice (sanity only)",
                          "area_lap_excess", "area_lap_deficit",
                          "slice_dice_by_role (end_cap is the hard stratum)"],
            "anti_hack_guards": (
                "accuracy term must dominate; watch area_lap_deficit for "
                "extrusion, never use abs_diff alone (see Phase0-B)"),
        },
        "statistics": {
            "volume_level": {
                "test": "paired bootstrap over 12 volumes",
                "n_boot": 10000, "seed": 20260731, "ties_eps": 1e-9,
                "caveat": "12 clusters is weak; report but do not gate on it",
            },
            "slice_level": {
                "test": "paired bootstrap + W/T/L over non-empty slices",
                "n": int(len(slice_rows)
                         - role_totals.get("empty", 0)),
                "exclusion": "empty GT slices excluded (evaluator scores them "
                             "dice=1.0 and they dilute every mean)",
            },
            "contour_level": {
                "test": "paired mean gt_dist per contour, W/T/L",
                "n_in_scope": scope_totals.get("in_scope", 0),
                "note": "highest power; matches reward granularity",
            },
        },
        "scope_rule": {
            "in_scope": "median per-contour gt_dist <= {}".format(
                ARGS.max_dist_in_scope),
            "rationale": "beyond that it is a detection failure, not "
                         "refinable by boundary actions (p99 of gt_dist ~8.3)",
        },
        "data_hygiene": (
            "these 12 val volumes are EVAL-ONLY. RL rollouts/updates must use "
            "train-split volumes; never optimise against this set."),
        "md5": {"eval_slices_v1.csv": md5_of(slices_csv),
                "refine_subset_v1.csv": md5_of(subset_csv)},
    }
    spec_path = os.path.join(ARGS.out_dir, "eval_spec_v1.json")
    m3d.save_report(spec, spec_path)
    print("[spec] slices={} (empty={}) contours={} in_scope={} "
          "detection_failure={} no_gt_match={}".format(
              len(slice_rows), role_totals.get("empty", 0),
              len(contour_rows), scope_totals.get("in_scope", 0),
              scope_totals.get("detection_failure", 0),
              scope_totals.get("no_gt_match", 0)), flush=True)
    print("[spec] md5 slices={} subset={}".format(
        spec["md5"]["eval_slices_v1.csv"],
        spec["md5"]["refine_subset_v1.csv"]), flush=True)
    print("[spec] wrote {}".format(spec_path), flush=True)


if __name__ == "__main__":
    main()
