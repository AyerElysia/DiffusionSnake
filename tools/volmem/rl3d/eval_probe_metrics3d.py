"""Phase0-B closure: run the 3D refinement metric suite on real probe dumps.

CPU-only, read-only. Consumes Phase0-A npz dumps (predicted contours) plus the
dataset's GT masks, rasterises predictions exactly the way the probe did
(cv2.fillPoly on np.rint(poly), labels in stored order), and runs
refine_metrics3d.evaluate_volume per volume.

Self-check: the probe recorded per-slice pred_fg / gt_fg pixel counts in
index.json. This runner recomputes both and aborts on any mismatch, so the
rasterisation here is guaranteed bit-compatible with what the probe saw.
"""
import argparse
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
    parser.add_argument("--cache-root", default=None)
    return parser.parse_args()


ARGS = parse_args()
os.environ["CFG_FILE"] = ARGS.cfg_file
sys.argv = [sys.argv[0], "--cfg_file", ARGS.cfg_file]

import cv2
import numpy as np

from lib.config import cfg
from lib.evaluators.sagittal_2d_fixed import Evaluator
from volmem.adapters import (
    configure_single_slice_compatibility,
    make_single_slice_dataset_class,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refine_metrics3d as m3d


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

    summary = {
        "probe_dir": ARGS.probe_dir,
        "checkpoint": probe_index.get("checkpoint"),
        "checkpoint_step": probe_index.get("checkpoint_step"),
        "memory_mode": probe_index.get("memory_mode"),
        "box_mode": probe_index.get("box_mode"),
        "volumes": {},
        "per_class_rows": [],
    }

    for vol_entry in probe_index["volumes"]:
        volume_id = vol_entry["volume_id"]
        npz = np.load(os.path.join(ARGS.probe_dir, vol_entry["npz"]))
        polys = npz["poly"]          # [N, P, 2] float32
        labels = npz["label"]        # [N] already actual GT label values
        slice_of = npz["slice_idx"]  # [N]

        items = sorted(by_volume[volume_id])
        slice_ids = [s for s, _ in items]
        if sorted({r["slice_idx"] for r in vol_entry["slices"]}) != slice_ids:
            raise RuntimeError(volume_id + ": probe/dataset slice sets differ")
        contiguous = all(b - a == 1 for a, b in zip(slice_ids, slice_ids[1:]))

        gt_stack, pred_stack = [], []
        probe_counts = {r["slice_idx"]: (r["pred_fg"], r["gt_fg"])
                        for r in vol_entry["slices"]}
        for slice_idx, ds_idx in items:
            gt = np.asarray(
                evaluator._read_mask(dataset.records[ds_idx]["mask_path"]))
            pred = np.zeros(gt.shape, dtype=np.uint16)
            for n in np.flatnonzero(slice_of == slice_idx):
                poly = polys[n]
                if poly.shape[0] >= 3:
                    cv2.fillPoly(pred, [np.rint(poly).astype(np.int32)],
                                 int(labels[n]))
            want_pred, want_gt = probe_counts[slice_idx]
            got_pred, got_gt = int((pred > 0).sum()), int((gt > 0).sum())
            if got_pred != want_pred or got_gt != want_gt:
                raise RuntimeError(
                    "{} slice {}: fg mismatch pred {}!={} gt {}!={}".format(
                        volume_id, slice_idx, got_pred, want_pred,
                        got_gt, want_gt))
            gt_stack.append(gt.astype(np.int32))
            pred_stack.append(pred.astype(np.int32))

        report = m3d.evaluate_volume(
            np.stack(pred_stack), np.stack(gt_stack), axis=0)
        report["volume_id"] = volume_id
        report["contiguous_slices"] = contiguous
        m3d.save_report(report, os.path.join(
            ARGS.out_dir, volume_id + ".metrics3d.json"))

        fg = report["foreground"]
        summary["volumes"][volume_id] = {
            "n_slices": len(slice_ids),
            "contiguous": contiguous,
            "role_counts": report["role_counts"],
            "dice": fg.get("dice"),
            "nsd@1": fg.get("nsd@1"),
            "nsd@2": fg.get("nsd@2"),
            "nsd@3": fg.get("nsd@3"),
            "asd": fg.get("asd"),
            "hd95": fg.get("hd95"),
            "area_lap_excess": fg.get("area_lap_excess"),
            "area_lap_deficit": fg.get("area_lap_deficit"),
            "slice_dice_by_role": report["slice_dice_by_role"],
        }
        for label, entry in report["per_class"].items():
            summary["per_class_rows"].append({
                "volume_id": volume_id,
                "label": int(label),
                "dice": entry.get("dice"),
                "nsd@1": entry.get("nsd@1"),
                "nsd@2": entry.get("nsd@2"),
                "asd": entry.get("asd"),
                "hd95": entry.get("hd95"),
                "gt_voxels": entry.get("gt_voxels"),
                "area_lap_excess": entry.get("area_lap_excess"),
                "area_lap_deficit": entry.get("area_lap_deficit"),
            })
        print("[m3d] {} slices={} dice={:.4f} nsd@1={:.4f} nsd@2={:.4f} "
              "asd={:.3f} hd95={:.3f}".format(
                  volume_id, len(slice_ids), fg.get("dice", float("nan")),
                  fg.get("nsd@1", float("nan")), fg.get("nsd@2", float("nan")),
                  fg.get("asd", float("nan")), fg.get("hd95", float("nan"))),
              flush=True)

    vols = summary["volumes"]
    pooled = {}
    for key in ("dice", "nsd@1", "nsd@2", "nsd@3", "asd", "hd95",
                "area_lap_excess", "area_lap_deficit"):
        vals = [v[key] for v in vols.values()
                if v.get(key) is not None and np.isfinite(v[key])]
        pooled[key] = float(np.mean(vals)) if vals else None
    rows = summary["per_class_rows"]
    finite = [r for r in rows if r["dice"] is not None
              and np.isfinite(r["dice"])]
    dead = [r for r in finite if r["dice"] < 0.1]
    weak = [r for r in finite if 0.1 <= r["dice"] < 0.5]
    alive = [r for r in finite if r["dice"] >= 0.5]
    pooled["n_class_instances"] = len(finite)
    pooled["n_dead(dice<0.1)"] = len(dead)
    pooled["n_weak(0.1-0.5)"] = len(weak)
    pooled["n_alive(>=0.5)"] = len(alive)
    pooled["alive_mean_dice"] = (
        float(np.mean([r["dice"] for r in alive])) if alive else None)
    pooled["alive_mean_nsd@1"] = (
        float(np.mean([r["nsd@1"] for r in alive
                       if np.isfinite(r["nsd@1"])])) if alive else None)
    pooled["alive_mean_nsd@2"] = (
        float(np.mean([r["nsd@2"] for r in alive
                       if np.isfinite(r["nsd@2"])])) if alive else None)
    summary["pooled"] = pooled

    m3d.save_report(summary, os.path.join(ARGS.out_dir, "summary.json"))
    print("[m3d] pooled:", json.dumps(pooled, sort_keys=True), flush=True)
    print("[m3d] done volumes={}".format(len(vols)), flush=True)


if __name__ == "__main__":
    main()
