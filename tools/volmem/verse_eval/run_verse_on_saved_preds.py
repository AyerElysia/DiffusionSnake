#!/usr/bin/env python3
"""Run VerSe-2021 standard evaluation on saved prediction masks.

Loads GT and prediction PNG stacks, resamples to 1mm isotropic,
computes per-vertebra Dice, ID rate, HD95(mm), NSD@1mm/2mm.
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import cv2
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add verse_eval to path to import verse_metrics directly
_VERSE_EVAL_DIR = pathlib.Path(__file__).resolve().parent
if str(_VERSE_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_VERSE_EVAL_DIR))

import verse_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-masks-dir", required=True, help="Root dir with pred_masks/sub-verseXXX/")
    parser.add_argument("--gt-root", required=True, help="Root dir with GT masks (sagittal_2d_fixed/)")
    parser.add_argument("--manifest", required=True, help="Path to slice_manifest.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory for results")
    parser.add_argument("--cases", default="", help="Comma-separated volume IDs (empty = all pred volumes)")
    return parser.parse_args()


def load_png_stack(volume_dir, volume_id, manifest_records):
    """Load PNG stack for one volume, return 3D array (X, Y, Z) uint16."""
    slices_for_volume = [r for r in manifest_records if r["case_id"] == volume_id]
    if not slices_for_volume:
        raise ValueError(f"No manifest records for {volume_id}")

    slices_for_volume.sort(key=lambda r: r["slice_idx"])

    stack = []
    for record in slices_for_volume:
        slice_idx = record["slice_idx"]
        png_path = volume_dir / f"{volume_id}_x{slice_idx:04d}_pred_mask.png"
        if not png_path.exists():
            # Missing slice → insert empty
            h, w = record.get("canonical_shape_y", 512), record.get("canonical_shape_z", 512)
            stack.append(np.zeros((h, w), dtype=np.uint16))
        else:
            img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Failed to read {png_path}")
            # PNG is (H, W) = (Z, Y) in sagittal → transpose to (Y, Z)
            stack.append(img.T)

    # stack is list of (Y, Z), concat along X → (X, Y, Z)
    volume_3d = np.stack(stack, axis=0)
    return volume_3d.astype(np.uint16)


def load_gt_stack(gt_root, volume_id, manifest_records):
    """Load GT PNG stack, same logic."""
    slices_for_volume = [r for r in manifest_records if r["case_id"] == volume_id]
    slices_for_volume.sort(key=lambda r: r["slice_idx"])

    stack = []
    for record in slices_for_volume:
        mask_path = pathlib.Path(record["mask_path"])
        if not mask_path.is_absolute():
            mask_path = gt_root / mask_path
        img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"GT mask not found: {mask_path}")
        stack.append(img.T)

    volume_3d = np.stack(stack, axis=0)
    return volume_3d.astype(np.uint16)


def load_manifest(manifest_path):
    """Parse CSV manifest."""
    import csv
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["slice_idx"] = int(row["slice_idx"])
            records.append(row)
    return records


def main():
    args = parse_args()

    pred_root = pathlib.Path(args.pred_masks_dir)
    gt_root = pathlib.Path(args.gt_root)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = load_manifest(args.manifest)

    if args.cases:
        volume_ids = args.cases.split(",")
    else:
        # Auto-discover from pred_masks/
        volume_ids = sorted([d.name for d in pred_root.iterdir() if d.is_dir()])

    print(f"[*] Will evaluate {len(volume_ids)} volumes: {volume_ids}")

    results_per_volume = {}

    for volume_id in volume_ids:
        print(f"\n[case] {volume_id}")

        pred_volume_dir = pred_root / volume_id
        if not pred_volume_dir.exists():
            print(f"  [skip] pred dir not found: {pred_volume_dir}")
            continue

        pred_3d = load_png_stack(pred_volume_dir, volume_id, manifest_records)
        gt_3d = load_gt_stack(gt_root, volume_id, manifest_records)

        print(f"  [shape] pred={pred_3d.shape} gt={gt_3d.shape}")

        # Evaluate with VerSe-2021 protocol
        result = verse_metrics.evaluate_volume_pair(
            pred_volume=pred_3d,
            gt_volume=gt_3d,
            spacing=(1.0, 1.0, 1.0),  # PNG stacks are already 1mm isotropic
        )

        results_per_volume[volume_id] = result

        print(f"  [dice_all_gt_mean] {result['dice_all_gt_mean']:.4f}")
        print(f"  [id_rate] {result['id_rate']:.4f}")
        print(f"  [hd95_directed_max_mm_mean] {result.get('hd95_directed_max_mm_mean', 'N/A')}")
        print(f"  [nsd@1mm_mean] {result.get('nsd@1mm_mean', 'N/A')}")

    # Aggregate across volumes
    all_dice = [r["dice_all_gt_mean"] for r in results_per_volume.values() if r["dice_all_gt_mean"] is not None]
    all_id_rate = [r["id_rate"] for r in results_per_volume.values() if r["id_rate"] is not None]

    summary = {
        "mean_dice_all_gt": float(np.mean(all_dice)) if all_dice else None,
        "mean_id_rate": float(np.mean(all_id_rate)) if all_id_rate else None,
        "num_volumes": len(volume_ids),
        "per_volume": results_per_volume,
    }

    out_json = out_dir / "verse_eval_results.json"
    out_json.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    print(f"\n[out] {out_json}")
    print(f"\n[summary]")
    print(f"  mean_dice_all_gt: {summary['mean_dice_all_gt']:.4f}")
    print(f"  mean_id_rate: {summary['mean_id_rate']:.4f}")
    print(f"  num_volumes: {summary['num_volumes']}")


if __name__ == "__main__":
    main()
