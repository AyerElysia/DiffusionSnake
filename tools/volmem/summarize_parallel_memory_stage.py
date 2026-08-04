#!/usr/bin/env python3
"""Aggregate paired MemFlowDiT Memory evaluations into JSON and Markdown."""

import argparse
import json
import pathlib


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Directory containing one result subdirectory per mode")
    parser.add_argument("--baseline", default="off")
    return parser.parse_args()


def main():
    args = parse_args()
    root = pathlib.Path(args.root)
    rows = []
    for summary_path in sorted(root.glob("*/summary.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({
            "name": summary_path.parent.name,
            "memory_mode": data.get("memory_mode"),
            "checkpoint_step": data.get("checkpoint_step"),
            "num_volumes": data.get("num_volumes"),
            "volume_mean_dice": data.get("volume_mean_dice"),
            "volume_mean_iou": data.get("volume_mean_iou"),
            "foreground_slice_mean_dice": data.get("foreground_slice_mean_dice"),
            "mean_memory_read_delta": data.get("mean_memory_read_delta"),
            "evaluation_seconds": data.get("evaluation_seconds"),
            "slices_per_second": data.get("slices_per_second"),
            "peak_cuda_memory_gb": data.get("peak_cuda_memory_gb"),
            "memory_capacity": data.get("memory_capacity"),
            "memory_pool_size": data.get("memory_pool_size"),
            "memory_selection_policy": data.get("memory_selection_policy"),
            "summary_path": str(summary_path),
        })
    if not rows:
        raise RuntimeError("no summary.json files found under {}".format(root))

    baseline = next((row for row in rows if row["name"] == args.baseline), None)
    baseline_dice = baseline.get("volume_mean_dice") if baseline else None
    for row in rows:
        value = row.get("volume_mean_dice")
        row["dice_delta_vs_baseline"] = (
            float(value) - float(baseline_dice)
            if value is not None and baseline_dice is not None else None
        )

    payload = {
        "root": str(root),
        "baseline": args.baseline,
        "complete_runs": len(rows),
        "rows": rows,
    }
    (root / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    header = (
        "| Run | Mode | Volumes | Volume Dice | Delta vs off | Read delta | "
        "slice/s | Peak GB |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for row in rows:
        def fmt(value, digits=6):
            return "-" if value is None else ("{:.%df}" % digits).format(float(value))

        lines.append(
            "| {name} | {mode} | {volumes} | {dice} | {delta} | {read} | "
            "{speed} | {peak} |\n".format(
                name=row["name"],
                mode=row["memory_mode"],
                volumes=row["num_volumes"],
                dice=fmt(row["volume_mean_dice"]),
                delta=fmt(row["dice_delta_vs_baseline"]),
                read=fmt(row["mean_memory_read_delta"]),
                speed=fmt(row["slices_per_second"], 3),
                peak=fmt(row["peak_cuda_memory_gb"], 3),
            )
        )
    (root / "comparison.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
