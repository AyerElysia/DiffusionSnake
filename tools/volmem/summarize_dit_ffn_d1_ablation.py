#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


LABELS = ("dense6", "odd3", "all6", "sharedodd3")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def average_runs(root, label, batch):
    runs = [
        load(root / "{}_{}_parallel-off_batch{}/summary.json".format(round_id, label, batch))
        for round_id in ("round1", "round2")
    ]
    scalar_keys = (
        "volume_mean_dice",
        "volume_mean_iou",
        "foreground_slice_mean_dice",
        "class_mean_dice",
        "slices_per_second",
        "peak_cuda_memory_gb",
    )
    return {key: statistics.mean(float(run[key]) for run in runs) for key in scalar_keys}


def route_health(summary):
    modules = summary.get("moe_diagnostics", {})
    return {
        "num_moe_layers": len(modules),
        "dead_experts": sum(float(row.get("dead_experts_lt_1pct", 0.0)) for row in modules.values()),
        "max_hard_cv": max([float(row.get("hard_cv", 0.0)) for row in modules.values()] or [0.0]),
        "layer_hard_load": [row.get("hard_load", []) for _, row in sorted(modules.items())],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    root = Path(args.result_root)
    rows = {}
    for label in LABELS:
        b1 = average_runs(root, label, 1)
        b8 = average_runs(root, label, 8)
        params = load(root / "params/{}.json".format(label))
        diagnostic_summary = load(root / "round1_{}_parallel-off_batch8/summary.json".format(label))
        rows[label] = {
            "total_params_million": float(params["total_params_million"]),
            "per_route_params_million": float(params["per_route_conditional_params_million"]),
            "volume_mean_dice": b8["volume_mean_dice"],
            "foreground_slice_mean_dice": b8["foreground_slice_mean_dice"],
            "class_mean_dice": b8["class_mean_dice"],
            "batch1_slices_per_second": b1["slices_per_second"],
            "batch8_slices_per_second": b8["slices_per_second"],
            "batch8_peak_cuda_memory_gb": b8["peak_cuda_memory_gb"],
            "route_health": route_health(diagnostic_summary),
        }
    for baseline in ("dense6", "odd3"):
        for label in LABELS:
            rows[label]["delta_dice_vs_{}".format(baseline)] = (
                rows[label]["volume_mean_dice"] - rows[baseline]["volume_mean_dice"]
            )
            rows[label]["speed_delta_batch8_vs_{}_pct".format(baseline)] = 100.0 * (
                rows[label]["batch8_slices_per_second"]
                / rows[baseline]["batch8_slices_per_second"] - 1.0
            )
    output = {"checkpoint_step": 1000, "rows": rows}
    (root / "comparison.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
