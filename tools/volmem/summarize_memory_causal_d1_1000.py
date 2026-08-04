#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compact(summary):
    keys = (
        "memory_mode",
        "memory_capacity",
        "memory_selection_policy",
        "memory_evidence_source",
        "volume_mean_dice",
        "foreground_slice_mean_dice",
        "class_mean_dice",
        "mean_memory_read_delta",
        "slices_per_second",
        "peak_cuda_memory_gb",
        "effective_contour_passes",
    )
    return {key: summary.get(key) for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--head-confirm-root", required=True)
    args = parser.parse_args()
    root = Path(args.result_root)
    rows = {}
    for summary_path in sorted(root.glob("*/summary.json")):
        rows[summary_path.parent.name] = compact(load(summary_path))

    head_root = Path(args.head_confirm_root)
    off_runs = [
        load(head_root / "round1_d1_parallel-off_batch8/summary.json"),
        load(head_root / "round2_d1_parallel-off_batch8/summary.json"),
    ]
    off = compact(off_runs[0])
    for key in (
        "volume_mean_dice",
        "foreground_slice_mean_dice",
        "class_mean_dice",
        "slices_per_second",
        "peak_cuda_memory_gb",
    ):
        off[key] = sum(float(run[key]) for run in off_runs) / len(off_runs)
    rows["full_off_reused"] = off
    rows["full_ar_k4_reused"] = compact(
        load(head_root / "d1_autoregressive_batch1/summary.json")
    )

    quick_off = float(rows["quick_off"]["volume_mean_dice"])
    full_off = float(rows["full_off_reused"]["volume_mean_dice"])
    for label, row in rows.items():
        baseline = full_off if label.startswith("full_") else quick_off
        row["delta_volume_dice_vs_matching_off"] = (
            float(row["volume_mean_dice"]) - baseline
        )

    causal = rows["full_frozen_causal_k4"]
    shuffled = rows["full_frozen_shuffled_k4"]
    content_margin = (
        float(causal["volume_mean_dice"])
        - float(shuffled["volume_mean_dice"])
    )
    verdict = {
        "normal_history_beats_shuffled_margin": content_margin,
        "content_dependent": content_margin >= 0.001,
        "normal_history_beats_off": (
            float(causal["volume_mean_dice"]) - full_off >= 0.001
        ),
        "autoregressive_beats_off": (
            float(rows["full_ar_k4_reused"]["volume_mean_dice"]) - full_off >= 0.001
        ),
        "all_history_beats_bounded_k4": (
            float(rows["full_frozen_all_k256"]["volume_mean_dice"])
            - float(causal["volume_mean_dice"]) >= 0.001
        ),
    }
    output = {"checkpoint_step": 1000, "rows": rows, "verdict": verdict}
    (root / "comparison.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
