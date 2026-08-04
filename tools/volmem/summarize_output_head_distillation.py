#!/usr/bin/env python3
"""Summarize the controlled H0/H1/H2 output-head experiment."""

import argparse
import json
import pathlib


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        default="data/outputs/volmem/output_head_h0_h1_h2_20260803",
    )
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def eval_metrics(path):
    if not path.exists():
        return None
    data = read_json(path)
    return {
        "volume_mean_dice": data["volume_mean_dice"],
        "foreground_slice_mean_dice": data["foreground_slice_mean_dice"],
        "all_slice_mean_dice": data["all_slice_mean_dice"],
        "evaluation_seconds": data["evaluation_seconds"],
        "slices_per_second": data["slices_per_second"],
        "peak_cuda_memory_gb": data["peak_cuda_memory_gb"],
        "moe_diagnostics": data.get("moe_diagnostics", {}),
        "source": str(path),
    }


def main():
    args = parse_args()
    root = pathlib.Path(args.experiment_root)
    eval_root = root / "distilled_eval"
    variants = {
        "H0_legacy_E8_top2": {
            "cost": root / "cost/h0.json",
            "batch8": eval_root / "h0_rerun_parallel_off_batch8/summary.json",
            "batch1": eval_root / "h0_rerun_parallel_off_batch1/summary.json",
        },
        "H1_dense_residual": {
            "cost": root / "cost/h1.json",
            "batch8": eval_root / "h1_parallel_off_batch8/summary.json",
            "batch1": eval_root / "h1_parallel_off_batch1/summary.json",
        },
        "H2_E4_top1_pilot": {
            "cost": root / "cost/h2_e4.json",
            "batch8": eval_root / "h2_parallel_off_batch8/summary.json",
            "batch1": eval_root / "h2_parallel_off_batch1/summary.json",
        },
        "H2_E2_top1_anti_collapse": {
            "cost": root / "cost/h2_e2.json",
            "batch8": eval_root / "h2e2_parallel_off_batch8/summary.json",
            "batch1": eval_root / "h2e2_parallel_off_batch1/summary.json",
        },
    }
    report = {
        "protocol": {
            "split": "val",
            "volumes": ["sub-verse010", "sub-verse011", "sub-verse013"],
            "slices": 333,
            "box_mode": "gt",
            "memory_mode": "parallel-off",
            "seed": 20260731,
        },
        "variants": {},
    }
    for name, paths in variants.items():
        cost = read_json(paths["cost"])
        report["variants"][name] = {
            "total_params": cost["total_params"],
            "total_params_million": cost["total_params_million"],
            "batch8": eval_metrics(paths["batch8"]),
            "batch1": eval_metrics(paths["batch1"]),
        }

    h0 = report["variants"]["H0_legacy_E8_top2"]
    h1 = report["variants"]["H1_dense_residual"]
    h1_b8 = h1["batch8"]
    h0_b8 = h0["batch8"]
    report["H1_vs_H0_batch8"] = {
        "dice_delta": h1_b8["volume_mean_dice"] - h0_b8["volume_mean_dice"],
        "parameter_delta": h1["total_params"] - h0["total_params"],
        "parameter_fraction": h1["total_params"] / h0["total_params"] - 1.0,
        "seconds_delta": h1_b8["evaluation_seconds"] - h0_b8["evaluation_seconds"],
        "seconds_fraction": h1_b8["evaluation_seconds"] / h0_b8["evaluation_seconds"] - 1.0,
        "throughput_fraction": h1_b8["slices_per_second"] / h0_b8["slices_per_second"] - 1.0,
    }
    report["decision"] = {
        "recommended_output_head": "H1_dense_residual",
        "reason": (
            "H1 preserves H0 Dice while using fewer parameters and the least "
            "end-to-end time. E4 collapses; E2 fixes collapse but is still "
            "strictly dominated by H1."
        ),
    }
    path = root / "comparison.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
