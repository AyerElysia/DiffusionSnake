#!/usr/bin/env python3
"""Summarize the five-way 2026 output-head screening experiment."""

import argparse
import json
from pathlib import Path

import torch


VARIANTS = {
    "d0_dense": "data/outputs/volmem/verse_memflowdit_output_head_d0_dense_gpu4",
    "d1_dense_residual": "data/outputs/volmem/verse_memflowdit_output_head_d1_dense_residual_gpu5",
    "l0_legacy": "data/outputs/volmem/verse_memflowdit_output_head_l0_legacy_gpu4_queued",
    "m1_modern_k2": "data/outputs/volmem/verse_memflowdit_output_head_m1_modern_k2_gpu6",
    "m1_modern_k1": "data/outputs/volmem/verse_memflowdit_output_head_m1_modern_k1_gpu7",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_last_jsonl(path):
    last = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last or {}


def checkpoint_counts(path):
    checkpoint = torch.load(str(path), map_location="cpu")
    state = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    tensors = {key: value for key, value in state.items() if torch.is_tensor(value)}
    head = {
        key: value
        for key, value in tensors.items()
        if ".final_layer." in key
    }
    experts = {
        key: value
        for key, value in head.items()
        if ".experts." in key
        or ".expert_" in key
        or ".shared_mlp." in key
    }
    return {
        "checkpoint_tensor_parameters": int(sum(v.numel() for v in tensors.values())),
        "output_head_parameters": int(sum(v.numel() for v in head.values())),
        "output_specialist_parameters": int(sum(v.numel() for v in experts.values())),
    }


def compact_eval(summary):
    final_head_route = None
    for module_name, diagnostics in summary.get("moe_diagnostics", {}).items():
        if module_name.endswith("final_layer"):
            final_head_route = diagnostics
            break
    return {
        "volume_mean_dice": summary.get("volume_mean_dice"),
        "volume_mean_iou": summary.get("volume_mean_iou"),
        "foreground_slice_mean_dice": summary.get("foreground_slice_mean_dice"),
        "class_mean_dice": summary.get("class_mean_dice"),
        "slices_per_second": summary.get("slices_per_second"),
        "peak_cuda_memory_gb": summary.get("peak_cuda_memory_gb"),
        "evaluation_seconds": summary.get("evaluation_seconds"),
        "final_head_route": final_head_route,
    }


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    result_root = (project_root / args.result_root).resolve()
    rows = {}
    for label, relative_run_dir in VARIANTS.items():
        run_dir = project_root / relative_run_dir
        checkpoint = run_dir / "checkpoints" / "step_000300.pt"
        row = {
            "parameters": checkpoint_counts(checkpoint),
            "train_last": load_last_jsonl(run_dir / "train.jsonl"),
        }
        for memory_mode, batch_size in (
            ("parallel-off", 1),
            ("parallel-off", 8),
            ("autoregressive", 1),
        ):
            summary_path = (
                result_root
                / "{}_{}_batch{}".format(label, memory_mode, batch_size)
                / "summary.json"
            )
            row["{}_batch{}".format(memory_mode, batch_size)] = compact_eval(
                load_json(summary_path)
            )
        rows[label] = row

    dense_control = rows["d1_dense_residual"]["parallel-off_batch8"]
    dense_dice = float(dense_control["volume_mean_dice"])
    dense_speed = float(dense_control["slices_per_second"])
    for label, row in rows.items():
        batch8 = row["parallel-off_batch8"]
        row["delta_vs_d1"] = {
            "volume_mean_dice": float(batch8["volume_mean_dice"]) - dense_dice,
            "throughput_percent": (
                float(batch8["slices_per_second"]) / dense_speed - 1.0
            ) * 100.0,
        }

    comparison = {
        "experiment": "output_head_moe_2026_screen",
        "checkpoint_step": 300,
        "quality_control": "d1_dense_residual",
        "quality_gate_delta_dice": 0.002,
        "latency_gate_max_slowdown_percent": 10.0,
        "rows": rows,
    }
    comparison_path = result_root / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    header = (
        "label\tparams_M\thead_params_M\tvolume_dice_b8\t"
        "delta_vs_D1\tslice_dice\tslice_per_s_b1\tslice_per_s_b8\t"
        "speed_delta_vs_D1_pct\tautoregressive_dice\toutput_hard_cv\tdead_experts"
    )
    lines = [header]
    for label, row in rows.items():
        params = row["parameters"]
        batch1 = row["parallel-off_batch1"]
        batch8 = row["parallel-off_batch8"]
        autoregressive = row["autoregressive_batch1"]
        route = batch8.get("final_head_route") or {}
        delta = row["delta_vs_d1"]
        lines.append(
            "\t".join([
                label,
                "{:.6f}".format(params["checkpoint_tensor_parameters"] / 1e6),
                "{:.6f}".format(params["output_head_parameters"] / 1e6),
                "{:.6f}".format(batch8["volume_mean_dice"]),
                "{:+.6f}".format(delta["volume_mean_dice"]),
                "{:.6f}".format(batch8["foreground_slice_mean_dice"]),
                "{:.6f}".format(batch1["slices_per_second"]),
                "{:.6f}".format(batch8["slices_per_second"]),
                "{:+.2f}".format(delta["throughput_percent"]),
                "{:.6f}".format(autoregressive["volume_mean_dice"]),
                str(route.get("hard_cv", "-")),
                str(route.get("dead_experts_lt_1pct", "-")),
            ])
        )
    text = "\n".join(lines) + "\n"
    (result_root / "summary.tsv").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
