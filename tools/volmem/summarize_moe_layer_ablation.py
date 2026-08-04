#!/usr/bin/env python3
import argparse
import json
import pathlib
import statistics


parser = argparse.ArgumentParser()
parser.add_argument("--result-root", required=True)
parser.add_argument("--odd-train", required=True)
parser.add_argument("--all-train", required=True)
args = parser.parse_args()
root = pathlib.Path(args.result_root)


def load(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def train_summary(path):
    rows = [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]
    tail = rows[-100:]
    return {
        "steps": len(rows),
        "final_step": int(rows[-1]["step"]),
        "last100_mean_time_ms": statistics.mean(float(row["time_ms"]) for row in tail),
        "last100_mean_loss": statistics.mean(float(row["loss"]) for row in tail),
        "peak_memory_gb": max(float(row["peak_memory_gb"]) for row in rows),
        "final_moe_routing": rows[-1].get("moe_routing", {}),
    }


evaluations = {}
for round_id in (1, 2):
    for batch in (1, 8):
        for variant in ("odd3", "all6"):
            key = f"round{round_id}_batch{batch}_{variant}"
            evaluations[key] = load(root / key / "summary.json")

odd_quality = evaluations["round1_batch1_odd3"]
all_quality = evaluations["round1_batch1_all6"]
quality_fields = (
    "volume_mean_dice",
    "volume_mean_iou",
    "foreground_slice_mean_dice",
    "all_slice_mean_dice",
    "class_mean_dice",
)
quality = {
    field: {
        "odd3": float(odd_quality[field]),
        "all6": float(all_quality[field]),
        "delta": float(all_quality[field]) - float(odd_quality[field]),
    }
    for field in quality_fields
}

speed = {}
for batch in (1, 8):
    odd_runs = [evaluations[f"round{i}_batch{batch}_odd3"]["slices_per_second"] for i in (1, 2)]
    all_runs = [evaluations[f"round{i}_batch{batch}_all6"]["slices_per_second"] for i in (1, 2)]
    odd_mean = statistics.mean(odd_runs)
    all_mean = statistics.mean(all_runs)
    speed[f"batch{batch}"] = {
        "odd3_runs": odd_runs,
        "all6_runs": all_runs,
        "odd3_mean_slices_per_second": odd_mean,
        "all6_mean_slices_per_second": all_mean,
        "all6_speed_delta_fraction": all_mean / odd_mean - 1.0,
    }

odd_params = load(root / "odd3_params.json")
all_params = load(root / "all6_params.json")
payload = {
    "experiment": "odd three E4 Top-1 DiT-MoE layers versus all six",
    "training_seed": 20260802,
    "evaluation_seed": 20260731,
    "quality": quality,
    "speed": speed,
    "parameters": {
        "odd3_total": int(odd_params["total_params"]),
        "all6_total": int(all_params["total_params"]),
        "all6_total_delta_fraction": float(all_params["total_params"]) / float(odd_params["total_params"]) - 1.0,
        "odd3_active": int(odd_params["per_route_conditional_params"]),
        "all6_active": int(all_params["per_route_conditional_params"]),
        "all6_active_delta_fraction": float(all_params["per_route_conditional_params"]) / float(odd_params["per_route_conditional_params"]) - 1.0,
    },
    "training": {
        "odd3": train_summary(args.odd_train),
        "all6": train_summary(args.all_train),
    },
}
(root / "comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
