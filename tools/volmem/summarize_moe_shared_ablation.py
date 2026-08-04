#!/usr/bin/env python3
import argparse
import json
import pathlib
import statistics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--control-train", required=True)
    parser.add_argument("--shared-train", required=True)
    return parser.parse_args()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def read_train(path):
    rows = [
        json.loads(line)
        for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tail = rows[-100:]
    return {
        "steps": len(rows),
        "final_step": int(rows[-1]["step"]),
        "last100_mean_time_ms": statistics.mean(float(row["time_ms"]) for row in tail),
        "last100_mean_loss": statistics.mean(float(row["loss"]) for row in tail),
        "peak_memory_gb": max(float(row["peak_memory_gb"]) for row in rows),
        "final_moe_routing": rows[-1].get("moe_routing", {}),
    }


def mean(values):
    return statistics.mean(float(value) for value in values)


args = parse_args()
root = pathlib.Path(args.result_root)
params = {
    "control": read_json(root / "control_params.json"),
    "shared": read_json(root / "shared_params.json"),
}
evaluations = {}
for round_name in ("round1", "round2"):
    for batch in (1, 8):
        for variant in ("control", "shared"):
            key = f"{round_name}_batch{batch}_{variant}"
            evaluations[key] = read_json(root / key / "summary.json")

quality_control = evaluations["round1_batch1_control"]
quality_shared = evaluations["round1_batch1_shared"]
quality_fields = (
    "volume_mean_dice",
    "volume_mean_iou",
    "foreground_slice_mean_dice",
    "all_slice_mean_dice",
    "class_mean_dice",
)
quality = {
    field: {
        "control": float(quality_control[field]),
        "shared": float(quality_shared[field]),
        "delta": float(quality_shared[field]) - float(quality_control[field]),
    }
    for field in quality_fields
}

speed = {}
for batch in (1, 8):
    control_values = [
        evaluations[f"round{round_id}_batch{batch}_control"]["slices_per_second"]
        for round_id in (1, 2)
    ]
    shared_values = [
        evaluations[f"round{round_id}_batch{batch}_shared"]["slices_per_second"]
        for round_id in (1, 2)
    ]
    control_mean = mean(control_values)
    shared_mean = mean(shared_values)
    speed[f"batch{batch}"] = {
        "control_runs": control_values,
        "shared_runs": shared_values,
        "control_mean_slices_per_second": control_mean,
        "shared_mean_slices_per_second": shared_mean,
        "shared_speed_delta_fraction": shared_mean / control_mean - 1.0,
    }

parameter_delta = (
    float(params["shared"]["total_params"])
    / float(params["control"]["total_params"])
    - 1.0
)
active_parameter_delta = (
    float(params["shared"]["per_route_conditional_params"])
    / float(params["control"]["per_route_conditional_params"])
    - 1.0
)

payload = {
    "experiment": "half-width shared plus half-width E4 Top-1 versus full-width routed-only E4 Top-1",
    "training_seed": 20260802,
    "evaluation_seed": 20260731,
    "quality": quality,
    "speed": speed,
    "parameters": {
        "control_total": int(params["control"]["total_params"]),
        "shared_total": int(params["shared"]["total_params"]),
        "shared_total_delta_fraction": parameter_delta,
        "control_active": int(params["control"]["per_route_conditional_params"]),
        "shared_active": int(params["shared"]["per_route_conditional_params"]),
        "shared_active_delta_fraction": active_parameter_delta,
    },
    "training": {
        "control": read_train(args.control_train),
        "shared": read_train(args.shared_train),
    },
}
(root / "comparison.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
