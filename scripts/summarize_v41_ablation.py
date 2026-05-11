#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    ("control", "btcv_ablate_v41_00_v34_detail_control"),
    ("full", "btcv_ablate_v41_01_full"),
    ("no_delta", "btcv_ablate_v41_02_no_delta"),
    ("no_curv", "btcv_ablate_v41_03_no_curv"),
    ("no_small", "btcv_ablate_v41_04_no_small_disp"),
    ("delta_only", "btcv_ablate_v41_05_delta_only"),
    ("curv_only", "btcv_ablate_v41_06_curv_only"),
    ("small_only", "btcv_ablate_v41_07_small_only"),
]


def read_last_jsonl(path: Path):
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def read_latest_eval(run_stem: str):
    eval_dir = ROOT / "data" / "outputs" / "v4_1_mechanism_ablation" / "eval_final" / run_stem
    candidates = sorted(eval_dir.glob("v3_7_full_test_iou_*.json"))
    if not candidates:
        return None
    with candidates[-1].open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("| run | step | epoch | loss | IoU | mBoundF | eval samples |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for label, stem in RUNS:
        log = read_last_jsonl(ROOT / "data" / "outputs" / stem / "logs.jsonl")
        ev = read_latest_eval(stem)
        step = log.get("step", "-") if log else "-"
        epoch = log.get("epoch", "-") if log else "-"
        loss = log.get("loss", None) if log else None
        loss_s = f"{loss:.6f}" if isinstance(loss, (int, float)) else "-"
        if ev:
            iou = ev.get("mean_iou_sample_avg", 0.0)
            mbf = ev.get("mean_mboundf_sample_avg", 0.0)
            n = ev.get("evaluated_samples", 0)
            print(f"| {label} | {step} | {epoch} | {loss_s} | {iou:.6f} | {mbf:.6f} | {n} |")
        else:
            print(f"| {label} | {step} | {epoch} | {loss_s} | - | - | - |")


if __name__ == "__main__":
    main()
