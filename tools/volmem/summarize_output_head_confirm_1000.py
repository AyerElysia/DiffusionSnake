#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    return parser.parse_args()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean_metric(root, label, batch, key):
    rows = [
        load(root / ("round1_{}_parallel-off_batch{}/summary.json".format(label, batch))),
        load(root / ("round2_{}_parallel-off_batch{}/summary.json".format(label, batch))),
    ]
    return statistics.mean(float(row[key]) for row in rows)


def main():
    args = parse_args()
    root = Path(args.result_root)
    rows = {}
    for label in ("d1", "l0"):
        autoregressive = load(root / ("{}_autoregressive_batch1/summary.json".format(label)))
        rows[label] = {
            "memory_off_volume_dice": mean_metric(root, label, 8, "volume_mean_dice"),
            "foreground_slice_dice": mean_metric(root, label, 8, "foreground_slice_mean_dice"),
            "batch1_slices_per_second": mean_metric(root, label, 1, "slices_per_second"),
            "batch8_slices_per_second": mean_metric(root, label, 8, "slices_per_second"),
            "autoregressive_volume_dice": float(autoregressive["volume_mean_dice"]),
        }
    rows["delta_d1_minus_l0"] = {
        key: rows["d1"][key] - rows["l0"][key]
        for key in rows["d1"]
    }
    output = {"checkpoint_step": 1000, "rows": rows}
    (root / "comparison.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
