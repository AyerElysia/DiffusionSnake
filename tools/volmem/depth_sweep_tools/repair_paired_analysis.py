"""Recompute the P0(L=6) vs P1(L=8) paired comparison from raw train.jsonl.

Reads nothing but the on-disk logs. Prints every number with enough precision
that a successor can re-derive it. No prior result is assumed or reused.
"""
import json
import math
import os

BASE = "data/outputs/depth_sweep"
ARMS = {
    "P0_L6": "depth_sweep_p0_l6",
    "P1_L8": "depth_sweep_p1_l8",
}


def load(run_dir):
    path = os.path.join(BASE, run_dir, "train.jsonl")
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    data = {name: load(d) for name, d in ARMS.items()}
    for name, rows in data.items():
        print("{}: {} rows, steps {}..{}".format(
            name, len(rows), rows[0]["step"], rows[-1]["step"]))

    a = data["P0_L6"]
    b = data["P1_L8"]
    n = min(len(a), len(b))
    print("paired over n = {} steps".format(n))

    # --- pairing validity: identical batches at every step? ---
    batch_mismatch = []
    for i in range(n):
        if a[i]["step"] != b[i]["step"]:
            batch_mismatch.append((i, "step"))
        elif a[i].get("volume_ids") != b[i].get("volume_ids"):
            batch_mismatch.append((a[i]["step"], "volume_ids"))
    print("batch mismatches: {}".format(len(batch_mismatch)))
    if batch_mismatch:
        print("  first 10: {}".format(batch_mismatch[:10]))

    # --- step-1 identity check ---
    print("step1 P0 loss = {!r}".format(a[0]["loss"]))
    print("step1 P1 loss = {!r}".format(b[0]["loss"]))
    print("step1 bit-identical: {}".format(a[0]["loss"] == b[0]["loss"]))

    # --- paired deltas (P1 - P0); negative == deeper is better ---
    d = [b[i]["loss"] - a[i]["loss"] for i in range(n)]
    absd = [abs(x) for x in d]
    md = mean(d)
    sd = stdev(d)
    sem = sd / math.sqrt(n)
    lo, hi = md - 1.96 * sem, md + 1.96 * sem
    wins = sum(1 for x in d if x < 0)

    print("")
    print("mean delta (P1-P0)      = {:.6e}".format(md))
    print("stdev                   = {:.6e}".format(sd))
    print("sem                     = {:.6e}".format(sem))
    print("95% CI                  = [{:.6e}, {:.6e}]".format(lo, hi))
    print("CI excludes zero        = {}".format(lo > 0 or hi < 0))
    print("mean |delta|            = {:.6e}".format(mean(absd)))
    print("P1 wins (lower loss)    = {}/{} = {:.2f}%".format(
        wins, n, 100.0 * wins / n))

    # --- windowed means, to test transience ---
    print("")
    print("{:>12} {:>6} {:>14} {:>14} {:>10} {:>9}".format(
        "window", "n", "mean_delta", "mean_P0_loss", "rel_pct", "win_pct"))
    windows = [(1, 249), (250, 499), (500, 999), (1000, 1249),
               (1250, 1499), (1500, 1749), (1750, 2000)]
    for s0, s1 in windows:
        idx = [i for i in range(n) if s0 <= a[i]["step"] <= s1]
        if not idx:
            continue
        dw = [d[i] for i in idx]
        p0w = [a[i]["loss"] for i in idx]
        w = sum(1 for i in idx if d[i] < 0)
        rel = 100.0 * mean(dw) / mean(p0w) if mean(p0w) else float("nan")
        print("{:>12} {:>6} {:>14.4e} {:>14.6f} {:>10.4f} {:>9.1f}".format(
            "{}-{}".format(s0, s1), len(idx), mean(dw), mean(p0w), rel,
            100.0 * w / len(idx)))

    # --- endpoint losses and memory ---
    print("")
    for name, rows in (("P0_L6", a), ("P1_L8", b)):
        tail = rows[-250:]
        print("{}: last250 mean loss = {:.8f}   final step loss = {:.8f}   "
              "max peak_mem = {:.2f} GB".format(
                  name, mean([r["loss"] for r in tail]), rows[-1]["loss"],
                  max(r.get("peak_memory_gb", 0.0) for r in rows)))

    # --- last-250 paired summary (the decision window) ---
    idx = list(range(n - 250, n))
    dw = [d[i] for i in idx]
    p0w = [a[i]["loss"] for i in idx]
    w = sum(1 for i in idx if d[i] < 0)
    sdw = stdev(dw)
    semw = sdw / math.sqrt(len(dw))
    print("")
    print("last250 mean delta = {:.6e}  95% CI [{:.6e}, {:.6e}]  "
          "rel = {:.4f}%  win = {:.1f}%".format(
              mean(dw), mean(dw) - 1.96 * semw, mean(dw) + 1.96 * semw,
              100.0 * mean(dw) / mean(p0w), 100.0 * w / len(dw)))


if __name__ == "__main__":
    main()
