# Paired per-step loss analysis across depth-sweep arms.
# Pure stdlib: remote python3.8 has no scipy guarantee.
import math
import os
import random
import re
import sys

ARMS = ["depth_sweep_p0_l6", "depth_sweep_p1_l8",
        "depth_sweep_p2_l10", "depth_sweep_p3_l12"]
ROOT = "data/outputs/depth_sweep"
STEP_RE = re.compile(r"\[step (\d+)\] loss=([0-9.eE+-]+)")


def read_losses(arm):
    path = os.path.join(ROOT, arm, "train.log")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = STEP_RE.search(line)
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


def mean(xs):
    return sum(xs) / float(len(xs)) if xs else float("nan")


def wilcoxon(diffs):
    """Signed-rank, normal approx with tie + continuity correction."""
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n < 10:
        return float("nan")
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    mu = n * (n + 1) / 4.0
    tie = 0.0
    i = 0
    srt = sorted(abs(d) for d in nz)
    while i < n:
        j = i
        while j + 1 < n and srt[j + 1] == srt[i]:
            j += 1
        t = j - i + 1
        if t > 1:
            tie += t ** 3 - t
        i = j + 1
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0
    if var <= 0:
        return float("nan")
    z = (abs(w_plus - mu) - 0.5) / math.sqrt(var)
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def sign_test(diffs):
    wins = sum(1 for d in diffs if d < 0)
    n = sum(1 for d in diffs if d != 0.0)
    if n == 0:
        return 0, 0, float("nan")
    z = (abs(wins - n / 2.0) - 0.5) / math.sqrt(n / 4.0)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return wins, n, p


def boot_ci(diffs, iters=4000, seed=12345):
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"))
    ms = []
    for _ in range(iters):
        ms.append(mean([diffs[rng.randrange(n)] for _ in range(n)]))
    ms.sort()
    return (ms[int(0.025 * iters)], ms[int(0.975 * iters)])


def main():
    data = {a: read_losses(a) for a in ARMS}
    for a in ARMS:
        d = data[a]
        print("{:22s} steps={:5d} max_step={}".format(
            a, len(d), max(d) if d else 0))
    base = data[ARMS[0]]
    if not base:
        print("no control data yet")
        return
    print("\n=== identity-at-init check (first 2 common steps) ===")
    for s in (1, 2, 3):
        row = ["{}={:.6f}".format(a.split("_")[-1], data[a][s])
               for a in ARMS if s in data[a]]
        print("  step {}: {}".format(s, "  ".join(row)))

    for a in ARMS[1:]:
        cur = data[a]
        common = sorted(set(base) & set(cur))
        if len(common) < 50:
            print("\n--- {} vs L6: only {} common steps, skip".format(
                a, len(common)))
            continue
        diffs = [cur[s] - base[s] for s in common]
        mb, mc = mean([base[s] for s in common]), mean([cur[s] for s in common])
        rel = 100.0 * (mc - mb) / mb if mb else float("nan")
        lo, hi = boot_ci(diffs)
        wins, n, ps = sign_test(diffs)
        print("\n--- {} vs L6  (n={} paired steps) ---".format(a, len(common)))
        print("  mean loss   L6={:.6f}  {}={:.6f}   rel={:+.3f}%".format(
            mb, a.split("_")[-1], mc, rel))
        print("  mean delta  {:+.3e}   95%CI [{:+.3e}, {:+.3e}]".format(
            mean(diffs), lo, hi))
        print("  mean |delta| {:.3e}".format(mean([abs(d) for d in diffs])))
        print("  win rate    {}/{} = {:.1f}%   sign p={:.4g}".format(
            wins, n, 100.0 * wins / n if n else float("nan"), ps))
        print("  wilcoxon    p={:.4g}".format(wilcoxon(diffs)))
        # windowed trend: is the gap widening as new layers train?
        k = max(1, len(common) // 4)
        for qi in range(4):
            seg = common[qi * k:(qi + 1) * k] if qi < 3 else common[3 * k:]
            if not seg:
                continue
            sd = [cur[s] - base[s] for s in seg]
            print("    Q{} steps {:>5}-{:<5} mean_delta={:+.3e}".format(
                qi + 1, seg[0], seg[-1], mean(sd)))


if __name__ == "__main__":
    main()
