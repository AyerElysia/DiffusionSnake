"""Probe adaLN gate-chunk norms from real depth-sweep checkpoints.

Reports, per checkpoint step, the mean L2 norm of the three gate chunks
(gate_sa=chunk2, gate_ca=chunk5, gate_ff=chunk8) of
adaLN_modulation[1].weight, for pretrained layers (idx < BASE) vs new
layers (idx >= BASE), and the new/pretrained ratio.

Loss-independent test of whether newly added depth is actually used.
"""
import os
import re
import sys
import glob

import torch

CKPT_DIR = sys.argv[1]
BASE = int(sys.argv[2])   # first new layer index
DIM = int(sys.argv[3])    # residual dim

KEY_RE = re.compile(r"dit_layers\.(\d+)\.adaLN_modulation\.1\.weight$")


def unwrap(obj):
    if not isinstance(obj, dict):
        return None
    for k in ("net", "model", "state_dict"):
        if k in obj and isinstance(obj[k], dict):
            inner = unwrap(obj[k])
            return inner if inner is not None else obj[k]
    # already a flat state dict?
    for k in obj:
        if isinstance(k, str) and "dit_layers" in k:
            return obj
    return None


paths = sorted(glob.glob(os.path.join(CKPT_DIR, "step_*.pt")))
if not paths:
    print("no checkpoints in", CKPT_DIR)
    sys.exit(1)

# discover keys once
first = torch.load(paths[0], map_location="cpu")
sd = unwrap(first)
if sd is None:
    print("could not locate state dict; top keys =", list(first.keys())[:20])
    sys.exit(1)

keys = {}
for k in sd:
    m = KEY_RE.search(k)
    if m:
        keys[int(m.group(1))] = k
if not keys:
    cand = [k for k in sd if "adaLN" in k]
    print("no adaLN_modulation.1.weight matches. adaLN-ish keys sample:")
    for k in cand[:20]:
        print("   ", k, tuple(sd[k].shape))
    sys.exit(1)

layer_ids = sorted(keys)
print("checkpoint dir :", CKPT_DIR)
print("dit layers found:", layer_ids)
print("base (first new):", BASE, " dim:", DIM)
w0 = sd[keys[layer_ids[0]]]
print("adaLN weight shape:", tuple(w0.shape), "=> expect (9*dim, dim) =", (9 * DIM, DIM))
print()

CHUNKS = {"gate_sa": 2, "gate_ca": 5, "gate_ff": 8}

hdr = "{:>7} {:>9} {:>12} {:>12} {:>9}".format(
    "step", "gate", "pretrained", "new", "new/pre%")
print(hdr)
print("-" * len(hdr))

del first, sd

for p in paths:
    step = int(re.search(r"step_(\d+)\.pt", p).group(1))
    obj = torch.load(p, map_location="cpu")
    sd = unwrap(obj)
    if sd is None:
        print(step, "unreadable")
        continue
    for name, ci in CHUNKS.items():
        pre_vals, new_vals = [], []
        for li in layer_ids:
            w = sd[keys[li]]
            chunk = w[ci * DIM:(ci + 1) * DIM, :]
            n = float(chunk.norm())
            (new_vals if li >= BASE else pre_vals).append(n)
        pre_mean = sum(pre_vals) / len(pre_vals) if pre_vals else float("nan")
        new_mean = sum(new_vals) / len(new_vals) if new_vals else float("nan")
        ratio = (new_mean / pre_mean * 100.0) if pre_mean else float("nan")
        print("{:>7} {:>9} {:>12.6f} {:>12.6f} {:>9.2f}".format(
            step, name, pre_mean, new_mean, ratio))
    del obj, sd
    print()
