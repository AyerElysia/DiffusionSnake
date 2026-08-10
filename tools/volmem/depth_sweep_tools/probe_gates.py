"""Measure whether freshly-added DiT layers actually lift off their zero-init gates.

DiTBlockV3.adaLN_modulation[-1] is zero-init (weight and bias), and it produces
9 chunks: shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca,
shift_ff, scale_ff, gate_ff.  Chunks 2/5/8 are the residual gates -- while they
are exactly zero the block is an identity map, so a new layer contributes
nothing no matter how many parameters it holds.

This reads each arm's checkpoint and reports, per DiT layer, the L2 norm of the
full modulation weight plus the three gate chunks, expressed as a fraction of
the *pretrained* layers' mean norm in the same checkpoint.  That gives an
absolute, loss-independent answer to "did the new depth get used?".
"""
import os
import re
import sys

import torch

DIM = 256
GATE_CHUNKS = {"gate_sa": 2, "gate_ca": 5, "gate_ff": 8}
BASE_DEPTH = 6  # layers 0..5 come from the pretrained checkpoint

KEY_RE = re.compile(r"\.dit_layers\.(\d+)\.adaLN_modulation\.1\.(weight|bias)$")


def load_state(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "net"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key], obj.get("step")
    return obj, None


def chunk_norm(tensor, index):
    lo, hi = index * DIM, (index + 1) * DIM
    return float(tensor[lo:hi].reshape(-1).norm())


def probe(path, label):
    if not os.path.exists(path):
        print("{:8s} MISSING {}".format(label, path))
        return
    state, step = load_state(path)
    layers = {}
    for key, value in state.items():
        match = KEY_RE.search(key)
        if match is None:
            continue
        idx = int(match.group(1))
        layers.setdefault(idx, {})[match.group(2)] = value.float()

    if not layers:
        print("{:8s} no adaLN_modulation keys found".format(label))
        return

    print("{:8s} step={} layers={}".format(label, step, len(layers)))
    rows = []
    for idx in sorted(layers):
        weight = layers[idx].get("weight")
        bias = layers[idx].get("bias")
        row = {
            "layer": idx,
            "new": idx >= BASE_DEPTH,
            "w_norm": float(weight.reshape(-1).norm()) if weight is not None else float("nan"),
        }
        for name, ci in GATE_CHUNKS.items():
            row[name] = chunk_norm(bias, ci) if bias is not None else float("nan")
            row[name + "_w"] = (
                float(weight[ci * DIM:(ci + 1) * DIM].reshape(-1).norm())
                if weight is not None
                else float("nan")
            )
        rows.append(row)

    mature = [r for r in rows if not r["new"]]
    ref_w = sum(r["w_norm"] for r in mature) / max(len(mature), 1)
    ref_gates = {
        name: sum(r[name + "_w"] for r in mature) / max(len(mature), 1)
        for name in GATE_CHUNKS
    }

    print(
        "  {:>5s} {:>4s} {:>10s} {:>7s} {:>10s} {:>10s} {:>10s}".format(
            "layer", "new", "|W|", "%mature", "gate_sa_w", "gate_ca_w", "gate_ff_w"
        )
    )
    for r in rows:
        pct = 100.0 * r["w_norm"] / ref_w if ref_w > 0 else float("nan")
        print(
            "  {:5d} {:>4s} {:10.5f} {:6.1f}% {:10.6f} {:10.6f} {:10.6f}".format(
                r["layer"],
                "YES" if r["new"] else "-",
                r["w_norm"],
                pct,
                r["gate_sa_w"],
                r["gate_ca_w"],
                r["gate_ff_w"],
            )
        )

    new_rows = [r for r in rows if r["new"]]
    if new_rows:
        print("  mature mean |W|={:.5f}  gate_w ref: {}".format(
            ref_w,
            " ".join("{}={:.6f}".format(k, v) for k, v in ref_gates.items()),
        ))
        for name in GATE_CHUNKS:
            new_mean = sum(r[name + "_w"] for r in new_rows) / len(new_rows)
            ref = ref_gates[name]
            frac = 100.0 * new_mean / ref if ref > 0 else float("nan")
            print("  NEW-LAYER {:8s} mean={:.6f}  = {:5.1f}% of mature".format(
                name, new_mean, frac))
        alive = sum(
            1 for r in new_rows
            if max(r[n + "_w"] for n in GATE_CHUNKS) > 0.0
        )
        print("  new layers with any non-zero gate: {}/{}".format(alive, len(new_rows)))
    print("")


if __name__ == "__main__":
    for path_label in sys.argv[1:]:
        if ":" in path_label:
            label, path = path_label.split(":", 1)
        else:
            label, path = os.path.basename(os.path.dirname(path_label)), path_label
        probe(path, label)
