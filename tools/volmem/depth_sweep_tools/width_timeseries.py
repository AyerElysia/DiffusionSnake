"""Track the 'extra width' slice of the 5 widened tensors across training checkpoints.

If the extra slice is exactly 0 at step 32 AND still exactly 0 at step 2000,
the width arms were mathematically inert: zero-init -> zero grad -> zero forever.
"""
import io
import os
import pickle
import sys

import torch

ROOT = ("/home/medteam/Zhrch/DiffusionSnake-12-30-pure2d-scaleup-outputs-20260808/"
        "fair_rng_six_arm_matrix_v1_r1_20260808/training")

STEPS = ["step_32", "step_500", "step_1000", "step_1500", "step_2000"]

# tensors that differ in shape vs A0, and the axis along which they were widened
WIDENED = [
    ("net.locate_feat_replacer.proj.7.weight", 0),
    ("net.locate_feat_replacer.proj.7.bias", 0),
    ("net.gcn.denoiser.global_compressor.input_proj.weight", 1),
    ("net.gcn.denoiser.local_proj.0.weight", 1),
    ("net.gcn.denoiser.point_embed.feat_embed.0.weight", 1),
]

BASE = 256  # A0 width


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Stub()

    def __setstate__(self, s):
        pass


_cache = {}


class _U(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            key = (module, name)
            if key not in _cache:
                _cache[key] = type(name, (_Stub,), {})
            return _cache[key]


class _FakePickle:
    Unpickler = _U
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
    Pickler = pickle.Pickler
    PickleError = pickle.PickleError
    UnpicklingError = pickle.UnpicklingError


def load_sd(path):
    obj = torch.load(path, map_location="cpu", pickle_module=_FakePickle)
    for k in ("state_dict", "net", "model"):
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
            return obj[k]
    return obj


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "B1"
    print("=== arm {} : |extra slice| over training ===".format(arm))
    hdr = "{:<52}".format("tensor")
    for s in STEPS:
        hdr += "{:>14}".format(s.replace("step_", "s"))
    print(hdr)
    for key, axis in WIDENED:
        row = "{:<52}".format(key.replace("net.gcn.denoiser.", "dit.").replace("net.", ""))
        for s in STEPS:
            p = os.path.join(ROOT, arm, "checkpoints", s + ".pt")
            if not os.path.exists(p):
                row += "{:>14}".format("--")
                continue
            sd = load_sd(p)
            t = sd.get(key)
            if t is None:
                row += "{:>14}".format("absent")
                continue
            t = t.float()
            if t.shape[axis] <= BASE:
                row += "{:>14}".format("n/a")
                continue
            extra = t.narrow(axis, BASE, t.shape[axis] - BASE)
            row += "{:>14.3e}".format(float(extra.abs().max()))
        print(row)

    # also: did the OLD slice move at all? (proves the arm trained something)
    print()
    print("=== arm {} : |old slice - its own step32| (did anything train?) ===".format(arm))
    ref = {}
    p32 = os.path.join(ROOT, arm, "checkpoints", "step_32.pt")
    sd32 = load_sd(p32)
    for key, axis in WIDENED:
        t = sd32.get(key)
        if t is not None:
            ref[key] = t.float().narrow(axis, 0, BASE).clone()
    for key, axis in WIDENED:
        if key not in ref:
            continue
        row = "{:<52}".format(key.replace("net.gcn.denoiser.", "dit.").replace("net.", ""))
        for s in STEPS:
            p = os.path.join(ROOT, arm, "checkpoints", s + ".pt")
            if not os.path.exists(p):
                row += "{:>14}".format("--")
                continue
            sd = load_sd(p)
            t = sd.get(key)
            if t is None:
                row += "{:>14}".format("absent")
                continue
            cur = t.float().narrow(axis, 0, BASE)
            row += "{:>14.3e}".format(float((cur - ref[key]).abs().max()))
        print(row)


main()
