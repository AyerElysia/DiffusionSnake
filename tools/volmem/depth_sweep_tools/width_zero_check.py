"""Load the real weights (bypassing the missing `dill` via a stub pickle_module)
and test whether the width arms' EXTRA channels are exactly zero -- i.e. whether
the F384/F512 arms added dead capacity that can never learn.
"""
import io
import os
import pickle
import types

import torch

ROOT = ('/home/medteam/Zhrch/DiffusionSnake-12-30-pure2d-scaleup-outputs-20260808/'
        'fair_rng_six_arm_matrix_v1_r1_20260808/training')

_stub_cache = {}


class Stub(object):
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super(StubUnpickler, self).find_class(module, name)
        except Exception:
            key = (module, name)
            if key not in _stub_cache:
                _stub_cache[key] = type('Stub_%s' % name, (Stub,), {})
            return _stub_cache[key]


fake = types.ModuleType('stub_pickle')
fake.Unpickler = StubUnpickler
fake.load = pickle.load
fake.loads = pickle.loads
fake.dumps = pickle.dumps
fake.dump = pickle.dump
fake.Pickler = pickle.Pickler
fake.HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
fake.UnpicklingError = pickle.UnpicklingError


def load_sd(arm):
    p = os.path.join(ROOT, arm, 'checkpoints', 'step_2000.pt')
    obj = torch.load(p, map_location='cpu', pickle_module=fake)
    sd = obj.get('state_dict', obj) if isinstance(obj, dict) else obj
    if isinstance(sd, dict) and 'net' in sd and isinstance(sd['net'], dict):
        sd = sd['net']
    return sd


# tensors whose shape grows with F, and the axis along which the NEW slice lives
TARGETS = [
    ('locate_feat_replacer.proj.7.weight', 0),   # extra OUTPUT channels
    ('locate_feat_replacer.proj.7.bias', 0),
    ('gcn.denoiser.global_compressor.input_proj.weight', 1),  # extra INPUT cols
    ('gcn.denoiser.local_proj.0.weight', 1),
    ('gcn.denoiser.point_embed.feat_embed.0.weight', 1),
]

BASE_F = 256

for arm, cfg in [('A0', 'L6/F256'), ('B1', 'L6/F384'), ('B2', 'L6/F512'), ('C1', 'L8/F384')]:
    try:
        sd = load_sd(arm)
    except Exception as exc:  # noqa: BLE001
        print('FAIL %s: %r' % (arm, exc))
        continue
    print('=== %s (%s) ===' % (arm, cfg))
    for key, axis in TARGETS:
        t = None
        for k in sd:
            if k.endswith(key):
                t = sd[k]
                break
        if t is None:
            print('  %-52s MISSING' % key)
            continue
        n = t.shape[axis]
        if n <= BASE_F:
            print('  %-52s %-22s (not widened)' % (key, tuple(t.shape)))
            continue
        old = t.narrow(axis, 0, BASE_F)
        new = t.narrow(axis, BASE_F, n - BASE_F)
        print('  %-52s %-22s old|max|=%.6e  NEW|max|=%.6e  NEW|mean|=%.6e  new_all_zero=%s'
              % (key, tuple(t.shape), old.abs().max().item(),
                 new.abs().max().item(), new.abs().mean().item(),
                 bool(new.abs().max().item() == 0.0)))
    print()
