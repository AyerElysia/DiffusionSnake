#!/usr/bin/env python3
"""Probe: how do DiT state_dict keys grow with dit_num_layers?

Builds the network at several depths on CPU and diffs the per-layer key sets,
so we can see exactly which extra keys the newly-added layers carry.
"""
import os
import re
import sys
import pathlib

WT = pathlib.Path('/home/medteam/Zhrch/DiffusionSnake-12-30-depth-sweep-20260809')
CFG = str(WT / 'configs/volmem/depth_sweep/depth_sweep_p0_l6.yaml')

os.chdir(str(WT))
if str(WT) not in sys.path:
    sys.path.insert(0, str(WT))

os.environ['CFG_FILE'] = CFG
sys.argv = [sys.argv[0], '--cfg_file', CFG]

from lib.config import cfg          # noqa: E402
from lib.networks import make_network  # noqa: E402

LAYER_RE = re.compile(r'dit_layers\.(\d+)\.(.+)$')


def layer_key_map(depth):
    cfg.dit_num_layers = int(depth)
    net = make_network(cfg)
    sd = net.state_dict()
    per_layer = {}
    total = 0
    for k in sd:
        total += 1
        m = LAYER_RE.search(k)
        if m is None:
            continue
        idx = int(m.group(1))
        per_layer.setdefault(idx, set()).add(m.group(2))
    del net
    return total, per_layer


base_total, base_layers = layer_key_map(6)
base_suffixes = base_layers.get(0, set())
print('L=6  total_keys={}  layers={}'.format(base_total, sorted(base_layers)))
print('L=6  keys per layer: {}'.format(
    {i: len(v) for i, v in sorted(base_layers.items())}))
print()

for depth in (8, 10, 12):
    total, layers = layer_key_map(depth)
    counts = {i: len(v) for i, v in sorted(layers.items())}
    print('L={:<3} total_keys={}  keys per layer: {}'.format(depth, total, counts))
    # which layers differ from layer 0 of the L=6 build?
    for idx in sorted(layers):
        extra = layers[idx] - base_suffixes
        missing = base_suffixes - layers[idx]
        if extra or missing:
            print('   layer {:>2}: +{} extra, -{} missing'.format(
                idx, len(extra), len(missing)))
            for s in sorted(extra):
                print('        EXTRA  {}'.format(s))
            for s in sorted(missing):
                print('        MISSING {}'.format(s))
    print()
