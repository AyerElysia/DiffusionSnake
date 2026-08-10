#!/usr/bin/env python3
"""Generate the 4 depth-sweep arm configs from the Route-B + v4_10 base config."""
import pathlib
import re
import sys

WT = pathlib.Path('/home/medteam/Zhrch/DiffusionSnake-12-30-depth-sweep-20260809')

CANDIDATES = [
    WT / 'configs/volmem/init_unify_route_B_v410.yaml',
    pathlib.Path('/home/medteam/Zhrch/DiffusionSnake-12-30-route-B-v410-20260809/'
                 'configs/volmem/init_unify_route_B_v410.yaml'),
    pathlib.Path('/home/medteam/Zhrch/DiffusionSnake-12-30/'
                 'configs/volmem/init_unify_route_B_v410.yaml'),
]

base = None
for cand in CANDIDATES:
    if cand.exists():
        base = cand
        break
if base is None:
    print('BASE CONFIG NOT FOUND in any candidate path')
    for cand in CANDIDATES:
        print('  missing:', cand)
    sys.exit(1)

print('base config =', base)
txt = base.read_text()

outdir = WT / 'configs/volmem/depth_sweep'
outdir.mkdir(parents=True, exist_ok=True)

NEW_MULT = 20.0
BASE_DEPTH = 6
ARMS = [('p0', 6), ('p1', 8), ('p2', 10), ('p3', 12)]


def sub_once(pattern, repl, text, label):
    new_text, n = re.subn(pattern, repl, text)
    if n != 1:
        raise SystemExit('pattern {!r} matched {} times (expected 1)'.format(label, n))
    return new_text


for tag, layers in ARMS:
    name = 'depth_sweep_{}_l{}'.format(tag, layers)
    t = txt
    t = sub_once(r"(?m)^model:\s*'[^']*'[ \t]*$",
                 "model: '{}'".format(name), t, 'model')
    t = sub_once(r"(?m)^model_dir:\s*'[^']*'[ \t]*$",
                 "model_dir: 'data/outputs/depth_sweep/{}'".format(name), t, 'model_dir')
    t = sub_once(r"(?m)^dit_num_layers:\s*\d+[ \t]*$",
                 "dit_num_layers: {}".format(layers), t, 'dit_num_layers')

    def inject(match):
        indent = match.group(1)
        return ("{0}warmup_steps: 200\n"
                "{0}new_layer_lr_multiplier: {1}\n"
                "{0}new_layer_base_depth: {2}").format(indent, NEW_MULT, BASE_DEPTH)

    t = sub_once(r"(?m)^([ \t]+)warmup_steps:\s*\d+[ \t]*$", inject, t, 'warmup_steps')

    header = (
        "# AUTO-GENERATED depth-sweep arm {tag}: dit_num_layers={L}\n"
        "# Base: init_unify_route_B_v410.yaml (Route B octagon init + v4_10 continuous sampling)\n"
        "#\n"
        "# Fresh DiT layers (index >= {bd}) are EXACTLY identity at init:\n"
        "#   DiTBlockV3.adaLN_modulation[-1] is zero-init (dit_blocks_v3.py:248-249) and all\n"
        "#   three residual branches are gated by it -> x = x + 0*sa + 0*ca + 0*ff = x.\n"
        "# So every arm is bit-identical to p0 at step 0; any divergence is the new capacity.\n"
        "# They get lr = base_lr * {mult} so the gates can lift off zero inside a short screen.\n"
        "# base_lr stays 1e-5, identical to mainline, so pretrained weights evolve as control.\n"
    ).format(tag=tag, L=layers, bd=BASE_DEPTH, mult=NEW_MULT)

    path = outdir / '{}.yaml'.format(name)
    path.write_text(header + t)
    print('wrote', path)

print('OK')
