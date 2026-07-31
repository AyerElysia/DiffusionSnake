#!/usr/bin/env python3
"""
Standalone eval script for KV-cache fix validation.
Loads .pt checkpoint directly (bypassing load_network which only handles .pth files).

Usage:
    CUDA_VISIBLE_DEVICES=0 \
    CFG_FILE=configs/eval_ep3200_ab2_s10o20.yaml \
    CKPT=data/outputs/1232_final_v4_6c_2d_fm_s_cond_gpu2/checkpoints/latest.pt \
    SAVE_DIR=visual/ab2_s10_kv_fixed \
    python test/run_eval_kv_fixed.py
"""

import os
import sys
import importlib.util
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# Pull config from environment before any lib.config import
cfg_file = os.environ.get('CFG_FILE', 'configs/eval_ep3200_ab2_s10o20.yaml')
ckpt_path = os.environ.get(
    'CKPT',
    'data/outputs/1232_final_v4_6c_2d_fm_s_cond_gpu2/checkpoints/latest.pt',
)
save_dir_str = os.environ.get('SAVE_DIR', 'visual/ab2_s10_kv_fixed')
os.environ['CFG_FILE'] = cfg_file   # lib.config reads this

import torch
from lib.config import cfg
from lib.networks import make_network

save_dir = Path(_ROOT) / save_dir_str
save_dir.mkdir(parents=True, exist_ok=True)
cfg.test.visual_save_root = str(save_dir)
cfg.use_gt_det = False          # avoid ct_01 KeyError during eval

kv_status = 'DISABLED' if os.environ.get('FLOW_DISABLE_KV_CACHE', '') in ('1', 'true', 'yes') else 'ENABLED'
print(f'[eval] cfg_file   : {cfg_file}')
print(f'[eval] checkpoint : {ckpt_path}')
print(f'[eval] save_dir   : {save_dir}')
print(f'[eval] kv_cache   : {kv_status}')

# Build network on GPU
network = make_network(cfg).cuda()
network.eval()

# Load .pt checkpoint directly
full_ckpt = os.path.join(_ROOT, ckpt_path) if not os.path.isabs(ckpt_path) else ckpt_path
print(f'[eval] Loading: {full_ckpt}')
ckpt_obj = torch.load(full_ckpt, map_location='cpu')
sd = ckpt_obj.get('state_dict') or ckpt_obj.get('net')
if sd is None:
    raise KeyError(f'No state_dict/net in checkpoint; keys={list(ckpt_obj.keys())}')

# Checkpoint is saved from DiffusionPretrainNetworkWrapper (self.net = network),
# so all keys carry a "net." prefix that the bare make_network() output does not have.
# Strip it so load_state_dict can align keys correctly.
_PREFIX = 'net.'
if all(k.startswith(_PREFIX) for k in sd.keys()):
    sd = {k[len(_PREFIX):]: v for k, v in sd.items()}
    print(f'[eval] Stripped "{_PREFIX}" prefix from all {len(sd)} checkpoint keys')
elif any(k.startswith(_PREFIX) for k in sd.keys()):
    sd = {(k[len(_PREFIX):] if k.startswith(_PREFIX) else k): v for k, v in sd.items()}
    print(f'[eval] Stripped "{_PREFIX}" prefix from some checkpoint keys')

info = network.load_state_dict(sd, strict=False)
print(f'[eval] Loaded (epoch={ckpt_obj.get("epoch","?")}): '
      f'missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}')
if info.missing_keys:
    print(f'  missing (first 5): {info.missing_keys[:5]}')
if info.unexpected_keys:
    print(f'  unexpected (first 5): {info.unexpected_keys[:5]}')

# Monkey-patch so test_medical.TEST() skips its own model loading
import lib.utils.net_utils as _nu
import lib.networks as _ln

_orig_load_network = _nu.load_network
_orig_make_network = _ln.make_network


def _noop_load(net, model_dir, resume=True, epoch=-1, strict=False):
    print('[eval] Skipping load_network (checkpoint already loaded)')
    return (ckpt_obj.get('epoch') or 0) + 1


def _cached_make_network(cfg_arg):
    print('[eval] Returning pre-loaded network')
    return network


_nu.load_network = _noop_load
_ln.make_network = _cached_make_network

try:
    test_file = Path(_ROOT) / 'test' / 'test_medical.py'
    spec = importlib.util.spec_from_file_location('test_medical', test_file)
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    tm.TEST()
finally:
    _nu.load_network = _orig_load_network
    _ln.make_network = _orig_make_network
