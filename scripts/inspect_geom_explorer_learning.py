#!/usr/bin/env python3
"""Inspect whether the V5 geom FourierExplorer moved off zero-init."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT_DIR))

_DEFAULT_CFG = _ROOT_DIR / 'configs' / '1232_final_v5_geom_learned_probe_gpu0.yaml'
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--cfg_file', default=str(_DEFAULT_CFG), type=str)
_pre_args, _remaining_argv = _pre_parser.parse_known_args()
os.environ['CFG_FILE'] = str(_pre_args.cfg_file)
sys.argv = [sys.argv[0], '--cfg_file', str(_pre_args.cfg_file)]

from lib.config import cfg
from lib.datasets import make_data_loader
from grpo_train_v5_geom_action import (
    FourierExplorer,
    _flatten_valid_polys,
    _fourier_band_basis,
    _make_py_ind,
)


FRACTIONS = [0.2, 0.25, 0.3333, 0.5, 1.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inspect V5 geom-action FourierExplorer low-mode learning.'
    )
    parser.add_argument('--cfg_file', default=str(_pre_args.cfg_file), type=str)
    parser.add_argument('--ckpt', default='', type=str, help='Checkpoint .pt path.')
    parser.add_argument('--ckpt_dir', default='', type=str, help='Directory to scan for latest .pt.')
    parser.add_argument('--compare_ckpts', nargs='*', default=[], help='Checkpoint paths to report in order.')
    parser.add_argument('--zero_init_only', action='store_true', help='Skip checkpoint load.')
    parser.add_argument('--max_samples', default=30, type=int)
    return parser.parse_args([sys.argv[1], sys.argv[2], *_remaining_argv])


def resolve_latest_ckpt(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    candidates = [p for p in ckpt_dir.glob('*.pt') if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f'No .pt checkpoints found in {ckpt_dir}')
    return max(candidates, key=lambda p: p.stat().st_mtime)


def make_explorer(device):
    detail_modes = _fourier_band_basis(
        128, 9, 24, torch.device('cpu'), torch.float32
    ).size(1)
    explorer = FourierExplorer(
        low_modes=8,
        detail_modes=detail_modes,
        hidden_dim=64,
        mu_max=0.50,
        logstd_min=-1.5,
        logstd_max=0.7,
    )
    return explorer.to(device).eval(), int(detail_modes)


def load_explorer(ckpt_path, device):
    explorer, detail_modes = make_explorer(device)
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    if not isinstance(ckpt, dict):
        raise TypeError(f'Checkpoint must be a dict, got {type(ckpt).__name__}: {ckpt_path}')
    state_dict = ckpt.get('explorer_state_dict')
    if state_dict is None:
        print(f'[*] {ckpt_path}: explorer_state_dict is None; using zero-init explorer.')
        return explorer, detail_modes
    info = explorer.load_state_dict(state_dict, strict=True)
    print(
        f'[*] Loaded explorer: {ckpt_path} | detail_modes={detail_modes} '
        f'missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}'
    )
    return explorer, detail_modes


def move_batch(batch, device):
    for key in list(batch.keys()):
        if key in ('meta', 'orig_img', 'img_path'):
            continue
        value = batch[key]
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)
    return batch


@torch.no_grad()
def collect_polys(device, max_samples):
    loader = make_data_loader(cfg, is_train=True, is_distributed=False)
    polys = []
    seen_samples = 0
    for batch in loader:
        batch = move_batch(batch, device)
        poly = _flatten_valid_polys(batch, 'i_it_py', device=device)
        _make_py_ind(batch, poly.size(0), device=device)
        if poly.numel() > 0:
            polys.append(poly)
        batch_size = int(batch['inp'].size(0)) if isinstance(batch.get('inp'), torch.Tensor) else 1
        seen_samples += batch_size
        if max_samples > 0 and seen_samples >= max_samples:
            break
    if not polys:
        raise RuntimeError('No valid i_it_py contours found in the requested samples.')
    return torch.cat(polys, dim=0)


@torch.no_grad()
def forward_low_stats(explorer, polys):
    low_mu_all = []
    low_logstd_all = []
    for frac in FRACTIONS:
        low_mu, low_logstd, _, _ = explorer(polys, float(frac))
        low_mu_all.append(low_mu.detach().cpu())
        low_logstd_all.append(low_logstd.detach().cpu())
    return torch.cat(low_mu_all, dim=0), torch.cat(low_logstd_all, dim=0)


def summarize(label, explorer, zero_explorer, polys):
    low_mu, low_logstd = forward_low_stats(explorer, polys)
    zero_mu, zero_logstd = forward_low_stats(zero_explorer, polys)

    logstd_mean = low_logstd.mean(dim=0).numpy()
    logstd_std = low_logstd.std(dim=0, unbiased=False).numpy()
    abs_mu_mean = low_mu.abs().mean(dim=0).numpy()
    max_logstd_dev = float((low_logstd - zero_logstd).abs().max().item())
    max_mu_dev = float((low_mu - zero_mu).abs().max().item())

    print('')
    print(f'=== {label} ===')
    print(f'contours={polys.size(0)} fractions={FRACTIONS} total_forward={low_mu.size(0)}')
    print('per-mode low_logstd mean:')
    print(np.array2string(logstd_mean, precision=6, floatmode='fixed'))
    print('per-mode low_logstd std:')
    print(np.array2string(logstd_std, precision=6, floatmode='fixed'))
    print('per-mode |low_mu| mean:')
    print(np.array2string(abs_mu_mean, precision=6, floatmode='fixed'))
    print(f'max |logstd| deviation from zero-init: {max_logstd_dev:.6f}')
    print(f'max |mu| deviation from zero-init: {max_mu_dev:.6f}')
    if max_logstd_dev > 0.05 or max_mu_dev > 0.02:
        print('LEARNING: explorer has moved off zero-init')
    else:
        print('FLAT: explorer still ~zero-init (advantage signal too weak)')


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[*] cfg_file={args.cfg_file}')
    print(f'[*] device={device}')

    polys = collect_polys(device, int(args.max_samples))
    zero_explorer, detail_modes = make_explorer(device)
    print(f'[*] detail_modes={detail_modes}')

    if args.zero_init_only:
        explorer, _ = make_explorer(device)
        summarize('zero-init explorer', explorer, zero_explorer, polys)
        return

    ckpt_paths = []
    if args.compare_ckpts:
        ckpt_paths.extend(args.compare_ckpts)
    elif args.ckpt:
        ckpt_paths.append(args.ckpt)
    elif args.ckpt_dir:
        ckpt_paths.append(str(resolve_latest_ckpt(args.ckpt_dir)))
    else:
        default_dir = _ROOT_DIR / 'data' / 'outputs' / '1232_final_v5_geom_learned_probe_gpu0' / 'checkpoints'
        ckpt_paths.append(str(resolve_latest_ckpt(default_dir)))

    for ckpt_path in ckpt_paths:
        explorer, _ = load_explorer(ckpt_path, device)
        summarize(str(ckpt_path), explorer, zero_explorer, polys)


if __name__ == '__main__':
    main()
