#!/usr/bin/env python3
"""Evaluate MoonViT sagittal pseudo-3D predictions without starting training."""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_CFG_FILE = _ROOT / 'configs' / 'sagittal_2d_pseudo3d_moonvit.yaml'


def _cfg_file_from_argv():
    for index, value in enumerate(sys.argv[1:]):
        if value == '--cfg_file' and index + 2 <= len(sys.argv[1:]):
            return sys.argv[index + 2]
        if value.startswith('--cfg_file='):
            return value.split('=', 1)[1]
    return os.environ.get('CFG_FILE', '')


def _resolve_cfg_file():
    return _cfg_file_from_argv() or str(_DEFAULT_CFG_FILE)


_original_argv = sys.argv[:]
_original_cfg_file = os.environ.get('CFG_FILE')
_cfg_file = _resolve_cfg_file()
os.environ['CFG_FILE'] = _cfg_file
try:
    sys.argv = [sys.argv[0]]
    from lib.config import cfg
    from lib.datasets.make_dataset import make_data_loader
    from lib.evaluators.sagittal_2d_fixed import Evaluator, configure_box_mode
    from lib.networks import make_network
finally:
    sys.argv = _original_argv
    if _original_cfg_file is None:
        os.environ.pop('CFG_FILE', None)
    else:
        os.environ['CFG_FILE'] = _original_cfg_file


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg_file', default=_cfg_file, help='YAML config file')
    parser.add_argument('--ckpt', required=True, help='checkpoint to evaluate')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--box-mode', choices=('gt', 'predicted'), default='predicted')
    parser.add_argument('--result-dir', required=True)
    parser.add_argument('--max-slices', type=int, default=None)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return parser.parse_args()


def _resolve_device(requested):
    requested = str(requested).strip().lower()
    if requested == 'cpu':
        return torch.device('cpu')
    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but torch.cuda.is_available() is false')
        return torch.device('cuda:0')
    if requested == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    raise ValueError('Unsupported device: {!r}'.format(requested))


def _move_tensors(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        moved = {}
        for key, item in value.items():
            if isinstance(key, str) and (key == 'locate_feat' or key.startswith('locate_feat_')):
                moved[key] = item
            else:
                moved[key] = _move_tensors(item, device)
        return moved
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    return value


def _strip_repeated_prefixes(key):
    prefixes = ('module.', 'net.')
    clean = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                changed = True
                break
    return clean


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError('Checkpoint must be a mapping or a state-dict mapping')
    for key in ('state_dict', 'net', 'model'):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            return nested
    return checkpoint


def _load_checkpoint_file(checkpoint_path):
    """Load trusted local checkpoints across PyTorch versions."""
    try:
        return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location='cpu')


def load_checkpoint(network, checkpoint_path):
    checkpoint = _load_checkpoint_file(checkpoint_path)
    source = _extract_state_dict(checkpoint)
    state_dict = OrderedDict()
    for key, value in source.items():
        clean_key = _strip_repeated_prefixes(key)
        if clean_key in state_dict:
            raise RuntimeError('Checkpoint key collision after prefix stripping: {}'.format(clean_key))
        state_dict[clean_key] = value

    target_state = network.state_dict()
    missing_keys = sorted(set(target_state) - set(state_dict))
    unexpected_keys = sorted(set(state_dict) - set(target_state))
    if missing_keys or unexpected_keys:
        details = []
        if missing_keys:
            details.append('missing keys: {}'.format(missing_keys))
        if unexpected_keys:
            details.append('unexpected keys: {}'.format(unexpected_keys))
        raise RuntimeError(
            'Checkpoint/model architecture mismatch; verify --cfg_file:\n{}'.format(
                '\n'.join(details)
            )
        )

    mismatches = []
    for key, value in state_dict.items():
        target = target_state[key]
        if not torch.is_tensor(value):
            mismatches.append('{}: checkpoint value is not a tensor'.format(key))
            continue
        source_shape = tuple(value.shape)
        target_shape = tuple(target.shape)
        if source_shape != target_shape:
            mismatches.append('{}: checkpoint {} vs model {}'.format(
                key, source_shape, target_shape
            ))
        elif value.layout != target.layout:
            mismatches.append('{}: checkpoint layout {} vs model {}'.format(
                key, value.layout, target.layout
            ))
    if mismatches:
        raise RuntimeError('Checkpoint/model tensor mismatch:\n{}'.format('\n'.join(mismatches)))

    result = network.load_state_dict(state_dict, strict=True)
    print('checkpoint loaded: missing=0 unexpected=0')
    return result


def preflight_locate_cache(dataset, max_slices=None):
    """Reject incomplete MoonViT evaluation subsets before model startup."""
    if not getattr(dataset, 'locate_feat_enabled', False):
        return 0
    limit = len(dataset.records)
    if max_slices is not None:
        limit = min(limit, int(max_slices))
    missing = []
    for row in dataset.records[:limit]:
        path = dataset._sagittal_moonvit_cache_path(row)
        if not os.path.isfile(path):
            missing.append(path)
    if missing:
        raise FileNotFoundError(
            'MoonViT cache is incomplete for the requested evaluation subset: '
            '{} of {} slices missing; first missing {}. Extract the cache or '
            'reduce --max-slices to a fully cached case.'.format(
                len(missing), limit, missing[0]
            )
        )
    return limit


def main():
    args = _parse_args()
    if args.max_slices is not None and args.max_slices < 0:
        raise ValueError('--max-slices must be non-negative')
    if args.cfg_file:
        cfg_file = os.fspath(args.cfg_file)
        if not os.path.isfile(cfg_file):
            raise FileNotFoundError('Config file not found: {}'.format(cfg_file))

    dataset_name = 'SagittalPseudo3DVal' if args.split == 'val' else 'SagittalPseudo3DTest'
    cfg.test.dataset = dataset_name
    cfg.test.batch_size = 1
    cfg.train.num_workers = 0
    cfg.result_dir = args.result_dir
    configure_box_mode(cfg, args.box_mode)

    device = _resolve_device(args.device)
    loader = make_data_loader(cfg, is_train=False)
    preflight_locate_cache(loader.dataset, args.max_slices)
    network = make_network(cfg).to(device)
    load_checkpoint(network, args.ckpt)
    network.eval()

    evaluator = Evaluator(args.result_dir)
    processed = 0
    loader_iter = iter(loader)
    with torch.no_grad():
        while args.max_slices is None or processed < args.max_slices:
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            batch = _move_tensors(batch, device)
            output = network(batch['inp'].to(device), batch)
            evaluator.evaluate(output, batch)
            processed += 1

    summary = evaluator.summarize()
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return summary


if __name__ == '__main__':
    main()
