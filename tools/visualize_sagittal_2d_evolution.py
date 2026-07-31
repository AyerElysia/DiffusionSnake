#!/usr/bin/env python3
"""Render GT-box sagittal evolution overlays for selected validation slices."""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg-file', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--indices', required=True, nargs='+', type=int)
    parser.add_argument(
        '--output-dir',
        default=str(_ROOT / 'visual' / 'sagittal_2d_evolution'),
    )
    parser.add_argument('--metrics-json', default='')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    return parser.parse_args()


def move_tensors(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors(item, device) for item in value)
    return value


def load_checkpoint(network, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    source = checkpoint.get('state_dict', checkpoint.get('net', checkpoint))
    state_dict = OrderedDict()
    for key, value in source.items():
        clean_key = str(key)
        while clean_key.startswith(('module.', 'net.')):
            clean_key = clean_key.split('.', 1)[1]
        state_dict[clean_key] = value
    result = network.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError('Checkpoint did not load strictly')


def sample_label(batch, index, metrics):
    meta = batch.get('meta', {})
    case_id = meta.get('case_id', ['unknown'])
    slice_idx = meta.get('slice_idx', [-1])
    if isinstance(case_id, (list, tuple)):
        case_id = case_id[0]
    if torch.is_tensor(case_id):
        case_id = case_id.reshape(-1)[0].item()
    if torch.is_tensor(slice_idx):
        slice_idx = int(slice_idx.reshape(-1)[0].item())
    elif isinstance(slice_idx, (list, tuple)):
        slice_idx = int(slice_idx[0])
    else:
        slice_idx = int(slice_idx)
    dice = metrics.get(index, {}).get('foreground_dice')
    suffix = '' if dice is None else ' Dice={:.3f}'.format(float(dice))
    return '{} slice {}{}'.format(case_id, slice_idx, suffix)


def annotate(image, title):
    canvas = cv2.copyMakeBorder(image, 46, 8, 8, 8, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(canvas, title, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(
        canvas, 'pred red | GT blue | init yellow | GT box green', (12, 39),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1, cv2.LINE_AA,
    )
    return canvas


def save_montage(paths, output_path):
    tiles = [cv2.imread(path, cv2.IMREAD_COLOR) for path in paths]
    if not tiles or any(tile is None for tile in tiles):
        raise RuntimeError('Failed to read one or more rendered overlays')
    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    normalized = [cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA) for tile in tiles]
    columns = 3
    rows = (len(normalized) + columns - 1) // columns
    blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    grid_rows = []
    for row in range(rows):
        chunk = normalized[row * columns:(row + 1) * columns]
        chunk += [blank] * (columns - len(chunk))
        grid_rows.append(np.hstack(chunk))
    cv2.imwrite(output_path, np.vstack(grid_rows))


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
    os.environ['CFG_FILE'] = os.path.abspath(args.cfg_file)

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        from lib.config import cfg
        from lib.datasets.make_dataset import make_data_loader
        from lib.evaluators.sagittal_2d_fixed import configure_box_mode
        from lib.networks import make_network
        from lib.visualizers.diffusion_one_sample import save_affine_visualization
    finally:
        sys.argv = original_argv

    cfg.test.dataset = 'SagittalPseudo3DVal'
    cfg.test.batch_size = 1
    cfg.train.num_workers = 0
    configure_box_mode(cfg, 'gt')

    metrics = {}
    if args.metrics_json:
        with open(args.metrics_json, 'r') as handle:
            metrics = {index: row for index, row in enumerate(json.load(handle))}

    loader = make_data_loader(cfg, is_train=False)
    network = make_network(cfg).to(device)
    load_checkpoint(network, args.ckpt)
    network.eval()

    requested = set(args.indices)
    if min(requested) < 0:
        raise ValueError('indices must be non-negative')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index > max(requested):
                break
            if index not in requested:
                continue
            batch = move_tensors(batch, device)
            output = network(batch['inp'], batch)
            tag = 'idx{:03d}'.format(index)
            save_affine_visualization(
                output=output,
                batch=batch,
                tag=tag,
                save_dir=str(output_dir),
            )
            path = output_dir / 'vis_affine_{}.png'.format(tag)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError('Overlay was not written: {}'.format(path))
            cv2.imwrite(str(path), annotate(image, sample_label(batch, index, metrics)))
            saved_paths.append(str(path))

    if len(saved_paths) != len(requested):
        raise RuntimeError('Rendered {} of {} requested slices'.format(len(saved_paths), len(requested)))
    montage_path = output_dir / 'montage.png'
    save_montage(saved_paths, str(montage_path))
    print('montage={}'.format(montage_path))
    for path in saved_paths:
        print('overlay={}'.format(path))


if __name__ == '__main__':
    main()
