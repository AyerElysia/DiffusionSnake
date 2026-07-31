#!/usr/bin/env python3
"""Evaluate GT-box sagittal temporal propagation and render selected slices."""

import argparse
import json
import os
import random
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
    parser.add_argument('--max-slices', type=int, default=64)
    parser.add_argument(
        '--visual-indices', nargs='*', type=int,
        default=[45, 52, 53, 54, 57, 59],
    )
    parser.add_argument(
        '--output-dir',
        default=str(_ROOT / 'visual' / 'sagittal_temporal'),
    )
    parser.add_argument('--seed', type=int, default=20260727)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    return parser.parse_args()


def move_tensors(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
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


def meta_scalar(meta, key, default='unknown'):
    value = meta.get(key, default) if isinstance(meta, dict) else default
    if torch.is_tensor(value):
        value = value.reshape(-1)[0].item() if value.numel() else default
    elif isinstance(value, (list, tuple)):
        value = value[0] if value else default
    return value


def annotate(image, title, init_source):
    canvas = cv2.copyMakeBorder(image, 66, 8, 8, 8, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(canvas, title, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(
        canvas, 'pred red | GT blue | init yellow | GT box green', (12, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, init_source, (12, 57),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 230, 180), 1, cv2.LINE_AA,
    )
    return canvas


def save_montage(paths, output_path):
    tiles = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
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
    if not cv2.imwrite(str(output_path), np.vstack(grid_rows)):
        raise RuntimeError('Failed to write montage: {}'.format(output_path))


def main():
    args = parse_args()
    if args.max_slices <= 0:
        raise ValueError('--max-slices must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    if min(args.visual_indices, default=0) < 0:
        raise ValueError('--visual-indices must be non-negative')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
    os.environ['CFG_FILE'] = os.path.abspath(args.cfg_file)

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        from lib.config import cfg
        from lib.datasets.make_dataset import make_data_loader
        from lib.evaluators.sagittal_2d_fixed import configure_box_mode
        from lib.evaluators.sagittal_2d_fixed.snake import Evaluator
        from lib.networks import make_network
        from lib.visualizers.diffusion_one_sample import save_affine_visualization
        from lib.utils.snake.prev_contour_init import cache_previous_predictions
    finally:
        sys.argv = original_argv

    cfg.train_or_test = 'test'
    cfg.test.dataset = 'SagittalPseudo3DVal'
    cfg.test.batch_size = 1
    cfg.train.num_workers = 0
    configure_box_mode(cfg, 'gt')
    # The GT path supplies boxes directly; do not execute the detector at all.
    cfg.freeze_heatmap_detector = True
    cfg.skip_heatmap_detector_when_gt = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = make_data_loader(cfg, is_train=False, is_distributed=False)
    network = make_network(cfg).to(device)
    load_checkpoint(network, args.ckpt)
    network.eval()
    evaluator = Evaluator(str(output_dir), config=cfg)

    requested = set(args.visual_indices)
    saved_paths = []
    prev_contour_cache = {}
    prev_case_id = None

    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= args.max_slices:
                break
            batch = move_tensors(batch, device)
            meta = batch.get('meta', {})
            case_id = str(meta_scalar(meta, 'case_id'))
            slice_idx = int(meta_scalar(meta, 'slice_idx', -1))
            if case_id != prev_case_id:
                prev_contour_cache = {}
                prev_case_id = case_id

            cache_before = prev_contour_cache
            batch['prev_contour_cache'] = cache_before
            output = network(batch['inp'], batch)
            evaluator.evaluate(output, batch)
            metrics = evaluator.results[-1]

            if index in requested:
                visual_output = dict(output)
                if torch.is_tensor(output.get('i_it_py')):
                    visual_output['it_py'] = output['i_it_py']
                tag = 'idx{:03d}_slice{:04d}'.format(index, slice_idx)
                from lib.visualizers.diffusion_one_sample import save_affine_visualization
                save_affine_visualization(
                    output=visual_output,
                    batch=batch,
                    tag=tag,
                    save_dir=str(output_dir),
                )
                path = output_dir / 'vis_affine_{}.png'.format(tag)
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError('Overlay was not written: {}'.format(path))
                detection = output['detection'][0]
                current_labels = {
                    int(label)
                    for label in detection[detection[:, 4] > 1e-4, 5].detach().cpu().tolist()
                }
                propagated = len(current_labels.intersection(cache_before))
                init_source = 'temporal init: {} propagated, {} GT-box fallback'.format(
                    propagated, len(current_labels) - propagated,
                )
                title = '{} slice {} Dice={:.3f} IoU={:.3f}'.format(
                    case_id,
                    slice_idx,
                    float(metrics['foreground_dice']),
                    float(metrics['foreground_iou']),
                )
                if not cv2.imwrite(str(path), annotate(image, title, init_source)):
                    raise RuntimeError('Failed to annotate overlay: {}'.format(path))
                saved_paths.append(path)

            prev_contour_cache = cache_previous_predictions(output)

    if len(evaluator.results) != args.max_slices:
        raise RuntimeError('Evaluated {} of {} requested slices'.format(
            len(evaluator.results), args.max_slices,
        ))
    missing = sorted(requested.difference(range(args.max_slices)))
    if missing:
        raise ValueError('--visual-indices outside evaluated range: {}'.format(missing))
    summary = evaluator.summarize()
    summary.update({
        'temporal_propagation': True,
        'detector_executed': False,
        'box_mode': 'gt',
        'checkpoint': os.path.abspath(args.ckpt),
        'max_slices': int(args.max_slices),
        'seed': int(args.seed),
    })
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)

    montage_path = output_dir / 'montage.png'
    if saved_paths:
        save_montage(saved_paths, montage_path)
    print('summary={}'.format(output_dir / 'summary.json'))
    if saved_paths:
        print('montage={}'.format(montage_path))
    print('evaluated_slices={}'.format(len(evaluator.results)))
    print('foreground_dice={:.6f}'.format(summary['foreground_slice_mean_dice']))
    print('foreground_iou={:.6f}'.format(summary['foreground_slice_mean_iou']))


if __name__ == '__main__':
    main()
