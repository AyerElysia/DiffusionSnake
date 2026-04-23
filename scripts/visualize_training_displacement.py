import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize training displacement targets.')
    parser.add_argument('--cfg', type=str, default='', help='Path to config yaml.')
    parser.add_argument('--sample-index', type=int, default=0, help='Dataset sample index.')
    parser.add_argument('--contour-index', type=int, nargs='*', default=None, help='Optional contour indices.')
    parser.add_argument('--max-contours', type=int, default=4, help='Max contours to save when contour index is omitted.')
    parser.add_argument('--vector-stride', type=int, default=0, help='Arrow stride. 0 means auto.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for deterministic augmentation.')
    parser.add_argument('--output-dir', type=str, default='', help='Directory to save figures.')
    return parser.parse_args()


def main():
    args = parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root_dir))

    cfg_path = args.cfg or os.environ.get('CFG_FILE', '')
    if cfg_path:
        os.environ['CFG_FILE'] = os.path.abspath(cfg_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sys.argv = [sys.argv[0]]
    from lib.config import cfg
    from lib.datasets.make_dataset import make_dataset
    from lib.datasets.transforms import make_transforms
    from lib.utils.snake import snake_config
    from lib.visualizers.training_displacement import save_sample_training_displacements

    snake_config.data_rng = np.random.RandomState(args.seed)

    dataset = make_dataset(cfg, cfg.train.dataset, make_transforms(cfg, is_train=False), is_train=False)
    sample = dataset[args.sample_index]

    cfg_stem = Path(os.environ.get('CFG_FILE', 'default')).stem
    output_dir = args.output_dir
    if not output_dir:
        output_dir = root_dir / 'visual' / 'training_displacement' / cfg_stem / f'sample_{args.sample_index:05d}'
    output_dir = os.path.abspath(str(output_dir))

    saved_paths = save_sample_training_displacements(
        sample=sample,
        save_dir=output_dir,
        sample_index=args.sample_index,
        contour_indices=args.contour_index,
        max_contours=args.max_contours,
        vector_stride=(args.vector_stride if args.vector_stride > 0 else None),
    )

    print(f'cfg={os.environ.get("CFG_FILE", "")}')
    print(f'init_mode={snake_config.evolve_init}')
    print(f'sample_index={args.sample_index}')
    print(f'saved={len(saved_paths)}')
    for path in saved_paths:
        print(path)


if __name__ == '__main__':
    main()
