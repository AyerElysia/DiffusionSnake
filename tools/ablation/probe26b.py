#!/usr/bin/env python3
"""Locate the exact filter that drops 26 foreground val slices."""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
os.environ.setdefault('CFG_FILE', 'configs/ablation/abl_a0_u2_single.yaml')
_argv = sys.argv[:]
sys.argv = [sys.argv[0], '--cfg_file', os.environ['CFG_FILE']]
from lib.config import cfg  # noqa: E402
sys.argv = _argv

from lib.datasets.sagittal_2d_fixed.snake import Dataset  # noqa: E402
from lib.utils.snake import snake_voc_utils, snake_config  # noqa: E402

ROOT = '/home/medteam/Zhrch/detect_3D_lgz2/datasets/sagittal_2d_fixed'
ds = Dataset(os.path.join(ROOT, 'manifests', 'slice_manifest.csv'), ROOT, 'val')

targets = [('sub-verse013', 63), ('sub-verse016', 12), ('sub-verse010', 146),
           ('sub-verse016', 40)]
idx = {}
for i, row in enumerate(ds.records):
    key = (row['case_id'], int(row['slice_idx']))
    if key in targets:
        idx[key] = i

for key in targets:
    i = idx[key]
    row = ds.records[i]
    img = ds._read_grayscale_image(row['image_path'])
    image = np.repeat(img[:, :, None], 3, axis=2)
    height, width = img.shape
    mask = ds._read_mask(row['mask_path'], img.shape)
    polys0, cls_ids = ds._mask_to_instances(mask)

    (augmented, inp, trans_input, trans_output, flipped, center, scale,
     inp_out_hw) = snake_voc_utils.augment(
        image, ds.split, snake_config.data_rng, snake_config.eig_val,
        snake_config.eig_vec, ds.mean, ds.std, polys0,
        color_aug=ds.color_aug, lr_flip=ds.lr_flip, random_crop=ds.random_crop)

    p1 = ds.transform_original_data(polys0, flipped, width, trans_output, inp_out_hw)
    n_after_tf = sum(len(x) for x in p1)
    areas = []
    for inst in p1:
        for q in inst:
            if len(q) >= 4:
                try:
                    areas.append(round(float(Polygon(q).area), 3))
                except Exception:
                    areas.append(-1.0)
    p2 = ds.get_valid_polys(p1, inp_out_hw)
    n_after_valid = sum(len(x) for x in p2)

    print('{} orig_hw={} inp_out_hw={} n_mask_inst={} n_after_transform={} '
          'n_after_get_valid={}'.format(
              key, (height, width), inp_out_hw, len(polys0), n_after_tf, n_after_valid))
    print('    output-space polygon areas (filter keeps area>5): {}'.format(areas))
