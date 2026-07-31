#!/usr/bin/env python3
"""Predict the exact GT box count arm H should report, as a self-check.

get_valid_polys applies the output-space area floor, but prepare_detection is
then skipped again by the `h <= 1 or w <= 1` guard at snake.py:1146. Only the
instances surviving BOTH are turned into GT boxes, and with --box-mode gt the
evaluator's total_predicted_boxes equals exactly that count.

F/A0 reported total_predicted_boxes=1228, which matches kept_inst=1228 at the
inherited floor of 5.0 - i.e. at that floor the second guard removes nothing.
This script computes the same number at floor 0.5 so the H eval can be checked
against a prediction made before it finished.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
_argv = sys.argv[:]
sys.argv = [sys.argv[0], '--cfg_file', 'configs/ablation/abl_a0_u2_single.yaml']
from lib.config import cfg  # noqa: E402
sys.argv = _argv

cfg.train.bbox_2d_manifest = ('/home/medteam/Zhrch/detect_3D_lgz2/datasets/'
                              'sagittal_2d_fixed/manifests/bbox_2d_manifest.csv')

from lib.datasets.sagittal_2d_fixed.snake import Dataset  # noqa: E402
from lib.utils.snake import snake_voc_utils  # noqa: E402

ROOT = '/home/medteam/Zhrch/detect_3D_lgz2/datasets/sagittal_2d_fixed'
ds = Dataset(os.path.join(ROOT, 'manifests', 'slice_manifest.csv'), ROOT, 'val')

N = 400
INP, OUT = 512, 128
scale_hw = (INP, INP, OUT, OUT)

for floor in (5.0, 1.0, 0.5):
    cfg.min_poly_area_output = floor
    kept = boxed = 0
    zero_box_slices = 0
    fg = 0
    for i in range(N):
        row = ds.records[i]
        mask = cv2.imread(row['mask_path'], cv2.IMREAD_UNCHANGED)
        if mask is None or not (mask > 0).any():
            continue
        fg += 1
        polys, _ = ds._mask_to_instances(mask)
        h, w = mask.shape
        c = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        s = float(max(h, w))
        trans = snake_voc_utils.data_utils.get_affine_transform(c, s, 0, [OUT, OUT])
        t = ds.transform_original_data(polys, False, w, trans, scale_hw)
        v = ds.get_valid_polys(t, scale_hw)
        n_box = 0
        for inst in v:
            for poly in inst:
                kept += 1
                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                ph, pw = y_max - y_min + 1, x_max - x_min + 1
                if ph <= 1 or pw <= 1:      # snake.py:1146, second gate
                    continue
                n_box += 1
        boxed += n_box
        if n_box == 0:
            zero_box_slices += 1
    print('floor={:<4} fg={} kept_by_area={} -> GT_boxes={} (lost_to_hw_guard={}) '
          'zero_box_slices={}'.format(floor, fg, kept, boxed, kept - boxed,
                                      zero_box_slices))
