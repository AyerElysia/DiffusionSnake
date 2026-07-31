#!/usr/bin/env python3
"""Quantify how much GT the output-space area floor discards over the eval subset,
and verify the full __getitem__ pipeline stays finite at the lowered floor."""
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

stats = {}
for floor in (5.0, 1.0, 0.5):
    cfg.min_poly_area_output = floor
    n_fg = n_zero = n_inst = n_mask_inst = 0
    for i in range(N):
        row = ds.records[i]
        mask = cv2.imread(row['mask_path'], cv2.IMREAD_UNCHANGED)
        if mask is None or not (mask > 0).any():
            continue
        n_fg += 1
        polys, _ = ds._mask_to_instances(mask)
        n_mask_inst += len(polys)
        h, w = mask.shape
        c = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        s = float(max(h, w))
        trans = snake_voc_utils.data_utils.get_affine_transform(c, s, 0, [OUT, OUT])
        t = ds.transform_original_data(polys, False, w, trans, scale_hw)
        v = ds.get_valid_polys(t, scale_hw)
        k = sum(len(x) for x in v)
        n_inst += k
        if k == 0:
            n_zero += 1
    stats[floor] = (n_fg, n_zero, n_inst, n_mask_inst)
    print('floor={:<4} fg_slices={} zero_inst_slices={} kept_inst={} mask_inst={} '
          'kept_frac={:.3f}'.format(floor, n_fg, n_zero, n_inst, n_mask_inst,
                                    n_inst / float(max(n_mask_inst, 1))))

# Full pipeline sanity at the lowered floor on previously-empty slices.
cfg.min_poly_area_output = 0.5
targets = [('sub-verse013', 63), ('sub-verse016', 12), ('sub-verse010', 146)]
idx = {}
for i, row in enumerate(ds.records):
    key = (row['case_id'], int(row['slice_idx']))
    if key in targets:
        idx[key] = i
for key in targets:
    ret = ds[idx[key]]
    py = np.asarray(ret['i_it_py'])
    gt = np.asarray(ret['i_gt_py'])
    ok = (py.size == 0) or (np.isfinite(py).all() and np.isfinite(gt).all())
    print('{} n_inst={} i_it_py={} finite={}'.format(
        key, len(ret['ct_ind']), py.shape, ok))
