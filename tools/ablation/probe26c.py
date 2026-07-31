#!/usr/bin/env python3
"""Verify cfg.min_poly_area_output: default keeps old behaviour, lower recovers GT."""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
_argv = sys.argv[:]
sys.argv = [sys.argv[0], '--cfg_file', 'configs/ablation/abl_a0_u2_single.yaml']
from lib.config import cfg  # noqa: E402
sys.argv = _argv

MANIFEST = ('/home/medteam/Zhrch/detect_3D_lgz2/datasets/sagittal_2d_fixed/'
            'manifests/bbox_2d_manifest.csv')
cfg.train.bbox_2d_manifest = MANIFEST

from lib.datasets.sagittal_2d_fixed.snake import Dataset  # noqa: E402

ROOT = '/home/medteam/Zhrch/detect_3D_lgz2/datasets/sagittal_2d_fixed'
ds = Dataset(os.path.join(ROOT, 'manifests', 'slice_manifest.csv'), ROOT, 'val')

# the 26 zero-prediction slices found in the gt400 eval
BAD = [('sub-verse010', 45), ('sub-verse010', 146), ('sub-verse010', 147),
       ('sub-verse013', 35), ('sub-verse013', 36), ('sub-verse013', 37),
       ('sub-verse013', 38), ('sub-verse013', 39), ('sub-verse013', 63),
       ('sub-verse013', 64), ('sub-verse013', 65), ('sub-verse013', 66),
       ('sub-verse013', 67), ('sub-verse013', 68), ('sub-verse016', 8),
       ('sub-verse016', 9), ('sub-verse016', 10), ('sub-verse016', 11),
       ('sub-verse016', 12), ('sub-verse016', 13), ('sub-verse016', 36),
       ('sub-verse016', 37), ('sub-verse016', 38), ('sub-verse016', 39),
       ('sub-verse016', 40), ('sub-verse016', 41)]
bad_set = set(BAD)
idx = {}
for i, row in enumerate(ds.records):
    key = (row['case_id'], int(row['slice_idx']))
    if key in bad_set:
        idx[key] = i

for thr in (5.0, 1.0, 0.5, 0.1):
    cfg.min_poly_area_output = thr
    nz = 0
    total_inst = 0
    for key in BAD:
        ret = ds[idx[key]]
        n = len(ret['ct_ind'])
        total_inst += n
        if n == 0:
            nz += 1
    print('min_area={:<4} zero-instance slices {:2d}/26  total instances {}'.format(
        thr, nz, total_inst))

# sanity: a normal slice must be unaffected by the threshold
probe = [('sub-verse010', 100), ('sub-verse013', 100)]
pidx = {}
for i, row in enumerate(ds.records):
    if (row['case_id'], int(row['slice_idx'])) in probe:
        pidx[(row['case_id'], int(row['slice_idx']))] = i
for key, i in sorted(pidx.items()):
    counts = []
    for thr in (5.0, 0.5):
        cfg.min_poly_area_output = thr
        counts.append(len(ds[i]['ct_ind']))
    print('normal slice', key, 'n_inst at 5.0 / 0.5 =', counts)
