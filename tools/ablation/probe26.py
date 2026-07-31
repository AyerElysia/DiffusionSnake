#!/usr/bin/env python3
"""Probe 26 zero-instance slices + effect of the missing bbox_2d_manifest."""
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
os.environ.setdefault('CFG_FILE', 'configs/ablation/abl_a0_u2_single.yaml')
_argv = sys.argv[:]
sys.argv = [sys.argv[0], '--cfg_file', os.environ['CFG_FILE']]
from lib.config import cfg  # noqa: E402
sys.argv = _argv

# Inject the manifest the ablation configs forgot, so foreground_flags is real.
MANIFEST = ('/home/medteam/Zhrch/detect_3D_lgz2/datasets/sagittal_2d_fixed/'
            'manifests/bbox_2d_manifest.csv')
cfg.train.bbox_2d_manifest = MANIFEST

from lib.datasets.sagittal_2d_fixed.snake import Dataset  # noqa: E402

ROOT = '/home/medteam/Zhrch/detect_3D_lgz2/datasets/sagittal_2d_fixed'
ds = Dataset(os.path.join(ROOT, 'manifests', 'slice_manifest.csv'), ROOT, 'val')
n = len(ds)
nfg = sum(bool(f) for f in ds.foreground_flags)
print('len={} fg={} bg={} fg_frac={:.4f}'.format(n, nfg, n - nfg, nfg / float(n)))

targets = [('sub-verse013', 63), ('sub-verse016', 12), ('sub-verse010', 146),
           ('sub-verse013', 35), ('sub-verse016', 40)]
idx = {}
for i, row in enumerate(ds.records):
    key = (row['case_id'], int(row['slice_idx']))
    if key in targets:
        idx[key] = i

for key in targets:
    i = idx[key]
    row = ds.records[i]
    import cv2
    mask = cv2.imread(row['mask_path'], cv2.IMREAD_UNCHANGED)
    polys, cls_ids = ds._mask_to_instances(mask)
    npts = [len(p[0]) for p in polys]
    hw = []
    for p in polys:
        q = p[0]
        hw.append((int(q[:, 1].max() - q[:, 1].min() + 1),
                   int(q[:, 0].max() - q[:, 0].min() + 1)))
    print(key, 'fg_flag', ds.foreground_flags[i],
          'mask_px', int((mask > 0).sum()),
          'n_inst_from_mask', len(polys), 'cls', cls_ids,
          'npts', npts, 'hw', hw)
