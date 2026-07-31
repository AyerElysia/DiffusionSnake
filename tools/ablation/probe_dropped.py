"""Why do 26 foreground val slices reach the model with n_inst=0?

summarize_all.py attributes them to the dataset filter at
lib/datasets/sagittal_2d_fixed/snake.py:1117 (`if h <= 1 or w <= 1: continue`).
That is worth checking rather than assuming: one dropped slice carries 1516 GT
foreground pixels, and a 1516-px object cannot be 1 pixel wide.

Run:
  python tools/ablation/probe_dropped.py --cfg_file configs/ablation/abl_a0_u2_single.yaml
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/home/medteam/Zhrch/DiffusionSnake-12-30')

SLICES = ('data/outputs/abl_a0_u2_single/eval_gt400_ep7/slices.json')


def main():
    from lib.config import cfg
    from lib.datasets import make_data_loader

    with open(SLICES) as f:
        rows = json.load(f)
    dropped = {}
    for r in rows:
        if r.get('gt_foreground_pixels', 0) > 0 and r.get('n_pred', 0) == 0:
            dropped.setdefault(r['case_id'], []).append(r['slice_idx'])
    print('dropped slices:', {k: sorted(v) for k, v in dropped.items()})

    loader = make_data_loader(cfg, is_train=False)
    ds = loader.dataset
    print('dataset size', len(ds), type(ds).__name__)

    # Map (case, slice) -> dataset index via the annotation list the dataset keeps.
    hits = 0
    for i in range(len(ds)):
        meta = ds.anns[i] if hasattr(ds, 'anns') else None
        if meta is None:
            print('no .anns attribute; cannot map indices')
            break
        path = meta if isinstance(meta, str) else str(meta)
        for case, idxs in dropped.items():
            if case not in path:
                continue
            for sidx in idxs:
                if 'x%04d' % sidx not in path:
                    continue
                item = ds[i]
                n_inst = len(item.get('i_it_py', []))
                print('%s x%04d -> n_inst=%d' % (case, sidx, n_inst))
                hits += 1
                if hits >= 6:
                    return
    print('probed', hits)


if __name__ == '__main__':
    main()
