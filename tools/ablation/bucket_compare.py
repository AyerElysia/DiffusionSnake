#!/usr/bin/env python
"""Paired per-slice comparison of two arms, bucketed by GT foreground area.

If the bottleneck were spatial sampling resolution, a higher-capacity arm must
win on the small-object buckets. A uniform loss across all buckets points at
optimization difficulty instead.

Usage:
  python tools/ablation/bucket_compare.py base=<slices.json> cand=<slices.json>
"""
from __future__ import print_function
import json
import sys

BUCKETS = [('tiny <300', 0, 300), ('small 300-1k', 300, 1000),
           ('mid 1k-3k', 1000, 3000), ('large >3k', 3000, float('inf'))]


def load(path):
    out = {}
    for s in json.load(open(path)):
        if s.get('gt_foreground_pixels', 0) <= 0:
            continue          # pure-background slices score ~1.0 and only add noise
        if s.get('n_pred', 0) <= 0:
            continue          # dropped by the dataset h<=1/w<=1 filter, not by the model
        out[(s['case_id'], s['slice_idx'])] = s
    return out


def main():
    args = dict(a.split('=', 1) for a in sys.argv[1:])
    base, cand = load(args['base']), load(args['cand'])
    keys = sorted(set(base) & set(cand))
    print('paired slices: %d' % len(keys))
    print('%-14s %5s %8s %8s %9s %8s' % ('bucket', 'n', 'base', 'cand', 'delta', 'win/loss'))
    for name, lo, hi in BUCKETS:
        sel = [k for k in keys if lo <= base[k]['gt_foreground_pixels'] < hi]
        if not sel:
            continue
        b = sum(base[k]['foreground_dice'] for k in sel) / len(sel)
        c = sum(cand[k]['foreground_dice'] for k in sel) / len(sel)
        win = sum(1 for k in sel if cand[k]['foreground_dice'] > base[k]['foreground_dice'] + 1e-6)
        loss = sum(1 for k in sel if cand[k]['foreground_dice'] < base[k]['foreground_dice'] - 1e-6)
        print('%-14s %5d %8.4f %8.4f %+9.4f %5d/%d' % (name, len(sel), b, c, c - b, win, loss))
    b = sum(base[k]['foreground_dice'] for k in keys) / len(keys)
    c = sum(cand[k]['foreground_dice'] for k in keys) / len(keys)
    win = sum(1 for k in keys if cand[k]['foreground_dice'] > base[k]['foreground_dice'] + 1e-6)
    loss = sum(1 for k in keys if cand[k]['foreground_dice'] < base[k]['foreground_dice'] - 1e-6)
    print('%-14s %5d %8.4f %8.4f %+9.4f %5d/%d' % ('ALL', len(keys), b, c, c - b, win, loss))


if __name__ == '__main__':
    main()
