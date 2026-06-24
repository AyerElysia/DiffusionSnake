#!/usr/bin/env python
"""Convert MoonViT locate-feature cache from compressed .npz to uncompressed .npy.

The .npz files are zip containers; np.load must inflate ~18MB per file even when the
bytes are fully resident in the page cache (~0.4s/file decompress => 6.3s/step @ bs16).
MoonViT features are near-incompressible, so writing a raw uncompressed .npy lets the
dataloader memmap them with zero decompress (~6ms/step), and disk barely grows.

For each <stem>.npz it writes:
  <dst>/<split>/<stem>.npy       -> concat(keys, axis=0) fp16, shape (sum_C, H, W)
  <dst>/<split>/<stem>.meta.npz  -> all remaining (small scalar) arrays, uncompressed

Resumable: existing outputs are skipped.
"""
import argparse
import glob
import os
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--src-root', default='data/f1_cache/locate896_full/locate_896',
                   help='cache root containing <split>/<stem>.npz')
    p.add_argument('--split', default='both', choices=['train', 'test', 'both'])
    p.add_argument('--keys', default='layer_18,layer_26',
                   help='comma-separated feature keys, concatenated in this order along axis 0')
    p.add_argument('--dst-suffix', default='_npy',
                   help='dst root = src_root.rstrip("/") + dst_suffix')
    return p.parse_args()


def convert_split(src_root, dst_root, split, keys):
    src_dir = os.path.join(src_root, split)
    dst_dir = os.path.join(dst_root, split)
    if not os.path.isdir(src_dir):
        print(f'[skip] split dir not found: {src_dir}')
        return 0, 0
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, '*.npz')))
    print(f'[{split}] {len(files)} npz files -> {dst_dir}')
    done, skipped = 0, 0
    t0 = time.time()
    for i, f in enumerate(files):
        stem = os.path.splitext(os.path.basename(f))[0]
        npy_path = os.path.join(dst_dir, f'{stem}.npy')
        meta_path = os.path.join(dst_dir, f'{stem}.meta.npz')
        if os.path.exists(npy_path) and os.path.exists(meta_path):
            skipped += 1
            continue
        with np.load(f) as npz:
            missing = [k for k in keys if k not in npz.files]
            if missing:
                raise KeyError(f'{f}: missing keys={missing}; available={list(npz.files)}')
            arrays = [np.asarray(npz[k], dtype=np.float16) for k in keys]
            feat = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
            meta = {k: np.asarray(npz[k]) for k in npz.files if k not in keys}
        # atomic-ish: write to .tmp then rename. np.save/np.savez append their own
        # extension to the given path, so name the tmp files accordingly.
        tmp_npy = npy_path + '.tmp.npy'
        tmp_meta = meta_path + '.tmp.npz'
        np.save(tmp_npy[:-4], np.ascontiguousarray(feat))   # np.save adds .npy
        os.replace(tmp_npy, npy_path)
        np.savez(tmp_meta[:-4], **meta)                     # np.savez adds .npz
        os.replace(tmp_meta, meta_path)
        done += 1
        if (i + 1) % 100 == 0:
            dt = time.time() - t0
            print(f'  [{split}] {i + 1}/{len(files)} (done={done} skip={skipped}) {dt:.0f}s', flush=True)
    dt = time.time() - t0
    print(f'[{split}] finished: done={done} skipped={skipped} in {dt:.0f}s')
    return done, skipped


def dir_size_gb(path):
    total = 0
    for root, _, fnames in os.walk(path):
        for fn in fnames:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total / 1e9


def main():
    args = parse_args()
    keys = [k.strip() for k in args.keys.split(',') if k.strip()]
    src_root = args.src_root.rstrip('/')
    dst_root = src_root + args.dst_suffix
    splits = ['train', 'test'] if args.split == 'both' else [args.split]
    print(f'src_root={src_root}\ndst_root={dst_root}\nkeys={keys}\nsplits={splits}')
    total_done, total_skip = 0, 0
    t0 = time.time()
    for sp in splits:
        d, s = convert_split(src_root, dst_root, sp, keys)
        total_done += d
        total_skip += s
    print(f'\n=== ALL DONE: converted={total_done} skipped={total_skip} '
          f'in {time.time() - t0:.0f}s ===')
    if os.path.isdir(dst_root):
        print(f'dst size: {dir_size_gb(dst_root):.1f} GB ({dst_root})')


if __name__ == '__main__':
    main()
