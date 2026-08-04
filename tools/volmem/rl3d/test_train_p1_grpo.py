#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import numpy as np

from train_p1_grpo import load_caches, split_by_last_cache


def write_cache(path: Path, count: int, label: int) -> None:
    points = np.zeros((count, 128, 2), dtype=np.float32)
    np.savez_compressed(
        path,
        poly=points,
        gt_target=points,
        gt_poly=points,
        gt_dist=np.zeros((count, 128), dtype=np.float32),
        normal=points,
        point_feat=np.zeros((count, 128, 4), dtype=np.float32),
        slice_idx=np.arange(count, dtype=np.int32),
        label=np.full(count, label, dtype=np.int32),
        orig_hw=np.full((count, 2), 64, dtype=np.int32),
        n_gt_boundary=np.ones(count, dtype=np.int32),
    )


class MultiCacheTests(unittest.TestCase):
    def test_last_cache_is_a_disjoint_volume_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.npz"
            second = Path(tmp) / "second.npz"
            write_cache(first, 24, 8)
            write_cache(second, 27, 16)
            arrays, counts = load_caches([str(first), str(second)])
            train, holdout = split_by_last_cache(arrays["cache_index"], 2)
            self.assertEqual(counts, [24, 27])
            self.assertEqual(train.size, 24)
            self.assertEqual(holdout.size, 27)
            self.assertTrue(np.all(arrays["cache_index"][train] == 0))
            self.assertTrue(np.all(arrays["cache_index"][holdout] == 1))
            self.assertTrue(np.all(arrays["label"][holdout] == 16))


if __name__ == "__main__":
    unittest.main()
