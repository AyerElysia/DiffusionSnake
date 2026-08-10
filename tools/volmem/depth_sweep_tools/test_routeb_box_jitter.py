#!/usr/bin/env python3
"""Contracts for Route-B detector-box robustness augmentation."""

import unittest
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.utils.snake import snake_gcn_utils, snake_config


def make_config(probabilities, min_iou=0.20):
    namespace = SimpleNamespace(
        routeb_box_jitter_enabled=True,
        routeb_box_jitter_probabilities=probabilities,
        routeb_box_jitter_shift_fractions=[0.0, 0.05, 0.10, 0.15],
        routeb_box_jitter_log_scale_fractions=[0.0, 0.10, 0.20, 0.30],
        routeb_box_jitter_edge_fractions=[0.0, 0.03, 0.08, 0.15],
        routeb_box_jitter_min_iou=min_iou,
    )
    return snake_gcn_utils.resolve_routeb_box_jitter_config(namespace)


class RouteBBoxJitterTest(unittest.TestCase):
    def setUp(self):
        self.box = torch.tensor(
            [
                [20.0, 30.0, 60.0, 90.0],
                [1.0, 2.0, 18.0, 24.0],
                [90.0, 80.0, 126.0, 126.0],
            ],
            dtype=torch.float32,
        )

    def test_clean_severity_is_exact(self):
        config = make_config([1.0, 0.0, 0.0, 0.0])
        torch.manual_seed(11)
        expected_next_random = torch.rand(8)
        torch.manual_seed(11)
        actual, stats = snake_gcn_utils.jitter_routeb_boxes_xyxy(
            self.box, config, image_hw=(128, 128)
        )
        actual_next_random = torch.rand(8)
        self.assertTrue(torch.equal(actual, self.box))
        self.assertTrue(torch.equal(actual_next_random, expected_next_random))
        self.assertEqual(float(stats['routeb_box_jitter_count']), 0.0)
        self.assertEqual(float(stats['routeb_box_jitter_mean_iou']), 1.0)

    def test_forced_jitter_is_valid_and_respects_iou_floor(self):
        config = make_config([0.0, 0.0, 0.0, 1.0], min_iou=0.35)
        torch.manual_seed(20260810)
        actual, stats = snake_gcn_utils.jitter_routeb_boxes_xyxy(
            self.box, config, image_hw=(128, 128)
        )
        self.assertFalse(torch.equal(actual, self.box))
        self.assertTrue(bool(torch.all(actual[:, :2] >= 0.0)))
        self.assertTrue(bool(torch.all(actual[:, 2:] <= 127.0)))
        self.assertTrue(bool(torch.all(actual[:, 2:] > actual[:, :2])))
        iou = snake_gcn_utils._aligned_box_iou(actual, self.box)
        self.assertGreaterEqual(float(iou.min()), 0.35 - 1e-6)
        self.assertGreaterEqual(
            float(stats['routeb_box_jitter_min_iou']), 0.35 - 1e-6
        )

    def test_mixture_probabilities_are_applied_per_instance(self):
        config = make_config([0.35, 0.40, 0.20, 0.05])
        boxes = self.box[:1].repeat(10000, 1)
        torch.manual_seed(71)
        _, stats = snake_gcn_utils.jitter_routeb_boxes_xyxy(
            boxes, config, image_hw=(128, 128)
        )
        fractions = stats['routeb_box_jitter_severity_counts'] / boxes.size(0)
        expected = torch.tensor([0.35, 0.40, 0.20, 0.05])
        self.assertTrue(torch.all(torch.abs(fractions.cpu() - expected) < 0.02))

    def test_clean_members_in_a_mixed_batch_remain_exact(self):
        config = make_config([0.50, 0.50, 0.0, 0.0])
        boxes = self.box[:1].repeat(256, 1)
        torch.manual_seed(810)
        actual, stats = snake_gcn_utils.jitter_routeb_boxes_xyxy(
            boxes, config, image_hw=(128, 128)
        )
        exact_members = torch.all(actual == boxes, dim=1).sum()
        self.assertEqual(
            int(exact_members), int(stats['routeb_box_jitter_clean_count'])
        )

    def test_40_and_128_point_inits_share_one_jittered_box(self):
        gt_poly = torch.tensor(
            [
                [
                    [20.0, 30.0],
                    [60.0, 30.0],
                    [60.0, 90.0],
                    [20.0, 90.0],
                ]
            ],
            dtype=torch.float32,
        )
        config = make_config([0.0, 0.0, 1.0, 0.0])
        torch.manual_seed(1234)
        replaced, stats = (
            snake_gcn_utils.replace_training_init_with_gt_box_octagon(
                {'i_gt_py': gt_poly},
                jitter_config=config,
                image_hw=(128, 128),
                return_jitter_stats=True,
            )
        )
        full_box = torch.cat(
            [
                replaced['i_it_py'].min(dim=1)[0],
                replaced['i_it_py'].max(dim=1)[0],
            ],
            dim=1,
        )
        coarse_box = torch.cat(
            [
                replaced['i_it_4py'].min(dim=1)[0],
                replaced['i_it_4py'].max(dim=1)[0],
            ],
            dim=1,
        )
        self.assertEqual(replaced['i_it_py'].shape[1], snake_config.poly_num)
        self.assertEqual(replaced['i_it_4py'].shape[1], snake_config.init_poly_num)
        self.assertTrue(torch.allclose(full_box, coarse_box, atol=1e-5, rtol=0.0))
        self.assertEqual(float(stats['routeb_box_jitter_count']), 1.0)

    def test_invalid_config_fails_closed(self):
        with self.assertRaises(ValueError):
            make_config([0.5, -0.5, 1.0, 0.0])
        namespace = SimpleNamespace(
            routeb_box_jitter_enabled=True,
            routeb_box_jitter_probabilities=[0.5, 0.5],
            routeb_box_jitter_shift_fractions=[0.01, 0.05],
            routeb_box_jitter_log_scale_fractions=[0.0, 0.10],
            routeb_box_jitter_edge_fractions=[0.0, 0.03],
            routeb_box_jitter_min_iou=0.20,
        )
        with self.assertRaises(ValueError):
            snake_gcn_utils.resolve_routeb_box_jitter_config(namespace)


if __name__ == '__main__':
    unittest.main()
