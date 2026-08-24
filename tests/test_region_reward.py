import unittest

import torch

from lib.train.rewards.region_reward import (
    compute_delta_nsd_reward,
    compute_nsd_score,
)


def rectangle(x0: float, y0: float, x1: float, y1: float) -> torch.Tensor:
    return torch.tensor(
        [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]],
        dtype=torch.float32,
    )


class NormalizedSurfaceDiceTest(unittest.TestCase):
    def score(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        return float(
            compute_nsd_score(
                prediction,
                target,
                H=64,
                W=64,
                delta_px=2.0,
            ).item()
        )

    def test_identical_boundaries_score_one(self):
        target = rectangle(16, 16, 40, 40)
        self.assertAlmostEqual(self.score(target, target), 1.0, places=7)

    def test_nsd_is_symmetric(self):
        first = rectangle(12, 14, 36, 42)
        second = rectangle(17, 11, 41, 39)
        self.assertAlmostEqual(
            self.score(first, second),
            self.score(second, first),
            places=7,
        )

    def test_near_boundary_beats_deterministic_baseline(self):
        target = rectangle(16, 16, 40, 40)
        deterministic = rectangle(23, 16, 47, 40)
        sampled = rectangle(17, 16, 41, 40)
        delta_nsd = self.score(sampled, target) - self.score(deterministic, target)
        self.assertGreater(delta_nsd, 0.0)

    def test_distant_boundaries_score_zero(self):
        target = rectangle(4, 4, 16, 16)
        prediction = rectangle(40, 40, 56, 56)
        self.assertAlmostEqual(self.score(prediction, target), 0.0, places=7)

    def test_delta_nsd_reward_uses_absolute_sampled_burr(self):
        reward = compute_delta_nsd_reward(
            sampled_nsd=torch.tensor([0.80, 0.70]),
            deterministic_nsd=torch.tensor([0.75, 0.72]),
            sampled_burr=torch.tensor([0.50, 0.25]),
            burr_weight=0.06,
        )
        expected = torch.tensor([0.05 - 0.06 * 0.50, -0.02 - 0.06 * 0.25])
        self.assertTrue(torch.allclose(reward, expected, atol=1e-7, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
