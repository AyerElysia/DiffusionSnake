import math
import unittest

import torch

from lib.rl.fourier import (
    fourier_action_logprob,
    fourier_mean_kl,
    low_frequency_delta,
    stage_progress,
    standard_normal_logprob,
)


class FourierPolicyTest(unittest.TestCase):
    def setUp(self):
        angles = (
            torch.arange(128, dtype=torch.float64)
            * (2.0 * torch.pi / 128.0)
        )
        self.contour = torch.stack(
            (32.0 + 12.0 * angles.cos(), 32.0 + 12.0 * angles.sin()),
            dim=-1,
        ).unsqueeze(0)

    def test_five_stage_progress_is_20_percent_per_action(self):
        progress = stage_progress([0.2, 0.25, 0.3333, 0.5, 1.0])
        expected = [0.0, 0.2, 0.4, 0.59998, 0.79999]
        self.assertEqual(len(progress), 5)
        for actual, target in zip(progress, expected):
            self.assertAlmostEqual(actual, target, places=10)

    def test_projected_logprob_recovers_sampled_coefficients(self):
        torch.manual_seed(20260824)
        coefficients = torch.randn(1, 8, dtype=torch.float64)
        mean = torch.zeros_like(self.contour)
        action = mean + low_frequency_delta(
            self.contour, coefficients, sigma=0.8
        )
        projected = fourier_action_logprob(
            action, mean, self.contour, sigma=0.8, n_modes=8
        )
        direct = standard_normal_logprob(coefficients)
        self.assertTrue(torch.allclose(projected, direct, atol=1e-10, rtol=0.0))

    def test_documented_point_rms_matches_orthonormal_basis(self):
        coefficients = torch.ones(1, 8, dtype=torch.float64)
        delta = low_frequency_delta(
            self.contour, coefficients, sigma=0.8
        )
        observed = delta.square().sum(dim=-1).mean().sqrt().item()
        expected = 0.8 * math.sqrt(8.0 / 128.0)
        self.assertAlmostEqual(observed, expected, places=10)

    def test_explicit_kl_has_finite_flow_mean_gradient(self):
        current = torch.zeros_like(self.contour, requires_grad=True)
        reference = torch.full_like(self.contour, 0.01)
        loss = fourier_mean_kl(
            current, reference, self.contour, sigma=0.8, n_modes=8
        ).mean()
        loss.backward()
        self.assertIsNotNone(current.grad)
        self.assertTrue(torch.isfinite(current.grad).all())


if __name__ == "__main__":
    unittest.main()
