#!/usr/bin/env python3

import unittest

import numpy as np
import torch

from p1_core import (
    FourierPolicy,
    apply_fourier_action,
    contour_quality,
    delta_nsd_burr_reward,
    fourier_basis,
    oracle_coefficients,
    polygon_normals,
)


class P1CoreTests(unittest.TestCase):
    def test_zero_action_is_identity_and_bound_holds(self):
        theta = torch.arange(128, dtype=torch.float32) * (2.0 * torch.pi / 128.0)
        poly = torch.stack([32.0 + 15.0 * torch.cos(theta),
                            32.0 + 12.0 * torch.sin(theta)], dim=-1).unsqueeze(0)
        normals = polygon_normals(poly)
        basis = fourier_basis()
        zero = torch.zeros(1, 11)
        unchanged, field = apply_fourier_action(poly, normals, zero, basis)
        self.assertTrue(torch.equal(unchanged, poly))
        self.assertTrue(torch.equal(field, torch.zeros_like(field)))

        huge = torch.full((4, 1, 11), 100.0)
        _, bounded = apply_fourier_action(poly, normals, huge, basis)
        self.assertLessEqual(float(bounded.abs().max()), 3.0 + 1e-6)

    def test_global_limiter_preserves_fourier_subspace(self):
        basis = fourier_basis()
        poly = torch.zeros(2, 128, 2)
        poly[..., 0] = torch.arange(128).float()
        normals = torch.zeros_like(poly)
        normals[..., 1] = 1.0
        coefficients = torch.randn(2, 11) * 10.0
        _, field = apply_fourier_action(poly, normals, coefficients, basis)
        reconstructed = basis @ torch.linalg.lstsq(basis, field.T).solution
        self.assertLess(float((reconstructed.T - field).abs().max()), 1e-4)

    def test_policy_starts_at_exact_zero(self):
        policy = FourierPolicy(feature_dim=64)
        features = torch.randn(3, 128, 64)
        theta = torch.arange(128).float() * (2.0 * torch.pi / 128.0)
        base = torch.stack([40.0 + 10.0 * torch.cos(theta),
                            40.0 + 8.0 * torch.sin(theta)], dim=-1)
        poly = base.unsqueeze(0).repeat(3, 1, 1)
        normals = polygon_normals(poly)
        output = policy(features, poly, normals, torch.tensor([1, 2, 3]))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_one_based_vertebra_labels_map_to_distinct_embeddings(self):
        policy = FourierPolicy(feature_dim=4)
        captured = []
        hook = policy.class_embedding.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone()))
        features = torch.zeros(3, 128, 4)
        theta = torch.arange(128).float() * (2.0 * torch.pi / 128.0)
        base = torch.stack([32.0 + 8.0 * torch.cos(theta),
                            32.0 + 8.0 * torch.sin(theta)], dim=-1)
        poly = base.unsqueeze(0).repeat(3, 1, 1)
        policy(features, poly, polygon_normals(poly), torch.tensor([1, 25, 26]))
        hook.remove()
        self.assertTrue(torch.equal(captured[0], torch.tensor([0, 24, 25])))

    def test_exact_match_scores_better_than_shifted_polygon(self):
        gt = np.asarray([[20, 20], [40, 20], [40, 40], [20, 40]], dtype=np.float32)
        shifted = gt + np.asarray([-5.0, 0.0], dtype=np.float32)
        hw = np.asarray([64, 64], dtype=np.int32)
        good = contour_quality(gt, gt, hw)
        bad = contour_quality(shifted, gt, hw)
        self.assertGreater(good["nsd"], bad["nsd"])
        self.assertLess(good["mean_distance"], bad["mean_distance"])

    def test_delta_nsd_reward_uses_absolute_sample_burr(self):
        sample = {
            "nsd": np.asarray([0.80, 0.70], dtype=np.float32),
            "burr": np.asarray([0.50, 0.25], dtype=np.float32),
        }
        baseline_nsd = np.asarray([0.75, 0.72], dtype=np.float32)
        reward = delta_nsd_burr_reward(sample, baseline_nsd)
        expected = np.asarray([0.05 - 0.06 * 0.50,
                               -0.02 - 0.06 * 0.25], dtype=np.float32)
        np.testing.assert_allclose(reward, expected, rtol=0.0, atol=1e-7)

    def test_oracle_normal_action_reduces_radial_error(self):
        theta = torch.arange(128).float() * (2.0 * torch.pi / 128.0)
        poly = torch.stack([32.0 + 20.0 * torch.cos(theta),
                            32.0 + 20.0 * torch.sin(theta)], dim=-1).unsqueeze(0)
        target = torch.stack([32.0 + 18.0 * torch.cos(theta),
                              32.0 + 18.0 * torch.sin(theta)], dim=-1).unsqueeze(0)
        normals = polygon_normals(poly)
        basis = fourier_basis()
        coefficients = oracle_coefficients(poly, target, normals, basis)
        refined, _ = apply_fourier_action(poly, normals, coefficients, basis)
        before = (poly - target).norm(dim=-1).mean()
        after = (refined - target).norm(dim=-1).mean()
        self.assertLess(float(after), float(before) * 0.15)


if __name__ == "__main__":
    unittest.main()
