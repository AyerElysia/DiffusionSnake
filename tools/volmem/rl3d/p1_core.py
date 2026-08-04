"""Core policy, action, and reward primitives for VolMem RL3D P1."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from torch import nn


def fourier_basis(point_count: int = 128, max_frequency: int = 5,
                  device=None, dtype=torch.float32) -> torch.Tensor:
    theta = torch.arange(point_count, device=device, dtype=dtype)
    theta = theta * (2.0 * math.pi / float(point_count))
    columns = [torch.ones_like(theta)]
    for frequency in range(1, int(max_frequency) + 1):
        columns.extend([
            torch.cos(float(frequency) * theta),
            torch.sin(float(frequency) * theta),
        ])
    return torch.stack(columns, dim=1)


def polygon_normals(poly: torch.Tensor) -> torch.Tensor:
    if poly.dim() != 3 or poly.size(-1) != 2:
        raise ValueError("poly must be [B,P,2]")
    tangent = torch.roll(poly, -1, dims=1) - torch.roll(poly, 1, dims=1)
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    outward = poly - poly.mean(dim=1, keepdim=True)
    flip = (normal * outward).sum(dim=-1, keepdim=True) < 0
    return torch.where(flip, -normal, normal)


def apply_fourier_action(
    poly: torch.Tensor,
    normals: torch.Tensor,
    coefficients: torch.Tensor,
    basis: torch.Tensor,
    max_displacement: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply an exactly low-frequency normal field with smooth radial limiting.

    The same scalar rescales the entire reconstructed field, so limiting never
    creates frequencies outside the k<=5 subspace.
    """
    if coefficients.size(-1) != basis.size(1):
        raise ValueError("coefficient count does not match Fourier basis")
    field = torch.einsum("...k,pk->...p", coefficients, basis.to(coefficients))
    magnitude = field.abs().amax(dim=-1, keepdim=True)
    limit = float(max_displacement)
    radial_scale = limit * torch.tanh(magnitude / limit) / magnitude.clamp_min(1e-8)
    radial_scale = torch.where(magnitude < 1e-8, torch.ones_like(radial_scale), radial_scale)
    field = field * radial_scale

    target_dim = field.dim() + 1
    action_poly = poly
    action_normals = normals
    while action_poly.dim() < target_dim:
        action_poly = action_poly.unsqueeze(0)
        action_normals = action_normals.unsqueeze(0)
    refined = action_poly + field.unsqueeze(-1) * action_normals
    return refined, field


class FourierPolicy(nn.Module):
    """Phase-aware small policy head over frozen per-point Flow features."""

    def __init__(self, feature_dim: int, point_count: int = 128,
                 max_frequency: int = 5, hidden_dim: int = 64) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.point_count = int(point_count)
        self.max_frequency = int(max_frequency)
        basis = fourier_basis(point_count, max_frequency)
        self.register_buffer("basis", basis, persistent=True)
        self.register_buffer("feature_mean", torch.zeros(feature_dim), persistent=True)
        self.register_buffer("feature_std", torch.ones(feature_dim), persistent=True)

        point_hidden = 16
        self.point_encoder = nn.Sequential(
            nn.Linear(feature_dim + 6, 32),
            nn.SiLU(),
            nn.Linear(32, point_hidden),
            nn.SiLU(),
        )
        self.class_embedding = nn.Embedding(26, 8)
        global_dim = point_hidden * basis.size(1) + 4 + 8
        self.trunk = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, basis.size(1))
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

    @property
    def action_dim(self) -> int:
        return int(self.basis.size(1))

    @torch.no_grad()
    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        mean = mean.reshape(-1).to(self.feature_mean)
        std = std.reshape(-1).to(self.feature_std)
        if mean.numel() != self.feature_dim or std.numel() != self.feature_dim:
            raise ValueError("feature statistics have the wrong channel count")
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std.clamp_min(1e-4))

    def forward(self, point_features: torch.Tensor, poly: torch.Tensor,
                normals: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if point_features.shape[:2] != poly.shape[:2]:
            raise ValueError("point feature and polygon axes must align")
        if point_features.size(-1) != self.feature_dim:
            raise ValueError("unexpected point feature channel count")
        if poly.size(1) != self.point_count or normals.shape != poly.shape:
            raise ValueError("policy expects canonical [B,128,2] contours")

        feat = (point_features - self.feature_mean) / self.feature_std
        centered = poly - poly.mean(dim=1, keepdim=True)
        radius = centered.norm(dim=-1, keepdim=True)
        scale = radius.mean(dim=1, keepdim=True).clamp_min(1e-3)
        centered_unit = centered / scale
        radius_unit = radius / scale
        curvature = (
            torch.roll(poly, 1, dims=1) - 2.0 * poly + torch.roll(poly, -1, dims=1)
        ).norm(dim=-1, keepdim=True) / scale
        geometry = torch.cat([centered_unit, normals, radius_unit, curvature], dim=-1)
        point_state = self.point_encoder(torch.cat([feat, geometry], dim=-1))
        moments = torch.einsum(
            "bpd,pk->bdk", point_state, self.basis.to(point_state)
        ) / float(self.point_count)

        x, y = poly[..., 0], poly[..., 1]
        signed_area = 0.5 * torch.sum(x * torch.roll(y, -1, 1) - y * torch.roll(x, -1, 1), dim=1)
        perimeter = (torch.roll(poly, -1, dims=1) - poly).norm(dim=-1).sum(dim=1)
        scale_flat = scale[:, 0, 0]
        global_geometry = torch.stack([
            torch.log1p(scale_flat),
            signed_area.abs() / scale_flat.square().clamp_min(1e-6),
            perimeter / scale_flat.clamp_min(1e-6),
            curvature.mean(dim=(1, 2)),
        ], dim=1)
        # Cached vertebra labels are 1-based (1..26), while Embedding is 0-based.
        class_state = self.class_embedding((labels.long() - 1).clamp(0, 25))
        state = torch.cat([moments.flatten(1), global_geometry, class_state], dim=1)
        return self.mean_head(self.trunk(state))


def gaussian_log_prob(action: torch.Tensor, mean: torch.Tensor,
                      sigma: float) -> torch.Tensor:
    variance = float(sigma) ** 2
    return -0.5 * (
        ((action - mean) ** 2) / variance
        + math.log(2.0 * math.pi * variance)
    ).sum(dim=-1)


def _poly_to_mask(poly: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.rint(poly).astype(np.int32)
    if points.ndim == 2 and points.shape[0] >= 3:
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 1)
    return mask


def _boundary(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask, dtype=np.uint8)
    if contours:
        cv2.drawContours(result, contours, -1, 1, thickness=1)
    return result.astype(bool)


def _local_polygons(pred: np.ndarray, gt: np.ndarray, orig_hw: np.ndarray,
                    margin: int = 8) -> Tuple[np.ndarray, np.ndarray, int, int]:
    height, width = int(orig_hw[0]), int(orig_hw[1])
    both = np.concatenate([pred, gt], axis=0)
    x0 = max(int(math.floor(float(np.nanmin(both[:, 0])))) - margin, 0)
    y0 = max(int(math.floor(float(np.nanmin(both[:, 1])))) - margin, 0)
    x1 = min(int(math.ceil(float(np.nanmax(both[:, 0])))) + margin + 1, width)
    y1 = min(int(math.ceil(float(np.nanmax(both[:, 1])))) + margin + 1, height)
    x1 = max(x1, x0 + 4)
    y1 = max(y1, y0 + 4)
    offset = np.asarray([x0, y0], dtype=np.float32)
    return pred - offset, gt - offset, y1 - y0, x1 - x0


def contour_quality(pred: np.ndarray, gt: np.ndarray, orig_hw: np.ndarray,
                    nsd_delta: float = 2.0, burr_weight: float = 0.06,
                    burr_margin: float = 0.5, burr_max: float = 1.5) -> Dict[str, float]:
    pred_local, gt_local, height, width = _local_polygons(pred, gt, orig_hw)
    pred_boundary = _boundary(_poly_to_mask(pred_local, height, width))
    gt_boundary = _boundary(_poly_to_mask(gt_local, height, width))
    if not pred_boundary.any() or not gt_boundary.any():
        nsd = 0.0
    else:
        gt_dt = cv2.distanceTransform((~gt_boundary).astype(np.uint8), cv2.DIST_L2, 3)
        pred_dt = cv2.distanceTransform((~pred_boundary).astype(np.uint8), cv2.DIST_L2, 3)
        pred_hit = float(np.mean(gt_dt[pred_boundary] <= float(nsd_delta)))
        gt_hit = float(np.mean(pred_dt[gt_boundary] <= float(nsd_delta)))
        nsd = 0.5 * (pred_hit + gt_hit)

    lap_pred = np.linalg.norm(
        np.roll(pred, 1, axis=0) - 2.0 * pred + np.roll(pred, -1, axis=0), axis=1)
    lap_gt = np.linalg.norm(
        np.roll(gt, 1, axis=0) - 2.0 * gt + np.roll(gt, -1, axis=0), axis=1)
    excess = np.maximum(lap_pred - lap_gt - float(burr_margin), 0.0)
    burr_raw = float(np.percentile(excess, 95.0))
    burr = float(np.clip(burr_raw / max(float(burr_max), 1e-6), 0.0, 2.0))

    distances = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=-1)
    mean_distance = float(np.min(distances, axis=1).mean())
    return {
        "quality": float(nsd - float(burr_weight) * burr),
        "nsd": float(nsd),
        "burr": burr,
        "burr_raw": burr_raw,
        "mean_distance": mean_distance,
    }


def score_contours(pred: np.ndarray, gt: np.ndarray,
                   orig_hw: np.ndarray) -> Dict[str, np.ndarray]:
    if pred.ndim != 3 or gt.shape != pred.shape or orig_hw.shape != (pred.shape[0], 2):
        raise ValueError("score_contours expects [N,P,2], matching GT, and [N,2] hw")
    rows = [contour_quality(p, g, hw) for p, g, hw in zip(pred, gt, orig_hw)]
    return {
        key: np.asarray([row[key] for row in rows], dtype=np.float32)
        for key in rows[0]
    } if rows else {
        key: np.zeros((0,), dtype=np.float32)
        for key in ("quality", "nsd", "burr", "burr_raw", "mean_distance")
    }


def delta_nsd_burr_reward(sample_scores: Dict[str, np.ndarray],
                          baseline_nsd: np.ndarray,
                          burr_weight: float = 0.06) -> np.ndarray:
    """The exact reward used by the strongest 2D delta-NSD experiment."""
    return (
        np.asarray(sample_scores["nsd"], dtype=np.float32)
        - np.asarray(baseline_nsd, dtype=np.float32)
        - float(burr_weight) * np.asarray(sample_scores["burr"], dtype=np.float32)
    )


def oracle_coefficients(poly: torch.Tensor, gt_target: torch.Tensor,
                        normals: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    residual = ((gt_target - poly) * normals).sum(dim=-1)
    design = basis.to(device=poly.device, dtype=poly.dtype)
    solution = torch.linalg.lstsq(
        design.unsqueeze(0).expand(poly.size(0), -1, -1),
        residual.unsqueeze(-1),
    ).solution.squeeze(-1)
    return solution
