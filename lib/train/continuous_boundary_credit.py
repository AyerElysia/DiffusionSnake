from __future__ import annotations

import torch


def point_to_closed_polyline_distance(
    points: torch.Tensor,
    polylines: torch.Tensor,
) -> torch.Tensor:
    """Return Euclidean distance from points to closed polyline segments."""
    if points.shape[-1] != 2 or polylines.shape[-1] != 2:
        raise ValueError("points and polylines must end with xy coordinates")
    if polylines.shape[-2] < 2:
        raise ValueError("a closed polyline requires at least two vertices")

    segment_start = polylines
    segment_end = torch.roll(polylines, shifts=-1, dims=-2)
    segment = segment_end - segment_start
    point_offset = points.unsqueeze(-2) - segment_start.unsqueeze(-3)
    segment_sq_norm = segment.square().sum(dim=-1).unsqueeze(-2)
    eps = torch.finfo(points.dtype).eps
    projection = (point_offset * segment.unsqueeze(-3)).sum(dim=-1)
    projection = (projection / segment_sq_norm.clamp_min(eps)).clamp(0.0, 1.0)
    closest = segment_start.unsqueeze(-3) + projection.unsqueeze(-1) * segment.unsqueeze(-3)
    distances = (points.unsqueeze(-2) - closest).square().sum(dim=-1).sqrt()
    return distances.min(dim=-1).values


def continuous_boundary_quality_delta(
    sampled_points: torch.Tensor,
    pure_points: torch.Tensor,
    gt_polylines: torch.Tensor,
    coord_scale: float,
    dist_max_px: float,
    clamp: bool = True,
) -> torch.Tensor:
    """Score sampled endpoints against pure-FM endpoints on a continuous boundary."""
    if dist_max_px <= 0:
        raise ValueError("dist_max_px must be positive")
    sampled_dist_px = point_to_closed_polyline_distance(
        sampled_points, gt_polylines
    ) * float(coord_scale)
    pure_dist_px = point_to_closed_polyline_distance(
        pure_points, gt_polylines
    ) * float(coord_scale)
    quality = (pure_dist_px - sampled_dist_px) / float(dist_max_px)
    return quality.clamp(-1.0, 1.0) if clamp else quality
