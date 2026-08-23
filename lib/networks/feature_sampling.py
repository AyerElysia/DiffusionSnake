"""Small tensor helpers used by the Flow context encoder."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def project_and_pad_feature_maps(
    raw_maps: Sequence[torch.Tensor],
    projector: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project variable-size maps and return padding plus a valid-pixel mask."""
    if not isinstance(raw_maps, (list, tuple)) or not raw_maps:
        raise ValueError("raw_maps must be a non-empty list or tuple")

    projected: list[torch.Tensor] = []
    max_h = 0
    max_w = 0
    for raw in raw_maps:
        raw = torch.as_tensor(raw, device=device, dtype=dtype)
        if raw.ndim == 3:
            raw = raw.unsqueeze(0)
        if raw.ndim != 4 or raw.shape[0] != 1:
            raise ValueError(f"each cached feature must be [C,H,W], got {tuple(raw.shape)}")
        feature = projector(raw)
        projected.append(feature)
        max_h = max(max_h, int(feature.shape[-2]))
        max_w = max(max_w, int(feature.shape[-1]))

    padded: list[torch.Tensor] = []
    valid: list[torch.Tensor] = []
    for feature in projected:
        height, width = map(int, feature.shape[-2:])
        padded.append(F.pad(feature, (0, max_w - width, 0, max_h - height)))
        mask = torch.zeros((1, max_h, max_w), device=device, dtype=torch.bool)
        mask[:, :height, :width] = True
        valid.append(mask)
    return torch.cat(padded, dim=0), torch.cat(valid, dim=0)


def contour_geometry_features(
    points: torch.Tensor,
    py_ind: torch.Tensor,
    inp_out_hw: torch.Tensor,
) -> torch.Tensor:
    """Encode contour center, size, aspect and area in output coordinates."""
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f"expected contour points [N,P,2], got {tuple(points.shape)}")
    py_ind = py_ind.to(device=points.device, dtype=torch.long)
    inp_out_hw = inp_out_hw.to(device=points.device, dtype=points.dtype)
    if inp_out_hw.ndim != 2 or inp_out_hw.shape[1] < 4:
        raise ValueError("inp_out_hw must be [B,4+] with input/output H,W")

    out_h = inp_out_hw[py_ind, 2].clamp_min(2.0)
    out_w = inp_out_hw[py_ind, 3].clamp_min(2.0)
    point_min = points.amin(dim=1)
    point_max = points.amax(dim=1)
    center = 0.5 * (point_min + point_max)
    span = (point_max - point_min).clamp_min(1.0)
    center_x = 2.0 * center[:, 0] / (out_w - 1.0) - 1.0
    center_y = 2.0 * center[:, 1] / (out_h - 1.0) - 1.0
    width = (span[:, 0] / out_w).clamp_min(1e-6)
    height = (span[:, 1] / out_h).clamp_min(1e-6)
    return torch.stack(
        (
            center_x,
            center_y,
            width.log(),
            height.log(),
            (width / height).log(),
            (width * height).log(),
        ),
        dim=-1,
    )
