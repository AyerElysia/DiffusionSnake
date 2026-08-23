"""Tensor-shape helpers shared by experimental pure-2D context modules."""

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def project_and_pad_feature_maps(
    raw_maps: Sequence[torch.Tensor],
    projector: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project variable-size maps and pad them with an attention-valid mask."""
    if not isinstance(raw_maps, (list, tuple)) or len(raw_maps) == 0:
        raise ValueError("raw_maps must be a non-empty list/tuple")
    projected = []
    max_h = 0
    max_w = 0
    for raw in raw_maps:
        if not torch.is_tensor(raw):
            raw = torch.as_tensor(raw)
        raw = raw.to(device=device, dtype=dtype, non_blocking=True)
        if raw.dim() == 3:
            raw = raw.unsqueeze(0)
        if raw.dim() != 4 or raw.size(0) != 1:
            raise ValueError(f"each cached feature must be [C,H,W], got {tuple(raw.shape)}")
        feat = projector(raw)
        projected.append(feat)
        max_h = max(max_h, int(feat.size(-2)))
        max_w = max(max_w, int(feat.size(-1)))

    padded = []
    valid = []
    for feat in projected:
        h, w = int(feat.size(-2)), int(feat.size(-1))
        padded.append(F.pad(feat, (0, max_w - w, 0, max_h - h)))
        mask = torch.zeros((1, max_h, max_w), device=device, dtype=torch.bool)
        mask[:, :h, :w] = True
        valid.append(mask)
    return torch.cat(padded, dim=0), torch.cat(valid, dim=0)


def contour_geometry_features(
    points: torch.Tensor,
    py_ind: torch.Tensor,
    inp_out_hw: torch.Tensor,
) -> torch.Tensor:
    """Encode target center, size, aspect and area in output coordinates."""
    if points.dim() != 3 or points.size(-1) != 2:
        raise ValueError(f"expected contour points [N,P,2], got {tuple(points.shape)}")
    py_ind = py_ind.to(device=points.device, dtype=torch.long)
    inp_out_hw = inp_out_hw.to(device=points.device, dtype=points.dtype)
    if inp_out_hw.dim() != 2 or inp_out_hw.size(1) < 4:
        raise ValueError("inp_out_hw must be [B,4+] with input/output H,W")

    out_h = inp_out_hw[py_ind, 2].clamp_min(2.0)
    out_w = inp_out_hw[py_ind, 3].clamp_min(2.0)
    p_min = points.amin(dim=1)
    p_max = points.amax(dim=1)
    center = 0.5 * (p_min + p_max)
    span = (p_max - p_min).clamp_min(1.0)
    cx = 2.0 * center[:, 0] / (out_w - 1.0) - 1.0
    cy = 2.0 * center[:, 1] / (out_h - 1.0) - 1.0
    frac_w = (span[:, 0] / out_w).clamp_min(1e-6)
    frac_h = (span[:, 1] / out_h).clamp_min(1e-6)
    log_w = frac_w.log()
    log_h = frac_h.log()
    log_aspect = (frac_w / frac_h).log()
    log_area = (frac_w * frac_h).log()
    return torch.stack([cx, cy, log_w, log_h, log_aspect, log_area], dim=-1)

