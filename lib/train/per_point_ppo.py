"""Small PPO helpers for masked per-point flow-matching actions."""

from __future__ import annotations

import math

import torch


def per_point_squashed_gaussian_logprob(
    raw_sample: torch.Tensor,
    mu: torch.Tensor,
    logstd: torch.Tensor,
    max_scale: float,
) -> torch.Tensor:
    """Return one tanh-squashed Gaussian log-probability per contour point."""
    var = torch.exp(2.0 * logstd).clamp_min(1e-12)
    log_prob = -0.5 * (
        (raw_sample - mu) ** 2 / var
        + 2.0 * logstd
        + math.log(2.0 * math.pi)
    )
    tanh_raw = torch.tanh(raw_sample)
    log_jacobian = math.log(max_scale) + torch.log1p(
        -tanh_raw.pow(2).clamp(max=1.0 - 1e-6)
    )
    return log_prob - log_jacobian


def masked_pointwise_ppo_loss(
    current_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    advantage: torch.Tensor,
    mask: torch.Tensor,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PPO independently at selected points; unselected points are absent."""
    ratio = torch.exp(current_logprob - old_logprob)
    unclipped = -advantage * ratio
    clipped = -advantage * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    selected = torch.maximum(unclipped, clipped).masked_select(mask)
    if selected.numel() == 0:
        return current_logprob.sum() * 0.0, ratio.masked_select(mask)
    return selected.mean(), ratio.masked_select(mask)
