"""Five-action Fourier exploration aligned with the deployed AB2 Flow solver.

The public mainline has one RL action family: a low-frequency scalar field is
sampled in an orthonormal Fourier basis and applied along contour normals.  The
Flow prediction is the policy mean.  The four AB2 evaluations used to obtain
that mean are deterministic solver work, not additional RL actions.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch


def stage_progress(fractions: Sequence[float]) -> list[float]:
    """Return the stage-progress conditioning value before every residual step."""

    progress = 0.0
    values: list[float] = []
    for fraction in fractions:
        fraction = float(fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"residual fraction must be in (0, 1], got {fraction}")
        values.append(progress)
        progress += (1.0 - progress) * fraction
    return values


def standard_normal_logprob(z: torch.Tensor) -> torch.Tensor:
    """Mean log-probability per contour under a standard normal."""

    logprob = -0.5 * z.square() - 0.5 * math.log(2.0 * math.pi)
    return logprob.mean(dim=tuple(range(1, logprob.ndim)))


def contour_normals(poly: torch.Tensor) -> torch.Tensor:
    """Unit normals for a closed, cyclic polygon tensor shaped ``(B, N, 2)``."""

    tangent = torch.roll(poly, shifts=-1, dims=1) - torch.roll(poly, shifts=1, dims=1)
    normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)
    return normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _basis(n_points: int, n_modes: int, device, dtype) -> torch.Tensor:
    """Orthonormal low-frequency real Fourier basis, including the DC mode."""

    n_points = int(n_points)
    n_modes = int(n_modes)
    if n_points <= 0 or not 1 <= n_modes <= n_points:
        raise ValueError(f"invalid Fourier shape: points={n_points}, modes={n_modes}")
    theta = torch.arange(n_points, device=device, dtype=dtype)
    theta = theta * (2.0 * math.pi / float(n_points))
    columns = [torch.ones_like(theta) / math.sqrt(float(n_points))]
    frequency = 1
    while len(columns) < n_modes:
        scale = math.sqrt(2.0 / float(n_points))
        columns.append(scale * torch.cos(float(frequency) * theta))
        if len(columns) < n_modes:
            columns.append(scale * torch.sin(float(frequency) * theta))
        frequency += 1
    return torch.stack(columns, dim=1)


def low_frequency_delta(
    poly: torch.Tensor,
    coefficients: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Map Fourier coefficients to a normal-direction contour displacement."""

    basis = _basis(
        poly.size(1), coefficients.size(1), poly.device, poly.dtype
    )
    scalar = coefficients @ basis.transpose(0, 1)
    return contour_normals(poly) * (scalar * float(sigma)).unsqueeze(-1)


def _project_coefficients(
    state: torch.Tensor,
    residual: torch.Tensor,
    sigma: float,
    n_modes: int,
) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError(f"Fourier sigma must be positive, got {sigma}")
    basis = _basis(state.size(1), n_modes, state.device, state.dtype)
    scalar = (residual * contour_normals(state)).sum(dim=-1) / sigma
    return scalar @ basis


def fourier_action_logprob(
    action: torch.Tensor,
    mean: torch.Tensor,
    state: torch.Tensor,
    sigma: float,
    n_modes: int,
) -> torch.Tensor:
    """Recompute an outer action's log-probability under the current Flow mean."""

    coefficients = _project_coefficients(
        state, action.detach() - mean, sigma, n_modes
    )
    return standard_normal_logprob(coefficients)


def fourier_mean_kl(
    current_mean: torch.Tensor,
    reference_mean: torch.Tensor,
    state: torch.Tensor,
    sigma: float,
    n_modes: int,
) -> torch.Tensor:
    """Gaussian KL induced by a Flow-mean shift in Fourier coordinates."""

    coefficients = _project_coefficients(
        state, current_mean - reference_mean.detach(), sigma, n_modes
    )
    return 0.5 * coefficients.square().mean(dim=1)


def _ab2_step(
    flow,
    cnn_feature: torch.Tensor,
    i_state: torch.Tensor,
    c_state: torch.Tensor,
    py_ind: torch.Tensor,
    x_t: torch.Tensor,
    step_index: int,
    total_steps: int,
    previous_velocity: Optional[torch.Tensor],
    sampled_feat: torch.Tensor,
    detail_feat: Optional[torch.Tensor],
    contour_scale: torch.Tensor,
    x_self_cond: Optional[torch.Tensor],
    s_value: float,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """One deterministic production-equivalent Adams--Bashforth-2 step."""

    total_steps = int(total_steps)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    dt = 1.0 / float(total_steps)
    t_value = float(step_index) * dt
    t_tensor = torch.full(
        (x_t.size(0),), t_value, device=x_t.device, dtype=x_t.dtype
    )
    s_tensor = None
    if getattr(flow, "_use_s_cond", False):
        s_tensor = torch.full(
            (x_t.size(0),), float(s_value), device=x_t.device, dtype=x_t.dtype
        )
    velocity, _ = flow.predict_velocity(
        cnn_feature,
        i_state,
        c_state,
        sampled_feat,
        detail_feat,
        py_ind,
        x_t,
        t_tensor,
        contour_scale=contour_scale.view(-1).to(x_t),
        x_self_cond=x_self_cond,
        s=s_tensor,
    )
    next_self_cond = None
    if getattr(flow, "_use_self_conditioning", False):
        next_self_cond = (x_t + (1.0 - t_value) * velocity).detach()
    if step_index == 0:
        if previous_velocity is not None:
            raise RuntimeError("AB2 bootstrap unexpectedly received velocity history")
        effective_velocity = velocity
    else:
        if previous_velocity is None:
            raise RuntimeError("AB2 history missing after bootstrap step")
        effective_velocity = 1.5 * velocity - 0.5 * previous_velocity
    next_x = x_t + dt * effective_velocity
    if getattr(flow, "_ode_smooth_k", 0) > 0:
        next_x = flow.fourier_smooth(next_x, flow._ode_smooth_k)
    return next_x, next_self_cond, velocity


def _flow_displacement(
    flow,
    cnn_feature: torch.Tensor,
    i_state: torch.Tensor,
    c_state: torch.Tensor,
    py_ind: torch.Tensor,
    latent: torch.Tensor,
    steps: int,
    s_value: float,
) -> torch.Tensor:
    context = flow.prepare_sampling_context(cnn_feature, i_state, py_ind)
    x = latent
    self_condition = (
        torch.zeros_like(x) if getattr(flow, "_use_self_conditioning", False) else None
    )
    previous_velocity = None
    for step_index in range(int(steps)):
        x, next_self_condition, velocity = _ab2_step(
            flow,
            cnn_feature,
            i_state,
            c_state,
            py_ind,
            x,
            step_index,
            int(steps),
            previous_velocity,
            context["sampled_feat"],
            context["detail_feat"],
            context["contour_scale"],
            self_condition,
            s_value,
        )
        previous_velocity = velocity.detach()
        if getattr(flow, "_use_self_conditioning", False):
            self_condition = next_self_condition
    displacement = flow.denormalize_pred_disp(x, context["contour_scale"])
    return flow.clamp_pred_disp(displacement, i_state)


def outer_action_mean(
    flow,
    cnn_feature: torch.Tensor,
    i_state: torch.Tensor,
    c_state: torch.Tensor,
    py_ind: torch.Tensor,
    fraction: float,
    steps: int,
    s_value: float,
    latent: torch.Tensor,
) -> torch.Tensor:
    """Flow policy mean for one residual outer stage."""

    return _flow_displacement(
        flow,
        cnn_feature,
        i_state,
        c_state,
        py_ind,
        latent,
        steps,
        s_value,
    ) * float(fraction)
