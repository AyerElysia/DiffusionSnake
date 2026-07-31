#!/usr/bin/env python3
"""Offline per-point policy bottleneck diagnostics.

The tool consumes a trusted NPZ/PT cache and never imports or modifies training code.
GT-derived targets are diagnostic oracles only; they must not be fed into a deployed
or training policy.

Cache schema ``perpoint_policy_bottlenecks.v1``
------------------------------------------------
Required arrays (C contours, P contour vertices):

``schema_version``
    Scalar string equal to ``perpoint_policy_bottlenecks.v1``.
``gt_point_semantics``
    Scalar string equal to ``closed_polyline_vertices``. GT indices are explicitly
    not assumed to correspond to current-point indices; geometry targets use nearest
    projection onto closed GT line segments.
``image_id``, ``group_id``, ``contour_id``
    Shape ``[C]``. Splits are made on image/group IDs; stable contour IDs identify
    contour-level bootstrap clusters and may repeat across cached states/steps.
``current_points``, ``fm_velocity``, ``gt_points``
    Float arrays with shape ``[C, P, 2]``. Pure-FM endpoints are
    ``current_points + fm_velocity``; ``gt_points`` are closed-polyline vertices.
``current_features``
    Float array ``[C, P, F]`` containing the cached policy-visible state features.

Optional arrays:

``valid_mask``
    Boolean ``[C, P]``; defaults to all valid.
``richer_features``
    Float ``[C, P, R]`` containing the complete richer candidate representation.
``counterfactual_mask``
    Boolean ``[C, P]`` identifying points with truly computed counterfactual metrics.
``reward_credit``
    Float ``[C, P]`` continuous-boundary credit for the same +/- counterfactual.
``delta_iou``, ``delta_dice``, ``delta_mboundf``, ``delta_nsd``
    Optional float ``[C, P]`` real counterfactual metric deltas. Correlation and sign
    agreement are reported only where ``counterfactual_mask`` is true.

The real extractor is ``scripts.extract_perpoint_policy_bottlenecks:extract_cache``.
It receives ``source=Path|None`` plus JSON keyword arguments and returns the same
mapping as an NPZ/PT cache. No extractor is imported during normal cache use.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = "perpoint_policy_bottlenecks.v1"
GT_POINT_SEMANTICS = "closed_polyline_vertices"
DELTA_FIELDS = ("delta_iou", "delta_dice", "delta_mboundf", "delta_nsd")
PROBE_FEATURE_SET_CHOICES = ("current", "richer")
PROBE_TARGET_CHOICES = (
    "scale_current",
    "normal_residual",
    "residual_2d",
    "reward_credit",
)
DEFAULT_PROBE_TARGETS = ("scale_current", "normal_residual", "residual_2d")
PROBE_SPLIT_CHOICES = ("image", "group")
PROBE_MODEL_CHOICES = ("linear", "h64", "h256")
SCHEMA = {
    "schema_version": SCHEMA_VERSION,
    "dimensions": {"C": "contours", "P": "contour vertices"},
    "required": {
        "schema_version": "scalar string",
        "gt_point_semantics": f"scalar string equal to {GT_POINT_SEMANTICS!r}",
        "image_id": "[C] string/int",
        "group_id": "[C] string/int",
        "contour_id": "[C] string/int, stable bootstrap cluster ID",
        "current_points": "[C,P,2] float",
        "fm_velocity": "[C,P,2] float",
        "gt_points": "[C,P,2] float, closed GT polyline vertices (not index-aligned)",
        "current_features": "[C,P,F] float, cached policy-visible state",
    },
    "optional": {
        "valid_mask": "[C,P] bool; omitted means all valid",
        "richer_features": "[C,P,R] float, complete richer representation",
        "counterfactual_mask": "[C,P] bool, true only for real metric evaluation",
        "reward_credit": "[C,P] float",
        "delta_iou": "[C,P] float",
        "delta_dice": "[C,P] float",
        "delta_mboundf": "[C,P] float",
        "delta_nsd": "[C,P] float",
    },
    "safety": (
        "gt_points and all GT-derived oracle targets are diagnostic-only and must "
        "not be added to the formal training policy"
    ),
}


@dataclass(frozen=True)
class DiagnosticCache:
    gt_point_semantics: str
    image_id: np.ndarray
    group_id: np.ndarray
    contour_id: np.ndarray
    current_points: np.ndarray
    fm_velocity: np.ndarray
    gt_points: np.ndarray
    current_features: np.ndarray
    valid_mask: np.ndarray
    counterfactual_mask: np.ndarray
    has_counterfactual_mask: bool
    richer_features: Optional[np.ndarray]
    reward_credit: Optional[np.ndarray]
    metric_deltas: Dict[str, np.ndarray]

    @property
    def num_contours(self) -> int:
        return int(self.current_points.shape[0])

    @property
    def num_points(self) -> int:
        return int(self.current_points.shape[1])


@dataclass
class ProbeModel:
    kind: str
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    params: Dict[str, np.ndarray]
    epochs: int = 0


def _as_scalar_string(value: Any) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("schema_version must be a scalar string")
    item = array.reshape(()).item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def _as_id_array(value: Any, name: str, length: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}], got {array.shape}")
    return array.astype(str)


def _as_float_array(value: Any, name: str, shape_prefix: Tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != len(shape_prefix) or array.shape[: len(shape_prefix)] != shape_prefix:
        raise ValueError(f"{name} must have shape {shape_prefix}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got {array.dtype}")
    return array.astype(np.float64, copy=False)


def validate_cache(mapping: Mapping[str, Any]) -> DiagnosticCache:
    """Validate and normalize an in-memory cache mapping without side effects."""
    required = set(SCHEMA["required"])
    missing = sorted(required.difference(mapping))
    if missing:
        raise ValueError(f"cache is missing required fields: {', '.join(missing)}")
    version = _as_scalar_string(mapping["schema_version"])
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION!r}")
    gt_point_semantics = _as_scalar_string(mapping["gt_point_semantics"])
    if gt_point_semantics != GT_POINT_SEMANTICS:
        raise ValueError(
            f"unsupported gt_point_semantics {gt_point_semantics!r}; "
            f"expected {GT_POINT_SEMANTICS!r}"
        )

    current = np.asarray(mapping["current_points"])
    if current.ndim != 3 or current.shape[-1] != 2:
        raise ValueError(f"current_points must have shape [C,P,2], got {current.shape}")
    contours, points = int(current.shape[0]), int(current.shape[1])
    if contours < 1 or points < 3:
        raise ValueError("cache must contain at least one contour with at least three points")
    geom_shape = (contours, points, 2)
    current = _as_float_array(current, "current_points", geom_shape)
    velocity = _as_float_array(mapping["fm_velocity"], "fm_velocity", geom_shape)
    gt = _as_float_array(mapping["gt_points"], "gt_points", geom_shape)

    features = np.asarray(mapping["current_features"])
    if features.ndim != 3 or features.shape[:2] != (contours, points) or features.shape[2] < 1:
        raise ValueError(
            "current_features must have shape [C,P,F] with F >= 1, "
            f"got {features.shape}"
        )
    if not np.issubdtype(features.dtype, np.number):
        raise ValueError("current_features must be numeric")
    features = features.astype(np.float64, copy=False)

    if "valid_mask" in mapping:
        valid = np.asarray(mapping["valid_mask"], dtype=bool)
        if valid.shape != (contours, points):
            raise ValueError(
                f"valid_mask must have shape {(contours, points)}, got {valid.shape}"
            )
    else:
        valid = np.ones((contours, points), dtype=bool)

    richer = None
    if "richer_features" in mapping:
        richer = np.asarray(mapping["richer_features"])
        if richer.ndim != 3 or richer.shape[:2] != (contours, points) or richer.shape[2] < 1:
            raise ValueError(
                "richer_features must have shape [C,P,R] with R >= 1, "
                f"got {richer.shape}"
            )
        if not np.issubdtype(richer.dtype, np.number):
            raise ValueError("richer_features must be numeric")
        richer = richer.astype(np.float64, copy=False)

    has_counterfactual_mask = "counterfactual_mask" in mapping
    if has_counterfactual_mask:
        counterfactual_mask = np.asarray(mapping["counterfactual_mask"], dtype=bool)
        if counterfactual_mask.shape != (contours, points):
            raise ValueError(
                "counterfactual_mask must have shape "
                f"{(contours, points)}, got {counterfactual_mask.shape}"
            )
    else:
        counterfactual_mask = valid.copy()

    reward = None
    if "reward_credit" in mapping:
        reward = _as_float_array(
            mapping["reward_credit"], "reward_credit", (contours, points)
        )

    deltas = {}
    for field in DELTA_FIELDS:
        if field in mapping:
            deltas[field] = _as_float_array(mapping[field], field, (contours, points))
    if deltas and reward is None:
        raise ValueError("metric delta fields require reward_credit")
    if reward is not None and not np.isfinite(reward[counterfactual_mask]).all():
        raise ValueError("reward_credit must be finite where counterfactual_mask is true")
    for field, values in deltas.items():
        if not np.isfinite(values[counterfactual_mask]).all():
            raise ValueError(f"{field} must be finite where counterfactual_mask is true")

    finite_geometry = (
        np.isfinite(current).all(axis=-1)
        & np.isfinite(velocity).all(axis=-1)
        & np.isfinite(gt).all(axis=-1)
    )
    valid = valid & finite_geometry
    counterfactual_mask = counterfactual_mask & valid
    if not valid.any():
        raise ValueError("cache has no valid finite geometry points")

    contour_id = _as_id_array(mapping["contour_id"], "contour_id", contours)

    return DiagnosticCache(
        gt_point_semantics=gt_point_semantics,
        image_id=_as_id_array(mapping["image_id"], "image_id", contours),
        group_id=_as_id_array(mapping["group_id"], "group_id", contours),
        contour_id=contour_id,
        current_points=current,
        fm_velocity=velocity,
        gt_points=gt,
        current_features=features,
        valid_mask=valid,
        counterfactual_mask=counterfactual_mask,
        has_counterfactual_mask=has_counterfactual_mask,
        richer_features=richer,
        reward_credit=reward,
        metric_deltas=deltas,
    )


def _mapping_from_pt(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("loading PT caches requires the project's existing PyTorch environment") from exc
    try:
        loaded = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(str(path), map_location="cpu")
    if isinstance(loaded, Mapping) and "cache" in loaded and isinstance(loaded["cache"], Mapping):
        loaded = loaded["cache"]
    if not isinstance(loaded, Mapping):
        raise ValueError("PT cache must contain a mapping, optionally under key 'cache'")
    converted: Dict[str, Any] = {}
    for key, value in loaded.items():
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            value = value.detach().cpu().numpy()
        converted[str(key)] = value
    return converted


def load_cache(path: Path) -> DiagnosticCache:
    """Load an NPZ or trusted PT cache on CPU and validate its schema."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            mapping = {key: archive[key] for key in archive.files}
    elif suffix in (".pt", ".pth"):
        mapping = _mapping_from_pt(path)
    else:
        raise ValueError("cache path must end in .npz, .pt, or .pth")
    return validate_cache(mapping)


def load_extractor(spec: str) -> Callable[..., Mapping[str, Any]]:
    """Resolve a future cache extractor without coupling this module to model code."""
    if ":" not in spec:
        raise ValueError("extractor must use package.module:function syntax")
    module_name, function_name = spec.rsplit(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"extractor {spec!r} is not callable")
    return function


def contour_normals(points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute unit normals from centered closed-contour finite differences."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("points must have shape [C,P,2]")
    tangent = np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1)
    length = np.linalg.norm(tangent, axis=-1, keepdims=True)
    tangent = np.divide(tangent, np.maximum(length, eps))
    return np.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)


def project_to_closed_polylines(
    points: np.ndarray,
    polylines: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Project each point onto the nearest segment of its closed GT polyline."""
    points = np.asarray(points, dtype=np.float64)
    polylines = np.asarray(polylines, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("points must have shape [C,P,2]")
    if polylines.ndim != 3 or polylines.shape[-1] != 2 or polylines.shape[1] < 2:
        raise ValueError("polylines must have shape [C,G,2] with G >= 2")
    if points.shape[0] != polylines.shape[0]:
        raise ValueError("points and polylines must have the same contour count")

    start = polylines
    segment = np.roll(polylines, -1, axis=1) - start
    offset = points[:, :, None, :] - start[:, None, :, :]
    denominator = np.sum(segment * segment, axis=-1)[:, None, :]
    projection = np.sum(offset * segment[:, None, :, :], axis=-1)
    projection = np.divide(
        projection,
        np.maximum(denominator, eps),
        out=np.zeros_like(projection),
    )
    projection = np.clip(projection, 0.0, 1.0)
    candidates = start[:, None, :, :] + projection[..., None] * segment[:, None, :, :]
    squared_distance = np.sum((points[:, :, None, :] - candidates) ** 2, axis=-1)
    nearest = np.argmin(squared_distance, axis=-1)
    return np.take_along_axis(candidates, nearest[..., None, None], axis=2).squeeze(2)


def optimal_scale_residual(
    fm_velocity: np.ndarray,
    target_residual: np.ndarray,
    bound: Optional[float],
    eps: float = 1e-12,
) -> np.ndarray:
    """Return pointwise least-squares FM scale residuals, optionally bounded."""
    velocity = np.asarray(fm_velocity, dtype=np.float64)
    residual = np.asarray(target_residual, dtype=np.float64)
    denominator = np.sum(velocity * velocity, axis=-1)
    numerator = np.sum(velocity * residual, axis=-1)
    scale = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > eps,
    )
    if bound is not None:
        if bound <= 0:
            raise ValueError("scale bound must be positive")
        scale = np.clip(scale, -float(bound), float(bound))
    return scale


def zero_mean_bounded_scale(
    fm_velocity: np.ndarray,
    target_residual: np.ndarray,
    valid_mask: np.ndarray,
    bound: float,
    iterations: int = 80,
    eps: float = 1e-12,
) -> np.ndarray:
    """Solve the per-contour bounded scale oracle with mean(scale)=0."""
    if bound <= 0:
        raise ValueError("bound must be positive")
    velocity = np.asarray(fm_velocity, dtype=np.float64)
    residual = np.asarray(target_residual, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    unconstrained = optimal_scale_residual(velocity, residual, bound=None, eps=eps)
    weights = np.sum(velocity * velocity, axis=-1)
    result = np.zeros_like(unconstrained)

    for contour_index in range(velocity.shape[0]):
        selected = valid[contour_index] & (weights[contour_index] > eps)
        if not selected.any():
            continue
        a = unconstrained[contour_index, selected]
        w = weights[contour_index, selected]

        def constrained_sum(lagrange: float) -> float:
            values = np.clip(a - lagrange / (2.0 * w), -bound, bound)
            return float(values.sum())

        low, high = -1.0, 1.0
        while constrained_sum(low) < 0.0:
            low *= 2.0
        while constrained_sum(high) > 0.0:
            high *= 2.0
        for _ in range(int(iterations)):
            middle = 0.5 * (low + high)
            if constrained_sum(middle) > 0.0:
                low = middle
            else:
                high = middle
        values = np.clip(a - 0.5 * (low + high) / (2.0 * w), -bound, bound)
        # Numerical cleanup keeps the diagnostic invariant explicit.
        if abs(float(values.sum())) > 1e-9:
            values -= float(values.mean())
            values = np.clip(values, -bound, bound)
        result[contour_index, selected] = values
    return result


def _finite_percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else float("nan")


def _mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    statistic: Callable[[np.ndarray], float] = _mean,
    samples: int = 1000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Percentile cluster bootstrap for a one-array statistic."""
    values = np.asarray(values)
    clusters = np.asarray(clusters).astype(str)
    finite = np.isfinite(values)
    values, clusters = values[finite], clusters[finite]
    unique = np.unique(clusters)
    estimate = float(statistic(values)) if values.size else float("nan")
    if values.size == 0 or unique.size == 0 or samples <= 0:
        return {
            "estimate": estimate,
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "bootstrap_valid": 0,
        }
    indices = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(samples)):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        draw = np.concatenate([indices[cluster] for cluster in sampled])
        value = float(statistic(values[draw]))
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        low = high = float("nan")
    else:
        low, high = np.percentile(np.asarray(draws), [2.5, 97.5])
    return {
        "estimate": estimate,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_valid": len(draws),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, implemented without SciPy."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if left.size < 2:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(np.sum(left * left) * np.sum(right * right)))
    return float(np.sum(left * right) / denominator) if denominator > 0 else float("nan")


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if left.size < 2:
        return float("nan")
    return pearson_correlation(_rankdata(left), _rankdata(right))


def sign_metrics(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    finite = np.isfinite(prediction) & np.isfinite(target)
    prediction, target = prediction[finite], target[finite]
    selected = (np.abs(prediction) > eps) & (np.abs(target) > eps)
    prediction, target = prediction[selected], target[selected]
    if target.size == 0:
        return {
            "sign_agreement": float("nan"),
            "balanced_sign_agreement": float("nan"),
            "positive_recall": float("nan"),
            "negative_recall": float("nan"),
            "sign_n": 0,
        }
    pred_positive = prediction > 0
    target_positive = target > 0
    positive = target_positive
    negative = ~target_positive
    positive_recall = _mean(pred_positive[positive].astype(float)) if positive.any() else float("nan")
    negative_recall = _mean((~pred_positive[negative]).astype(float)) if negative.any() else float("nan")
    recalls = [value for value in (positive_recall, negative_recall) if np.isfinite(value)]
    return {
        "sign_agreement": float(np.mean(pred_positive == target_positive)),
        "balanced_sign_agreement": float(np.mean(recalls)) if recalls else float("nan"),
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "sign_n": int(target.size),
    }


def paired_cluster_bootstrap_ci(
    left: np.ndarray,
    right: np.ndarray,
    clusters: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    clusters = np.asarray(clusters).astype(str)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right, clusters = left[finite], right[finite], clusters[finite]
    unique = np.unique(clusters)
    estimate = float(statistic(left, right)) if left.size else float("nan")
    if left.size == 0 or unique.size == 0 or samples <= 0:
        return {
            "estimate": estimate,
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "bootstrap_valid": 0,
        }
    indices = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(samples)):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        draw = np.concatenate([indices[cluster] for cluster in sampled])
        value = float(statistic(left[draw], right[draw]))
        if np.isfinite(value):
            draws.append(value)
    if draws:
        low, high = np.percentile(np.asarray(draws), [2.5, 97.5])
    else:
        low = high = float("nan")
    return {
        "estimate": estimate,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_valid": len(draws),
    }


def compute_geometry_oracles(
    cache: DiagnosticCache,
    current_scale_bound: float = 0.10,
    wider_scale_bound: float = 1.0,
) -> Dict[str, Any]:
    """Compute GT-boundary oracles using nearest closed-segment projection."""
    if current_scale_bound <= 0 or wider_scale_bound <= current_scale_bound:
        raise ValueError("wider_scale_bound must be larger than positive current_scale_bound")
    if cache.gt_point_semantics != GT_POINT_SEMANTICS:
        raise ValueError(f"geometry oracle requires {GT_POINT_SEMANTICS!r} GT semantics")
    pure = cache.current_points + cache.fm_velocity
    target_points = project_to_closed_polylines(pure, cache.gt_points)
    residual = target_points - pure
    current_scale = optimal_scale_residual(
        cache.fm_velocity, residual, bound=current_scale_bound
    )
    wider_scale = optimal_scale_residual(
        cache.fm_velocity, residual, bound=wider_scale_bound
    )
    unbounded_scale = optimal_scale_residual(cache.fm_velocity, residual, bound=None)
    zero_mean_scale = zero_mean_bounded_scale(
        cache.fm_velocity,
        residual,
        cache.valid_mask,
        bound=current_scale_bound,
    )
    normals = contour_normals(pure)
    normal_amount = np.sum(residual * normals, axis=-1)

    endpoints = {
        "pure_fm": pure,
        "scale_current": pure + cache.fm_velocity * current_scale[..., None],
        "scale_wider": pure + cache.fm_velocity * wider_scale[..., None],
        "scale_unbounded": pure + cache.fm_velocity * unbounded_scale[..., None],
        "scale_zero_mean": pure + cache.fm_velocity * zero_mean_scale[..., None],
        "normal_residual": pure + normals * normal_amount[..., None],
        "residual_2d": pure + residual,
    }
    errors = {
        name: np.linalg.norm(endpoint - target_points, axis=-1)
        for name, endpoint in endpoints.items()
    }
    base_error = errors["pure_fm"]
    valid = cache.valid_mask
    base_sq_sum = float(np.sum(np.square(base_error[valid])))
    rows = []
    scale_lookup = {
        "scale_current": (current_scale, current_scale_bound),
        "scale_wider": (wider_scale, wider_scale_bound),
        "scale_unbounded": (unbounded_scale, None),
        "scale_zero_mean": (zero_mean_scale, current_scale_bound),
    }
    for name, error in errors.items():
        selected = error[valid]
        gain = base_error[valid] - selected
        row = {
            "oracle": name,
            "n_points": int(selected.size),
            "error_mean": _mean(selected),
            "error_rmse": float(np.sqrt(_mean(np.square(selected)))),
            "error_median": _finite_percentile(selected, 50),
            "error_p90": _finite_percentile(selected, 90),
            "error_p95": _finite_percentile(selected, 95),
            "gain_vs_pure_mean": _mean(gain),
            "improved_fraction": _mean((gain > 1e-12).astype(float)),
            "squared_error_reduction_fraction": (
                1.0 - float(np.sum(np.square(selected))) / base_sq_sum
                if base_sq_sum > 0
                else float("nan")
            ),
        }
        if name in scale_lookup:
            scale, bound = scale_lookup[name]
            selected_scale = scale[valid]
            row.update(
                {
                    "scale_mean": _mean(selected_scale),
                    "scale_abs_mean": _mean(np.abs(selected_scale)),
                    "scale_p95_abs": _finite_percentile(np.abs(selected_scale), 95),
                    "scale_saturation_fraction": (
                        _mean((np.abs(selected_scale) >= bound - 1e-9).astype(float))
                        if bound is not None
                        else 0.0
                    ),
                }
            )
        rows.append(row)
    return {
        "rows": rows,
        "endpoints": endpoints,
        "errors": errors,
        "scales": {
            "scale_current": current_scale,
            "scale_wider": wider_scale,
            "scale_unbounded": unbounded_scale,
            "scale_zero_mean": zero_mean_scale,
        },
        "target_points": target_points,
        "target_semantics": "nearest_projection_on_closed_gt_segments_from_pure_fm_endpoint",
        "normal_residual": normal_amount,
        "residual_2d": residual,
    }


def geometry_bootstrap(
    cache: DiagnosticCache,
    geometry: Mapping[str, Any],
    samples: int,
    seed: int,
) -> list[Dict[str, Any]]:
    valid = cache.valid_mask
    image_clusters = np.repeat(cache.image_id[:, None], cache.num_points, axis=1)[valid]
    contour_clusters = np.repeat(cache.contour_id[:, None], cache.num_points, axis=1)[valid]
    base = geometry["errors"]["pure_fm"][valid]
    records = []
    for oracle_index, (oracle, error) in enumerate(geometry["errors"].items()):
        selected = error[valid]
        for cluster_index, (unit, clusters) in enumerate(
            (("image", image_clusters), ("contour", contour_clusters))
        ):
            for statistic_index, (statistic_name, values) in enumerate(
                (("error_mean", selected), ("gain_vs_pure_mean", base - selected))
            ):
                result = cluster_bootstrap_ci(
                    values,
                    clusters,
                    samples=samples,
                    seed=seed + oracle_index * 20 + cluster_index * 4 + statistic_index,
                )
                records.append(
                    {
                        "section": "geometry_oracle",
                        "item": oracle,
                        "statistic": statistic_name,
                        "cluster_unit": unit,
                        "n_clusters": int(np.unique(clusters).size),
                        "bootstrap_samples": int(samples),
                        **result,
                    }
                )
    return records


def grouped_split_masks(
    group_ids: np.ndarray,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create leakage-free train/validation/test masks from cluster IDs."""
    groups = np.asarray(group_ids).astype(str)
    unique = np.unique(groups)
    if unique.size < 3:
        raise ValueError("a grouped split requires at least three unique IDs")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must leave non-empty train, validation, and test sets")
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_validation = max(1, int(round(unique.size * validation_fraction)))
    n_test = max(1, int(round(unique.size * (1.0 - train_fraction - validation_fraction))))
    if n_validation + n_test >= unique.size:
        n_validation, n_test = 1, 1
    validation_groups = shuffled[:n_validation]
    test_groups = shuffled[n_validation : n_validation + n_test]
    train_groups = shuffled[n_validation + n_test :]
    return (
        np.isin(groups, train_groups),
        np.isin(groups, validation_groups),
        np.isin(groups, test_groups),
    )


def _fit_standardization(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    y_mean = y.mean(axis=0)
    y_scale = y.std(axis=0)
    y_scale[y_scale < 1e-8] = 1.0
    return (
        (x - x_mean) / x_scale,
        (y - y_mean) / y_scale,
        x_mean,
        x_scale,
        y_mean,
        y_scale,
    )


def fit_linear_probe(x: np.ndarray, y: np.ndarray, ridge_alpha: float = 1e-3) -> ProbeModel:
    """Fit a standardized multi-output ridge probe."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("x/y must be aligned 2D arrays with at least two samples")
    xs, ys, x_mean, x_scale, y_mean, y_scale = _fit_standardization(x, y)
    sample_count, feature_count = xs.shape
    if feature_count <= sample_count:
        gram = xs.T @ xs + sample_count * float(ridge_alpha) * np.eye(feature_count)
        weight = np.linalg.solve(gram, xs.T @ ys)
    else:
        dual = xs @ xs.T + sample_count * float(ridge_alpha) * np.eye(sample_count)
        weight = xs.T @ np.linalg.solve(dual, ys)
    return ProbeModel(
        kind="linear",
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        params={"weight": weight},
    )


def fit_mlp_probe(
    x: np.ndarray,
    y: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    hidden_dim: int,
    seed: int,
    epochs: int = 100,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
) -> ProbeModel:
    """Fit a one-hidden-layer ReLU MLP with NumPy Adam and validation stopping."""
    if hidden_dim <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("hidden_dim, epochs, and batch_size must be positive")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xv = np.asarray(x_validation, dtype=np.float64)
    yv = np.asarray(y_validation, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if yv.ndim == 1:
        yv = yv[:, None]
    xs, ys, x_mean, x_scale, y_mean, y_scale = _fit_standardization(x, y)
    xvs = (xv - x_mean) / x_scale
    yvs = (yv - y_mean) / y_scale

    rng = np.random.default_rng(seed)
    input_dim, output_dim = xs.shape[1], ys.shape[1]
    params = {
        "w1": rng.normal(0.0, math.sqrt(2.0 / input_dim), (input_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim, dtype=np.float64),
        "w2": rng.normal(0.0, math.sqrt(2.0 / hidden_dim), (hidden_dim, output_dim)),
        "b2": np.zeros(output_dim, dtype=np.float64),
    }
    first = {key: np.zeros_like(value) for key, value in params.items()}
    second = {key: np.zeros_like(value) for key, value in params.items()}
    beta1, beta2 = 0.9, 0.999
    update = 0
    best_loss = float("inf")
    best_params = {key: value.copy() for key, value in params.items()}
    stale = 0
    epochs_run = 0

    for epoch in range(int(epochs)):
        order = rng.permutation(xs.shape[0])
        for start in range(0, xs.shape[0], int(batch_size)):
            batch = order[start : start + int(batch_size)]
            xb, yb = xs[batch], ys[batch]
            pre = xb @ params["w1"] + params["b1"]
            hidden = np.maximum(pre, 0.0)
            prediction = hidden @ params["w2"] + params["b2"]
            grad_prediction = 2.0 * (prediction - yb) / max(yb.size, 1)
            gradients = {
                "w2": hidden.T @ grad_prediction + weight_decay * params["w2"],
                "b2": grad_prediction.sum(axis=0),
            }
            grad_hidden = grad_prediction @ params["w2"].T
            grad_pre = grad_hidden * (pre > 0.0)
            gradients["w1"] = xb.T @ grad_pre + weight_decay * params["w1"]
            gradients["b1"] = grad_pre.sum(axis=0)
            update += 1
            for key in params:
                first[key] = beta1 * first[key] + (1.0 - beta1) * gradients[key]
                second[key] = beta2 * second[key] + (1.0 - beta2) * gradients[key] ** 2
                first_hat = first[key] / (1.0 - beta1**update)
                second_hat = second[key] / (1.0 - beta2**update)
                params[key] -= learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)

        val_hidden = np.maximum(xvs @ params["w1"] + params["b1"], 0.0)
        val_prediction = val_hidden @ params["w2"] + params["b2"]
        val_loss = float(np.mean((val_prediction - yvs) ** 2))
        epochs_run = epoch + 1
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_params = {key: value.copy() for key, value in params.items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break

    return ProbeModel(
        kind=f"h{hidden_dim}",
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        params=best_params,
        epochs=epochs_run,
    )


def predict_probe(model: ProbeModel, x: np.ndarray) -> np.ndarray:
    xs = (np.asarray(x, dtype=np.float64) - model.x_mean) / model.x_scale
    if model.kind == "linear":
        standardized = xs @ model.params["weight"]
    else:
        hidden = np.maximum(xs @ model.params["w1"] + model.params["b1"], 0.0)
        standardized = hidden @ model.params["w2"] + model.params["b2"]
    return standardized * model.y_scale + model.y_mean


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    error = prediction - target
    denominator = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    metrics = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": 1.0 - float(np.sum(error**2)) / denominator if denominator > 0 else float("nan"),
        "pearson": pearson_correlation(prediction.reshape(-1), target.reshape(-1)),
    }
    metrics.update(sign_metrics(prediction, target))
    if target.ndim == 2 and target.shape[1] == 2:
        dot = np.sum(prediction * target, axis=1)
        norm = np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1)
        cosine = np.divide(dot, norm, out=np.full_like(dot, np.nan), where=norm > 1e-12)
        metrics["cosine_mean"] = _mean(cosine)
    else:
        metrics["cosine_mean"] = float("nan")
    return metrics


def regression_cluster_bootstrap(
    prediction: np.ndarray,
    target: np.ndarray,
    clusters: np.ndarray,
    samples: int,
    seed: int,
) -> list[Dict[str, Any]]:
    """Bootstrap regression R2/RMSE/MAE from per-cluster sufficient statistics."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    clusters = np.asarray(clusters).astype(str)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must be aligned 2D arrays")
    finite = np.isfinite(prediction).all(axis=1) & np.isfinite(target).all(axis=1)
    prediction, target, clusters = prediction[finite], target[finite], clusters[finite]
    unique, inverse = np.unique(clusters, return_inverse=True)
    if unique.size == 0 or samples <= 0:
        return []

    output_dim = target.shape[1]
    counts = np.bincount(inverse, minlength=unique.size).astype(np.float64)
    target_sum = np.zeros((unique.size, output_dim), dtype=np.float64)
    target_sq_sum = np.zeros_like(target_sum)
    squared_error = np.zeros(unique.size, dtype=np.float64)
    absolute_error = np.zeros(unique.size, dtype=np.float64)
    np.add.at(target_sum, inverse, target)
    np.add.at(target_sq_sum, inverse, target**2)
    np.add.at(squared_error, inverse, np.sum((prediction - target) ** 2, axis=1))
    np.add.at(absolute_error, inverse, np.sum(np.abs(prediction - target), axis=1))

    draws = {"r2": [], "rmse": [], "mae": []}
    rng = np.random.default_rng(seed)
    for _ in range(int(samples)):
        sampled = rng.integers(0, unique.size, size=unique.size)
        weights = np.bincount(sampled, minlength=unique.size).astype(np.float64)
        n = float(weights @ counts)
        sum_y = weights @ target_sum
        sum_y2 = weights @ target_sq_sum
        sse = float(weights @ squared_error)
        sae = float(weights @ absolute_error)
        denominator = float(np.sum(sum_y2 - (sum_y**2) / max(n, 1.0)))
        draws["r2"].append(1.0 - sse / denominator if denominator > 0 else float("nan"))
        draws["rmse"].append(math.sqrt(sse / max(n * output_dim, 1.0)))
        draws["mae"].append(sae / max(n * output_dim, 1.0))

    estimate = regression_metrics(prediction, target)
    records = []
    for statistic, values in draws.items():
        finite_values = np.asarray(values, dtype=np.float64)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size:
            low, high = np.percentile(finite_values, [2.5, 97.5])
        else:
            low = high = float("nan")
        records.append(
            {
                "statistic": statistic,
                "estimate": estimate[statistic],
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_valid": int(finite_values.size),
                "n_clusters": int(unique.size),
                "bootstrap_samples": int(samples),
            }
        )
    return records


def _probe_targets(
    cache: DiagnosticCache,
    geometry: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    targets = {
        "scale_current": np.asarray(geometry["scales"]["scale_current"])[..., None],
        "normal_residual": np.asarray(geometry["normal_residual"])[..., None],
        "residual_2d": np.asarray(geometry["residual_2d"]),
    }
    if cache.reward_credit is not None:
        targets["reward_credit"] = np.asarray(cache.reward_credit)[..., None]
    return targets


def _validate_probe_selection(
    name: str,
    values: Sequence[str],
    choices: Sequence[str],
) -> Tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be a sequence of values")
    selected = tuple(values)
    if not selected:
        raise ValueError(f"{name} must contain at least one value")
    unknown = [value for value in selected if value not in choices]
    if unknown:
        raise ValueError(
            f"unsupported {name}: {', '.join(repr(value) for value in unknown)}; "
            f"choose from {', '.join(choices)}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError(f"{name} must not contain duplicate values")
    return selected


def run_supervised_probes(
    cache: DiagnosticCache,
    geometry: Mapping[str, Any],
    seeds: Sequence[int] = (13, 37, 101),
    ridge_alpha: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    patience: int = 12,
    bootstrap_records: Optional[list[Dict[str, Any]]] = None,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 20260710,
    probe_feature_sets: Sequence[str] = PROBE_FEATURE_SET_CHOICES,
    probe_targets: Sequence[str] = DEFAULT_PROBE_TARGETS,
    probe_splits: Sequence[str] = PROBE_SPLIT_CHOICES,
    probe_models: Sequence[str] = PROBE_MODEL_CHOICES,
) -> list[Dict[str, Any]]:
    """Run selected supervised probes with leakage-free held-out splits."""
    if len(seeds) != 3:
        raise ValueError("exactly three probe seeds are required")
    selected_feature_sets = _validate_probe_selection(
        "probe feature sets", probe_feature_sets, PROBE_FEATURE_SET_CHOICES
    )
    selected_targets = _validate_probe_selection(
        "probe targets", probe_targets, PROBE_TARGET_CHOICES
    )
    selected_splits = _validate_probe_selection(
        "probe splits", probe_splits, PROBE_SPLIT_CHOICES
    )
    selected_models = _validate_probe_selection(
        "probe models", probe_models, PROBE_MODEL_CHOICES
    )
    if "reward_credit" in selected_targets:
        if cache.reward_credit is None:
            raise ValueError(
                "reward_credit probe target requested but cache is missing reward_credit"
            )
        if not cache.has_counterfactual_mask:
            raise ValueError(
                "reward_credit probe target requested but cache is missing "
                "counterfactual_mask"
            )
        if "image" not in selected_splits:
            raise ValueError("reward_credit probes require the image-held-out split")

    available_feature_sets = {"current": cache.current_features}
    if cache.richer_features is not None:
        available_feature_sets["richer"] = cache.richer_features
    feature_sets = {
        name: available_feature_sets[name]
        for name in selected_feature_sets
        if name in available_feature_sets
    }
    available_targets = _probe_targets(cache, geometry)
    targets = {name: available_targets[name] for name in selected_targets}
    contour_images = np.repeat(cache.image_id[:, None], cache.num_points, axis=1)
    contour_groups = np.repeat(cache.group_id[:, None], cache.num_points, axis=1)
    contour_ids = np.repeat(cache.contour_id[:, None], cache.num_points, axis=1)
    records = []

    for feature_index, (feature_name, feature_array) in enumerate(feature_sets.items()):
        flat_features = feature_array.reshape(-1, feature_array.shape[-1])
        for target_index, (target_name, target_array) in enumerate(targets.items()):
            flat_target = target_array.reshape(-1, target_array.shape[-1])
            target_mask = (
                cache.counterfactual_mask
                if target_name == "reward_credit"
                else cache.valid_mask
            )
            valid = target_mask.reshape(-1).copy()
            valid &= np.isfinite(flat_features).all(axis=1)
            valid &= np.isfinite(flat_target).all(axis=1)
            x, y = flat_features[valid], flat_target[valid]
            image_ids = contour_images.reshape(-1)[valid]
            point_contour_ids = contour_ids.reshape(-1)[valid]
            available_splits = {
                "image": image_ids,
                "group": contour_groups.reshape(-1)[valid],
            }
            target_splits = (
                ("image",) if target_name == "reward_credit" else selected_splits
            )
            ids_by_split = {name: available_splits[name] for name in target_splits}
            for split_index, (split_name, split_ids) in enumerate(ids_by_split.items()):
                for seed_index, seed in enumerate(seeds):
                    train, validation, test = grouped_split_masks(split_ids, int(seed))
                    models = []
                    for model_name in selected_models:
                        if model_name == "linear":
                            model = fit_linear_probe(
                                x[train], y[train], ridge_alpha=ridge_alpha
                            )
                        else:
                            model = fit_mlp_probe(
                                x[train],
                                y[train],
                                x[validation],
                                y[validation],
                                hidden_dim=int(model_name[1:]),
                                seed=int(seed),
                                epochs=epochs,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                patience=patience,
                            )
                        models.append(model)
                    for model_index, model in enumerate(models):
                        test_prediction = predict_probe(model, x[test])
                        test_metrics = regression_metrics(test_prediction, y[test])
                        phase_metrics = {}
                        for phase_name, phase_mask in (
                            ("train", train),
                            ("validation", validation),
                            ("test", test),
                        ):
                            phase_prediction = (
                                test_prediction
                                if phase_name == "test"
                                else predict_probe(model, x[phase_mask])
                            )
                            for metric_name, metric_value in regression_metrics(
                                phase_prediction, y[phase_mask]
                            ).items():
                                phase_metrics[f"{phase_name}_{metric_name}"] = metric_value
                        records.append(
                            {
                                "feature_set": feature_name,
                                "target": target_name,
                                "split": split_name,
                                "seed": int(seed),
                                "model": model.kind,
                                "feature_dim": int(x.shape[1]),
                                "target_dim": int(y.shape[1]),
                                "train_n": int(train.sum()),
                                "validation_n": int(validation.sum()),
                                "test_n": int(test.sum()),
                                "epochs": int(model.epochs),
                                **test_metrics,
                                **phase_metrics,
                            }
                        )
                        if bootstrap_records is not None and bootstrap_samples > 0:
                            clusters_by_unit = {
                                "image": image_ids[test],
                                "contour": point_contour_ids[test],
                            }
                            offset = (
                                feature_index * 10000
                                + target_index * 2000
                                + split_index * 600
                                + seed_index * 100
                                + model_index * 10
                            )
                            for unit_index, (unit, clusters) in enumerate(
                                clusters_by_unit.items()
                            ):
                                bootstrapped = regression_cluster_bootstrap(
                                    test_prediction,
                                    y[test],
                                    clusters,
                                    samples=bootstrap_samples,
                                    seed=bootstrap_seed + offset + unit_index,
                                )
                                for row in bootstrapped:
                                    bootstrap_records.append(
                                        {
                                            "section": "supervised_probe",
                                            "item": target_name,
                                            "feature_set": feature_name,
                                            "split": split_name,
                                            "seed": int(seed),
                                            "model": model.kind,
                                            "cluster_unit": unit,
                                            **row,
                                        }
                                    )
    return records


def aggregate_probe_records(records: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    keys = ("feature_set", "target", "split", "model")
    metrics = ("r2", "rmse", "mae", "pearson", "balanced_sign_agreement", "cosine_mean")
    grouped: Dict[Tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in records:
        grouped.setdefault(tuple(str(row[key]) for key in keys), []).append(row)
    result = []
    for group_key, rows in sorted(grouped.items()):
        output: Dict[str, Any] = dict(zip(keys, group_key))
        output["seeds"] = len(rows)
        for metric in metrics:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            finite_values = values[np.isfinite(values)]
            output[f"{metric}_mean"] = _mean(finite_values)
            output[f"{metric}_std"] = (
                float(finite_values.std()) if finite_values.size else float("nan")
            )
        result.append(output)
    return result


def _metric_name(field: str) -> str:
    return field[len("delta_") :] if field.startswith("delta_") else field


def analyze_reward_credit(
    cache: DiagnosticCache,
    bootstrap_samples: int,
    seed: int,
) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Compare point reward credit with saved task-metric deltas."""
    if cache.reward_credit is None:
        return [], []
    valid = cache.valid_mask & cache.counterfactual_mask
    image_clusters = np.repeat(cache.image_id[:, None], cache.num_points, axis=1)
    contour_clusters = np.repeat(cache.contour_id[:, None], cache.num_points, axis=1)
    rows = []
    bootstrap = []
    statistic_functions = {
        "pearson": pearson_correlation,
        "spearman": spearman_correlation,
        "sign_agreement": lambda x, y: sign_metrics(x, y)["sign_agreement"],
        "balanced_sign_agreement": lambda x, y: sign_metrics(x, y)[
            "balanced_sign_agreement"
        ],
    }
    for metric_index, (field, delta) in enumerate(sorted(cache.metric_deltas.items())):
        selected = valid & np.isfinite(cache.reward_credit) & np.isfinite(delta)
        reward = cache.reward_credit[selected]
        target = delta[selected]
        row = {
            "metric": _metric_name(field),
            "n_points": int(reward.size),
            "reward_mean": _mean(reward),
            "delta_mean": _mean(target),
            "pearson": pearson_correlation(reward, target),
            "spearman": spearman_correlation(reward, target),
            **sign_metrics(reward, target),
        }
        rows.append(row)
        clusters_by_unit = {
            "image": image_clusters[selected],
            "contour": contour_clusters[selected],
        }
        for unit_index, (unit, clusters) in enumerate(clusters_by_unit.items()):
            for statistic_index, (name, function) in enumerate(statistic_functions.items()):
                result = paired_cluster_bootstrap_ci(
                    reward,
                    target,
                    clusters,
                    statistic=function,
                    samples=bootstrap_samples,
                    seed=seed + metric_index * 50 + unit_index * 10 + statistic_index,
                )
                bootstrap.append(
                    {
                        "section": "reward_credit",
                        "item": _metric_name(field),
                        "statistic": name,
                        "cluster_unit": unit,
                        "n_clusters": int(np.unique(clusters).size),
                        "bootstrap_samples": int(bootstrap_samples),
                        **result,
                    }
                )
    return rows, bootstrap


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if _json_safe(row.get(key)) is None else _json_safe(row.get(key))
                    for key in fieldnames
                }
            )


def prepare_output_dir(path: Path) -> Path:
    """Create an output directory, refusing to overwrite any existing report."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_plots(
    out_dir: Path,
    oracle_rows: Sequence[Mapping[str, Any]],
    geometry_bootstrap_rows: Sequence[Mapping[str, Any]],
    probe_aggregate: Sequence[Mapping[str, Any]],
    reward_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Generate compact PNG summaries with the existing Matplotlib dependency."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = []
    names = [str(row["oracle"]) for row in oracle_rows]
    means = np.asarray([row["error_mean"] for row in oracle_rows], dtype=float)
    image_ci = {
        (row["item"], row["statistic"]): row
        for row in geometry_bootstrap_rows
        if row["cluster_unit"] == "image"
    }
    low = np.asarray(
        [image_ci[(name, "error_mean")]["ci95_low"] for name in names], dtype=float
    )
    high = np.asarray(
        [image_ci[(name, "error_mean")]["ci95_high"] for name in names], dtype=float
    )
    yerr = np.vstack((np.maximum(means - low, 0.0), np.maximum(high - means, 0.0)))
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(np.arange(len(names)), means, yerr=yerr, capsize=3)
    axis.set_xticks(np.arange(len(names)), names, rotation=25, ha="right")
    axis.set_ylabel("mean aligned endpoint error")
    axis.set_title("GT geometry oracle capacity (image-cluster 95% CI)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "oracle_errors.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    output.append(path.name)

    if probe_aggregate:
        target_names = sorted({str(row["target"]) for row in probe_aggregate})
        fig, axes = plt.subplots(1, len(target_names), figsize=(6 * len(target_names), 5), squeeze=False)
        for axis, target in zip(axes[0], target_names):
            selected = [
                row
                for row in probe_aggregate
                if row["target"] == target and row["split"] == "image"
            ]
            labels = [f"{row['feature_set']}\n{row['model']}" for row in selected]
            values = [row["r2_mean"] for row in selected]
            errors = [row["r2_std"] for row in selected]
            axis.bar(np.arange(len(selected)), values, yerr=errors, capsize=3)
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_xticks(np.arange(len(selected)), labels, rotation=30, ha="right")
            axis.set_title(target)
            axis.set_ylabel("held-out R2 (mean +/- seed std)")
            axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = out_dir / "probe_performance.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        output.append(path.name)

    if reward_rows:
        metrics = [str(row["metric"]) for row in reward_rows]
        x = np.arange(len(metrics))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].bar(x - 0.18, [row["pearson"] for row in reward_rows], 0.36, label="Pearson")
        axes[0].bar(x + 0.18, [row["spearman"] for row in reward_rows], 0.36, label="Spearman")
        axes[0].axhline(0.0, color="black", linewidth=0.8)
        axes[0].set_xticks(x, metrics)
        axes[0].set_title("Reward credit correlation")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.25)
        axes[1].bar(x - 0.18, [row["sign_agreement"] for row in reward_rows], 0.36, label="sign")
        axes[1].bar(
            x + 0.18,
            [row["balanced_sign_agreement"] for row in reward_rows],
            0.36,
            label="balanced sign",
        )
        axes[1].axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_xticks(x, metrics)
        axes[1].set_title("Reward credit sign agreement")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = out_dir / "reward_credit_alignment.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        output.append(path.name)
    return output


def run_analysis(
    cache: DiagnosticCache,
    out_dir: Path,
    seeds: Sequence[int] = (13, 37, 101),
    current_scale_bound: float = 0.10,
    wider_scale_bound: float = 1.0,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 20260710,
    ridge_alpha: float = 1e-3,
    probe_epochs: int = 100,
    probe_batch_size: int = 2048,
    probe_learning_rate: float = 1e-3,
    probe_patience: int = 12,
    skip_probes: bool = False,
    no_plots: bool = False,
    probe_feature_sets: Sequence[str] = PROBE_FEATURE_SET_CHOICES,
    probe_targets: Sequence[str] = DEFAULT_PROBE_TARGETS,
    probe_splits: Sequence[str] = PROBE_SPLIT_CHOICES,
    probe_models: Sequence[str] = PROBE_MODEL_CHOICES,
) -> Dict[str, Any]:
    """Run all diagnostics and write a new, non-overwriting report directory."""
    selected_feature_sets = _validate_probe_selection(
        "probe feature sets", probe_feature_sets, PROBE_FEATURE_SET_CHOICES
    )
    selected_targets = _validate_probe_selection(
        "probe targets", probe_targets, PROBE_TARGET_CHOICES
    )
    selected_splits = _validate_probe_selection(
        "probe splits", probe_splits, PROBE_SPLIT_CHOICES
    )
    selected_models = _validate_probe_selection(
        "probe models", probe_models, PROBE_MODEL_CHOICES
    )
    out_dir = prepare_output_dir(out_dir)
    geometry = compute_geometry_oracles(
        cache,
        current_scale_bound=current_scale_bound,
        wider_scale_bound=wider_scale_bound,
    )
    geometry_bootstrap_rows = geometry_bootstrap(
        cache, geometry, samples=bootstrap_samples, seed=bootstrap_seed
    )
    probe_rows = []
    probe_bootstrap_rows: list[Dict[str, Any]] = []
    if not skip_probes:
        probe_rows = run_supervised_probes(
            cache,
            geometry,
            seeds=seeds,
            ridge_alpha=ridge_alpha,
            epochs=probe_epochs,
            batch_size=probe_batch_size,
            learning_rate=probe_learning_rate,
            patience=probe_patience,
            bootstrap_records=probe_bootstrap_rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 2000,
            probe_feature_sets=selected_feature_sets,
            probe_targets=selected_targets,
            probe_splits=selected_splits,
            probe_models=selected_models,
        )
    probe_aggregate = aggregate_probe_records(probe_rows)
    reward_rows, reward_bootstrap_rows = analyze_reward_credit(
        cache, bootstrap_samples=bootstrap_samples, seed=bootstrap_seed + 1000
    )
    bootstrap_rows = (
        geometry_bootstrap_rows + probe_bootstrap_rows + reward_bootstrap_rows
    )

    write_json(out_dir / "cache_schema.json", SCHEMA)
    write_csv(out_dir / "oracle_metrics.csv", geometry["rows"])
    write_csv(out_dir / "probe_metrics.csv", probe_rows)
    write_csv(out_dir / "reward_credit_metrics.csv", reward_rows)
    write_csv(out_dir / "bootstrap_metrics.csv", bootstrap_rows)
    plot_files = []
    if not no_plots:
        plot_files = generate_plots(
            out_dir,
            geometry["rows"],
            geometry_bootstrap_rows,
            probe_aggregate,
            reward_rows,
        )

    warnings = []
    if cache.richer_features is None and "richer" in selected_feature_sets:
        warnings.append("richer_features absent: richer-feature probes were skipped")
    if cache.reward_credit is None or not cache.metric_deltas:
        warnings.append("reward_credit or metric deltas absent: credit-alignment analysis was skipped")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "safety": SCHEMA["safety"],
        "cache": {
            "num_contours": cache.num_contours,
            "num_points_per_contour": cache.num_points,
            "valid_points": int(cache.valid_mask.sum()),
            "counterfactual_points": int(cache.counterfactual_mask.sum()),
            "num_images": int(np.unique(cache.image_id).size),
            "num_groups": int(np.unique(cache.group_id).size),
            "current_feature_dim": int(cache.current_features.shape[-1]),
            "richer_feature_dim": (
                int(cache.richer_features.shape[-1]) if cache.richer_features is not None else None
            ),
            "available_metric_deltas": sorted(cache.metric_deltas),
        },
        "config": {
            "current_scale_bound": float(current_scale_bound),
            "wider_scale_bound": float(wider_scale_bound),
            "probe_seeds": [int(value) for value in seeds],
            "probe_feature_sets": list(selected_feature_sets),
            "probe_targets": list(selected_targets),
            "probe_target_masks": {
                target: (
                    "counterfactual_mask_and_finite"
                    if target == "reward_credit"
                    else "valid_mask_and_finite"
                )
                for target in selected_targets
            },
            "probe_target_splits": {
                target: (
                    ["image"] if target == "reward_credit" else list(selected_splits)
                )
                for target in selected_targets
            },
            "probe_splits": list(selected_splits),
            "probe_models": list(selected_models),
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "geometry_oracles": geometry["rows"],
        "probe_aggregate": probe_aggregate,
        "reward_credit": reward_rows,
        "bootstrap": bootstrap_rows,
        "warnings": warnings,
        "output_files": [
            "summary.json",
            "cache_schema.json",
            "oracle_metrics.csv",
            "probe_metrics.csv",
            "reward_credit_metrics.csv",
            "bootstrap_metrics.csv",
            *plot_files,
        ],
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def parse_seeds(value: str) -> Tuple[int, int, int]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if len(seeds) != 3:
        raise argparse.ArgumentTypeError("exactly three seeds are required")
    return seeds  # type: ignore[return-value]


def parse_probe_selection(
    value: str,
    name: str,
    choices: Sequence[str],
) -> Tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(","))
    if any(not item for item in selected):
        raise argparse.ArgumentTypeError(
            f"{name} must be a non-empty comma-separated list without empty values"
        )
    try:
        return _validate_probe_selection(name, selected, choices)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline GT-oracle, feature-probe, and reward-credit bottleneck diagnostics."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--cache", type=Path, help="Input .npz/.pt/.pth cache.")
    source.add_argument(
        "--extractor",
        help="Future extractor in package.module:function form; returns the v1 cache mapping.",
    )
    parser.add_argument("--extract-source", type=Path)
    parser.add_argument("--extractor-kwargs", default="{}", help="JSON object passed to extractor.")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--print-schema", action="store_true")
    parser.add_argument("--seeds", type=parse_seeds, default=(13, 37, 101))
    parser.add_argument("--current-scale-bound", type=float, default=0.10)
    parser.add_argument("--wider-scale-bound", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--probe-batch-size", type=int, default=2048)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-patience", type=int, default=12)
    parser.add_argument(
        "--probe-feature-sets",
        type=lambda value: parse_probe_selection(
            value, "probe feature sets", PROBE_FEATURE_SET_CHOICES
        ),
        default=PROBE_FEATURE_SET_CHOICES,
        metavar="FEATURE_SET,...",
    )
    parser.add_argument(
        "--probe-targets",
        type=lambda value: parse_probe_selection(
            value, "probe targets", PROBE_TARGET_CHOICES
        ),
        default=DEFAULT_PROBE_TARGETS,
        metavar="TARGET,...",
    )
    parser.add_argument(
        "--probe-splits",
        type=lambda value: parse_probe_selection(
            value, "probe splits", PROBE_SPLIT_CHOICES
        ),
        default=PROBE_SPLIT_CHOICES,
        metavar="SPLIT,...",
    )
    parser.add_argument(
        "--probe-models",
        type=lambda value: parse_probe_selection(
            value, "probe models", PROBE_MODEL_CHOICES
        ),
        default=PROBE_MODEL_CHOICES,
        metavar="MODEL,...",
    )
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.print_schema:
        print(json.dumps(SCHEMA, indent=2, ensure_ascii=False))
        return 0
    if args.cache is None and args.extractor is None:
        parser.error("one of --cache or --extractor is required unless --print-schema is used")

    if args.cache is not None:
        cache = load_cache(args.cache)
        source_stem = args.cache.stem
    else:
        try:
            kwargs = json.loads(args.extractor_kwargs)
        except json.JSONDecodeError as exc:
            parser.error(f"invalid --extractor-kwargs JSON: {exc}")
        if not isinstance(kwargs, dict):
            parser.error("--extractor-kwargs must decode to a JSON object")
        extractor = load_extractor(args.extractor)
        mapping = extractor(source=args.extract_source, **kwargs)
        cache = validate_cache(mapping)
        source_stem = args.extract_source.stem if args.extract_source else "extracted_cache"

    out_dir = args.out_dir
    if out_dir is None:
        root = Path(__file__).resolve().parents[1]
        out_dir = root / "report" / "perpoint_policy_bottlenecks" / source_stem
    summary = run_analysis(
        cache,
        out_dir=out_dir,
        seeds=args.seeds,
        current_scale_bound=args.current_scale_bound,
        wider_scale_bound=args.wider_scale_bound,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        ridge_alpha=args.ridge_alpha,
        probe_epochs=args.probe_epochs,
        probe_batch_size=args.probe_batch_size,
        probe_learning_rate=args.probe_learning_rate,
        probe_patience=args.probe_patience,
        probe_feature_sets=args.probe_feature_sets,
        probe_targets=args.probe_targets,
        probe_splits=args.probe_splits,
        probe_models=args.probe_models,
        skip_probes=args.skip_probes,
        no_plots=args.no_plots,
    )
    print(f"Wrote {len(summary['output_files'])} artifacts to {out_dir}")
    for name in summary["output_files"]:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
