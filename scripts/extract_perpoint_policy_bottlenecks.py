#!/usr/bin/env python3
"""Extract real final-step per-point FM bottleneck states and counterfactuals.

The model path deliberately reuses ``eval_v37_full_iou`` for checkpoint loading,
manual GT-init context construction, and the deterministic five-step ODE rollout.
The saved GT vertices are a closed polyline, not point-index correspondence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "perpoint_policy_bottlenecks.v1"
DEFAULT_CFG = ROOT / "configs/1232_final_v5_perpoint_fmscale_v15_last2_gpu5.yaml"
DEFAULT_CKPT = (
    ROOT
    / "data/outputs/1232_final_v5_perpoint_fmscale_bs6_gpu5/checkpoints/step1000.pt"
)


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _parse_indices(value: str) -> list[int]:
    try:
        result = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("step indices must be comma-separated integers") from exc
    if not result or any(index < 0 or index >= 5 for index in result):
        raise argparse.ArgumentTypeError("cache step indices must be in [0, 4]")
    return result


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lines(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _point_major(tensor, num_points: int):
    if tensor is None or tensor.ndim != 3:
        return None
    if tensor.size(1) == num_points:
        return tensor
    if tensor.size(2) == num_points:
        return tensor.transpose(1, 2)
    return None


def _contour_geometry(points, fm_velocity, frac: float):
    import torch

    eps = 1e-6
    center = points.mean(dim=1, keepdim=True)
    centered = points - center
    bbox_min = points.amin(dim=1, keepdim=True)
    bbox_size = (points.amax(dim=1, keepdim=True) - bbox_min).clamp_min(eps)
    canonical = (points - bbox_min) / bbox_size
    contour_scale = bbox_size.norm(dim=-1, keepdim=True).clamp_min(eps)
    centered_norm = centered / contour_scale

    tangent = torch.roll(points, -1, dims=1) - torch.roll(points, 1, dims=1)
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(eps)
    normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)
    edge_prev = points - torch.roll(points, 1, dims=1)
    edge_next = torch.roll(points, -1, dims=1) - points
    edge_prev = edge_prev / edge_prev.norm(dim=-1, keepdim=True).clamp_min(eps)
    edge_next = edge_next / edge_next.norm(dim=-1, keepdim=True).clamp_min(eps)
    signed_curvature = (
        edge_prev[..., 0] * edge_next[..., 1]
        - edge_prev[..., 1] * edge_next[..., 0]
    ).unsqueeze(-1)

    magnitude = fm_velocity.norm(dim=-1, keepdim=True)
    magnitude_norm = magnitude / magnitude.mean(dim=1, keepdim=True).clamp_min(eps)
    frac_feature = points.new_full((*points.shape[:2], 1), float(frac))
    geometry = torch.cat(
        [
            centered_norm,
            canonical,
            fm_velocity,
            magnitude,
            magnitude_norm,
            tangent,
            normal,
            signed_curvature,
            frac_feature,
        ],
        dim=-1,
    )
    names = [
        "centered_x_norm",
        "centered_y_norm",
        "canonical_x",
        "canonical_y",
        "fm_x",
        "fm_y",
        "fm_magnitude",
        "fm_magnitude_norm",
        "tangent_x",
        "tangent_y",
        "normal_x",
        "normal_y",
        "signed_curvature",
        "fraction",
    ]
    return geometry, magnitude_norm, names


def _build_features(record, step_embeddings):
    import torch

    points = record["state"]
    num_points = int(points.size(1))
    sampled = _point_major(record["sampled_feat"], num_points)
    if sampled is None:
        raise RuntimeError(
            f"sampled_feat is not point-aligned: {tuple(record['sampled_feat'].shape)}"
        )
    fm_velocity = record["fm_velocity"]
    geometry, magnitude_norm, geometry_names = _contour_geometry(
        points, fm_velocity, record["fraction"]
    )

    # Exact input seen by the deployed point head: local sampled feature,
    # outer-step embedding, and normalized FM magnitude.  The contour-global
    # pooled feature only feeds the separate global branch (and its mu output is
    # discarded when zero_mean_local=True), so keep it out of this capacity
    # probe and expose it only through the richer candidate representation.
    pooled = sampled.mean(dim=1, keepdim=True).expand(-1, num_points, -1)
    step_embedding = step_embeddings[int(record["step_index"])].to(
        device=points.device, dtype=points.dtype
    )
    step_embedding = step_embedding.view(1, 1, -1).expand(
        points.size(0), num_points, -1
    )
    current = torch.cat([sampled, step_embedding, magnitude_norm], dim=-1)

    neighbor_mean = 0.5 * (
        torch.roll(sampled, 1, dims=1) + torch.roll(sampled, -1, dims=1)
    )
    neighbor_diff = 0.5 * (
        torch.roll(sampled, -1, dims=1) - torch.roll(sampled, 1, dims=1)
    )
    richer_parts = [geometry, sampled, pooled, neighbor_mean, neighbor_diff]
    components = [
        {"name": "geometry", "dim": int(geometry.size(-1)), "fields": geometry_names},
        {"name": "sampled_feat", "dim": int(sampled.size(-1))},
        {"name": "sampled_feat_global_mean_repeated", "dim": int(pooled.size(-1))},
        {"name": "cyclic_neighbor_sampled_feat_mean", "dim": int(sampled.size(-1))},
        {"name": "cyclic_neighbor_sampled_feat_difference", "dim": int(sampled.size(-1))},
    ]
    detail = _point_major(record.get("detail_feat"), num_points)
    if detail is not None and detail.shape[:2] == points.shape[:2]:
        richer_parts.append(detail)
        components.append({"name": "ctx_detail_feat", "dim": int(detail.size(-1))})
    richer = torch.cat(richer_parts, dim=-1)
    current_components = [
        {"name": "sampled_feat_local", "dim": int(sampled.size(-1))},
        {"name": "checkpoint_step_embedding_repeated", "dim": int(step_embedding.size(-1))},
        {"name": "fm_magnitude_normalized", "dim": 1},
    ]
    return current, richer, current_components, components


def _selected_point_indices(num_points: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or max_points >= num_points:
        return np.arange(num_points, dtype=np.int64)
    return np.unique(
        np.floor(np.arange(max_points, dtype=np.float64) * num_points / max_points).astype(
            np.int64
        )
    )


def _counterfactual_metrics(
    eval_mod,
    current,
    fm_velocity,
    gt,
    image_hw,
    scale_delta: float,
    max_points: int,
    nsd_delta_px: float,
):
    import torch

    from lib.train.continuous_boundary_credit import continuous_boundary_quality_delta
    from lib.utils.snake import snake_config

    contours, points = int(current.size(0)), int(current.size(1))
    selected = _selected_point_indices(points, max_points)
    mask = np.zeros((contours, points), dtype=bool)
    fields = {
        name: np.full((contours, points), np.nan, dtype=np.float32)
        for name in ("reward_credit", "delta_iou", "delta_dice", "delta_mboundf", "delta_nsd")
    }
    if selected.size == 0:
        return mask, fields

    pure = current + fm_velocity
    plus_points = current[:, selected] + fm_velocity[:, selected] * (1.0 + scale_delta)
    minus_points = current[:, selected] + fm_velocity[:, selected] * (1.0 - scale_delta)
    dist_max_px = float(getattr(eval_mod.cfg, "rl_v4_reward_dist_max_px", 8.0))
    credit = continuous_boundary_quality_delta(
        plus_points,
        minus_points,
        gt,
        coord_scale=float(snake_config.down_ratio),
        dist_max_px=dist_max_px,
    )
    fields["reward_credit"][:, selected] = credit.detach().cpu().numpy().astype(np.float32)

    dr = float(snake_config.down_ratio)
    height, width = int(image_hw[0]), int(image_hw[1])
    pure_np = (pure.detach().cpu().numpy() * dr).astype(np.float32)
    current_np = (current.detach().cpu().numpy() * dr).astype(np.float32)
    velocity_np = (fm_velocity.detach().cpu().numpy() * dr).astype(np.float32)
    gt_np = (gt.detach().cpu().numpy() * dr).astype(np.float32)
    for contour_index in range(contours):
        gt_mask = eval_mod.poly_to_mask(gt_np[contour_index], height, width)
        for point_index in selected.tolist():
            plus = pure_np[contour_index].copy()
            minus = pure_np[contour_index].copy()
            plus[point_index] = (
                current_np[contour_index, point_index]
                + velocity_np[contour_index, point_index] * (1.0 + scale_delta)
            )
            minus[point_index] = (
                current_np[contour_index, point_index]
                + velocity_np[contour_index, point_index] * (1.0 - scale_delta)
            )
            plus_mask = eval_mod.poly_to_mask(plus, height, width)
            minus_mask = eval_mod.poly_to_mask(minus, height, width)
            plus_iou = eval_mod.compute_iou(plus_mask, gt_mask)
            minus_iou = eval_mod.compute_iou(minus_mask, gt_mask)
            plus_metrics = (
                plus_iou,
                eval_mod.compute_dice_from_iou(plus_iou),
                eval_mod.compute_mboundf(plus_mask, gt_mask),
                float(eval_mod._calc_nsd(plus_mask, gt_mask, delta_px=nsd_delta_px)),
            )
            minus_metrics = (
                minus_iou,
                eval_mod.compute_dice_from_iou(minus_iou),
                eval_mod.compute_mboundf(minus_mask, gt_mask),
                float(eval_mod._calc_nsd(minus_mask, gt_mask, delta_px=nsd_delta_px)),
            )
            for field, plus_value, minus_value in zip(
                ("delta_iou", "delta_dice", "delta_mboundf", "delta_nsd"),
                plus_metrics,
                minus_metrics,
            ):
                fields[field][contour_index, point_index] = plus_value - minus_value
            mask[contour_index, point_index] = True
    return mask, fields


def _batch_to_device(batch: Dict[str, Any], device) -> Dict[str, Any]:
    import torch

    for key, value in list(batch.items()):
        if key == "locate_feat" or str(key).startswith("locate_feat_"):
            continue
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def _metric_image_hw(batch: Mapping[str, Any], fallback: Sequence[int]) -> tuple[int, int]:
    original = batch.get("orig_img")
    if isinstance(original, (list, tuple)) and original:
        shape = original[0].shape
        if len(shape) >= 2:
            return int(shape[0]), int(shape[1])
    return int(fallback[0]), int(fallback[1])


def _sample_path_list(dataset) -> list[str]:
    for name in ("train_images_path", "images", "image_paths"):
        values = getattr(dataset, name, None)
        if isinstance(values, (list, tuple)):
            return [str(value) for value in values]
    return []


def _group_id(sample_id: str) -> str:
    value = re.sub(r"(?:_image)?$", "", str(sample_id))
    return value or str(sample_id)


def _append(store: Dict[str, list[np.ndarray]], key: str, value: Any) -> None:
    store.setdefault(key, []).append(np.asarray(value))


def _finalize(store: Dict[str, list[np.ndarray]]) -> Dict[str, np.ndarray]:
    mapping = {key: np.concatenate(parts, axis=0) for key, parts in store.items()}
    mapping["schema_version"] = np.asarray(SCHEMA_VERSION)
    mapping["gt_point_semantics"] = np.asarray("closed_polyline_vertices")
    return mapping


def _save_cache(
    mapping: Mapping[str, np.ndarray], output: Path, provenance: Path
) -> None:
    output = Path(output)
    provenance = Path(provenance)
    if output.exists() or provenance.exists():
        raise FileExistsError(
            f"refusing to overwrite cache/provenance: {output}, {provenance}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **mapping)


def extract_cache(
    source: Optional[Path] = None,
    cfg_path: Path | str = DEFAULT_CFG,
    checkpoint: Path | str = DEFAULT_CKPT,
    start_index: int = 0,
    end_index: Optional[int] = None,
    max_samples: Optional[int] = 1,
    cache_step_indices: Sequence[int] = (4,),
    counterfactual_max_points: int = 16,
    counterfactual_scale: float = 0.02,
    seed: int = 20260710,
    gpu: Optional[int] = None,
    output: Optional[Path | str] = None,
    provenance_output: Optional[Path | str] = None,
) -> Mapping[str, np.ndarray]:
    """Run real model extraction and return a validated-v1-compatible mapping."""
    cfg_path = Path(cfg_path).expanduser().resolve()
    checkpoint = Path(source or checkpoint).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    cache_step_indices = sorted({int(value) for value in cache_step_indices})
    if not cache_step_indices or any(value < 0 or value >= 5 for value in cache_step_indices):
        raise ValueError("cache_step_indices must be a non-empty subset of [0, 4]")
    if counterfactual_scale <= 0:
        raise ValueError("counterfactual_scale must be positive")

    os.environ["CFG_FILE"] = str(cfg_path)
    os.environ["EVAL_ABLATION_MODE"] = "gt_init"
    os.environ["EVAL_FM_POLICY"] = "off"
    if gpu is not None:
        os.environ["EVAL_GPU"] = str(int(gpu))

    import torch

    original_argv = sys.argv[:]
    try:
        sys.argv[:] = [sys.argv[0]]
        from lib.datasets.collate_batch import make_collator
        from lib.datasets.make_dataset import make_dataset
        from lib.datasets.transforms import make_transforms
        from scripts import eval_v37_full_iou as eval_mod
    finally:
        sys.argv[:] = original_argv

    try:
        checkpoint_obj = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint_obj = torch.load(str(checkpoint), map_location="cpu")
    policy_state = (
        checkpoint_obj.get("fm_velocity_policy_state_dict")
        if isinstance(checkpoint_obj, dict)
        else None
    )
    if not isinstance(policy_state, dict) or "step_embed.weight" not in policy_state:
        raise RuntimeError("Checkpoint has no per-point policy step embedding")
    step_embeddings = policy_state["step_embed.weight"].detach().cpu()
    del checkpoint_obj, policy_state

    eval_mod.apply_gpu_override()
    eval_mod.apply_ablation_mode()
    eval_mod.set_eval_seed(int(seed))
    model, device, loaded_checkpoint, policy, rollout = eval_mod.load_model(
        str(checkpoint), return_policy=True
    )
    if policy is not None or rollout["effective_mode"] != "off":
        raise RuntimeError("Extractor requires deterministic pure-FM policy-off rollout")
    if rollout["outer_steps"] != 5 or len(rollout["fractions"]) != 5:
        raise RuntimeError(f"Expected unified deterministic five-step rollout, got {rollout}")

    dataset = make_dataset(
        eval_mod.cfg,
        eval_mod.cfg.test.dataset,
        make_transforms(eval_mod.cfg, False),
        False,
    )
    collator = make_collator(eval_mod.cfg)
    dataset_size = len(dataset)
    start = max(int(start_index), 0)
    stop = dataset_size if end_index is None else min(int(end_index), dataset_size)
    if stop <= start:
        raise ValueError(f"empty sample range [{start}, {stop}) for dataset size {dataset_size}")
    indices = list(range(start, stop))
    if max_samples is not None and int(max_samples) >= 0:
        indices = indices[: int(max_samples)]
    if not indices:
        raise ValueError("MAX_SAMPLES selected no samples")

    core = model.net if hasattr(model, "net") else model
    store: Dict[str, list[np.ndarray]] = {}
    sample_ids: list[str] = []
    sample_paths: list[str] = []
    failures: list[Dict[str, Any]] = []
    current_components = None
    richer_components = None
    nsd_delta_px = float(os.environ.get("NSD_DELTA_PX", "2.0"))

    with torch.no_grad():
        for dataset_index in indices:
            try:
                sample = dataset[dataset_index]
                identity = eval_mod._sample_identity(sample, dataset_index)
                batch = _batch_to_device(collator([sample]), device)
                context = eval_mod.prepare_manual_gt_init_context(core, batch, device)
                _disp, _states, records = eval_mod._deterministic_unified_rollout(
                    core.gcn,
                    None,
                    context["cnn_feature"],
                    context["i_it_py"],
                    context["py_ind"],
                    rollout["fractions"],
                    rollout["actual_ode_steps"],
                    active_step_indices=rollout["active_step_indices"],
                    return_states=True,
                    return_step_records=True,
                )
                sample_id = str(identity["sample_id"])
                sample_path = str(identity["sample_path"] or "")
                sample_ids.append(sample_id)
                sample_paths.append(sample_path)
                for step_index in cache_step_indices:
                    record = records[step_index]
                    current_features, richer_features, current_parts, richer_parts = (
                        _build_features(record, step_embeddings)
                    )
                    if current_components is None:
                        current_components = current_parts
                        richer_components = richer_parts
                    elif current_parts != current_components or richer_parts != richer_components:
                        raise RuntimeError("feature composition changed across cached states")
                    current = record["state"]
                    gt = context["i_gt_py"]
                    contour_count, point_count = current.shape[:2]
                    cf_mask, fields = _counterfactual_metrics(
                        eval_mod,
                        current,
                        record["fm_velocity"],
                        gt,
                        _metric_image_hw(batch, context["image_hw"]),
                        float(counterfactual_scale),
                        int(counterfactual_max_points),
                        nsd_delta_px,
                    )
                    image_ids = np.asarray([sample_id] * contour_count)
                    group_ids = np.asarray([_group_id(sample_id)] * contour_count)
                    contour_ids = np.asarray(
                        [f"{sample_id}:contour:{index}" for index in range(contour_count)]
                    )
                    labels = np.asarray(
                        [f"before_outer_step_{step_index + 1}"] * contour_count
                    )
                    _append(store, "image_id", image_ids)
                    _append(store, "group_id", group_ids)
                    _append(store, "contour_id", contour_ids)
                    _append(store, "sample_index", np.full(contour_count, dataset_index))
                    _append(store, "state_step_index", np.full(contour_count, step_index))
                    _append(store, "state_label", labels)
                    _append(
                        store,
                        "current_points",
                        current.detach().cpu().numpy().astype(np.float32),
                    )
                    _append(
                        store,
                        "fm_velocity",
                        record["fm_velocity"].detach().cpu().numpy().astype(np.float32),
                    )
                    _append(
                        store,
                        "gt_points",
                        gt.detach().cpu().numpy().astype(np.float32),
                    )
                    _append(
                        store,
                        "current_features",
                        current_features.detach().cpu().numpy().astype(np.float32),
                    )
                    _append(
                        store,
                        "richer_features",
                        richer_features.detach().cpu().numpy().astype(np.float32),
                    )
                    _append(
                        store,
                        "valid_mask",
                        np.ones((contour_count, point_count), dtype=bool),
                    )
                    _append(store, "counterfactual_mask", cf_mask)
                    for field, values in fields.items():
                        _append(store, field, values)
                print(
                    f"[{len(sample_ids)}/{len(indices)}] sample={dataset_index} "
                    f"id={sample_id} contours={context['i_it_py'].size(0)}",
                    flush=True,
                )
            except Exception as exc:
                failures.append({"sample_index": dataset_index, "error": repr(exc)})
                print(f"[!] sample={dataset_index} failed: {exc}", file=sys.stderr, flush=True)

    if not store:
        raise RuntimeError(f"No states extracted; failures={failures}")
    mapping = _finalize(store)
    all_dataset_paths = _sample_path_list(dataset)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(Path(loaded_checkpoint).resolve()),
        "checkpoint_sha256": _sha256_file(Path(loaded_checkpoint).resolve()),
        "config": str(cfg_path),
        "config_sha256": _sha256_file(cfg_path),
        "dataset": str(eval_mod.cfg.test.dataset),
        "dataset_size": int(dataset_size),
        "dataset_list_sha256": _sha256_lines(all_dataset_paths) if all_dataset_paths else None,
        "selected_list_sha256": _sha256_lines(sample_paths),
        "requested_indices": indices,
        "sample_ids": sample_ids,
        "sample_paths": sample_paths,
        "failures": failures,
        "rollout": {
            "backend": rollout["rollout_backend"],
            "deterministic": bool(rollout["deterministic"]),
            "policy_mode": rollout["effective_mode"],
            "outer_steps": int(rollout["outer_steps"]),
            "actual_ode_steps": int(rollout["actual_ode_steps"]),
            "fractions": [float(value) for value in rollout["fractions"]],
            "cached_state_step_indices": cache_step_indices,
            "cached_state_labels": [
                f"before_outer_step_{value + 1}" for value in cache_step_indices
            ],
        },
        "gt_point_semantics": "closed_polyline_vertices",
        "fm_velocity_semantics": (
            "full deterministic zero-latent ODE displacement from the cached state; "
            "the outer action is fm_velocity * fraction"
        ),
        "counterfactual": {
            "scale_delta": float(counterfactual_scale),
            "metric_delta_semantics": "metric(+scale_delta)-metric(-scale_delta)",
            "max_points_per_contour": int(counterfactual_max_points),
            "selected_points_per_contour": int(mapping["counterfactual_mask"][0].sum()),
            "reward_credit": (
                "continuous_boundary_quality_delta(plus_point, minus_point, gt_closed_polyline)"
            ),
            "reward_dist_max_px": float(
                getattr(eval_mod.cfg, "rl_v4_reward_dist_max_px", 8.0)
            ),
            "nsd_delta_px": nsd_delta_px,
        },
        "features": {
            "current_features": current_components,
            "richer_features": richer_components,
            "ctx_detail_feat_included": any(
                part["name"] == "ctx_detail_feat" for part in (richer_components or [])
            ),
        },
        "array_shapes": {key: list(value.shape) for key, value in mapping.items()},
    }

    if output is not None:
        output_path = Path(output).expanduser().resolve()
        provenance_path = (
            Path(provenance_output).expanduser().resolve()
            if provenance_output is not None
            else output_path.with_suffix(".provenance.json")
        )
        _save_cache(mapping, output_path, provenance_path)
        with provenance_path.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(provenance), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"Saved cache: {output_path}")
        print(f"Saved provenance: {provenance_path}")
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract real deterministic final-step per-point bottleneck cache."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--provenance-output", type=Path)
    parser.add_argument("--start-index", type=int, default=_env_int("START_INDEX", 0))
    parser.add_argument("--end-index", type=int, default=_env_int("END_INDEX", None))
    parser.add_argument("--max-samples", type=int, default=_env_int("MAX_SAMPLES", 1))
    parser.add_argument(
        "--cache-step-indices",
        type=_parse_indices,
        default=_parse_indices(os.environ.get("CACHE_STEP_INDICES", "4")),
        help="0-based outer-step states; default 4 means after four deterministic steps.",
    )
    parser.add_argument(
        "--counterfactual-max-points",
        type=int,
        default=_env_int("CF_MAX_POINTS_PER_CONTOUR", 16),
        help="Evenly spaced real region counterfactuals per contour; <=0 computes all points.",
    )
    parser.add_argument(
        "--counterfactual-scale",
        type=float,
        default=float(os.environ.get("CF_SCALE", "0.02")),
    )
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--gpu", type=int, default=_env_int("EVAL_GPU", None))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = (
            ROOT
            / "data/cache/perpoint_policy_bottlenecks"
            / f"step1000_finalstate_{timestamp}.npz"
        )
    extract_cache(
        cfg_path=args.config,
        checkpoint=args.checkpoint,
        start_index=args.start_index,
        end_index=args.end_index,
        max_samples=args.max_samples,
        cache_step_indices=args.cache_step_indices,
        counterfactual_max_points=args.counterfactual_max_points,
        counterfactual_scale=args.counterfactual_scale,
        seed=args.seed,
        gpu=args.gpu,
        output=args.output,
        provenance_output=args.provenance_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
