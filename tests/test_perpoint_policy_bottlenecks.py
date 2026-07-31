import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


_MODULE_PATH = (
    Path(__file__).parents[1] / "scripts/analyze_perpoint_policy_bottlenecks.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "analyze_perpoint_policy_bottlenecks", _MODULE_PATH
)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

SCHEMA_VERSION = _MODULE.SCHEMA_VERSION
aggregate_probe_records = _MODULE.aggregate_probe_records
analyze_reward_credit = _MODULE.analyze_reward_credit
build_parser = _MODULE.build_parser
compute_geometry_oracles = _MODULE.compute_geometry_oracles
grouped_split_masks = _MODULE.grouped_split_masks
load_cache = _MODULE.load_cache
project_to_closed_polylines = _MODULE.project_to_closed_polylines
run_analysis = _MODULE.run_analysis
run_supervised_probes = _MODULE.run_supervised_probes
validate_cache = _MODULE.validate_cache
zero_mean_bounded_scale = _MODULE.zero_mean_bounded_scale


def make_synthetic_mapping(contours=12, points=6):
    rng = np.random.default_rng(7)
    angles = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    base = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    current = np.stack(
        [base + np.array([4.0 * index, 0.25 * index]) for index in range(contours)]
    )
    current_features = rng.normal(size=(contours, points, 4))
    richer_features = np.concatenate(
        (current_features, np.sin(current_features), np.cos(current_features)), axis=-1
    )
    velocity = np.empty_like(current)
    velocity[..., 0] = 1.0 + 0.1 * current_features[..., 1]
    velocity[..., 1] = 0.4 + 0.05 * current_features[..., 2]
    pure = current + velocity
    scale = 0.08 * np.tanh(current_features[..., 0])
    normals = _MODULE.contour_normals(pure)
    normal_extra = 0.03 * current_features[..., 3]
    residual = velocity * scale[..., None] + normals * normal_extra[..., None]
    gt = pure + residual
    reward = scale + 0.01 * current_features[..., 2]
    return {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "gt_point_semantics": np.asarray("closed_polyline_vertices"),
        "image_id": np.asarray([f"image-{index}" for index in range(contours)]),
        "group_id": np.asarray([f"group-{index // 2}" for index in range(contours)]),
        "contour_id": np.asarray([f"contour-{index}" for index in range(contours)]),
        "current_points": current,
        "fm_velocity": velocity,
        "gt_points": gt,
        "current_features": current_features,
        "richer_features": richer_features,
        "valid_mask": np.ones((contours, points), dtype=bool),
        "counterfactual_mask": np.ones((contours, points), dtype=bool),
        "reward_credit": reward,
        "delta_iou": reward.copy(),
        "delta_dice": -reward,
        "delta_mboundf": 0.5 * reward,
        "delta_nsd": reward + 0.001 * rng.normal(size=reward.shape),
    }


def test_validate_cache_rejects_schema_and_accepts_repeated_contour_clusters():
    mapping = make_synthetic_mapping()
    mapping["schema_version"] = np.asarray("future.v2")
    with pytest.raises(ValueError, match="unsupported schema_version"):
        validate_cache(mapping)

    mapping = make_synthetic_mapping()
    mapping["contour_id"][1] = mapping["contour_id"][0]
    cache = validate_cache(mapping)
    assert cache.contour_id[1] == cache.contour_id[0]

    mapping = make_synthetic_mapping()
    mapping["gt_point_semantics"] = np.asarray("point_aligned")
    with pytest.raises(ValueError, match="unsupported gt_point_semantics"):
        validate_cache(mapping)


def test_nearest_closed_segment_projection_avoids_index_correspondence():
    points = np.asarray([[[0.5, -0.25], [1.25, 0.5], [0.5, 1.25], [-0.25, 0.5]]])
    # Deliberately rotate the GT vertex order; nearest boundary projection is invariant.
    polyline = np.asarray([[[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]])

    projected = project_to_closed_polylines(points, polyline)

    expected = np.asarray([[[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]]])
    np.testing.assert_allclose(projected, expected)


def test_npz_and_pt_cache_loading(tmp_path):
    mapping = make_synthetic_mapping(contours=4, points=5)
    npz_path = tmp_path / "cache.npz"
    np.savez_compressed(npz_path, **mapping)

    from_npz = load_cache(npz_path)

    assert from_npz.current_points.shape == (4, 5, 2)
    assert from_npz.current_features.shape[-1] == 4

    torch = pytest.importorskip("torch")
    pt_path = tmp_path / "cache.pt"
    torch_mapping = {}
    for key, value in mapping.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "fbi":
            torch_mapping[key] = torch.from_numpy(value)
        elif isinstance(value, np.ndarray) and value.ndim == 0:
            torch_mapping[key] = value.item()
        elif isinstance(value, np.ndarray):
            torch_mapping[key] = value.tolist()
        else:
            torch_mapping[key] = value
    torch.save({"cache": torch_mapping}, pt_path)

    from_pt = load_cache(pt_path)

    np.testing.assert_allclose(from_pt.gt_points, from_npz.gt_points)


def test_geometry_oracles_form_expected_capacity_ladder():
    mapping = make_synthetic_mapping(contours=4, points=8)
    cache = validate_cache(mapping)
    geometry = compute_geometry_oracles(
        cache, current_scale_bound=0.02, wider_scale_bound=0.10
    )
    rows = {row["oracle"]: row for row in geometry["rows"]}

    assert rows["scale_wider"]["error_mean"] < rows["scale_current"]["error_mean"]
    assert rows["scale_unbounded"]["error_mean"] <= rows["scale_wider"]["error_mean"]
    assert rows["normal_residual"]["error_mean"] < rows["pure_fm"]["error_mean"]
    assert rows["residual_2d"]["error_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert rows["scale_current"]["scale_saturation_fraction"] > 0.0


def test_zero_mean_oracle_enforces_contour_constraint():
    velocity = np.ones((2, 5, 2), dtype=np.float64)
    target = np.zeros_like(velocity)
    target[0, :, 0] = np.asarray([2.0, 1.0, -0.5, -1.0, 0.25])
    target[1, :, 0] = 3.0
    valid = np.ones((2, 5), dtype=bool)

    scale = zero_mean_bounded_scale(velocity, target, valid, bound=0.1)

    np.testing.assert_allclose(scale.mean(axis=1), 0.0, atol=1e-9)
    assert np.max(np.abs(scale)) <= 0.1 + 1e-12


def test_grouped_split_has_no_cluster_leakage():
    ids = np.repeat(np.asarray([f"image-{index}" for index in range(10)]), 4)
    train, validation, test = grouped_split_masks(ids, seed=13)

    train_ids = set(ids[train])
    validation_ids = set(ids[validation])
    test_ids = set(ids[test])
    assert train.any() and validation.any() and test.any()
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)


def test_supervised_probes_cover_models_features_splits_and_three_seeds():
    cache = validate_cache(make_synthetic_mapping(contours=12, points=6))
    geometry = compute_geometry_oracles(cache)

    bootstrap = []
    rows = run_supervised_probes(
        cache,
        geometry,
        seeds=(3, 5, 7),
        epochs=2,
        batch_size=64,
        patience=1,
        bootstrap_records=bootstrap,
        bootstrap_samples=3,
    )
    aggregate = aggregate_probe_records(rows)

    assert {row["model"] for row in rows} == {"linear", "h64", "h256"}
    assert {row["feature_set"] for row in rows} == {"current", "richer"}
    assert {row["split"] for row in rows} == {"image", "group"}
    assert {row["seed"] for row in rows} == {3, 5, 7}
    assert {row["target"] for row in rows} == {
        "scale_current",
        "normal_residual",
        "residual_2d",
    }
    assert len(rows) == 2 * 3 * 2 * 3 * 3
    assert all(row["seeds"] == 3 for row in aggregate)
    assert {row["cluster_unit"] for row in bootstrap} == {"image", "contour"}
    assert {row["statistic"] for row in bootstrap} == {"r2", "rmse", "mae"}
    assert all(row["bootstrap_samples"] == 3 for row in bootstrap)


def test_probe_cli_defaults_filters_and_strict_validation():
    parser = build_parser()

    defaults = parser.parse_args(["--print-schema"])
    assert defaults.probe_feature_sets == ("current", "richer")
    assert defaults.probe_targets == ("scale_current", "normal_residual", "residual_2d")
    assert defaults.probe_splits == ("image", "group")
    assert defaults.probe_models == ("linear", "h64", "h256")

    selected = parser.parse_args(
        [
            "--print-schema",
            "--probe-feature-sets",
            "current,richer",
            "--probe-targets",
            "reward_credit",
            "--probe-splits",
            "image",
            "--probe-models",
            "linear,h64,h256",
        ]
    )
    assert selected.probe_feature_sets == ("current", "richer")
    assert selected.probe_targets == ("reward_credit",)
    assert selected.probe_splits == ("image",)
    assert selected.probe_models == ("linear", "h64", "h256")

    with pytest.raises(SystemExit):
        parser.parse_args(["--probe-targets", "scale_current,unknown"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--probe-models", "linear,linear"])


def test_supervised_probe_filters_and_analysis_summary(tmp_path):
    cache = validate_cache(make_synthetic_mapping(contours=12, points=6))
    geometry = compute_geometry_oracles(cache)
    selection = {
        "probe_feature_sets": ("richer",),
        "probe_targets": ("scale_current",),
        "probe_splits": ("image",),
        "probe_models": ("linear",),
    }

    rows = run_supervised_probes(
        cache,
        geometry,
        seeds=(3, 5, 7),
        **selection,
    )

    assert len(rows) == 3
    assert {row["feature_set"] for row in rows} == {"richer"}
    assert {row["target"] for row in rows} == {"scale_current"}
    assert {row["split"] for row in rows} == {"image"}
    assert {row["model"] for row in rows} == {"linear"}
    assert {row["seed"] for row in rows} == {3, 5, 7}

    summary = run_analysis(
        cache,
        tmp_path / "filtered",
        seeds=(3, 5, 7),
        bootstrap_samples=0,
        no_plots=True,
        **selection,
    )
    assert summary["config"]["probe_feature_sets"] == ["richer"]
    assert summary["config"]["probe_targets"] == ["scale_current"]
    assert summary["config"]["probe_splits"] == ["image"]
    assert summary["config"]["probe_models"] == ["linear"]
    assert len(summary["probe_aggregate"]) == 1

    with pytest.raises(ValueError, match="unsupported probe models"):
        run_supervised_probes(
            cache,
            geometry,
            probe_models=("unknown",),
        )


def test_reward_credit_probe_uses_real_finite_counterfactual_points(tmp_path):
    mapping = make_synthetic_mapping(contours=12, points=6)
    counterfactual_mask = np.zeros((12, 6), dtype=bool)
    counterfactual_mask[:, ::2] = True
    mapping["counterfactual_mask"] = counterfactual_mask
    mapping["reward_credit"] = (
        2.0 * mapping["current_features"][..., 0]
        - mapping["current_features"][..., 2]
    )
    mapping["reward_credit"][~counterfactual_mask] = np.nan
    mapping["current_features"][0, 2, 0] = np.nan
    cache = validate_cache(mapping)
    geometry = compute_geometry_oracles(cache)
    selection = {
        "probe_feature_sets": ("current",),
        "probe_targets": ("reward_credit",),
        "probe_splits": ("image", "group"),
        "probe_models": ("linear",),
    }

    rows = run_supervised_probes(
        cache,
        geometry,
        seeds=(3, 5, 7),
        **selection,
    )

    assert len(rows) == 3
    assert {row["target"] for row in rows} == {"reward_credit"}
    assert {row["split"] for row in rows} == {"image"}
    assert all(
        row["train_n"] + row["validation_n"] + row["test_n"] == 35
        for row in rows
    )
    assert all(row["r2"] > 0.99 for row in rows)
    assert all("balanced_sign_agreement" in row for row in rows)

    summary = run_analysis(
        cache,
        tmp_path / "reward-credit",
        seeds=(3, 5, 7),
        bootstrap_samples=0,
        no_plots=True,
        **selection,
    )
    assert summary["cache"]["counterfactual_points"] == 36
    assert summary["config"]["probe_target_masks"] == {
        "reward_credit": "counterfactual_mask_and_finite"
    }
    assert summary["config"]["probe_target_splits"] == {
        "reward_credit": ["image"]
    }


def test_reward_credit_probe_requires_cache_fields():
    mapping = make_synthetic_mapping()
    mapping.pop("reward_credit")
    for field in _MODULE.DELTA_FIELDS:
        mapping.pop(field)
    cache = validate_cache(mapping)
    geometry = compute_geometry_oracles(cache)
    with pytest.raises(ValueError, match="cache is missing reward_credit"):
        run_supervised_probes(
            cache,
            geometry,
            probe_targets=("reward_credit",),
            probe_splits=("image",),
        )

    mapping = make_synthetic_mapping()
    mapping.pop("counterfactual_mask")
    cache = validate_cache(mapping)
    geometry = compute_geometry_oracles(cache)
    with pytest.raises(ValueError, match="cache is missing counterfactual_mask"):
        run_supervised_probes(
            cache,
            geometry,
            probe_targets=("reward_credit",),
            probe_splits=("image",),
        )


def test_reward_credit_correlations_and_sign_agreement():
    cache = validate_cache(make_synthetic_mapping())

    rows, bootstrap = analyze_reward_credit(cache, bootstrap_samples=20, seed=11)
    by_metric = {row["metric"]: row for row in rows}

    assert by_metric["iou"]["pearson"] == pytest.approx(1.0)
    assert by_metric["iou"]["spearman"] == pytest.approx(1.0)
    assert by_metric["iou"]["sign_agreement"] == pytest.approx(1.0)
    assert by_metric["dice"]["pearson"] == pytest.approx(-1.0)
    assert by_metric["dice"]["sign_agreement"] == pytest.approx(0.0)
    assert {row["cluster_unit"] for row in bootstrap} == {"image", "contour"}
    assert all(row["bootstrap_valid"] > 0 for row in bootstrap)


def test_run_analysis_writes_json_csv_png_without_overwrite(tmp_path):
    cache = validate_cache(make_synthetic_mapping())
    output = tmp_path / "diagnostic"

    summary = run_analysis(
        cache,
        output,
        bootstrap_samples=20,
        skip_probes=True,
    )

    expected = {
        "summary.json",
        "cache_schema.json",
        "oracle_metrics.csv",
        "probe_metrics.csv",
        "reward_credit_metrics.csv",
        "bootstrap_metrics.csv",
        "oracle_errors.png",
        "reward_credit_alignment.png",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    assert summary["safety"].startswith("gt_points")
    assert (output / "oracle_errors.png").read_bytes().startswith(b"\x89PNG")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_analysis(cache, output, bootstrap_samples=5, skip_probes=True)
