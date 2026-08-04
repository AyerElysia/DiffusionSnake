"""Self-tests for the Phase0-C oracle analysis.

These check the algebra on synthetic residuals whose correct answer is known in
closed form. The oracle numbers are going to decide which action space the 3D RL
uses, so a silent bug here would be expensive in exactly the way the 2D
FM-scaling detour was expensive.
"""
import numpy as np

import oracle_action_families as O


def make_circle(n_points=128, radius=10.0, center=(20.0, 20.0)):
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    poly = np.stack([center[0] + radius * np.cos(theta),
                     center[1] + radius * np.sin(theta)], axis=1)
    normals = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return poly, normals, theta


def test_pure_normal_residual_is_fully_captured():
    """If the error is entirely along the normal, the normal family gets 100%."""
    poly, normals, _ = make_circle()
    rng = np.random.RandomState(0)
    residual = normals * rng.normal(0, 1.0, (poly.shape[0], 1))
    value = O.family_per_point_normal(poly, normals, residual)
    assert abs(value - 1.0) < 1e-9, value
    print("ok  pure normal residual -> per_point_normal = {:.6f}".format(value))


def test_pure_tangential_residual_is_not_captured():
    """And if it is entirely tangential, the normal family gets 0%.

    This is the term that caps the 2D reference at 85.5% rather than 100%: the
    missing 14.5% is boundary-sliding that no normal action can fix (and which
    mostly does not matter, since sliding does not change the shape).
    """
    poly, normals, _ = make_circle()
    tangents = np.stack([-normals[:, 1], normals[:, 0]], axis=1)
    rng = np.random.RandomState(1)
    residual = tangents * rng.normal(0, 1.0, (poly.shape[0], 1))
    value = O.family_per_point_normal(poly, normals, residual)
    assert value < 1e-9, value
    print("ok  pure tangential residual -> per_point_normal = {:.2e}".format(value))


def test_uniform_dilation_captured_by_single_mode():
    """A uniform grow/shrink is mode 0 alone: 1 parameter should suffice."""
    poly, normals, _ = make_circle()
    residual = normals * 1.7                      # push every point out by 1.7
    r0, _ = O.family_slice_angular(poly, normals, residual, 0)
    assert abs(r0 - 1.0) < 1e-9, r0
    print("ok  uniform dilation captured by k=0 (dim 1): {:.6f}".format(r0))


def test_translation_captured_by_rigid_family():
    """A whole-contour shift is inside translate+scale, so the rigid family = 1."""
    poly, _, _ = make_circle()
    residual = np.tile(np.array([[2.0, -3.0]]), (poly.shape[0], 1))
    value, _ = O.family_slice_rigid(poly, residual)
    assert abs(value - 1.0) < 1e-9, value
    print("ok  pure translation captured by rigid t+s (dim 3): {:.6f}".format(value))


def test_angular_families_are_nested_and_monotone():
    """Higher K is a strict superset, so the ceiling can only rise with K.

    Non-monotonicity would mean the least-squares is misconditioned - worth
    catching, since high-order modes on a near-degenerate contour do go
    collinear.
    """
    poly, normals, theta = make_circle()
    rng = np.random.RandomState(2)
    residual = normals * (
        0.8 * np.cos(2 * theta) + 0.4 * np.sin(5 * theta)
        + rng.normal(0, 0.2, theta.shape))[:, None]
    values = [O.family_slice_angular(poly, normals, residual, k)[0]
              for k in (0, 1, 2, 3, 5, 8, 16)]
    for a, b in zip(values, values[1:]):
        assert b >= a - 1e-9, values
    # The signal is concentrated at modes 2 and 5, so K=5 should already be high
    # while K=1 should not be - if K=1 were high the test would be vacuous.
    assert values[1] < 0.5, values
    assert values[4] > 0.85, values
    print("ok  angular monotone k=0,1,2,3,5,8,16 -> " +
          " ".join("{:.3f}".format(v) for v in values))


def test_per_slice_upper_bounds_cylindrical():
    """The key structural invariant behind the cross-slice comparison.

    At the same angular resolution K, the cylindrical family constrains each
    angular coefficient to vary as a low-order polynomial along the slice axis,
    whereas the per-slice family lets every slice choose freely. So per-slice is
    a strict superset and must always score at least as high.

    That is exactly why the comparison in `analyse` is reported as
    (reduction, parameter count) and must never be read as "cylindrical wins
    because its number is bigger" - it cannot be bigger. The question is how
    little it gives up for how many fewer parameters.
    """
    rng = np.random.RandomState(3)
    polys, normals, residuals, positions = [], [], [], []
    for s in range(8):
        poly, normal, theta = make_circle()
        residual = normal * rng.normal(0, 1.0, (poly.shape[0], 1))
        polys.append(poly)
        normals.append(normal)
        residuals.append(residual)
        positions.append(s)
    cyl, _ = O.family_cylindrical(polys, normals, residuals, positions, 3, 2)
    per_slice, dim, _ = O.family_slice_angular_pooled(polys, normals, residuals, 3)
    assert per_slice >= cyl - 1e-9, (cyl, per_slice)
    print("ok  per-slice ({:.4f}, {} params) upper-bounds cylindrical "
          "({:.4f}, {} params)".format(
              per_slice, dim, cyl, (2 * 3 + 1) * (2 + 1)))


def test_cylindrical_is_parameter_efficient_on_axial_trend():
    """The case that would justify a 3D-native action space.

    Construct an error that is a uniform dilation whose magnitude drifts
    linearly along the slice axis - i.e. "this vertebra is progressively too fat
    towards the top". A (K=0, A=1) cylindrical family has 2 parameters for the
    whole track; the per-slice family needs 1 per slice to say the same thing.
    Both should reach ~100%, and that ratio is the whole argument.
    """
    n_slices = 10
    polys, normals, residuals, positions = [], [], [], []
    for s in range(n_slices):
        poly, normal, _ = make_circle()
        magnitude = -1.0 + 0.35 * s          # linear drift along the axis
        residuals.append(normal * magnitude)
        polys.append(poly)
        normals.append(normal)
        positions.append(s)
    cyl, _ = O.family_cylindrical(polys, normals, residuals, positions, 0, 1)
    per_slice, dim, _ = O.family_slice_angular_pooled(polys, normals, residuals, 0)
    assert cyl > 0.999, cyl
    assert per_slice > 0.999, per_slice
    print("ok  axial trend: cylindrical {:.4f} with 2 params vs per-slice "
          "{:.4f} with {} params ({:.0f}x cheaper)".format(
              cyl, per_slice, dim, dim / 2.0))


def test_cylindrical_cannot_fake_independent_noise():
    """The negative control for the test above.

    When each slice's error is independent, there is no axial structure to
    share, so a cheap cylindrical family should capture only a small fraction
    while the per-slice family captures everything. Without this control, the
    previous test could be passed by a basis that trivially spans everything.
    """
    rng = np.random.RandomState(4)
    polys, normals, residuals, positions = [], [], [], []
    for s in range(12):
        poly, normal, _ = make_circle()
        residuals.append(normal * rng.normal(0, 1.0))   # one scalar per slice
        polys.append(poly)
        normals.append(normal)
        positions.append(s)
    cyl, _ = O.family_cylindrical(polys, normals, residuals, positions, 0, 1)
    per_slice, dim, _ = O.family_slice_angular_pooled(polys, normals, residuals, 0)
    assert per_slice > 0.999, per_slice
    assert cyl < 0.6, cyl
    print("ok  independent noise: cylindrical only {:.4f} (2 params) vs "
          "per-slice {:.4f} ({} params) -> no free lunch".format(
              cyl, per_slice, dim))


def test_oracle_reduction_matches_closed_form():
    """Cross-check lstsq against the analytic projection on an orthonormal basis."""
    rng = np.random.RandomState(5)
    n = 40
    residual = rng.normal(0, 1, (n, 2))
    basis = np.zeros((n, 2, 2))
    basis[:, 0, 0] = 1.0 / np.sqrt(n)      # unit-norm constant x displacement
    basis[:, 1, 1] = 1.0 / np.sqrt(n)      # unit-norm constant y displacement
    got, rank = O.oracle_reduction(residual, basis)
    flat = residual.reshape(-1)
    b = basis.reshape(-1, 2)
    expected = float(((b.T @ flat) ** 2).sum() / (flat ** 2).sum())
    assert abs(got - expected) < 1e-12, (got, expected)
    assert rank == 2, rank
    print("ok  lstsq matches closed-form projection: {:.8f} == {:.8f}".format(
        got, expected))


def test_analyse_end_to_end(tmp="/tmp/_oracle_selftest"):
    """Smoke-test the driver on a fabricated npz with the probe's schema."""
    import os
    os.makedirs(tmp, exist_ok=True)
    rng = np.random.RandomState(6)
    polys, targets, normals, labels, slices = [], [], [], [], []
    for s in range(6):
        poly, normal, _ = make_circle()
        residual = normal * (0.5 + 0.1 * s) + rng.normal(0, 0.05, poly.shape)
        polys.append(poly)
        normals.append(normal)
        targets.append(poly + residual)
        labels.append(1)
        slices.append(s)
    path = os.path.join(tmp, "fake_volume.npz")
    np.savez_compressed(
        path,
        poly=np.stack(polys).astype(np.float32),
        gt_target=np.stack(targets).astype(np.float32),
        normal=np.stack(normals).astype(np.float32),
        gt_dist=np.zeros((6, 128), dtype=np.float32),
        label=np.asarray(labels, dtype=np.int32),
        slice_idx=np.asarray(slices, dtype=np.int32),
        score=np.ones(6, dtype=np.float32),
        n_gt_boundary=np.full(6, 400, dtype=np.int32))

    report = O.analyse([path])
    assert report["n_contours"] == 6, report
    assert report["families"]["per_point_normal"]["reduction"] > 0.9, report
    assert report["cross_slice"], report
    print("ok  analyse end-to-end: contours={} per_point_normal={:.4f} "
          "k0={:.4f} rigid={:.4f}".format(
              report["n_contours"],
              report["families"]["per_point_normal"]["reduction"],
              report["families"]["slice_angular_k0"]["reduction"],
              report["families"]["slice_rigid_t+s"]["reduction"]))
    row = report["cross_slice"][0]
    print("    cross-slice K={} A={}: cyl {:.4f} ({} params) vs per-slice "
          "{:.4f} ({:.0f} params)".format(
              row["n_angular"], row["n_axial"], row["cylindrical_reduction"],
              row["cylindrical_params_per_track"], row["per_slice_reduction"],
              row["per_slice_params_per_track_mean"]))


def test_duplicate_slice_indices_do_not_crash(tmp="/tmp/_oracle_dupetest"):
    """Regression: two contours of the same label in the same slice.

    Real data has this - sub-verse011 dumps 554 contours over 38 slices for 12
    labels, so a label routinely owns several contours in one slice (a vertebra
    can appear as disjoint pieces in a sagittal cut). Sorting track entries
    without an explicit key then compares the ndarray payloads and raises.
    """
    import os
    os.makedirs(tmp, exist_ok=True)
    polys, targets, normals, labels, slices = [], [], [], [], []
    for s in [0, 0, 1, 1, 2, 3]:            # note the repeated slice indices
        poly, normal, _ = make_circle()
        polys.append(poly)
        normals.append(normal)
        targets.append(poly + normal * 0.7)
        labels.append(1)
        slices.append(s)
    path = os.path.join(tmp, "dupe.npz")
    np.savez_compressed(
        path,
        poly=np.stack(polys).astype(np.float32),
        gt_target=np.stack(targets).astype(np.float32),
        normal=np.stack(normals).astype(np.float32),
        gt_dist=np.zeros((6, 128), dtype=np.float32),
        label=np.asarray(labels, dtype=np.int32),
        slice_idx=np.asarray(slices, dtype=np.int32),
        score=np.ones(6, dtype=np.float32),
        n_gt_boundary=np.full(6, 400, dtype=np.int32))
    report = O.analyse([path])       # must not raise
    assert report["cross_slice"], report
    print("ok  duplicate slice indices handled: {} tracks, cyl={:.4f}".format(
        report["cross_slice"][0]["n_tracks"],
        report["cross_slice"][0]["cylindrical_reduction"]))


if __name__ == "__main__":
    test_pure_normal_residual_is_fully_captured()
    test_pure_tangential_residual_is_not_captured()
    test_uniform_dilation_captured_by_single_mode()
    test_translation_captured_by_rigid_family()
    test_angular_families_are_nested_and_monotone()
    test_per_slice_upper_bounds_cylindrical()
    test_cylindrical_is_parameter_efficient_on_axial_trend()
    test_cylindrical_cannot_fake_independent_noise()
    test_oracle_reduction_matches_closed_form()
    test_analyse_end_to_end()
    test_duplicate_slice_indices_do_not_crash()
    print("\nall oracle tests passed")
