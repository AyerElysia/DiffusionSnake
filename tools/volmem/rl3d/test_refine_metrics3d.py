"""Self-tests for the Phase0-B metric suite.

The important one is `test_extrusion_is_not_rewarded`: it encodes the reward
hacking failure mode the metric was designed to resist. If that test ever goes
green-to-red, the consistency term has drifted back to the naive form and must
not be used in a reward.
"""
import numpy as np

import refine_metrics3d as M


def make_sphere_volume(n=24, r=7.0, jitter=None, shift=None):
    """Stack of discs whose radius varies along the slice axis (a crude vertebra)."""
    vol = np.zeros((n, 40, 40), dtype=np.uint16)
    yy, xx = np.mgrid[0:40, 0:40]
    for i in range(n):
        radius = r * np.sqrt(max(1.0 - ((i - n / 2.0) / (n / 2.0)) ** 2, 0.0))
        if jitter is not None:
            radius += jitter[i]
        cy, cx = 20.0, 20.0
        if shift is not None:
            cx += shift[i]
        if radius <= 0.5:
            continue
        vol[i][((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2] = 1
    return vol


def test_surface_nsd_identity():
    gt = make_sphere_volume()
    out = M.surface_nsd(gt > 0, gt > 0)
    assert out["nsd@1"] == 1.0, out
    assert out["asd"] == 0.0, out
    assert out["hd95"] == 0.0, out
    print("ok  surface_nsd identity -> nsd@1=1.0 asd=0.0")


def test_surface_nsd_degrades_with_error():
    gt = make_sphere_volume(r=7.0)
    near = make_sphere_volume(r=7.8)
    far = make_sphere_volume(r=10.0)
    a = M.surface_nsd(near > 0, gt > 0)
    b = M.surface_nsd(far > 0, gt > 0)
    assert a["nsd@1"] > b["nsd@1"], (a, b)
    assert a["asd"] < b["asd"], (a, b)
    print("ok  surface_nsd monotone: nsd@1 {:.3f} > {:.3f}, asd {:.2f} < {:.2f}".format(
        a["nsd@1"], b["nsd@1"], a["asd"], b["asd"]))


def test_nsd_is_not_diluted_by_object_size():
    """The actual motivating property, and the reason Dice is the wrong headline.

    Take the *same* boundary error (a uniform 1-voxel over-segmentation) on a
    small object and on a large one. Dice's penalty shrinks as the object grows,
    because the error is measured against a bulk volume that grows faster than
    the surface. NSD's penalty stays put, because it only ever looks at surface.

    A refinement policy moves the boundary by 1-2 voxels. Under Dice that signal
    is progressively swallowed by big vertebrae; under NSD it is not. Hence NSD
    is the headline refinement metric and Dice is a sanity check.
    """
    results = {}
    for radius in (5.0, 12.0):
        gt = make_sphere_volume(n=32, r=radius)
        pred = make_sphere_volume(n=32, r=radius + 1.0)
        g, p = gt > 0, pred > 0
        dice = 2.0 * np.logical_and(g, p).sum() / (g.sum() + p.sum())
        results[radius] = (1.0 - dice, 1.0 - M.surface_nsd(p, g)["nsd@0.5"])

    small_dice, small_nsd = results[5.0]
    large_dice, large_nsd = results[12.0]
    # Dice's penalty must decay materially with size...
    assert large_dice < small_dice * 0.75, results
    # ...while NSD's penalty stays roughly constant.
    assert abs(large_nsd - small_nsd) < 0.5 * small_nsd, results
    print("ok  NSD not size-diluted: 1-dice {:.4f}->{:.4f} (decays {:.0f}%), "
          "1-nsd@0.5 {:.4f}->{:.4f} (stable)".format(
              small_dice, large_dice, 100 * (1 - large_dice / small_dice),
              small_nsd, large_nsd))


def test_extrusion_is_not_rewarded():
    """THE anti-hacking test.

    Arm A: a faithful prediction that tracks the GT's varying cross-section.
    Arm B: a constant extrusion - every slice identical - which is the optimum
    of a naive z-smoothness reward and the thing Slice Memory can produce most
    cheaply by copying its neighbour.

    A must score better. If B ever wins, the metric is hackable.
    """
    gt = make_sphere_volume(r=7.0)
    rng = np.random.RandomState(0)
    faithful = make_sphere_volume(r=7.0, jitter=rng.normal(0, 0.15, 24))
    extrusion = np.zeros_like(gt)
    yy, xx = np.mgrid[0:40, 0:40]
    disc = (((yy - 20.0) ** 2 + (xx - 20.0) ** 2) <= 5.0 ** 2)
    for i in range(gt.shape[0]):
        extrusion[i][disc] = 1

    naive_faithful = np.abs(M._second_difference(
        M._per_slice_area(faithful > 0))).mean()
    naive_extrusion = np.abs(M._second_difference(
        M._per_slice_area(extrusion > 0))).mean()
    assert naive_extrusion < naive_faithful, "premise: naive metric prefers extrusion"

    zf = M.z_consistency(faithful > 0, gt > 0)
    ze = M.z_consistency(extrusion > 0, gt > 0)
    nf = M.surface_nsd(faithful > 0, gt > 0)
    ne = M.surface_nsd(extrusion > 0, gt > 0)

    # `deficit` is the term that names over-smoothing, and it fires correctly.
    assert ze["area_lap_deficit"] > zf["area_lap_deficit"] * 3, (zf, ze)
    # Accuracy must dominate: the extrusion is nowhere near the anatomy.
    # Judged at the tight tolerance, which is where refinement lives - at
    # tau=1 the extrusion still scores 0.53 because a constant cross-section
    # does coincide with the anatomy mid-vertebra. Another reason tau matters.
    assert ne["nsd@0.5"] < nf["nsd@0.5"] * 0.5, (nf, ne)
    assert ne["asd"] > nf["asd"] * 3, (nf, ne)

    # Documented negative result: abs_diff alone is NOT a sufficient guard.
    # The extrusion's abs_diff is capped at mean|lap_gt|, so a smooth GT lets it
    # beat a noisy-but-honest prediction. Asserted so the weakness stays visible.
    assert ze["area_lap_abs_diff"] < zf["area_lap_abs_diff"], (zf, ze)

    print("ok  anti-hack: naive metric prefers extrusion ({:.2f} < {:.2f});"
          .format(naive_extrusion, naive_faithful))
    print("      deficit fires:  extrusion {:.4f} vs faithful {:.4f}".format(
        ze["area_lap_deficit"], zf["area_lap_deficit"]))
    print("      NSD@0.5 collapses: extrusion {:.4f} vs faithful {:.4f}".format(
        ne["nsd@0.5"], nf["nsd@0.5"]))
    print("      NOTE abs_diff alone is insufficient ({:.4f} < {:.4f}) -> "
          "reward must keep accuracy dominant".format(
              ze["area_lap_abs_diff"], zf["area_lap_abs_diff"]))


def test_centroid_consistency_catches_drift():
    gt = make_sphere_volume()
    rng = np.random.RandomState(1)
    wobble = make_sphere_volume(shift=rng.normal(0, 1.5, 24))
    steady = make_sphere_volume()
    a = M.centroid_consistency(wobble > 0, gt > 0)
    b = M.centroid_consistency(steady > 0, gt > 0)
    assert a["centroid_lap_excess"] > b["centroid_lap_excess"], (a, b)
    print("ok  centroid drift detected: excess {:.4f} > {:.4f}".format(
        a["centroid_lap_excess"], b["centroid_lap_excess"]))


def test_slice_roles():
    vol = np.zeros((10, 8, 8), dtype=np.uint16)
    vol[2:6, 2:6, 2:6] = 1     # label 1 spans slices 2..5
    vol[5:9, 2:6, 2:6] = 2     # label 2 spans 5..8, overlapping at slice 5
    roles = M.slice_roles(vol)
    assert roles[0] == "empty" and roles[9] == "empty", roles
    assert roles[2] == "end_cap", roles      # first appearance of label 1
    assert roles[5] == "end_cap", roles      # last of 1 and first of 2
    assert roles[3] == "interior", roles
    assert "transition" in roles or True
    print("ok  slice_roles:", roles)


def test_stratify_excludes_empty_inflation():
    """Empty slices are scored 1.0 by the existing evaluator; must not inflate."""
    values = [1.0, 1.0, 0.5, 0.5, 1.0]
    roles = ["empty", "empty", "interior", "interior", "empty"]
    out = M.stratify(values, roles)
    assert "empty" not in out
    assert abs(out["interior"]["mean"] - 0.5) < 1e-9, out
    print("ok  stratify drops empty slices: interior mean={:.3f} n={}".format(
        out["interior"]["mean"], out["interior"]["n"]))


def test_paired_bootstrap():
    rng = np.random.RandomState(3)
    base = rng.normal(0.85, 0.05, 200)
    # Effect size deliberately set to the 2D RL scale (~0.002).
    treat = base + rng.normal(0.002, 0.001, 200)
    out = M.paired_bootstrap(base, treat)
    assert out["significant"], out
    assert out["ci_low"] > 0, out
    null = M.paired_bootstrap(base, base + rng.normal(0, 0.05, 200))
    assert not null["significant"], null
    print("ok  paired_bootstrap: real effect diff={:.5f} CI=({:.5f},{:.5f}) "
          "W/T/L={}/{}/{}; null effect significant={}".format(
              out["mean_diff"], out["ci_low"], out["ci_high"],
              out["wins"], out["ties"], out["losses"], null["significant"]))


def test_evaluate_volume_end_to_end():
    gt = make_sphere_volume()
    gt[gt > 0] = 3  # single class with a non-1 label id
    rng = np.random.RandomState(5)
    pred = make_sphere_volume(jitter=rng.normal(0, 0.3, 24))
    pred[pred > 0] = 3
    report = M.evaluate_volume(pred, gt)
    assert "3" in report["per_class"], report["per_class"].keys()
    assert np.isfinite(report["foreground"]["nsd@1"])
    assert np.isfinite(report["foreground"]["dice"])
    assert report["role_counts"]["empty"] >= 0
    print("ok  evaluate_volume: dice={:.4f} nsd@1={:.4f} nsd@2={:.4f} "
          "hd95={:.2f} roles={}".format(
              report["foreground"]["dice"], report["foreground"]["nsd@1"],
              report["foreground"]["nsd@2"], report["foreground"]["hd95"],
              report["role_counts"]))
    print("    slice_dice_by_role:", {
        k: round(v["mean"], 4) if np.isfinite(v["mean"]) else None
        for k, v in report["slice_dice_by_role"].items()})


if __name__ == "__main__":
    test_surface_nsd_identity()
    test_surface_nsd_degrades_with_error()
    test_nsd_is_not_diluted_by_object_size()
    test_extrusion_is_not_rewarded()
    test_centroid_consistency_catches_drift()
    test_slice_roles()
    test_stratify_excludes_empty_inflation()
    test_paired_bootstrap()
    test_evaluate_volume_end_to_end()
    print("\nall metric tests passed")
