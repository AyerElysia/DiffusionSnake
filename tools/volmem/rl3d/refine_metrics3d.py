"""Phase0-B: 3D refinement metrics for the VolMem route.

Why this file exists
--------------------
Everything currently in `lib/evaluators/` and `tools/volmem/` scores a volume by
stacking independent 2D predictions and counting voxels (Dice / IoU). Two
consequences, both fatal for 3D RL:

  1. Voxel Dice is dominated by the bulk interior. A refinement policy moves the
     boundary by 1-2 px; that is well inside Dice's noise floor. The 2D work
     already hit this - RL gains were ~0.002 IoU and only visible under paired
     bootstrap.
  2. Nothing in the codebase can *see* cross-slice coherence. A volume made of
     40 individually-plausible but mutually-inconsistent slices scores the same
     as a smooth one. So neither the optimiser nor the report card can reward
     the Slice Memory for doing its job - which is the leading explanation for
     why memory is currently inert (feeding it GT evidence scores -0.000243).

This module supplies the missing measurements. It is read-only: no training
loop imports it, and nothing here defines a reward. Reward design comes later
and must be reviewed first; these are the *metrics* the reward will be
validated against.

Three families
-------------
`surface_nsd`      True 3D surface distance, not per-slice. Boundary voxels are
                   extracted from the volume, so the axis-normal direction is
                   measured too. This is the headline refinement metric:
                   sensitive to 1px boundary moves, insensitive to bulk volume.

`z_consistency`    Cross-slice coherence, in GT-relative form. See the long
                   comment on `z_consistency` - the naive version is trivially
                   hackable and must not be used.

`stratify`         Splits every metric by slice role (interior / end-cap /
                   transition). A refinement that helps on interior slices while
                   degrading end-caps is memory copying its neighbour, not
                   learning anatomy. Aggregate numbers hide exactly that.

Plus `paired_bootstrap`, because at these effect sizes an unpaired point
estimate is meaningless.

Spacing note
------------
The sagittal_2d_fixed manifest carries no physical voxel spacing, so all
distances are in *voxels* and NSD thresholds are voxel thresholds. That is fine
for A/B comparison (both arms use the same grid) but the numbers are not
comparable to published mm-based NSD. Anything reported externally needs
spacing recovered from the source NIfTI first.
"""
import json
import os

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------
# Surface extraction
# --------------------------------------------------------------------------

def surface_voxels(mask):
    """Boundary voxels of a binary volume, as an [N,3] float array.

    A voxel is on the surface if it is foreground and at least one 6-neighbour
    is background. Using 6-connectivity (rather than 26) keeps the surface thin,
    which matters because a thick surface biases NSD optimistically.

    The volume is padded before erosion so that foreground touching the array
    edge is treated as a real surface rather than silently closed off - VerSe
    vertebrae routinely run off the end of the cropped field of view.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros((0, 3), dtype=np.float32)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    eroded = ndimage.binary_erosion(padded, structure=structure)
    boundary = padded & ~eroded
    coords = np.argwhere(boundary[1:-1, 1:-1, 1:-1])
    return coords.astype(np.float32)


def surface_distances(pred_mask, gt_mask):
    """Symmetric surface distances in voxels: (pred->gt, gt->pred).

    Both directions are returned rather than a single summary because they mean
    different things: pred->gt penalises spurious surface (over-segmentation),
    gt->pred penalises missed surface (under-segmentation). A refinement policy
    can trade one against the other, and collapsing them early hides that.
    """
    pred_pts = surface_voxels(pred_mask)
    gt_pts = surface_voxels(gt_mask)
    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty
    d_pred = cKDTree(gt_pts).query(pred_pts, k=1)[0].astype(np.float32)
    d_gt = cKDTree(pred_pts).query(gt_pts, k=1)[0].astype(np.float32)
    return d_pred, d_gt


def surface_nsd(pred_mask, gt_mask, taus=(0.5, 1.0, 2.0, 3.0)):
    """Normalized Surface Dice at several voxel tolerances.

    NSD@tau = fraction of surface (both directions pooled) that lies within tau
    of the other surface. Reported at multiple tau because the right tolerance
    is an empirical question: too tight and it is pure label noise, too loose
    and it saturates. Phase 0 picks tau by looking at where the metric actually
    separates checkpoints.

    Also returns mean/95th-percentile surface distance and Hausdorff95, which
    are the standard companions and catch the case where NSD looks fine but a
    small region is catastrophically wrong.
    """
    d_pred, d_gt = surface_distances(pred_mask, gt_mask)
    result = {
        "n_surface_pred": int(d_pred.shape[0]),
        "n_surface_gt": int(d_gt.shape[0]),
    }
    if d_pred.shape[0] == 0 or d_gt.shape[0] == 0:
        for tau in taus:
            result["nsd@{:g}".format(tau)] = float("nan")
        result.update({"asd": float("nan"), "hd95": float("nan")})
        return result
    pooled = np.concatenate([d_pred, d_gt])
    for tau in taus:
        result["nsd@{:g}".format(tau)] = float((pooled <= tau).mean())
    result["asd"] = float(pooled.mean())
    result["hd95"] = float(np.percentile(pooled, 95))
    return result


# --------------------------------------------------------------------------
# Cross-slice consistency
# --------------------------------------------------------------------------

def _per_slice_area(mask, axis=0):
    mask = np.asarray(mask, dtype=bool)
    other = tuple(i for i in range(mask.ndim) if i != axis)
    return mask.sum(axis=other).astype(np.float64)


def _second_difference(series):
    """Discrete Laplacian along the slice axis, interior samples only."""
    series = np.asarray(series, dtype=np.float64)
    if series.shape[0] < 3:
        return np.zeros((0,), dtype=np.float64)
    return series[2:] - 2.0 * series[1:-1] + series[:-2]


def z_consistency(pred_mask, gt_mask, axis=0):
    """GT-relative cross-slice smoothness. Read this before using it.

    The obvious formulation - reward low curvature of the predicted contour
    stack along the slice axis - is **reward hacking bait**. Its global optimum
    is a constant extrusion: make every slice identical and the smoothness term
    is perfect. Slice Memory is *especially* well placed to exploit this,
    because copying the previous slice is exactly what a memory read can do
    cheaply. Optimising the naive term would therefore produce a model that
    looks like it finally "uses memory" while being anatomically worse.

    So consistency is measured **relative to the GT's own roughness**, the same
    trick the 2D work used for the burr penalty (`|lap_pred| - |lap_gt|`).
    Anatomy has real curvature along the spine; the target is to match it, not
    to minimise it. Under this form, a constant extrusion scores badly because
    GT curvature is nonzero.

    `area_lap_excess` > 0 means the prediction wobbles more than the anatomy.
    `area_lap_deficit` > 0 means it is over-smoothed - flagged separately
    precisely so the extrusion failure mode is visible rather than rewarded.

    IMPORTANT - do not use this term standalone, and do not use `abs_diff` as
    the anti-hack guard. Empirically (see test_extrusion_is_not_rewarded) a
    constant extrusion scores `abs_diff` = mean|lap_gt|, which is *bounded and
    small* whenever the GT is smooth, while any realistically noisy prediction
    can exceed it. So `abs_diff` alone can still rank the extrusion first.

    What actually defeats the extrusion is the pair (surface NSD, deficit):
      - NSD collapses, because a constant cross-section does not match the
        anatomy anywhere except mid-vertebra;
      - `area_lap_deficit` rises, which names the failure explicitly.
    Any reward built on this must therefore keep an accuracy term dominant and
    use consistency only as a shaping/diagnostic term. `deficit` is the signal
    to watch for over-smoothing; `excess` for jitter.
    """
    pred_area = _per_slice_area(pred_mask, axis=axis)
    gt_area = _per_slice_area(gt_mask, axis=axis)
    valid = gt_area > 0
    if valid.sum() < 3:
        return {"area_lap_excess": float("nan"),
                "area_lap_deficit": float("nan"),
                "area_lap_abs_diff": float("nan"),
                "n_interior": 0}

    # Normalise by mean area so volumes of different size are comparable.
    scale = max(gt_area[valid].mean(), 1.0)
    lap_pred = np.abs(_second_difference(pred_area / scale))
    lap_gt = np.abs(_second_difference(gt_area / scale))
    n = min(lap_pred.shape[0], lap_gt.shape[0])
    lap_pred, lap_gt = lap_pred[:n], lap_gt[:n]
    diff = lap_pred - lap_gt
    return {
        "area_lap_excess": float(np.clip(diff, 0, None).mean()),
        "area_lap_deficit": float(np.clip(-diff, 0, None).mean()),
        "area_lap_abs_diff": float(np.abs(diff).mean()),
        "n_interior": int(n),
    }


def centroid_consistency(pred_mask, gt_mask, axis=0):
    """Same GT-relative logic applied to the per-slice centroid track.

    Area alone is blind to a contour that drifts sideways while keeping its
    size. Centroid drift is the other half of cross-slice coherence, and it is
    the one a slice-sequential model is most likely to get wrong at vertebra
    transitions.
    """
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    n_slices = pred_mask.shape[axis]
    pred_c, gt_c, keep = [], [], []
    for i in range(n_slices):
        p = np.take(pred_mask, i, axis=axis)
        g = np.take(gt_mask, i, axis=axis)
        if g.sum() == 0 or p.sum() == 0:
            continue
        pred_c.append(ndimage.center_of_mass(p))
        gt_c.append(ndimage.center_of_mass(g))
        keep.append(i)
    if len(keep) < 3:
        return {"centroid_lap_excess": float("nan"),
                "centroid_drift": float("nan"), "n_interior": 0}
    pred_c = np.asarray(pred_c, dtype=np.float64)
    gt_c = np.asarray(gt_c, dtype=np.float64)
    lap_pred = np.linalg.norm(
        np.stack([_second_difference(pred_c[:, k]) for k in range(2)], axis=1),
        axis=1)
    lap_gt = np.linalg.norm(
        np.stack([_second_difference(gt_c[:, k]) for k in range(2)], axis=1),
        axis=1)
    return {
        "centroid_lap_excess": float(np.clip(lap_pred - lap_gt, 0, None).mean()),
        "centroid_drift": float(np.linalg.norm(pred_c - gt_c, axis=1).mean()),
        "n_interior": int(lap_pred.shape[0]),
    }


# --------------------------------------------------------------------------
# Slice-role stratification
# --------------------------------------------------------------------------

def slice_roles(gt_label_volume, axis=0):
    """Label each slice interior / end_cap / transition / empty, per the GT.

    Rationale: these three regimes have different failure modes and a single
    average lets a gain in one hide a loss in another.

      end_cap    - first/last slice on which a given label appears. The contour
                   is small and the previous slice is a poor prior, so this is
                   where "copy the neighbour" behaviour breaks.
      transition - the label set changes between neighbouring slices (one
                   vertebra ending, another starting). Hardest case for a
                   sequential model.
      interior   - label set stable. Easiest, and where naive smoothness
                   rewards pay off - which is why it must be reported apart.
    """
    volume = np.asarray(gt_label_volume)
    n_slices = volume.shape[axis]
    per_slice_labels = []
    for i in range(n_slices):
        plane = np.take(volume, i, axis=axis)
        per_slice_labels.append(frozenset(int(v) for v in np.unique(plane) if v > 0))

    first_seen, last_seen = {}, {}
    for i, labels in enumerate(per_slice_labels):
        for label in labels:
            first_seen.setdefault(label, i)
            last_seen[label] = i

    roles = []
    for i, labels in enumerate(per_slice_labels):
        if not labels:
            roles.append("empty")
            continue
        if any(first_seen[l] == i or last_seen[l] == i for l in labels):
            roles.append("end_cap")
            continue
        prev_labels = per_slice_labels[i - 1] if i > 0 else labels
        next_labels = (per_slice_labels[i + 1]
                       if i + 1 < n_slices else labels)
        if labels != prev_labels or labels != next_labels:
            roles.append("transition")
        else:
            roles.append("interior")
    return roles


def stratify(values, roles, exclude_empty=True):
    """Mean of `values` grouped by slice role.

    `exclude_empty` defaults to True on purpose. The existing evaluator records
    empty slices as `foreground_dice: 1.0`, which inflates any average that
    includes them - a volume that is mostly background can look excellent while
    every real slice is wrong. Empty slices carry no refinement signal, so they
    are dropped rather than scored.
    """
    values = np.asarray(values, dtype=np.float64)
    out = {}
    for role in ("interior", "end_cap", "transition", "empty"):
        if exclude_empty and role == "empty":
            continue
        idx = [i for i, r in enumerate(roles) if r == role and i < values.shape[0]]
        picked = values[idx] if idx else np.zeros((0,))
        picked = picked[np.isfinite(picked)]
        out[role] = {
            "mean": float(picked.mean()) if picked.size else float("nan"),
            "n": int(picked.size),
        }
    finite_all = values[np.isfinite(values)]
    out["all_nonempty"] = {
        "mean": float(finite_all.mean()) if finite_all.size else float("nan"),
        "n": int(finite_all.size),
    }
    return out


# --------------------------------------------------------------------------
# Paired statistics
# --------------------------------------------------------------------------

def paired_bootstrap(baseline, treatment, n_boot=10000, seed=20260731,
                     alpha=0.05):
    """Paired bootstrap CI on the mean difference, plus W/T/L.

    Mandatory for this project. 2D RL gains were ~0.002 IoU against a per-sample
    spread orders of magnitude larger; unpaired comparison cannot separate that
    from noise, and a point estimate without a CI has repeatedly been misread as
    a result. Pairs must be the same units under both arms, in the same order.

    `ties_eps` treats differences below float noise as ties so that W/T/L is not
    dominated by numerically identical predictions.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)
    if baseline.shape != treatment.shape:
        raise ValueError("paired arrays must have identical shape")
    finite = np.isfinite(baseline) & np.isfinite(treatment)
    baseline, treatment = baseline[finite], treatment[finite]
    n = baseline.shape[0]
    if n == 0:
        return {"n": 0, "mean_diff": float("nan")}
    diff = treatment - baseline

    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    ties_eps = 1e-9
    wins = int((diff > ties_eps).sum())
    losses = int((diff < -ties_eps).sum())
    return {
        "n": int(n),
        "baseline_mean": float(baseline.mean()),
        "treatment_mean": float(treatment.mean()),
        "mean_diff": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool(lo > 0 or hi < 0),
        "wins": wins,
        "ties": int(n - wins - losses),
        "losses": losses,
    }


# --------------------------------------------------------------------------
# Volume-level driver
# --------------------------------------------------------------------------

def evaluate_volume(pred_label_volume, gt_label_volume, axis=0,
                    taus=(0.5, 1.0, 2.0, 3.0), per_class=True):
    """Full metric bundle for one volume.

    Per-class as well as pooled, because the base model's class_mean_dice
    (0.564) is far below its volume_dice (0.703): a few vertebra classes are
    near zero. Those are detection/data failures that refinement RL cannot fix,
    and pooled numbers would let them swamp the signal we care about. Phase0-E
    uses these per-class numbers to carve out the "roughly right, locally
    wrong" subset that refinement is actually aimed at.
    """
    pred_label_volume = np.asarray(pred_label_volume)
    gt_label_volume = np.asarray(gt_label_volume)
    if pred_label_volume.shape != gt_label_volume.shape:
        raise ValueError("pred/gt volume shape mismatch: {} vs {}".format(
            pred_label_volume.shape, gt_label_volume.shape))

    roles = slice_roles(gt_label_volume, axis=axis)
    pred_fg = pred_label_volume > 0
    gt_fg = gt_label_volume > 0

    report = {
        "shape": list(pred_label_volume.shape),
        "axis": axis,
        "role_counts": {r: roles.count(r)
                        for r in ("interior", "end_cap", "transition", "empty")},
        "foreground": {},
        "per_class": {},
    }
    report["foreground"].update(surface_nsd(pred_fg, gt_fg, taus=taus))
    report["foreground"].update(z_consistency(pred_fg, gt_fg, axis=axis))
    report["foreground"].update(centroid_consistency(pred_fg, gt_fg, axis=axis))
    inter = float(np.logical_and(pred_fg, gt_fg).sum())
    denom = float(pred_fg.sum() + gt_fg.sum())
    report["foreground"]["dice"] = (2.0 * inter / denom) if denom > 0 else float("nan")

    # Per-slice Dice, stratified by role: the diagnostic that tells us whether a
    # change helps everywhere or just on the easy interior slices.
    per_slice_dice = []
    for i in range(pred_label_volume.shape[axis]):
        p = np.take(pred_fg, i, axis=axis)
        g = np.take(gt_fg, i, axis=axis)
        d = float(p.sum() + g.sum())
        per_slice_dice.append(
            2.0 * float(np.logical_and(p, g).sum()) / d if d > 0 else np.nan)
    report["slice_dice_by_role"] = stratify(per_slice_dice, roles)

    if per_class:
        labels = [int(v) for v in np.unique(gt_label_volume) if v > 0]
        for label in labels:
            p = pred_label_volume == label
            g = gt_label_volume == label
            entry = surface_nsd(p, g, taus=taus)
            d = float(p.sum() + g.sum())
            entry["dice"] = (2.0 * float(np.logical_and(p, g).sum()) / d
                             if d > 0 else float("nan"))
            entry["gt_voxels"] = int(g.sum())
            entry.update(z_consistency(p, g, axis=axis))
            report["per_class"][str(label)] = entry
    return report


def save_report(report, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(report, handle, indent=1)
    os.replace(tmp, path)  # atomic, per project checkpoint-safety convention
    return path
