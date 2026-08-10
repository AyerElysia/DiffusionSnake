#!/usr/bin/env python3
"""Faithful VerSe-2021 evaluation primitives, in physical (mm) units.

Every metric below is traceable to a published definition. Where a published
definition is ambiguous, BOTH variants are computed and reported so the choice
is visible instead of hidden.

  per-vertebra Dice   official VerSe evaluator `compute_dice`
                      (anjany/verse @02b292b -> lib/evaluators/verse2021_3d)
  ID rate (20 mm)     official VerSe evaluator `get_hits`
  Hausdorff / HD95    surface-to-surface distances, mm. Two HD95 conventions
                      computed: max-of-directed-percentiles (medpy / MONAI)
                      and percentile-of-pooled-distances.
  NSD @ tau mm        Nikolov et al. surface Dice. NOT part of official VerSe;
                      reported as a modern complement only.
  pooled binary Dice  NOT a standard metric. This is a reimplementation of what
                      tools/volmem/eval_memflowdit_v03.py currently reports, kept
                      here purely so the two can be printed side by side.

Units contract: every distance returned by this module is in millimetres.
Callers must pass `spacing` as the (mm) size of one voxel along each axis of the
arrays they hand in, and both arrays must already live on the SAME grid.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

# --------------------------------------------------------------- numpy compat
# The vendored official evaluator calls `np.bool`, which numpy>=1.24 removed.
# Restore the alias BEFORE importing it so the vendored source can stay
# byte-identical to upstream (do not patch third-party code in place).
if not hasattr(np, "bool"):
    np.bool = np.bool_  # type: ignore[attr-defined]

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_VERSE_EVAL_DIR = _PROJECT_ROOT / "lib" / "evaluators" / "verse2021_3d"
if str(_VERSE_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_VERSE_EVAL_DIR))

from scipy import ndimage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

# Import directly from eval_utilities module to bypass lib.evaluators.__init__
# which triggers global config initialization with incompatible argparse
from eval_utilities import (  # noqa: E402
    compute_dice,
    get_hits,
)

MAX_VERT_IDX = 28
ID_RATE_THRESHOLD_MM = 20.0  # VerSe: correct label must be closest AND < 20 mm


# ---------------------------------------------------------------- label utils
def present_labels(volume, max_vert_idx=MAX_VERT_IDX):
    """Sorted list of vertebra labels present in `volume` (background excluded)."""
    values = np.unique(volume)
    values = values[(values > 0) & (values <= max_vert_idx)]
    return [int(v) for v in values]


def centroids_mm(volume, spacing, max_vert_idx=MAX_VERT_IDX):
    """Per-label centre of mass, in mm, laid out for the official `get_hits`.

    Returns an (max_vert_idx, 3) array; row (label-1) holds the centroid of that
    label, NaN when the label is absent. This mirrors `data_utilities.calc_centroids`
    (centre of mass of the label mask) rather than reading the VerSe centroid JSON,
    which is the standard surrogate when only a mask is available.
    """
    spacing = np.asarray(spacing, dtype=float)
    out = np.full((max_vert_idx, 3), np.nan, dtype=float)
    for label in present_labels(volume, max_vert_idx):
        com = ndimage.center_of_mass(volume == label)
        out[label - 1] = np.asarray(com, dtype=float) * spacing
    return out


def id_rate(gt_volume, pred_volume, spacing, max_vert_idx=MAX_VERT_IDX):
    """Official VerSe identification rate.

    A vertebra counts as identified when the predicted centroid carrying that
    label has the same-label GT centroid as its nearest GT centroid, and that
    distance is < 20 mm. Returns (hits, hit_list, n_gt, cent_gt, cent_pred).
    `hit_list[label-1]` is NaN (absent in GT), 0 (missed) or 1 (identified).
    """
    cent_gt = centroids_mm(gt_volume, spacing, max_vert_idx)
    cent_pred = centroids_mm(pred_volume, spacing, max_vert_idx)
    hits, hit_list = get_hits(cent_gt, cent_pred, max_vert_idx)
    n_gt = int(np.count_nonzero(~np.isnan(cent_gt[:, 0])))
    return int(hits), hit_list, n_gt, cent_gt, cent_pred


# -------------------------------------------------------------- surface utils
def _crop_to_union_bbox(mask_a, mask_b, pad=2):
    """Crop both masks to their union bounding box plus `pad` voxels of margin.

    The padding guarantees a ring of background around the objects, so surface
    extraction never sees an object clipped by the crop edge.
    """
    union = mask_a | mask_b
    if not union.any():
        return mask_a, mask_b
    slices = []
    for axis, size in enumerate(union.shape):
        idx = np.where(union.any(axis=tuple(i for i in range(union.ndim) if i != axis)))[0]
        lo = max(int(idx[0]) - pad, 0)
        hi = min(int(idx[-1]) + pad + 1, size)
        slices.append(slice(lo, hi))
    sl = tuple(slices)
    return mask_a[sl], mask_b[sl]


def surface_mask(mask, border_value=1):
    """Boundary voxels of a binary mask: mask AND NOT erode(mask).

    6-connected structuring element (generate_binary_structure(3, 1)).
    `border_value=1` treats voxels outside the array as foreground, so an object
    truncated by the field of view does not gain a phantom surface along the FOV
    cut. After `_crop_to_union_bbox` padding this choice is inert for objects
    fully inside the volume; it only matters for FOV-truncated vertebrae.
    """
    structure = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=border_value)
    return mask & ~eroded


def surface_distances_mm(pred_mask, gt_mask, spacing, pad=2):
    """Directed surface distances in mm, between voxel-centre point sets.

    Returns (d_pred_to_gt, d_gt_to_pred) or None when either surface is empty.
    Approximation note: distances are nearest-neighbour distances between surface
    VOXEL CENTRES, not between triangulated surfaces. On a 1 mm grid the
    discretisation error is sub-voxel, but it is an approximation, and it is the
    same approximation medpy/MONAI make.
    """
    spacing = np.asarray(spacing, dtype=float)
    pred_c, gt_c = _crop_to_union_bbox(pred_mask, gt_mask, pad=pad)
    pred_surf = surface_mask(pred_c)
    gt_surf = surface_mask(gt_c)
    if pred_surf.sum() == 0 or gt_surf.sum() == 0:
        return None
    pred_pts = np.argwhere(pred_surf) * spacing
    gt_pts = np.argwhere(gt_surf) * spacing
    d_pred_to_gt, _ = cKDTree(gt_pts).query(pred_pts, workers=-1)
    d_gt_to_pred, _ = cKDTree(pred_pts).query(gt_pts, workers=-1)
    return d_pred_to_gt, d_gt_to_pred


def hausdorff_mm(d_pred_to_gt, d_gt_to_pred, percentile=95.0):
    """Both HD95 conventions plus symmetric HD and mean surface distance (mm).

    hd95_directed_max : max(P95(d_p2g), P95(d_g2p))  <- medpy.hd95, MONAI default
    hd95_pooled       : P95(concat(d_p2g, d_g2p))    <- also seen in the wild
    They are not equal; reporting both makes the convention auditable.
    """
    return {
        "hd_mm": float(max(d_pred_to_gt.max(), d_gt_to_pred.max())),
        "hd95_directed_max_mm": float(
            max(np.percentile(d_pred_to_gt, percentile),
                np.percentile(d_gt_to_pred, percentile))
        ),
        "hd95_pooled_mm": float(
            np.percentile(np.concatenate([d_pred_to_gt, d_gt_to_pred]), percentile)
        ),
        "assd_mm": float(
            (d_pred_to_gt.sum() + d_gt_to_pred.sum())
            / (d_pred_to_gt.size + d_gt_to_pred.size)
        ),
    }


def nsd(d_pred_to_gt, d_gt_to_pred, tau_mm):
    """Normalised surface distance (surface Dice) at tolerance tau, in mm.

    Fraction of surface points, pooled over both directions, whose nearest
    counterpart lies within tau. Voxel-count weighted (not area weighted); the
    area-weighted DeepMind variant needs a marching-cubes surface and is not
    used here.
    """
    within = int((d_pred_to_gt <= tau_mm).sum() + (d_gt_to_pred <= tau_mm).sum())
    total = int(d_pred_to_gt.size + d_gt_to_pred.size)
    return float(within) / total if total else float("nan")


# ------------------------------------------------- the current non-standard one
def pooled_binary_dice(pred_volume, gt_volume):
    """Reimplementation of the CURRENT eval_memflowdit_v03.py Dice.

    Binarises everything (`> 0`), pools all foreground voxels of the whole
    volume, then computes one Dice. Label identity is discarded entirely, so an
    off-by-one labelling error is invisible to it. Kept only for side-by-side
    comparison; it is not a VerSe number.
    """
    gt_bin = gt_volume > 0
    pred_bin = pred_volume > 0
    denom = int(gt_bin.sum()) + int(pred_bin.sum())
    if denom == 0:
        return 1.0
    inter = int(np.logical_and(gt_bin, pred_bin).sum())
    return 2.0 * inter / denom


# ------------------------------------------------------------------ top level
def evaluate_volume_pair(
    pred_volume,
    gt_volume,
    spacing,
    taus_mm=(1.0, 2.0),
    max_vert_idx=MAX_VERT_IDX,
):
    """Full VerSe-standard evaluation of one predicted label volume.

    Both volumes must be integer label maps on the SAME grid. `spacing` is the
    voxel size in mm.

    Reported Dice aggregations, both over the vertebrae PRESENT IN GT:
      dice_all_gt_mean   every GT vertebra contributes, a mislabelled vertebra
                         contributes its true (near-zero) Dice. This is the
                         leaderboard-comparable number.
      dice_id_gated_mean only correctly identified vertebrae contribute. Strictly
                         optimistic; a diagnostic, never a headline number.
    """
    pred_volume = np.asarray(pred_volume)
    gt_volume = np.asarray(gt_volume)
    if pred_volume.shape != gt_volume.shape:
        raise ValueError(
            "grid mismatch: pred {} vs gt {}".format(pred_volume.shape, gt_volume.shape)
        )
    spacing = np.asarray(spacing, dtype=float)

    hits, hit_list, n_gt, cent_gt, cent_pred = id_rate(
        gt_volume, pred_volume, spacing, max_vert_idx
    )

    gt_labels = present_labels(gt_volume, max_vert_idx)
    pred_labels = present_labels(pred_volume, max_vert_idx)

    per_vertebra = []
    for label in gt_labels:
        gt_mask = gt_volume == label
        pred_mask = pred_volume == label
        identified = bool(hit_list[label - 1] == 1.0)

        # Dice is computed with the official primitive, unconditionally: it does
        # not depend on identification, and gating it would hide failures.
        dice = float(compute_dice(gt_mask, pred_mask))

        entry = {
            "label": label,
            "identified": identified,
            "dice": dice,
            "gt_voxels": int(gt_mask.sum()),
            "pred_voxels": int(pred_mask.sum()),
            "centroid_error_mm": None,
            "hd_mm": None,
            "hd95_directed_max_mm": None,
            "hd95_pooled_mm": None,
            "assd_mm": None,
        }
        for tau in taus_mm:
            entry["nsd_{:g}mm".format(tau)] = None

        if not np.isnan(cent_pred[label - 1, 0]):
            entry["centroid_error_mm"] = float(
                np.linalg.norm(cent_pred[label - 1] - cent_gt[label - 1])
            )

        dists = None
        if pred_mask.any() and gt_mask.any():
            dists = surface_distances_mm(pred_mask, gt_mask, spacing)
        if dists is not None:
            d_p2g, d_g2p = dists
            entry.update(hausdorff_mm(d_p2g, d_g2p))
            for tau in taus_mm:
                entry["nsd_{:g}mm".format(tau)] = nsd(d_p2g, d_g2p, tau)
        per_vertebra.append(entry)

    def _mean(key, rows):
        vals = [r[key] for r in rows if r[key] is not None and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else None

    id_rows = [r for r in per_vertebra if r["identified"]]
    dice_all = [r["dice"] for r in per_vertebra]
    dice_gated = [r["dice"] for r in id_rows]

    summary = {
        "n_gt_vertebrae": n_gt,
        "n_pred_vertebrae": len(pred_labels),
        "gt_labels": gt_labels,
        "pred_labels": pred_labels,
        "id_hits": hits,
        "id_rate": float(hits) / n_gt if n_gt else None,
        "dice_all_gt_mean": float(np.mean(dice_all)) if dice_all else None,
        "dice_all_gt_std": float(np.std(dice_all)) if dice_all else None,
        "dice_id_gated_mean": float(np.mean(dice_gated)) if dice_gated else None,
        "pooled_binary_dice_NONSTANDARD": pooled_binary_dice(pred_volume, gt_volume),
        "hd_mm_mean": _mean("hd_mm", per_vertebra),
        "hd95_directed_max_mm_mean": _mean("hd95_directed_max_mm", per_vertebra),
        "hd95_pooled_mm_mean": _mean("hd95_pooled_mm", per_vertebra),
        "assd_mm_mean": _mean("assd_mm", per_vertebra),
        "centroid_error_mm_mean": _mean("centroid_error_mm", per_vertebra),
        "spacing_mm": [float(s) for s in spacing],
    }
    for tau in taus_mm:
        key = "nsd_{:g}mm".format(tau)
        summary[key + "_mean"] = _mean(key, per_vertebra)

    summary["per_vertebra"] = per_vertebra
    return summary
