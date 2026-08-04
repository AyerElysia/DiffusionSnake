"""Phase0-C: oracle ceilings for candidate 3D action families.

The question this answers
------------------------
The 2D work burned v10->v15 on a per-point FM-velocity-scaling action space
before an oracle analysis showed why it could never work: that family could
remove only **7.34%** of the squared boundary error, against **85.50%** for a
plain per-point normal residual. No amount of policy capacity fixes a subspace
that does not contain the answer - and indeed a bigger policy net (H256) never
beat H64, while a reward-credit probe found R^2 = 0.008.

So before writing any 3D RL, measure the ceiling of every action family we might
use. An oracle here is generous by construction: it is the *best possible*
action, chosen with full knowledge of the GT. A family whose oracle is low is
dead on arrival; a family whose oracle is high is merely *permitted*, not proven.

Method
------
For each predicted contour point p_i we have the nearest GT boundary point
g_i, so the residual is r_i = g_i - p_i. An action family is a linear subspace
of displacement fields spanned by a basis Phi. The oracle displacement is the
least-squares projection of r onto that subspace, and

    reduction = 1 - ||r - Phi a*||^2 / ||r||^2

is the fraction of squared error the family can remove. This is exactly the
quantity the 2D analysis reported, so the numbers are directly comparable.

Two things are reported that the 2D analysis did not, and both matter for 3D:

  * **Ceiling per parameter.** A family that reaches 90% with 400 free
    parameters per contour is worse for RL than one that reaches 80% with 6.
    Credit assignment difficulty scales with the action dimension - that is the
    whole reason 2D's per-point space failed. So the deliverable is a
    ceiling-vs-dimension curve, not a single number.

  * **3D-native vs per-slice bases at matched dimension.** The user's question
    is whether the 2D solution is optimal in 3D or merely transferable. The
    honest test is to give a per-slice family and a cross-slice family the same
    parameter budget and see which buys more. If the cylindrical (theta, x)
    basis wins at matched budget, a 3D-native action space is justified; if it
    does not, we should say so and keep the simpler per-slice actions.

Caveats, stated up front
------------------------
Nearest-neighbour correspondence understates the true error (it is a lower
bound on any consistent matching), so all ceilings here are *optimistic*. That
is acceptable because we use them to *reject* families: a family that cannot
clear an optimistic bar certainly cannot clear a real one. It does mean a high
ceiling must not be read as a promise.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


# --------------------------------------------------------------------------
# Basis construction
# --------------------------------------------------------------------------

def contour_theta(poly):
    """Angular parameter per contour point, measured from the centroid.

    Used instead of arc length because it is what a cylindrical harmonic basis
    needs and because it is stable under the resampling the snake head applies
    (points are already distributed around a closed loop).
    """
    centroid = poly.mean(axis=0, keepdims=True)
    delta = poly - centroid
    return np.arctan2(delta[:, 1], delta[:, 0])


def angular_basis(theta, n_modes):
    """[P, 2*n_modes+1] Fourier design matrix on the contour angle.

    Mode 0 is a uniform normal offset (grow/shrink), modes 1..K are progressively
    finer angular detail. Truncating at K is what makes this a *low-frequency*
    family: it can move the boundary smoothly but cannot chase per-point noise.
    """
    columns = [np.ones_like(theta)]
    for k in range(1, n_modes + 1):
        columns.append(np.cos(k * theta))
        columns.append(np.sin(k * theta))
    return np.stack(columns, axis=1)


def axial_basis(positions, n_modes):
    """[S, n_modes+1] polynomial basis on the (normalised) slice coordinate.

    Polynomial rather than Fourier because a vertebra's profile along the spine
    axis is not periodic - it has ends. Degree 0 is a constant offset shared by
    the whole 3D structure, which is exactly the cross-slice coupling a
    per-slice action family cannot express.
    """
    positions = np.asarray(positions, dtype=np.float64)
    span = positions.max() - positions.min()
    if span <= 0:
        normalized = np.zeros_like(positions)
    else:
        normalized = 2.0 * (positions - positions.min()) / span - 1.0
    return np.stack([normalized ** d for d in range(n_modes + 1)], axis=1)


# --------------------------------------------------------------------------
# Oracle projection
# --------------------------------------------------------------------------

def oracle_reduction(residual, basis_vectors):
    """Fraction of ||residual||^2 removed by the best element of a subspace.

    `residual`      [M, 2] stacked per-point residual vectors.
    `basis_vectors` [M, 2, D] displacement produced by each of D unit actions.

    Solved as a plain least-squares in the flattened 2M-dimensional space, with
    `rcond` left to lstsq so rank-deficient bases (which do occur: a degenerate
    contour makes high-order modes collinear) degrade gracefully instead of
    exploding.
    """
    residual = np.asarray(residual, dtype=np.float64)
    total = float((residual ** 2).sum())
    if total <= 0 or basis_vectors.shape[2] == 0:
        return 0.0, 0
    flat_r = residual.reshape(-1)
    flat_b = basis_vectors.reshape(-1, basis_vectors.shape[2])
    solution, _, rank, _ = np.linalg.lstsq(flat_b, flat_r, rcond=None)
    leftover = flat_r - flat_b @ solution
    return float(1.0 - (leftover ** 2).sum() / total), int(rank)


def normal_displacement_basis(normals, profile):
    """Combine a per-point scalar profile with the outward normal.

    `profile` [P, D] scalar weight of each action on each point.
    Returns   [P, 2, D] displacement, i.e. every action moves points along their
    own normal. Restricting to the normal direction is deliberate: tangential
    motion just slides points along the boundary without changing the shape, so
    including it would inflate the ceiling without any real benefit.
    """
    return normals[:, :, None] * profile[:, None, :]


# --------------------------------------------------------------------------
# Families
# --------------------------------------------------------------------------

def family_per_point_normal(poly, normals, residual):
    """Upper bound for any normal-direction action. The 2D reference: 85.50%.

    Every point moves independently along its normal, so the achievable subspace
    is the whole normal component of the residual and what is left is purely
    tangential. This is the ceiling all cheaper normal-based families are
    measured against - it is not itself a usable action space (P free parameters
    per contour is precisely what failed in 2D).
    """
    along = (residual * normals).sum(axis=1)
    total = float((residual ** 2).sum())
    if total <= 0:
        return 0.0
    return float((along ** 2).sum() / total)


def family_per_point_free(residual):
    """Sanity check: unrestricted per-point 2D motion is 100% by construction."""
    return 1.0 if float((residual ** 2).sum()) > 0 else 0.0


def family_slice_angular(poly, normals, residual, n_modes):
    """Per-slice low-frequency normal deformation. Dimension 2*n_modes+1."""
    profile = angular_basis(contour_theta(poly), n_modes)
    return oracle_reduction(residual, normal_displacement_basis(normals, profile))


def family_slice_rigid(poly, residual):
    """Per-slice translation + isotropic scale. Dimension 3.

    The cheapest plausible action space, and a useful floor: if most of the
    error is a whole-contour offset then RL barely needs to shape anything.
    """
    n_points = poly.shape[0]
    centered = poly - poly.mean(axis=0, keepdims=True)
    basis = np.zeros((n_points, 2, 3), dtype=np.float64)
    basis[:, 0, 0] = 1.0                 # translate x
    basis[:, 1, 1] = 1.0                 # translate y
    basis[:, :, 2] = centered            # isotropic scale about the centroid
    return oracle_reduction(residual, basis)


def family_cylindrical(polys, normals, residuals, positions,
                       n_angular, n_axial):
    """3D-native (theta, x) harmonics shared across the slices of one vertebra.

    This is the family that can express "this vertebra is systematically 1px too
    fat towards the top", which no per-slice family can represent with shared
    parameters. Dimension (2*n_angular+1) * (n_axial+1) for the *whole* label
    track, not per slice - so at matched budget it is far cheaper than a
    per-slice family, which is exactly the point being tested.
    """
    axial = axial_basis(positions, n_axial)          # [S, A]
    blocks, stacked_residual = [], []
    for s, (poly, normal, residual) in enumerate(zip(polys, normals, residuals)):
        angular = angular_basis(contour_theta(poly), n_angular)   # [P, T]
        # Outer product of the angular and axial profiles for this slice.
        profile = (angular[:, :, None] * axial[s][None, None, :]).reshape(
            angular.shape[0], -1)                                  # [P, T*A]
        blocks.append(normal_displacement_basis(normal, profile))
        stacked_residual.append(residual)
    return oracle_reduction(np.concatenate(stacked_residual, axis=0),
                            np.concatenate(blocks, axis=0))


def family_slice_angular_pooled(polys, normals, residuals, n_modes):
    """Per-slice angular family with *independent* parameters per slice.

    The matched-budget comparison partner for `family_cylindrical`: same angular
    resolution, but every slice gets its own coefficients. Reported alongside its
    parameter count so the two can be compared honestly rather than by ceiling
    alone.
    """
    blocks, stacked_residual = [], []
    total_dim = 0
    for poly, normal, residual in zip(polys, normals, residuals):
        angular = angular_basis(contour_theta(poly), n_modes)
        blocks.append(normal_displacement_basis(normal, angular))
        stacked_residual.append(residual)
        total_dim += angular.shape[1]

    # Block-diagonal assembly: slice s's actions may only move slice s.
    n_rows = sum(b.shape[0] for b in blocks)
    combined = np.zeros((n_rows, 2, total_dim), dtype=np.float64)
    row, col = 0, 0
    for block in blocks:
        p, _, d = block.shape
        combined[row:row + p, :, col:col + d] = block
        row += p
        col += d
    reduction, rank = oracle_reduction(
        np.concatenate(stacked_residual, axis=0), combined)
    return reduction, total_dim, rank


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def load_volume(npz_path):
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}


def analyse(npz_paths, angular_modes=(0, 1, 2, 3, 5, 8, 16),
            cylindrical_specs=((1, 1), (2, 1), (2, 2), (3, 2), (5, 3)),
            max_dist=None):
    """Aggregate oracle ceilings over every dumped volume.

    Ceilings are pooled by summing squared error across contours rather than
    averaging per-contour percentages: a contour that is already nearly perfect
    contributes almost no error and should not get an equal vote. This matches
    how the 2D analysis pooled its 692,608 points.
    """
    per_point_normal_num = 0.0
    per_point_total = 0.0
    slice_family_num = defaultdict(float)
    rigid_num = 0.0
    n_contours = 0
    n_points = 0
    dropped_no_gt = 0

    # Grouped by (volume, label) so cross-slice families see a vertebra track.
    tracks = defaultdict(list)

    for path in npz_paths:
        data = load_volume(path)
        volume_id = os.path.splitext(os.path.basename(path))[0]
        poly_all = data["poly"]
        target_all = data["gt_target"]
        normal_all = data["normal"]
        label_all = data["label"]
        slice_all = data["slice_idx"]

        for i in range(poly_all.shape[0]):
            poly = poly_all[i].astype(np.float64)
            target = target_all[i].astype(np.float64)
            normal = normal_all[i].astype(np.float64)
            if not np.isfinite(target).all():
                dropped_no_gt += 1
                continue
            residual = target - poly
            if max_dist is not None:
                # Optionally ignore points whose nearest GT is implausibly far;
                # those are detection failures, not refinement targets.
                keep = np.linalg.norm(residual, axis=1) <= max_dist
                if keep.sum() < 8:
                    dropped_no_gt += 1
                    continue
                poly, target = poly[keep], target[keep]
                normal, residual = normal[keep], residual[keep]

            total = float((residual ** 2).sum())
            if total <= 0:
                continue
            per_point_total += total
            per_point_normal_num += family_per_point_normal(
                poly, normal, residual) * total
            rigid_num += family_slice_rigid(poly, residual)[0] * total
            for modes in angular_modes:
                slice_family_num[modes] += family_slice_angular(
                    poly, normal, residual, modes)[0] * total
            n_contours += 1
            n_points += poly.shape[0]
            tracks[(volume_id, int(label_all[i]))].append(
                (int(slice_all[i]), poly, normal, residual))

    report = {
        "n_contours": n_contours,
        "n_points": n_points,
        "dropped_contours_no_gt": dropped_no_gt,
        "max_dist_filter": max_dist,
        "families": {},
    }
    if per_point_total <= 0:
        return report

    report["families"]["per_point_free_2d"] = {
        "reduction": 1.0, "dim_per_contour": "2P", "note": "trivial upper bound"}
    report["families"]["per_point_normal"] = {
        "reduction": per_point_normal_num / per_point_total,
        "dim_per_contour": "P",
        "note": "2D reference value was 0.8550"}
    report["families"]["slice_rigid_t+s"] = {
        "reduction": rigid_num / per_point_total, "dim_per_contour": 3}
    for modes in angular_modes:
        report["families"]["slice_angular_k{}".format(modes)] = {
            "reduction": slice_family_num[modes] / per_point_total,
            "dim_per_contour": 2 * modes + 1}

    # ---- cross-slice comparison at matched parameter budget ----
    cross = []
    # Sort on the slice index ONLY. A label can own more than one contour in the
    # same slice (verse011: 554 contours over 38 slices and 12 labels), and a
    # bare sorted() would fall through to comparing the ndarray payloads and
    # raise "truth value of an array is ambiguous".
    usable = {k: sorted(v, key=lambda entry: entry[0])
              for k, v in tracks.items() if len(v) >= 4}
    for n_angular, n_axial in cylindrical_specs:
        cyl_num = cyl_den = 0.0
        slice_num = slice_den = 0.0
        cyl_dim_total = slice_dim_total = 0
        for (volume_id, label), entries in usable.items():
            positions = [e[0] for e in entries]
            polys = [e[1] for e in entries]
            normals = [e[2] for e in entries]
            residuals = [e[3] for e in entries]
            total = float(sum((r ** 2).sum() for r in residuals))
            if total <= 0:
                continue
            cyl_r, _ = family_cylindrical(
                polys, normals, residuals, positions, n_angular, n_axial)
            cyl_num += cyl_r * total
            cyl_den += total
            cyl_dim_total += (2 * n_angular + 1) * (n_axial + 1)

            sl_r, sl_dim, _ = family_slice_angular_pooled(
                polys, normals, residuals, n_angular)
            slice_num += sl_r * total
            slice_den += total
            slice_dim_total += sl_dim
        if cyl_den > 0:
            cross.append({
                "n_angular": n_angular,
                "n_axial": n_axial,
                "cylindrical_reduction": cyl_num / cyl_den,
                "cylindrical_params_per_track": (2 * n_angular + 1) * (n_axial + 1),
                "per_slice_reduction": slice_num / slice_den,
                "per_slice_params_per_track_mean": (
                    slice_dim_total / max(len(usable), 1)),
                "n_tracks": len(usable),
            })
    report["cross_slice"] = cross
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-dist", type=float, default=None,
                        help="Drop points whose nearest GT is further than this "
                             "(voxels). Separates refinement from detection "
                             "failure; run with and without.")
    args = parser.parse_args()

    paths = sorted(
        os.path.join(args.probe_dir, name)
        for name in os.listdir(args.probe_dir)
        if name.endswith(".npz"))
    if not paths:
        raise RuntimeError("no npz dumps in " + args.probe_dir)
    print("[oracle] analysing {} volumes".format(len(paths)), flush=True)

    report = analyse(paths, max_dist=args.max_dist)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)

    print("\n{:<28} {:>10} {:>12}".format("family", "reduction", "dim/contour"))
    print("-" * 52)
    for name, entry in report["families"].items():
        print("{:<28} {:>10.4f} {:>12}".format(
            name, entry["reduction"], str(entry["dim_per_contour"])))
    if report.get("cross_slice"):
        print("\ncross-slice at matched angular resolution "
              "({} vertebra tracks)".format(report["cross_slice"][0]["n_tracks"]))
        print("{:>4} {:>4} {:>14} {:>8} {:>14} {:>10}".format(
            "K", "A", "cylindrical", "params", "per-slice", "params"))
        for row in report["cross_slice"]:
            print("{:>4} {:>4} {:>14.4f} {:>8} {:>14.4f} {:>10.0f}".format(
                row["n_angular"], row["n_axial"],
                row["cylindrical_reduction"],
                row["cylindrical_params_per_track"],
                row["per_slice_reduction"],
                row["per_slice_params_per_track_mean"]))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
