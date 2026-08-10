#!/usr/bin/env python3
"""Run the faithful VerSe-2021 metrics on real data and show exactly what they do.

This is a METRIC CALIBRATION study, not a model evaluation. The "predictions" are
real ground-truth volumes put through controlled, known perturbations (dilate by
1 voxel, shift by 2 mm, shift every label by +1, drop a vertebra, ...). Because
the perturbation is known exactly, every number the evaluator prints can be
checked against what it *should* be. That is the point: it pins down what each
metric measures before any model number is allowed to lean on it.

It also settles the data questions the protocol depends on, empirically:
  * what spacing the source NIfTI actually has
  * whether the 2D PNG slice stack is a faithful 1 mm rendering of that NIfTI
  * whether the PNG masks carry per-vertebra labels or only foreground

Outputs: results.json + a self-contained report.html (figures inlined as base64).

Example:
  python tools/volmem/verse_eval/run_verse_calibration.py \
      --nii-root /home/medteam/Zhrch/detect_3D_lgz2/datasets/raw_nii_spacing1p5_lgz/validation \
      --cases sub-verse022 \
      --out-dir data/outputs/verse_eval_calibration_20260809
"""
from __future__ import annotations

import argparse
import base64
import csv
import html as html_mod
import inspect
import io
import json
import pathlib
import sys
from collections import OrderedDict

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import cv2  # noqa: E402
import nibabel as nib  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import verse_metrics as vm  # noqa: E402

# Import data_utilities directly to bypass lib.evaluators.__init__ which triggers
# global config initialization with incompatible argparse
_VERSE_DATA_UTILS_DIR = _PROJECT_ROOT / "lib" / "evaluators" / "verse2021_3d"
if str(_VERSE_DATA_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_VERSE_DATA_UTILS_DIR))
from data_utilities import resample_nib, v_dict  # noqa: E402

DEFAULT_MANIFEST = (
    "/home/medteam/Zhrch/detect_3D_lgz2/datasets/sagittal_2d_fixed/"
    "manifests/slice_manifest.csv"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--split", default="validation")
    p.add_argument("--nii-root", required=True,
                   help="dir containing rawdata/ and derivatives/")
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--taus-mm", nargs="+", type=float, default=[1.0, 2.0])
    return p.parse_args()


# ------------------------------------------------------------------ data load
def read_manifest(path, split, case_id):
    """Rows for one case, sorted by sagittal slice index."""
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == split and row["case_id"] == case_id:
                rows.append(row)
    rows.sort(key=lambda r: int(r["slice_idx"]))
    return rows


def gt_volume_from_png_stack(rows):
    """Stack per-slice sagittal PNG label masks into an (X, Y, Z) label volume.

    Manifest contract for this dataset (verified against the files):
      png_axis            = sagittal_x        -> slice_idx indexes X
      png_width_meaning   = original_Y_...    -> PNG width  is Y
      png_height_meaning  = original_Z_...    -> PNG height is Z
    so a PNG read by OpenCV has shape (H, W) = (Z, Y) and volume[x] = png.T.
    """
    # Infer shape: X = number of slices, (Y, Z) from first PNG
    r0 = rows[0]
    first_png = cv2.imread(r0["mask_path"], cv2.IMREAD_UNCHANGED)
    if first_png is None:
        raise RuntimeError("Cannot read first PNG: {}".format(r0["mask_path"]))
    png_h, png_w = first_png.shape
    shape = (len(rows), png_w, png_h)  # (X, Y, Z) where PNG is (H, W) = (Z, Y)
    vol = np.zeros(shape, dtype=np.uint8)
    notes = {"expected_slices": shape[0], "rows_in_manifest": len(rows),
             "missing": [], "shape_mismatch": [], "dtypes": set(), "max_label": 0}
    for row in rows:
        x = int(row["slice_idx"])
        path = row["mask_path"]
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            notes["missing"].append(path)
            continue
        notes["dtypes"].add(str(img.dtype))
        notes["max_label"] = max(notes["max_label"], int(img.max()))
        if img.shape != (shape[2], shape[1]):
            notes["shape_mismatch"].append((path, img.shape))
            continue
        vol[x] = img.T.astype(np.uint8)
    notes["dtypes"] = sorted(notes["dtypes"])
    # PNG generation used nib.as_closest_canonical which flips L->R (LAS->RAS).
    # Flip X axis back to match original NIfTI orientation for evaluation.
    vol = vol[::-1]
    return vol, shape, notes


def resample_labels_to_1mm(nii):
    """Official resample_nib at order=0, with the label-safety fix it needs.

    resample_nib hardcodes cval=-1024 (sensible for CT HU, wrong for a label
    map): voxels pulled from outside the FOV come back negative and would wrap
    to ~64000 when cast to an unsigned type. Clip to >= 0 before casting.
    """
    out = resample_nib(nii, voxel_spacing=(1, 1, 1), order=0)
    data = np.asanyarray(out.dataobj)
    n_negative = int((data < 0).sum())
    data = np.clip(np.rint(data), 0, vm.MAX_VERT_IDX).astype(np.uint8)
    return data, n_negative


def align_grids(a, b):
    """Crop two volumes to their common shape, reporting what was trimmed."""
    common = tuple(min(x, y) for x, y in zip(a.shape, b.shape))
    sl = tuple(slice(0, c) for c in common)
    return a[sl], b[sl], {"shape_a": list(a.shape), "shape_b": list(b.shape),
                          "common": list(common),
                          "trimmed": [int(max(x, y) - c)
                                      for x, y, c in zip(a.shape, b.shape, common)]}


# --------------------------------------------------------------- perturbations
def p_identity(vol):
    return vol.copy()


def p_dilate(vol, iters=1):
    """Grow every label by `iters` voxels, resolving collisions by nearest label."""
    from scipy import ndimage
    fg = vol > 0
    grown = ndimage.binary_dilation(
        fg, structure=ndimage.generate_binary_structure(3, 1), iterations=iters)
    idx = ndimage.distance_transform_edt(
        ~fg, return_distances=False, return_indices=True)
    out = vol[tuple(idx)]
    out[~grown] = 0
    return out.astype(vol.dtype)


def p_erode(vol, iters=1):
    from scipy import ndimage
    out = vol.copy()
    st = ndimage.generate_binary_structure(3, 1)
    for label in vm.present_labels(vol):
        m = vol == label
        out[m & ~ndimage.binary_erosion(m, structure=st, iterations=iters)] = 0
    return out


def p_shift(vol, mm, spacing):
    """Translate the whole label map along +Z by `mm` (spacing is 1 mm here)."""
    shift = int(round(mm / spacing[2]))
    out = np.zeros_like(vol)
    if shift == 0:
        return vol.copy()
    out[:, :, shift:] = vol[:, :, :-shift]
    return out


def p_label_shift(vol, delta=1):
    """Relabel every vertebra L -> L+delta. Foreground is untouched.

    This is the off-by-one labelling failure VerSe cares about, and the exact
    case a pooled foreground Dice cannot see.
    """
    out = np.zeros_like(vol)
    for label in vm.present_labels(vol):
        new = label + delta
        if 1 <= new <= vm.MAX_VERT_IDX:
            out[vol == label] = new
    return out


def p_drop_middle(vol):
    """Delete the middle vertebra entirely (a false negative)."""
    labels = vm.present_labels(vol)
    if not labels:
        return vol.copy()
    out = vol.copy()
    out[vol == labels[len(labels) // 2]] = 0
    return out


def p_merge_adjacent(vol):
    """Merge the middle vertebra into its neighbour (a label collision)."""
    labels = vm.present_labels(vol)
    if len(labels) < 2:
        return vol.copy()
    i = len(labels) // 2
    out = vol.copy()
    out[vol == labels[i]] = labels[i - 1]
    return out


def build_perturbations(spacing):
    return OrderedDict([
        ("identity", ("perfect prediction; every metric must hit its ideal value",
                      lambda v: p_identity(v))),
        ("dilate_1vox", ("every label grown 1 voxel (over-segmentation)",
                         lambda v: p_dilate(v, 1))),
        ("erode_1vox", ("every label shrunk 1 voxel (under-segmentation)",
                        lambda v: p_erode(v, 1))),
        ("shift_1mm", ("whole volume translated 1 mm along +Z",
                       lambda v: p_shift(v, 1.0, spacing))),
        ("shift_2mm", ("whole volume translated 2 mm along +Z",
                       lambda v: p_shift(v, 2.0, spacing))),
        ("shift_5mm", ("whole volume translated 5 mm along +Z",
                       lambda v: p_shift(v, 5.0, spacing))),
        ("label_shift_+1", ("correct masks, every label off by one (L -> L+1)",
                            lambda v: p_label_shift(v, 1))),
        ("drop_middle_vert", ("middle vertebra deleted (false negative)",
                              lambda v: p_drop_middle(v))),
        ("merge_adjacent", ("middle vertebra merged into its neighbour",
                            lambda v: p_merge_adjacent(v))),
    ])


# -------------------------------------------------------------------- figures
def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="#0F131C")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _label_rgb(vol_slice):
    """Colour a 2D label slice with the official VerSe ITK colour table."""
    # Import directly from data_utilities to bypass lib.evaluators.__init__
    from data_utilities import colors_itk
    rgb = np.zeros(vol_slice.shape + (3,), dtype=float)
    for label in np.unique(vol_slice):
        if label == 0:
            continue
        rgb[vol_slice == label] = colors_itk[int(label) - 1]
    return rgb


def fig_overlays(gt, variants, ct_slice_idx):
    """Mid-sagittal slice of GT and of each perturbed volume, same slice index."""
    names = list(variants.keys())
    ncol = min(5, len(names) + 1)
    nrow = int(np.ceil((len(names) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.4 * nrow),
                             facecolor="#0F131C")
    axes = np.atleast_1d(axes).ravel()
    panels = [("ground truth", gt)] + [(n, variants[n]) for n in names]
    for ax, (title, vol) in zip(axes, panels):
        ax.imshow(np.rot90(_label_rgb(vol[ct_slice_idx])), interpolation="nearest")
        ax.set_title(title, color="#E4E7EB", fontsize=9)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle("sagittal slice x={}, official VerSe label colours".format(ct_slice_idx),
                 color="#9CA3AF", fontsize=10)
    return _fig_to_b64(fig)


def fig_headline(results):
    """The one chart that matters: pooled Dice vs per-vertebra Dice vs ID rate."""
    names = [r["perturbation"] for r in results]
    pooled = [r["summary"]["pooled_binary_dice_NONSTANDARD"] for r in results]
    per_v = [r["summary"]["dice_all_gt_mean"] or 0.0 for r in results]
    idr = [r["summary"]["id_rate"] or 0.0 for r in results]
    x = np.arange(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 4.4), facecolor="#0F131C")
    ax.set_facecolor("#0F131C")
    ax.bar(x - w, pooled, w, label="pooled binary Dice (current, non-standard)",
           color="#F87171")
    ax.bar(x, per_v, w, label="per-vertebra Dice (VerSe)", color="#38BDF8")
    ax.bar(x + w, idr, w, label="ID rate @20 mm (VerSe)", color="#6EE7B7")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", color="#E4E7EB", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="#9CA3AF")
    for s in ax.spines.values():
        s.set_color("#1E2636")
    ax.grid(axis="y", color="#1E2636")
    ax.legend(facecolor="#161D2B", edgecolor="#1E2636", labelcolor="#E4E7EB",
              fontsize=9)
    ax.set_title("Same predictions, three metrics. The red bar is what we report today.",
                 color="#E4E7EB", fontsize=11)
    return _fig_to_b64(fig)


def fig_per_vertebra(results, keys):
    """Per-vertebra Dice for a few selected perturbations."""
    sel = [r for r in results if r["perturbation"] in keys]
    fig, ax = plt.subplots(figsize=(11, 4.0), facecolor="#0F131C")
    ax.set_facecolor("#0F131C")
    labels = [v["label"] for v in sel[0]["summary"]["per_vertebra"]]
    x = np.arange(len(labels))
    w = 0.8 / max(len(sel), 1)
    palette = ["#38BDF8", "#6EE7B7", "#FBBF24", "#F87171", "#C084FC"]
    for i, r in enumerate(sel):
        vals = [v["dice"] for v in r["summary"]["per_vertebra"]]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=r["perturbation"],
               color=palette[i % len(palette)])
    ax.set_xticks(x)
    ax.set_xticklabels(["{} ({})".format(v_dict.get(l, l), l) for l in labels],
                       color="#E4E7EB", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("per-vertebra Dice", color="#9CA3AF")
    ax.tick_params(colors="#9CA3AF")
    for s in ax.spines.values():
        s.set_color("#1E2636")
    ax.grid(axis="y", color="#1E2636")
    ax.legend(facecolor="#161D2B", edgecolor="#1E2636", labelcolor="#E4E7EB",
              fontsize=9)
    ax.set_title("Dice is computed per vertebra, then averaged over GT vertebrae",
                 color="#E4E7EB", fontsize=11)
    return _fig_to_b64(fig)


def fig_distance_hist(gt, pred, spacing, label):
    """Where HD95 / NSD actually come from: the surface distance distribution."""
    d = vm.surface_distances_mm(pred == label, gt == label, spacing)
    if d is None:
        return None
    d_p2g, d_g2p = d
    pooled = np.concatenate([d_p2g, d_g2p])
    stats = vm.hausdorff_mm(d_p2g, d_g2p)
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor="#0F131C")
    ax.set_facecolor("#0F131C")
    ax.hist(pooled, bins=60, color="#38BDF8", alpha=0.85)
    for val, col, txt in [
        (stats["assd_mm"], "#6EE7B7", "ASSD {:.2f}".format(stats["assd_mm"])),
        (stats["hd95_pooled_mm"], "#FBBF24",
         "HD95 pooled {:.2f}".format(stats["hd95_pooled_mm"])),
        (stats["hd95_directed_max_mm"], "#C084FC",
         "HD95 directed-max {:.2f}".format(stats["hd95_directed_max_mm"])),
        (stats["hd_mm"], "#F87171", "HD max {:.2f}".format(stats["hd_mm"])),
    ]:
        ax.axvline(val, color=col, ls="--", lw=1.6, label=txt)
    ax.set_xlabel("surface-to-surface distance (mm)", color="#9CA3AF")
    ax.set_ylabel("surface voxels", color="#9CA3AF")
    ax.tick_params(colors="#9CA3AF")
    for s in ax.spines.values():
        s.set_color("#1E2636")
    ax.legend(facecolor="#161D2B", edgecolor="#1E2636", labelcolor="#E4E7EB",
              fontsize=9)
    ax.set_title("vertebra {} ({}), shift_2mm: every distance metric is a "
                 "statistic of THIS histogram".format(label, v_dict.get(label, label)),
                 color="#E4E7EB", fontsize=11)
    return _fig_to_b64(fig)


# ----------------------------------------------------------------------- html
def esc(x):
    return html_mod.escape(str(x))


def fmt(v, spec="{:.4f}"):
    if v is None:
        return "&mdash;"
    if isinstance(v, float) and not np.isfinite(v):
        return "&mdash;"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return spec.format(v)


CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { font-family: ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
       max-width: 1220px; margin: 0 auto; padding: 48px 24px 96px; background: #0A0D12;
       color: #E4E7EB; line-height: 1.65; }
h1 { font-size: 30px; font-weight: 650; letter-spacing: -0.02em; margin: 0 0 6px; }
h2 { font-size: 20px; font-weight: 650; margin: 52px 0 14px; padding-bottom: 8px;
     border-bottom: 1px solid #1E2636; letter-spacing: -0.01em; }
h3 { font-size: 15px; font-weight: 650; margin: 28px 0 10px; color: #CBD5E1; }
p, li { font-size: 14.5px; color: #C9D1DA; }
code { font-family: ui-monospace,"SF Mono",Menlo,Consolas,monospace; font-size: 12.5px;
       background: #161D2B; padding: 1px 5px; border-radius: 4px; color: #A5D8FF; }
pre { background: #0F131C; border: 1px solid #1E2636; border-radius: 8px; padding: 16px;
      overflow-x: auto; font-family: ui-monospace,"SF Mono",Menlo,Consolas,monospace;
      font-size: 12px; line-height: 1.55; color: #C9D1DA; }
table { width: 100%; border-collapse: collapse; margin: 14px 0 6px; background: #0F131C;
        border: 1px solid #1E2636; border-radius: 8px; font-variant-numeric: tabular-nums; }
th, td { padding: 9px 12px; text-align: right; border-bottom: 1px solid #1E2636;
         font-size: 13px; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { background: #161D2B; color: #9CA3AF; font-size: 11px; text-transform: uppercase;
     letter-spacing: 0.06em; font-weight: 600; position: sticky; top: 0; }
tr:last-child td { border-bottom: none; }
tbody tr:hover { background: #131926; }
.wrap { overflow-x: auto; }
.note { background: #0F131C; border-left: 3px solid #38BDF8; border-radius: 0 8px 8px 0;
        padding: 14px 18px; margin: 18px 0; }
.warn { border-left-color: #F87171; }
.ok { border-left-color: #6EE7B7; }
.sub { color: #7D8796; font-size: 13px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(190px,1fr)); gap: 12px;
        margin: 18px 0; }
.card { background: #0F131C; border: 1px solid #1E2636; border-radius: 8px; padding: 15px 17px; }
.card .k { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.07em;
           color: #7D8796; margin-bottom: 7px; }
.card .v { font-size: 23px; font-weight: 650; color: #38BDF8; letter-spacing: -0.02em; }
.hit { color: #6EE7B7; } .miss { color: #F87171; } .bad { color: #F87171; font-weight: 600; }
.pass { color: #6EE7B7; font-weight: 600; }
img { max-width: 100%; border: 1px solid #1E2636; border-radius: 8px; margin: 10px 0;
      display: block; }
.tag { display: inline-block; font-size: 10.5px; text-transform: uppercase; letter-spacing: .06em;
       padding: 2px 7px; border-radius: 4px; background: #161D2B; color: #9CA3AF;
       margin-left: 8px; vertical-align: middle; }
"""


def render_html(payload, figures, out_path):
    H = []
    a = H.append
    a("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    a("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    a("<title>VerSe metric calibration</title><style>{}</style></head><body>".format(CSS))

    a("<h1>VerSe-2021 evaluation, pinned down</h1>")
    a("<p class='sub'>Official primitives (<code>anjany/verse @02b292b</code>) run on real "
      "VerSe validation data. The predictions here are ground truth put through "
      "<b>known</b> perturbations, so every printed number can be checked against what it "
      "must be. This is a metric calibration, not a model score.</p>")

    # ---- section 1: data facts
    a("<h2>1. What the data actually is</h2>")
    a("<p>Protocol decisions depend on these facts, so they are measured, not assumed.</p>")
    for case in payload["cases"]:
        f = case["data_facts"]
        a("<h3>{}</h3>".format(esc(case["case_id"])))
        a("<div class='grid'>")
        a("<div class='card'><div class='k'>source NIfTI spacing</div><div class='v'>{}</div></div>"
          .format(" &times; ".join("{:g}".format(z) for z in f["nii_zooms_mm"])))
        a("<div class='card'><div class='k'>NIfTI orientation</div><div class='v'>{}</div></div>"
          .format(esc("".join(f["nii_axcodes"]))))
        a("<div class='card'><div class='k'>PNG mask dtype</div><div class='v'>{}</div></div>"
          .format(esc(", ".join(f["png_dtypes"]))))
        a("<div class='card'><div class='k'>GT vertebrae</div><div class='v'>{}</div></div>"
          .format(len(f["png_labels"])))
        a("<div class='card'><div class='k'>eval grid</div><div class='v'>{}</div></div>"
          .format(esc("&times;".join(str(s) for s in f["eval_shape"]))))
        a("</div>")
        a("<table><tbody>")
        rows = [
            ("source mask", f["nii_mask_path"]),
            ("NIfTI shape @ {:g} mm".format(f["nii_zooms_mm"][0]), f["nii_shape"]),
            ("NIfTI resampled to 1 mm (order=0)", f["nii_1mm_shape"]),
            ("PNG stack shape (X,Y,Z)", f["png_shape"]),
            ("PNG slices found / expected", "{} / {}".format(
                f["png_rows"], f["png_expected_slices"])),
            ("labels in PNG stack", f["png_labels"]),
            ("labels in resampled NIfTI", f["nii_1mm_labels"]),
            ("voxels pulled from outside FOV by resample (cval=-1024)",
             f["resample_negative_voxels"]),
            ("grid trim to align the two", f["grid_align"]["trimmed"]),
        ]
        for k, v in rows:
            a("<tr><td>{}</td><td>{}</td></tr>".format(esc(k), esc(v)))
        a("</tbody></table>")

        cc = f["png_vs_nii"]
        cls = "ok" if cc["mean_dice"] is not None and cc["mean_dice"] > 0.95 else "warn"
        a("<div class='note {}'>".format(cls))
        a("<b>Cross-check: is the PNG slice stack a faithful 1 mm rendering of the NIfTI?</b><br>"
          "Per-vertebra Dice between the PNG stack and the independently resampled NIfTI: "
          "mean <b>{}</b>, min <b>{}</b> over {} vertebrae. ".format(
              fmt(cc["mean_dice"]), fmt(cc["min_dice"]), cc["n_labels"]))
        a("Pooled foreground Dice {}.".format(fmt(cc["pooled_dice"])))
        if cls == "ok":
            a(" The two agree, so the PNG grid is already 1 mm isotropic and "
              "VerSe's resampling requirement is satisfied by construction on this dataset. "
              "Evaluation runs on the PNG grid.")
        else:
            a(" They do <b>not</b> agree. The PNG grid cannot be assumed to be a 1 mm "
              "rendering of the source and this must be resolved before any number is trusted.")
        a("</div>")

    # ---- section 2: the headline
    a("<h2>2. The headline: three metrics, same predictions</h2>")
    a("<img src='data:image/png;base64,{}' alt='headline'>".format(figures["headline"]))
    a("<div class='note warn'><b>Read the <code>label_shift_+1</code> column.</b> "
      "The masks are pixel-perfect; only the vertebra labels are off by one, which is the "
      "single most common failure mode in vertebra labelling. Pooled foreground Dice "
      "reports <b>1.0000</b>. VerSe per-vertebra Dice reports <b>~0</b> and the ID rate "
      "reports <b>0</b>. Pooled Dice is structurally blind to identity, so it cannot be "
      "used to rank vertebra-labelling models, and it is what "
      "<code>eval_memflowdit_v03.py</code> reports today.</div>")

    # ---- section 3: full metric table
    a("<h2>3. Every metric, every perturbation</h2>")
    for case in payload["cases"]:
        a("<h3>{}<span class='tag'>{} GT vertebrae</span></h3>".format(
            esc(case["case_id"]), len(case["data_facts"]["png_labels"])))
        a("<div class='wrap'><table><thead><tr>")
        cols = ["perturbation", "what it does", "ID rate", "hits",
                "Dice (all GT)", "Dice (ID-gated)", "pooled Dice",
                "HD (mm)", "HD95 dir-max", "HD95 pooled", "ASSD",
                "NSD@1mm", "NSD@2mm", "centroid err"]
        for c in cols:
            a("<th>{}</th>".format(esc(c)))
        a("</tr></thead><tbody>")
        for r in case["results"]:
            s = r["summary"]
            a("<tr>")
            a("<td><code>{}</code></td>".format(esc(r["perturbation"])))
            a("<td style='text-align:left;white-space:normal' class='sub'>{}</td>".format(
                esc(r["description"])))
            a("<td>{}</td>".format(fmt(s["id_rate"], "{:.3f}")))
            a("<td>{}/{}</td>".format(s["id_hits"], s["n_gt_vertebrae"]))
            a("<td>{}</td>".format(fmt(s["dice_all_gt_mean"])))
            a("<td>{}</td>".format(fmt(s["dice_id_gated_mean"])))
            pooled = s["pooled_binary_dice_NONSTANDARD"]
            flag = " class='bad'" if (
                s["dice_all_gt_mean"] is not None
                and pooled - s["dice_all_gt_mean"] > 0.20) else ""
            a("<td{}>{}</td>".format(flag, fmt(pooled)))
            a("<td>{}</td>".format(fmt(s["hd_mm_mean"], "{:.2f}")))
            a("<td>{}</td>".format(fmt(s["hd95_directed_max_mm_mean"], "{:.2f}")))
            a("<td>{}</td>".format(fmt(s["hd95_pooled_mm_mean"], "{:.2f}")))
            a("<td>{}</td>".format(fmt(s["assd_mm_mean"], "{:.2f}")))
            a("<td>{}</td>".format(fmt(s.get("nsd_1mm_mean"))))
            a("<td>{}</td>".format(fmt(s.get("nsd_2mm_mean"))))
            a("<td>{}</td>".format(fmt(s["centroid_error_mm_mean"], "{:.2f}")))
            a("</tr>")
        a("</tbody></table></div>")
        a("<p class='sub'>Red pooled-Dice cells are where pooled Dice overstates "
          "per-vertebra Dice by more than 0.20.</p>")

    # ---- section 4: self-test
    a("<h2>4. Self-test: does the evaluator agree with ground truth it already knows?</h2>")
    a("<p>An evaluator that cannot score a perfect prediction perfectly is not measuring "
      "what it claims. These assertions are checked on every run.</p>")
    a("<table><thead><tr><th>check</th><th>expected</th><th>observed</th><th>result</th>"
      "</tr></thead><tbody>")
    for c in payload["self_test"]:
        cls = "pass" if c["passed"] else "bad"
        a("<tr><td>{}</td><td>{}</td><td>{}</td><td class='{}'>{}</td></tr>".format(
            esc(c["check"]), esc(c["expected"]), esc(c["observed"]),
            cls, "PASS" if c["passed"] else "FAIL"))
    a("</tbody></table>")

    # ---- section 5: visuals
    a("<h2>5. What the perturbations look like</h2>")
    a("<img src='data:image/png;base64,{}' alt='overlays'>".format(figures["overlays"]))
    a("<h3>Dice is per vertebra, then averaged</h3>")
    a("<img src='data:image/png;base64,{}' alt='per vertebra'>".format(
        figures["per_vertebra"]))
    a("<p class='sub'>Under <code>label_shift_+1</code> every bar is ~0 even though the "
      "predicted foreground is exact, because Dice for GT label L is computed against "
      "predicted label L only.</p>")
    if figures.get("dist_hist"):
        a("<h3>Where HD95, ASSD and NSD come from</h3>")
        a("<img src='data:image/png;base64,{}' alt='distance histogram'>".format(
            figures["dist_hist"]))
        a("<p class='sub'>One vertebra under a known 2 mm translation. Each distance metric "
          "is a different statistic of the same surface-distance histogram, which is why the "
          "convention has to be stated: the two HD95 conventions differ on identical data.</p>")

    # ---- section 6: decisions
    a("<h2>6. Decisions this run forces us to make explicit</h2>")
    a("<table><thead><tr><th>question</th><th>choice</th><th>why</th></tr></thead><tbody>")
    for q, ch, why in payload["decisions"]:
        a("<tr><td style='white-space:normal'>{}</td>"
          "<td style='white-space:normal'>{}</td>"
          "<td style='white-space:normal' class='sub'>{}</td></tr>".format(
              esc(q), esc(ch), esc(why)))
    a("</tbody></table>")

    # ---- section 7: code
    a("<h2>7. The code that produced every number above</h2>")
    for title, src in payload["code"]:
        a("<h3>{}</h3><pre>{}</pre>".format(esc(title), esc(src)))

    a("<h2>8. Provenance</h2>")
    a("<table><tbody>")
    for k, v in payload["provenance"].items():
        a("<tr><td>{}</td><td style='white-space:normal'>{}</td></tr>".format(
            esc(k), esc(v)))
    a("</tbody></table>")
    a("</body></html>")

    out_path.write_text("\n".join(H), encoding="utf-8")


# ----------------------------------------------------------------------- main
def main():
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nii_root = pathlib.Path(args.nii_root)
    spacing = np.array([1.0, 1.0, 1.0])  # PNG grid; verified below

    payload = {"cases": [], "self_test": [], "decisions": [], "code": [],
               "provenance": {}}
    figures = {}

    for case_id in args.cases:
        print("[case] {}".format(case_id), flush=True)
        rows = read_manifest(args.manifest, args.split, case_id)
        if not rows:
            print("  [skip] not in manifest split={}".format(args.split))
            continue

        gt_png, png_shape, notes = gt_volume_from_png_stack(rows)

        nii_mask = nii_root / "derivatives" / case_id / "{}_seg-vert_msk.nii.gz".format(case_id)
        nii = nib.load(str(nii_mask))
        import nibabel.orientations as nio
        nii_axcodes = nio.aff2axcodes(nii.affine)
        nii_zooms = [float(z) for z in nii.header.get_zooms()[:3]]
        gt_nii_1mm, n_neg = resample_labels_to_1mm(nii)

        gt_a, gt_b, align = align_grids(gt_png, gt_nii_1mm)
        shared = sorted(set(vm.present_labels(gt_a)) & set(vm.present_labels(gt_b)))
        cc_dice = [float(vm.compute_dice(gt_a == l, gt_b == l)) for l in shared]
        png_vs_nii = {
            "n_labels": len(shared),
            "mean_dice": float(np.mean(cc_dice)) if cc_dice else None,
            "min_dice": float(np.min(cc_dice)) if cc_dice else None,
            "per_label": {int(l): d for l, d in zip(shared, cc_dice)},
            "pooled_dice": vm.pooled_binary_dice(gt_b, gt_a),
        }

        data_facts = {
            "nii_mask_path": str(nii_mask),
            "nii_shape": list(nii.shape),
            "nii_zooms_mm": nii_zooms,
            "nii_axcodes": list(nii_axcodes),
            "nii_1mm_shape": list(gt_nii_1mm.shape),
            "nii_1mm_labels": vm.present_labels(gt_nii_1mm),
            "resample_negative_voxels": n_neg,
            "png_shape": list(png_shape),
            "png_rows": notes["rows_in_manifest"],
            "png_expected_slices": notes["expected_slices"],
            "png_dtypes": notes["dtypes"],
            "png_max_label": notes["max_label"],
            "png_missing": notes["missing"],
            "png_shape_mismatch": [list(x[1]) for x in notes["shape_mismatch"]],
            "png_labels": vm.present_labels(gt_png),
            "grid_align": align,
            "png_vs_nii": png_vs_nii,
            "eval_shape": list(gt_png.shape),
        }

        perts = build_perturbations(spacing)
        variants, results = OrderedDict(), []
        for name, (desc, fn) in perts.items():
            print("  [pert] {}".format(name), flush=True)
            pred = fn(gt_png)
            variants[name] = pred
            summary = vm.evaluate_volume_pair(
                pred, gt_png, spacing, taus_mm=tuple(args.taus_mm))
            results.append({"perturbation": name, "description": desc,
                            "summary": summary})

        payload["cases"].append({"case_id": case_id, "data_facts": data_facts,
                                 "results": results})

        if not figures:
            mid_x = int(np.median(np.argwhere(gt_png > 0)[:, 0]))
            figures["overlays"] = fig_overlays(gt_png, variants, mid_x)
            figures["headline"] = fig_headline(results)
            figures["per_vertebra"] = fig_per_vertebra(
                results, {"identity", "erode_1vox", "shift_2mm", "label_shift_+1"})
            mid_label = vm.present_labels(gt_png)[len(vm.present_labels(gt_png)) // 2]
            figures["dist_hist"] = fig_distance_hist(
                gt_png, variants["shift_2mm"], spacing, mid_label)

            ident = results[0]["summary"]
            lshift = next(r["summary"] for r in results
                          if r["perturbation"] == "label_shift_+1")
            payload["self_test"] = [
                {"check": "identity: per-vertebra Dice == 1",
                 "expected": "1.0", "observed": fmt(ident["dice_all_gt_mean"]),
                 "passed": abs((ident["dice_all_gt_mean"] or 0) - 1.0) < 1e-9},
                {"check": "identity: ID rate == 1",
                 "expected": "1.0", "observed": fmt(ident["id_rate"]),
                 "passed": abs((ident["id_rate"] or 0) - 1.0) < 1e-9},
                {"check": "identity: HD == 0 mm",
                 "expected": "0.00", "observed": fmt(ident["hd_mm_mean"], "{:.2f}"),
                 "passed": ident["hd_mm_mean"] is not None and ident["hd_mm_mean"] < 1e-9},
                {"check": "identity: NSD@1mm == 1",
                 "expected": "1.0", "observed": fmt(ident["nsd_1mm_mean"]),
                 "passed": abs((ident["nsd_1mm_mean"] or 0) - 1.0) < 1e-9},
                {"check": "identity: centroid error == 0 mm",
                 "expected": "0.00",
                 "observed": fmt(ident["centroid_error_mm_mean"], "{:.2f}"),
                 "passed": ident["centroid_error_mm_mean"] is not None and ident["centroid_error_mm_mean"] < 1e-9},
                {"check": "label_shift_+1: pooled Dice stays 1 (blind to identity)",
                 "expected": "1.0",
                 "observed": fmt(lshift["pooled_binary_dice_NONSTANDARD"]),
                 "passed": abs(lshift["pooled_binary_dice_NONSTANDARD"] - 1.0) < 1e-9},
                {"check": "label_shift_+1: VerSe Dice collapses (< 0.05)",
                 "expected": "< 0.05", "observed": fmt(lshift["dice_all_gt_mean"]),
                 "passed": lshift["dice_all_gt_mean"] is not None and lshift["dice_all_gt_mean"] < 0.05},
                {"check": "label_shift_+1: ID rate collapses to 0",
                 "expected": "0.0", "observed": fmt(lshift["id_rate"], "{:.3f}"),
                 "passed": lshift["id_rate"] is not None and lshift["id_rate"] < 1e-9},
                {"check": "PNG stack == resampled NIfTI (mean Dice > 0.95)",
                 "expected": "> 0.95", "observed": fmt(png_vs_nii["mean_dice"]),
                 "passed": (png_vs_nii["mean_dice"] or 0) > 0.95},
            ]

    payload["decisions"] = [
        ("Which grid does evaluation run on?",
         "The 1 mm PNG grid, after proving per-vertebra agreement with the "
         "independently resampled source NIfTI.",
         "VerSe requires 1 mm isotropic. Section 1 measures the source at 1.5 mm and shows "
         "the PNG stack already sits on the 1 mm grid, so no second resampling is applied "
         "and no interpolation error is added."),
        ("Dice pooled or per vertebra?",
         "Per vertebra, averaged over the vertebrae present in GT.",
         "VerSe's own evaluator computes Dice per instance. Section 2 shows pooled Dice "
         "scores an off-by-one labelling as perfect."),
        ("Do mis-identified vertebrae count in Dice?",
         "Yes. `dice_all_gt_mean` is the headline; `dice_id_gated_mean` is a diagnostic only.",
         "Gating Dice on identification hides the failure that matters most. Both are "
         "reported so the gap is visible."),
        ("Which HD95 convention?",
         "max of the two directed 95th percentiles (medpy / MONAI). The pooled-percentile "
         "variant is reported alongside.",
         "The two disagree on identical data, so the choice must be stated rather than "
         "inherited from whichever library got imported."),
        ("Is NSD part of VerSe?",
         "No. Reported as a modern complement, clearly labelled.",
         "VerSe 2021 reports Dice and Hausdorff. NSD comes from the surface-Dice line of "
         "work and must not be presented as a VerSe leaderboard metric."),
        ("Surface definition",
         "mask AND NOT binary_erosion(mask), 6-connected, border_value=1, on a "
         "union-bbox crop with 2-voxel padding.",
         "border_value=1 stops a vertebra truncated by the field of view from growing a "
         "phantom surface along the cut; the padded crop keeps the crop edge from doing "
         "the same."),
        ("Centroids for ID rate",
         "Centre of mass per label on the 1 mm grid.",
         "Matches `data_utilities.calc_centroids`. The VerSe centroid JSON is the primary "
         "source when available; centre of mass is the standard surrogate from a mask."),
        ("Label resampling safety",
         "Clip to >= 0 after `resample_nib`.",
         "`resample_nib` hardcodes cval=-1024 for CT. On a label map, out-of-FOV voxels "
         "come back negative and would wrap to ~64000 in an unsigned cast."),
        ("np.bool in the vendored evaluator",
         "Aliased at import time; the vendored file is left byte-identical to upstream.",
         "numpy >= 1.24 removed np.bool, so `compute_dice` would raise. Patching vendored "
         "third-party code in place would break provenance against the upstream commit."),
    ]

    payload["code"] = [
        ("verse_metrics.evaluate_volume_pair", inspect.getsource(vm.evaluate_volume_pair)),
        ("verse_metrics.id_rate + centroids_mm",
         inspect.getsource(vm.centroids_mm) + "\n" + inspect.getsource(vm.id_rate)),
        ("verse_metrics.surface_mask / surface_distances_mm / hausdorff_mm / nsd",
         inspect.getsource(vm.surface_mask) + "\n"
         + inspect.getsource(vm.surface_distances_mm) + "\n"
         + inspect.getsource(vm.hausdorff_mm) + "\n" + inspect.getsource(vm.nsd)),
        ("official VerSe compute_dice (vendored, unmodified)",
         inspect.getsource(vm.compute_dice)),
        ("official VerSe get_hits (vendored, unmodified)",
         inspect.getsource(vm.get_hits)),
        ("the current non-standard pooled Dice, for comparison",
         inspect.getsource(vm.pooled_binary_dice)),
    ]

    import subprocess
    def _sh(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, cwd=str(_PROJECT_ROOT),
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "n/a"

    payload["provenance"] = {
        "generated": _sh("date -Is"),
        "git HEAD": _sh("git rev-parse HEAD"),
        "git branch": _sh("git rev-parse --abbrev-ref HEAD"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "nibabel": nib.__version__,
        "official evaluator": "lib/evaluators/verse2021_3d (anjany/verse @02b292b, MIT)",
        "manifest": args.manifest,
        "nii root": str(nii_root),
        "cases": ", ".join(args.cases),
        "nature of predictions": "GT + known perturbations (metric calibration, "
                                 "NOT a model evaluation)",
    }

    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    render_html(payload, figures, out_dir / "report.html")

    n_fail = sum(1 for c in payload["self_test"] if not c["passed"])
    print("\n[self-test] {}/{} passed".format(
        len(payload["self_test"]) - n_fail, len(payload["self_test"])))
    for c in payload["self_test"]:
        print("  [{}] {} -> expected {}, observed {}".format(
            "PASS" if c["passed"] else "FAIL", c["check"], c["expected"], c["observed"]))
    print("\n[out] {}".format(out_dir / "results.json"))
    print("[out] {}".format(out_dir / "report.html"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
