#!/usr/bin/env python3
"""Thin original-space VerSe-style 3D export and evaluation utilities.

This module deliberately contains no model imports.  It consumes the existing
original-image multiclass masks emitted by the frozen 2D evaluator, restores
the source NIfTI voxel order, and computes scan-equal 3D metrics.
"""

from __future__ import print_function

import csv
import gzip
import hashlib
import json
import math
import os
import pathlib
import struct
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


VERSE_LABELS = tuple(list(range(1, 26)) + [28])
DEV8_CASES = (
    "sub-verse004",
    "sub-verse005",
    "sub-verse033",
    "sub-verse043",
    "sub-verse060",
    "sub-verse064",
    "sub-verse082",
    "sub-verse410_split-verse267",
)
LOCKED_CASES = ("sub-verse010", "sub-verse011", "sub-verse013")


def sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _open_nifti(path, mode="rb"):
    if not str(path).lower().endswith(".gz"):
        return open(path, mode)
    if "w" in mode:
        return gzip.GzipFile(filename=os.fspath(path), mode=mode, mtime=0)
    return gzip.open(path, mode)


def _qform_affine(header, endian, pixdim):
    b, c, d = struct.unpack(endian + "3f", header[256:268])
    x, y, z = struct.unpack(endian + "3f", header[268:280])
    a = math.sqrt(max(1.0 - b * b - c * c - d * d, 0.0))
    qfac = -1.0 if pixdim[0] < 0 else 1.0
    rotation = np.asarray(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ],
        dtype=np.float64,
    )
    spacing = np.asarray([pixdim[1], pixdim[2], pixdim[3] * qfac], dtype=np.float64)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = rotation * spacing[None, :]
    affine[:3, 3] = [x, y, z]
    return affine


def axis_codes(affine):
    matrix = np.asarray(affine, dtype=np.float64)[:3, :3]
    positive = ("R", "A", "S")
    negative = ("L", "P", "I")
    codes = []
    used = set()
    for input_axis in range(3):
        world_axis = int(np.argmax(np.abs(matrix[:, input_axis])))
        if world_axis in used:
            raise ValueError("affine orientation is not a signed axis permutation")
        used.add(world_axis)
        codes.append(positive[world_axis] if matrix[world_axis, input_axis] >= 0 else negative[world_axis])
    return tuple(codes)


def orientation_to_canonical(affine):
    matrix = np.asarray(affine, dtype=np.float64)[:3, :3]
    input_for_world = []
    signs = []
    for world_axis in range(3):
        candidates = [
            input_axis
            for input_axis in range(3)
            if int(np.argmax(np.abs(matrix[:, input_axis]))) == world_axis
        ]
        if len(candidates) != 1:
            raise ValueError("affine orientation is ambiguous")
        input_axis = candidates[0]
        input_for_world.append(input_axis)
        signs.append(1 if matrix[world_axis, input_axis] >= 0 else -1)
    return tuple(input_for_world), tuple(signs)


@dataclass
class NiftiHeader:
    path: str
    raw_prefix: bytes
    endian: str
    shape: tuple
    datatype: int
    bitpix: int
    vox_offset: int
    affine: np.ndarray
    spacing: tuple
    axis_codes: tuple
    permutation: tuple
    signs: tuple

    @classmethod
    def read(cls, path):
        path = os.path.abspath(os.fspath(path))
        with _open_nifti(path, "rb") as handle:
            prefix = handle.read(4096)
        if len(prefix) < 348:
            raise ValueError("truncated NIfTI header: {}".format(path))
        if struct.unpack("<i", prefix[:4])[0] == 348:
            endian = "<"
        elif struct.unpack(">i", prefix[:4])[0] == 348:
            endian = ">"
        else:
            raise ValueError("not a NIfTI-1 header: {}".format(path))
        dims = struct.unpack(endian + "8h", prefix[40:56])
        ndim = int(dims[0])
        if ndim != 3:
            raise ValueError("only 3D NIfTI is supported: {} dims={}".format(path, dims))
        shape = tuple(int(x) for x in dims[1:4])
        datatype = int(struct.unpack(endian + "h", prefix[70:72])[0])
        bitpix = int(struct.unpack(endian + "h", prefix[72:74])[0])
        pixdim = struct.unpack(endian + "8f", prefix[76:108])
        vox_offset = int(round(struct.unpack(endian + "f", prefix[108:112])[0]))
        if vox_offset < 352:
            raise ValueError("invalid vox_offset {} in {}".format(vox_offset, path))
        qform_code, sform_code = struct.unpack(endian + "2h", prefix[252:256])
        if sform_code > 0:
            affine = np.asarray(
                [
                    struct.unpack(endian + "4f", prefix[280:296]),
                    struct.unpack(endian + "4f", prefix[296:312]),
                    struct.unpack(endian + "4f", prefix[312:328]),
                    (0.0, 0.0, 0.0, 1.0),
                ],
                dtype=np.float64,
            )
        elif qform_code > 0:
            affine = _qform_affine(prefix, endian, pixdim)
        else:
            raise ValueError("NIfTI has neither qform nor sform: {}".format(path))
        spacing = tuple(float(np.linalg.norm(affine[:3, axis])) for axis in range(3))
        permutation, signs = orientation_to_canonical(affine)
        with _open_nifti(path, "rb") as handle:
            raw_prefix = handle.read(vox_offset)
        if len(raw_prefix) != vox_offset:
            raise ValueError("NIfTI prefix shorter than vox_offset: {}".format(path))
        return cls(
            path=path,
            raw_prefix=raw_prefix,
            endian=endian,
            shape=shape,
            datatype=datatype,
            bitpix=bitpix,
            vox_offset=vox_offset,
            affine=affine,
            spacing=spacing,
            axis_codes=axis_codes(affine),
            permutation=permutation,
            signs=signs,
        )

    def to_json(self):
        return {
            "path": self.path,
            "shape": list(self.shape),
            "datatype": self.datatype,
            "bitpix": self.bitpix,
            "vox_offset": self.vox_offset,
            "affine": self.affine.tolist(),
            "spacing_mm": list(self.spacing),
            "axis_codes": list(self.axis_codes),
            "canonical_permutation": list(self.permutation),
            "canonical_signs": list(self.signs),
        }


_DTYPES = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
}


def load_nifti(path, header=None):
    header = header or NiftiHeader.read(path)
    try:
        dtype = np.dtype(_DTYPES[header.datatype]).newbyteorder(header.endian)
    except KeyError as exc:
        raise ValueError("unsupported NIfTI datatype {}".format(header.datatype)) from exc
    count = int(np.prod(header.shape))
    with _open_nifti(path, "rb") as handle:
        handle.seek(header.vox_offset)
        payload = handle.read(count * dtype.itemsize)
    if len(payload) != count * dtype.itemsize:
        raise ValueError("truncated NIfTI payload: {}".format(path))
    return np.frombuffer(payload, dtype=dtype, count=count).reshape(header.shape, order="F")


def canonical_from_source(array, header):
    array = np.asarray(array)
    if tuple(array.shape) != tuple(header.shape):
        raise ValueError("source shape mismatch")
    result = np.transpose(array, axes=header.permutation)
    for axis, sign in enumerate(header.signs):
        if sign < 0:
            result = np.flip(result, axis=axis)
    return result


def source_from_canonical(array, header):
    result = np.asarray(array)
    expected = tuple(header.shape[index] for index in header.permutation)
    if tuple(result.shape) != expected:
        raise ValueError("canonical shape mismatch: {} vs {}".format(result.shape, expected))
    for axis, sign in enumerate(header.signs):
        if sign < 0:
            result = np.flip(result, axis=axis)
    inverse = np.argsort(np.asarray(header.permutation))
    result = np.transpose(result, axes=tuple(int(x) for x in inverse))
    if tuple(result.shape) != tuple(header.shape):
        raise AssertionError("inverse orientation changed shape")
    return result


def write_uint8_like(path, array, reference_header):
    array = np.asarray(array)
    if tuple(array.shape) != tuple(reference_header.shape):
        raise ValueError("prediction/source NIfTI shape mismatch")
    if reference_header.datatype != 2 or reference_header.bitpix != 8:
        raise ValueError("reference GT NIfTI must be uint8 for byte-preserving export")
    if np.any(array < 0) or np.any(array > 255):
        raise ValueError("prediction labels exceed uint8")
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.asarray(array, dtype=np.uint8).tobytes(order="F")
    with _open_nifti(path, "wb") as handle:
        handle.write(reference_header.raw_prefix)
        handle.write(payload)


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_centroids(path, header):
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload or "direction" not in payload[0]:
        raise ValueError("invalid VerSe centroid JSON")
    direction = tuple(str(x) for x in payload[0]["direction"])
    if direction != tuple(header.axis_codes):
        raise ValueError("centroid direction {} != source axis codes {}".format(direction, header.axis_codes))
    rows = {}
    for item in payload[1:]:
        label = int(item["label"])
        if label in rows:
            raise ValueError("duplicate centroid label {}".format(label))
        source_voxel = np.asarray([item["X"], item["Y"], item["Z"]], dtype=np.float64)
        world = (header.affine @ np.r_[source_voxel, 1.0])[:3]
        rows[label] = {
            "label": label,
            "source_voxel_xyz": source_voxel.tolist(),
            "world_mm": world.tolist(),
        }
    return {"direction": list(direction), "rows": rows}


def _parse_shape(value):
    return tuple(int(x) for x in str(value).lower().replace(" ", "").split("x"))


def build_scan_geometry(case_id, slice_rows, case_row, validate_png_reader=None):
    if case_id in LOCKED_CASES:
        raise RuntimeError("locked case selected: {}".format(case_id))
    rows = sorted(slice_rows, key=lambda row: int(row["slice_idx"]))
    if not rows:
        raise ValueError("scan has no slice rows: {}".format(case_id))
    indices = [int(row["slice_idx"]) for row in rows]
    if indices != list(range(len(indices))):
        raise ValueError("slice mapping must be complete and start at zero: {}".format(case_id))
    if len({row["image_path"] for row in rows}) != len(rows):
        raise ValueError("duplicate image path: {}".format(case_id))
    if len({row["mask_path"] for row in rows}) != len(rows):
        raise ValueError("duplicate mask path: {}".format(case_id))
    gt_path = os.path.abspath(case_row["mask_nii_path"])
    ct_path = os.path.abspath(case_row["image_nii_path"])
    centroid_path = gt_path.replace("_seg-vert_msk.nii.gz", "_seg-subreg_ctd.json")
    if not os.path.isfile(centroid_path):
        raise FileNotFoundError("signed centroid JSON is missing: {}".format(centroid_path))
    gt_header = NiftiHeader.read(gt_path)
    ct_header = NiftiHeader.read(ct_path)
    canonical_shape = tuple(gt_header.shape[index] for index in gt_header.permutation)
    if canonical_shape != _parse_shape(case_row["canonical_shape"]):
        raise ValueError("case_metadata canonical shape mismatch")
    if len(rows) != canonical_shape[0]:
        raise ValueError("slice count does not cover complete canonical sagittal FoV")
    expected_hw = (canonical_shape[2], canonical_shape[1])
    for row in rows:
        if (int(row["image_height"]), int(row["image_width"])) != expected_hw:
            raise ValueError("PNG shape metadata does not match canonical NIfTI")
        if row["png_axis"] != "sagittal_x":
            raise ValueError("unexpected PNG axis")
    if tuple(ct_header.shape) != tuple(gt_header.shape):
        raise ValueError("CT/GT source shape mismatch")
    if not np.allclose(ct_header.affine, gt_header.affine, rtol=0.0, atol=1e-4):
        raise ValueError("CT/GT affine mismatch")
    if not np.allclose(ct_header.spacing, gt_header.spacing, rtol=0.0, atol=1e-5):
        raise ValueError("CT/GT spacing mismatch")
    centroids = load_centroids(centroid_path, gt_header)
    if validate_png_reader is not None:
        gt = canonical_from_source(load_nifti(gt_path, gt_header), gt_header)
        for index, row in enumerate(rows):
            mask = np.asarray(validate_png_reader(row["mask_path"]))
            expected = gt[index, :, :].T
            if mask.shape != expected.shape or not np.array_equal(mask, expected):
                raise ValueError("PNG/GT NIfTI round-trip mismatch at {} slice {}".format(case_id, index))
    return {
        "case_id": case_id,
        "rows": rows,
        "gt_path": gt_path,
        "ct_path": ct_path,
        "centroid_path": centroid_path,
        "gt_sha256": sha256_file(gt_path),
        "ct_sha256": sha256_file(ct_path),
        "centroid_sha256": sha256_file(centroid_path),
        "gt_header": gt_header,
        "ct_header": ct_header,
        "centroids": centroids,
        "canonical_shape": canonical_shape,
        "expected_png_hw": expected_hw,
        "mapping_digest": hashlib.sha256(
            json.dumps(
                [[int(row["slice_idx"]), row["image_path"], row["mask_path"]] for row in rows],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def prepare_dev8_geometry(slice_manifest, case_metadata, png_reader=None):
    all_slice_rows = read_csv(slice_manifest)
    all_case_rows = read_csv(case_metadata)
    case_lookup = {row["case_id"]: row for row in all_case_rows}
    selected = {case_id: [] for case_id in DEV8_CASES}
    for row in all_slice_rows:
        if row["case_id"] in LOCKED_CASES:
            continue
        if row["case_id"] in selected:
            selected[row["case_id"]].append(row)
    if sum(len(rows) for rows in selected.values()) != 1123:
        raise ValueError("Dev8 row count must be exactly 1123")
    result = {}
    for case_id in DEV8_CASES:
        if case_id not in case_lookup:
            raise ValueError("case_metadata missing {}".format(case_id))
        result[case_id] = build_scan_geometry(
            case_id,
            selected[case_id],
            case_lookup[case_id],
            validate_png_reader=png_reader,
        )
    return result


class PredictionVolumeCollector:
    def __init__(self, geometries):
        self.geometries = geometries
        self.canonical = {
            case_id: np.zeros(info["canonical_shape"], dtype=np.uint8)
            for case_id, info in geometries.items()
        }
        self.seen = {case_id: set() for case_id in geometries}
        self.capture_calls = 0

    def add(self, case_id, slice_idx, label_mask):
        case_id = str(case_id)
        slice_idx = int(slice_idx)
        if case_id not in self.geometries:
            raise ValueError("unexpected case {}".format(case_id))
        if slice_idx in self.seen[case_id]:
            raise ValueError("duplicate predicted plane {}:{}".format(case_id, slice_idx))
        mask = np.asarray(label_mask)
        if tuple(mask.shape) != tuple(self.geometries[case_id]["expected_png_hw"]):
            raise ValueError("predicted plane shape mismatch")
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError("predicted multiclass plane must have an integer dtype")
        labels = set(int(x) for x in np.unique(mask))
        invalid = labels.difference({0}.union(VERSE_LABELS))
        if invalid:
            raise ValueError("invalid predicted labels: {}".format(sorted(invalid)))
        self.canonical[case_id][slice_idx, :, :] = mask.T.astype(np.uint8, copy=False)
        self.seen[case_id].add(slice_idx)
        self.capture_calls += 1

    def assert_complete(self):
        for case_id, info in self.geometries.items():
            expected = set(range(info["canonical_shape"][0]))
            if self.seen[case_id] != expected:
                raise ValueError(
                    "incomplete predicted planes for {}: missing={} extra={}".format(
                        case_id,
                        sorted(expected.difference(self.seen[case_id])),
                        sorted(self.seen[case_id].difference(expected)),
                    )
                )

    def source_volume(self, case_id):
        self.assert_complete()
        info = self.geometries[case_id]
        return source_from_canonical(self.canonical[case_id], info["gt_header"])


def dice_score(gt, pred):
    gt = np.asarray(gt, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    gt_count = int(gt.sum())
    pred_count = int(pred.sum())
    if gt_count <= 0:
        raise ValueError("GT label must be non-empty")
    if pred_count <= 0:
        return 0.0
    return float(2.0 * np.logical_and(gt, pred).sum() / float(gt_count + pred_count))


def world_points(indices, affine):
    indices = np.asarray(indices, dtype=np.float64)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices must have shape [N,3]")
    return indices @ np.asarray(affine, dtype=np.float64)[:3, :3].T + np.asarray(affine)[:3, 3]


def mask_centroid_world(mask, affine):
    points = np.argwhere(np.asarray(mask, dtype=bool))
    if not len(points):
        return None
    return world_points(points.mean(axis=0, keepdims=True), affine)[0]


def surface_world(mask, affine):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros((0, 3), dtype=np.float64)
    structure = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, iterations=1, border_value=0)
    surface = np.logical_xor(mask, eroded)
    return world_points(np.argwhere(surface), affine)


def maximum_hausdorff_mm(gt, pred, affine):
    gt_points = surface_world(gt, affine)
    pred_points = surface_world(pred, affine)
    if not len(gt_points) or not len(pred_points):
        return None
    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)
    pred_to_gt = gt_tree.query(pred_points, k=1)[0]
    gt_to_pred = pred_tree.query(gt_points, k=1)[0]
    return float(max(float(np.max(pred_to_gt)), float(np.max(gt_to_pred))))


def _finite(values):
    return [float(x) for x in values if x is not None and math.isfinite(float(x))]


def _mean(values):
    values = _finite(values)
    return float(np.mean(values)) if values else None


def _median(values):
    values = _finite(values)
    return float(np.median(values)) if values else None


def evaluate_scan(case_id, gt_volume, pred_volume, affine, signed_centroids):
    gt_volume = np.asarray(gt_volume)
    pred_volume = np.asarray(pred_volume)
    if gt_volume.shape != pred_volume.shape:
        raise ValueError("GT/pred volume shape mismatch")
    gt_labels = sorted(int(x) for x in np.unique(gt_volume) if int(x) > 0)
    pred_labels = sorted(int(x) for x in np.unique(pred_volume) if int(x) > 0)
    if not gt_labels:
        raise ValueError("scan has no GT vertebra labels")
    invalid = set(gt_labels + pred_labels).difference(VERSE_LABELS)
    if invalid:
        raise ValueError("unsupported anatomical labels {}".format(sorted(invalid)))
    missing_centroids = sorted(set(gt_labels).difference(signed_centroids["rows"]))
    if missing_centroids:
        raise ValueError("signed centroids missing GT labels {}".format(missing_centroids))
    pred_centroids = {
        label: mask_centroid_world(pred_volume == label, affine) for label in pred_labels
    }
    pred_centroids = {label: point for label, point in pred_centroids.items() if point is not None}
    per_label = []
    for label in gt_labels:
        gt_mask = gt_volume == label
        pred_mask = pred_volume == label
        missing = not bool(pred_mask.any())
        dice = dice_score(gt_mask, pred_mask)
        hd = maximum_hausdorff_mm(gt_mask, pred_mask, affine) if not missing else None
        gt_point = np.asarray(signed_centroids["rows"][label]["world_mm"], dtype=np.float64)
        distances = {
            int(pred_label): float(np.linalg.norm(point - gt_point))
            for pred_label, point in pred_centroids.items()
        }
        nearest_label = min(distances, key=lambda key: (distances[key], key)) if distances else None
        same_label_distance = distances.get(label)
        hit = bool(
            same_label_distance is not None
            and nearest_label == label
            and same_label_distance < 20.0
        )
        if same_label_distance is None:
            reason = "missing_same_label_prediction"
        elif nearest_label != label:
            reason = "same_label_not_nearest"
        elif not same_label_distance < 20.0:
            reason = "same_label_distance_not_strictly_below_20mm"
        else:
            reason = "hit"
        per_label.append({
            "case_id": case_id,
            "label": label,
            "gt_voxels": int(gt_mask.sum()),
            "pred_voxels": int(pred_mask.sum()),
            "missing": missing,
            "dice": dice,
            "maximum_hausdorff_mm": hd,
            "gt_centroid_world_mm": gt_point.tolist(),
            "pred_centroid_world_mm": pred_centroids[label].tolist() if label in pred_centroids else None,
            "same_label_distance_mm": same_label_distance,
            "nearest_prediction_label": nearest_label,
            "nearest_prediction_distance_mm": distances.get(nearest_label) if nearest_label is not None else None,
            "distance_matrix_row_mm": {str(key): value for key, value in sorted(distances.items())},
            "identification_hit_strict_20mm": hit,
            "identification_failure_reason": reason,
        })
    finite_hd = _finite([row["maximum_hausdorff_mm"] for row in per_label])
    hit_distances = _finite([
        row["same_label_distance_mm"]
        if row["identification_hit_strict_20mm"] else None
        for row in per_label
    ])
    n_scan = len(gt_labels)
    missing_count = sum(bool(row["missing"]) for row in per_label)
    failed_count = sum(not bool(row["identification_hit_strict_20mm"]) for row in per_label)
    scan = {
        "case_id": case_id,
        "N_scan": n_scan,
        "gt_fov_labels": gt_labels,
        "predicted_labels": pred_labels,
        "missing_gt_labels": [row["label"] for row in per_label if row["missing"]],
        "extra_predicted_labels": sorted(set(pred_labels).difference(gt_labels)),
        "scan_dice": float(np.mean([row["dice"] for row in per_label])),
        "scan_maximum_hd_mm_mean_finite": float(np.mean(finite_hd)) if finite_hd else None,
        "scan_maximum_hd_mm_median_finite": float(np.median(finite_hd)) if finite_hd else None,
        "scan_maximum_hd_mm_max_finite": float(np.max(finite_hd)) if finite_hd else None,
        "hd_finite_count": len(finite_hd),
        "hd_missing_count": missing_count,
        "identification_hits": n_scan - failed_count,
        "identification_failures": failed_count,
        "identification_rate": float((n_scan - failed_count) / float(n_scan)),
        "dmean_mm_hits_only": float(np.mean(hit_distances)) if hit_distances else None,
        "dmean_hit_count": len(hit_distances),
        "dmean_excluded_count": failed_count,
        "penalty_100mm_hd_scan_mean": float(np.mean([
            row["maximum_hausdorff_mm"] if row["maximum_hausdorff_mm"] is not None else 100.0
            for row in per_label
        ])),
        "penalty_1000mm_dmean_scan_mean": float(np.mean([
            row["same_label_distance_mm"]
            if row["identification_hit_strict_20mm"] else 1000.0
            for row in per_label
        ])),
    }
    return {"per_label": per_label, "scan": scan}


def summarize_cohort(scan_results):
    if not scan_results:
        raise ValueError("empty cohort")
    case_ids = [row["scan"]["case_id"] for row in scan_results]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate scan result")
    scans = [row["scan"] for row in scan_results]
    all_labels = [item for row in scan_results for item in row["per_label"]]
    finite_scan_hd = _finite([row["scan_maximum_hd_mm_mean_finite"] for row in scans])
    finite_scan_dmean = _finite([row["dmean_mm_hits_only"] for row in scans])
    return {
        "case_count": len(scans),
        "case_ids": case_ids,
        "main": {
            "scan_equal_dice_mean": float(np.mean([row["scan_dice"] for row in scans])),
            "scan_equal_dice_median": float(np.median([row["scan_dice"] for row in scans])),
            "scan_equal_maximum_hd_mm_mean_unpenalized": float(np.mean(finite_scan_hd)) if finite_scan_hd else None,
            "scan_equal_maximum_hd_mm_median_unpenalized": float(np.median(finite_scan_hd)) if finite_scan_hd else None,
            "hd_included_scan_count": len(finite_scan_hd),
            "hd_excluded_scan_count": len(scans) - len(finite_scan_hd),
            "scan_equal_identification_rate_mean": float(np.mean([row["identification_rate"] for row in scans])),
            "scan_equal_identification_rate_median": float(np.median([row["identification_rate"] for row in scans])),
            "scan_equal_dmean_mm_mean_hits_only": float(np.mean(finite_scan_dmean)) if finite_scan_dmean else None,
            "scan_equal_dmean_mm_median_hits_only": float(np.median(finite_scan_dmean)) if finite_scan_dmean else None,
            "dmean_included_scan_count": len(finite_scan_dmean),
            "dmean_excluded_scan_count": len(scans) - len(finite_scan_dmean),
        },
        "sensitivity": {
            "penalty_100mm_hd_scan_equal_mean": float(np.mean([row["penalty_100mm_hd_scan_mean"] for row in scans])),
            "penalty_100mm_hd_scan_equal_median": float(np.median([row["penalty_100mm_hd_scan_mean"] for row in scans])),
            "penalty_1000mm_dmean_scan_equal_mean": float(np.mean([row["penalty_1000mm_dmean_scan_mean"] for row in scans])),
            "penalty_1000mm_dmean_scan_equal_median": float(np.median([row["penalty_1000mm_dmean_scan_mean"] for row in scans])),
        },
        "diagnostic": {
            "pooled_per_vertebra_dice_mean": float(np.mean([row["dice"] for row in all_labels])),
            "pooled_per_vertebra_dice_median": float(np.median([row["dice"] for row in all_labels])),
            "missing_count": int(sum(bool(row["missing"]) for row in all_labels)),
            "missing_rate": float(sum(bool(row["missing"]) for row in all_labels) / float(len(all_labels))),
        },
        "per_scan": scans,
    }


def _label_voxel_counts(volume):
    labels, counts = np.unique(np.asarray(volume), return_counts=True)
    return {
        str(int(label)): int(count)
        for label, count in zip(labels.tolist(), counts.tolist())
        if int(label) > 0
    }


def _write_jsonl(path, rows):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def write_sha256_manifest(root, paths, name="SHA256SUMS.txt"):
    root = pathlib.Path(root).resolve()
    manifest = root / name
    rows = []
    for path in sorted({pathlib.Path(path).resolve() for path in paths}):
        if path == manifest:
            continue
        rows.append("{}  {}".format(sha256_file(path), path.relative_to(root).as_posix()))
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest


def finalize_export(
    output_root,
    collector,
    arm,
    step,
    checkpoint_path,
    checkpoint_sha256,
    contract_path,
    contract_sha256,
    native_summary_path,
    native_entry_path,
    native_entry_sha256,
    command_argv,
    environment,
    observer_additional_calls,
):
    """Persist original-space volumes and separate main/sensitivity summaries."""
    output_root = pathlib.Path(output_root).resolve()
    native_summary_path = pathlib.Path(native_summary_path).resolve()
    if not native_summary_path.is_file():
        raise FileNotFoundError("native evaluator summary is missing")
    if int(observer_additional_calls) != 0:
        raise RuntimeError("thin exporter must add zero model/DiT/Flow calls")
    collector.assert_complete()
    export_root = output_root / "verse2021_3d"
    if export_root.exists():
        raise FileExistsError("VerSe export root already exists: {}".format(export_root))
    prediction_root = export_root / "predictions"
    sidecar_root = export_root / "sidecars"
    scan_root = export_root / "per_scan"
    prediction_root.mkdir(parents=True)
    sidecar_root.mkdir(parents=True)
    scan_root.mkdir(parents=True)

    all_per_label = []
    scan_results = []
    created = []
    for case_id in DEV8_CASES:
        info = collector.geometries[case_id]
        header = info["gt_header"]
        prediction = collector.source_volume(case_id)
        gt_volume = load_nifti(info["gt_path"], header)
        prediction_path = prediction_root / "{}_pred-multiclass.nii.gz".format(case_id)
        write_uint8_like(prediction_path, prediction, header)
        prediction_header = NiftiHeader.read(prediction_path)
        prediction_roundtrip = load_nifti(prediction_path, prediction_header)
        if prediction_header.shape != header.shape:
            raise AssertionError("prediction NIfTI shape drift")
        if not np.array_equal(prediction_header.affine, header.affine):
            raise AssertionError("prediction NIfTI affine drift")
        if prediction_header.axis_codes != header.axis_codes:
            raise AssertionError("prediction NIfTI axis drift")
        if not np.array_equal(prediction_roundtrip, prediction):
            raise AssertionError("prediction NIfTI payload round-trip drift")
        result = evaluate_scan(
            case_id,
            gt_volume,
            prediction,
            header.affine,
            info["centroids"],
        )
        all_per_label.extend(result["per_label"])
        scan_results.append(result)
        scan_payload = {
            "schema": "diffusionsnake.verse2021_3d_per_scan.v1",
            "arm": str(arm),
            "step": int(step),
            "source": {
                "ct_nifti_path": info["ct_path"],
                "ct_nifti_sha256": info["ct_sha256"],
                "gt_nifti_path": info["gt_path"],
                "gt_nifti_sha256": info["gt_sha256"],
                "centroid_json_path": info["centroid_path"],
                "centroid_json_sha256": info["centroid_sha256"],
            },
            "geometry": header.to_json(),
            "prediction": {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
                "shape": list(prediction_header.shape),
                "dtype": "uint8",
                "affine": prediction_header.affine.tolist(),
                "axis_codes": list(prediction_header.axis_codes),
                "spacing_mm": list(prediction_header.spacing),
                "slice_count": len(info["rows"]),
                "slice_mapping_digest": info["mapping_digest"],
                "label_voxel_counts": _label_voxel_counts(prediction),
            },
            "metrics": result["scan"],
            "per_label": result["per_label"],
        }
        scan_path = scan_root / "{}.json".format(case_id)
        dump_json(scan_path, scan_payload)
        sidecar_path = sidecar_root / "{}.json".format(case_id)
        dump_json(sidecar_path, {
            "schema": "diffusionsnake.verse2021_3d_export_sidecar.v1",
            "arm": str(arm),
            "step": int(step),
            "checkpoint_path": os.path.abspath(os.fspath(checkpoint_path)),
            "checkpoint_sha256": str(checkpoint_sha256),
            "parameter_identity": {
                "architecture_arm": str(arm),
                "checkpoint_sha256": str(checkpoint_sha256),
            },
            "case_id": case_id,
            "gt_fov_labels": result["scan"]["gt_fov_labels"],
            "missing_gt_labels": result["scan"]["missing_gt_labels"],
            "extra_predicted_labels": result["scan"]["extra_predicted_labels"],
            "source": scan_payload["source"],
            "prediction": scan_payload["prediction"],
            "native_evaluator_summary_path": str(native_summary_path),
            "native_evaluator_summary_sha256": sha256_file(native_summary_path),
            "observer_additional_model_dit_flow_calls": 0,
        })
        created.extend([prediction_path, scan_path, sidecar_path])

    per_label_path = export_root / "per_vertebra.jsonl"
    _write_jsonl(per_label_path, all_per_label)
    cohort = summarize_cohort(scan_results)
    main_path = export_root / "cohort_main.json"
    sensitivity_path = export_root / "cohort_penalty_sensitivity.json"
    dump_json(main_path, {
        "schema": "diffusionsnake.verse2021_3d_cohort_main.v1",
        "arm": str(arm),
        "step": int(step),
        "oracle_conditioned": True,
        "development_cohort": True,
        "native_3d_model": False,
        "main": cohort["main"],
        "diagnostic": cohort["diagnostic"],
        "per_scan": cohort["per_scan"],
        "penalty_values_present": False,
    })
    dump_json(sensitivity_path, {
        "schema": "diffusionsnake.verse2021_3d_penalty_sensitivity.v1",
        "arm": str(arm),
        "step": int(step),
        "sensitivity_only": True,
        "penalty_100mm_hd": {
            key: value for key, value in cohort["sensitivity"].items()
            if key.startswith("penalty_100mm")
        },
        "penalty_1000mm_dmean": {
            key: value for key, value in cohort["sensitivity"].items()
            if key.startswith("penalty_1000mm")
        },
        "per_scan": [{
            "case_id": row["case_id"],
            "N_scan": row["N_scan"],
            "penalty_100mm_hd_scan_mean": row["penalty_100mm_hd_scan_mean"],
            "penalty_1000mm_dmean_scan_mean": row["penalty_1000mm_dmean_scan_mean"],
        } for row in cohort["per_scan"]],
        "main_unpenalized_values_present": False,
    })
    identity_path = export_root / "evaluation_identity.json"
    dump_json(identity_path, {
        "schema": "diffusionsnake.verse2021_3d_evaluation_identity.v1",
        "arm": str(arm),
        "step": int(step),
        "checkpoint_path": os.path.abspath(os.fspath(checkpoint_path)),
        "checkpoint_sha256": str(checkpoint_sha256),
        "contract_path": os.path.abspath(os.fspath(contract_path)),
        "contract_sha256": str(contract_sha256),
        "native_entry_path": os.path.abspath(os.fspath(native_entry_path)),
        "native_entry_sha256": str(native_entry_sha256),
        "native_summary_path": str(native_summary_path),
        "native_summary_sha256": sha256_file(native_summary_path),
        "command_argv": list(command_argv),
        "environment": dict(environment),
        "dev8_case_ids": list(DEV8_CASES),
        "dev8_case_count": 8,
        "dev8_slice_count": collector.capture_calls,
        "locked_case_opens": 0,
        "observer_additional_model_dit_flow_calls": 0,
    })
    created.extend([per_label_path, main_path, sensitivity_path, identity_path])
    artifact_manifest = write_sha256_manifest(export_root, created)
    return {
        "export_root": str(export_root),
        "cohort_main_path": str(main_path),
        "cohort_main_sha256": sha256_file(main_path),
        "sensitivity_path": str(sensitivity_path),
        "sensitivity_sha256": sha256_file(sensitivity_path),
        "per_vertebra_path": str(per_label_path),
        "per_vertebra_sha256": sha256_file(per_label_path),
        "identity_path": str(identity_path),
        "identity_sha256": sha256_file(identity_path),
        "artifact_manifest_path": str(artifact_manifest),
        "artifact_manifest_sha256": sha256_file(artifact_manifest),
        "main": cohort["main"],
    }
