#!/usr/bin/env python3
"""Add canonical sagittal millimetre positions to the slice manifest."""

import argparse
import csv
import gzip
import math
import pathlib
import struct


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-manifest", required=True)
    parser.add_argument("--case-metadata", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _read_header(path):
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rb") as handle:
        header = handle.read(348)
    if len(header) != 348:
        raise ValueError("incomplete NIfTI header: {}".format(path))
    if struct.unpack("<i", header[:4])[0] == 348:
        endian = "<"
    elif struct.unpack(">i", header[:4])[0] == 348:
        endian = ">"
    else:
        raise ValueError("invalid NIfTI header: {}".format(path))
    dim = struct.unpack(endian + "8h", header[40:56])
    pixdim = struct.unpack(endian + "8f", header[76:108])
    qform_code, sform_code = struct.unpack(endian + "2h", header[252:256])
    return header, endian, dim, pixdim, qform_code, sform_code


def _affine_columns(header, endian, pixdim, qform_code, sform_code):
    if sform_code > 0:
        rows = [
            struct.unpack(endian + "4f", header[offset:offset + 16])
            for offset in (280, 296, 312)
        ]
        return [[rows[row][column] for row in range(3)] for column in range(3)]
    if qform_code > 0:
        b, c, d = struct.unpack(endian + "3f", header[256:268])
        a = math.sqrt(max(1.0 - (b * b + c * c + d * d), 0.0))
        rotation = (
            (a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)),
            (2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)),
            (2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b),
        )
        scales = [abs(float(pixdim[index])) for index in (1, 2, 3)]
        if float(pixdim[0]) < 0:
            scales[2] *= -1.0
        return [
            [rotation[row][column] * scales[column] for row in range(3)]
            for column in range(3)
        ]
    return [
        [abs(float(pixdim[column + 1])) if row == column else 0.0 for row in range(3)]
        for column in range(3)
    ]


def canonical_sagittal_geometry(path):
    header, endian, dim, pixdim, qform_code, sform_code = _read_header(path)
    columns = _affine_columns(header, endian, pixdim, qform_code, sform_code)
    source_axis = max(range(3), key=lambda column: abs(columns[column][0]))
    spacing = math.sqrt(sum(value * value for value in columns[source_axis]))
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("invalid sagittal spacing in {}".format(path))
    return int(dim[source_axis + 1]), float(spacing), int(source_axis)


def main():
    args = parse_args()
    with open(args.case_metadata, newline="", encoding="utf-8-sig") as handle:
        case_rows = list(csv.DictReader(handle))
    case_map = {row["case_id"]: row for row in case_rows}
    geometry = {}
    for case_id, row in case_map.items():
        geometry[case_id] = canonical_sagittal_geometry(row["image_nii_path"])

    with open(args.slice_manifest, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for field in ("slice_spacing_mm", "slice_position_mm", "source_sagittal_axis"):
        if field not in fieldnames:
            fieldnames.append(field)

    counts = {}
    for row in rows:
        case_id = row["case_id"]
        if case_id not in geometry:
            raise ValueError("missing case metadata for {}".format(case_id))
        expected_slices, spacing, source_axis = geometry[case_id]
        slice_index = int(row["slice_idx"])
        if slice_index < 0 or slice_index >= expected_slices:
            raise ValueError("slice index outside NIfTI geometry for {}".format(case_id))
        if int(row["canonical_shape_x"]) != expected_slices:
            raise ValueError("canonical_shape_x mismatch for {}".format(case_id))
        row["slice_spacing_mm"] = "{:.9g}".format(spacing)
        row["slice_position_mm"] = "{:.9g}".format(slice_index * spacing)
        row["source_sagittal_axis"] = str(source_axis)
        counts[case_id] = counts.get(case_id, 0) + 1
    for case_id, count in counts.items():
        if count != geometry[case_id][0]:
            raise ValueError("manifest slice count mismatch for {}".format(case_id))

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {} rows across {} volumes to {}".format(len(rows), len(counts), output))


if __name__ == "__main__":
    main()
