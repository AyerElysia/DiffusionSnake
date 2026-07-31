#!/usr/bin/env python3
"""Extract frozen MoonViT features for sagittal slices listed in slice_manifest.csv.

Each manifest slice is processed independently: its single grayscale image is
replicated into RGB. Adjacent sagittal slices are never used as color channels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import f1_extract_locate_layers as f1  # noqa: E402


DEFAULT_MANIFEST = Path(
    "/home/medteam/Zhrch/detect_3D_lgz2/datasets/sagittal_2d_fixed/"
    "manifests/slice_manifest.csv"
)
DEFAULT_CHECKPOINT = f1.DEFAULT_CHECKPOINT
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "sagittal_moonvit_cache"
MANIFEST_SPLITS = ("training", "validation", "test")
CLI_TO_MANIFEST_SPLIT = {
    "train": "training",
    "validation": "validation",
    "test": "test",
}
REQUIRED_COLUMNS = {"split", "case_id", "slice_idx", "image_path"}
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)
NORMALIZATION_NAME = "moonvit_pretrained_rgb_mean_std"
ESTIMATED_NPZ_METADATA_BYTES = 4096


@dataclass(frozen=True)
class SliceRecord:
    manifest_split: str
    case_id: str
    slice_idx: int
    image_path: Path
    row_number: int
    declared_width: int | None
    declared_height: int | None

    @property
    def key(self) -> tuple[str, str, int]:
        return self.manifest_split, self.case_id, self.slice_idx

    def output_path(self, out_root: Path) -> Path:
        return out_root / self.manifest_split / self.case_id / f"x{self.slice_idx:04d}.npz"


@dataclass(frozen=True)
class ImageInfo:
    height: int
    width: int
    mode: str
    grid_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    padded_hw: tuple[int, int]
    input_hw: tuple[int, int]
    pad: tuple[int, int, int, int]
    scale: float


@dataclass(frozen=True)
class MoonViTMetadata:
    hidden_size: int
    num_hidden_layers: int
    patch_size: int
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    normalization_source: str
    checkpoint_shards: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to sagittal slice_manifest.csv",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "all"),
        default="all",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Maximum unique selected slices after filtering; 0 means all",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Restrict to one case ID; repeat or pass comma-separated IDs",
    )
    parser.add_argument("--input-size", type=int, choices=(448, 896), default=448)
    parser.add_argument("--layers", default="18", help="Comma-separated 1-based encoder layers")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def _parse_optional_positive_int(value: str, field: str, row_number: int) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {field}={value!r} at manifest row {row_number}"
        ) from exc
    if parsed <= 0:
        raise ValueError(
            f"{field} must be positive at manifest row {row_number}, got {parsed}"
        )
    return parsed


def _validate_case_id(case_id: str, row_number: int) -> str:
    case_id = str(case_id or "").strip()
    if not case_id:
        raise ValueError(f"Empty case_id at manifest row {row_number}")
    if (
        case_id in (".", "..")
        or "/" in case_id
        or "\\" in case_id
        or Path(case_id).name != case_id
    ):
        raise ValueError(
            f"Unsafe case_id={case_id!r} at manifest row {row_number}; "
            "case IDs must be one path component"
        )
    return case_id


def _resolve_image_path(raw_path: str, data_root: Path, row_number: int) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise ValueError(f"Empty image_path at manifest row {row_number}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = data_root / path
    return path.resolve()


def read_manifest(manifest: Path) -> tuple[list[SliceRecord], int]:
    if not manifest.exists():
        raise FileNotFoundError(f"Slice manifest not found: {manifest}")
    if not manifest.is_file():
        raise ValueError(f"Slice manifest is not a regular file: {manifest}")
    if manifest.name != "slice_manifest.csv":
        raise ValueError(
            f"Expected a file named slice_manifest.csv, got: {manifest.name}"
        )

    data_root = manifest.parent.parent
    records_by_key: dict[tuple[str, str, int], SliceRecord] = {}
    path_to_key: dict[Path, tuple[str, str, int]] = {}
    duplicate_count = 0

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing_columns = sorted(REQUIRED_COLUMNS.difference(fieldnames))
        if missing_columns:
            raise ValueError(
                f"Slice manifest is missing required columns {missing_columns}: {manifest}"
            )

        for row_number, row in enumerate(reader, start=2):
            manifest_split = str(row.get("split") or "").strip().lower()
            if manifest_split not in MANIFEST_SPLITS:
                raise ValueError(
                    f"Invalid split={manifest_split!r} at manifest row {row_number}; "
                    f"expected one of {MANIFEST_SPLITS}"
                )
            case_id = _validate_case_id(row.get("case_id", ""), row_number)
            try:
                slice_idx = int(str(row.get("slice_idx") or "").strip())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid slice_idx={row.get('slice_idx')!r} at manifest row {row_number}"
                ) from exc
            if slice_idx < 0:
                raise ValueError(
                    f"slice_idx must be non-negative at manifest row {row_number}, got {slice_idx}"
                )

            image_path = _resolve_image_path(row.get("image_path", ""), data_root, row_number)
            filename_match = re.search(r"_x(\d+)$", image_path.stem)
            if filename_match and int(filename_match.group(1)) != slice_idx:
                raise ValueError(
                    f"slice_idx/path mismatch at manifest row {row_number}: "
                    f"slice_idx={slice_idx}, image={image_path.name}"
                )

            record = SliceRecord(
                manifest_split=manifest_split,
                case_id=case_id,
                slice_idx=slice_idx,
                image_path=image_path,
                row_number=row_number,
                declared_width=_parse_optional_positive_int(
                    row.get("image_width", ""), "image_width", row_number
                ),
                declared_height=_parse_optional_positive_int(
                    row.get("image_height", ""), "image_height", row_number
                ),
            )
            previous = records_by_key.get(record.key)
            if previous is not None:
                if (
                    previous.image_path != record.image_path
                    or previous.declared_width != record.declared_width
                    or previous.declared_height != record.declared_height
                ):
                    raise ValueError(
                        f"Conflicting duplicate slice {record.key}: manifest rows "
                        f"{previous.row_number} and {row_number}"
                    )
                duplicate_count += 1
                continue

            previous_key = path_to_key.get(record.image_path)
            if previous_key is not None and previous_key != record.key:
                raise ValueError(
                    f"Image path is assigned to multiple slices: {record.image_path}; "
                    f"keys={previous_key} and {record.key}"
                )
            records_by_key[record.key] = record
            path_to_key[record.image_path] = record.key

    if not records_by_key:
        raise ValueError(f"Slice manifest has no data rows: {manifest}")

    split_order = {name: index for index, name in enumerate(MANIFEST_SPLITS)}
    records = sorted(
        records_by_key.values(),
        key=lambda rec: (
            split_order[rec.manifest_split],
            rec.case_id,
            rec.slice_idx,
            str(rec.image_path),
        ),
    )
    return records, duplicate_count


def parse_case_ids(values: Iterable[str]) -> set[str]:
    case_ids: set[str] = set()
    for raw in values:
        for value in str(raw).split(","):
            value = value.strip()
            if value:
                _validate_case_id(value, row_number=0)
                case_ids.add(value)
    return case_ids


def select_records(
    records: list[SliceRecord],
    split: str,
    case_ids: set[str],
    max_records: int,
) -> list[SliceRecord]:
    if max_records < 0:
        raise ValueError(f"--max-records must be non-negative, got {max_records}")

    selected_splits = (
        set(MANIFEST_SPLITS)
        if split == "all"
        else {CLI_TO_MANIFEST_SPLIT[split]}
    )
    split_records = [rec for rec in records if rec.manifest_split in selected_splits]
    if case_ids:
        available = {rec.case_id for rec in split_records}
        missing_case_ids = sorted(case_ids.difference(available))
        if missing_case_ids:
            raise ValueError(
                f"Requested case IDs are absent from split={split}: {missing_case_ids}"
            )
        split_records = [rec for rec in split_records if rec.case_id in case_ids]

    if max_records > 0:
        split_records = split_records[:max_records]
    if not split_records:
        raise ValueError(
            f"No unique slices selected for split={split}, case_ids={sorted(case_ids)}"
        )
    return split_records


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {description} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _normalization_triplet(
    value: Any,
    default: tuple[float, float, float],
    field: str,
) -> tuple[float, float, float]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"MoonViT {field} must have three values, got {value!r}")
    triplet = tuple(float(item) for item in value)
    if not all(np.isfinite(item) for item in triplet):
        raise ValueError(f"MoonViT {field} must be finite, got {triplet}")
    if field == "image_std" and any(item <= 0.0 for item in triplet):
        raise ValueError(f"MoonViT image_std must be positive, got {triplet}")
    return triplet


def load_checkpoint_metadata(checkpoint: Path, layers: list[int]) -> MoonViTMetadata:
    if not checkpoint.exists():
        raise FileNotFoundError(f"MoonViT checkpoint directory not found: {checkpoint}")
    if not checkpoint.is_dir():
        raise ValueError(f"MoonViT checkpoint must be a directory: {checkpoint}")

    config = _load_json_object(checkpoint / "config.json", "checkpoint config.json")
    vision_config = config.get("vision_config")
    if not isinstance(vision_config, dict):
        raise ValueError(f"checkpoint config lacks vision_config: {checkpoint / 'config.json'}")
    try:
        hidden_size = int(vision_config["hidden_size"])
        num_hidden_layers = int(vision_config["num_hidden_layers"])
        patch_size = int(vision_config["patch_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid MoonViT hidden/layer/patch metadata in {checkpoint / 'config.json'}"
        ) from exc
    if hidden_size <= 0 or num_hidden_layers <= 0 or patch_size <= 0:
        raise ValueError(
            f"MoonViT dimensions must be positive: hidden={hidden_size}, "
            f"layers={num_hidden_layers}, patch={patch_size}"
        )
    if max(layers) > num_hidden_layers:
        raise ValueError(
            f"Requested layer {max(layers)} exceeds MoonViT num_hidden_layers={num_hidden_layers}"
        )

    index = _load_json_object(
        checkpoint / "model.safetensors.index.json", "safetensors index"
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(
            f"safetensors index lacks a weight_map: {checkpoint / 'model.safetensors.index.json'}"
        )
    shards = tuple(
        sorted(
            {
                str(filename)
                for key, filename in weight_map.items()
                if str(key).startswith("vision_model.")
            }
        )
    )
    if not shards:
        raise ValueError(f"Checkpoint has no vision_model weights: {checkpoint}")
    missing_shards = [name for name in shards if not (checkpoint / name).is_file()]
    if missing_shards:
        raise FileNotFoundError(
            f"Checkpoint is missing MoonViT shard(s): {missing_shards}"
        )

    processor_path = checkpoint / "preprocessor_config.json"
    if processor_path.exists():
        processor = _load_json_object(processor_path, "preprocessor config")
        mean = _normalization_triplet(processor.get("image_mean"), DEFAULT_MEAN, "image_mean")
        std = _normalization_triplet(processor.get("image_std"), DEFAULT_STD, "image_std")
        processor_patch = int(processor.get("patch_size", patch_size))
        if processor_patch != patch_size:
            raise ValueError(
                f"Patch-size mismatch: vision_config={patch_size}, "
                f"preprocessor_config={processor_patch}"
            )
        normalization_source = str(processor_path.resolve())
    else:
        mean = DEFAULT_MEAN
        std = DEFAULT_STD
        normalization_source = "f1_extract_locate_layers defaults"

    return MoonViTMetadata(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        patch_size=patch_size,
        normalization_mean=mean,
        normalization_std=std,
        normalization_source=normalization_source,
        checkpoint_shards=shards,
    )


def read_grayscale_slice(path: Path) -> tuple[np.ndarray, str]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest image is missing: {path}")
    if not path.is_file():
        raise ValueError(f"Manifest image is not a regular file: {path}")
    try:
        with Image.open(path) as source:
            source.load()
            mode = str(source.mode)
            array = np.asarray(source)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot decode manifest image {path}: {exc}") from exc

    if array.dtype != np.uint8:
        raise ValueError(f"Expected uint8 sagittal slice at {path}, got {array.dtype}")
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] in (3, 4):
        rgb = array[:, :, :3]
        if not (
            np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
            and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
        ):
            raise ValueError(
                f"Expected a grayscale sagittal slice at {path}, got non-identical RGB channels"
            )
        gray = rgb[:, :, 0]
    else:
        raise ValueError(
            f"Expected a 2D grayscale slice at {path}, got shape={array.shape}, mode={mode}"
        )
    if gray.size == 0 or gray.shape[0] <= 0 or gray.shape[1] <= 0:
        raise ValueError(f"Empty sagittal slice: {path}")
    return np.ascontiguousarray(gray), mode


def inspect_record(
    record: SliceRecord,
    input_size: int,
    patch_size: int,
) -> ImageInfo:
    gray, mode = read_grayscale_slice(record.image_path)
    height, width = int(gray.shape[0]), int(gray.shape[1])
    if record.declared_width is not None and record.declared_width != width:
        raise ValueError(
            f"Manifest/image width mismatch at row {record.row_number}: "
            f"declared={record.declared_width}, actual={width}, path={record.image_path}"
        )
    if record.declared_height is not None and record.declared_height != height:
        raise ValueError(
            f"Manifest/image height mismatch at row {record.row_number}: "
            f"declared={record.declared_height}, actual={height}, path={record.image_path}"
        )

    geometry = f1.resized_padded_hw(width, height, input_size, patch_size)
    return ImageInfo(
        height=height,
        width=width,
        mode=mode,
        grid_hw=tuple(int(v) for v in geometry["grid_hw"]),
        resized_hw=tuple(int(v) for v in geometry["resized_hw"]),
        padded_hw=tuple(int(v) for v in geometry["padded_hw"]),
        input_hw=tuple(int(v) for v in geometry["padded_hw"]),
        pad=tuple(int(v) for v in geometry["pad"]),
        scale=float(geometry["scale"]),
    )


def validate_records(
    records: list[SliceRecord],
    input_size: int,
    patch_size: int,
) -> dict[tuple[str, str, int], ImageInfo]:
    infos: dict[tuple[str, str, int], ImageInfo] = {}
    case_shapes: dict[tuple[str, str], tuple[int, int]] = {}
    for record in records:
        info = inspect_record(record, input_size, patch_size)
        case_key = (record.manifest_split, record.case_id)
        shape = (info.height, info.width)
        previous_shape = case_shapes.get(case_key)
        if previous_shape is not None and previous_shape != shape:
            raise ValueError(
                f"Inconsistent image size within case {case_key}: "
                f"expected={previous_shape}, got={shape} at {record.image_path}"
            )
        case_shapes[case_key] = shape
        infos[record.key] = info
    return infos


def preprocess_record(
    record: SliceRecord,
    expected: ImageInfo,
    input_size: int,
    metadata: MoonViTMetadata,
) -> tuple[torch.Tensor, dict[str, Any]]:
    gray, source_mode = read_grayscale_slice(record.image_path)
    height, width = int(gray.shape[0]), int(gray.shape[1])
    if (height, width) != (expected.height, expected.width):
        raise RuntimeError(
            f"Image changed after preflight validation: {record.image_path}; "
            f"expected={(expected.height, expected.width)}, got={(height, width)}"
        )

    # One center slice only: repeat its grayscale pixels into R/G/B.
    gray_image = Image.fromarray(gray, mode="L")
    rgb_image = Image.merge("RGB", (gray_image, gray_image, gray_image))
    resized_h, resized_w = expected.resized_hw
    padded_h, padded_w = expected.padded_hw
    resized = rgb_image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
    resized_array = np.asarray(resized, dtype=np.float32) / 255.0

    mean = np.asarray(metadata.normalization_mean, dtype=np.float32)
    std = np.asarray(metadata.normalization_std, dtype=np.float32)
    padded = np.empty((padded_h, padded_w, 3), dtype=np.float32)
    padded[...] = mean
    padded[:resized_h, :resized_w, :] = resized_array
    normalized = (padded - mean) / std
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).contiguous()

    patch_size = metadata.patch_size
    patches = tensor.reshape(
        3,
        padded_h // patch_size,
        patch_size,
        padded_w // patch_size,
        patch_size,
    )
    patches = (
        patches.permute(1, 3, 0, 2, 4)
        .contiguous()
        .view(-1, 3, patch_size, patch_size)
    )
    meta: dict[str, Any] = {
        "grid_hw": expected.grid_hw,
        "orig_hw": (height, width),
        "resized_hw": expected.resized_hw,
        "padded_hw": expected.padded_hw,
        "input_hw": expected.input_hw,
        "pad": expected.pad,
        "scale": expected.scale,
        "image_path": str(record.image_path),
        "case_id": record.case_id,
        "slice_idx": record.slice_idx,
        "manifest_split": record.manifest_split,
        "source_mode": source_mode,
        "input_size": input_size,
    }
    return patches, meta


def _scalar(npz: Any, key: str) -> Any:
    value = np.asarray(npz[key])
    if value.size != 1:
        raise ValueError(f"metadata {key} must contain one value, got shape={value.shape}")
    return value.reshape(-1)[0].item()


def validate_existing_cache(
    out_path: Path,
    record: SliceRecord,
    info: ImageInfo,
    layers: list[int],
    input_size: int,
    checkpoint: Path,
    metadata: MoonViTMetadata,
) -> None:
    try:
        with np.load(out_path, allow_pickle=False) as npz:
            required = {
                *(f"layer_{layer}" for layer in layers),
                "grid_hw",
                "orig_hw",
                "resized_hw",
                "padded_hw",
                "input_hw",
                "pad",
                "scale",
                "patch_size",
                "input_size",
                "image_path",
                "case_id",
                "slice_idx",
                "manifest_split",
                "normalization",
                "normalization_mean",
                "normalization_std",
                "checkpoint",
            }
            missing = sorted(required.difference(npz.files))
            if missing:
                raise ValueError(f"missing keys={missing}")

            expected_shape = (
                metadata.hidden_size,
                int(info.grid_hw[0]),
                int(info.grid_hw[1]),
            )
            for layer in layers:
                actual = np.asarray(npz[f"layer_{layer}"])
                actual_shape = tuple(actual.shape)
                if actual_shape != expected_shape:
                    raise ValueError(
                        f"layer_{layer} shape={actual_shape}, expected={expected_shape}"
                    )
                if actual.dtype != np.float16:
                    raise ValueError(
                        f"layer_{layer} dtype={actual.dtype}, expected float16"
                    )

            exact_arrays = {
                "grid_hw": np.asarray(info.grid_hw, dtype=np.int32),
                "orig_hw": np.asarray([info.height, info.width], dtype=np.int32),
                "resized_hw": np.asarray(info.resized_hw, dtype=np.int32),
                "padded_hw": np.asarray(info.padded_hw, dtype=np.int32),
                "input_hw": np.asarray(info.input_hw, dtype=np.int32),
                "pad": np.asarray(info.pad, dtype=np.int32),
            }
            for key, expected_array in exact_arrays.items():
                if not np.array_equal(np.asarray(npz[key]), expected_array):
                    raise ValueError(
                        f"{key}={np.asarray(npz[key]).tolist()}, "
                        f"expected={expected_array.tolist()}"
                    )
            if not np.allclose(
                np.asarray(npz["scale"], dtype=np.float32),
                np.asarray([info.scale], dtype=np.float32),
                rtol=1e-6,
                atol=1e-7,
            ):
                raise ValueError(f"scale mismatch in {out_path}")
            if not np.allclose(
                np.asarray(npz["normalization_mean"], dtype=np.float32),
                np.asarray(metadata.normalization_mean, dtype=np.float32),
            ):
                raise ValueError(f"normalization_mean mismatch in {out_path}")
            if not np.allclose(
                np.asarray(npz["normalization_std"], dtype=np.float32),
                np.asarray(metadata.normalization_std, dtype=np.float32),
            ):
                raise ValueError(f"normalization_std mismatch in {out_path}")

            exact_scalars = {
                "patch_size": metadata.patch_size,
                "input_size": input_size,
                "image_path": str(record.image_path),
                "case_id": record.case_id,
                "slice_idx": record.slice_idx,
                "manifest_split": record.manifest_split,
                "normalization": NORMALIZATION_NAME,
                "checkpoint": str(checkpoint),
            }
            for key, expected_value in exact_scalars.items():
                if _scalar(npz, key) != expected_value:
                    raise ValueError(
                        f"{key}={_scalar(npz, key)!r}, expected={expected_value!r}"
                    )
            cached_layers = np.asarray(
                npz["layers"] if "layers" in npz.files else layers,
                dtype=np.int32,
            ).tolist()
            if cached_layers != layers:
                raise ValueError(f"layers={cached_layers}, expected={layers}")
    except Exception as exc:
        raise RuntimeError(
            f"Existing cache is inconsistent: {out_path}: {exc}. "
            "Pass --overwrite to replace it."
        ) from exc


def estimate_cache(
    records: list[SliceRecord],
    infos: dict[tuple[str, str, int], ImageInfo],
    out_root: Path,
    layers: list[int],
    metadata: MoonViTMetadata,
    overwrite: bool,
    input_size: int,
    checkpoint: Path,
) -> dict[str, Any]:
    total_bytes = 0
    pending_bytes = 0
    existing = 0
    pending = 0
    grid_counts: dict[tuple[int, int], int] = {}
    split_counts: dict[str, int] = {}

    for record in records:
        info = infos[record.key]
        gh, gw = info.grid_hw
        record_bytes = int(
            gh * gw * metadata.hidden_size * len(layers) * np.dtype(np.float16).itemsize
            + ESTIMATED_NPZ_METADATA_BYTES
        )
        total_bytes += record_bytes
        grid_counts[(gh, gw)] = grid_counts.get((gh, gw), 0) + 1
        split_counts[record.manifest_split] = split_counts.get(record.manifest_split, 0) + 1

        out_path = record.output_path(out_root)
        if out_path.exists() and not overwrite:
            if not out_path.is_file():
                raise ValueError(f"Cache output path is not a regular file: {out_path}")
            validate_existing_cache(
                out_path,
                record,
                info,
                layers,
                input_size,
                checkpoint,
                metadata,
            )
            existing += 1
        else:
            pending += 1
            pending_bytes += record_bytes

    return {
        "records": len(records),
        "existing": existing,
        "pending": pending,
        "bytes": total_bytes,
        "pending_bytes": pending_bytes,
        "gb": total_bytes / float(1000**3),
        "gib": total_bytes / float(1024**3),
        "pending_gib": pending_bytes / float(1024**3),
        "grid_counts": {
            f"{height}x{width}": count
            for (height, width), count in sorted(grid_counts.items())
        },
        "split_counts": {key: split_counts[key] for key in MANIFEST_SPLITS if key in split_counts},
    }


def save_cache(
    out_path: Path,
    features: dict[str, np.ndarray],
    meta: dict[str, Any],
    layers: list[int],
    checkpoint: Path,
    metadata: MoonViTMetadata,
) -> None:
    gh, gw = (int(v) for v in meta["grid_hw"])
    oh, ow = (int(v) for v in meta["orig_hw"])
    rh, rw = (int(v) for v in meta["resized_hw"])
    ph, pw = (int(v) for v in meta["padded_hw"])
    ih, iw = (int(v) for v in meta["input_hw"])
    pad_left, pad_top, pad_right, pad_bottom = (int(v) for v in meta["pad"])
    expected_shape = (metadata.hidden_size, gh, gw)
    for layer in layers:
        key = f"layer_{layer}"
        if key not in features:
            raise RuntimeError(f"MoonViT extraction did not return {key}")
        if tuple(features[key].shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected {key} shape={features[key].shape}, expected={expected_shape}"
            )
        if features[key].dtype != np.float16:
            raise RuntimeError(f"Expected float16 {key}, got {features[key].dtype}")

    payload: dict[str, Any] = {
        **features,
        # Standard keys consumed by the Locate feature loader.
        "grid_hw": np.asarray([gh, gw], dtype=np.int32),
        "orig_hw": np.asarray([oh, ow], dtype=np.int32),
        "resized_hw": np.asarray([rh, rw], dtype=np.int32),
        "padded_hw": np.asarray([ph, pw], dtype=np.int32),
        "input_hw": np.asarray([ih, iw], dtype=np.int32),
        "pad": np.asarray(
            [pad_left, pad_top, pad_right, pad_bottom], dtype=np.int32
        ),
        "scale": np.asarray([meta["scale"]], dtype=np.float32),
        "patch_size": np.asarray([metadata.patch_size], dtype=np.int32),
        "input_size": np.asarray([meta["input_size"]], dtype=np.int32),
        "long_side": np.asarray([meta["input_size"]], dtype=np.int32),
        "layers": np.asarray(layers, dtype=np.int32),
        # Explicit geometry aliases make the cache self-describing.
        "grid": np.asarray([gh, gw], dtype=np.int32),
        "orig": np.asarray([oh, ow], dtype=np.int32),
        "resized": np.asarray([rh, rw], dtype=np.int32),
        "padded": np.asarray([ph, pw], dtype=np.int32),
        "input": np.asarray([ih, iw], dtype=np.int32),
        "patch": np.asarray(
            [metadata.patch_size, metadata.patch_size], dtype=np.int32
        ),
        # Slice identity and preprocessing provenance.
        "image_path": np.asarray(meta["image_path"]),
        "case_id": np.asarray(meta["case_id"]),
        "slice_idx": np.asarray([meta["slice_idx"]], dtype=np.int32),
        "manifest_split": np.asarray(meta["manifest_split"]),
        "source_mode": np.asarray(meta["source_mode"]),
        "rgb_source": np.asarray("single_grayscale_slice_repeated_3x"),
        "normalization": np.asarray(NORMALIZATION_NAME),
        "normalization_mean": np.asarray(
            metadata.normalization_mean, dtype=np.float32
        ),
        "normalization_std": np.asarray(
            metadata.normalization_std, dtype=np.float32
        ),
        "normalization_source": np.asarray(metadata.normalization_source),
        "padding_value_rgb": np.asarray(
            metadata.normalization_mean, dtype=np.float32
        ),
        "checkpoint": np.asarray(str(checkpoint)),
        "checkpoint_shards": np.asarray(metadata.checkpoint_shards),
        "vision_hidden_size": np.asarray([metadata.hidden_size], dtype=np.int32),
        "vision_num_hidden_layers": np.asarray(
            [metadata.num_hidden_layers], dtype=np.int32
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("wb") as handle:
            np.savez(handle, **payload)
        os.replace(temp_path, out_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(f"No existing parent for output path: {path}")
        candidate = candidate.parent
    if not candidate.is_dir():
        candidate = candidate.parent
    return candidate


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    layers = f1.parse_layers(args.layers)
    case_ids = parse_case_ids(args.case_id)

    records_all, duplicate_count = read_manifest(manifest)
    records = select_records(
        records_all,
        split=args.split,
        case_ids=case_ids,
        max_records=int(args.max_records),
    )
    moonvit_meta = load_checkpoint_metadata(checkpoint, layers)
    if args.input_size % moonvit_meta.patch_size != 0:
        raise ValueError(
            f"--input-size {args.input_size} must be divisible by MoonViT "
            f"patch_size={moonvit_meta.patch_size}"
        )

    infos = validate_records(records, args.input_size, moonvit_meta.patch_size)
    estimate = estimate_cache(
        records,
        infos,
        out_root,
        layers,
        moonvit_meta,
        bool(args.overwrite),
        args.input_size,
        checkpoint,
    )
    print(
        f"[*] Manifest={manifest} unique_total={len(records_all)} "
        f"deduplicated={duplicate_count} selected={estimate['records']} "
        f"split_counts={estimate['split_counts']}",
        flush=True,
    )
    print(
        f"[*] Cache estimate: input={args.input_size} layers={layers} "
        f"hidden={moonvit_meta.hidden_size} grids={estimate['grid_counts']} "
        f"total={estimate['bytes']} bytes ({estimate['gb']:.2f} GB / "
        f"{estimate['gib']:.2f} GiB) existing={estimate['existing']} "
        f"pending={estimate['pending']} ({estimate['pending_gib']:.2f} GiB)",
        flush=True,
    )
    print(
        f"[*] Normalization={NORMALIZATION_NAME} "
        f"mean={moonvit_meta.normalization_mean} std={moonvit_meta.normalization_std} "
        f"source={moonvit_meta.normalization_source}",
        flush=True,
    )
    if args.estimate_only:
        return
    if estimate["pending"] == 0:
        print(f"[*] Nothing to extract; all selected caches are valid under {out_root}")
        return

    output_parent = nearest_existing_parent(out_root)
    free_bytes = shutil.disk_usage(output_parent).free
    if estimate["pending_bytes"] > free_bytes:
        raise OSError(
            f"Insufficient free space for estimated pending cache: "
            f"need={estimate['pending_bytes']} bytes, free={free_bytes} bytes under "
            f"{output_parent}"
        )

    device = f1.resolve_device(args.device)
    dtype = f1.resolve_dtype(args.dtype, device)
    model, model_cfg = f1.load_vision_model(checkpoint, device, dtype)
    if int(model_cfg.hidden_size) != moonvit_meta.hidden_size:
        raise RuntimeError(
            f"Loaded MoonViT hidden_size={model_cfg.hidden_size}, "
            f"checkpoint metadata={moonvit_meta.hidden_size}"
        )
    if int(model_cfg.patch_size) != moonvit_meta.patch_size:
        raise RuntimeError(
            f"Loaded MoonViT patch_size={model_cfg.patch_size}, "
            f"checkpoint metadata={moonvit_meta.patch_size}"
        )

    written = 0
    skipped = 0
    for index, record in enumerate(records, start=1):
        out_path = record.output_path(out_root)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        patches, meta = preprocess_record(
            record,
            infos[record.key],
            args.input_size,
            moonvit_meta,
        )
        features = f1.extract_layers(
            model,
            patches,
            meta["grid_hw"],
            layers,
            device,
            dtype,
        )
        save_cache(out_path, features, meta, layers, checkpoint, moonvit_meta)
        written += 1

        if index <= 2 or args.progress_every > 0 and index % args.progress_every == 0:
            shapes = {key: tuple(value.shape) for key, value in features.items()}
            print(
                f"[{index}/{len(records)}] split={record.manifest_split} "
                f"case={record.case_id} slice={record.slice_idx} shapes={shapes} "
                f"grid={meta['grid_hw']} file={out_path.stat().st_size / (1024**2):.2f} MiB "
                f"-> {out_path}",
                flush=True,
            )

    print(
        f"[*] Finished. written={written} skipped_valid={skipped} root={out_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
