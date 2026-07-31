import json
import math
import numbers
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


_REF_BOX_RE = re.compile(
    r"<ref>(?P<label>[^<]+)</ref>\s*<box>\s*"
    r"<\s*(?P<x1>\d+(?:\.\d+)?)\s*>\s*"
    r"<\s*(?P<y1>\d+(?:\.\d+)?)\s*>\s*"
    r"<\s*(?P<x2>\d+(?:\.\d+)?)\s*>\s*"
    r"<\s*(?P<y2>\d+(?:\.\d+)?)\s*>\s*</box>",
    re.IGNORECASE,
)
_PLAIN_BOX_RE = re.compile(
    r"<box>\s*<?\s*(?P<x1>\d+(?:\.\d+)?)\s*>?\s*[, ]+"
    r"<?\s*(?P<y1>\d+(?:\.\d+)?)\s*>?\s*[, ]+"
    r"<?\s*(?P<x2>\d+(?:\.\d+)?)\s*>?\s*[, ]+"
    r"<?\s*(?P<y2>\d+(?:\.\d+)?)\s*>?\s*</box>",
    re.IGNORECASE,
)
_LEGACY_CLASS_LABEL_RE = re.compile(r"^\s*class[_\s-]*(\d+)\s*$", re.IGNORECASE)
_VERTEBRA_LABEL_RE = re.compile(r"^\s*([CTL])(\d+)\s+vertebra\s*$", re.IGNORECASE)
_VERTEBRA_OFFSETS = {"C": 0, "T": 7, "L": 19}
_VERTEBRA_LIMITS = {"C": 7, "T": 12, "L": 6}


@dataclass(frozen=True)
class DetectionPolicy:
    """Quality gates for cached LocateAnything detections."""

    min_score: float = 1e-4
    min_box_side: float = 1.0
    min_box_area: float = 4.0
    nms_iou: float = 0.5
    max_detections: int = 32
    class_aware_nms: bool = True
    missing: str = "error"

    def validate(self) -> None:
        float_values = {
            "min_score": self.min_score,
            "min_box_side": self.min_box_side,
            "min_box_area": self.min_box_area,
            "nms_iou": self.nms_iou,
        }
        for name, value in float_values.items():
            try:
                finite = math.isfinite(value)
            except TypeError as exc:
                raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
            if not finite:
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.min_score < 0.0:
            raise ValueError("min_score must be non-negative")
        if self.min_box_side < 0.0 or self.min_box_area < 0.0:
            raise ValueError("box size gates must be non-negative")
        if not 0.0 <= self.nms_iou <= 1.0:
            raise ValueError("nms_iou must be in [0, 1]")
        if isinstance(self.max_detections, bool) or not isinstance(
                self.max_detections, numbers.Integral):
            raise ValueError(
                f"max_detections must be an integer, got {self.max_detections!r}"
            )
        if self.max_detections < 0:
            raise ValueError("max_detections must be non-negative")
        if self.missing not in {"error", "empty"}:
            raise ValueError("missing must be 'error' or 'empty'")


@dataclass(frozen=True)
class CachedDetection:
    bbox: Tuple[float, float, float, float]
    score: float
    class_id: int
    label: str = ""


def _strict_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    raise ValueError(f"{name} must be an integer, got {value!r}")


def _canonical_class_id(value: object) -> int:
    class_id = _strict_integer(value, "canonical class_id")
    if not 0 <= class_id <= 24:
        raise ValueError(f"canonical class_id must be in [0, 24], got {class_id}")
    return class_id


def _legacy_label_id(value: object) -> int:
    label_id = _strict_integer(value, "legacy VerSe label id")
    if not 1 <= label_id <= 25:
        raise ValueError(f"legacy VerSe label id must be in [1, 25], got {label_id}")
    return label_id - 1


def _class_from_text(label: object, default: int = 0) -> int:
    if label is None or not str(label).strip():
        return _canonical_class_id(default)
    text = str(label)
    vertebra = _VERTEBRA_LABEL_RE.match(text)
    if vertebra:
        region = vertebra.group(1).upper()
        level = int(vertebra.group(2))
        if not 1 <= level <= _VERTEBRA_LIMITS[region]:
            raise ValueError(f"invalid vertebra label: {text}")
        return _VERTEBRA_OFFSETS[region] + level - 1
    legacy = _LEGACY_CLASS_LABEL_RE.match(text)
    if legacy:
        return _legacy_label_id(legacy.group(1))
    raise ValueError(f"unsupported detection label: {text}")


def _clip_xyxy(
    bbox: Sequence[float],
    width: float,
    height: float,
) -> Optional[Tuple[float, float, float, float]]:
    if len(bbox) < 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError(f"bbox coordinates must be finite, got {bbox[:4]}")
    x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
    y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def parse_raw_response(
    text: str,
    image_width: int,
    image_height: int,
    default_class_id: int = 0,
) -> List[CachedDetection]:
    """Parse only class-aware LocateAnything groups on the normalized 0..1000 grid.

    ``default_class_id`` is retained for API compatibility but is deliberately
    unused. A bare box or a generic ``vertebra`` label has no defensible
    absolute class and therefore cannot enter the Snake contract.
    """
    del default_class_id
    detections: List[CachedDetection] = []
    for match in _REF_BOX_RE.finditer(text or ""):
        label = match.group("label").strip()
        class_id = _class_from_text(label)
        values = [float(match.group(name)) for name in ("x1", "y1", "x2", "y2")]
        scaled = [
            values[0] / 1000.0 * image_width,
            values[1] / 1000.0 * image_height,
            values[2] / 1000.0 * image_width,
            values[3] / 1000.0 * image_height,
        ]
        bbox = _clip_xyxy(scaled, float(image_width), float(image_height))
        if bbox is not None:
            detections.append(CachedDetection(bbox, 1.0, class_id, label))
    return detections


def _parse_instances(
    instances: Iterable[Mapping[str, object]],
    width: int,
    height: int,
) -> List[CachedDetection]:
    detections: List[CachedDetection] = []
    for instance in instances:
        bbox_obj = instance.get("bbox", instance.get("box"))
        if not isinstance(bbox_obj, (list, tuple)):
            continue
        label = instance.get("label", "")
        if "class_id" in instance:
            class_id = _canonical_class_id(instance["class_id"])
        elif label:
            class_id = _class_from_text(label)
        else:
            legacy_value = instance.get(
                "label_id",
                instance.get("cls_id", instance.get("category_id", 1)),
            )
            class_id = _legacy_label_id(legacy_value)
        score = float(instance.get("score", instance.get("confidence", 1.0)))
        if not math.isfinite(score):
            raise ValueError(f"detection score must be finite, got {score}")
        bbox = _clip_xyxy(bbox_obj, float(width), float(height))
        if bbox is None:
            continue
        detections.append(CachedDetection(bbox, score, class_id, str(label)))
    return detections


def _path_keys(path: object, include_absolute_aliases: bool = True) -> Tuple[str, ...]:
    value = str(path)
    p = Path(value)
    keys = [value, str(p)]
    if p.is_absolute():
        keys.append(str(p.resolve()))
        if include_absolute_aliases:
            keys.extend((p.name, p.stem))
    else:
        keys.extend((p.name, p.stem))
    return tuple(dict.fromkeys(keys))


class LocateAnythingCache:
    """Read-only image-keyed cache independent from the 3B inference project."""

    def __init__(self, records: Sequence[Mapping[str, object]]) -> None:
        self._records: Dict[str, Tuple[CachedDetection, ...]] = {}
        for record in records:
            width = int(record.get("width", record.get("image_width", 0)) or 0)
            height = int(record.get("height", record.get("image_height", 0)) or 0)
            if width <= 0 or height <= 0:
                raise ValueError("each cache record must declare positive width and height")
            structured_key = next(
                (name for name in ("instances", "detections", "boxes") if name in record),
                None,
            )
            if structured_key is not None:
                instances = record[structured_key]
                if not isinstance(instances, list):
                    raise TypeError(f"{structured_key} must be a list")
                parsed = _parse_instances(instances, width, height)
            else:
                parsed = parse_raw_response(str(record.get("raw_response", "")), width, height)
            keys = set()
            for name in ("img_path", "image_path", "path", "file_name", "image", "image_rel", "id"):
                value = record.get(name)
                if value:
                    keys.update(_path_keys(value))
            if not keys:
                raise ValueError("cache record has no image identity key")
            frozen = tuple(parsed)
            for key in keys:
                if key in self._records:
                    raise ValueError(f"ambiguous cache key appears in multiple records: {key}")
                self._records[key] = frozen

    @classmethod
    def from_path(cls, path: str) -> "LocateAnythingCache":
        cache_path = Path(path)
        if cache_path.suffix.lower() == ".jsonl":
            with cache_path.open("r", encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
        else:
            with cache_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            records = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError("LocateAnything cache must be JSONL records or a JSON samples list")
        return cls(records)

    def lookup(self, image_path: object, missing: str = "error") -> Tuple[CachedDetection, ...]:
        for key in _path_keys(image_path, include_absolute_aliases=False):
            if key in self._records:
                return self._records[key]
        if missing == "empty":
            return ()
        raise KeyError(f"no LocateAnything cache entry for image: {image_path}")


def _box_iou(one: CachedDetection, two: CachedDetection) -> float:
    x1 = max(one.bbox[0], two.bbox[0])
    y1 = max(one.bbox[1], two.bbox[1])
    x2 = min(one.bbox[2], two.bbox[2])
    y2 = min(one.bbox[3], two.bbox[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_one = (one.bbox[2] - one.bbox[0]) * (one.bbox[3] - one.bbox[1])
    area_two = (two.bbox[2] - two.bbox[0]) * (two.bbox[3] - two.bbox[1])
    union = area_one + area_two - intersection
    return intersection / union if union > 0.0 else 0.0


def filter_detections(
    detections: Sequence[CachedDetection],
    policy: DetectionPolicy,
) -> List[CachedDetection]:
    policy.validate()
    candidates = []
    for detection in detections:
        width = detection.bbox[2] - detection.bbox[0]
        height = detection.bbox[3] - detection.bbox[1]
        if detection.score <= 1e-4:
            continue
        if detection.score < policy.min_score:
            continue
        if min(width, height) < policy.min_box_side or width * height < policy.min_box_area:
            continue
        candidates.append(detection)
    candidates.sort(key=lambda item: item.score, reverse=True)
    kept: List[CachedDetection] = []
    for candidate in candidates:
        suppressed = False
        for existing in kept:
            same_group = (candidate.class_id == existing.class_id) or not policy.class_aware_nms
            if same_group and _box_iou(candidate, existing) > policy.nms_iou:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)
            if policy.max_detections and len(kept) >= policy.max_detections:
                break
    return kept


def flip_detection(
    detection: CachedDetection,
    original_width: int,
) -> CachedDetection:
    """Mirror an original-image box exactly as the slice dataset mirrors pixels."""
    if original_width <= 0:
        raise ValueError("original_width must be positive for flipped detections")
    x1, y1, x2, y2 = detection.bbox
    mirrored_x1 = float(original_width) - x2
    mirrored_x2 = float(original_width) - x1
    return CachedDetection(
        (mirrored_x1, y1, mirrored_x2, y2),
        detection.score,
        detection.class_id,
        detection.label,
    )


def transform_detection(
    detection: CachedDetection,
    transform: np.ndarray,
    input_width: int,
    input_height: int,
) -> Optional[CachedDetection]:
    matrix = np.asarray(transform, dtype=np.float32)
    if matrix.shape != (2, 3):
        raise ValueError(f"trans_input must have shape (2, 3), got {matrix.shape}")
    x1, y1, x2, y2 = detection.bbox
    corners = np.asarray(
        [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
        dtype=np.float32,
    )
    mapped = corners @ matrix.T
    bbox = _clip_xyxy(
        [mapped[:, 0].min(), mapped[:, 1].min(), mapped[:, 0].max(), mapped[:, 1].max()],
        float(input_width),
        float(input_height),
    )
    if bbox is None:
        return None
    return CachedDetection(bbox, detection.score, detection.class_id, detection.label)


def _batch_values(value: object, batch_size: int, name: str) -> List[object]:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return [value.item()] * batch_size
        if int(value.shape[0]) != batch_size:
            raise ValueError(f"{name} batch dimension does not match inp")
        return [value[index].detach().cpu().numpy() for index in range(batch_size)]
    if isinstance(value, np.ndarray):
        if value.ndim == 2 and value.shape == (2, 3) and batch_size == 1:
            return [value]
        if int(value.shape[0]) != batch_size:
            raise ValueError(f"{name} batch dimension does not match inp")
        return [value[index] for index in range(batch_size)]
    if isinstance(value, (list, tuple)):
        if len(value) != batch_size:
            raise ValueError(f"{name} length does not match inp")
        return list(value)
    if batch_size == 1:
        return [value]
    raise TypeError(f"{name} must provide one value per batch item")


def build_detection_tensor(
    cache: LocateAnythingCache,
    batch: Mapping[str, object],
    policy: DetectionPolicy,
) -> torch.Tensor:
    """Build padded [B, N, 6] detections in current network-input pixels."""
    inp = batch.get("inp")
    if not torch.is_tensor(inp) or inp.ndim != 4:
        raise TypeError("batch['inp'] must be a [B,C,H,W] tensor")
    batch_size, _, input_height, input_width = inp.shape
    image_paths = _batch_values(batch.get("img_path"), batch_size, "img_path")
    meta = batch.get("meta")
    if not isinstance(meta, Mapping) or "trans_input" not in meta:
        raise KeyError("batch['meta']['trans_input'] is required for cached detections")
    transforms = _batch_values(meta["trans_input"], batch_size, "trans_input")
    flipped_values = (
        _batch_values(meta["flipped"], batch_size, "flipped")
        if "flipped" in meta
        else [False] * batch_size
    )
    original_shapes = (
        _batch_values(meta["orig_hw"], batch_size, "orig_hw")
        if "orig_hw" in meta
        else [None] * batch_size
    )

    rows_by_image: List[List[List[float]]] = []
    for image_path, transform, flipped_value, original_shape in zip(
        image_paths,
        transforms,
        flipped_values,
        original_shapes,
    ):
        original = cache.lookup(image_path, missing=policy.missing)
        filtered = filter_detections(original, policy)
        flipped = bool(np.asarray(flipped_value).reshape(-1)[0])
        if flipped:
            shape = np.asarray(original_shape).reshape(-1)
            if shape.size != 2:
                raise ValueError("meta['orig_hw'] must contain [height, width] for flipped samples")
            original_width = int(shape[1])
            filtered = [flip_detection(detection, original_width) for detection in filtered]
        transformed = [
            mapped
            for detection in filtered
            for mapped in [transform_detection(detection, transform, input_width, input_height)]
            if mapped is not None
        ]
        rows_by_image.append([
            [*detection.bbox, detection.score, float(detection.class_id)]
            for detection in transformed
        ])

    max_count = max((len(rows) for rows in rows_by_image), default=0)
    output = inp.new_zeros((batch_size, max_count, 6))
    for index, rows in enumerate(rows_by_image):
        if rows:
            output[index, : len(rows)] = torch.as_tensor(rows, device=inp.device, dtype=inp.dtype)
    return output
