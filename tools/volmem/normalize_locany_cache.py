#!/usr/bin/env python3
"""Normalize LocateAnything raw-response batch_infer JSONL into a canonical box cache.

Input rows use the ``image/query/raw_response/stats`` schema.
"""

import argparse
from collections.abc import Mapping
import json
import sys
from pathlib import Path
from typing import Dict, List

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from volmem.detection import parse_raw_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="LocateAnything batch_infer raw-response JSONL (image/query/raw_response/stats)",
    )
    parser.add_argument("--output", required=True, help="Canonical cache JSON path")
    parser.add_argument("--image-root", default="", help="Optional root for relative image paths")
    parser.add_argument("--strict", action="store_true", help="Fail when a response has no valid box")
    return parser.parse_args()


def normalize_row(row: Dict[str, object], image_root: Path, strict: bool) -> Dict[str, object]:
    if "raw_response" not in row:
        raise ValueError("input row is missing raw_response")
    raw_response = row["raw_response"]
    if not isinstance(raw_response, str):
        raise TypeError("input row raw_response must be a string")
    if "query" not in row:
        raise ValueError("input row is missing query")
    query = row["query"]
    if not isinstance(query, str):
        raise TypeError("input row query must be a string")
    if "stats" not in row:
        raise ValueError("input row is missing stats")
    if not isinstance(row["stats"], Mapping):
        raise TypeError("input row stats must be an object")
    if "image" not in row:
        raise ValueError("input row is missing image")
    image_value = row["image"]
    if not isinstance(image_value, str):
        raise TypeError("input row image must be a string")
    if not image_value:
        raise ValueError("input row image must be non-empty")
    image_path = Path(image_value)
    if not image_path.is_absolute():
        image_path = image_root / image_path
    image_path = image_path.resolve()
    with Image.open(image_path) as image:
        width, height = image.size
    detections = parse_raw_response(raw_response, width, height)
    if strict and not detections:
        raise ValueError(f"no valid box in LocateAnything response for {image_path}")
    instances: List[Dict[str, object]] = []
    for detection in detections:
        instances.append({
            "bbox": list(detection.bbox),
            "score": detection.score,
            "label": detection.label,
            "class_id": detection.class_id,
            "source": "LocateAnything",
        })
    return {
        "id": image_path.stem,
        "img_path": str(image_path),
        "image_path": str(image_path),
        "image_rel": image_value,
        "width": width,
        "height": height,
        "query": query,
        "raw_response": raw_response,
        "instances": instances,
    }


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root or ".").resolve()
    records = []
    with Path(args.input).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(normalize_row(json.loads(line), image_root, args.strict))
            except Exception as error:
                raise RuntimeError(f"failed to normalize input line {line_number}: {error}") from error
    payload = {
        "format": "volmem_locany_cache_v1",
        "coordinate_space": "original_image_pixels",
        "score_policy": "LocateAnything text output is uncalibrated; parsed boxes use score=1.0",
        "samples": records,
        "summary": {
            "samples": len(records),
            "samples_with_boxes": sum(bool(record["instances"]) for record in records),
            "boxes": sum(len(record["instances"]) for record in records),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
