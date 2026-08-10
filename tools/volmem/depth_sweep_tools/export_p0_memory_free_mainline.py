#!/usr/bin/env python3
"""Export the validated P0 base network without Memory or internal detector.

This is a mechanical, fail-closed state-dict transformation.  It does not
construct a model and it never carries optimizer or Memory state forward.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


SOURCE_PREFIX = "contour_adapter.slice_loss_wrapper."
WRAPPED_DENOISER_PREFIX = "net.gcn.denoiser.base_denoiser."
PLAIN_DENOISER_PREFIX = "net.gcn.denoiser."


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_schema_digest(state):
    rows = []
    for key in sorted(state):
        value = state[key]
        rows.append({
            "key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
        })
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0-checkpoint", required=True)
    parser.add_argument("--reference-pure2d-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    source_path = Path(args.p0_checkpoint).resolve()
    reference_path = Path(args.reference_pure2d_checkpoint).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite {output_path}")
    for path in (source_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_root = torch.load(source_path, map_location="cpu")
    reference_root = torch.load(reference_path, map_location="cpu")
    source_state = source_root.get("state_dict")
    reference_state = reference_root.get("state_dict")
    if not isinstance(source_state, dict) or not isinstance(reference_state, dict):
        raise TypeError("both checkpoints must contain state_dict mappings")

    exported = {}
    excluded_memory = []
    excluded_detector = []
    for key, value in source_state.items():
        if key.startswith("memory_encoder.") or key.startswith("memflow_controller."):
            excluded_memory.append(key)
            continue
        if not key.startswith(SOURCE_PREFIX):
            raise RuntimeError(f"unclassified source key: {key}")
        plain_key = key[len(SOURCE_PREFIX):]
        if plain_key.startswith("net.heatmap_detector."):
            excluded_detector.append(key)
            continue
        if plain_key.startswith(WRAPPED_DENOISER_PREFIX):
            plain_key = PLAIN_DENOISER_PREFIX + plain_key[len(WRAPPED_DENOISER_PREFIX):]
        if plain_key in exported:
            raise RuntimeError(f"duplicate mapped key: {plain_key}")
        exported[plain_key] = value.detach().cpu()

    missing = sorted(set(reference_state) - set(exported))
    unexpected = sorted(set(exported) - set(reference_state))
    shape_mismatch = [
        {
            "key": key,
            "expected": list(reference_state[key].shape),
            "observed": list(exported[key].shape),
        }
        for key in sorted(set(reference_state) & set(exported))
        if tuple(reference_state[key].shape) != tuple(exported[key].shape)
    ]
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(json.dumps({
            "missing": missing,
            "unexpected": unexpected,
            "shape_mismatch": shape_mismatch,
        }, sort_keys=True))
    if not excluded_memory or not excluded_detector:
        raise RuntimeError("source did not contain both Memory and detector parameters")
    forbidden = [
        key for key in exported
        if "memory_encoder" in key or "memflow_controller" in key or "heatmap_detector" in key
    ]
    if forbidden:
        raise RuntimeError(f"forbidden exported keys: {forbidden[:20]}")

    output_path.parent.mkdir(parents=True, exist_ok=False)
    payload = {
        "format_version": 1,
        "schema": "diffusionsnake.pure2d_memory_free_p0_step2000.v1",
        "step": int(source_root.get("step", 0)),
        "state_dict": exported,
        "source": {
            "p0_checkpoint": str(source_path),
            "p0_checkpoint_sha256": sha256_file(source_path),
            "reference_pure2d_checkpoint": str(reference_path),
            "reference_pure2d_checkpoint_sha256": sha256_file(reference_path),
            "source_state_keys": len(source_state),
            "excluded_memory_state_keys": len(excluded_memory),
            "excluded_detector_state_keys": len(excluded_detector),
        },
        "architecture": {
            "dit_layers": 6,
            "state_dim": 256,
            "heads": 8,
            "memory_parameters": 0,
            "internal_detector_parameters": 0,
            "model_parameters": 14373444,
            "state_keys": len(exported),
            "state_numel_including_buffers": sum(int(value.numel()) for value in exported.values()),
            "state_schema_sha256": state_schema_digest(exported),
        },
    }
    torch.save(payload, output_path)
    report = {
        "status": "PASS_MEMORY_AND_INTERNAL_DETECTOR_PHYSICALLY_REMOVED",
        "checkpoint": str(output_path),
        "checkpoint_sha256": sha256_file(output_path),
        "checkpoint_bytes": os.path.getsize(output_path),
        "schema": payload["schema"],
        "step": payload["step"],
        "source": payload["source"],
        "architecture": payload["architecture"],
        "strict_key_set_equal_to_reference": True,
        "strict_shape_equal_to_reference": True,
        "optimizer_state_carried": False,
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
