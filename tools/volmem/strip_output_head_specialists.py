#!/usr/bin/env python3
"""Create a fair output-head initialization checkpoint.

The source V4.6c checkpoint contains trained legacy output experts.  Keeping
those tensors would give the legacy control an initialization advantage over
new dense-residual and modern sparse heads.  This utility removes only
specialist/router tensors while retaining the shared norm, adaLN, and linear
displacement predictor.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch


SHARED_SUFFIXES = (
    "norm.weight",
    "linear.weight",
    "linear.bias",
    "adaLN.1.weight",
    "adaLN.1.bias",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    source_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else output_path.with_suffix(output_path.suffix + ".json")
    )
    if output_path.exists():
        raise FileExistsError("refusing to overwrite {}".format(output_path))

    checkpoint = torch.load(str(source_path), map_location="cpu")
    state_key = next(
        (key for key in ("state_dict", "model", "net") if key in checkpoint),
        None,
    )
    if state_key is None:
        raise KeyError("checkpoint does not contain state_dict/model/net")
    source_state = checkpoint[state_key]
    kept_state = source_state.__class__()
    removed = []
    retained_shared = []
    marker = "gcn.denoiser.final_layer."
    for key, value in source_state.items():
        if not key.startswith(marker):
            kept_state[key] = value
            continue
        suffix = key[len(marker):]
        if suffix in SHARED_SUFFIXES:
            kept_state[key] = value
            retained_shared.append(key)
        else:
            removed.append(key)

    if set(key[len(marker):] for key in retained_shared) != set(SHARED_SUFFIXES):
        raise RuntimeError("shared output-head tensors are incomplete")
    if not removed:
        raise RuntimeError("no specialist tensors were found")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint[state_key] = kept_state
    torch.save(checkpoint, str(output_path))
    manifest = {
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "state_key": state_key,
        "source_tensor_count": len(source_state),
        "output_tensor_count": len(kept_state),
        "retained_shared": retained_shared,
        "removed_specialists": removed,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
