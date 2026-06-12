#!/usr/bin/env python3
"""
F1 MoonViT layer feature extraction for boundary probe experiments.

Run with locany311:
  CUDA_VISIBLE_DEVICES=0 /home/medteam/Zhrch/.venvs/locany311/bin/python scripts/f1_extract_locate_layers.py --split train
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "Eagle"
    / "Embodied"
    / "work_dirs"
    / "1232_final_locany_full_more10000"
    / "checkpoint-3000"
)
DEFAULT_DATA_ROOT = Path("/home/medteam/Zhrch/Datasets/1232_final")
DEFAULT_JSONL_ROOT = REPO_ROOT / "Eagle" / "Embodied" / "locany_recipe" / "1232_final"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "f1_locate_layer_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--jsonl-root", default=str(DEFAULT_JSONL_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--subset-size", type=int, default=300, help="train uses first N; test is always full")
    parser.add_argument("--layers", default="5,9,13,18,22,26", help="1-based encoder layer indices")
    parser.add_argument("--input-size", type=int, choices=(448, 896), default=448)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--allow-large-cache", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def parse_layers(text: str) -> list[int]:
    layers = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            layers.append(int(item))
    if not layers:
        raise ValueError("At least one layer is required")
    if any(x <= 0 for x in layers):
        raise ValueError(f"Layers are 1-based positive indices: {layers}")
    return sorted(set(layers))


def resolve_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(text)


def resolve_dtype(text: str, device: torch.device) -> torch.dtype:
    if text == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[text]


def load_jsonl_records(jsonl_root: Path, split: str) -> list[dict[str, Any]]:
    path = jsonl_root / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing jsonl: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_image_path(data_root: Path, split: str, rec: dict[str, Any]) -> Path:
    image = rec.get("image") or rec.get("image_path") or rec.get("img_path") or rec.get("file_name")
    if image is None:
        raise KeyError(f"Record lacks image field: keys={sorted(rec.keys())}")
    p = Path(str(image))
    if p.is_absolute():
        return p
    if len(p.parts) > 1 and p.parts[0] in ("train", "test"):
        return data_root / p
    return data_root / split / p


def select_records(records: list[dict[str, Any]], split: str, subset_size: int) -> list[dict[str, Any]]:
    if split == "train":
        n = max(int(subset_size), 0)
        return records[:n] if n > 0 else records
    return records


def resized_padded_hw(width: int, height: int, input_size: int, patch_size: int) -> dict[str, Any]:
    scale = float(input_size) / float(max(width, height))
    resized_w = max(int(round(width * scale)), 1)
    resized_h = max(int(round(height * scale)), 1)
    padded_w = int(math.ceil(resized_w / patch_size) * patch_size)
    padded_h = int(math.ceil(resized_h / patch_size) * patch_size)
    return {
        "scale": scale,
        "resized_hw": (resized_h, resized_w),
        "padded_hw": (padded_h, padded_w),
        "grid_hw": (padded_h // patch_size, padded_w // patch_size),
        "pad": (0, 0, padded_w - resized_w, padded_h - resized_h),
    }


def preprocess_image(path: Path, input_size: int, patch_size: int) -> tuple[torch.Tensor, dict[str, Any]]:
    img = Image.open(path).convert("RGB")
    width, height = img.size
    meta = resized_padded_hw(width, height, input_size, patch_size)
    resized_h, resized_w = meta["resized_hw"]
    padded_h, padded_w = meta["padded_hw"]
    resized = img.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    padded = np.full((padded_h, padded_w, 3), 0.5, dtype=np.float32)
    padded[:resized_h, :resized_w, :] = arr
    tensor = torch.from_numpy(padded).permute(2, 0, 1).contiguous()
    tensor = (tensor - 0.5) / 0.5
    patches = tensor.reshape(3, padded_h // patch_size, patch_size, padded_w // patch_size, patch_size)
    patches = patches.permute(1, 3, 0, 2, 4).contiguous().view(-1, 3, patch_size, patch_size)
    meta.update(
        {
            "orig_hw": (height, width),
            "image_path": str(path),
            "image_name": path.name,
            "input_hw": (padded_h, padded_w),
        }
    )
    return patches, meta


def estimate_cache(paths: list[Path], input_size: int, patch_size: int, channels: int) -> dict[str, Any]:
    total_bytes = 0
    grid_counts: dict[tuple[int, int], int] = {}
    for path in paths:
        with Image.open(path) as img:
            width, height = img.size
        meta = resized_padded_hw(width, height, input_size, patch_size)
        gh, gw = meta["grid_hw"]
        total_bytes += int(gh * gw * channels * 2)
        grid_counts[(gh, gw)] = grid_counts.get((gh, gw), 0) + 1
    return {
        "images": len(paths),
        "channels": channels,
        "bytes": total_bytes,
        "gb": total_bytes / float(1000**3),
        "gib": total_bytes / float(1024**3),
        "grid_counts": {f"{k[0]}x{k[1]}": v for k, v in sorted(grid_counts.items())},
    }


def load_moonvit_module(checkpoint: Path):
    module_path = checkpoint / "modeling_vit.py"
    if not module_path.exists():
        module_path = REPO_ROOT / "Eagle" / "Embodied" / "eaglevl" / "model" / "moon_vit" / "modeling_vit.py"
    spec = importlib.util.spec_from_file_location("f1_moon_vit", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MoonViT module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("f1_moon_vit", module)
    spec.loader.exec_module(module)
    return module


def load_vision_model(checkpoint: Path, device: torch.device, dtype: torch.dtype):
    config = json.loads((checkpoint / "config.json").read_text())
    vision_cfg = dict(config["vision_config"])
    vision_cfg["_attn_implementation"] = vision_cfg.get("_attn_implementation") or config.get("_attn_implementation") or "sdpa"
    module = load_moonvit_module(checkpoint)
    moon_cfg = module.MoonViTConfig(**vision_cfg)
    model = module.MoonVitPretrainedModel(moon_cfg)

    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise ImportError(f"safetensors is required in locany311: {exc}") from exc

    weight_map = json.loads(index_path.read_text()).get("weight_map", {})
    shards = sorted({filename for key, filename in weight_map.items() if key.startswith("vision_model.")})
    state = {}
    for shard in shards:
        shard_path = checkpoint / shard
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("vision_model."):
                    state[key[len("vision_model.") :]] = f.get_tensor(key)

    info = model.load_state_dict(state, strict=False)
    if info.missing_keys or info.unexpected_keys:
        print(
            f"[WARN] vision load missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}",
            flush=True,
        )
    print(
        f"[*] Loaded MoonViT vision weights: keys={len(state)} shards={shards} "
        f"hidden={moon_cfg.hidden_size} layers={moon_cfg.num_hidden_layers} patch={moon_cfg.patch_size}",
        flush=True,
    )
    model.encoder.gradient_checkpointing = False
    model.to(device=device, dtype=dtype).eval()
    return model, moon_cfg


@torch.inference_mode()
def extract_layers(
    model: torch.nn.Module,
    patches: torch.Tensor,
    grid_hw: tuple[int, int],
    layers: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    grid_hws = torch.tensor([[gh, gw]], device=device, dtype=torch.int32)
    patches = patches.to(device=device, dtype=dtype, non_blocking=(device.type == "cuda"))

    try:
        out = model(patches, grid_hws, output_hidden_states=True, return_dict=True)
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is not None:
            feats = {}
            for layer_idx in layers:
                hidden = hidden_states[layer_idx]
                feats[f"layer_{layer_idx}"] = (
                    hidden.view(gh, gw, -1).permute(2, 0, 1).contiguous().to(torch.float16).cpu().numpy()
                )
            return feats
    except TypeError:
        pass

    hidden = model.patch_embed(patches, grid_hws)
    rope_freqs_cis = model.encoder.rope_2d.get_freqs_cis(grid_hws=grid_hws)
    lengths = torch.cat(
        (
            torch.zeros(1, device=device, dtype=grid_hws.dtype),
            grid_hws[:, 0] * grid_hws[:, 1],
        )
    )
    cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)

    captures = {}
    requested = set(layers)
    max_layer = max(layers)
    for layer_idx, block in enumerate(model.encoder.blocks, start=1):
        hidden = block(hidden, cu_seqlens, rope_freqs_cis=rope_freqs_cis)
        if layer_idx in requested:
            captures[layer_idx] = hidden.detach()
        if layer_idx >= max_layer:
            break

    feats = {}
    for layer_idx in layers:
        if layer_idx not in captures:
            raise RuntimeError(f"Layer {layer_idx} was not captured")
        feat = captures[layer_idx].view(gh, gw, -1).permute(2, 0, 1).contiguous()
        feats[f"layer_{layer_idx}"] = feat.to(torch.float16).cpu().numpy()
    return feats


def save_npz(out_path: Path, feats: dict[str, np.ndarray], meta: dict[str, Any], layers: list[int], patch_size: int, input_size: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gh, gw = meta["grid_hw"]
    oh, ow = meta["orig_hw"]
    rh, rw = meta["resized_hw"]
    ph, pw = meta["padded_hw"]
    ih, iw = meta["input_hw"]
    pad_left, pad_top, pad_right, pad_bottom = meta["pad"]
    payload: dict[str, Any] = {
        **feats,
        "grid_hw": np.asarray([gh, gw], dtype=np.int32),
        "orig_hw": np.asarray([oh, ow], dtype=np.int32),
        "resized_hw": np.asarray([rh, rw], dtype=np.int32),
        "padded_hw": np.asarray([ph, pw], dtype=np.int32),
        "input_hw": np.asarray([ih, iw], dtype=np.int32),
        "pad": np.asarray([pad_left, pad_top, pad_right, pad_bottom], dtype=np.int32),
        "scale": np.asarray([meta["scale"]], dtype=np.float32),
        "layers": np.asarray(layers, dtype=np.int32),
        "patch_size": np.asarray([patch_size], dtype=np.int32),
        "input_size": np.asarray([input_size], dtype=np.int32),
        "long_side": np.asarray([input_size], dtype=np.int32),
        "image_path": np.asarray(meta["image_path"]),
    }
    np.savez(out_path, **payload)


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    jsonl_root = Path(args.jsonl_root).resolve()
    out_root = Path(args.out_dir).resolve()
    layers = parse_layers(args.layers)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    records = select_records(load_jsonl_records(jsonl_root, args.split), args.split, args.subset_size)
    paths = [resolve_image_path(data_root, args.split, rec) for rec in records]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing image: {missing[0]}")

    channels = 1152 * len(layers)
    estimate = estimate_cache(paths, args.input_size, args.patch_size, channels)
    print(
        f"[*] Cache estimate: split={args.split} images={estimate['images']} "
        f"layers={layers} channels_total={channels} "
        f"bytes={estimate['bytes']} ({estimate['gb']:.2f} GB / {estimate['gib']:.2f} GiB) "
        f"grid_counts={estimate['grid_counts']}",
        flush=True,
    )
    if estimate["bytes"] > 40 * 1024**3 and not args.allow_large_cache:
        raise RuntimeError("Estimated cache exceeds 40 GiB; reduce --subset-size/--layers or pass --allow-large-cache")
    if args.estimate_only:
        return

    model, moon_cfg = load_vision_model(checkpoint, device, dtype)
    if max(layers) > int(moon_cfg.num_hidden_layers):
        raise ValueError(f"Requested layer {max(layers)} > num_hidden_layers={moon_cfg.num_hidden_layers}")

    done = 0
    out_dir = out_root / f"locate_{args.input_size}" / args.split
    with torch.no_grad():
        for idx, image_path in enumerate(paths, start=1):
            out_path = out_dir / f"{image_path.stem}.npz"
            if out_path.exists() and not args.overwrite:
                continue
            patches, meta = preprocess_image(image_path, args.input_size, args.patch_size)
            feats = extract_layers(model, patches, meta["grid_hw"], layers, device, dtype)
            save_npz(out_path, feats, meta, layers, args.patch_size, args.input_size)
            done += 1
            if idx <= 2 or (args.progress_every > 0 and idx % args.progress_every == 0):
                shapes = {k: tuple(v.shape) for k, v in feats.items()}
                print(
                    f"[{args.split} {idx}/{len(paths)}] {image_path.name} shapes={shapes} "
                    f"grid_hw={meta['grid_hw']} orig_hw={meta['orig_hw']} "
                    f"file={out_path.stat().st_size / (1024 ** 2):.2f} MiB -> {out_path}",
                    flush=True,
                )

    print(f"[*] Finished. New/overwritten files: {done} under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
