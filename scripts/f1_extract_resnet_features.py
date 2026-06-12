#!/usr/bin/env python3
"""
F1 ResNet34 stride-4 feature extraction for boundary probe experiments.

Run with snake1:
  CUDA_VISIBLE_DEVICES=0 /home/medteam/miniconda3/envs/snake1/bin/python scripts/f1_extract_resnet_features.py --split train
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO_ROOT / "configs" / "e3_v8_2_boxjitter_mixinit_gpu7.yaml"
DEFAULT_CKPT = REPO_ROOT / "data" / "outputs" / "e3_v8_2_boxjitter_mixinit_gpu7" / "checkpoints" / "latest.pt"
DEFAULT_DATA_ROOT = Path("/home/medteam/Zhrch/Datasets/1232_final")
DEFAULT_JSONL_ROOT = REPO_ROOT / "Eagle" / "Embodied" / "locany_recipe" / "1232_final"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "f1_resnet_feature_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-file", default=str(DEFAULT_CFG))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--jsonl-root", default=str(DEFAULT_JSONL_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--subset-size", type=int, default=300, help="train uses first N; test is always full")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--visible-gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--allow-large-cache", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


ARGS = parse_args()

# Project convention: CFG_FILE must be set before importing lib.* modules.
os.environ["CFG_FILE"] = str(Path(ARGS.cfg_file).resolve())
sys.argv = [sys.argv[0], "--cfg_file", os.environ["CFG_FILE"]]
if ARGS.visible_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.visible_gpu)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

from lib.config import cfg  # noqa: E402
from lib.networks import make_network  # noqa: E402
from lib.utils import data_utils  # noqa: E402
from lib.utils.snake import snake_config  # noqa: E402


def resolve_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(text)


def set_seed(seed: int) -> None:
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def apply_extract_overrides(input_size: int) -> None:
    cfg.gpus = [0]
    cfg.train_or_test = "test"
    cfg.use_gt_det = False
    cfg.skip_diffusion_forward = True
    cfg.use_extreme_refine = False
    cfg.locate_feat_inject = False
    snake_config.voc_input_h = int(input_size)
    snake_config.voc_input_w = int(input_size)


def load_jsonl_records(jsonl_root: Path, split: str) -> list[dict[str, Any]]:
    path = jsonl_root / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing jsonl: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_records(records: list[dict[str, Any]], split: str, subset_size: int) -> list[dict[str, Any]]:
    if split == "train":
        n = max(int(subset_size), 0)
        return records[:n] if n > 0 else records
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


def square_affine_meta(width: int, height: int, input_size: int) -> dict[str, Any]:
    center = np.asarray([width / 2.0, height / 2.0], dtype=np.float32)
    scale = np.asarray([max(width, height), max(width, height)], dtype=np.float32)
    trans_input = data_utils.get_affine_transform(center, scale, 0, [input_size, input_size])
    return {
        "center": center,
        "affine_scale": scale,
        "trans_input": trans_input,
        "scale": float(input_size) / float(max(width, height)),
        "resized_hw": (
            max(int(round(height * float(input_size) / float(max(width, height)))), 1),
            max(int(round(width * float(input_size) / float(max(width, height)))), 1),
        ),
        "padded_hw": (int(input_size), int(input_size)),
        "input_hw": (int(input_size), int(input_size)),
        "grid_hw": (int(input_size) // 4, int(input_size) // 4),
        "pad": (0, 0, 0, 0),
    }


def preprocess_image(path: Path, input_size: int) -> tuple[torch.Tensor, dict[str, Any]]:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    height, width = img.shape[:2]
    meta = square_affine_meta(width, height, input_size)
    inp = cv2.warpAffine(img, meta["trans_input"], (input_size, input_size), flags=cv2.INTER_LINEAR)
    arr = inp.astype(np.float32) / 255.0
    arr = (arr - snake_config.mean) / snake_config.std
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).contiguous().float()
    meta.update(
        {
            "orig_hw": (height, width),
            "image_path": str(path),
            "image_name": path.name,
        }
    )
    return tensor, meta


def estimate_cache(paths: list[Path], input_size: int, channels: int = 64) -> dict[str, Any]:
    feat_h = int(input_size) // 4
    feat_w = int(input_size) // 4
    total_bytes = len(paths) * feat_h * feat_w * channels * 2
    return {
        "images": len(paths),
        "channels": channels,
        "bytes": total_bytes,
        "gb": total_bytes / float(1000**3),
        "gib": total_bytes / float(1024**3),
        "grid_counts": {f"{feat_h}x{feat_w}": len(paths)},
    }


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    network = make_network(cfg)
    ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt_obj.get("state_dict") or ckpt_obj.get("model") or ckpt_obj.get("net") or ckpt_obj
    from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict

    state = remap_legacy_state_dict(state)
    target = network.state_dict()
    reusable = {}
    for key, value in state.items():
        clean = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "network.", "net."):
                if clean.startswith(prefix):
                    clean = clean[len(prefix) :]
                    changed = True
        if clean in target and hasattr(value, "shape") and tuple(value.shape) == tuple(target[clean].shape):
            reusable[clean] = value
    info = network.load_state_dict(reusable, strict=False)
    print(
        f"[*] Loaded checkpoint {ckpt_path} | reused={len(reusable)}/{len(target)} "
        f"missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}",
        flush=True,
    )
    network.to(device).eval()
    return network


@torch.inference_mode()
def extract_feature(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> np.ndarray:
    core = model.net if hasattr(model, "net") else model
    x = x.to(device=device, non_blocking=(device.type == "cuda"))
    if not hasattr(core, "heatmap_detector"):
        raise RuntimeError("Expected heatmap_detector on E3 ResNet network")
    feat, _ct_hm, _wh, mask_logits = core.heatmap_detector(x)
    if mask_logits is not None:
        alpha = float(getattr(cfg, "heatmap_mask_guidance_alpha", 0.0))
        if alpha > 0.0:
            guidance = torch.sigmoid(mask_logits).amax(dim=1, keepdim=True)
            feat = feat * (1.0 + alpha * guidance)
    return feat[0].detach().to(torch.float16).cpu().numpy()


def save_npz(out_path: Path, feat: np.ndarray, meta: dict[str, Any], input_size: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gh, gw = meta["grid_hw"]
    oh, ow = meta["orig_hw"]
    rh, rw = meta["resized_hw"]
    ph, pw = meta["padded_hw"]
    ih, iw = meta["input_hw"]
    pad_left, pad_top, pad_right, pad_bottom = meta["pad"]
    np.savez(
        out_path,
        feat=feat,
        resnet_stride4=feat,
        grid_hw=np.asarray([gh, gw], dtype=np.int32),
        orig_hw=np.asarray([oh, ow], dtype=np.int32),
        resized_hw=np.asarray([rh, rw], dtype=np.int32),
        padded_hw=np.asarray([ph, pw], dtype=np.int32),
        input_hw=np.asarray([ih, iw], dtype=np.int32),
        pad=np.asarray([pad_left, pad_top, pad_right, pad_bottom], dtype=np.int32),
        scale=np.asarray([meta["scale"]], dtype=np.float32),
        trans_input=np.asarray(meta["trans_input"], dtype=np.float32),
        center=np.asarray(meta["center"], dtype=np.float32),
        affine_scale=np.asarray(meta["affine_scale"], dtype=np.float32),
        stride=np.asarray([4], dtype=np.int32),
        input_size=np.asarray([input_size], dtype=np.int32),
        image_path=np.asarray(meta["image_path"]),
    )


def main() -> None:
    args = ARGS
    cfg_file = Path(args.cfg_file).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    data_root = Path(args.data_root).resolve()
    jsonl_root = Path(args.jsonl_root).resolve()
    out_root = Path(args.out_dir).resolve()
    device = resolve_device(args.device)
    set_seed(args.seed)
    apply_extract_overrides(args.input_size)

    records = select_records(load_jsonl_records(jsonl_root, args.split), args.split, args.subset_size)
    paths = [resolve_image_path(data_root, args.split, rec) for rec in records]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing image: {missing[0]}")

    estimate = estimate_cache(paths, args.input_size, channels=64)
    print(
        f"[*] Cache estimate: split={args.split} images={estimate['images']} input={args.input_size} "
        f"channels=64 bytes={estimate['bytes']} ({estimate['gb']:.2f} GB / {estimate['gib']:.2f} GiB) "
        f"grid_counts={estimate['grid_counts']}",
        flush=True,
    )
    if estimate["bytes"] > 40 * 1024**3 and not args.allow_large_cache:
        raise RuntimeError("Estimated cache exceeds 40 GiB; reduce --subset-size or pass --allow-large-cache")
    if args.estimate_only:
        return

    print(f"[*] cfg={cfg_file} ckpt={ckpt_path} device={device}", flush=True)
    model = load_model(ckpt_path, device)

    done = 0
    out_dir = out_root / f"resnet_{args.input_size}" / args.split
    with torch.no_grad():
        for idx, image_path in enumerate(paths, start=1):
            out_path = out_dir / f"{image_path.stem}.npz"
            if out_path.exists() and not args.overwrite:
                continue
            x, meta = preprocess_image(image_path, args.input_size)
            feat = extract_feature(model, x, device)
            save_npz(out_path, feat, meta, args.input_size)
            done += 1
            if idx <= 2 or (args.progress_every > 0 and idx % args.progress_every == 0):
                print(
                    f"[{args.split} {idx}/{len(paths)}] {image_path.name} feat_shape={tuple(feat.shape)} "
                    f"orig_hw={meta['orig_hw']} file={out_path.stat().st_size / (1024 ** 2):.2f} MiB -> {out_path}",
                    flush=True,
                )

    print(f"[*] Finished. New/overwritten files: {done} under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
