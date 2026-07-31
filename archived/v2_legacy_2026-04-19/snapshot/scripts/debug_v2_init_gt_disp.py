#!/usr/bin/env python3
import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parent
REPO_DIR = ROOT_DIR.parent
DEFAULT_CFG = REPO_DIR / "configs" / "btcv_diffusion_dit_v2.yaml"
DEFAULT_CKPT = REPO_DIR / "data" / "outputs" / "btcv_diffusion_dit_v2" / "checkpoints" / "latest.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Debug V2 init / GT / displacement visualization")
    parser.add_argument("--index", type=int, default=-1, help="Dataset index. Random if negative.")
    parser.add_argument("--instance", type=int, default=0, help="Instance index to zoom into.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when index < 0.")
    parser.add_argument("--save_dir", type=str, default=str(ROOT_DIR / "visual" / "v2_debug_init_gt_disp"))
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[*] Ignoring extra args: {' '.join(unknown)}")
    return args


ARGS = parse_args()
if not os.environ.get("CFG_FILE"):
    os.environ["CFG_FILE"] = str(DEFAULT_CFG)

# Keep lib.config from seeing our custom CLI args.
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(REPO_DIR))

from lib.config import cfg
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.datasets.collate_batch import make_collator
from lib.utils.snake import snake_config, snake_gcn_utils


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_uint8(img):
    arr = to_numpy(img)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


def signed_area(poly):
    x = poly[..., 0]
    y = poly[..., 1]
    x1 = torch.roll(x, shifts=-1, dims=1)
    y1 = torch.roll(y, shifts=-1, dims=1)
    return 0.5 * torch.sum(x * y1 - x1 * y, dim=1)


def align_gt_to_init(init_py, gt_py):
    if init_py.numel() == 0 or gt_py.numel() == 0:
        return gt_py
    gt_py = gt_py.clone()
    area_init = signed_area(init_py)
    area_gt = signed_area(gt_py)
    orient_mismatch = ((area_init >= 0) ^ (area_gt >= 0))
    if orient_mismatch.any():
        gt_py[orient_mismatch] = torch.flip(gt_py[orient_mismatch], dims=[1])

    d2 = (init_py[:, :1, :] - gt_py).pow(2).sum(-1)
    nearest = torch.argmin(d2, dim=1)
    rolled = []
    for i in range(gt_py.size(0)):
        s = int(nearest[i].item())
        rolled.append(torch.roll(gt_py[i], shifts=-s, dims=0) if s != 0 else gt_py[i])
    return torch.stack(rolled, dim=0)


def poly_to_input(poly):
    return to_numpy(poly).astype(np.float32) * float(snake_config.down_ratio)


def draw_poly(img, poly, color, thickness=2):
    poly = np.asarray(poly)
    if poly.size == 0:
        return
    pts = poly.astype(np.int32)
    cv2.polylines(img, [pts], True, color, thickness)


def draw_vectors(img, init_poly, gt_poly, color=(0, 220, 0), max_arrows=16):
    init_arr = np.asarray(init_poly)
    gt_arr = np.asarray(gt_poly)
    if init_arr.ndim != 3 or gt_arr.ndim != 3:
        return
    num_poly = min(init_arr.shape[0], gt_arr.shape[0])
    for k in range(num_poly):
        src = init_arr[k]
        dst = gt_arr[k]
        n = min(src.shape[0], dst.shape[0])
        if n == 0:
            continue
        step = max(1, n // max_arrows)
        for i in range(0, n, step):
            p0 = tuple(np.round(src[i]).astype(np.int32))
            p1 = tuple(np.round(dst[i]).astype(np.int32))
            if p0 == p1:
                continue
            cv2.arrowedLine(img, p0, p1, color, 1, tipLength=0.18)
            cv2.circle(img, p0, 2, color, -1)


def select_batch(dataset, collate_fn):
    if ARGS.index >= 0:
        sample = dataset[ARGS.index]
        return collate_fn([sample]), ARGS.index

    rng = random.Random(ARGS.seed)
    for _ in range(200):
        idx = rng.randrange(len(dataset))
        batch = collate_fn([dataset[idx]])
        init = snake_gcn_utils.prepare_training({}, batch)
        if init["i_it_py"].numel() > 0 and init["i_gt_py"].numel() > 0:
            return batch, idx
    raise RuntimeError("No valid sample found.")


def save_full_view(batch, init, save_path):
    img = ensure_uint8(batch["orig_img"][0])
    init_orig = poly_to_input(init["i_it_py"].detach().cpu())
    gt_orig = poly_to_input(init["i_gt_py"].detach().cpu())

    canvas = img.copy()
    for poly in gt_orig:
        draw_poly(canvas, poly, (255, 0, 0), thickness=2)
    for poly in init_orig:
        draw_poly(canvas, poly, (0, 255, 255), thickness=1)
    draw_vectors(canvas, init_orig, gt_orig)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, canvas)


def save_zoom_view(batch, init, instance_idx, save_path):
    img = ensure_uint8(batch["orig_img"][0])
    init_orig = poly_to_input(init["i_it_py"].detach().cpu())
    gt_orig = poly_to_input(init["i_gt_py"].detach().cpu())

    if init_orig.shape[0] == 0:
        return

    idx = max(0, min(instance_idx, init_orig.shape[0] - 1))
    src = init_orig[idx]
    dst = gt_orig[idx]

    pts = np.concatenate([src, dst], axis=0)
    x1 = int(np.floor(pts[:, 0].min()) - 40)
    y1 = int(np.floor(pts[:, 1].min()) - 40)
    x2 = int(np.ceil(pts[:, 0].max()) + 40)
    y2 = int(np.ceil(pts[:, 1].max()) + 40)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)

    crop = img[y1:y2, x1:x2].copy()
    off = np.array([x1, y1], dtype=np.float32)

    draw_poly(crop, dst - off, (255, 0, 0), thickness=2)
    draw_poly(crop, src - off, (0, 255, 255), thickness=1)
    draw_vectors(crop, (src - off)[None], (dst - off)[None])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, crop)


def main():
    if not Path(ARGS.ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ARGS.ckpt}")

    print(f"[*] CFG_FILE={os.environ.get('CFG_FILE')}")
    print(f"[*] Reference checkpoint: {ARGS.ckpt}")

    cfg.train.num_workers = 0
    dataset = make_dataset(cfg, cfg.train.dataset, make_transforms(cfg, is_train=True), is_train=True)
    collate_fn = make_collator(cfg)
    batch, idx = select_batch(dataset, collate_fn)
    init = snake_gcn_utils.prepare_training({}, batch)

    if init["i_it_py"].numel() == 0 or init["i_gt_py"].numel() == 0:
        raise RuntimeError("Selected sample has no valid instances.")

    init["i_gt_py"] = align_gt_to_init(init["i_it_py"], init["i_gt_py"])

    stem = Path(str(batch["img_path"][0])).stem if "img_path" in batch else f"idx{idx}"
    save_dir = Path(ARGS.save_dir)
    full_path = save_dir / f"v2_debug_full_{idx}_{stem}.png"
    zoom_path = save_dir / f"v2_debug_zoom_{idx}_{stem}_i{ARGS.instance}.png"

    save_full_view(batch, init, str(full_path))
    save_zoom_view(batch, init, ARGS.instance, str(zoom_path))

    print(f"[*] Sample index: {idx}")
    print(f"[*] Full view: {full_path}")
    print(f"[*] Zoom view: {zoom_path}")


if __name__ == "__main__":
    main()
