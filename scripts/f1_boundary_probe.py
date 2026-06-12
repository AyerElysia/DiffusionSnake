#!/usr/bin/env python3
"""
F1 boundary information probe on frozen feature caches.

Examples:
  /home/medteam/miniconda3/envs/snake1/bin/python scripts/f1_boundary_probe.py \
    --feat-dir data/f1_locate_layer_cache/locate_448 --feat-key layer_9

  /home/medteam/miniconda3/envs/snake1/bin/python scripts/f1_boundary_probe.py --smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/medteam/Zhrch/Datasets/1232_final")
DEFAULT_JSONL_ROOT = REPO_ROOT / "Eagle" / "Embodied" / "locany_recipe" / "1232_final"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "f1_boundary_probe_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feat-dir", default="", help="Root containing train/ and test/ feature npz files")
    parser.add_argument("--train-feat-dir", default="", help="Optional explicit train feature directory")
    parser.add_argument("--test-feat-dir", default="", help="Optional explicit test feature directory")
    parser.add_argument("--feat-key", default="feat", help="npz key to train on, e.g. layer_9 or resnet_stride4")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--jsonl-root", default=str(DEFAULT_JSONL_ROOT))
    parser.add_argument("--train-subset-size", type=int, default=300)
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_DIR / "latest.json"))
    parser.add_argument("--summary-json", default=str(DEFAULT_OUT_DIR / "summary.json"))
    parser.add_argument(
        "--debug-vis",
        nargs="?",
        const=str(DEFAULT_OUT_DIR / "debug_vis"),
        default="",
        help="Save first 3 GT boundary overlays to this directory",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a 2-epoch CPU smoke test with fake npz and fake polygons")
    return parser.parse_args()


ARGS = parse_args()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def set_seed(seed: int) -> None:
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def resolve_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(text)


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


def select_records(records: list[dict[str, Any]], split: str, train_subset_size: int) -> list[dict[str, Any]]:
    if split == "train":
        n = max(int(train_subset_size), 0)
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


def resolve_split_feat_dir(args: argparse.Namespace, split: str) -> Path:
    explicit = getattr(args, f"{split}_feat_dir")
    if explicit:
        return Path(explicit).resolve()
    root = Path(args.feat_dir).resolve() if args.feat_dir else None
    if root is None:
        raise ValueError("Provide --feat-dir or both --train-feat-dir/--test-feat-dir")
    candidate = root / split
    return candidate if candidate.exists() else root


def feature_path_for(feat_dir: Path, image_path: Path) -> Path:
    return feat_dir / f"{image_path.stem}.npz"


def _as_poly_array(value: Any) -> list[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return []
    if arr.size < 6:
        return []
    polys = []
    if arr.ndim == 1 and arr.size % 2 == 0:
        arr = arr.reshape(-1, 2)
        if arr.shape[0] >= 3:
            polys.append(arr)
    elif arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 3:
        polys.append(arr)
    elif arr.ndim == 3 and arr.shape[-1] == 2:
        for poly in arr:
            if poly.shape[0] >= 3:
                polys.append(np.asarray(poly, dtype=np.float32))
    return polys


def polygons_from_record(rec: dict[str, Any]) -> list[np.ndarray]:
    poly_keys = ("polygon", "polygons", "segmentation", "segments", "contour", "contours")
    out: list[np.ndarray] = []

    def visit(obj: Any, from_poly_key: bool = False) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_l = str(key).lower()
                if key_l in poly_keys:
                    visit(value, True)
                elif key_l in ("instances", "objects", "labels", "annotations"):
                    visit(value, False)
        elif isinstance(obj, list):
            if from_poly_key:
                parsed = _as_poly_array(obj)
                if parsed:
                    out.extend(parsed)
                    return
            for item in obj:
                visit(item, from_poly_key)
        elif from_poly_key:
            out.extend(_as_poly_array(obj))

    visit(rec, False)
    return out


def polygons_from_masks(image_path: Path) -> list[np.ndarray]:
    stem = image_path.stem
    base = stem.split("_")[0]
    mask_paths = sorted(image_path.parent.glob(f"{base}_mask*"))
    polys: list[np.ndarray] = []
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        binary = (mask > 0).astype(np.uint8)
        found = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = found[0] if len(found) == 2 else found[1]
        for contour in contours:
            pts = contour.reshape(-1, 2).astype(np.float32)
            if pts.shape[0] >= 3:
                polys.append(pts)
    return polys


def load_polygons(rec: dict[str, Any], image_path: Path) -> list[np.ndarray]:
    polys = polygons_from_record(rec)
    if polys:
        return polys
    return polygons_from_masks(image_path)


def npz_get(npz: Any, key: str, default: Any) -> np.ndarray:
    if key in npz.files:
        return np.asarray(npz[key])
    return np.asarray(default)


def npz_meta(npz: Any, feat_shape: tuple[int, int, int]) -> dict[str, Any]:
    _c, h, w = feat_shape
    input_default = npz_get(npz, "padded_hw", [h, w])
    return {
        "orig_hw": npz_get(npz, "orig_hw", [0, 0]).astype(np.int32),
        "resized_hw": npz_get(npz, "resized_hw", input_default).astype(np.int32),
        "padded_hw": npz_get(npz, "padded_hw", input_default).astype(np.int32),
        "input_hw": npz_get(npz, "input_hw", input_default).astype(np.int32),
        "grid_hw": npz_get(npz, "grid_hw", [h, w]).astype(np.int32),
        "pad": npz_get(npz, "pad", [0, 0, 0, 0]).astype(np.float32),
        "scale": float(npz_get(npz, "scale", [1.0]).reshape(-1)[0]),
        "trans_input": np.asarray(npz["trans_input"], dtype=np.float32) if "trans_input" in npz.files else None,
    }


def load_feature_array(npz: Any, feat_key: str) -> np.ndarray:
    if feat_key in npz.files:
        arr = np.asarray(npz[feat_key])
    elif feat_key.startswith("layer_") and "feat" in npz.files and "layers" in npz.files:
        layer = int(feat_key.split("_", 1)[1])
        layers = [int(x) for x in np.asarray(npz["layers"]).reshape(-1).tolist()]
        if layer not in layers:
            raise KeyError(f"{feat_key} not in npz layers={layers}")
        feat = np.asarray(npz["feat"])
        if feat.ndim != 3 or feat.shape[0] % len(layers) != 0:
            raise ValueError(f"Cannot slice concatenated feat with shape={feat.shape} layers={layers}")
        c = feat.shape[0] // len(layers)
        pos = layers.index(layer)
        arr = feat[pos * c : (pos + 1) * c]
    else:
        raise KeyError(f"Feature key {feat_key!r} missing. Available keys: {npz.files}")

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected feature shape C,H,W for key={feat_key}, got {arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def transform_polygons_to_target(polys: list[np.ndarray], meta: dict[str, Any], target_size: int) -> list[np.ndarray]:
    input_h, input_w = [float(x) for x in np.asarray(meta["input_hw"]).reshape(-1)[:2]]
    input_h = max(input_h, 1.0)
    input_w = max(input_w, 1.0)
    out = []
    trans = meta.get("trans_input")
    pad = np.asarray(meta.get("pad", [0, 0, 0, 0]), dtype=np.float32).reshape(-1)
    pad_left = float(pad[0]) if pad.size >= 1 else 0.0
    pad_top = float(pad[1]) if pad.size >= 2 else 0.0
    for poly in polys:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2).copy()
        if pts.shape[0] < 3:
            continue
        if trans is not None:
            pts = np.dot(pts, trans[:, :2].T) + trans[:, 2]
        else:
            pts[:, 0] = pts[:, 0] * float(meta["scale"]) + pad_left
            pts[:, 1] = pts[:, 1] * float(meta["scale"]) + pad_top
        pts[:, 0] = pts[:, 0] * float(target_size) / input_w
        pts[:, 1] = pts[:, 1] * float(target_size) / input_h
        pts[:, 0] = np.clip(pts[:, 0], 0, target_size - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, target_size - 1)
        if np.unique(np.round(pts).astype(np.int32), axis=0).shape[0] >= 3:
            out.append(pts)
    return out


def rasterize_boundary(polys_target: list[np.ndarray], target_size: int) -> np.ndarray:
    boundary = np.zeros((target_size, target_size), dtype=np.uint8)
    for poly in polys_target:
        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        if pts.shape[0] >= 3:
            cv2.polylines(boundary, [pts], isClosed=True, color=1, thickness=1, lineType=cv2.LINE_8)
    return boundary


class BoundaryFeatureDataset:
    def __init__(
        self,
        records: list[dict[str, Any]],
        split: str,
        data_root: Path,
        feat_dir: Path,
        feat_key: str,
        target_size: int,
    ) -> None:
        self.records = records
        self.split = split
        self.data_root = data_root
        self.feat_dir = feat_dir
        self.feat_key = feat_key
        self.target_size = int(target_size)
        self.items = []
        missing = []
        no_poly = []
        for rec in records:
            image_path = resolve_image_path(data_root, split, rec)
            feat_path = feature_path_for(feat_dir, image_path)
            if not feat_path.exists():
                missing.append(feat_path)
                continue
            polys = load_polygons(rec, image_path)
            if not polys:
                no_poly.append(image_path)
                continue
            self.items.append({"record": rec, "image_path": image_path, "feat_path": feat_path, "polygons": polys})
        if missing:
            raise FileNotFoundError(f"Missing feature cache for {len(missing)} samples, first={missing[0]}")
        if no_poly:
            raise RuntimeError(f"No polygons/masks for {len(no_poly)} samples, first={no_poly[0]}")
        if not self.items:
            raise RuntimeError(f"No usable samples for split={split}")

    def __len__(self) -> int:
        return len(self.items)

    def load_npz_sample(self, idx: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[np.ndarray]]:
        item = self.items[idx]
        with np.load(item["feat_path"]) as npz:
            feat = load_feature_array(npz, self.feat_key)
            meta = npz_meta(npz, tuple(feat.shape))
        polys_target = transform_polygons_to_target(item["polygons"], meta, self.target_size)
        boundary = rasterize_boundary(polys_target, self.target_size)
        return feat, boundary, meta, polys_target

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        feat, boundary, _meta, _polys_target = self.load_npz_sample(idx)
        feat_t = torch.from_numpy(np.ascontiguousarray(feat)).float()
        target_t = torch.from_numpy(boundary[None].astype(np.float32))
        return feat_t, target_t


class BoundaryProbe(nn.Module):
    def __init__(self, in_channels: int, feat_hw: tuple[int, int], target_size: int = 128, hidden: int = 64) -> None:
        super().__init__()
        self.target_size = int(target_size)
        self.stem = nn.Sequential(
            nn.Conv2d(int(in_channels), hidden, kernel_size=1, bias=True),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        max_hw = max(int(feat_hw[0]), int(feat_hw[1]), 1)
        stages = 0
        cur = max_hw
        while cur < self.target_size:
            stages += 1
            cur *= 2
        stages = min(stages, 2)
        blocks = []
        for _ in range(stages):
            blocks.extend(
                [
                    nn.Conv2d(hidden, hidden * 4, kernel_size=3, padding=1, bias=True),
                    nn.PixelShuffle(2),
                    nn.GroupNorm(8, hidden),
                    nn.GELU(),
                ]
            )
        self.up = nn.Sequential(*blocks)
        self.out = nn.Conv2d(hidden, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.mid(x)
        x = self.up(x)
        if x.shape[-2:] != (self.target_size, self.target_size):
            x = F.interpolate(x, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        return self.out(x)


def probe_param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def boundary_fscore(pred: np.ndarray, gt: np.ndarray, tolerance: int) -> float:
    pred_u = (pred > 0).astype(np.uint8)
    gt_u = (gt > 0).astype(np.uint8)
    pred_count = int(pred_u.sum())
    gt_count = int(gt_u.sum())
    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0
    dist_to_gt = cv2.distanceTransform((1 - gt_u).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_pred = cv2.distanceTransform((1 - pred_u).astype(np.uint8), cv2.DIST_L2, 3)
    precision = float(((pred_u > 0) & (dist_to_gt <= float(tolerance))).sum()) / float(pred_count)
    recall = float(((gt_u > 0) & (dist_to_pred <= float(tolerance))).sum()) / float(gt_count)
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def boundary_iou(pred: np.ndarray, gt: np.ndarray, dilation_px: int = 2) -> float:
    pred_u = (pred > 0).astype(np.uint8)
    gt_u = (gt > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    pred_d = cv2.dilate(pred_u, kernel, iterations=int(dilation_px))
    gt_d = cv2.dilate(gt_u, kernel, iterations=int(dilation_px))
    inter = np.logical_and(pred_d > 0, gt_d > 0).sum()
    union = np.logical_or(pred_d > 0, gt_d > 0).sum()
    return float(inter) / float(union) if union > 0 else 1.0


@torch.no_grad()
def evaluate(model: nn.Module, dataset: BoundaryFeatureDataset, device: torch.device, threshold: float) -> dict[str, float]:
    model.eval()
    bf1_vals = []
    bf2_vals = []
    biou_vals = []
    for idx in range(len(dataset)):
        feat, target = dataset[idx]
        logits = model(feat.unsqueeze(0).to(device))
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        pred = (prob >= float(threshold)).astype(np.uint8)
        gt = target[0].numpy().astype(np.uint8)
        bf1_vals.append(boundary_fscore(pred, gt, tolerance=1))
        bf2_vals.append(boundary_fscore(pred, gt, tolerance=2))
        biou_vals.append(boundary_iou(pred, gt, dilation_px=2))
    return {
        "bf1_1px": float(np.mean(bf1_vals)) if bf1_vals else 0.0,
        "bf1_2px": float(np.mean(bf2_vals)) if bf2_vals else 0.0,
        "biou": float(np.mean(biou_vals)) if biou_vals else 0.0,
    }


def train_one_epoch(
    model: nn.Module,
    dataset: BoundaryFeatureDataset,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    seed: int,
) -> float:
    model.train()
    order = np.arange(len(dataset))
    rng = np.random.default_rng(int(seed) + int(epoch))
    rng.shuffle(order)
    losses = []
    for idx in order.tolist():
        feat, target = dataset[idx]
        feat = feat.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        logits = model(feat)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        loss = bce + dice_loss_from_logits(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def canonical_image_128(image_path: Path, meta: dict[str, Any], target_size: int) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    input_h, input_w = [int(x) for x in np.asarray(meta["input_hw"]).reshape(-1)[:2]]
    input_h = max(input_h, 1)
    input_w = max(input_w, 1)
    trans = meta.get("trans_input")
    if trans is not None:
        canvas = cv2.warpAffine(img, trans, (input_w, input_h), flags=cv2.INTER_LINEAR)
    else:
        resized_h, resized_w = [int(x) for x in np.asarray(meta["resized_hw"]).reshape(-1)[:2]]
        resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((input_h, input_w, 3), 128, dtype=np.uint8)
        canvas[:resized_h, :resized_w] = resized
    return cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def save_debug_visuals(dataset: BoundaryFeatureDataset, out_dir: Path, count: int = 3) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(min(count, len(dataset))):
        _feat, boundary, meta, polys_target = dataset.load_npz_sample(idx)
        item = dataset.items[idx]
        vis = canonical_image_128(item["image_path"], meta, dataset.target_size)
        overlay = vis.copy()
        overlay[boundary > 0] = (0, 0, 255)
        vis = cv2.addWeighted(overlay, 0.65, vis, 0.35, 0)
        for poly in polys_target:
            cv2.polylines(vis, [np.round(poly).astype(np.int32)], True, (0, 255, 255), 1, lineType=cv2.LINE_AA)
        out_path = out_dir / f"{dataset.split}_{idx:02d}_{item['image_path'].stem}_gt_boundary.png"
        cv2.imwrite(str(out_path), vis)
    print(f"[*] Debug GT overlays saved to {out_dir}", flush=True)


def write_results(result: dict[str, Any], out_json: Path, summary_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    if summary_json.exists():
        try:
            existing = json.loads(summary_json.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    else:
        existing = []
    if isinstance(existing, dict) and isinstance(existing.get("runs"), list):
        existing["runs"].append(result)
        payload = existing
    elif isinstance(existing, list):
        existing.append(result)
        payload = existing
    else:
        payload = [result]
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_root = Path(args.data_root).resolve()
    jsonl_root = Path(args.jsonl_root).resolve()
    train_feat_dir = resolve_split_feat_dir(args, "train")
    test_feat_dir = resolve_split_feat_dir(args, "test")

    train_records = select_records(load_jsonl_records(jsonl_root, "train"), "train", args.train_subset_size)
    test_records = select_records(load_jsonl_records(jsonl_root, "test"), "test", args.train_subset_size)
    train_set = BoundaryFeatureDataset(train_records, "train", data_root, train_feat_dir, args.feat_key, args.target_size)
    test_set = BoundaryFeatureDataset(test_records, "test", data_root, test_feat_dir, args.feat_key, args.target_size)

    first_feat, _first_target = train_set[0]
    in_channels = int(first_feat.shape[0])
    feat_hw = (int(first_feat.shape[1]), int(first_feat.shape[2]))
    model = BoundaryProbe(in_channels, feat_hw, target_size=args.target_size).to(device)
    param_count = probe_param_count(model)
    print(
        f"[*] Probe: feat_key={args.feat_key} in_channels={in_channels} feat_hw={feat_hw} "
        f"params={param_count} ({param_count / 1e6:.3f}M)",
        flush=True,
    )
    if param_count >= 500_000:
        raise RuntimeError(f"Probe parameter count must stay <0.5M, got {param_count}")

    if args.debug_vis:
        save_debug_visuals(test_set, Path(args.debug_vis).resolve(), count=3)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_score = -1.0
    best_epoch = 0
    best_metrics = {"bf1_1px": 0.0, "bf1_2px": 0.0, "biou": 0.0}
    best_state = None
    bad_epochs = 0
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_one_epoch(model, train_set, device, optimizer, epoch, args.seed)
        metrics = evaluate(model, test_set, device, threshold=args.threshold)
        print(
            f"[epoch {epoch:02d}] loss={train_loss:.4f} "
            f"bf1_1px={metrics['bf1_1px']:.4f} bf1_2px={metrics['bf1_2px']:.4f} biou={metrics['biou']:.4f}",
            flush=True,
        )
        score = metrics["bf1_2px"]
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                print(f"[*] Early stop at epoch={epoch} patience={args.patience}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    result = {
        "config": {
            "feat_key": str(args.feat_key),
            "feat_dir": str(Path(args.feat_dir).resolve()) if args.feat_dir else "",
            "train_feat_dir": str(train_feat_dir),
            "test_feat_dir": str(test_feat_dir),
            "data_root": str(data_root),
            "jsonl_root": str(jsonl_root),
            "train_subset_size": int(args.train_subset_size),
            "train_samples": int(len(train_set)),
            "test_samples": int(len(test_set)),
            "target_size": int(args.target_size),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "threshold": float(args.threshold),
            "seed": int(args.seed),
            "device": str(device),
            "in_channels": int(in_channels),
            "feat_hw_first": [int(feat_hw[0]), int(feat_hw[1])],
            "probe_params": int(param_count),
        },
        "bf1_1px": float(best_metrics["bf1_1px"]),
        "bf1_2px": float(best_metrics["bf1_2px"]),
        "biou": float(best_metrics["biou"]),
        "best_epoch": int(best_epoch),
    }
    write_results(result, Path(args.out_json).resolve(), Path(args.summary_json).resolve())
    print(f"[*] Wrote result: {Path(args.out_json).resolve()}", flush=True)
    print(f"[*] Appended summary: {Path(args.summary_json).resolve()}", flush=True)
    return result


def make_smoke_case() -> argparse.Namespace:
    root = Path(tempfile.mkdtemp(prefix="f1_boundary_probe_smoke_", dir="/tmp"))
    data_root = root / "data"
    jsonl_root = root / "jsonl"
    feat_root = root / "features"
    rng = np.random.default_rng(20260613)
    for split, count in (("train", 4), ("test", 3)):
        (data_root / split).mkdir(parents=True, exist_ok=True)
        (feat_root / split).mkdir(parents=True, exist_ok=True)
        records = []
        for idx in range(count):
            image_rel = f"{split}/{idx}_image.png"
            image_path = data_root / image_rel
            img = np.full((128, 128, 3), 40, dtype=np.uint8)
            offset = 12 + idx * 3
            cv2.rectangle(img, (offset, 20), (104, 96), (90, 120, 180), -1)
            cv2.imwrite(str(image_path), img)
            poly = [[offset, 20], [104, 20], [104, 96], [offset, 96]]
            records.append({"image": image_rel, "polygons": [poly]})
            feat = rng.normal(size=(8, 32, 32)).astype(np.float16)
            np.savez(
                feat_root / split / f"{idx}_image.npz",
                feat=feat,
                grid_hw=np.asarray([32, 32], dtype=np.int32),
                orig_hw=np.asarray([128, 128], dtype=np.int32),
                resized_hw=np.asarray([128, 128], dtype=np.int32),
                padded_hw=np.asarray([128, 128], dtype=np.int32),
                input_hw=np.asarray([128, 128], dtype=np.int32),
                pad=np.asarray([0, 0, 0, 0], dtype=np.int32),
                scale=np.asarray([1.0], dtype=np.float32),
                image_path=np.asarray(str(image_path)),
            )
        jsonl_root.mkdir(parents=True, exist_ok=True)
        with (jsonl_root / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    smoke_args = copy.copy(ARGS)
    smoke_args.feat_dir = str(feat_root)
    smoke_args.train_feat_dir = ""
    smoke_args.test_feat_dir = ""
    smoke_args.feat_key = "feat"
    smoke_args.data_root = str(data_root)
    smoke_args.jsonl_root = str(jsonl_root)
    smoke_args.train_subset_size = 4
    smoke_args.target_size = 128
    smoke_args.epochs = 2
    smoke_args.patience = 2
    smoke_args.device = "cpu"
    smoke_args.out_json = str(root / "result.json")
    smoke_args.summary_json = str(root / "summary.json")
    if ARGS.debug_vis:
        smoke_args.debug_vis = str(root / "debug_vis")
    print(f"[*] Smoke workspace: {root}", flush=True)
    return smoke_args


def main() -> None:
    if ARGS.smoke:
        smoke_args = make_smoke_case()
        result = run_experiment(smoke_args)
        print(
            f"[*] Smoke OK: best_epoch={result['best_epoch']} "
            f"bf1_1px={result['bf1_1px']:.4f} bf1_2px={result['bf1_2px']:.4f} biou={result['biou']:.4f}",
            flush=True,
        )
        return
    run_experiment(ARGS)


if __name__ == "__main__":
    main()
