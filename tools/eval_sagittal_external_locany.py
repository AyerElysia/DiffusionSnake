#!/usr/bin/env python3
"""Evaluate sagittal DiffusionSnake from external LocateAnything detections."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import traceback
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _cfg_file_from_argv() -> str:
    for index, value in enumerate(sys.argv[1:]):
        if value == "--cfg_file" and index + 2 <= len(sys.argv[1:]):
            return sys.argv[index + 2]
        if value.startswith("--cfg_file="):
            return value.split("=", 1)[1]
    return os.environ.get("CFG_FILE", str(_ROOT / "configs" / "sagittal_2d_v4_6c_moonvit_train.yaml"))


_ORIGINAL_ARGV = sys.argv[:]
_ORIGINAL_CFG = os.environ.get("CFG_FILE")
_CFG_FILE = _cfg_file_from_argv()
os.environ["CFG_FILE"] = _CFG_FILE
try:
    sys.argv = [sys.argv[0]]
    from lib.config import cfg
    from lib.datasets.make_dataset import make_data_loader
    from lib.evaluators.sagittal_2d_fixed.snake import (
        binary_iou_dice,
        inverse_affine_points,
        rasterize_polygons,
    )
    from lib.networks import make_network
    from lib.utils.snake import snake_config
finally:
    sys.argv = _ORIGINAL_ARGV
    if _ORIGINAL_CFG is None:
        os.environ.pop("CFG_FILE", None)
    else:
        os.environ["CFG_FILE"] = _ORIGINAL_CFG


CLASS_ID_TO_NAME = {
    **{i: f"C{i} vertebra" for i in range(1, 8)},
    **{i + 7: f"T{i} vertebra" for i in range(1, 13)},
    **{i + 19: f"L{i} vertebra" for i in range(1, 7)},
}
BOUND_TH = 0.008
_DILL_FALLBACK = Path("/home/medteam/miniconda3/envs/sam1_lgz/lib/python3.11/site-packages/dill")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg_file", default=_CFG_FILE)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--bbox-manifest", type=Path, default=Path("/home/medteam/Zhrch/detect_3D_lgz2/datasets/sagittal_2d_fixed/manifests/bbox_2d_component_class_manifest.csv"))
    parser.add_argument("--max-slices", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--max-overlays", type=int, default=40)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def move_tensors(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if isinstance(key, str) and (key == "locate_feat" or key.startswith("locate_feat_")):
                out[key] = item
            else:
                out[key] = move_tensors(item, device)
        return out
    if isinstance(value, list):
        return [move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors(item, device) for item in value)
    return value


def strip_prefix(key: str) -> str:
    clean = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "net."):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                changed = True
                break
    return clean


def load_checkpoint(network: torch.nn.Module, checkpoint_path: str) -> dict[str, Any]:
    if "dill" not in sys.modules and _DILL_FALLBACK.exists():
        spec = importlib.util.spec_from_file_location(
            "dill",
            str(_DILL_FALLBACK / "__init__.py"),
            submodule_search_locations=[str(_DILL_FALLBACK)],
        )
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules["dill"] = module
            spec.loader.exec_module(module)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = OrderedDict((strip_prefix(k), v) for k, v in source.items())
    target_keys = set(network.state_dict())
    missing = sorted(target_keys - set(state_dict))
    unexpected = sorted(set(state_dict) - target_keys)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint/model mismatch missing={missing} unexpected={unexpected}")
    network.load_state_dict(state_dict, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {}


def path_keys(path: str | Path) -> set[str]:
    p = Path(path)
    keys = {str(p), str(p.resolve())}
    parts = p.parts
    for marker in ("training", "validation", "test"):
        if marker in parts:
            idx = parts.index(marker)
            keys.add("/".join(parts[idx:]))
    return keys


def load_detections(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = row.get("absolute_image_path") or row.get("image_path")
            if not image_path:
                raise ValueError(f"detection line {line_number} lacks image_path")
            for key in path_keys(image_path):
                records[key] = row
    return records


def load_gt_boxes(path: Path, split: str) -> dict[tuple[str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            item = {
                "label_id": int(float(row["label_id"])),
                "class_name": row["class_name"],
                "bbox": [
                    float(row["x_min"]),
                    float(row["y_min"]),
                    float(row["x_max"]),
                    float(row["y_max"]),
                ],
            }
            out[(row["case_id"], int(float(row["slice_idx"])))].append(item)
    return out


def as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def select_meta(batch: dict[str, Any], key: str, index: int = 0) -> Any:
    value = batch.get("meta", {}).get(key)
    arr = as_numpy(value)
    if key == "trans_input" and arr.ndim == 3:
        return arr[index]
    if key == "inv_trans_input" and arr.ndim == 3:
        return arr[index]
    if key == "orig_hw" and arr.ndim >= 2:
        return arr[index]
    if key == "flipped" and arr.size > 1:
        return arr.reshape(-1)[index]
    if key in ("case_id",):
        return value[index] if isinstance(value, (list, tuple)) else value
    if arr.ndim >= 1 and arr.shape[0] > index and arr.size != 2:
        return arr[index]
    return value


def select_path(batch: dict[str, Any], index: int = 0) -> str:
    value = batch.get("img_path")
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return str(value[index])
    arr = as_numpy(value)
    return str(arr[index])


def transform_box_xyxy(box: list[float], trans_input: np.ndarray, orig_hw: np.ndarray, flipped: Any) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    height, width = [int(v) for v in np.asarray(orig_hw).reshape(-1)[:2]]
    points = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    if bool(np.asarray(flipped).reshape(-1)[0]) if np.asarray(flipped).size else False:
        points[:, 0] = float(width) - points[:, 0] - 1.0
    hom = np.concatenate([points, np.ones((4, 1), dtype=np.float32)], axis=1)
    restored = hom @ np.asarray(trans_input, dtype=np.float32).T
    min_xy = restored.min(axis=0)
    max_xy = restored.max(axis=0)
    return [float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1])]


def make_external_detection(batch: dict[str, Any], det_row: dict[str, Any] | None) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    inp = batch["inp"]
    bsz = int(inp.shape[0]) if torch.is_tensor(inp) and inp.ndim >= 4 else 1
    per_sample: list[list[list[float]]] = []
    used: list[dict[str, Any]] = []
    for b in range(bsz):
        image_path = select_path(batch, b)
        row = det_row if bsz == 1 else None
        if row is None:
            row = {}
        trans_input = select_meta(batch, "trans_input", b)
        orig_hw = select_meta(batch, "orig_hw", b)
        flipped = select_meta(batch, "flipped", b)
        detections = []
        for pred in row.get("predictions", []):
            label_id = int(pred["label_id"])
            box_in = transform_box_xyxy(pred["bbox_original_xyxy"], trans_input, orig_hw, flipped)
            x1, y1, x2, y2 = box_in
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append([x1, y1, x2, y2, 1.0, float(label_id - 1)])
            used.append({**pred, "image_path": image_path, "bbox_snake_input_xyxy": [x1, y1, x2, y2]})
        per_sample.append(detections)
    max_len = max((len(items) for items in per_sample), default=0)
    tensor = torch.zeros((bsz, max_len, 6), dtype=torch.float32)
    for b, items in enumerate(per_sample):
        if items:
            tensor[b, : len(items)] = torch.tensor(items, dtype=torch.float32)
    return tensor, used


def final_contours(output: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    py = output.get("py")
    if isinstance(py, (list, tuple)):
        py = py[-1] if py else None
    if py is None:
        return np.zeros((0, int(snake_config.poly_num), 2), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    contours = as_numpy(py).astype(np.float32, copy=False)
    if contours.ndim == 2:
        contours = contours[None]
    detection = as_numpy(output["detection"])
    valid = detection[..., 4] > 1e-4
    labels = np.rint(detection[..., 5][valid]).astype(np.int64)
    scores = detection[..., 4][valid].astype(np.float32)
    if labels.size != contours.shape[0]:
        raise RuntimeError(f"contour/detection mismatch contours={contours.shape[0]} labels={labels.size}")
    return contours, labels, scores


def binary_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(boundary, contours, -1, 1, 1)
    return boundary.astype(bool)


def boundary_f(pred: np.ndarray, gt: np.ndarray, bound_th: float = BOUND_TH) -> float:
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    pred_b = binary_boundary(pred)
    gt_b = binary_boundary(gt)
    if not pred_b.any() and not gt_b.any():
        return 1.0
    if not pred_b.any() or not gt_b.any():
        return 0.0
    h, w = pred.shape
    tolerance = max(1, int(np.ceil(bound_th * np.sqrt(float(h * h + w * w)))))
    gt_dist = cv2.distanceTransform((~gt_b).astype(np.uint8), cv2.DIST_L2, 3)
    pred_dist = cv2.distanceTransform((~pred_b).astype(np.uint8), cv2.DIST_L2, 3)
    precision = float(np.mean(gt_dist[pred_b] <= tolerance))
    recall = float(np.mean(pred_dist[gt_b] <= tolerance))
    return 0.0 if precision + recall == 0 else float(2.0 * precision * recall / (precision + recall))


def read_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint16, copy=False)


def match_detection_stats(preds: list[dict[str, Any]], gt_items: list[dict[str, Any]], iou_thr: float = 0.10) -> tuple[int, int]:
    matched_gt = set()
    matched_pred = set()
    candidates = []
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gt_items):
            if int(pred.get("label_id", -1)) != int(gt["label_id"]):
                continue
            iou, _ = binary_iou_dice_box(pred["bbox_original_xyxy"], gt["bbox"])
            candidates.append((iou, pi, gi))
    for iou, pi, gi in sorted(candidates, reverse=True):
        if iou < iou_thr or pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
    return len(gt_items) - len(matched_gt), len(preds) - len(matched_pred)


def binary_iou_dice_box(a: list[float], b: list[float]) -> tuple[float, float]:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    iou = inter / union if union > 0 else 1.0
    dice = 2 * inter / (aa + bb) if aa + bb > 0 else 1.0
    return float(iou), float(dice)


def overlay_image(image_path: str, gt_mask: np.ndarray, pred_mask: np.ndarray, boxes: list[dict[str, Any]], contours: list[np.ndarray], out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    base = np.asarray(image).copy()
    if gt_mask.any():
        base[gt_mask > 0] = (0.55 * base[gt_mask > 0] + 0.45 * np.array([0, 190, 0])).astype(np.uint8)
    if pred_mask.any():
        base[pred_mask > 0] = (0.55 * base[pred_mask > 0] + 0.45 * np.array([230, 50, 40])).astype(np.uint8)
    canvas = Image.fromarray(base)
    draw = ImageDraw.Draw(canvas)
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box["bbox_original_xyxy"]]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 220, 0), width=1)
        draw.text((x1, max(0, y1 - 10)), str(box.get("class_name", ""))[:4], fill=(255, 220, 0))
    for contour in contours:
        pts = [(float(x), float(y)) for x, y in np.asarray(contour).reshape(-1, 2)]
        if len(pts) > 2:
            draw.line(pts + [pts[0]], fill=(0, 170, 255), width=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    for sub in ("masks", "contours", "overlays", "metrics", "logs"):
        (result_dir / sub).mkdir(parents=True, exist_ok=True)

    dataset_name = "SagittalPseudo3DVal" if args.split == "validation" else "SagittalPseudo3DTest"
    cfg.test.dataset = dataset_name
    cfg.test.batch_size = 1
    cfg.train.num_workers = 0
    cfg.result_dir = str(result_dir)
    cfg.use_gt_det = False
    cfg.use_gt_det_train_only = False
    cfg.sagittal_eval_box_mode = "external_locany"

    detections = load_detections(args.detections)
    gt_boxes = load_gt_boxes(args.bbox_manifest, args.split)
    device = resolve_device(args.device)
    loader = make_data_loader(cfg, is_train=False)
    network = make_network(cfg).to(device)
    checkpoint = load_checkpoint(network, args.ckpt)
    network.eval()

    run_info = {
        "cfg_file": str(Path(args.cfg_file).resolve()),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "checkpoint_run_id": checkpoint.get("run_id"),
        "checkpoint_lineage_step": checkpoint.get("lineage_step"),
        "split": args.split,
        "device": str(device),
        "snake_poly_num": int(snake_config.poly_num),
        "snake_init_poly_num": int(snake_config.init_poly_num),
        "boundary_f": {
            "implementation": "fallback DAVIS/Perazzi-style distance-transform boundary F",
            "bound_th": BOUND_TH,
            "pixel_tolerance": "ceil(bound_th * sqrt(H^2 + W^2)), minimum 1",
        },
        "gt_leakage_check": "external_detection is built only from LocateAnything predicted bbox/class; GT masks/boxes are used only for metrics and detection-error audit",
    }
    (result_dir / "protocol_snake.json").write_text(json.dumps(run_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "snake_ready", **run_info}, sort_keys=True), flush=True)

    per_image: list[dict[str, Any]] = []
    per_class_values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    failure_rows: list[dict[str, Any]] = []
    class_slice_iou: list[float] = []
    class_slice_dice: list[float] = []
    class_slice_bf: list[float] = []
    foreground_iou: list[float] = []
    foreground_dice: list[float] = []
    foreground_bf: list[float] = []
    stats: Counter = Counter()
    overlays_written = 0

    with torch.no_grad():
        for index, batch in enumerate(loader):
            if args.max_slices is not None and index >= args.max_slices:
                break
            try:
                image_path = select_path(batch, 0)
                det_row = None
                for key in path_keys(image_path):
                    if key in detections:
                        det_row = detections[key]
                        break
                if det_row is None:
                    det_row = {"predictions": [], "raw_output": "", "image_path": image_path}
                    stats["missing_detection_rows"] += 1
                external_detection, used_boxes = make_external_detection(batch, det_row)
                batch["external_detection"] = external_detection
                batch["external_detection_source"] = "LocateAnything-3B checkpoint-20000"
                moved = move_tensors(batch, device)
                output = network(moved["inp"].to(device), moved)

                contours_input, labels0, scores = final_contours(output)
                inv_trans = np.asarray(select_meta(batch, "inv_trans_input", 0), dtype=np.float32)
                orig_hw = np.asarray(select_meta(batch, "orig_hw", 0), dtype=np.float32)
                flipped = select_meta(batch, "flipped", 0)
                contours_orig = []
                labels = []
                for contour, label0 in zip(contours_input, labels0):
                    restored = inverse_affine_points(
                        contour * float(snake_config.down_ratio),
                        inv_trans,
                        orig_hw,
                        flipped=flipped,
                    )
                    contours_orig.append(restored)
                    labels.append(int(label0) + 1)
                gt_mask = read_mask(batch["mask_path"][0] if isinstance(batch.get("mask_path"), (list, tuple)) else loader.dataset.records[index]["mask_path"])
                pred_mask = rasterize_polygons(contours_orig, gt_mask.shape, labels)

                fg_iou, fg_dice = binary_iou_dice(pred_mask > 0, gt_mask > 0)
                fg_bf = boundary_f(pred_mask > 0, gt_mask > 0)
                foreground_iou.append(fg_iou)
                foreground_dice.append(fg_dice)
                foreground_bf.append(fg_bf)

                class_metrics = {}
                class_union = sorted(set(int(x) for x in np.unique(gt_mask) if int(x) > 0) | set(int(x) for x in np.unique(pred_mask) if int(x) > 0))
                for class_id in class_union:
                    pred_c = pred_mask == class_id
                    gt_c = gt_mask == class_id
                    iou, dice = binary_iou_dice(pred_c, gt_c)
                    bf = boundary_f(pred_c, gt_c)
                    class_slice_iou.append(iou)
                    class_slice_dice.append(dice)
                    class_slice_bf.append(bf)
                    per_class_values[class_id]["iou"].append(iou)
                    per_class_values[class_id]["dice"].append(dice)
                    per_class_values[class_id]["bf"].append(bf)
                    class_metrics[str(class_id)] = {"iou": iou, "dice": dice, "bf": bf}

                case_id = str(select_meta(batch, "case_id", 0))
                slice_idx = int(np.asarray(select_meta(batch, "slice_idx", 0)).reshape(-1)[0])
                missed, false_pos = match_detection_stats(
                    det_row.get("predictions", []),
                    gt_boxes.get((case_id, slice_idx), []),
                )
                stats["gt_instances"] += len(gt_boxes.get((case_id, slice_idx), []))
                stats["detector_predictions"] += len(det_row.get("predictions", []))
                stats["missed_detections_iou010"] += missed
                stats["false_positive_detections_iou010"] += false_pos
                if not det_row.get("predictions"):
                    stats["no_prediction_images"] += 1
                if len(contours_orig) != len(det_row.get("predictions", [])):
                    stats["snake_count_mismatch"] += 1

                mask_rel = f"{case_id}/{Path(image_path).stem}_pred_mask.png"
                if args.save_masks:
                    mask_path = result_dir / "masks" / mask_rel
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(mask_path), pred_mask.astype(np.uint16))
                contour_record = {
                    "case_id": case_id,
                    "slice_idx": slice_idx,
                    "image_path": image_path,
                    "detections_used": used_boxes,
                    "labels": labels,
                    "scores": [float(x) for x in scores.tolist()],
                    "contours_original": [np.asarray(c).round(4).tolist() for c in contours_orig],
                }
                with (result_dir / "contours" / f"{args.split}_contours.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(contour_record, ensure_ascii=False, sort_keys=True) + "\n")
                if args.save_overlays and overlays_written < args.max_overlays:
                    overlay_image(
                        image_path,
                        gt_mask,
                        pred_mask,
                        det_row.get("predictions", []),
                        contours_orig,
                        result_dir / "overlays" / case_id / f"{Path(image_path).stem}_overlay.png",
                    )
                    overlays_written += 1
                row = {
                    "case_id": case_id,
                    "slice_idx": slice_idx,
                    "image_path": image_path,
                    "n_gt_instances": len(gt_boxes.get((case_id, slice_idx), [])),
                    "n_detector_predictions": len(det_row.get("predictions", [])),
                    "n_snake_contours": len(contours_orig),
                    "missed_detections_iou010": missed,
                    "false_positive_detections_iou010": false_pos,
                    "foreground_iou": fg_iou,
                    "foreground_dice": fg_dice,
                    "foreground_bf": fg_bf,
                    "class_metrics": class_metrics,
                }
                per_image.append(row)
                stats["processed_slices"] += 1
                if (index + 1) % 50 == 0:
                    print(json.dumps({"event": "progress", "processed": index + 1}, sort_keys=True), flush=True)
            except Exception as exc:  # noqa: BLE001 - continue and record sample-level failures.
                failure_rows.append(
                    {
                        "index": index,
                        "image_path": select_path(batch, 0) if isinstance(batch, dict) else "",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                stats["snake_failures"] += 1

    summary = {
        "split": args.split,
        "num_slices": int(stats["processed_slices"]),
        "failures": int(stats["snake_failures"]),
        "gt_instances": int(stats["gt_instances"]),
        "detector_predictions": int(stats["detector_predictions"]),
        "missed_detections_iou010": int(stats["missed_detections_iou010"]),
        "false_positive_detections_iou010": int(stats["false_positive_detections_iou010"]),
        "no_prediction_images": int(stats["no_prediction_images"]),
        "primary_class_slice_mean": {
            "mIoU": mean(class_slice_iou),
            "mDC": mean(class_slice_dice),
            "mDice": mean(class_slice_dice),
            "mBF": mean(class_slice_bf),
            "aggregation": "mean over non-background class masks present in GT or prediction per slice",
        },
        "foreground_slice_mean": {
            "mIoU": mean(foreground_iou),
            "mDC": mean(foreground_dice),
            "mDice": mean(foreground_dice),
            "mBF": mean(foreground_bf),
            "aggregation": "mean over binary foreground masks for every processed slice",
        },
        "boundary_f": {
            "bound_th": BOUND_TH,
            "pixel_tolerance": "ceil(bound_th * sqrt(H^2 + W^2)), minimum 1",
        },
        "extra_counts": dict(stats),
    }
    metrics_dir = result_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (metrics_dir / f"per_image_metrics_{args.split}.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "case_id",
            "slice_idx",
            "image_path",
            "n_gt_instances",
            "n_detector_predictions",
            "n_snake_contours",
            "missed_detections_iou010",
            "false_positive_detections_iou010",
            "foreground_iou",
            "foreground_dice",
            "foreground_bf",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_image:
            writer.writerow({key: row.get(key) for key in fieldnames})
    with (metrics_dir / f"per_class_metrics_{args.split}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label_id", "class_name", "mean_iou", "mean_dice", "mean_bf", "n_class_slices"])
        writer.writeheader()
        for class_id in range(1, 26):
            values = per_class_values.get(class_id, {})
            writer.writerow(
                {
                    "label_id": class_id,
                    "class_name": CLASS_ID_TO_NAME[class_id],
                    "mean_iou": mean(values.get("iou", [])),
                    "mean_dice": mean(values.get("dice", [])),
                    "mean_bf": mean(values.get("bf", [])),
                    "n_class_slices": len(values.get("iou", [])),
                }
            )
    with (metrics_dir / "failure_cases.csv").open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "index", "image_path", "error", "traceback"])
        if handle.tell() == 0:
            writer.writeheader()
        for row in failure_rows:
            writer.writerow({"split": args.split, **row})
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
