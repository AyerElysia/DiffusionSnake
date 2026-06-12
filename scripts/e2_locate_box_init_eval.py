#!/usr/bin/env python3
"""
E2 LocateAnything box-initialized V8.2 full-chain evaluation.

Run with snake1:
  CUDA_VISIBLE_DEVICES=2 /home/medteam/miniconda3/envs/snake1/bin/python scripts/e2_locate_box_init_eval.py --det-source locate
  CUDA_VISIBLE_DEVICES=2 /home/medteam/miniconda3/envs/snake1/bin/python scripts/e2_locate_box_init_eval.py --det-source v8_2
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO_ROOT / "configs" / "1232_final_v8_2_mged_final_gpu6.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-file", default=str(DEFAULT_CFG))
    parser.add_argument("--det-source", choices=("locate", "v8_2"), required=True)
    parser.add_argument(
        "--locate-json",
        default=str(REPO_ROOT / "data" / "eagle_teacher" / "1232_final_test_locateanything_ckpt3000.json"),
    )
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--limit", type=int, default=0, help="0 means full test split")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visible-gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "2"))
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--ode-steps", type=int, default=0)
    parser.add_argument("--det-conf-thresh", type=float, default=None)
    parser.add_argument("--det-iou-thresh", type=float, default=None)
    parser.add_argument("--det-max-det", type=int, default=None)
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


ARGS = parse_args()

# Project convention: CFG_FILE must be set before importing lib.* modules.
os.environ["CFG_FILE"] = str(Path(ARGS.cfg_file).resolve())
sys.argv = [sys.argv[0], "--cfg_file", os.environ["CFG_FILE"]]
if ARGS.visible_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.visible_gpu)

sys.path.insert(0, str(REPO_ROOT))

from lib.config import cfg  # noqa: E402
from lib.datasets.collate_batch import make_collator  # noqa: E402
from lib.datasets.make_dataset import make_dataset  # noqa: E402
from lib.datasets.transforms import make_transforms  # noqa: E402
from lib.networks import make_network  # noqa: E402
from lib.utils import data_utils  # noqa: E402
from lib.utils.snake import snake_config  # noqa: E402


def apply_eval_overrides() -> None:
    cfg.gpus = [0]
    cfg.train_or_test = "test"
    cfg.use_gt_det = False
    cfg.skip_diffusion_forward = False
    cfg.detector_only_warmup = False
    cfg.use_extreme_refine = True
    cfg.use_pred_extreme_init_for_inference = True
    cfg.contour_init_method = "octagon"
    if ARGS.det_conf_thresh is not None:
        cfg.det_conf_thresh = float(ARGS.det_conf_thresh)
    if ARGS.det_iou_thresh is not None:
        cfg.det_iou_thresh = float(ARGS.det_iou_thresh)
    if ARGS.det_max_det is not None:
        cfg.det_max_det = int(ARGS.det_max_det)


def set_eval_seed(seed: int) -> None:
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def resolve_checkpoint(path_arg: str) -> Path:
    if path_arg:
        return Path(path_arg).resolve()
    ckpt_dir = (REPO_ROOT / str(cfg.model_dir) / "checkpoints").resolve()
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return latest
    candidates = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")
    return candidates[-1]


def load_model(ckpt_path: Path) -> torch.nn.Module:
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
        f"[*] Loaded checkpoint {ckpt_path} | "
        f"reused={len(reusable)}/{len(target)} missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}"
    )
    return network


def load_locate_cache(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    records = obj.get("samples") if isinstance(obj, dict) else obj
    if not isinstance(records, list):
        raise ValueError(f"Locate cache must contain a samples list: {path}")

    by_key: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        instances = rec.get("instances") or []
        keys = set()
        for name in ("img_path", "image_path", "path", "file_name", "image", "image_rel"):
            val = rec.get(name)
            if val:
                p = Path(str(val))
                keys.add(str(val))
                keys.add(str(p))
                keys.add(p.name)
                keys.add(p.stem)
                if p.is_absolute():
                    keys.add(str(p.resolve()))
        for key in keys:
            by_key[key] = instances
    print(f"[*] Loaded Locate cache records={len(records)} keys={len(by_key)} from {path}")
    return by_key


def _path_keys(path: str | Path) -> list[str]:
    p = Path(str(path))
    keys = [str(path), str(p), p.name, p.stem]
    if p.is_absolute():
        keys.append(str(p.resolve()))
    return keys


def locate_instances_for_batch(batch: dict[str, Any], cache: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    img_path = batch.get("img_path")
    if isinstance(img_path, (list, tuple)):
        img_path = img_path[0]
    if torch.is_tensor(img_path):
        img_path = str(img_path.item())
    for key in _path_keys(str(img_path)):
        if key in cache:
            return cache[key]
    raise KeyError(f"No Locate cache entry for image path: {img_path}")


def clip_box_np(box: np.ndarray, width: int, height: int) -> np.ndarray | None:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
    y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def get_original_to_input_transform(img_path: str | Path, input_hw: tuple[int, int]) -> tuple[np.ndarray, int, int]:
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image for coordinate transform: {img_path}")
    height, width = img.shape[:2]
    center = np.asarray([width / 2.0, height / 2.0], dtype=np.float32)
    scale = np.asarray([max(width, height), max(width, height)], dtype=np.float32)
    input_h, input_w = input_hw
    trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
    return trans_input, input_w, input_h


def original_box_to_input_box(
    box: list[float],
    trans_input: np.ndarray,
    input_w: int,
    input_h: int,
) -> np.ndarray | None:
    pts = np.asarray([[box[0], box[1]], [box[2], box[3]]], dtype=np.float32)
    pts = data_utils.affine_transform(pts, trans_input)
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    return clip_box_np(np.asarray([x1, y1, x2, y2], dtype=np.float32), input_w, input_h)


def build_locate_detection(
    batch: dict[str, Any],
    locate_cache: dict[str, list[dict[str, Any]]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    instances = locate_instances_for_batch(batch, locate_cache)
    inp = batch["inp"]
    input_h, input_w = int(inp.shape[-2]), int(inp.shape[-1])
    img_path = batch.get("img_path")
    if isinstance(img_path, (list, tuple)):
        img_path = img_path[0]
    trans_input, input_w, input_h = get_original_to_input_transform(str(img_path), (input_h, input_w))
    rows = []
    for inst in instances:
        if not isinstance(inst, dict) or inst.get("bbox") is None:
            continue
        box = original_box_to_input_box(inst["bbox"], trans_input, input_w, input_h)
        if box is None:
            continue
        label = inst.get("label_id", inst.get("cls_id", inst.get("category_id", inst.get("label", 0))))
        try:
            label_i = int(label)
        except Exception:
            label_i = 0
        score = float(inst.get("score", inst.get("confidence", 1.0)))
        rows.append([float(box[0]), float(box[1]), float(box[2]), float(box[3]), score, float(label_i)])
    if rows:
        det = torch.as_tensor(rows, device=device, dtype=dtype).unsqueeze(0)
    else:
        det = torch.zeros((1, 0, 6), device=device, dtype=dtype)
    return det, len(rows)


def compute_detector_side_output(
    core: torch.nn.Module,
    x: torch.Tensor,
    batch: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], torch.Tensor]:
    backend = str(getattr(core, "detector_backend", "yolo") or "yolo").strip().lower()
    if backend.startswith("heatmap_") or backend.startswith("convnext") or backend.startswith("moonvit"):
        cnn_feature, ct_hm, wh, mask_logits = core.heatmap_detector(x)
        if mask_logits is not None:
            alpha = float(getattr(cfg, "heatmap_mask_guidance_alpha", 0.0))
            if alpha > 0.0:
                guidance = torch.sigmoid(mask_logits).amax(dim=1, keepdim=True)
                cnn_feature = cnn_feature * (1.0 + alpha * guidance)
        det_cnn_feature = cnn_feature
        locate_feat_stats = {}
        if hasattr(core, "apply_locate_feature_injection"):
            cnn_feature, locate_feat_stats = core.apply_locate_feature_injection(det_cnn_feature, batch)
        if hasattr(core, "apply_locate_feature_replacement"):
            cnn_feature, replace_stats = core.apply_locate_feature_replacement(cnn_feature, batch)
            locate_feat_stats.update(replace_stats)
        h, w = det_cnn_feature.size(2), det_cnn_feature.size(3)
        ct, raw_det = core.decode_detection_from_heatmap(ct_hm, wh)
        detection = core.filter_detection_candidates(raw_det)
        output = {
            "ct_hm": ct_hm,
            "wh": wh,
            "ct": ct,
            "detection": detection,
            "feat_hw": (h, w),
            "cnn_feature": cnn_feature,
        }
        output.update(locate_feat_stats)
        if mask_logits is not None:
            output["mask_logits"] = mask_logits
        return output, cnn_feature

    if backend == "yolo":
        yolo_out = core.yolo(x)
        if isinstance(yolo_out, tuple) and len(yolo_out) >= 2:
            yolo_y, yolo_feats = yolo_out[0], yolo_out[1]
        else:
            yolo_y, yolo_feats = yolo_out, []
        if getattr(core, "use_swin_snake_feature", False):
            cnn_feature = core.swin_snake_feature(x)
            p2 = cnn_feature
        else:
            p2 = yolo_feats[0] if isinstance(yolo_feats, (list, tuple)) and len(yolo_feats) > 0 else None
            if p2 is None:
                raise RuntimeError("YOLO head features are not available; expected P2 feature at index 0")
            cnn_feature = core.cnn_proj(p2)
            if getattr(core, "use_p3_features", False):
                p3 = yolo_feats[1] if isinstance(yolo_feats, (list, tuple)) and len(yolo_feats) > 1 else None
                if p3 is not None:
                    p3_up = F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
                    cnn_feature = cnn_feature + core.cnn_proj_p3(p3_up)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        h_img, w_img = int(round(h * float(core.down_ratio))), int(round(w * float(core.down_ratio)))
        raw_det = core.decode_detection_from_yolo(yolo_y, h_img, w_img)
        detection = core.filter_detection_candidates(raw_det)
        return {
            "detection": detection,
            "feat_hw": (h, w),
            "cnn_feature": cnn_feature,
            "yolo_preds": (yolo_y, yolo_feats),
        }, cnn_feature

    raise RuntimeError(f"E2 script does not support detector_backend={backend}")


def forward_with_detection_source(
    model: torch.nn.Module,
    batch: dict[str, Any],
    det_source: str,
    locate_cache: dict[str, list[dict[str, Any]]] | None,
) -> tuple[dict[str, Any], int]:
    core = model.net if hasattr(model, "net") else model
    x = batch["inp"]
    output, cnn_feature = compute_detector_side_output(core, x, batch)
    if det_source == "locate":
        if locate_cache is None:
            raise RuntimeError("Locate cache is required for det-source=locate")
        det, n_boxes = build_locate_detection(batch, locate_cache, x.device, x.dtype)
        output["detection"] = det
        if det.size(1) > 0:
            output["ct"] = ((det[..., :2] + det[..., 2:4]) * 0.5)
        else:
            output["ct"] = det.new_zeros((det.size(0), 0, 2))
    else:
        det = output.get("detection")
        n_boxes = int((det[..., 4] > 1e-4).sum().item()) if torch.is_tensor(det) and det.numel() > 0 else 0

    if getattr(cfg, "use_gt_det", False):
        raise RuntimeError("E2 requires cfg.use_gt_det=False")
    output = core.attach_extreme_prediction(output, cnn_feature, batch)
    if core.gcn is not None and not getattr(core, "freeze_snake", False):
        output = core.gcn(output, cnn_feature, batch)
    output["feat_hw"] = (cnn_feature.size(2), cnn_feature.size(3))
    output["cnn_feature"] = cnn_feature
    return output, n_boxes


def poly_to_mask(poly_pts: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly_pts).astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def compute_ordered_or_matched_iou(
    pred_polys: np.ndarray,
    gt_polys: np.ndarray,
    height: int,
    width: int,
    match_by_iou: bool,
) -> tuple[list[float], list[int]]:
    gt_masks = [poly_to_mask(poly, height, width) for poly in gt_polys]
    pred_masks = [poly_to_mask(poly, height, width) for poly in pred_polys]

    if (not match_by_iou) and len(pred_masks) == len(gt_masks):
        matched_pred = list(range(len(gt_masks)))
    else:
        matched_pred = [-1 for _ in gt_masks]
        if pred_masks and gt_masks:
            iou_mat = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
            for gi, gt_mask in enumerate(gt_masks):
                for pi, pred_mask in enumerate(pred_masks):
                    iou_mat[gi, pi] = compute_iou(pred_mask, gt_mask)
            while True:
                gi, pi = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[gi, pi] < 0:
                    break
                matched_pred[int(gi)] = int(pi)
                iou_mat[gi, :] = -1
                iou_mat[:, pi] = -1
                if np.all(iou_mat < 0):
                    break

    per_iou = []
    for gi, gt_mask in enumerate(gt_masks):
        pi = matched_pred[gi]
        if pi < 0 or pi >= len(pred_masks):
            per_iou.append(0.0)
        else:
            per_iou.append(compute_iou(pred_masks[pi], gt_mask))
    return per_iou, matched_pred


def valid_gt_polys_and_labels(batch: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    gt_all = batch["i_gt_py"]
    num_points = gt_all.shape[2]
    keep = batch["ct_01"].bool() if "ct_01" in batch else torch.ones(gt_all.shape[:2], dtype=torch.bool, device=gt_all.device)
    gt_polys = gt_all[keep].detach().cpu().numpy().reshape(-1, num_points, 2) * float(snake_config.down_ratio)
    labels = None
    if "ct_cls" in batch:
        labels = batch["ct_cls"][keep].detach().cpu().numpy().astype(np.int32)
    if labels is None:
        labels = np.zeros((gt_polys.shape[0],), dtype=np.int32)
    return gt_polys, labels


def tensor_polys_to_input_np(t: torch.Tensor) -> np.ndarray:
    if not torch.is_tensor(t) or t.numel() == 0:
        return np.zeros((0, int(snake_config.poly_num), 2), dtype=np.float32)
    return t.detach().cpu().numpy().astype(np.float32) * float(snake_config.down_ratio)


def extract_pred_labels(output: dict[str, Any]) -> np.ndarray:
    det = output.get("detection")
    if not torch.is_tensor(det) or det.numel() == 0 or det.size(-1) < 6:
        return np.zeros((0,), dtype=np.int32)
    flat = det.detach().reshape(-1, det.size(-1))
    flat = flat[flat[:, 4] > 1e-4]
    return flat[:, 5].detach().cpu().numpy().astype(np.int32)


def save_overlay(path: Path, img: np.ndarray, gt_polys: np.ndarray, init_polys: np.ndarray, pred_polys: np.ndarray) -> None:
    vis = img.copy()
    for poly in gt_polys:
        cv2.polylines(vis, [np.round(poly).astype(np.int32)], True, (80, 255, 80), 1, lineType=cv2.LINE_AA)
    for poly in init_polys:
        cv2.polylines(vis, [np.round(poly).astype(np.int32)], True, (255, 200, 80), 1, lineType=cv2.LINE_AA)
    for poly in pred_polys:
        cv2.polylines(vis, [np.round(poly).astype(np.int32)], True, (60, 60, 255), 1, lineType=cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)


def eval_sample(
    model: torch.nn.Module,
    device: torch.device,
    batch: dict[str, Any],
    det_source: str,
    locate_cache: dict[str, list[dict[str, Any]]] | None,
    save_visual_path: Path | None,
) -> dict[str, Any]:
    for key, value in list(batch.items()):
        if key == "locate_feat" or str(key).startswith("locate_feat_"):
            continue
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)

    with torch.no_grad():
        output, num_input_boxes = forward_with_detection_source(model, batch, det_source, locate_cache)

    gt_polys, gt_labels = valid_gt_polys_and_labels(batch)
    pred_polys = tensor_polys_to_input_np(output.get("py"))
    init_src = output.get("i_it_py", torch.zeros_like(output.get("py", torch.zeros(0, device=device))))
    init_polys = tensor_polys_to_input_np(init_src)
    pred_labels = extract_pred_labels(output)

    img_raw = batch.get("orig_img", [np.zeros((512, 512, 3), dtype=np.uint8)])[0]
    img = img_raw.detach().cpu().numpy() if torch.is_tensor(img_raw) else np.asarray(img_raw)
    img = img.astype(np.uint8)
    height, width = img.shape[:2]

    match_by_iou = True
    final_iou, matched_pred = compute_ordered_or_matched_iou(pred_polys, gt_polys, height, width, match_by_iou)
    init_iou, _ = compute_ordered_or_matched_iou(init_polys, gt_polys, height, width, match_by_iou)

    if save_visual_path is not None:
        save_overlay(save_visual_path, img, gt_polys, init_polys, pred_polys)

    img_path = batch.get("img_path")
    if isinstance(img_path, (list, tuple)):
        img_path = img_path[0]
    return {
        "img_path": str(img_path),
        "num_gt_contours": int(gt_polys.shape[0]),
        "num_input_boxes": int(num_input_boxes),
        "num_pred_contours": int(pred_polys.shape[0]),
        "mean_init_iou": float(np.mean(init_iou)) if init_iou else 0.0,
        "mean_final_iou": float(np.mean(final_iou)) if final_iou else 0.0,
        "per_contour_init_iou": [float(x) for x in init_iou],
        "per_contour_final_iou": [float(x) for x in final_iou],
        "matched_pred": [int(x) for x in matched_pred],
        "gt_labels": [int(x) for x in gt_labels.tolist()],
        "pred_labels": [int(x) for x in pred_labels.tolist()],
    }


def summarize(rows: list[dict[str, Any]], dataset_size: int, failed: list[dict[str, Any]], save_dir: Path, ckpt_path: Path) -> dict[str, Any]:
    sample_init = [r["mean_init_iou"] for r in rows]
    sample_final = [r["mean_final_iou"] for r in rows]
    contour_init = [x for r in rows for x in r["per_contour_init_iou"]]
    contour_final = [x for r in rows for x in r["per_contour_final_iou"]]
    box_counts = [r["num_input_boxes"] for r in rows]
    return {
        "task": "E2 LocateAnything box-initialized V8.2 full-chain evaluation",
        "timestamp": _dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "det_source": ARGS.det_source,
        "cfg_file": str(Path(ARGS.cfg_file).resolve()),
        "checkpoint": str(ckpt_path),
        "locate_json": str(Path(ARGS.locate_json).resolve()) if ARGS.det_source == "locate" else "",
        "save_dir": str(save_dir),
        "dataset": str(cfg.test.dataset),
        "test_img_path": str(getattr(cfg.test, "img_path", "")),
        "dataset_size": int(dataset_size),
        "start_index": int(ARGS.start_index),
        "limit": int(ARGS.limit),
        "evaluated_samples": int(len(rows)),
        "failed_samples": int(len(failed)),
        "failed": failed,
        "device_requested": str(ARGS.device),
        "visible_gpu": str(ARGS.visible_gpu),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "device_actual": "cuda" if torch.cuda.is_available() and ARGS.device.startswith("cuda") else "cpu",
        "det_conf_thresh": float(getattr(cfg, "det_conf_thresh", 0.0)),
        "det_iou_thresh": float(getattr(cfg, "det_iou_thresh", 0.0)),
        "det_max_det": int(getattr(cfg, "det_max_det", 0)),
        "locate_policy": "score=1.0 in cache; all Locate boxes are used before downstream model filtering" if ARGS.det_source == "locate" else "",
        "matching": "Pred/init contours are greedily one-to-one matched to GT by mask IoU per sample; sample mean averages GT contours in that sample.",
        "summary": {
            "mean_init_iou_sample_avg": float(np.mean(sample_init)) if sample_init else 0.0,
            "mean_final_iou_sample_avg": float(np.mean(sample_final)) if sample_final else 0.0,
            "mean_init_iou_contour_avg": float(np.mean(contour_init)) if contour_init else 0.0,
            "mean_final_iou_contour_avg": float(np.mean(contour_final)) if contour_final else 0.0,
            "median_init_iou_sample_avg": float(np.median(sample_init)) if sample_init else 0.0,
            "median_final_iou_sample_avg": float(np.median(sample_final)) if sample_final else 0.0,
            "avg_input_boxes_per_image": float(np.mean(box_counts)) if box_counts else 0.0,
            "total_gt_contours": int(sum(r["num_gt_contours"] for r in rows)),
            "total_input_boxes": int(sum(box_counts)),
            "total_pred_contours": int(sum(r["num_pred_contours"] for r in rows)),
        },
        "samples": rows,
    }


def main() -> None:
    apply_eval_overrides()
    set_eval_seed(ARGS.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and ARGS.device.startswith("cuda") else "cpu")
    if ARGS.ode_steps > 0:
        cfg.flow_ode_steps = int(ARGS.ode_steps)
        cfg.iterative_ode_steps = int(ARGS.ode_steps)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(ARGS.save_dir) if ARGS.save_dir else REPO_ROOT / "visual" / f"e2_locate_box_init_eval_{ts}"
    save_dir = save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    locate_cache = load_locate_cache(Path(ARGS.locate_json).resolve()) if ARGS.det_source == "locate" else None
    ckpt_path = resolve_checkpoint(ARGS.ckpt)
    model = load_model(ckpt_path).to(device).eval()

    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)
    dataset_size = len(dataset)
    start = max(int(ARGS.start_index), 0)
    limit = int(ARGS.limit) if ARGS.limit and ARGS.limit > 0 else dataset_size - start
    end = min(dataset_size, start + limit)
    indices = list(range(start, end))

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    print(f"[*] E2 evaluating det_source={ARGS.det_source} samples={len(indices)}/{dataset_size} device={device}")
    print(f"[*] save_dir={save_dir}")
    for ordinal, index in enumerate(indices, 1):
        try:
            batch = collator([dataset[index]])
            vis_path = save_dir / "visuals" / f"idx_{index:03d}.png" if ARGS.save_visuals else None
            row = eval_sample(model, device, batch, ARGS.det_source, locate_cache, vis_path)
            row["index"] = int(index)
            rows.append(row)
            if ARGS.progress_every > 0 and (ordinal % ARGS.progress_every == 0 or ordinal == len(indices)):
                print(
                    f"[{ordinal}/{len(indices)}] idx={index} boxes={row['num_input_boxes']} "
                    f"gt={row['num_gt_contours']} init={row['mean_init_iou']:.4f} final={row['mean_final_iou']:.4f}"
                )
        except Exception as exc:
            failed.append({"index": int(index), "error": str(exc)})
            print(f"[!] failed idx={index}: {exc}")

    report = summarize(rows, dataset_size, failed, save_dir, ckpt_path)
    out_path = save_dir / f"e2_locate_box_init_eval_{ARGS.det_source}_{report['timestamp']}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("\n" + "=" * 80)
    print(f"Saved report: {out_path}")
    print(f"evaluated_samples: {report['evaluated_samples']} failed_samples: {report['failed_samples']}")
    print(f"avg_input_boxes_per_image: {report['summary']['avg_input_boxes_per_image']:.4f}")
    print(f"mean_init_iou_sample_avg: {report['summary']['mean_init_iou_sample_avg']:.6f}")
    print(f"mean_final_iou_sample_avg: {report['summary']['mean_final_iou_sample_avg']:.6f}")
    print(f"mean_init_iou_contour_avg: {report['summary']['mean_init_iou_contour_avg']:.6f}")
    print(f"mean_final_iou_contour_avg: {report['summary']['mean_final_iou_contour_avg']:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
