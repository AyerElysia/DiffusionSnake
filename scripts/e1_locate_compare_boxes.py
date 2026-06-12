#!/usr/bin/env python3
"""
E1 LocateAnything vs V8.2 detection-head box comparison.

Run with:
  CUDA_VISIBLE_DEVICES=2 /home/medteam/miniconda3/envs/snake1/bin/python scripts/e1_locate_compare_boxes.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO_ROOT / "configs" / "1232_final_v8_2_mged_final_gpu6.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-file", default=str(DEFAULT_CFG))
    parser.add_argument(
        "--locate-json",
        default=str(REPO_ROOT / "data" / "eagle_teacher" / "1232_final_test_locateanything_ckpt3000.json"),
    )
    parser.add_argument("--data-root", default="/home/medteam/Zhrch/Datasets/1232_final")
    parser.add_argument("--split", default="test")
    parser.add_argument("--ckpt", default="")
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "data" / "eagle_teacher" / "e1_locate_vs_v8_2_detection_report.json"),
    )
    parser.add_argument(
        "--output-txt",
        default=str(REPO_ROOT / "data" / "eagle_teacher" / "e1_locate_vs_v8_2_detection_report.txt"),
    )
    parser.add_argument("--visible-gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "2"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="0 means all samples from Locate cache")
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--scan-thresholds",
        default="0.05,0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated V8.2 confidence thresholds for fair box-budget scan.",
    )
    parser.add_argument("--det-conf-thresh", type=float, default=None)
    parser.add_argument("--det-iou-thresh", type=float, default=None)
    parser.add_argument("--det-max-det", type=int, default=None)
    return parser.parse_args()


ARGS = parse_args()
os.environ["CFG_FILE"] = str(Path(ARGS.cfg_file).resolve())
sys.argv = [sys.argv[0], "--cfg_file", os.environ["CFG_FILE"]]

from lib.config import cfg  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.visible_gpu
cfg.gpus = [0]

import torch  # noqa: E402

from lib.networks import make_network  # noqa: E402
from lib.utils import data_utils  # noqa: E402
from lib.utils.snake import snake_config, snake_voc_utils  # noqa: E402


MASK_RE = re.compile(r"^(?P<stem>.+)_mask_(?P<class_id>\d+)\.png$")


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


def load_locate_samples(path: Path, limit: int) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or not isinstance(obj.get("samples"), list):
        raise ValueError(f"Locate cache must be a dict with a samples list: {path}")
    samples = obj["samples"]
    samples = sorted(samples, key=lambda rec: str(rec.get("image_rel") or rec.get("img_path") or ""))
    if limit and limit > 0:
        samples = samples[:limit]
    return samples


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a[:4]]
    bx1, by1, bx2, by2 = [float(x) for x in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def clip_box(box: list[float] | np.ndarray, width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
    y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def load_gt_boxes(image_path: Path, min_area: int) -> list[dict]:
    with Image.open(image_path) as img:
        width, height = img.size
    stem = image_path.name[: -len("_image.png")] if image_path.name.endswith("_image.png") else image_path.stem
    boxes = []
    for mask_path in sorted(image_path.parent.glob(f"{stem}_mask_*.png"), key=lambda p: p.name):
        match = MASK_RE.match(mask_path.name)
        if not match:
            continue
        class_id = int(match.group("class_id"))
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        binary = (mask > 0).astype(np.uint8)
        if binary.max() == 0:
            continue
        n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for comp_id in range(1, n):
            x, y, w, h, area = stats[comp_id]
            if int(area) < min_area:
                continue
            box = clip_box([float(x), float(y), float(x + w), float(y + h)], width, height)
            if box is not None:
                boxes.append({"bbox": box, "label_id": class_id})
    boxes.sort(key=lambda rec: (int(rec["label_id"]), rec["bbox"][1], rec["bbox"][0]))
    return boxes


def locate_pred_boxes(sample: dict) -> list[dict]:
    preds = []
    width = int(sample.get("width") or 0)
    height = int(sample.get("height") or 0)
    for inst in sample.get("instances") or []:
        box = inst.get("bbox")
        if box is None:
            continue
        clipped = clip_box(box, width, height) if width > 0 and height > 0 else [float(v) for v in box[:4]]
        if clipped is None:
            continue
        label = inst.get("label_id", inst.get("cls_id", inst.get("category_id", None)))
        try:
            label = int(label) if label is not None else None
        except Exception:
            label = None
        preds.append({"bbox": clipped, "label_id": label, "score": float(inst.get("score", inst.get("confidence", 1.0)))})
    return preds


def parse_scan_thresholds(text: str) -> list[float]:
    vals = []
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    if not vals:
        raise ValueError("At least one scan threshold is required")
    return sorted(set(vals))


def filter_preds_by_score(pred_by_sample: list[list[dict]], score_thresh: float) -> list[list[dict]]:
    return [[pred for pred in preds if float(pred.get("score", 0.0)) >= score_thresh] for preds in pred_by_sample]


def one_to_one_stats(gt_by_sample: list[list[dict]], pred_by_sample: list[list[dict]], class_aware: bool, iou_threshold: float) -> dict:
    tp = 0
    fp = 0
    gt_total = 0
    pred_total = 0
    for gts, preds in zip(gt_by_sample, pred_by_sample):
        gt_total += len(gts)
        pred_total += len(preds)
        used_gt = set()
        sorted_preds = sorted(preds, key=lambda p: float(p.get("score", 0.0)), reverse=True)
        for pred in sorted_preds:
            best_iou = 0.0
            best_idx = -1
            for gi, gt in enumerate(gts):
                if gi in used_gt:
                    continue
                if class_aware and pred.get("label_id") != gt.get("label_id"):
                    continue
                iou = box_iou(np.asarray(gt["bbox"]), np.asarray(pred["bbox"]))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi
            if best_iou >= iou_threshold and best_idx >= 0:
                tp += 1
                used_gt.add(best_idx)
            else:
                fp += 1
    fn = max(gt_total - tp, 0)
    return {
        "one_to_one_tp": int(tp),
        "one_to_one_fp": int(fp),
        "one_to_one_fn": int(fn),
        f"precision_iou_{iou_threshold:g}": float(tp / pred_total) if pred_total else 0.0,
        f"one_to_one_recall_iou_{iou_threshold:g}": float(tp / gt_total) if gt_total else 0.0,
    }


def aggregate_metrics(gt_by_sample: list[list[dict]], pred_by_sample: list[list[dict]], class_aware: bool, iou_threshold: float) -> dict:
    all_best = []
    pred_count = 0
    rows = []
    for idx, (gts, preds) in enumerate(zip(gt_by_sample, pred_by_sample)):
        pred_count += len(preds)
        sample_best = []
        for gt in gts:
            candidates = preds
            if class_aware:
                candidates = [p for p in preds if p.get("label_id") == gt.get("label_id")]
            best = max((box_iou(np.asarray(gt["bbox"]), np.asarray(pred["bbox"])) for pred in candidates), default=0.0)
            all_best.append(best)
            sample_best.append(best)
        rows.append(
            {
                "sample_index": idx,
                "gt_instances": len(gts),
                "pred_instances": len(preds),
                "mean_best_iou": float(np.mean(sample_best)) if sample_best else 0.0,
                f"recall_iou_{iou_threshold:g}": float(np.mean(np.asarray(sample_best) >= iou_threshold)) if sample_best else 0.0,
            }
        )
    best_arr = np.asarray(all_best, dtype=np.float32)
    mean_best = float(best_arr.mean()) if best_arr.size else 0.0
    out = {
        "samples": len(gt_by_sample),
        "gt_instances": int(best_arr.size),
        "pred_instances": int(pred_count),
        "avg_boxes_per_image": float(pred_count / len(gt_by_sample)) if gt_by_sample else 0.0,
        "mean_iou": mean_best,
        "mean_best_iou": mean_best,
        f"recall_iou_{iou_threshold:g}": float(np.mean(best_arr >= iou_threshold)) if best_arr.size else 0.0,
        "matched_gt_instances": int(np.sum(best_arr >= iou_threshold)) if best_arr.size else 0,
        "per_sample": rows,
    }
    out.update(one_to_one_stats(gt_by_sample, pred_by_sample, class_aware, iou_threshold))
    return out


def compact_metrics(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "per_sample"}


def load_v8_model(ckpt_path: Path, candidate_conf_thresh: float):
    cfg.use_gt_det = False
    cfg.skip_diffusion_forward = True
    cfg.detector_only_warmup = True
    cfg.use_extreme_refine = False
    cfg.contour_init_method = "octagon"
    cfg.det_conf_thresh = float(candidate_conf_thresh)
    if ARGS.det_iou_thresh is not None:
        cfg.det_iou_thresh = float(ARGS.det_iou_thresh)
    if ARGS.det_max_det is not None:
        cfg.det_max_det = int(ARGS.det_max_det)

    network = make_network(cfg)

    ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt_obj.get("state_dict") or ckpt_obj.get("model") or ckpt_obj.get("net") or ckpt_obj
    from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict

    sd = remap_legacy_state_dict(sd)
    target = network.state_dict()
    filtered = {}
    for key, value in sd.items():
        clean = key
        for prefix in ("module.", "net."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
        if clean in target and tuple(value.shape) == tuple(target[clean].shape):
            filtered[clean] = value
    info = network.load_state_dict(filtered, strict=False)
    loaded = len(filtered)
    print(
        f"Loaded V8.2 checkpoint: {ckpt_path} "
        f"({loaded}/{len(target)} network keys, missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)})"
    )
    return network


def preprocess_image_for_v8(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = img.shape[:2]
    _orig_img, inp, _trans_input, _trans_output, _flipped, center, scale, _inp_out_hw = snake_voc_utils.augment(
        img,
        "val",
        snake_config.data_rng,
        snake_config.eig_val,
        snake_config.eig_vec,
        snake_config.mean,
        snake_config.std,
        None,
    )
    inv_trans_input = data_utils.get_affine_transform(center, scale, 0, [512, 512], inv=1)
    tensor = torch.from_numpy(inp[None]).float()
    return tensor, inv_trans_input, width, height


def input_box_to_original(box: np.ndarray, inv_trans_input: np.ndarray, width: int, height: int) -> list[float] | None:
    pts = np.array([[box[0], box[1]], [box[2], box[3]]], dtype=np.float32)
    pts = data_utils.affine_transform(pts, inv_trans_input)
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    return clip_box([float(x1), float(y1), float(x2), float(y2)], width, height)


def run_v8_predictions(model, samples: list[dict]) -> list[list[dict]]:
    device = torch.device(ARGS.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    pred_by_sample = []
    with torch.no_grad():
        for idx, sample in enumerate(samples):
            image_path = Path(sample.get("img_path") or sample.get("image_path"))
            if not image_path.is_absolute():
                image_path = (Path(ARGS.data_root) / str(sample.get("image_rel", ""))).resolve()
            inp, inv_trans_input, width, height = preprocess_image_for_v8(image_path)
            output = model(inp.to(device), None)
            det = output.get("detection")
            preds = []
            if torch.is_tensor(det) and det.numel() > 0:
                det_np = det.detach().cpu().numpy()
                det_np = det_np.reshape(-1, det_np.shape[-1])
                for row in det_np:
                    if row.shape[0] < 6 or float(row[4]) <= 0.0:
                        continue
                    box = input_box_to_original(row[:4], inv_trans_input, width, height)
                    if box is None:
                        continue
                    preds.append({"bbox": box, "score": float(row[4]), "label_id": int(row[5])})
            pred_by_sample.append(preds)
            print(f"[{idx + 1}/{len(samples)}] V8.2 detections: {len(preds)}")
    return pred_by_sample


def write_text_report(path: Path, report: dict) -> None:
    threshold = report["matching"]["iou_threshold"]
    fair = report["metrics"]["fair"]
    aligned = fair["budget_aligned"]
    locate_agn = aligned["locate"]["class_agnostic"]
    locate_cls = aligned["locate"]["class_aware"]
    v8_agn = aligned["v8_2"]["class_agnostic"]
    v8_cls = aligned["v8_2"]["class_aware"]
    selected_thr = aligned["v8_2"]["det_conf_thresh"]
    legacy = report["metrics"]["legacy_inflated"]

    def metric_line(name: str, m: dict) -> str:
        return (
            f"{name:<18} {m['avg_boxes_per_image']:>8.2f} "
            f"{m['mean_best_iou']:>10.6f} {m[f'recall_iou_{threshold:g}']:>10.6f} "
            f"{m[f'precision_iou_{threshold:g}']:>10.6f}"
        )

    def scan_lines(rows: list[dict], key: str) -> list[str]:
        out = ["thr      avg_box   meanBestIoU  recall@0.5  precision@0.5"]
        for row in rows:
            m = row[key]
            out.append(
                f"{row['det_conf_thresh']:<7.2f} "
                f"{m['avg_boxes_per_image']:>8.2f} "
                f"{m['mean_best_iou']:>11.6f} "
                f"{m[f'recall_iou_{threshold:g}']:>11.6f} "
                f"{m[f'precision_iou_{threshold:g}']:>14.6f}"
            )
        return out

    lines = [
        "E1 LocateAnything detection quality comparison",
        "",
        f"Samples: {report['samples']}",
        f"GT instances: {report['gt_instances']}",
        f"Matching: {report['matching']['description']}",
        "",
        "Budget-aligned comparison (V8.2 threshold closest to Locate avg boxes/image)",
        f"Locate avg boxes/image: {fair['locate']['class_agnostic']['avg_boxes_per_image']:.2f}",
        f"Selected V8.2 det_conf_thresh: {selected_thr:.2f}",
        "",
        "Class-agnostic",
        "method              avg_box   meanBestIoU  recall@0.5  precision@0.5",
        metric_line("Locate", locate_agn),
        metric_line(f"V8.2@{selected_thr:.2f}", v8_agn),
        "",
        "Class-aware",
        "method              avg_box   meanBestIoU  recall@0.5  precision@0.5",
        metric_line("Locate", locate_cls),
        metric_line(f"V8.2@{selected_thr:.2f}", v8_cls),
        "",
        "V8.2 threshold scan: class-agnostic",
        *scan_lines(fair["v8_2_threshold_scan"], "class_agnostic"),
        "",
        "V8.2 threshold scan: class-aware",
        *scan_lines(fair["v8_2_threshold_scan"], "class_aware"),
        "",
        "Legacy inflated口径 (class-agnostic, per-GT best IoU over all predictions, no one-to-one assignment)",
        f"Locate meanBestIoU={legacy['locate_vs_gt']['mean_best_iou']:.6f} "
        f"recall@0.5={legacy['locate_vs_gt'][f'recall_iou_{threshold:g}']:.6f} "
        f"pred_boxes={legacy['locate_vs_gt']['pred_instances']}",
        f"V8.2@{legacy['v8_2_detection_vs_gt']['det_conf_thresh']:.2f} "
        f"meanBestIoU={legacy['v8_2_detection_vs_gt']['class_agnostic']['mean_best_iou']:.6f} "
        f"recall@0.5={legacy['v8_2_detection_vs_gt']['class_agnostic'][f'recall_iou_{threshold:g}']:.6f} "
        f"pred_boxes={legacy['v8_2_detection_vs_gt']['class_agnostic']['pred_instances']}",
        "",
        f"Locate confidence policy: {report['thresholds']['locate']}",
        f"V8.2 thresholds: {json.dumps(report['thresholds']['v8_2_detection_head'], ensure_ascii=False)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    locate_path = Path(ARGS.locate_json).resolve()
    data_root = Path(ARGS.data_root).resolve()
    out_json = Path(ARGS.output_json).resolve()
    out_txt = Path(ARGS.output_txt).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    samples = load_locate_samples(locate_path, ARGS.limit)
    gt_by_sample = []
    locate_by_sample = []
    image_rels = []
    for sample in samples:
        image_path = Path(sample.get("img_path") or sample.get("image_path"))
        if not image_path.is_absolute():
            image_rel = str(sample.get("image_rel") or sample.get("image") or "")
            image_path = data_root / image_rel
        image_rels.append(str(sample.get("image_rel") or image_path.name))
        gt_by_sample.append(load_gt_boxes(image_path, ARGS.min_area))
        locate_by_sample.append(locate_pred_boxes(sample))

    scan_thresholds = parse_scan_thresholds(ARGS.scan_thresholds)
    legacy_conf_thresh = float(ARGS.det_conf_thresh if ARGS.det_conf_thresh is not None else getattr(cfg, "det_conf_thresh", 0.01))
    candidate_conf_thresh = min([legacy_conf_thresh] + scan_thresholds)

    ckpt_path = resolve_checkpoint(ARGS.ckpt)
    model = load_v8_model(ckpt_path, candidate_conf_thresh)
    v8_by_sample = run_v8_predictions(model, samples)

    locate_agn = aggregate_metrics(gt_by_sample, locate_by_sample, False, ARGS.iou_threshold)
    locate_cls = aggregate_metrics(gt_by_sample, locate_by_sample, True, ARGS.iou_threshold)
    legacy_v8_preds = filter_preds_by_score(v8_by_sample, legacy_conf_thresh)
    legacy_v8_agn = aggregate_metrics(gt_by_sample, legacy_v8_preds, False, ARGS.iou_threshold)
    legacy_v8_cls = aggregate_metrics(gt_by_sample, legacy_v8_preds, True, ARGS.iou_threshold)

    scan_rows = []
    for thr in scan_thresholds:
        preds_thr = filter_preds_by_score(v8_by_sample, thr)
        scan_rows.append(
            {
                "det_conf_thresh": float(thr),
                "class_agnostic": compact_metrics(aggregate_metrics(gt_by_sample, preds_thr, False, ARGS.iou_threshold)),
                "class_aware": compact_metrics(aggregate_metrics(gt_by_sample, preds_thr, True, ARGS.iou_threshold)),
            }
        )

    target_avg_boxes = float(locate_agn["avg_boxes_per_image"])
    selected = min(scan_rows, key=lambda row: abs(row["class_agnostic"]["avg_boxes_per_image"] - target_avg_boxes))

    report = {
        "task": "E1 LocateAnything detection quality comparison",
        "samples": len(samples),
        "gt_instances": locate_agn["gt_instances"],
        "paths": {
            "locate_json": str(locate_path),
            "v8_2_cfg": str(Path(ARGS.cfg_file).resolve()),
            "v8_2_checkpoint": str(ckpt_path),
            "data_root": str(data_root),
        },
        "runtime": {
            "requested_visible_gpu": str(ARGS.visible_gpu),
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "requested_device": str(ARGS.device),
            "actual_v8_device": str(ARGS.device if torch.cuda.is_available() else "cpu"),
        },
        "matching": {
            "iou_threshold": float(ARGS.iou_threshold),
            "description": "meanBestIoU/recall: each GT takes max IoU in the same image. precision: predictions are sorted by confidence and greedily matched one-to-one to unused GT at IoU>=threshold. Both class-agnostic and class-aware variants are reported.",
        },
        "thresholds": {
            "locate": "no confidence threshold; LocateAnything cache stores score=confidence=1.0 for each parsed box",
            "v8_2_detection_head": {
                "candidate_det_conf_thresh_for_single_forward": float(candidate_conf_thresh),
                "legacy_det_conf_thresh": float(legacy_conf_thresh),
                "scan_det_conf_thresh": scan_thresholds,
                "det_iou_thresh": float(getattr(cfg, "det_iou_thresh", 0.0)),
                "det_max_det": int(getattr(cfg, "det_max_det", 0)),
                "per_class_nms": bool(getattr(cfg, "per_class_nms", True)),
                "use_nms_for_snake": bool(getattr(cfg, "use_nms_for_snake", True)),
                "use_gt_det": bool(getattr(cfg, "use_gt_det", False)),
                "use_gt_det_forced_false": True,
            },
        },
        "metrics": {
            "legacy_inflated": {
                "note": "Inflated because V8.2 at det_conf_thresh=0.01 returns far more boxes and old matching used per-GT max over all predictions without one-to-one assignment.",
                "locate_vs_gt": compact_metrics(locate_agn),
                "v8_2_detection_vs_gt": {
                    "det_conf_thresh": float(legacy_conf_thresh),
                    "class_agnostic": compact_metrics(legacy_v8_agn),
                    "class_aware": compact_metrics(legacy_v8_cls),
                },
            },
            "fair": {
                "locate": {
                    "class_agnostic": compact_metrics(locate_agn),
                    "class_aware": compact_metrics(locate_cls),
                },
                "v8_2_threshold_scan": scan_rows,
                "budget_aligned": {
                    "target_locate_avg_boxes_per_image": target_avg_boxes,
                    "selected_by": "minimum absolute difference in avg_boxes_per_image among scan thresholds",
                    "locate": {
                        "class_agnostic": compact_metrics(locate_agn),
                        "class_aware": compact_metrics(locate_cls),
                    },
                    "v8_2": selected,
                },
            },
        },
        "sample_order": image_rels,
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    write_text_report(out_txt, report)
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))
    print(f"Saved JSON report: {out_json}")
    print(f"Saved text report: {out_txt}")


if __name__ == "__main__":
    main()
