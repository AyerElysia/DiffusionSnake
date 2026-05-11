import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate heatmap detector + diffusion checkpoint on BTCV.")
    parser.add_argument("--cfg", default="configs/btcv_diffusion_dit_v3_heatmap_resnet18.yaml")
    parser.add_argument("--detector-ckpt", required=True)
    parser.add_argument("--diffusion-ckpt", required=True)
    parser.add_argument("--dataset", choices=["train", "val"], default="val")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--save-dir", default="data/eval/heatmap_diffusion_smoke")
    parser.add_argument("--save-overlays", action="store_true")
    return parser.parse_args()


def unwrap_state_dict(state_dict):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.net.") for k in keys):
        return {k[len("module.net."):]: v for k, v in state_dict.items()}
    if all(k.startswith("net.") for k in keys):
        return {k[4:]: v for k, v in state_dict.items()}
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint_state(path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("net") or ckpt
    return unwrap_state_dict(state)


def polygon_to_mask(poly, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly).astype(np.int32)
    if pts.shape[0] >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask


def polygon_iou(poly_a, poly_b, h, w):
    mask_a = polygon_to_mask(poly_a, h, w)
    mask_b = polygon_to_mask(poly_b, h, w)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def greedy_match(gt_polys, gt_clses, pred_polys, pred_clses, image_hw):
    h, w = image_hw
    pairs = []
    for gt_idx, gt_poly in enumerate(gt_polys):
        for pred_idx, pred_poly in enumerate(pred_polys):
            if int(pred_clses[pred_idx]) != int(gt_clses[gt_idx]):
                continue
            iou = polygon_iou(gt_poly, pred_poly, h, w)
            if iou > 0:
                pairs.append((iou, gt_idx, pred_idx))

    pairs.sort(reverse=True, key=lambda x: x[0])
    matched_gt = set()
    matched_pred = set()
    matches = []
    for iou, gt_idx, pred_idx in pairs:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        matches.append({"gt_index": gt_idx, "pred_index": pred_idx, "iou": float(iou)})
    return matches


def save_overlay(path, image, gt_polys, pred_polys, pred_clses, pred_scores):
    canvas = image.copy()
    for poly in gt_polys:
        cv2.polylines(canvas, [np.round(poly).astype(np.int32)], True, (0, 255, 0), 2)
    for poly, cls_id, score in zip(pred_polys, pred_clses, pred_scores):
        poly_i = np.round(poly).astype(np.int32)
        cv2.polylines(canvas, [poly_i], True, (0, 0, 255), 2)
        cx = int(poly_i[:, 0].mean())
        cy = int(poly_i[:, 1].mean())
        cv2.putText(canvas, f"{int(cls_id)}:{float(score):.2f}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, canvas)


def main():
    args = parse_args()
    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()

    os.environ["CFG_FILE"] = str(cfg_path)
    sys.argv = [sys.argv[0]]

    from lib.config import cfg
    from lib.datasets.collate_batch import make_collator
    from lib.datasets.make_dataset import make_dataset
    from lib.datasets.transforms import make_transforms
    from lib.networks import make_network
    from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
    from lib.utils.snake import snake_config

    cfg.train.data_path = "/home/medteam/Zhrch/Datasets/BTCV/btcv_png_new_snake"
    cfg.test.img_path = "/home/medteam/Zhrch/Datasets/BTCV/btcv_png_test_new_snake"

    diffusion_state = remap_legacy_state_dict(load_checkpoint_state(args.diffusion_ckpt))
    detector_state = load_checkpoint_state(args.detector_ckpt)

    merged = {}
    for key, value in diffusion_state.items():
        if key.startswith("yolo.") or key.startswith("cnn_proj"):
            continue
        merged[key] = value
    for key, value in detector_state.items():
        if key.startswith("heatmap_detector."):
            merged[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = make_network(cfg).to(device).eval()
    info = network.load_state_dict(merged, strict=False)
    print(
        json.dumps(
            {
                "loaded_keys": len(merged),
                "missing_keys": len(info.missing_keys),
                "unexpected_keys": len(info.unexpected_keys),
                "missing_preview": info.missing_keys[:10],
                "unexpected_preview": info.unexpected_keys[:10],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    dataset_name = cfg.train.dataset if args.dataset == "train" else cfg.test.dataset
    dataset = make_dataset(cfg, dataset_name, make_transforms(cfg, is_train=(args.dataset == "train")), is_train=(args.dataset == "train"))
    collator = make_collator(cfg)
    max_samples = min(len(dataset), max(int(args.max_samples), 1))
    save_dir = (ROOT / args.save_dir).resolve()
    os.makedirs(save_dir, exist_ok=True)

    dr = float(getattr(snake_config, "down_ratio", 4.0))

    total_gt = 0
    total_pred = 0
    total_matched = 0
    best_iou_sum = 0.0
    matched_iou_sum = 0.0
    recall50 = 0
    recall75 = 0
    rows = []

    for index in range(max_samples):
        sample = dataset[index]
        batch = collator([sample])
        image = batch["orig_img"][0]
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        image = image.astype(np.uint8)
        h, w = image.shape[:2]

        for key, value in list(batch.items()):
            if key in ("meta", "orig_img", "img_path"):
                continue
            if torch.is_tensor(value):
                batch[key] = value.to(device)

        with torch.no_grad():
            output = network(batch["inp"], batch)

        pred_polys = output["py"][-1] if isinstance(output.get("py"), list) else output.get("py")
        if pred_polys is None:
            raise RuntimeError("Network output does not contain 'py'")
        pred_polys = pred_polys.detach().cpu().numpy() * dr
        detection = output["detection"][0].detach().cpu().numpy()
        pred_scores = detection[:, 4] if len(detection) > 0 else np.zeros((0,), dtype=np.float32)
        pred_clses = detection[:, 5].astype(np.int32) if len(detection) > 0 else np.zeros((0,), dtype=np.int32)
        pred_count = min(len(pred_polys), len(pred_clses))
        pred_polys = pred_polys[:pred_count]
        pred_scores = pred_scores[:pred_count]
        pred_clses = pred_clses[:pred_count]

        gt_polys = batch["i_gt_py"][0]
        valid_mask = batch["ct_01"][0].bool() if "ct_01" in batch else torch.ones((gt_polys.shape[0],), dtype=torch.bool, device=gt_polys.device)
        gt_polys = gt_polys[valid_mask].detach().cpu().numpy() * dr
        gt_clses = sample["cls"].view(-1).numpy().astype(np.int32)

        matches = greedy_match(gt_polys, gt_clses, pred_polys, pred_clses, (h, w))
        matched_by_gt = {m["gt_index"]: m for m in matches}

        total_gt += len(gt_polys)
        total_pred += len(pred_polys)
        total_matched += len(matches)
        matched_iou_sum += sum(m["iou"] for m in matches)

        per_sample = {
            "index": index,
            "num_gt": int(len(gt_polys)),
            "num_pred": int(len(pred_polys)),
            "num_matched": int(len(matches)),
            "matches": matches,
        }
        rows.append(per_sample)

        if args.save_overlays:
            save_overlay(
                str(save_dir / f"sample_{index:03d}.png"),
                image,
                gt_polys,
                pred_polys,
                pred_clses,
                pred_scores,
            )

        for gt_idx in range(len(gt_polys)):
            best_iou = float(matched_by_gt.get(gt_idx, {}).get("iou", 0.0))
            best_iou_sum += best_iou
            if best_iou >= 0.5:
                recall50 += 1
            if best_iou >= 0.75:
                recall75 += 1

    summary = {
        "dataset": args.dataset,
        "max_samples": int(max_samples),
        "detector_ckpt": str(Path(args.detector_ckpt).resolve()),
        "diffusion_ckpt": str(Path(args.diffusion_ckpt).resolve()),
        "num_gt": int(total_gt),
        "num_pred": int(total_pred),
        "num_matched": int(total_matched),
        "precision_like": float(total_matched / max(total_pred, 1)),
        "recall50": float(recall50 / max(total_gt, 1)),
        "recall75": float(recall75 / max(total_gt, 1)),
        "mean_best_iou": float(best_iou_sum / max(total_gt, 1)),
        "matched_mean_iou": float(matched_iou_sum / max(total_matched, 1)) if total_matched > 0 else 0.0,
    }
    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(save_dir / "rows.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
