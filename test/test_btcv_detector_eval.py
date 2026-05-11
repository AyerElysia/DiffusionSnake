import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BTCV detector checkpoint on train/val split.")
    parser.add_argument("--cfg", default="configs/btcv_yolo_detect_only.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", choices=["train", "val"], default="val")
    parser.add_argument("--score-thresh", type=float, default=0.01)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-path", default="")
    parser.add_argument("--disable-aug", action="store_true")
    parser.add_argument("--yolo-num-classes", type=int, default=0)
    return parser.parse_args()


def _set_cfg_file(cfg_file: str):
    os.environ["CFG_FILE"] = cfg_file
    sys.argv = [sys.argv[0]]


def _unwrap_network_state_dict(state_dict):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict

    keys = list(state_dict.keys())
    if all(k.startswith("net.") for k in keys):
        return {k[4:]: v for k, v in state_dict.items()}
    if all(k.startswith("module.net.") for k in keys):
        return {k[len("module.net."):]: v for k, v in state_dict.items()}
    return state_dict


def _iou_xyxy(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = max(area_a + area_b - inter, 1e-6)
    return inter / union


def _build_gt_boxes(sample):
    inp_h, inp_w = sample["inp"].shape[1:]
    gt_boxes = []
    for box_xywh, cls_tensor in zip(sample["bboxes"].numpy(), sample["cls"].view(-1).numpy()):
        cx, cy, w, h = [float(x) for x in box_xywh]
        gt_boxes.append(
            [
                (cx - w / 2.0) * inp_w,
                (cy - h / 2.0) * inp_h,
                (cx + w / 2.0) * inp_w,
                (cy + h / 2.0) * inp_h,
                int(cls_tensor),
            ]
        )
    return gt_boxes


def _greedy_match(gt_boxes, detections):
    pairs = []
    for gt_idx, gt_box in enumerate(gt_boxes):
        for pred_idx, pred in enumerate(detections):
            if int(pred[5]) != gt_box[4]:
                continue
            iou = _iou_xyxy(gt_box[:4], pred[:4])
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
        matches.append(
            {
                "gt_index": gt_idx,
                "pred_index": pred_idx,
                "iou": float(iou),
                "score": float(detections[pred_idx][4]),
                "cls": int(gt_boxes[gt_idx][4]),
            }
        )
    return matches


def main():
    args = _parse_args()
    if args.disable_aug:
        os.environ["SNAKE_DISABLE_AUG"] = "1"
    _set_cfg_file(args.cfg)

    from lib.config import cfg
    from lib.datasets.collate_batch import make_collator
    from lib.datasets.make_dataset import make_dataset
    from lib.datasets.transforms import make_transforms
    from lib.networks import make_network

    cfg.train.data_path = "/home/medteam/Zhrch/Datasets/BTCV/btcv_png_new_snake"
    cfg.test.img_path = "/home/medteam/Zhrch/Datasets/BTCV/btcv_png_test_new_snake"
    if args.yolo_num_classes > 0:
        cfg.yolo_num_classes = args.yolo_num_classes

    dataset_name = cfg.train.dataset if args.dataset == "train" else cfg.test.dataset
    transforms = make_transforms(cfg, is_train=(args.dataset == "train"))
    dataset = make_dataset(cfg, dataset_name, transforms, is_train=(args.dataset == "train"))
    max_samples = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = make_network(cfg).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = _unwrap_network_state_dict(ckpt.get("state_dict", ckpt))
    network.load_state_dict(state_dict, strict=False)
    collator = make_collator(cfg)

    total_gt = 0
    total_pred = 0
    total_matched = 0
    matched_iou_sum = 0.0
    best_iou_sum = 0.0
    recall50 = 0
    recall75 = 0
    per_class = defaultdict(lambda: {"gt": 0, "pred": 0, "matched": 0, "recall50": 0, "recall75": 0, "matched_iou_sum": 0.0})

    for index in range(max_samples):
        sample = dataset[index]
        batch = collator([sample])
        for key, value in list(batch.items()):
            if key in ("meta", "orig_img", "img_path"):
                continue
            if torch.is_tensor(value):
                batch[key] = value.to(device)

        with torch.no_grad():
            output = network(batch["inp"], batch)

        detections = output["detection"][0].detach().cpu().numpy()
        detections = detections[detections[:, 4] > args.score_thresh]
        detections = detections[np.argsort(-detections[:, 4])]

        gt_boxes = _build_gt_boxes(sample)
        matches = _greedy_match(gt_boxes, detections)
        matched_by_gt = {m["gt_index"]: m for m in matches}

        total_gt += len(gt_boxes)
        total_pred += int(len(detections))
        total_matched += len(matches)
        matched_iou_sum += sum(m["iou"] for m in matches)

        for gt_idx, gt_box in enumerate(gt_boxes):
            cls_id = int(gt_box[4])
            per_class[cls_id]["gt"] += 1
            match = matched_by_gt.get(gt_idx)
            best_iou = float(match["iou"]) if match is not None else 0.0
            best_iou_sum += best_iou
            if best_iou >= 0.5:
                recall50 += 1
                per_class[cls_id]["recall50"] += 1
            if best_iou >= 0.75:
                recall75 += 1
                per_class[cls_id]["recall75"] += 1
            if match is not None:
                per_class[cls_id]["matched"] += 1
                per_class[cls_id]["matched_iou_sum"] += best_iou

        for pred in detections:
            per_class[int(pred[5])]["pred"] += 1

    per_class_summary = {}
    for cls_id, stats in sorted(per_class.items()):
        matched = max(stats["matched"], 1)
        gt = max(stats["gt"], 1)
        per_class_summary[str(cls_id)] = {
            "gt": int(stats["gt"]),
            "pred": int(stats["pred"]),
            "matched": int(stats["matched"]),
            "recall50": float(stats["recall50"] / gt),
            "recall75": float(stats["recall75"] / gt),
            "matched_mean_iou": float(stats["matched_iou_sum"] / matched) if stats["matched"] > 0 else 0.0,
        }

    summary = {
        "dataset": args.dataset,
        "max_samples": int(max_samples),
        "score_thresh": float(args.score_thresh),
        "num_gt": int(total_gt),
        "num_pred": int(total_pred),
        "num_matched": int(total_matched),
        "precision_like": float(total_matched / max(total_pred, 1)),
        "recall50": float(recall50 / max(total_gt, 1)),
        "recall75": float(recall75 / max(total_gt, 1)),
        "mean_best_iou": float(best_iou_sum / max(total_gt, 1)),
        "matched_mean_iou": float(matched_iou_sum / max(total_matched, 1)) if total_matched > 0 else 0.0,
        "per_class": per_class_summary,
    }

    save_path = args.save_path or str(Path(args.ckpt).resolve().parent.parent / f"{args.dataset}_eval_summary.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
