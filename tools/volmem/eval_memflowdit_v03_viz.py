#!/usr/bin/env python3
"""Evaluate VolMemSnake checkpoints with volume-scoped sequential memory and save visualization overlays."""

import argparse
import json
import os
import pathlib
import sys
from collections import OrderedDict, defaultdict
from contextlib import nullcontext

import cv2
import numpy as np
import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--memory-mode",
        choices=("autoregressive", "oracle", "off"),
        default="autoregressive",
    )
    parser.add_argument("--box-mode", choices=("gt", "predicted"), default="predicted")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-volumes", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--volume-start", type=int, default=0)
    parser.add_argument("--volume-end", type=int, default=0)
    parser.add_argument("--save-viz", action="store_true", help="save prediction overlay PNGs")
    parser.add_argument("--viz-every", type=int, default=1)
    return parser.parse_args()


ARGS = parse_args()
os.environ["CFG_FILE"] = ARGS.cfg_file
sys.argv = [sys.argv[0], "--cfg_file", ARGS.cfg_file]

from lib.config import cfg
from lib.datasets.collate_batch import snake_collator
from lib.evaluators.sagittal_2d_fixed import Evaluator, configure_box_mode
from lib.evaluators.sagittal_2d_fixed.snake import (
    inverse_affine_points,
    rasterize_polygons,
)
from lib.networks import make_network
from lib.train.trainers.make_trainer import _wrapper_factory
from lib.utils.snake import snake_config
from volmem.adapters import (
    V46cContourAdapter,
    configure_single_slice_compatibility,
    make_single_slice_dataset_class,
)
from volmem.adapters.legacy_dataset import align_mask_to_token_grid
from volmem.models import MemFlowDiTSnake, SliceSequenceMeta


def move_batch(batch, device):
    for key, value in list(batch.items()):
        if key == "locate_feat" or str(key).startswith("locate_feat_"):
            continue
        if torch.is_tensor(value):
            batch[key] = value.to(device=device, non_blocking=True)
    batch["locate_feat"] = [
        feature.to(device=device, dtype=torch.float16, non_blocking=True)
        for feature in batch["locate_feat"]
    ]
    return batch


def load_checkpoint_strict(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict") or checkpoint
    model_state = model.state_dict()
    clean = OrderedDict()
    incompatible = []
    for key, value in state.items():
        normalized = str(key)
        while normalized.startswith("module."):
            normalized = normalized[len("module."):]
        if normalized in clean:
            raise RuntimeError("checkpoint key collision: {}".format(normalized))
        if normalized in model_state and tuple(value.shape) == tuple(model_state[normalized].shape):
            clean[normalized] = value
        elif normalized in model_state:
            incompatible.append(normalized)
    minimum_compatible = max(1, len(model_state) // 2)
    if len(clean) < minimum_compatible:
        raise RuntimeError(
            "checkpoint is not a compatible MemFlowDiT checkpoint: "
            "matched {}/{} tensors".format(len(clean), len(model_state))
        )
    load_result = model.load_state_dict(clean, strict=False)
    print(
        "[checkpoint] compatible={}/{} missing={} shape_mismatch={}".format(
            len(clean),
            len(model_state),
            len(load_result.missing_keys),
            len(incompatible),
        ),
        flush=True,
    )
    return int(checkpoint.get("step", -1))


def build_model(device):
    base_network = make_network(cfg)
    slice_wrapper = _wrapper_factory(cfg, base_network)
    adapter = V46cContourAdapter(slice_wrapper)
    model = MemFlowDiTSnake(
        contour_adapter=adapter,
        feature_dim=int(cfg.locate_feat_dim),
        memory_dim=int(cfg.volmem.memory_dim),
        memory_capacity=int(cfg.volmem.memory_capacity),
        memory_heads=int(cfg.volmem.memory_heads),
        mask_channels=int(getattr(cfg.volmem, "mask_channels", 1)),
        memory_pool_size=int(cfg.volmem.memory_pool_size),
    )
    model.to(device)
    if ARGS.random_init:
        step = -1
    else:
        step = load_checkpoint_strict(model, ARGS.ckpt)
    model.eval()
    return model, step


def group_volume_indices(records):
    volumes = defaultdict(list)
    for idx, record in enumerate(records):
        volumes[record["case_id"]].append((record["slice_idx"], idx))
    ordered = []
    for volume_id in sorted(volumes.keys()):
        items = sorted(volumes[volume_id], key=lambda x: x[0])
        ordered.append((volume_id, items))
    return ordered


def prediction_mask(output, batch, evaluator):
    predictions = evaluator._prepare_predictions(output, 1)[0]
    image_path = batch["img_path"][0] if isinstance(batch["img_path"], (list, tuple)) else batch["img_path"]
    record = evaluator._record_for_path(image_path)
    gt_mask = evaluator._read_mask(record["mask_path"])
    _, inv_trans, orig_hw, flipped = evaluator._sample_metadata(
        batch, 0, 1, record, gt_mask.shape
    )
    label_mask = np.zeros(gt_mask.shape, dtype=np.uint16)
    for contour, label, _ in predictions:
        restored = inverse_affine_points(
            contour * float(snake_config.down_ratio),
            inv_trans,
            orig_hw,
            flipped=flipped,
        )
        polygon = np.rint(restored).astype(np.int32)
        if polygon.shape[0] >= 3:
            cv2.fillPoly(label_mask, [polygon], int(label) + 1)
    return label_mask


def token_evidence(sample, pred_label_mask, device):
    if ARGS.memory_mode == "oracle":
        grid = sample["volmem_mask_grid"]
    elif ARGS.memory_mode == "autoregressive":
        grid = align_mask_to_token_grid(
            pred_label_mask,
            sample,
            mask_channels=int(getattr(cfg.volmem, "mask_channels", 1)),
        )
    else:
        raise RuntimeError("off mode must not request memory evidence")
    return torch.as_tensor(grid, device=device, dtype=torch.float32).unsqueeze(0)


def save_overlay(image_path, gt_mask, pred_mask, volume_id, slice_idx, dice, iou, out_dir):
    """Save CT + GT(green) + Pred(red) overlay."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # normalize to 8-bit grayscale / RGB
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # window to 0-255
    p1, p99 = np.percentile(gray, (1, 99))
    gray = np.clip((gray.astype(np.float32) - p1) / max(p99 - p1, 1e-6), 0, 1)
    gray = (gray * 255).astype(np.uint8)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = rgb.shape[:2]

    overlay = rgb.copy()
    # GT mask in green
    gt_color = np.array([0, 255, 0], dtype=np.uint8)
    overlay[gt_mask] = (overlay[gt_mask] * 0.5 + gt_color * 0.5).astype(np.uint8)
    # Pred mask in red
    pred_color = np.array([0, 0, 255], dtype=np.uint8)
    overlay[pred_mask] = (overlay[pred_mask] * 0.5 + pred_color * 0.5).astype(np.uint8)

    # contours
    for mask, color in [(gt_mask, (0, 255, 0)), (pred_mask, (0, 0, 255))]:
        mask_u8 = mask.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, color, 1)

    panel = np.zeros((h, w * 2, 3), dtype=np.uint8)
    panel[:, :w] = rgb
    panel[:, w:] = overlay
    # labels
    cv2.putText(panel, "CT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(panel, "GT(green) + Pred(red)", (w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    label = "{} slice{:04d} Dice={:.3f} IoU={:.3f}".format(volume_id, slice_idx, dice, iou)
    cv2.putText(panel, label, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    out_path = pathlib.Path(out_dir) / "viz_{}_slice{:04d}.png".format(volume_id, slice_idx)
    cv2.imwrite(str(out_path), panel)
    return out_path


def main():
    configure_single_slice_compatibility(cfg)
    configure_box_mode(cfg, ARGS.box_mode)
    cfg.test.dataset = "VolMemVal" if ARGS.split == "val" else "VolMemTest"
    cfg.test.batch_size = 1
    cfg.train.num_workers = 0
    cfg.result_dir = ARGS.result_dir

    os.makedirs(ARGS.result_dir, exist_ok=True)
    viz_dir = pathlib.Path(ARGS.result_dir) / "viz" if ARGS.save_viz else None
    if viz_dir is not None:
        viz_dir.mkdir(exist_ok=True)

    device = torch.device(ARGS.device)
    dataset = make_single_slice_dataset_class()(
        ann_file=str(cfg.volmem.manifest_file),
        data_root=str(cfg.volmem.data_root),
        split=ARGS.split,
    )
    volumes = group_volume_indices(dataset.records)
    model, checkpoint_step = build_model(device)
    evaluator = Evaluator(ARGS.result_dir)
    amp_context = (
        lambda: torch.cuda.amp.autocast(dtype=torch.float16)
        if bool(cfg.use_amp) else nullcontext()
    )

    processed = 0
    read_deltas = []
    volume_stats = defaultdict(lambda: {"gt": 0, "pred": 0, "intersection": 0})
    with torch.no_grad():
        for volume_number, (volume_id, items) in enumerate(volumes, start=1):
            if volume_number <= ARGS.volume_start:
                continue
            if ARGS.volume_end and volume_number > ARGS.volume_end:
                break
            if ARGS.max_volumes and volume_number > ARGS.max_volumes:
                break
            bank = model.new_banks([volume_id])
            for slice_index, dataset_index in items:
                sample = dataset[dataset_index]
                batch = move_batch(snake_collator([sample]), device)
                meta = SliceSequenceMeta(
                    volume_id=volume_id,
                    slice_index=slice_index,
                    slice_position=float(slice_index),
                    position_unit="index",
                    sequence_direction="ascending",
                )
                with amp_context():
                    output, raw_features, read_delta = model.predict_step(
                        batch, [meta], bank
                    )
                pred_label_mask = prediction_mask(output, batch, evaluator)
                pred_mask = pred_label_mask > 0
                gt_mask = cv2.imread(
                    dataset.records[dataset_index]["mask_path"],
                    cv2.IMREAD_UNCHANGED,
                ) > 0
                stats = volume_stats[volume_id]
                stats["gt"] += int(gt_mask.sum())
                stats["pred"] += int(pred_mask.sum())
                stats["intersection"] += int(np.logical_and(gt_mask, pred_mask).sum())
                evaluator.evaluate(output, batch)
                read_deltas.append(float(read_delta.item()))

                if ARGS.save_viz and processed % max(ARGS.viz_every, 1) == 0:
                    dice = (
                        float(2 * np.logical_and(gt_mask, pred_mask).sum()) /
                        max(float(gt_mask.sum() + pred_mask.sum()), 1e-6)
                    )
                    iou = (
                        float(np.logical_and(gt_mask, pred_mask).sum()) /
                        max(float(np.logical_or(gt_mask, pred_mask).sum()), 1e-6)
                    )
                    save_overlay(
                        dataset.records[dataset_index]["image_path"],
                        gt_mask, pred_mask, volume_id, slice_index,
                        dice, iou, viz_dir,
                    )

                if ARGS.memory_mode != "off":
                    evidence = token_evidence(sample, pred_label_mask, device)
                    with amp_context():
                        model.write_step(raw_features, [evidence], [meta], bank)
                    model.detach_banks(bank, keep_recent=0)
                processed += 1
                if processed % max(ARGS.log_every, 1) == 0:
                    print(
                        "[eval] mode={} volumes={}/{} slices={} read_delta={:.6f}".format(
                            ARGS.memory_mode,
                            volume_number,
                            len(volumes),
                            processed,
                            read_deltas[-1],
                        ),
                        flush=True,
                    )
            del bank

    summary = evaluator.summarize()
    per_volume = {}
    for volume_id, stats in sorted(volume_stats.items()):
        union = stats["gt"] + stats["pred"] - stats["intersection"]
        denominator = stats["gt"] + stats["pred"]
        per_volume[volume_id] = {
            "iou": float(stats["intersection"]) / float(union) if union else 1.0,
            "dice": (
                float(2 * stats["intersection"]) / float(denominator)
                if denominator else 1.0
            ),
            "gt_foreground_voxels": int(stats["gt"]),
            "pred_foreground_voxels": int(stats["pred"]),
        }
    summary.update({
        "volume_mean_iou": float(np.mean([x["iou"] for x in per_volume.values()])) if per_volume else 0.0,
        "volume_mean_dice": float(np.mean([x["dice"] for x in per_volume.values()])) if per_volume else 0.0,
        "per_volume": per_volume,
        "checkpoint": os.path.abspath(ARGS.ckpt) if ARGS.ckpt else "random_init",
        "checkpoint_step": checkpoint_step,
        "memory_mode": ARGS.memory_mode,
        "num_volumes": len(volumes),
        "mean_memory_read_delta": (
            float(np.mean(read_deltas)) if read_deltas else 0.0
        ),
        "sequence_direction": "ascending",
        "memory_capacity": int(cfg.volmem.memory_capacity),
    })
    summary_path = pathlib.Path(ARGS.result_dir) / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print("Saved summary to", summary_path)
    if viz_dir is not None:
        print("Saved visualizations to", viz_dir)


if __name__ == "__main__":
    main()
