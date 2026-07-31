#!/usr/bin/env python3
"""Evaluate VolMemSnake checkpoints with volume-scoped sequential memory."""

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
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--volume-start", type=int, default=0)
    parser.add_argument("--volume-end", type=int, default=0)
    return parser.parse_args()


ARGS = parse_args()
os.environ["CFG_FILE"] = ARGS.cfg_file
# The inherited config module parses argv during import.
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
    clean = OrderedDict()
    for key, value in state.items():
        normalized = str(key)
        while normalized.startswith("module."):
            normalized = normalized[len("module."):]
        if normalized in clean:
            raise RuntimeError("checkpoint key collision: {}".format(normalized))
        clean[normalized] = value
    model.load_state_dict(clean, strict=False)
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
        mask_channels=1,
        memory_pool_size=int(cfg.volmem.memory_pool_size),
        dit_state_dim=int(cfg.dit_state_dim),
        distance_scale=float(cfg.volmem.relative_distance_scale),
    ).to(device)
    if ARGS.random_init or not ARGS.ckpt:
        print("[model] RANDOM INIT (no checkpoint)", flush=True)
        step = -1
    else:
        step = load_checkpoint_strict(model, ARGS.ckpt)
    return model.eval(), step


def group_volume_indices(records):
    grouped = defaultdict(list)
    for dataset_index, record in enumerate(records):
        grouped[str(record["case_id"])].append(
            (int(record["slice_idx"]), dataset_index)
        )
    ordered = []
    for volume_id in sorted(grouped):
        items = sorted(grouped[volume_id])
        slice_indices = [item[0] for item in items]
        if len(slice_indices) != len(set(slice_indices)):
            raise ValueError("duplicate slice index in volume {}".format(volume_id))
        if any(right != left + 1 for left, right in zip(slice_indices[:-1], slice_indices[1:])):
            raise ValueError("non-consecutive slice indices in volume {}".format(volume_id))
        ordered.append((volume_id, items))
    if ARGS.max_volumes is not None:
        if ARGS.max_volumes <= 0:
            raise ValueError("--max-volumes must be positive")
        ordered = ordered[:ARGS.max_volumes]
    if ARGS.volume_start > 0 or ARGS.volume_end > 0:
        end = ARGS.volume_end if ARGS.volume_end > 0 else len(ordered)
        ordered = ordered[ARGS.volume_start:end]
    return ordered


def prediction_mask(output, batch, evaluator):
    predictions = evaluator._prepare_predictions(output, 1)[0]
    image_path = batch["img_path"][0] if isinstance(batch["img_path"], (list, tuple)) else batch["img_path"]
    record = evaluator._record_for_path(image_path)
    gt_mask = evaluator._read_mask(record["mask_path"])
    _, inv_trans, orig_hw, flipped = evaluator._sample_metadata(
        batch, 0, 1, record, gt_mask.shape
    )
    contours = []
    for contour, _, _ in predictions:
        restored = inverse_affine_points(
            contour * float(snake_config.down_ratio),
            inv_trans,
            orig_hw,
            flipped=flipped,
        )
        contours.append(restored)
    return (rasterize_polygons(contours, gt_mask.shape) > 0).astype(np.float32)


def token_evidence(sample, pred_mask, device):
    if ARGS.memory_mode == "oracle":
        grid = sample["volmem_mask_grid"]
    elif ARGS.memory_mode == "autoregressive":
        grid = align_mask_to_token_grid(pred_mask, sample)
    else:
        raise RuntimeError("off mode must not request memory evidence")
    return torch.as_tensor(grid, device=device, dtype=torch.float32).unsqueeze(0)


def main():
    configure_single_slice_compatibility(cfg)
    configure_box_mode(cfg, ARGS.box_mode)
    cfg.test.dataset = "VolMemVal" if ARGS.split == "val" else "VolMemTest"
    cfg.test.batch_size = 1
    cfg.train.num_workers = 0
    cfg.result_dir = ARGS.result_dir

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
                pred_mask = prediction_mask(output, batch, evaluator).astype(bool)
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
                if ARGS.memory_mode != "off":
                    evidence = token_evidence(sample, pred_mask, device)
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
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return summary


if __name__ == "__main__":
    main()
