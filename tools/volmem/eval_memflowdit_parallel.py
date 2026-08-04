#!/usr/bin/env python3
"""Evaluate sequential and frozen-volume MemFlowDiT inference policies."""

import argparse
import json
import os
import pathlib
import random
import sys
import time
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
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--memory-mode",
        choices=(
            "autoregressive",
            "oracle",
            "off",
            "parallel-off",
            "frozen-causal",
            "frozen-bidirectional",
            "frozen-oracle-causal",
            "frozen-oracle-strided",
            "frozen-oracle-compact",
            "frozen-oracle-bidirectional",
            "frozen-feature-causal",
            "frozen-feature-strided",
            "frozen-feature-bidirectional",
            "frozen-key-similar",
            "frozen-shuffled",
        ),
        default="autoregressive",
    )
    parser.add_argument("--box-mode", choices=("gt", "predicted"), default="predicted")
    parser.add_argument(
        "--box-source",
        choices=("detector", "locany_cached"),
        default=None,
        help="Override cfg.box_source for this evaluation only",
    )
    parser.add_argument(
        "--locany-cache-path",
        default="",
        help="Canonical LocateAnything cache used with --box-source locany_cached",
    )
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-volumes", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--volume-start", type=int, default=0)
    parser.add_argument("--volume-end", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--memory-capacity",
        type=int,
        default=None,
        help="Evaluation-only bank capacity override.",
    )
    parser.add_argument(
        "--memory-pool-size",
        type=int,
        default=None,
        help="Evaluation-only spatial pooling override; checkpoint compatible.",
    )
    parser.add_argument(
        "--memory-global-pool-size",
        type=int,
        default=None,
        help="Compact all-history summary width; zero disables global tokens.",
    )
    parser.add_argument(
        "--parallel-batch-size",
        type=int,
        default=8,
        help="Number of independent target slices evaluated together in frozen modes.",
    )
    parser.add_argument(
        "--locate-feat-cache-root",
        default="",
        help="Optional fast local mirror of the MoonViT feature cache.",
    )
    parser.add_argument(
        "--memory-read-scale",
        type=float,
        default=1.0,
        help="Inference-only residual scale used to diagnose an under-opened Memory path.",
    )
    parser.add_argument(
        "--memory-value-position-scale",
        type=float,
        default=None,
        help=(
            "Inference-only scale for adding slice-distance encoding to Memory "
            "values; zero tests key-only positional encoding."
        ),
    )
    parser.add_argument(
        "--memory-stride",
        type=float,
        default=1.0,
        help="Spatial stride for frozen-oracle-strided, in SliceSequenceMeta units.",
    )
    parser.add_argument(
        "--conditional-moe-diagnostics",
        action="store_true",
        help="Break output-head routing down by class, scale, time, and point sector.",
    )
    parser.add_argument(
        "--final-head-cache-dir",
        default="",
        help="Optional directory for inference-distribution final-head distillation shards.",
    )
    parser.add_argument("--final-head-cache-shard-contours", type=int, default=256)
    parser.add_argument("--final-head-cache-max-contours", type=int, default=0)
    parser.add_argument("--final-head-cache-stride", type=int, default=1)
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
    build_detection_provider,
    configure_single_slice_compatibility,
    make_single_slice_dataset_class,
)
from volmem.adapters.legacy_dataset import align_mask_to_token_grid
from volmem.models import MemFlowDiTSnake, SliceSequenceMeta
from volmem.models.memory_bank import SliceMemoryBank, select_memory_states


FROZEN_MODES = {
    "parallel-off",
    "frozen-causal",
    "frozen-bidirectional",
    "frozen-oracle-causal",
    "frozen-oracle-strided",
    "frozen-oracle-compact",
    "frozen-oracle-bidirectional",
    "frozen-feature-causal",
    "frozen-feature-strided",
    "frozen-feature-bidirectional",
    "frozen-key-similar",
    "frozen-shuffled",
}


class FinalHeadCacheWriter:
    """Stream final-head inputs and teacher outputs to bounded CPU shards."""

    def __init__(self, model):
        self.output_dir = pathlib.Path(ARGS.final_head_cache_dir)
        self.shard_contours = max(int(ARGS.final_head_cache_shard_contours), 1)
        self.max_contours = max(int(ARGS.final_head_cache_max_contours), 0)
        self.stride = max(int(ARGS.final_head_cache_stride), 1)
        self.call_index = 0
        self.total_contours = 0
        self.shard_index = 0
        self.pending = []
        self.pending_contours = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = list(self.output_dir.glob("shard_*.pt"))
        if existing:
            raise RuntimeError(
                "final-head cache directory already contains shards: {}".format(
                    self.output_dir
                )
            )
        matches = [
            (name, module)
            for name, module in model.named_modules()
            if name.endswith("final_layer")
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "expected exactly one final_layer, found {}: {}".format(
                    len(matches), [name for name, _ in matches]
                )
            )
        self.module_name, module = matches[0]
        self.handle = module.register_forward_hook(self._capture)

    def _capture(self, _module, inputs, output):
        call_index = self.call_index
        self.call_index += 1
        if call_index % self.stride != 0:
            return
        if self.max_contours > 0 and self.total_contours >= self.max_contours:
            return
        if len(inputs) < 2 or not torch.is_tensor(output):
            raise RuntimeError("final-head cache requires tensor (x, t_emb) -> tensor")
        x, t_emb = inputs[:2]
        count = int(x.size(0))
        if self.max_contours > 0:
            count = min(count, self.max_contours - self.total_contours)
        if count <= 0:
            return
        self.pending.append({
            "x": x[:count].detach().to(device="cpu", dtype=torch.float16),
            "t_emb": t_emb[:count].detach().to(device="cpu", dtype=torch.float16),
            "target": output[:count].detach().to(device="cpu", dtype=torch.float16),
        })
        self.pending_contours += count
        self.total_contours += count
        if self.pending_contours >= self.shard_contours:
            self._flush()

    def _flush(self):
        if not self.pending:
            return
        payload = {
            "x": torch.cat([item["x"] for item in self.pending], dim=0),
            "t_emb": torch.cat([item["t_emb"] for item in self.pending], dim=0),
            "target": torch.cat([item["target"] for item in self.pending], dim=0),
        }
        path = self.output_dir / "shard_{:05d}.pt".format(self.shard_index)
        torch.save(payload, str(path))
        self.shard_index += 1
        self.pending = []
        self.pending_contours = 0

    def close(self):
        self._flush()
        self.handle.remove()
        manifest = {
            "format": "memflowdit-final-head-cache-v1",
            "module": self.module_name,
            "checkpoint": str(ARGS.ckpt),
            "config": str(ARGS.cfg_file),
            "seed": int(ARGS.seed),
            "memory_mode": str(ARGS.memory_mode),
            "box_mode": str(ARGS.box_mode),
            "stride": int(self.stride),
            "calls_observed": int(self.call_index),
            "contours": int(self.total_contours),
            "shards": int(self.shard_index),
        }
        path = self.output_dir / "manifest.json"
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest


def set_all_seeds(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


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
    normalized_state = OrderedDict()
    for key, value in state.items():
        normalized = str(key)
        while normalized.startswith("module."):
            normalized = normalized[len("module."):]
        if normalized in normalized_state:
            raise RuntimeError("checkpoint key collision: {}".format(normalized))
        normalized_state[normalized] = value
    clean = OrderedDict()
    incompatible = []
    adapted = []
    for normalized, value in normalized_state.items():
        if normalized in model_state and tuple(value.shape) == tuple(model_state[normalized].shape):
            clean[normalized] = value
        elif (
            normalized in {
                "memory_encoder.key_proj.weight",
                "memory_encoder.value_proj.0.weight",
            }
            and normalized in model_state
            and value.ndim == 4
            and model_state[normalized].ndim == 4
            and value.size(0) == model_state[normalized].size(0)
            and value.size(1) > model_state[normalized].size(1)
            and tuple(value.shape[2:]) == tuple(model_state[normalized].shape[2:])
        ):
            clean[normalized] = value[:, :model_state[normalized].size(1)].clone()
            adapted.append(normalized + "<-feature_channels")
        elif normalized in model_state:
            incompatible.append(normalized)
    for target_key, source_key in (
        ("memory_encoder.mask_key_proj.weight", "memory_encoder.key_proj.weight"),
        ("memory_encoder.mask_value_proj.weight", "memory_encoder.value_proj.0.weight"),
    ):
        if target_key not in model_state or target_key in clean:
            continue
        source = normalized_state.get(source_key)
        target = model_state[target_key]
        if (
            source is not None
            and source.ndim == 4
            and target.ndim == 4
            and source.size(0) == target.size(0)
            and source.size(1) >= target.size(1)
            and tuple(source.shape[2:]) == tuple(target.shape[2:])
        ):
            clean[target_key] = source[:, -target.size(1):].clone()
            adapted.append(target_key + "<-legacy_mask_channels")
    evidence_value_target = "memory_encoder.value_proj.0.weight"
    target = model_state.get(evidence_value_target)
    source_key = "memory_encoder.mask_value_proj.weight"
    source = normalized_state.get(source_key)
    if source is None:
        source_key = "memory_encoder.value_proj.0.weight"
        source = normalized_state.get(source_key)
    if (
        source is not None
        and target is not None
        and source.ndim == 4
        and target.ndim == 4
        and source.size(0) == target.size(0)
        and source.size(1) >= target.size(1)
        and tuple(source.shape[2:]) == tuple(target.shape[2:])
    ):
        clean[evidence_value_target] = source[:, -target.size(1):].clone()
        adapted.append(evidence_value_target + "<-" + source_key)
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
    if adapted:
        print("[checkpoint] adapted={}".format(",".join(adapted)), flush=True)
    return int(checkpoint.get("step", -1))


def build_model(device):
    base_network = make_network(cfg)
    slice_wrapper = _wrapper_factory(cfg, base_network)
    detection_cache, detection_policy = build_detection_provider(cfg)
    adapter = V46cContourAdapter(
            slice_wrapper,
            detection_cache=detection_cache,
            detection_policy=detection_policy,
        )
    memory_capacity = (
        int(ARGS.memory_capacity)
        if ARGS.memory_capacity is not None
        else int(cfg.volmem.memory_capacity)
    )
    memory_pool_size = (
        int(ARGS.memory_pool_size)
        if ARGS.memory_pool_size is not None
        else int(cfg.volmem.memory_pool_size)
    )
    memory_global_pool_size = (
        int(ARGS.memory_global_pool_size)
        if ARGS.memory_global_pool_size is not None
        else int(getattr(cfg.volmem, "memory_global_pool_size", 0))
    )
    if (
        memory_capacity <= 0
        or memory_pool_size <= 0
        or memory_global_pool_size < 0
    ):
        raise ValueError("memory capacity and pool size must be positive")
    distance_mode = (
        "absolute"
        if ARGS.memory_mode in {
            "frozen-bidirectional",
            "frozen-oracle-bidirectional",
            "frozen-feature-bidirectional",
            "frozen-shuffled",
        }
        else str(getattr(cfg.volmem, "relative_distance_mode", "signed"))
    )
    model = MemFlowDiTSnake(
        contour_adapter=adapter,
        feature_dim=int(cfg.locate_feat_dim),
        memory_dim=int(cfg.volmem.memory_dim),
        memory_capacity=memory_capacity,
        memory_heads=int(cfg.volmem.memory_heads),
        mask_channels=int(getattr(cfg.volmem, "mask_channels", 1)),
        memory_pool_size=memory_pool_size,
        dit_state_dim=int(cfg.dit_state_dim),
        distance_scale=float(cfg.volmem.relative_distance_scale),
        distance_mode=distance_mode,
        memory_mask_fusion_mode=str(
            getattr(cfg.volmem, "memory_mask_fusion_mode", "concat")
        ),
        memory_mask_evidence_scale=float(
            getattr(cfg.volmem, "memory_mask_evidence_scale", 0.25)
        ),
        memory_position_in_values=bool(
            getattr(cfg.volmem, "memory_position_in_values", True)
        ),
        memory_global_pool_size=memory_global_pool_size,
    ).to(device)
    model.memflow_controller.set_read_scale(ARGS.memory_read_scale)
    if ARGS.memory_value_position_scale is not None:
        model.memflow_controller.set_value_position_scale(
            ARGS.memory_value_position_scale
        )
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


def prediction_masks(output, batch, evaluator):
    batch_size = int(batch["inp"].size(0))
    predictions = evaluator._prepare_predictions(output, batch_size)
    image_paths = batch["img_path"]
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    masks = []
    for sample_index, sample_predictions in enumerate(predictions):
        image_path = image_paths[sample_index]
        record = evaluator._record_for_path(image_path)
        gt_mask = evaluator._read_mask(record["mask_path"])
        _, inv_trans, orig_hw, flipped = evaluator._sample_metadata(
            batch, sample_index, batch_size, record, gt_mask.shape
        )
        label_mask = np.zeros(gt_mask.shape, dtype=np.uint16)
        for contour, label, _ in sample_predictions:
            restored = inverse_affine_points(
                contour * float(snake_config.down_ratio),
                inv_trans,
                orig_hw,
                flipped=flipped,
            )
            polygon = np.rint(restored).astype(np.int32)
            if polygon.shape[0] >= 3:
                cv2.fillPoly(label_mask, [polygon], int(label) + 1)
        masks.append(label_mask)
    return masks


def prediction_mask(output, batch, evaluator):
    return prediction_masks(output, batch, evaluator)[0]


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


def _chunks(items, size):
    size = max(int(size), 1)
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _make_meta(volume_id, slice_index, record):
    position_unit = str(cfg.volmem.position_unit)
    if position_unit == "mm":
        value = record.get("slice_position_mm")
        if value is None or str(value).strip() == "":
            raise ValueError("mm position requires slice_position_mm in the manifest")
        slice_position = float(value)
    elif position_unit == "index":
        slice_position = float(slice_index)
    else:
        raise ValueError("position_unit must be index or mm")
    return SliceSequenceMeta(
        volume_id=volume_id,
        slice_index=int(slice_index),
        slice_position=slice_position,
        position_unit=position_unit,
        sequence_direction="ascending",
    )


def _make_batch(dataset, volume_id, item_chunk, device):
    samples = [dataset[dataset_index] for _, dataset_index in item_chunk]
    batch = move_batch(snake_collator(samples), device)
    metas = [
        _make_meta(volume_id, slice_index, dataset.records[dataset_index])
        for slice_index, dataset_index in item_chunk
    ]
    return samples, batch, metas


def _empty_banks(model, metas):
    return model.new_banks([meta.volume_id for meta in metas])


def _memory_evidence(sample, pred_label_mask, model, device, source):
    if source == "oracle":
        grid = sample["volmem_mask_grid"]
        return torch.as_tensor(
            grid, device=device, dtype=torch.float32
        ).unsqueeze(0)
    if source == "predicted":
        grid = align_mask_to_token_grid(
            pred_label_mask,
            sample,
            mask_channels=int(model.mask_channels),
        )
        return torch.as_tensor(
            grid, device=device, dtype=torch.float32
        ).unsqueeze(0)
    if source == "feature-only":
        return torch.zeros(
            (1, int(model.mask_channels), model.memory_pool_size, model.memory_pool_size),
            device=device,
            dtype=torch.float32,
        )
    raise ValueError("unsupported frozen memory evidence source: {}".format(source))


def _accumulate_volume_stats(
    volume_stats,
    volume_id,
    item_chunk,
    pred_label_masks,
    dataset,
):
    stats = volume_stats[volume_id]
    for (_, dataset_index), pred_label_mask in zip(item_chunk, pred_label_masks):
        pred_mask = pred_label_mask > 0
        gt_mask = cv2.imread(
            dataset.records[dataset_index]["mask_path"],
            cv2.IMREAD_UNCHANGED,
        ) > 0
        stats["gt"] += int(gt_mask.sum())
        stats["pred"] += int(pred_mask.sum())
        stats["intersection"] += int(np.logical_and(gt_mask, pred_mask).sum())


def run_frozen_evaluation(
    model,
    dataset,
    volumes,
    evaluator,
    conditional_collectors,
    amp_context,
    device,
):
    """Build a frozen whole-volume state table, then refine slices independently."""
    mode = ARGS.memory_mode
    if mode not in FROZEN_MODES:
        raise ValueError("run_frozen_evaluation requires a frozen memory mode")
    if ARGS.parallel_batch_size <= 0:
        raise ValueError("--parallel-batch-size must be positive")

    evidence_source = {
        "parallel-off": "none",
        "frozen-causal": "predicted",
        "frozen-bidirectional": "predicted",
        "frozen-oracle-causal": "oracle",
        "frozen-oracle-compact": "oracle",
        "frozen-oracle-strided": "oracle",
        "frozen-oracle-bidirectional": "oracle",
        "frozen-feature-causal": "feature-only",
        "frozen-feature-strided": "feature-only",
        "frozen-feature-bidirectional": "feature-only",
        "frozen-key-similar": "predicted",
        "frozen-shuffled": "predicted",
    }[mode]
    selection_policy = {
        "parallel-off": "none",
        "frozen-causal": "causal-nearest",
        "frozen-bidirectional": "bidirectional-nearest",
        "frozen-oracle-causal": "causal-nearest",
        "frozen-oracle-compact": "causal-all",
        "frozen-oracle-strided": "causal-strided",
        "frozen-oracle-bidirectional": "bidirectional-nearest",
        "frozen-feature-causal": "causal-nearest",
        "frozen-feature-strided": "causal-strided",
        "frozen-feature-bidirectional": "bidirectional-nearest",
        "frozen-key-similar": "causal-recent-key-similar",
        "frozen-shuffled": "shuffled",
    }[mode]
    requires_coarse_prediction = evidence_source == "predicted"

    processed = 0
    read_deltas = []
    state_build_seconds = 0.0
    refinement_seconds = 0.0
    volume_stats = defaultdict(lambda: {"gt": 0, "pred": 0, "intersection": 0})
    for volume_number, (volume_id, items) in enumerate(volumes, start=1):
        states = []
        if evidence_source != "none":
            built = 0
            build_start = time.perf_counter()
            for item_chunk in _chunks(items, ARGS.parallel_batch_size):
                samples, batch, metas = _make_batch(
                    dataset, volume_id, item_chunk, device
                )
                if requires_coarse_prediction:
                    with amp_context():
                        coarse_output, raw_features, _ = model.predict_step(
                            batch, metas, _empty_banks(model, metas)
                        )
                    coarse_masks = prediction_masks(coarse_output, batch, evaluator)
                else:
                    raw_features = model._raw_features(batch)
                    coarse_masks = [None] * len(samples)
                for sample, raw_feature, coarse_mask, meta in zip(
                    samples, raw_features, coarse_masks, metas
                ):
                    evidence = _memory_evidence(
                        sample,
                        coarse_mask,
                        model,
                        device,
                        evidence_source,
                    )
                    with amp_context():
                        states.append(model.memory_encoder(raw_feature, evidence, meta))
                built += len(item_chunk)
                if built % max(ARGS.log_every, 1) < len(item_chunk):
                    print(
                        "[memory-build] mode={} volumes={}/{} states={}/{}".format(
                            mode,
                            volume_number,
                            len(volumes),
                            built,
                            len(items),
                        ),
                        flush=True,
                    )
                del batch
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            state_build_seconds += time.perf_counter() - build_start

        states_by_slice = {state.slice_index: state for state in states}

        # The coarse pass consumes diffusion noise.  Reset before refinement so
        # every frozen mode and parallel-off use identical final-pass noise for
        # the same volume, making small Dice deltas causally interpretable.
        set_all_seeds(ARGS.seed + volume_number * 1_000_003)
        refine_start = time.perf_counter()
        for item_chunk in _chunks(items, ARGS.parallel_batch_size):
            _, batch, metas = _make_batch(dataset, volume_id, item_chunk, device)
            if selection_policy == "none":
                banks = _empty_banks(model, metas)
            else:
                banks = []
                for meta in metas:
                    bank = model.new_banks([volume_id])[0]
                    selection_capacity = model.memory_capacity
                    if selection_policy == "causal-all":
                        selection_capacity = max(len(states), 1)
                    selected_states = select_memory_states(
                        states,
                        meta,
                        capacity=selection_capacity,
                        policy=selection_policy,
                        seed=ARGS.seed,
                        stride=ARGS.memory_stride,
                        target_state=states_by_slice.get(meta.slice_index),
                    )
                    bank.extend(selected_states)
                    banks.append(bank)
            with amp_context():
                output, _, read_delta = model.predict_step(batch, metas, banks)
            for collector in conditional_collectors:
                collector.add_output(output)
            pred_label_masks = prediction_masks(output, batch, evaluator)
            _accumulate_volume_stats(
                volume_stats,
                volume_id,
                item_chunk,
                pred_label_masks,
                dataset,
            )
            evaluator.evaluate(output, batch)
            read_deltas.extend(
                [float(read_delta.item())] * len(item_chunk)
            )
            processed += len(item_chunk)
            if processed % max(ARGS.log_every, 1) < len(item_chunk):
                print(
                    "[eval] mode={} volumes={}/{} slices={} read_delta={:.6f}".format(
                        mode,
                        volume_number,
                        len(volumes),
                        processed,
                        read_deltas[-1],
                    ),
                    flush=True,
                )
            del batch, banks
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        refinement_seconds += time.perf_counter() - refine_start
        del states

    return {
        "processed": processed,
        "read_deltas": read_deltas,
        "volume_stats": volume_stats,
        "state_build_seconds": state_build_seconds,
        "refinement_seconds": refinement_seconds,
        "requires_coarse_prediction": requires_coarse_prediction,
        "evidence_source": evidence_source,
        "selection_policy": selection_policy,
    }


def collect_moe_diagnostics(model):
    result = {}
    for name, module in model.named_modules():
        if not hasattr(module, "routing_diagnostics"):
            continue
        diagnostics = module.routing_diagnostics()
        converted = {}
        for key, value in diagnostics.items():
            if torch.is_tensor(value):
                if value.ndim == 0:
                    converted[key] = float(value.item())
                else:
                    converted[key] = [float(item) for item in value.tolist()]
            else:
                converted[key] = value
        result[name] = converted
    return result


def _safe_distribution(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return np.zeros_like(counts)
    return counts / total


def _normalized_mutual_information(joint_counts):
    joint = np.asarray(joint_counts, dtype=np.float64)
    total = float(joint.sum())
    if total <= 0:
        return 0.0
    joint = joint / total
    p_group = joint.sum(axis=1, keepdims=True)
    p_expert = joint.sum(axis=0, keepdims=True)
    expected = p_group @ p_expert
    valid = (joint > 0) & (expected > 0)
    mutual_information = float(
        (joint[valid] * np.log(joint[valid] / expected[valid])).sum()
    )
    expert_prob = p_expert.reshape(-1)
    expert_entropy = float(
        -(expert_prob[expert_prob > 0] * np.log(expert_prob[expert_prob > 0])).sum()
    )
    return mutual_information / max(expert_entropy, 1e-12)


class ConditionalMoECollector:
    """Evaluation-only analysis of whether skew reflects real specialization."""

    def __init__(self, module_name, module):
        self.module_name = str(module_name)
        self.module = module
        self.num_experts = int(module.router_num_experts)
        self.rows = []
        self.class_observations = defaultdict(int)
        self.mismatched_events = 0
        module.enable_conditional_routing_capture(True)

    def add_output(self, output):
        events = self.module.drain_conditional_routing_events()
        classes = output.get("py_cls") if isinstance(output, dict) else None
        if not torch.is_tensor(classes) and isinstance(output, dict):
            detection = output.get("detection")
            if torch.is_tensor(detection):
                valid = detection[..., 4] > 1e-4
                classes = detection[..., 5][valid].long()
        if not torch.is_tensor(classes):
            self.mismatched_events += len(events)
            return
        class_ids = classes.detach().long().cpu().numpy().reshape(-1)
        for class_id in class_ids:
            self.class_observations[int(class_id)] += 1
        for event in events:
            top1 = event["top1_sum"].numpy()
            if top1.shape[0] != class_ids.shape[0]:
                self.mismatched_events += 1
                continue
            self.rows.append({
                "class_ids": class_ids.copy(),
                "soft_sum": event["soft_sum"].numpy(),
                "hard_sum": event["hard_sum"].numpy(),
                "top1_sum": top1,
                "point_top1": event["point_top1"].numpy(),
                "expert_delta_l2_sum": event["expert_delta_l2_sum"].numpy(),
                "expert_delta_cross": event["expert_delta_cross"].numpy(),
                "diffusion_t": event["diffusion_t"].numpy(),
                "contour_scale": event["contour_scale"].numpy(),
                "points": int(event["points"]),
                "top_k": int(event["top_k"]),
            })

    @staticmethod
    def _group_summary(keys, top1, hard, soft, delta_l2, labels):
        result = {}
        joint = []
        token_denominator = top1.sum(axis=1)
        for key, label in zip(keys, labels):
            mask = key["mask"]
            top1_counts = top1[mask].sum(axis=0)
            hard_counts = hard[mask].sum(axis=0)
            soft_counts = soft[mask].sum(axis=0)
            tokens = float(token_denominator[mask].sum())
            joint.append(top1_counts)
            result[str(label)] = {
                "rows": int(mask.sum()),
                "top1_decisions": int(round(tokens)),
                "top1_load": _safe_distribution(top1_counts).tolist(),
                "hard_load": _safe_distribution(hard_counts).tolist(),
                "soft_load": _safe_distribution(soft_counts).tolist(),
                "expert_delta_l2_mean_per_point": (
                    delta_l2[mask].sum(axis=0) / max(tokens, 1.0)
                ).tolist(),
            }
        joint = np.stack(joint) if joint else np.zeros((0, top1.shape[1]))
        return result, joint

    def summarize(self):
        if not self.rows:
            return {
                "module": self.module_name,
                "num_events": 0,
                "mismatched_events": int(self.mismatched_events),
            }

        class_ids = np.concatenate([row["class_ids"] for row in self.rows], axis=0)
        soft = np.concatenate([row["soft_sum"] for row in self.rows], axis=0)
        hard = np.concatenate([row["hard_sum"] for row in self.rows], axis=0)
        top1 = np.concatenate([row["top1_sum"] for row in self.rows], axis=0)
        delta_l2 = np.concatenate(
            [row["expert_delta_l2_sum"] for row in self.rows], axis=0
        )
        diffusion_t = np.concatenate(
            [row["diffusion_t"] for row in self.rows], axis=0
        )
        contour_scale = np.concatenate(
            [row["contour_scale"] for row in self.rows], axis=0
        )
        point_top1 = np.concatenate(
            [row["point_top1"] for row in self.rows], axis=0
        )
        delta_cross = np.concatenate(
            [row["expert_delta_cross"] for row in self.rows], axis=0
        )
        nonfinite_counts = {
            "soft_sum": int((~np.isfinite(soft)).sum()),
            "hard_sum": int((~np.isfinite(hard)).sum()),
            "top1_sum": int((~np.isfinite(top1)).sum()),
            "expert_delta_l2_sum": int((~np.isfinite(delta_l2)).sum()),
            "expert_delta_cross": int((~np.isfinite(delta_cross)).sum()),
            "diffusion_t": int((~np.isfinite(diffusion_t)).sum()),
            "contour_scale": int((~np.isfinite(contour_scale)).sum()),
        }
        soft = np.nan_to_num(soft, nan=0.0, posinf=0.0, neginf=0.0)
        hard = np.nan_to_num(hard, nan=0.0, posinf=0.0, neginf=0.0)
        top1 = np.nan_to_num(top1, nan=0.0, posinf=0.0, neginf=0.0)
        delta_l2 = np.nan_to_num(delta_l2, nan=0.0, posinf=0.0, neginf=0.0)
        point_top1 = np.nan_to_num(
            point_top1, nan=0.0, posinf=0.0, neginf=0.0
        )
        delta_cross = np.nan_to_num(
            delta_cross, nan=0.0, posinf=0.0, neginf=0.0
        )

        unique_classes = sorted(int(value) for value in np.unique(class_ids))
        class_keys = [
            {"mask": class_ids == class_id} for class_id in unique_classes
        ]
        per_class, class_joint = self._group_summary(
            class_keys, top1, hard, soft, delta_l2, unique_classes
        )

        time_edges = np.asarray([0.0, 200.0, 400.0, 600.0, 800.0, 1000.0001])
        time_ids = np.clip(np.digitize(diffusion_t, time_edges[1:-1]), 0, 4)
        time_labels = [
            "{:.0f}-{:.0f}".format(time_edges[index], time_edges[index + 1])
            for index in range(5)
        ]
        time_keys = [{"mask": time_ids == index} for index in range(5)]
        per_time, time_joint = self._group_summary(
            time_keys, top1, hard, soft, delta_l2, time_labels
        )

        finite_scale = contour_scale[np.isfinite(contour_scale)]
        if finite_scale.size:
            scale_edges = np.quantile(finite_scale, [0.25, 0.50, 0.75])
            scale_ids = np.digitize(contour_scale, scale_edges, right=True)
            scale_labels = [
                "q1<=%.4f" % scale_edges[0],
                "q2<=%.4f" % scale_edges[1],
                "q3<=%.4f" % scale_edges[2],
                "q4>%.4f" % scale_edges[2],
            ]
            scale_keys = [{"mask": scale_ids == index} for index in range(4)]
            per_scale, scale_joint = self._group_summary(
                scale_keys, top1, hard, soft, delta_l2, scale_labels
            )
        else:
            scale_edges = np.asarray([])
            per_scale = {}
            scale_joint = np.zeros((0, self.num_experts))

        point_joint = point_top1.sum(axis=0)
        per_point_sector = {
            str(index): {
                "top1_load": _safe_distribution(point_joint[index]).tolist(),
                "top1_decisions": int(round(float(point_joint[index].sum()))),
            }
            for index in range(point_joint.shape[0])
        }

        cross = delta_cross.sum(axis=0)
        diagonal = np.clip(np.diag(cross), 1e-12, None)
        cosine = cross / np.sqrt(diagonal[:, None] * diagonal[None, :])
        cosine = np.nan_to_num(cosine, nan=0.0, posinf=0.0, neginf=0.0)

        class_distributions = np.stack([
            _safe_distribution(row) for row in class_joint
        ]) if class_joint.size else np.zeros((0, self.num_experts))
        class_balanced_load = (
            class_distributions.mean(axis=0)
            if class_distributions.size
            else np.zeros(self.num_experts)
        )
        global_load = _safe_distribution(top1.sum(axis=0))

        return {
            "module": self.module_name,
            "num_events": int(len(self.rows)),
            "num_contour_event_rows": int(top1.shape[0]),
            "mismatched_events": int(self.mismatched_events),
            "nonfinite_counts": nonfinite_counts,
            "class_observations_per_slice": {
                str(key): int(value)
                for key, value in sorted(self.class_observations.items())
            },
            "global_top1_load": global_load.tolist(),
            "class_balanced_top1_load": class_balanced_load.tolist(),
            "class_balance_counterfactual_l1": float(
                np.abs(global_load - class_balanced_load).sum()
            ),
            "normalized_mi": {
                "class_expert": _normalized_mutual_information(class_joint),
                "time_expert": _normalized_mutual_information(time_joint),
                "scale_expert": _normalized_mutual_information(scale_joint),
                "point_sector_expert": _normalized_mutual_information(point_joint),
            },
            "per_class": per_class,
            "per_diffusion_time": per_time,
            "scale_quantile_edges": scale_edges.tolist(),
            "per_scale_quantile": per_scale,
            "per_point_sector": per_point_sector,
            "expert_delta_cosine_similarity": cosine.tolist(),
        }


def enable_conditional_moe_diagnostics(model):
    collectors = []
    if not ARGS.conditional_moe_diagnostics:
        return collectors
    for name, module in model.named_modules():
        if not hasattr(module, "enable_conditional_routing_capture"):
            continue
        collectors.append(ConditionalMoECollector(name, module))
    if not collectors:
        raise RuntimeError("No conditional-routing-capable MoE output head found")
    print(
        "[moe-conditional] enabled modules={}".format(
            ",".join(collector.module_name for collector in collectors)
        ),
        flush=True,
    )
    return collectors


def main():
    set_all_seeds(ARGS.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    configure_single_slice_compatibility(cfg)
    configure_box_mode(cfg, ARGS.box_mode)
    if ARGS.box_source is not None:
        cfg.box_source = ARGS.box_source
    if ARGS.locany_cache_path:
        cfg.locany_cache_path = ARGS.locany_cache_path
    if str(cfg.box_source).strip().lower() == "locany_cached" and ARGS.box_mode != "predicted":
        raise ValueError("locany_cached requires --box-mode predicted")
    if ARGS.locate_feat_cache_root:
        cfg.locate_feat_cache_root = ARGS.locate_feat_cache_root
    cfg.test.dataset = (
        "VolMemTrain" if ARGS.split == "train"
        else "VolMemVal" if ARGS.split == "val"
        else "VolMemTest"
    )
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
    final_head_cache_writer = (
        FinalHeadCacheWriter(model)
        if str(ARGS.final_head_cache_dir).strip()
        else None
    )
    conditional_collectors = enable_conditional_moe_diagnostics(model)
    evaluator = Evaluator(ARGS.result_dir)
    amp_context = (
        lambda: torch.cuda.amp.autocast(dtype=torch.float16)
        if bool(cfg.use_amp) else nullcontext()
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    evaluation_start = time.perf_counter()
    with torch.no_grad():
        if ARGS.memory_mode in FROZEN_MODES:
            run = run_frozen_evaluation(
                model,
                dataset,
                volumes,
                evaluator,
                conditional_collectors,
                amp_context,
                device,
            )
            processed = run["processed"]
            read_deltas = run["read_deltas"]
            volume_stats = run["volume_stats"]
        else:
            processed = 0
            read_deltas = []
            volume_stats = defaultdict(
                lambda: {"gt": 0, "pred": 0, "intersection": 0}
            )
            for volume_number, (volume_id, items) in enumerate(volumes, start=1):
                bank = model.new_banks([volume_id])
                for slice_index, dataset_index in items:
                    sample = dataset[dataset_index]
                    batch = move_batch(snake_collator([sample]), device)
                    meta = _make_meta(
                        volume_id,
                        slice_index,
                        dataset.records[dataset_index],
                    )
                    with amp_context():
                        output, raw_features, read_delta = model.predict_step(
                            batch, [meta], bank
                        )
                    for collector in conditional_collectors:
                        collector.add_output(output)
                    pred_label_mask = prediction_mask(output, batch, evaluator)
                    _accumulate_volume_stats(
                        volume_stats,
                        volume_id,
                        [(slice_index, dataset_index)],
                        [pred_label_mask],
                        dataset,
                    )
                    evaluator.evaluate(output, batch)
                    read_deltas.append(float(read_delta.item()))
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
            run = {
                "state_build_seconds": 0.0,
                "refinement_seconds": 0.0,
                "requires_coarse_prediction": False,
                "evidence_source": (
                    "oracle" if ARGS.memory_mode == "oracle" else "predicted"
                    if ARGS.memory_mode == "autoregressive" else "none"
                ),
                "selection_policy": (
                    "fifo" if ARGS.memory_mode != "off" else "none"
                ),
            }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_seconds = time.perf_counter() - evaluation_start
    final_head_cache_manifest = (
        final_head_cache_writer.close()
        if final_head_cache_writer is not None
        else None
    )

    summary = evaluator.summarize()
    if final_head_cache_manifest is not None:
        summary["final_head_cache"] = final_head_cache_manifest
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
        "evaluation_seed": int(ARGS.seed),
        "refinement_seed_policy": (
            "per_volume_common_noise"
            if ARGS.memory_mode in FROZEN_MODES else "global_sequential"
        ),
        "memory_mode": ARGS.memory_mode,
        "num_volumes": len(volumes),
        "mean_memory_read_delta": (
            float(np.mean(read_deltas)) if read_deltas else 0.0
        ),
        "sequence_direction": (
            "frozen-independent"
            if ARGS.memory_mode in FROZEN_MODES else "ascending"
        ),
        "memory_capacity": int(model.memory_capacity),
        "memory_pool_size": int(model.memory_pool_size),
        "memory_global_pool_size": int(model.memory_global_pool_size),
        "memory_tokens_per_full_bank": int(
            model.memory_capacity * model.memory_pool_size * model.memory_pool_size
            + (
                model.memory_global_pool_size * model.memory_global_pool_size
                if model.memory_global_pool_size > 0 else 0
            )
        ),
        "memory_distance_mode": str(model.memflow_controller.distance_mode),
        "memory_read_scale": float(ARGS.memory_read_scale),
        "memory_value_position_scale": float(
            model.memflow_controller.value_position_scale
        ),
        "memory_evidence_source": run["evidence_source"],
        "memory_mask_fusion_mode": str(model.memory_encoder.fusion_mode),
        "memory_mask_evidence_scale": float(
            model.memory_encoder.mask_evidence_scale
        ),
        "memory_selection_policy": run["selection_policy"],
        "memory_selection_stride": float(ARGS.memory_stride),
        "parallel_batch_size": (
            int(ARGS.parallel_batch_size)
            if ARGS.memory_mode in FROZEN_MODES else 1
        ),
        "requires_coarse_prediction": bool(run["requires_coarse_prediction"]),
        "effective_contour_passes": (
            2 if run["requires_coarse_prediction"] else 1
        ),
        "processed_slices": int(processed),
        "evaluation_seconds": float(evaluation_seconds),
        "state_build_seconds": float(run["state_build_seconds"]),
        "refinement_seconds": float(run["refinement_seconds"]),
        "slices_per_second": (
            float(processed) / float(evaluation_seconds)
            if evaluation_seconds > 0 else 0.0
        ),
        "peak_cuda_memory_gb": (
            float(torch.cuda.max_memory_allocated(device)) / float(1024 ** 3)
            if device.type == "cuda" else 0.0
        ),
        "moe_diagnostics": collect_moe_diagnostics(model),
        "conditional_moe_diagnostics": {
            collector.module_name: collector.summarize()
            for collector in conditional_collectors
        },
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
