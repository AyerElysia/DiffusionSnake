#!/usr/bin/env python3
"""Evaluate VolMemSnake checkpoints with volume-scoped sequential memory."""

import argparse
import json
import os
import pathlib
import random
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
        "--conditional-moe-diagnostics",
        action="store_true",
        help="Break output-head routing down by class, scale, time, and point sector.",
    )
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
    detection_cache, detection_policy = build_detection_provider(cfg)
    adapter = V46cContourAdapter(
            slice_wrapper,
            detection_cache=detection_cache,
            detection_policy=detection_policy,
        )
    model = MemFlowDiTSnake(
        contour_adapter=adapter,
        feature_dim=int(cfg.locate_feat_dim),
        memory_dim=int(cfg.volmem.memory_dim),
        memory_capacity=int(cfg.volmem.memory_capacity),
        memory_heads=int(cfg.volmem.memory_heads),
        mask_channels=int(getattr(cfg.volmem, "mask_channels", 1)),
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
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed_all(ARGS.seed)
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
    conditional_collectors = enable_conditional_moe_diagnostics(model)
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
                for collector in conditional_collectors:
                    collector.add_output(output)
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
        "evaluation_seed": int(ARGS.seed),
        "memory_mode": ARGS.memory_mode,
        "num_volumes": len(volumes),
        "mean_memory_read_delta": (
            float(np.mean(read_deltas)) if read_deltas else 0.0
        ),
        "sequence_direction": "ascending",
        "memory_capacity": int(cfg.volmem.memory_capacity),
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
