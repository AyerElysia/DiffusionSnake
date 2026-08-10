#!/usr/bin/env python3
"""Machine audit for the frozen Dense-6+H1 pure-2D inference path.

This is an external audit/export tool.  It does not edit repository source,
the frozen configuration, or the checkpoint.
"""

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn


LOCKED_CASES = {"sub-verse010", "sub-verse011", "sub-verse013"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--external-cache", required=True)
    parser.add_argument("--observer-source", required=True)
    parser.add_argument("--case-id", default="sub-verse016")
    parser.add_argument("--slice-idx", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_target_manifest(source_path, output_path, case_id, slice_idx):
    with open(source_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [
            row for row in reader
            if str(row.get("case_id")) == case_id
            and int(row.get("slice_idx")) == int(slice_idx)
        ]
    if len(rows) != 1:
        raise RuntimeError("expected one target row, got {}".format(len(rows)))
    joined = " ".join(str(value) for value in rows[0].values())
    if any(case in joined for case in LOCKED_CASES):
        raise RuntimeError("target-only manifest contains a locked case")
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(rows[0])
    return rows[0]


class ParameterAccessTracker:
    """Record direct parameters of modules that execute in the formal forward."""

    def __init__(self, model):
        self.handles = []
        self.touched = set()
        self.module_calls = Counter()
        for module_name, module in model.named_modules():
            direct_names = [
                "{}.{}".format(module_name, local_name) if module_name else local_name
                for local_name, _ in module.named_parameters(recurse=False)
            ]
            if not direct_names:
                continue

            def pre_hook(current_module, _inputs, names=tuple(direct_names), qualified=module_name):
                # MemoryCrossAttention is invoked as a DiT block hook even when
                # no Memory exists; its early return reads none of its weights.
                if (
                    type(current_module).__name__ == "MemoryCrossAttention"
                    and getattr(current_module, "_cached_key", None) is None
                ):
                    return
                self.touched.update(names)
                self.module_calls[qualified] += 1

            self.handles.append(module.register_forward_pre_hook(pre_hook))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class ZeroHeatmapDetector(nn.Module):
    """Parameter-free detector stub for signed GT/external-box inference only."""

    def __init__(self, feature_channels, num_classes, down_ratio):
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.num_classes = int(num_classes)
        self.down_ratio = max(int(round(float(down_ratio))), 1)

    def forward(self, x):
        height = int(math.ceil(float(x.size(2)) / float(self.down_ratio)))
        width = int(math.ceil(float(x.size(3)) / float(self.down_ratio)))
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else x.dtype
        feature = x.new_zeros((x.size(0), self.feature_channels, height, width), dtype=dtype)
        heatmap = x.new_zeros((x.size(0), self.num_classes, height, width), dtype=dtype)
        wh = x.new_zeros((x.size(0), 2, height, width), dtype=dtype)
        return feature, heatmap, wh, None


class Pure2DInferenceSnake(nn.Module):
    """Minimal interface-compatible wrapper for Memory-off 2D prediction."""

    def __init__(self, contour_adapter):
        super().__init__()
        self.contour_adapter = contour_adapter

    @staticmethod
    def _raw_features(batch):
        raw_features = batch.get("locate_feat")
        if not isinstance(raw_features, (list, tuple)):
            raise TypeError("locate_feat must be a per-slice feature list")
        return [value.unsqueeze(0) if value.dim() == 3 else value for value in raw_features]

    def new_banks(self, volume_ids):
        return [None for _ in volume_ids]

    def predict_step(self, batch, metas, banks):
        del metas, banks
        output = self.contour_adapter.predict(batch)
        raw_features = self._raw_features(batch)
        return output, raw_features, raw_features[0].new_zeros(())


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def parameter_bytes(model):
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def state_bytes(model):
    return sum(value.numel() * value.element_size() for value in model.state_dict().values())


def normalization_parameter_names(model):
    names = set()
    norm_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)
    for module_name, module in model.named_modules():
        if isinstance(module, norm_types) or "norm" in type(module).__name__.lower():
            for local_name, _ in module.named_parameters(recurse=False):
                names.add("{}.{}".format(module_name, local_name) if module_name else local_name)
    return names


def parameter_category(name, normalization_names):
    if name.startswith("memory_encoder.") or name.startswith("memflow_controller."):
        return "unused_optional_modules"
    if ".heatmap_detector." in name:
        if any(token in name for token in (".ct_head.", ".wh_head.", ".mask_head.")):
            return "detection_heads"
        return "detector_backbone_neck"
    if ".extreme_fuse." in name or ".extreme_refiner." in name:
        return "extreme_head"
    if name in normalization_names:
        return "normalization"
    if ".final_layer." in name:
        return "output_head"
    if ".dit_layers." in name:
        return "flow_dit_blocks"
    if ".locate_feat_replacer." in name or ".locate_feat_adapter." in name:
        return "input_feature_projection"
    if ".gcn." in name:
        return "flow_input_context_projections"
    return "other_or_legacy"


def collect_parameter_records(model):
    normalization_names = normalization_parameter_names(model)
    return [
        {
            "name": name,
            "numel": int(parameter.numel()),
            "trainable": bool(parameter.requires_grad),
            "element_size": int(parameter.element_size()),
            "category": parameter_category(name, normalization_names),
        }
        for name, parameter in model.named_parameters()
    ]


def parameter_census(parameter_records, touched_by_arm):
    groups = defaultdict(lambda: {
        "total_parameters": 0,
        "trainable_parameters": 0,
        "parameter_bytes": 0,
        "touched_parameters": defaultdict(int),
        "untouched_parameter_names": [],
    })
    all_touched = set().union(*touched_by_arm.values()) if touched_by_arm else set()
    for record in parameter_records:
        name = record["name"]
        group = groups[record["category"]]
        count = record["numel"]
        group["total_parameters"] += count
        group["parameter_bytes"] += count * record["element_size"]
        if record["trainable"]:
            group["trainable_parameters"] += count
        for arm, touched in touched_by_arm.items():
            if name in touched:
                group["touched_parameters"][arm] += count
        if name not in all_touched:
            group["untouched_parameter_names"].append(name)
    result = {}
    for name, values in sorted(groups.items()):
        values["touched_parameters"] = dict(sorted(values["touched_parameters"].items()))
        values["untouched_parameter_count"] = len(values["untouched_parameter_names"])
        values["untouched_parameter_names"] = values["untouched_parameter_names"][:200]
        result[name] = values
    return result


def clone_flow_record(record):
    result = {}
    for key in ("stage_init", "outer_1", "outer_2_final", "py_ind", "output_py_ind"):
        value = record.get(key)
        result[key] = value.detach().cpu().clone() if torch.is_tensor(value) else value
    return result


def compare_records(left, right):
    fields = {}
    all_exact = True
    for key in ("stage_init", "outer_1", "outer_2_final", "py_ind", "output_py_ind"):
        a, b = left.get(key), right.get(key)
        if torch.is_tensor(a) and torch.is_tensor(b):
            exact = bool(torch.equal(a, b))
            max_abs = float((a.to(torch.float64) - b.to(torch.float64)).abs().max().item()) if a.numel() else 0.0
            fields[key] = {
                "torch_equal": exact,
                "max_abs": max_abs,
                "mismatch_count": int((a != b).sum().item()),
                "baseline_sha256": tensor_sha(a),
                "slim_sha256": tensor_sha(b),
                "shape": list(a.shape),
            }
            all_exact = all_exact and exact
        else:
            fields[key] = {
                "tensor_comparison_applicable": False,
                "baseline_type": type(a).__name__,
                "slim_type": type(b).__name__,
                "note": "authoritative final comparison uses returned ret.py/py_ind",
            }
    return {"all_exact": all_exact, "fields": fields}


def set_seed(evaluator, seed=20260731):
    evaluator.set_all_seeds(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_predict(model, evaluator, observer_module, batch, meta, amp_context, capture_access=False):
    observer = observer_module.PhysicalGateObserver(
        output_dir="/tmp/pure2d_slim_observer_unused",
        box_mode=str(evaluator.ARGS.box_mode),
        expected_outer_stages=2,
        expected_inner_nfe=4,
        require_instance_contract=False,
    )
    access = ParameterAccessTracker(model) if capture_access else None
    observer.install()
    try:
        set_seed(evaluator)
        banks = model.new_banks([str(meta.volume_id)])
        with torch.no_grad():
            with amp_context():
                output, _, _ = model.predict_step(batch, [meta], banks)
        torch.cuda.synchronize(batch["inp"].device)
    finally:
        observer.uninstall()
        if access is not None:
            access.remove()
    if len(observer.flow_records) != 1:
        raise RuntimeError("expected exactly one observer Flow record")
    return output, clone_flow_record(observer.flow_records[0]), (access.touched if access else set())


def profile_predict(model, evaluator, batch, meta, amp_context, repeats):
    for _ in range(2):
        set_seed(evaluator)
        with torch.no_grad(), amp_context():
            model.predict_step(batch, [meta], model.new_banks([str(meta.volume_id)]))
    torch.cuda.synchronize(batch["inp"].device)
    torch.cuda.reset_peak_memory_stats(batch["inp"].device)
    durations = []
    for _ in range(int(repeats)):
        set_seed(evaluator)
        torch.cuda.synchronize(batch["inp"].device)
        start = time.perf_counter()
        with torch.no_grad(), amp_context():
            model.predict_step(batch, [meta], model.new_banks([str(meta.volume_id)]))
        torch.cuda.synchronize(batch["inp"].device)
        durations.append(time.perf_counter() - start)
    return {
        "warmup_runs": 2,
        "measured_repeats": int(repeats),
        "seconds": durations,
        "mean_seconds": statistics.mean(durations),
        "stdev_seconds": statistics.pstdev(durations),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(batch["inp"].device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(batch["inp"].device)),
    }


def make_slim_model(full_model, evaluator):
    controller = full_model.memflow_controller
    for handle in list(getattr(controller, "_hook_handles", [])):
        handle.remove()
    network = full_model.contour_adapter.slice_loss_wrapper.net
    wrapped_denoiser = network.gcn.denoiser
    if not hasattr(wrapped_denoiser, "base_denoiser"):
        raise RuntimeError("expected MemFlowDiT denoiser wrapper")
    network.gcn.denoiser = wrapped_denoiser.base_denoiser
    network.heatmap_detector = ZeroHeatmapDetector(
        feature_channels=int(getattr(evaluator.cfg, "heatmap_feat_channels", 256)),
        num_classes=int(network.detector_num_classes),
        down_ratio=float(network.down_ratio),
    )
    contour_adapter = full_model.contour_adapter
    full_model.contour_adapter = nn.Identity()
    slim = Pure2DInferenceSnake(contour_adapter).eval()
    return slim


def configure_arm(evaluator, arm, target_manifest, cache):
    evaluator.configure_single_slice_compatibility(evaluator.cfg)
    evaluator.configure_box_mode(evaluator.cfg, arm)
    evaluator.cfg.volmem.manifest_file = str(target_manifest)
    evaluator.cfg.test.dataset = "VolMemVal"
    evaluator.cfg.test.batch_size = 1
    evaluator.cfg.train.num_workers = 0
    evaluator.cfg.iterative_num_steps = 2
    evaluator.cfg.iterative_ode_steps = 4
    evaluator.cfg.iterative_fractions = [0.6667, 1.0]
    evaluator.cfg.v3_7_ode_solver = "ab2"
    evaluator.ARGS.box_mode = arm
    evaluator.ARGS.memory_mode = "parallel-off"
    evaluator.ARGS.parallel_batch_size = 1
    evaluator.ARGS.iterative_num_steps = 2
    evaluator.ARGS.iterative_ode_steps = 4
    evaluator.ARGS.iterative_fractions = [0.6667, 1.0]
    evaluator.ARGS.ode_solver = "ab2"
    if arm == "predicted":
        evaluator.cfg.box_source = "locany_cached"
        evaluator.cfg.locany_cache_path = str(cache)
        evaluator.ARGS.box_source = "locany_cached"
        evaluator.ARGS.locany_cache_path = str(cache)
    else:
        evaluator.cfg.box_source = "detector"
        evaluator.ARGS.box_source = "detector"
        evaluator.ARGS.locany_cache_path = ""


def build_batch_and_model(evaluator, arm, target_manifest, target_row, cache, device):
    configure_arm(evaluator, arm, target_manifest, cache)
    dataset = evaluator.make_single_slice_dataset_class()(
        ann_file=str(target_manifest),
        data_root=str(evaluator.cfg.volmem.data_root),
        split="val",
    )
    if len(dataset.records) != 1:
        raise RuntimeError("target-only dataset did not remain one row")
    sample = dataset[0]
    batch = evaluator.move_batch(evaluator.snake_collator([sample]), device)
    meta = evaluator._make_meta(
        str(target_row["case_id"]), int(target_row["slice_idx"]), dataset.records[0]
    )
    model, checkpoint_step = evaluator.build_model(device)
    return model, batch, meta, checkpoint_step


def child_parameter_count(module, name):
    child = getattr(module, name, None)
    return sum(parameter.numel() for parameter in child.parameters()) if isinstance(child, nn.Module) else 0


def candidate_counts(evaluator):
    original = {
        "dit_num_layers": int(evaluator.cfg.dit_num_layers),
        "dit_state_dim": int(evaluator.cfg.dit_state_dim),
        "dit_num_heads": int(evaluator.cfg.dit_num_heads),
        "v5_2_output_dense_residual_hidden_dim": int(evaluator.cfg.v5_2_output_dense_residual_hidden_dim),
    }
    definitions = [
        ("conservative", 6, 256, 8, 512),
        ("medium", 4, 256, 8, 256),
        ("aggressive", 4, 192, 8, 256),
    ]
    results = []
    try:
        evaluator.configure_box_mode(evaluator.cfg, "gt")
        for tier, layers, state_dim, heads, hidden in definitions:
            evaluator.cfg.dit_num_layers = layers
            evaluator.cfg.dit_state_dim = state_dim
            evaluator.cfg.dit_num_heads = heads
            evaluator.cfg.v5_2_output_dense_residual_hidden_dim = hidden
            base = evaluator.make_network(evaluator.cfg)
            detector = child_parameter_count(base, "heatmap_detector")
            flow = child_parameter_count(base, "gcn")
            locate = child_parameter_count(base, "locate_feat_replacer")
            total = parameter_count(base)
            results.append({
                "tier": tier,
                "dit_layers": layers,
                "dit_state_dim": state_dim,
                "dit_heads": heads,
                "dense_output_hidden": hidden,
                "self_contained_base_parameters": total,
                "internal_detector_parameters": detector,
                "external_detector_pure2d_parameters": total - detector,
                "flow_parameters": flow,
                "locate_feature_replacer_parameters": locate,
                "r1b_optional_total_parameters": total - detector + 26123,
            })
            del base
            gc.collect()
    finally:
        for key, value in original.items():
            setattr(evaluator.cfg, key, value)
    return results


def main():
    args = parse_args()
    if args.case_id in LOCKED_CASES:
        raise RuntimeError("locked case is forbidden")
    worktree = pathlib.Path(args.worktree).resolve()
    config = pathlib.Path(args.config).resolve()
    checkpoint = pathlib.Path(args.checkpoint).resolve()
    cache = pathlib.Path(args.external_cache).resolve()
    observer_source = pathlib.Path(args.observer_source).resolve()
    output_root = pathlib.Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError("output root already exists")
    output_root.mkdir(parents=True)
    evaluator_path = worktree / "tools/volmem/eval_memflowdit_parallel.py"
    for path in (config, checkpoint, cache, observer_source, evaluator_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(worktree))
    saved_argv = list(sys.argv)
    sys.argv = [
        str(evaluator_path), "--cfg_file", str(config), "--ckpt", str(checkpoint),
        "--split", "val", "--memory-mode", "parallel-off", "--box-mode", "predicted",
        "--box-source", "locany_cached", "--locany-cache-path", str(cache),
        "--result-dir", str(output_root / "unused_evaluator"), "--device", args.device,
        "--max-volumes", "1", "--parallel-batch-size", "1", "--seed", "20260731",
    ]
    evaluator = import_file("pure2d_slim_evaluator", evaluator_path)
    sys.argv = saved_argv
    observer_module = import_file("pure2d_slim_observer", observer_source)

    source_manifest = pathlib.Path(str(evaluator.cfg.volmem.manifest_file))
    if not source_manifest.is_absolute():
        source_manifest = worktree / source_manifest
    target_manifest = output_root / "target_only_manifest.csv"
    target_row = write_target_manifest(
        source_manifest, target_manifest, args.case_id, args.slice_idx
    )
    device = torch.device(args.device)
    amp_context = (
        lambda: torch.cuda.amp.autocast(dtype=torch.float16)
        if bool(evaluator.cfg.use_amp) else nullcontext()
    )

    arms = {}
    touched_by_arm = {}
    reference_parameter_records = None
    reference_parameter_sizes = None
    current_flow_parameters = None
    slim_checkpoint_path = output_root / "dense_h1_pure2d_slim_external_or_gt.pt"
    # The long-training source is validated against the exact GT-box path used
    # by training.  External-box deployment has a separate flow_box_only gate.
    for arm in ("gt",):
        model, batch, meta, checkpoint_step = build_batch_and_model(
            evaluator, arm, target_manifest, target_row, cache, device
        )
        if reference_parameter_records is None:
            reference_parameter_records = collect_parameter_records(model)
            reference_parameter_sizes = {
                record["name"]: record["numel"] for record in reference_parameter_records
            }
            current_flow_parameters = sum(
                record["numel"] for record in reference_parameter_records
                if ".gcn." in record["name"]
            )
        full_counts = {
            "total_parameters": parameter_count(model),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "parameter_bytes": parameter_bytes(model),
            "state_dict_bytes": state_bytes(model),
        }
        baseline_output, baseline_record, touched = run_predict(
            model, evaluator, observer_module, batch, meta, amp_context, capture_access=True
        )
        touched_by_arm[arm] = set(touched)
        baseline_profile = profile_predict(
            model, evaluator, batch, meta, amp_context, args.timing_repeats
        )
        baseline_final_sha = tensor_sha(baseline_output["py"])

        slim = make_slim_model(model, evaluator)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        slim_counts = {
            "total_parameters": parameter_count(slim),
            "trainable_parameters": sum(p.numel() for p in slim.parameters() if p.requires_grad),
            "parameter_bytes": parameter_bytes(slim),
            "state_dict_bytes": state_bytes(slim),
        }
        slim_output, slim_record, _ = run_predict(
            slim, evaluator, observer_module, batch, meta, amp_context, capture_access=False
        )
        exact = compare_records(baseline_record, slim_record)
        exact["ret_py"] = {
            "torch_equal": bool(torch.equal(baseline_output["py"], slim_output["py"])),
            "max_abs": float((baseline_output["py"].to(torch.float64) - slim_output["py"].to(torch.float64)).abs().max().item()),
            "baseline_sha256": baseline_final_sha,
            "slim_sha256": tensor_sha(slim_output["py"]),
        }
        exact["returned_py_ind"] = {
            "torch_equal": bool(torch.equal(baseline_output["py_ind"], slim_output["py_ind"])),
            "max_abs": float((baseline_output["py_ind"].to(torch.float64) - slim_output["py_ind"].to(torch.float64)).abs().max().item()) if baseline_output["py_ind"].numel() else 0.0,
            "baseline_sha256": tensor_sha(baseline_output["py_ind"]),
            "slim_sha256": tensor_sha(slim_output["py_ind"]),
        }
        exact["all_exact"] = (
            exact["all_exact"]
            and exact["ret_py"]["torch_equal"]
            and exact["returned_py_ind"]["torch_equal"]
        )
        print("[slim-exact] arm={} {}".format(arm, json.dumps(exact, sort_keys=True)), flush=True)
        if not exact["all_exact"]:
            failure_path = output_root / ("SLIM_EXACT_FAILURE_{}.json".format(arm))
            failure_path.write_text(
                json.dumps({
                    "status": "FAIL_SLIM_EXACT_{}".format(arm.upper()),
                    "arm": arm,
                    "exact": exact,
                    "full_counts": full_counts,
                    "slim_counts": slim_counts,
                }, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            raise RuntimeError("slim exact gate failed for {}".format(arm))
        slim_profile = profile_predict(
            slim, evaluator, batch, meta, amp_context, args.timing_repeats
        )
        # Persist the exported model for whichever validated arm runs first.
        # The pure-2D training handoff deliberately audits the GT-box arm only,
        # so tying export to the historical predicted-box arm leaves the
        # otherwise successful exact audit without its packaged checkpoint.
        if not slim_checkpoint_path.exists():
            cpu_state = {name: value.detach().cpu() for name, value in slim.state_dict().items()}
            torch.save({
                "state_dict": cpu_state,
                "format": "dense_h1_pure2d_inference_slim_v1",
                "source_checkpoint_sha256": sha256_file(checkpoint),
                "supported_box_modes": ["gt", "external_detector_cache"],
                "memory": "removed",
                "internal_heatmap_detector": "replaced_by_parameter_free_zero_stub",
            }, slim_checkpoint_path)
        arms[arm] = {
            "checkpoint_step": int(checkpoint_step),
            "full": full_counts,
            "slim": slim_counts,
            "removed_parameters": full_counts["total_parameters"] - slim_counts["total_parameters"],
            "removed_fraction": float(full_counts["total_parameters"] - slim_counts["total_parameters"]) / float(full_counts["total_parameters"]),
            "exact": exact,
            "full_profile": baseline_profile,
            "slim_profile": slim_profile,
            "speedup_full_over_slim": baseline_profile["mean_seconds"] / slim_profile["mean_seconds"],
            "touched_parameter_names": sorted(touched),
            "touched_parameters": sum(reference_parameter_sizes[name] for name in touched),
        }
        del baseline_output, slim_output, baseline_record, slim_record, slim, batch
        gc.collect()
        torch.cuda.empty_cache()

    census = parameter_census(reference_parameter_records, touched_by_arm)
    candidate_table = candidate_counts(evaluator)
    for row in candidate_table:
        row["estimated_flow_parameter_ratio_vs_current"] = row["flow_parameters"] / float(current_flow_parameters)
        row["estimated_8nfe_dit_compute_ratio_vs_current"] = row["estimated_flow_parameter_ratio_vs_current"]

    full_checkpoint_bytes = checkpoint.stat().st_size
    slim_checkpoint_bytes = slim_checkpoint_path.stat().st_size
    payload = {
        "schema": "diffusionsnake.pure2d_dense_h1_slim_audit.v1",
        "status": "PASS_PURE2D_SLIM_EXACT_GT_TRAINING_PROTOCOL",
        "scope": "pure 2D only; Memory/PCAA/Oracle/Global Sparse/3D abandoned and absent",
        "identity": {
            "worktree": str(worktree),
            "worktree_head": subprocess.check_output(
                ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True
            ).strip(),
            "config": str(config),
            "config_sha256": sha256_file(config),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_file_bytes": full_checkpoint_bytes,
            "case_id": args.case_id,
            "slice_idx": args.slice_idx,
            "locked_cases_opened": 0,
            "seed": 20260731,
            "schedule": "2 outer x 4 inner AB2 = 8 NFE",
        },
        "external_detector": {
            "parameters_in_this_model": 0,
            "note": "External detector cost is outside the Dense-H1 model and must be reported separately.",
        },
        "arms": arms,
        "parameter_census": census,
        "slim_export": {
            "path": str(slim_checkpoint_path),
            "sha256": sha256_file(slim_checkpoint_path),
            "file_bytes": slim_checkpoint_bytes,
            "file_reduction_bytes": full_checkpoint_bytes - slim_checkpoint_bytes,
            "file_reduction_fraction": float(full_checkpoint_bytes - slim_checkpoint_bytes) / float(full_checkpoint_bytes),
            "loader_contract": "Pure2DInferenceSnake + ZeroHeatmapDetector; only GT/external signed boxes; no internal detector or Memory.",
        },
        "retraining_candidates": candidate_table,
        "r1b": {
            "optional_additional_parameters": 26123,
            "inference_overhead": "unknown_not_profiled_in_this_frozen_H1_audit",
            "boundary": "separate post-training plug-in; not part of recommended base architecture census",
        },
        "structural_zero_parameter_options": {
            "extreme_head": 0,
            "legacy_moe": 0,
            "pcaa": 0,
            "rl": 0,
            "sam_prev_contour": 0,
            "auxiliary_loss_observer": 0,
        },
        "boundaries": [
            "Slim exactness is established for the one nonlocked signed slice in GT and external-detector protocols; it is not a quality re-evaluation.",
            "Timing is a same-process light B1 GPU microprofile, not a full-volume E2E benchmark.",
            "Candidate parameter/FLOP ratios are architectural estimates and require retraining; no candidate quality is claimed.",
            "A self-contained internal-detector model retains detector parameters; the sub-30M result relies on GT or a separately accounted external detector.",
        ],
    }
    json_path = output_root / "PURE2D_DENSE_H1_SLIM_AUDIT_20260807.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    artifact_path = output_root / "SHA256SUMS.txt"
    artifacts = [target_manifest, slim_checkpoint_path, json_path]
    artifact_path.write_text(
        "".join("{}  {}\n".format(sha256_file(path), path) for path in artifacts),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "json": str(json_path),
        "json_sha256": sha256_file(json_path),
        "slim_checkpoint": str(slim_checkpoint_path),
        "slim_checkpoint_sha256": sha256_file(slim_checkpoint_path),
        "artifact_manifest": str(artifact_path),
        "artifact_manifest_sha256": sha256_file(artifact_path),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
