#!/usr/bin/env python3
import argparse
import atexit
import gc
import hashlib
import json
import os
import pathlib
import random
import subprocess
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="MemFlowDiT sequential pretraining")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--chunks_per_step", type=int, default=0)
    parser.add_argument("--chunk_length", type=int, default=0)
    parser.add_argument(
        "--output-dir-override",
        default="",
        help="Use a separate run directory for smoke tests.",
    )
    parser.add_argument(
        "--prediction-evidence-prob-override",
        type=float,
        default=-1.0,
        help="Test-only override; negative values use the configured schedule.",
    )
    return parser.parse_args()


ARGS = parse_args()
os.environ["CFG_FILE"] = ARGS.cfg_file
# The inherited config module parses argv during import. Expose only its own
# supported flag and keep VolMem-specific arguments inside this entry point.
sys.argv = [sys.argv[0], "--cfg_file", ARGS.cfg_file]

from lib.config import cfg
from lib.datasets.collate_batch import snake_collator
from lib.networks import make_network
from lib.train.trainers.make_trainer import _wrapper_factory
from volmem.adapters import (
    V46cContourAdapter,
    configure_single_slice_compatibility,
    make_single_slice_dataset_class,
)
from volmem.data import VolumeChunkSampler
from volmem.engine.run_guard import RunDirectoryLock
from volmem.models import MemFlowDiTSnake, SliceSequenceMeta


ALLOWED_MATURITY = {"prototype", "baseline", "validated"}


def validate_config():
    maturity = str(cfg.volmem.maturity)
    if maturity not in ALLOWED_MATURITY:
        raise RuntimeError("VolMem training requires prototype maturity or higher")
    if bool(cfg.volmem.native_3d_network):
        raise ValueError("prototype must remain slice-sequential")
    if str(cfg.volmem.position_unit) != "index":
        raise ValueError("current manifest supports position_unit=index only")
    feature_keys = list(cfg.locate_feat_keys)
    if feature_keys not in (["layer_18"], ["layer_18", "layer_26"]):
        raise ValueError("VolMem supports MoonViT layer_18, optionally with layer_26")
    expected_layers = len(feature_keys)
    if int(cfg.locate_feat_dim) != 1152 * expected_layers:
        raise ValueError("locate_feat_dim must match the selected MoonViT layers")
    if int(cfg.locate_feat_input_layers) != expected_layers:
        raise ValueError("locate_feat_input_layers must match locate_feat_keys")
    if str(cfg.gcn_sample_mode) != "half_pixel":
        raise ValueError("VolMem prototype requires half_pixel sampling")
    if str(cfg.gcn_sample_padding_mode) != "border":
        raise ValueError("MemFlowDiT full-mask training requires border point sampling")
    if int(cfg.volmem.mask_channels) != 26:
        raise ValueError("MemFlowDiT full-mask training requires 26 class-aware mask channels")
    if int(cfg.volmem.truncated_bptt_steps) != 2:
        raise ValueError("the first prototype requires truncated_bptt_steps=2")


def move_batch(batch, device):
    for key, value in list(batch.items()):
        if key == "locate_feat" or str(key).startswith("locate_feat_"):
            continue
        if torch.is_tensor(value):
            batch[key] = value.to(device=device, non_blocking=True)
    features = []
    for feature in batch["locate_feat"]:
        features.append(feature.to(device=device, dtype=torch.float16, non_blocking=True))
    batch["locate_feat"] = features
    return batch


def load_initial_weights(module, checkpoint_path):
    obj = torch.load(checkpoint_path, map_location="cpu")
    state = obj.get("state_dict") or obj.get("model") or obj.get("net") or obj
    current = module.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    adapted = []

    # A prototype-Phi MoE layer contains no inactive dense FFN.  When starting
    # from the proven dense checkpoint, clone each dense SwiGLU tensor into all
    # routed experts here, before optimizer construction.  New MoE checkpoints
    # already contain the expert keys and therefore bypass this bridge.
    expert_marker = ".prototype_phi_moe.experts."
    for target_key, target_value in current.items():
        if expert_marker not in target_key or target_key in compatible:
            continue
        prefix, expert_suffix = target_key.split(expert_marker, 1)
        expert_parts = expert_suffix.split(".", 1)
        if len(expert_parts) != 2:
            continue
        source_key = "{}.mlp.{}".format(prefix, expert_parts[1])
        source_value = state.get(source_key)
        if (
            source_value is not None
            and tuple(source_value.shape) == tuple(target_value.shape)
        ):
            compatible[target_key] = source_value
            adapted.append("{}<-{}".format(target_key, source_key))

    projection_key = "locate_feat_replacer.proj.0.weight"
    if projection_key in state and projection_key in current:
        source = state[projection_key]
        target = current[projection_key]
        if (
            source.dim() == 4
            and target.dim() == 4
            and source.size(0) == target.size(0)
            and source.shape[2:] == target.shape[2:]
            and target.size(1) == source.size(1) * 2
        ):
            # Start the dual-layer adapter as the proven layer-18 model:
            # [W_l18, 0_l26].  Layer 26 is introduced continuously by training
            # instead of randomising the entire 2304-channel entrance.
            expanded = torch.zeros_like(target)
            expanded[:, :source.size(1)].copy_(source.to(expanded))
            compatible[projection_key] = expanded
            adapted.append(
                "{}:{}->{}".format(
                    projection_key,
                    tuple(source.shape),
                    tuple(target.shape),
                )
            )
    info = module.load_state_dict(compatible, strict=False)
    if len(compatible) < int(len(state) * 0.80):
        raise RuntimeError(
            "checkpoint compatibility below 80%: {}/{}".format(
                len(compatible), len(state)
            )
        )
    print(
        "[init] checkpoint={} compatible={}/{} missing={} unexpected={}".format(
            checkpoint_path,
            len(compatible),
            len(state),
            len(info.missing_keys),
            len(info.unexpected_keys),
        ),
        flush=True,
    )
    if adapted:
        print(
            "[init] adapted_count={} sample={}".format(
                len(adapted), ",".join(adapted[:3])
            ),
            flush=True,
        )
    if projection_key not in compatible:
        raise RuntimeError(
            "MoonViT entrance projection was neither loaded nor adapted"
        )


def build_optimizer(model):
    if str(cfg.train.optim).strip().lower() != "adamw":
        raise ValueError("MemFlowDiT formal training currently requires AdamW")
    base_lr = float(cfg.train.lr)
    weight_decay = float(cfg.train.weight_decay)
    locate_mult = float(getattr(cfg.train, "locate_lr_multiplier", 1.0))
    memory_mult = float(getattr(cfg.train, "memory_lr_multiplier", 1.0))
    detail_mult = float(getattr(cfg.train, "detail_lr_multiplier", 1.0))
    groups = []
    counts = {"base": 0, "locate": 0, "memory": 0, "detail": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group_name = "base"
        multiplier = 1.0
        if name.startswith("memory_encoder.") or name.startswith("memflow_controller."):
            group_name, multiplier = "memory", memory_mult
        elif "locate_feat_replacer." in name:
            group_name, multiplier = "locate", locate_mult
        elif "detail_local_proj." in name or "detail_point_proj." in name:
            group_name, multiplier = "detail", detail_mult
        counts[group_name] += parameter.numel()
        groups.append({
            "params": [parameter],
            "lr": base_lr * multiplier,
            "target_lr": base_lr * multiplier,
            "weight_decay": weight_decay,
        })
    print(
        "[optim] base_lr={} multipliers=locate:{} memory:{} detail:{} params={}".format(
            base_lr,
            locate_mult,
            memory_mult,
            detail_mult,
            counts,
        ),
        flush=True,
    )
    return torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay)


def apply_warmup(optimizer, step):
    warmup_steps = max(int(getattr(cfg.train, "warmup_steps", 0)), 0)
    factor = (
        min(max(float(step) / float(warmup_steps), 0.0), 1.0)
        if warmup_steps > 0
        else 1.0
    )
    for group in optimizer.param_groups:
        group["lr"] = float(group.get("target_lr", group["lr"])) * factor
    return factor


def prediction_evidence_probability(step):
    if ARGS.prediction_evidence_prob_override >= 0.0:
        return min(max(float(ARGS.prediction_evidence_prob_override), 0.0), 1.0)
    start_step = int(getattr(cfg.volmem, "prediction_evidence_start_step", 0))
    ramp_steps = max(
        int(getattr(cfg.volmem, "prediction_evidence_ramp_steps", 1)),
        1,
    )
    maximum = min(
        max(float(getattr(cfg.volmem, "prediction_evidence_max_prob", 0.0)), 0.0),
        1.0,
    )
    if step < start_step:
        return 0.0
    progress = min(max(float(step - start_step) / float(ramp_steps), 0.0), 1.0)
    return maximum * progress


def save_checkpoint(output_dir, model, optimizer, step):
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "maturity": str(cfg.volmem.maturity),
    }
    latest = checkpoint_dir / "latest.pt"
    torch.save(payload, str(latest))
    if ARGS.save_every > 0 and step % ARGS.save_every == 0:
        torch.save(payload, str(checkpoint_dir / "step_{:06d}.pt".format(step)))
    del payload
    gc.collect()


def collect_moe_routing_diagnostics(model):
    result = {}
    for name, module in model.named_modules():
        diagnostics_fn = getattr(module, "routing_diagnostics", None)
        if not callable(diagnostics_fn):
            continue
        diagnostics = diagnostics_fn()
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


def main():
    configure_single_slice_compatibility(cfg)
    validate_config()
    random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed_all(ARGS.seed)
    device = torch.device(ARGS.device)

    chunks_per_step = (
        ARGS.chunks_per_step
        if ARGS.chunks_per_step > 0 else int(cfg.volmem.chunks_per_step)
    )
    chunk_length = (
        ARGS.chunk_length
        if ARGS.chunk_length > 0 else int(cfg.volmem.chunk_length)
    )
    if chunk_length % int(cfg.volmem.truncated_bptt_steps) != 0:
        raise ValueError("chunk_length must be divisible by truncated_bptt_steps")

    dataset_class = make_single_slice_dataset_class()
    dataset = dataset_class(
        ann_file=str(cfg.volmem.manifest_file),
        data_root=str(cfg.volmem.data_root),
        split="train",
    )
    chunk_sampler = VolumeChunkSampler(
        records=dataset.records,
        chunk_length=chunk_length,
        chunks_per_step=chunks_per_step,
        seed=ARGS.seed,
        steps_per_epoch=ARGS.max_steps,
    )

    base_network = make_network(cfg)
    load_initial_weights(base_network, str(cfg.resume_path))
    slice_wrapper = _wrapper_factory(cfg, base_network)
    contour_adapter = V46cContourAdapter(slice_wrapper)
    model = MemFlowDiTSnake(
        contour_adapter=contour_adapter,
        feature_dim=int(cfg.locate_feat_dim),
        memory_dim=int(cfg.volmem.memory_dim),
        memory_capacity=int(cfg.volmem.memory_capacity),
        memory_heads=int(cfg.volmem.memory_heads),
        mask_channels=int(cfg.volmem.mask_channels),
        memory_pool_size=int(cfg.volmem.memory_pool_size),
        dit_state_dim=int(cfg.dit_state_dim),
        distance_scale=float(cfg.volmem.relative_distance_scale),
    ).to(device).train()

    optimizer = build_optimizer(model)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.use_amp))
    amp_context = (
        lambda: torch.cuda.amp.autocast(dtype=torch.float16)
        if bool(cfg.use_amp) else nullcontext()
    )
    gradient_accumulation = int(cfg.train.gradient_accumulation_steps)
    output_dir = (
        PROJECT_ROOT / str(ARGS.output_dir_override)
        if str(ARGS.output_dir_override).strip()
        else PROJECT_ROOT / str(cfg.model_dir)
    )
    lock_experiment_id = (
        output_dir.name
        if str(ARGS.output_dir_override).strip()
        else str(cfg.model)
    )
    run_lock = RunDirectoryLock(output_dir, lock_experiment_id).acquire()
    atexit.register(run_lock.release)
    log_path = output_dir / "train.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    try:
        source_branch = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "branch", "--show-current"],
            text=True,
        ).strip()
        source_commit = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_branch, source_commit = "unknown", "unknown"
    runtime_paths = [
        "tools/volmem/train_memflowdit.py",
        "volmem/models/memflow_dit.py",
        "volmem/models/memflow_snake.py",
        "volmem/models/slice_memory.py",
        "lib/networks/diffusion/flow_matching_evolution.py",
        "lib/networks/diffusion/dit_denoiser_v4_1.py",
        "lib/networks/diffusion/dit_blocks_v3.py",
        "lib/networks/snake/ct_snake.py",
        "lib/datasets/sagittal_2d_fixed/snake.py",
        "lib/train/trainers/diffusion_trainer.py",
        "volmem/adapters/legacy_dataset.py",
        "volmem/adapters/v4_6c.py",
    ]
    runtime_file_hashes = {}
    for relative_path in runtime_paths:
        source_path = PROJECT_ROOT / relative_path
        runtime_file_hashes[relative_path] = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
    try:
        tracked_status = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        tracked_status = "unknown"
    run_manifest = {
        "experiment_id": lock_experiment_id,
        "configured_experiment_id": str(cfg.model),
        "method": "MemFlowDiT",
        "parent_experiment": "volmem-v0.2-prototype",
        "mechanism": "flow_dit_block_memory_cross_attention",
        "config": str(ARGS.cfg_file),
        "output_dir": str(output_dir),
        "source_branch": source_branch,
        "source_commit": source_commit,
        "source_tracked_dirty": bool(tracked_status.strip()),
        "runtime_file_hashes": runtime_file_hashes,
        "seed": int(ARGS.seed),
        "max_steps": int(ARGS.max_steps),
        "chunks_per_step": int(chunks_per_step),
        "chunk_length": int(chunk_length),
        "gradient_accumulation_steps": int(gradient_accumulation),
        "memory_mask_channels": int(cfg.volmem.mask_channels),
        "prediction_evidence_max_prob": float(cfg.volmem.prediction_evidence_max_prob),
    }
    manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + chr(10),
        encoding="utf-8",
    )
    optimizer.zero_grad(set_to_none=True)

    print(
        "[start] device={} chunks={} length={} micro_slices={} grad_accum={} effective_slices={}".format(
            device,
            chunks_per_step,
            chunk_length,
            chunks_per_step * chunk_length,
            gradient_accumulation,
            chunks_per_step * chunk_length * gradient_accumulation,
        ),
        flush=True,
    )

    for step, windows in enumerate(chunk_sampler, start=1):
        started = time.time()
        volume_ids = [window[0] for window in windows]
        banks = model.new_banks(volume_ids)
        step_loss_value = 0.0
        final_stats = {}
        pred_evidence_prob = prediction_evidence_probability(step)

        segment_losses = []
        tbptt_steps = int(cfg.volmem.truncated_bptt_steps)
        for offset in range(chunk_length):
            samples = [dataset[window[2][offset]] for window in windows]
            batch = move_batch(snake_collator(samples), device)
            metas = [
                SliceSequenceMeta(
                    volume_id=window[0],
                    slice_index=int(window[1][offset]),
                    slice_position=float(window[1][offset]),
                    position_unit="index",
                    sequence_direction="ascending",
                )
                for window in windows
            ]
            masks = [
                torch.as_tensor(
                    samples[item_index]["volmem_mask_grid"],
                    device=device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                for item_index in range(len(windows))
            ]
            with amp_context():
                loss, stats = model.forward_step(
                    batch,
                    metas,
                    masks,
                    banks,
                    prediction_evidence_probability=pred_evidence_prob,
                )
            segment_losses.append(loss)
            step_loss_value += float(loss.detach().item()) / float(chunk_length)
            final_stats = stats
            segment_boundary = (
                len(segment_losses) == tbptt_steps
                or offset + 1 == chunk_length
            )
            if segment_boundary:
                segment_loss = torch.stack(segment_losses).sum() / float(
                    chunk_length * gradient_accumulation
                )
                scaler.scale(segment_loss).backward()
                segment_losses = []
                model.detach_banks(banks, keep_recent=0)
                del segment_loss
            del batch, samples, masks, loss

        if step % gradient_accumulation == 0:
            warmup_factor = apply_warmup(optimizer, step)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg.train.gradient_clip),
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        peak_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
        elapsed_ms = (time.time() - started) * 1000.0
        row = {
            "step": step,
            "loss": step_loss_value,
            "diff_loss": float(final_stats.get("diff_loss", torch.tensor(0.0)).item()),
            "memory_size": float(final_stats["volmem_memory_size"].item()),
            "memory_read_delta": float(final_stats["volmem_memory_read_delta"].item()),
            "memflow_active_states": float(final_stats["memflow_active_states"].item()),
            "prediction_evidence_probability": pred_evidence_prob,
            "prediction_evidence_fraction": float(
                final_stats["volmem_prediction_evidence_fraction"].item()
            ),
            "warmup_factor": (
                warmup_factor
                if step % gradient_accumulation == 0
                else apply_warmup(optimizer, step)
            ),
            "peak_memory_gb": peak_gb,
            "time_ms": elapsed_ms,
            "volume_ids": volume_ids,
        }
        moe_diagnostics = collect_moe_routing_diagnostics(model)
        row["moe_routing"] = moe_diagnostics
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if step % max(ARGS.log_every, 1) == 0:
            print(
                "[step {}] loss={:.6f} diff={:.6f} mem={:.1f} active={:.0f} read_delta={:.6f} pred_mem={:.2f}/{:.2f} peak={:.2f}GB time={:.1f}ms".format(
                    step,
                    row["loss"],
                    row["diff_loss"],
                    row["memory_size"],
                    row["memflow_active_states"],
                    row["memory_read_delta"],
                    row["prediction_evidence_fraction"],
                    row["prediction_evidence_probability"],
                    row["peak_memory_gb"],
                    row["time_ms"],
                ),
                flush=True,
            )
        if moe_diagnostics and step % 10 == 0:
            entropies = [
                item["normalized_entropy"]
                for item in moe_diagnostics.values()
            ]
            cvs = [item["hard_cv"] for item in moe_diagnostics.values()]
            dead = [
                int(item["dead_experts_lt_1pct"])
                for item in moe_diagnostics.values()
            ]
            print(
                "[moe] entropy={:.4f} hard_cv={:.4f} dead_max={} layers={}".format(
                    sum(entropies) / len(entropies),
                    sum(cvs) / len(cvs),
                    max(dead),
                    len(moe_diagnostics),
                ),
                flush=True,
            )
        optimizer_updated = step % gradient_accumulation == 0
        periodic_checkpoint = (
            ARGS.save_every > 0
            and step % ARGS.save_every == 0
        )
        final_checkpoint = step == ARGS.max_steps
        if optimizer_updated and (periodic_checkpoint or final_checkpoint):
            save_checkpoint(output_dir, model, optimizer, step)


if __name__ == "__main__":
    main()
