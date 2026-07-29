#!/usr/bin/env python3
import argparse
import gc
import json
import os
import pathlib
import random
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="VolMemSnake prototype pretraining")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--chunks_per_step", type=int, default=0)
    parser.add_argument("--chunk_length", type=int, default=0)
    return parser.parse_args()


ARGS = parse_args()
os.environ["CFG_FILE"] = ARGS.cfg_file
# The inherited config module parses argv during import. Expose only its own
# supported flag and keep VolMem-specific arguments inside this entry point.
sys.argv = [sys.argv[0], "--cfg_file", ARGS.cfg_file]

from lib.config import cfg
from lib.datasets.collate_batch import snake_collator
from lib.networks import make_network
from lib.train.optimizer import make_optimizer
from lib.train.trainers.make_trainer import _wrapper_factory
from volmem.adapters import (
    V46cContourAdapter,
    configure_single_slice_compatibility,
    make_single_slice_dataset_class,
)
from volmem.data import VolumeChunkSampler
from volmem.models import SliceSequenceMeta, VolMemSnake


ALLOWED_MATURITY = {"prototype", "baseline", "validated"}


def validate_config():
    maturity = str(cfg.volmem.maturity)
    if maturity not in ALLOWED_MATURITY:
        raise RuntimeError("VolMem training requires prototype maturity or higher")
    if bool(cfg.volmem.native_3d_network):
        raise ValueError("prototype must remain slice-sequential")
    if str(cfg.volmem.position_unit) != "index":
        raise ValueError("current manifest supports position_unit=index only")
    if list(cfg.locate_feat_keys) != ["layer_18", "layer_26"]:
        raise ValueError("VolMem prototype requires MoonViT layers 18 and 26")
    if int(cfg.locate_feat_dim) != 2304:
        raise ValueError("VolMem prototype requires locate_feat_dim=2304")
    if int(cfg.locate_feat_input_layers) != 2:
        raise ValueError("VolMem prototype requires per-layer normalization")
    if str(cfg.gcn_sample_mode) != "half_pixel":
        raise ValueError("VolMem prototype requires half_pixel sampling")
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


def save_checkpoint(output_dir, model, optimizer, step):
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "maturity": "prototype",
    }
    latest = checkpoint_dir / "latest.pt"
    torch.save(payload, str(latest))
    if ARGS.save_every > 0 and step % ARGS.save_every == 0:
        torch.save(payload, str(checkpoint_dir / "step_{:06d}.pt".format(step)))
    del payload
    gc.collect()


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
    model = VolMemSnake(
        contour_adapter=contour_adapter,
        feature_dim=int(cfg.locate_feat_dim),
        memory_dim=int(cfg.volmem.memory_dim),
        memory_capacity=int(cfg.volmem.memory_capacity),
        memory_heads=int(cfg.volmem.memory_heads),
        mask_channels=1,
        memory_pool_size=int(cfg.volmem.memory_pool_size),
    ).to(device).train()

    optimizer = make_optimizer(cfg, model)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.use_amp))
    amp_context = (
        lambda: torch.cuda.amp.autocast(dtype=torch.float16)
        if bool(cfg.use_amp) else nullcontext()
    )
    gradient_accumulation = int(cfg.train.gradient_accumulation_steps)
    output_dir = PROJECT_ROOT / str(cfg.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.jsonl"
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
                loss, stats = model.forward_step(batch, metas, masks, banks)
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
            "peak_memory_gb": peak_gb,
            "time_ms": elapsed_ms,
            "volume_ids": volume_ids,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if step % max(ARGS.log_every, 1) == 0:
            print(
                "[step {}] loss={:.6f} diff={:.6f} mem={:.1f} read_delta={:.6f} peak={:.2f}GB time={:.1f}ms".format(
                    step,
                    row["loss"],
                    row["diff_loss"],
                    row["memory_size"],
                    row["memory_read_delta"],
                    row["peak_memory_gb"],
                    row["time_ms"],
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
