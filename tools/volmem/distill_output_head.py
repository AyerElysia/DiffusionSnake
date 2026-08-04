#!/usr/bin/env python3
"""Distill the mature v0.5 output MoE into efficient H1/H2 heads."""

import argparse
import collections
import copy
import json
import math
import os
import pathlib
import random
import sys
import time

import torch
import torch.nn.functional as F


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--teacher-ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--cfg_file",
        default="configs/volmem/verse_memflowdit_v0_5_minimal_gpu6.yaml",
        help="Configuration used only to initialize the repository module namespace.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--h1-steps", type=int, default=5000)
    parser.add_argument("--h2-steps", type=int, default=3000)
    parser.add_argument("--h2-num-experts", type=int, default=4)
    parser.add_argument("--h2-balance-bias-step", type=float, default=1e-3)
    parser.add_argument("--h2-balance-bias-limit", type=float, default=0.10)
    parser.add_argument("--batch-contours", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-validation-contours", type=int, default=2048)
    return parser.parse_args()


class ShardPool:
    def __init__(self, paths, cache_size=6):
        self.paths = list(paths)
        self.cache_size = int(max(cache_size, 1))
        self.cache = collections.OrderedDict()
        self._sample_path = None
        self._sample_uses = 0

    def load(self, path):
        path = pathlib.Path(path)
        key = str(path)
        if key in self.cache:
            payload = self.cache.pop(key)
            self.cache[key] = payload
            return payload
        payload = torch.load(str(path), map_location="cpu")
        required = {"x", "t_emb", "target"}
        if set(payload) != required:
            raise RuntimeError("invalid cache shard {}: {}".format(path, sorted(payload)))
        lengths = {int(payload[name].size(0)) for name in required}
        if len(lengths) != 1:
            raise RuntimeError("cache shard has inconsistent contour counts: {}".format(path))
        self.cache[key] = payload
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return payload

    def sample(self, batch_size, rng):
        # Reuse each loaded shard for several optimizer steps.  This keeps the
        # randomized stream while avoiding multi-gigabyte cache I/O per epoch.
        if self._sample_path is None or self._sample_uses >= 8:
            self._sample_path = rng.choice(self.paths)
            self._sample_uses = 0
        payload = self.load(self._sample_path)
        self._sample_uses += 1
        count = int(payload["x"].size(0))
        indices = torch.randint(0, count, (int(batch_size),))
        return tuple(payload[name][indices] for name in ("x", "t_emb", "target"))


def _head_prefix(state):
    suffix = ".final_layer.norm.weight"
    matches = [key[:-len("norm.weight")] for key in state if key.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError("expected one final-head prefix, found {}".format(matches))
    return matches[0]


def _teacher_head_state(state, prefix):
    return {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }


@torch.no_grad()
def initialize_h1_from_teacher(head, teacher):
    """Transfer the legacy shared path exactly; distill only routed behavior."""
    for name in (
        "norm.weight",
        "linear.weight",
        "linear.bias",
        "adaLN.1.weight",
        "adaLN.1.bias",
    ):
        target = head.state_dict()[name]
        source = teacher[name]
        if tuple(target.shape) != tuple(source.shape):
            raise RuntimeError("shape mismatch for {}".format(name))
        target.copy_(source.to(target))

    first_weight = teacher["shared_mlp.0.weight"]
    first_bias = teacher["shared_mlp.0.bias"]
    second_weight = teacher["shared_mlp.2.weight"]
    second_bias = teacher["shared_mlp.2.bias"]
    shared_width = int(first_weight.size(0))
    if shared_width > head.residual_mlp[0].out_features:
        raise RuntimeError("H1 hidden dimension cannot hold teacher shared MLP")
    head.residual_mlp[0].weight[:shared_width].copy_(first_weight.to(head.residual_mlp[0].weight))
    head.residual_mlp[0].bias[:shared_width].copy_(first_bias.to(head.residual_mlp[0].bias))
    projected_weight = teacher["linear.weight"] @ second_weight
    projected_bias = teacher["linear.weight"] @ second_bias
    head.residual_mlp[-1].weight.zero_()
    head.residual_mlp[-1].bias.copy_(projected_bias.to(head.residual_mlp[-1].bias))
    head.residual_mlp[-1].weight[:, :shared_width].copy_(
        projected_weight.to(head.residual_mlp[-1].weight)
    )


def _metrics(head, pool, paths, device, max_contours):
    head.eval()
    squared = 0.0
    absolute = 0.0
    maximum = 0.0
    target_squared = 0.0
    cosine_sum = 0.0
    vectors = 0
    elements = 0
    contours = 0
    with torch.no_grad():
        for path in paths:
            payload = pool.load(path)
            remaining = int(max_contours) - contours if max_contours > 0 else None
            count = int(payload["x"].size(0))
            if remaining is not None:
                count = min(count, remaining)
            if count <= 0:
                break
            for start in range(0, count, 64):
                end = min(start + 64, count)
                x = payload["x"][start:end].to(device=device, dtype=torch.float32)
                t_emb = payload["t_emb"][start:end].to(device=device, dtype=torch.float32)
                target = payload["target"][start:end].to(device=device, dtype=torch.float32)
                pred = head(x, t_emb)
                error = pred - target
                squared += float(error.square().sum().item())
                absolute += float(error.abs().sum().item())
                maximum = max(maximum, float(error.abs().max().item()))
                target_squared += float(target.square().sum().item())
                cosine_sum += float(F.cosine_similarity(
                    pred.reshape(-1, 2), target.reshape(-1, 2), dim=-1
                ).sum().item())
                vectors += int(target.numel() // 2)
                elements += int(target.numel())
                contours += int(end - start)
    return {
        "contours": contours,
        "rmse": math.sqrt(squared / max(elements, 1)),
        "mae": absolute / max(elements, 1),
        "max_abs": maximum,
        "relative_rmse": math.sqrt(squared / max(target_squared, 1e-20)),
        "mean_vector_cosine": cosine_sum / max(vectors, 1),
    }


def _train_head(
    name,
    head,
    trainable,
    train_pool,
    train_paths,
    val_pool,
    val_paths,
    steps,
    args,
    output_dir,
):
    device = torch.device(args.device)
    head.to(device=device, dtype=torch.float32)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    trainable_parameters = []
    for parameter in trainable(head):
        parameter.requires_grad_(True)
        trainable_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(args.learning_rate),
        weight_decay=1e-5,
    )
    rng = random.Random(args.seed + (1 if name == "h1" else 2))
    initial = _metrics(
        head, val_pool, val_paths, device, args.max_validation_contours
    )
    best = dict(initial)
    best_step = 0
    best_state = copy.deepcopy(head.state_dict())
    log_path = output_dir / "{}_distill.jsonl".format(name)
    started = time.time()
    for step in range(1, int(steps) + 1):
        head.train()
        x, t_emb, target = train_pool.sample(args.batch_contours, rng)
        x = x.to(device=device, dtype=torch.float32)
        t_emb = t_emb.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        pred = head(x, t_emb)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        should_validate = step % max(args.validate_every, 1) == 0 or step == steps
        if should_validate:
            metrics = _metrics(
                head, val_pool, val_paths, device, args.max_validation_contours
            )
            row = {
                "step": step,
                "train_mse": float(loss.detach().item()),
                "elapsed_seconds": time.time() - started,
                **metrics,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if metrics["rmse"] < best["rmse"]:
                best = dict(metrics)
                best_step = step
                best_state = copy.deepcopy(head.state_dict())
        if step % max(args.log_every, 1) == 0:
            print(
                "[{} step {}] mse={:.8f} best_rmse={:.8f}@{}".format(
                    name, step, float(loss.detach().item()), best["rmse"], best_step
                ),
                flush=True,
            )
    head.load_state_dict(best_state)
    reset_diagnostics = getattr(head, "reset_routing_diagnostics", None)
    if callable(reset_diagnostics):
        reset_diagnostics()
    final = _metrics(head, val_pool, val_paths, device, args.max_validation_contours)
    return {
        "initial": initial,
        "best": best,
        "best_step": best_step,
        "final": final,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "elapsed_seconds": time.time() - started,
    }


def _transplant_checkpoint(teacher_payload, teacher_state, prefix, head, metadata, path):
    state = {
        key: value.detach().cpu().clone()
        for key, value in teacher_state.items()
        if not key.startswith(prefix)
    }
    for name, value in head.state_dict().items():
        state[prefix + name] = value.detach().cpu().clone()
    payload = {
        "state_dict": state,
        "step": int(teacher_payload.get("step", -1)),
        "maturity": teacher_payload.get("maturity", "prototype"),
        "output_head_distillation": metadata,
    }
    torch.save(payload, str(path))
    return sum(tensor.numel() for tensor in state.values())


def main():
    args = parse_args()
    os.environ["CFG_FILE"] = str(args.cfg_file)
    sys.argv = [sys.argv[0], "--cfg_file", str(args.cfg_file)]
    global DenseResidualFinalHead, SharedDenseSparseResidualHead
    from lib.networks.diffusion.dit_denoiser_v4 import (
        DenseResidualFinalHead,
        SharedDenseSparseResidualHead,
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = pathlib.Path(args.cache_dir)
    shard_paths = sorted(cache_dir.glob("shard_*.pt"))
    if len(shard_paths) < 2:
        raise RuntimeError("distillation requires at least two cache shards")
    split = max(1, int(round(len(shard_paths) * 0.85)))
    split = min(split, len(shard_paths) - 1)
    train_paths = shard_paths[:split]
    val_paths = shard_paths[split:]
    train_pool = ShardPool(train_paths)
    val_pool = ShardPool(val_paths)

    teacher_payload = torch.load(args.teacher_ckpt, map_location="cpu")
    teacher_state = teacher_payload.get("state_dict") or teacher_payload.get("model")
    if teacher_state is None:
        raise RuntimeError("teacher checkpoint has no state_dict")
    prefix = _head_prefix(teacher_state)
    teacher_head = _teacher_head_state(teacher_state, prefix)

    h1 = DenseResidualFinalHead(dim=256, out_dim=2, hidden_dim=1024)
    initialize_h1_from_teacher(h1, teacher_head)
    h1_result = _train_head(
        "h1",
        h1,
        lambda module: module.residual_mlp.parameters(),
        train_pool,
        train_paths,
        val_pool,
        val_paths,
        args.h1_steps,
        args,
        output_dir,
    )
    torch.save(h1.state_dict(), str(output_dir / "h1_head.pt"))

    h2 = SharedDenseSparseResidualHead(
        dim=256,
        out_dim=2,
        shared_hidden_dim=1024,
        num_experts=args.h2_num_experts,
        expert_hidden_dim=128,
        router_temperature=0.50,
        load_ema_decay=0.99,
        balance_bias_step=args.h2_balance_bias_step,
        balance_bias_limit=args.h2_balance_bias_limit,
    )
    h2.load_state_dict(h1.state_dict(), strict=False)
    h2_result = _train_head(
        "h2",
        h2,
        lambda module: list(module.router.parameters()) + list(module.experts.parameters()),
        train_pool,
        train_paths,
        val_pool,
        val_paths,
        args.h2_steps,
        args,
        output_dir,
    )
    torch.save(h2.state_dict(), str(output_dir / "h2_head.pt"))

    old_head_params = sum(value.numel() for value in teacher_head.values())
    h1_head_params = sum(value.numel() for value in h1.parameters())
    h2_head_params = sum(value.numel() for value in h2.parameters())
    common_metadata = {
        "teacher_checkpoint": str(args.teacher_ckpt),
        "teacher_step": int(teacher_payload.get("step", -1)),
        "cache_dir": str(cache_dir),
        "seed": int(args.seed),
    }
    h2_label = "H2_shared_sparse_E{}_top1".format(args.h2_num_experts)
    h1_total = _transplant_checkpoint(
        teacher_payload,
        teacher_state,
        prefix,
        h1,
        {**common_metadata, "variant": "H1_dense_residual", "metrics": h1_result},
        output_dir / "h1_distilled_full.pt",
    )
    h2_total = _transplant_checkpoint(
        teacher_payload,
        teacher_state,
        prefix,
        h2,
        {**common_metadata, "variant": h2_label, "metrics": h2_result},
        output_dir / "h2_distilled_full.pt",
    )
    teacher_total = sum(value.numel() for value in teacher_state.values())
    report = {
        "teacher_checkpoint": str(args.teacher_ckpt),
        "cache": json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8")),
        "split": {
            "train_shards": len(train_paths),
            "validation_shards": len(val_paths),
        },
        "head_parameters": {
            "H0_legacy_E8_top2": old_head_params,
            "H1_dense_residual": h1_head_params,
            h2_label: h2_head_params,
        },
        # This includes persistent buffers because it is derived directly from
        # checkpoint state.  Formal trainable-parameter counts are measured by
        # audit_moe_cost.py on the constructed models after distillation.
        "full_checkpoint_tensors": {
            "H0_legacy_E8_top2": teacher_total,
            "H1_dense_residual": h1_total,
            h2_label: h2_total,
        },
        "H1": h1_result,
        "H2": h2_result,
        "H2_routing": {
            key: value.tolist() if torch.is_tensor(value) and value.ndim else (
                float(value.item()) if torch.is_tensor(value) else value
            )
            for key, value in h2.routing_diagnostics().items()
        },
    }
    (output_dir / "distillation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
