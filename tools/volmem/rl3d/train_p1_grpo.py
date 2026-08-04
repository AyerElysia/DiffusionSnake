#!/usr/bin/env python3
"""Train the frozen-base VolMem P1 Fourier policy with GRPO-style updates."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from p1_core import (
    FourierPolicy,
    apply_fourier_action,
    delta_nsd_burr_reward,
    gaussian_log_prob,
    oracle_coefficients,
    score_contours,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", action="append", required=True,
        help="Contour cache .npz. Repeat for multi-volume training; when "
             "--holdout-mode=last-cache, the final cache is evaluation-only.")
    parser.add_argument(
        "--holdout-mode", choices=("labels", "last-cache"), default="labels",
        help="Use the preregistered label split or reserve the final cache as "
             "a completely unseen-volume holdout.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--adv-std-floor", type=float, default=0.10)
    parser.add_argument("--adv-clip", type=float, default=2.0)
    parser.add_argument("--max-displacement", type=float, default=3.0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@dataclass
class EvalRow:
    count: int
    reward_mean: float
    nsd_delta: float
    mean_distance_before: float
    mean_distance_after: float
    mean_distance_reduction_frac: float
    burr_delta: float
    wins: int
    ties: int
    losses: int
    saturation_fraction: float


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_cache(path: str):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    required = {
        "poly", "gt_target", "gt_poly", "gt_dist", "normal", "point_feat",
        "slice_idx", "label", "orig_hw", "n_gt_boundary",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise RuntimeError("cache is missing fields: {}".format(missing))
    if arrays["poly"].shape[1:] != (128, 2):
        raise RuntimeError("P1 requires 128-point contours")
    valid = (
        np.isfinite(arrays["poly"]).all(axis=(1, 2))
        & np.isfinite(arrays["gt_target"]).all(axis=(1, 2))
        & np.isfinite(arrays["gt_poly"]).all(axis=(1, 2))
        & np.isfinite(arrays["point_feat"]).all(axis=(1, 2))
        & (arrays["n_gt_boundary"] > 0)
        & (np.nanmedian(arrays["gt_dist"], axis=1) <= 6.0)
    )
    arrays = {key: value[valid] for key, value in arrays.items()}
    if arrays["poly"].shape[0] < 20:
        raise RuntimeError("too few valid in-scope contours for P1")
    return arrays


def load_caches(paths):
    """Load compatible per-volume caches without mixing their split identity."""
    loaded = [load_cache(path) for path in paths]
    reference = loaded[0]
    required = tuple(reference.keys())
    for path, arrays in zip(paths[1:], loaded[1:]):
        missing = sorted(set(required) - set(arrays))
        if missing:
            raise RuntimeError("cache {} is missing fields: {}".format(path, missing))
        for key in required:
            if arrays[key].shape[1:] != reference[key].shape[1:]:
                raise RuntimeError(
                    "cache {} has incompatible {} shape {} vs {}".format(
                        path, key, arrays[key].shape[1:], reference[key].shape[1:]))
    arrays = {
        key: np.concatenate([item[key] for item in loaded], axis=0)
        for key in required
    }
    source_counts = [int(item["poly"].shape[0]) for item in loaded]
    arrays["cache_index"] = np.concatenate([
        np.full(count, index, dtype=np.int32)
        for index, count in enumerate(source_counts)
    ])
    return arrays, source_counts


def split_by_track(labels: np.ndarray):
    unique = np.asarray(sorted(int(x) for x in np.unique(labels)), dtype=np.int64)
    if unique.size < 2:
        raise RuntimeError("P1 track split requires at least two vertebra labels")
    holdout_labels = unique[::5]
    if holdout_labels.size == 0:
        holdout_labels = unique[-1:]
    holdout = np.isin(labels, holdout_labels)
    if holdout.all():
        holdout = labels == unique[-1]
        holdout_labels = unique[-1:]
    return np.flatnonzero(~holdout), np.flatnonzero(holdout), holdout_labels


def split_by_last_cache(cache_index: np.ndarray, cache_count: int):
    if cache_count < 2:
        raise RuntimeError("last-cache holdout requires at least two --cache inputs")
    holdout = cache_index == cache_count - 1
    train_indices = np.flatnonzero(~holdout)
    holdout_indices = np.flatnonzero(holdout)
    if train_indices.size < 20 or holdout_indices.size < 20:
        raise RuntimeError("last-cache split has too few valid contours")
    return train_indices, holdout_indices


def as_tensors(arrays, indices, device):
    return {
        "point_feat": torch.as_tensor(
            arrays["point_feat"][indices], device=device, dtype=torch.float32),
        "poly": torch.as_tensor(
            arrays["poly"][indices], device=device, dtype=torch.float32),
        "normal": torch.as_tensor(
            arrays["normal"][indices], device=device, dtype=torch.float32),
        "label": torch.as_tensor(
            arrays["label"][indices], device=device, dtype=torch.long),
    }


@torch.no_grad()
def evaluate(policy, arrays, indices, base_scores, device, max_displacement):
    tensors = as_tensors(arrays, indices, device)
    coefficients = policy(
        tensors["point_feat"], tensors["poly"], tensors["normal"], tensors["label"])
    refined, field = apply_fourier_action(
        tensors["poly"], tensors["normal"], coefficients,
        policy.basis, max_displacement=max_displacement)
    scores = score_contours(
        refined.cpu().numpy(), arrays["gt_poly"][indices], arrays["orig_hw"][indices])
    base = {key: value[indices] for key, value in base_scores.items()}
    nsd_delta = scores["nsd"] - base["nsd"]
    # Match the strongest 2D run exactly: delta NSD against the deterministic
    # rollout, followed by an absolute burr penalty on the sampled endpoint.
    reward = delta_nsd_burr_reward(scores, base["nsd"])
    before = float(base["mean_distance"].mean())
    after = float(scores["mean_distance"].mean())
    tolerance = 1e-7
    return EvalRow(
        count=int(len(indices)),
        reward_mean=float(reward.mean()),
        nsd_delta=float(nsd_delta.mean()),
        mean_distance_before=before,
        mean_distance_after=after,
        mean_distance_reduction_frac=float((before - after) / max(before, 1e-8)),
        burr_delta=float((scores["burr"] - base["burr"]).mean()),
        wins=int((nsd_delta > tolerance).sum()),
        ties=int((np.abs(nsd_delta) <= tolerance).sum()),
        losses=int((nsd_delta < -tolerance).sum()),
        saturation_fraction=float((field.abs() >= 0.95 * max_displacement).float().mean().item()),
    )


def main():
    args = parse_args()
    if args.rollouts != 8:
        raise ValueError("P1 is preregistered with G=8")
    set_seeds(args.seed)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays, cache_counts = load_caches(args.cache)
    if args.holdout_mode == "last-cache":
        train_indices, holdout_indices = split_by_last_cache(
            arrays["cache_index"], len(args.cache))
        holdout_labels = np.unique(arrays["label"][holdout_indices])
    else:
        train_indices, holdout_indices, holdout_labels = split_by_track(arrays["label"])
    device = torch.device(args.device)

    feature_dim = int(arrays["point_feat"].shape[-1])
    policy = FourierPolicy(feature_dim=feature_dim).to(device)
    train_features = torch.as_tensor(
        arrays["point_feat"][train_indices], dtype=torch.float32)
    feature_mean = train_features.mean(dim=(0, 1))
    feature_std = train_features.std(dim=(0, 1), unbiased=False).clamp_min(1e-4)
    policy.set_feature_stats(feature_mean, feature_std)
    del train_features

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=0.0)
    base_scores = score_contours(arrays["poly"], arrays["gt_poly"], arrays["orig_hw"])
    initial_train = evaluate(
        policy, arrays, train_indices, base_scores, device, args.max_displacement)
    initial_holdout = evaluate(
        policy, arrays, holdout_indices, base_scores, device, args.max_displacement)
    if (
        abs(initial_train.nsd_delta) > 1e-7
        or abs(initial_train.mean_distance_after - initial_train.mean_distance_before) > 1e-7
    ):
        raise RuntimeError("zero-initialized policy does not reproduce the frozen base")

    # Diagnostic upper bound under the deployed action transform.
    all_poly = torch.as_tensor(arrays["poly"], device=device, dtype=torch.float32)
    all_target = torch.as_tensor(arrays["gt_target"], device=device, dtype=torch.float32)
    all_normal = torch.as_tensor(arrays["normal"], device=device, dtype=torch.float32)
    with torch.no_grad():
        oracle_coef = oracle_coefficients(all_poly, all_target, all_normal, policy.basis)
        oracle_poly, oracle_field = apply_fourier_action(
            all_poly, all_normal, oracle_coef, policy.basis, args.max_displacement)
    oracle_scores = score_contours(
        oracle_poly.cpu().numpy(), arrays["gt_poly"], arrays["orig_hw"])
    oracle_summary = {
        "reward_mean": float(delta_nsd_burr_reward(
            oracle_scores, base_scores["nsd"]).mean()),
        "nsd_delta": float((oracle_scores["nsd"] - base_scores["nsd"]).mean()),
        "mean_distance_reduction_frac": float(
            (base_scores["mean_distance"].mean() - oracle_scores["mean_distance"].mean())
            / max(float(base_scores["mean_distance"].mean()), 1e-8)),
        "saturation_fraction": float(
            (oracle_field.abs() >= 0.95 * args.max_displacement).float().mean().item()),
    }

    manifest = {
        "args": vars(args),
        "caches": [os.path.abspath(path) for path in args.cache],
        "cache_valid_contours": cache_counts,
        "holdout_mode": args.holdout_mode,
        "valid_contours": int(arrays["poly"].shape[0]),
        "train_contours": int(train_indices.size),
        "holdout_contours": int(holdout_indices.size),
        "holdout_labels": holdout_labels.tolist(),
        "initial_train": asdict(initial_train),
        "initial_holdout": asdict(initial_holdout),
        "oracle": oracle_summary,
    }
    with open(output_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    rng = np.random.default_rng(args.seed)
    log_path = output_dir / "train.jsonl"
    best_holdout = -float("inf")
    start_time = time.time()
    for step in range(1, args.steps + 1):
        batch_indices = rng.choice(
            train_indices, size=args.batch_size,
            replace=args.batch_size > train_indices.size)
        batch = as_tensors(arrays, batch_indices, device)
        mean = policy(batch["point_feat"], batch["poly"], batch["normal"], batch["label"])
        with torch.no_grad():
            noise = torch.randn(
                args.rollouts, args.batch_size, policy.action_dim, device=device)
            sampled_action = mean.detach().unsqueeze(0) + args.sigma * noise
            sampled_poly, sampled_field = apply_fourier_action(
                batch["poly"], batch["normal"], sampled_action,
                policy.basis, args.max_displacement)
            deterministic_poly, _ = apply_fourier_action(
                batch["poly"], batch["normal"], mean.detach(),
                policy.basis, args.max_displacement)

            flat_sampled = sampled_poly.reshape(-1, 128, 2).cpu().numpy()
            repeated_gt = np.tile(arrays["gt_poly"][batch_indices], (args.rollouts, 1, 1))
            repeated_hw = np.tile(arrays["orig_hw"][batch_indices], (args.rollouts, 1))
            sampled_scores = score_contours(flat_sampled, repeated_gt, repeated_hw)
            deterministic_scores = score_contours(
                deterministic_poly.cpu().numpy(), arrays["gt_poly"][batch_indices],
                arrays["orig_hw"][batch_indices])
            sampled_reward_np = delta_nsd_burr_reward(
                {
                    "nsd": sampled_scores["nsd"].reshape(
                        args.rollouts, args.batch_size),
                    "burr": sampled_scores["burr"].reshape(
                        args.rollouts, args.batch_size),
                },
                deterministic_scores["nsd"][None, :],
            )
            sampled_reward = torch.as_tensor(sampled_reward_np, device=device)
            reward_std = sampled_reward.std(dim=0, unbiased=False)
            advantage = sampled_reward / reward_std.clamp_min(args.adv_std_floor)
            advantage = advantage.clamp(-args.adv_clip, args.adv_clip)
            old_mean = mean.detach()
            old_log_prob = gaussian_log_prob(
                sampled_action, old_mean.unsqueeze(0), args.sigma)

        losses = []
        clip_fractions = []
        approx_kls = []
        for _ in range(args.ppo_epochs):
            new_mean = policy(
                batch["point_feat"], batch["poly"], batch["normal"], batch["label"])
            new_log_prob = gaussian_log_prob(
                sampled_action, new_mean.unsqueeze(0), args.sigma)
            log_ratio = new_log_prob - old_log_prob
            ratio = torch.exp(log_ratio)
            unclipped = ratio * advantage
            clipped = ratio.clamp(1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * advantage
            loss = -torch.minimum(unclipped, clipped).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            clip_fractions.append(float((ratio.detach().sub(1.0).abs() > args.clip_ratio).float().mean().item()))
            approx_kls.append(float((-log_ratio.detach()).mean().item()))

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            train_eval = evaluate(
                policy, arrays, train_indices, base_scores, device, args.max_displacement)
            holdout_eval = evaluate(
                policy, arrays, holdout_indices, base_scores, device, args.max_displacement)
            row = {
                "step": step,
                "elapsed_seconds": time.time() - start_time,
                "loss": float(np.mean(losses)),
                "grad_norm": float(grad_norm),
                "clip_fraction": float(np.mean(clip_fractions)),
                "approx_kl": float(np.mean(approx_kls)),
                "reward_mean": float(sampled_reward.mean().item()),
                "reward_std": float(sampled_reward.std(unbiased=False).item()),
                "adv_abs_mean": float(advantage.abs().mean().item()),
                "sample_saturation_fraction": float(
                    (sampled_field.abs() >= 0.95 * args.max_displacement).float().mean().item()),
                "train": asdict(train_eval),
                "holdout": asdict(holdout_eval),
            }
            with open(log_path, "a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print("[P1] " + json.dumps(row, sort_keys=True), flush=True)
            checkpoint = {
                "step": step,
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "train_eval": asdict(train_eval),
                "holdout_eval": asdict(holdout_eval),
                "oracle": oracle_summary,
            }
            torch.save(checkpoint, output_dir / "latest.pt")
            if holdout_eval.nsd_delta > best_holdout:
                best_holdout = holdout_eval.nsd_delta
                torch.save(checkpoint, output_dir / "best_holdout.pt")

    final_train = evaluate(
        policy, arrays, train_indices, base_scores, device, args.max_displacement)
    final_holdout = evaluate(
        policy, arrays, holdout_indices, base_scores, device, args.max_displacement)
    passed = (
        final_train.mean_distance_reduction_frac >= 0.10
        and final_train.nsd_delta > 0.0
        and final_holdout.nsd_delta > 0.0
        and final_train.saturation_fraction < 0.10
    )
    summary = {
        "passed": bool(passed),
        "train": asdict(final_train),
        "holdout": asdict(final_holdout),
        "oracle": oracle_summary,
        "steps": args.steps,
        "seed": args.seed,
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("[P1][DONE] " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
