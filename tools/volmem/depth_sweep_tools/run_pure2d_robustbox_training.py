#!/usr/bin/env python3
"""Fail-closed launcher for the Memory-free Route-B robustness run."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys


EXPECTED_RESUME_SHA256 = "73bbbdb4d2acd1b55517f4a65bbbf1ac0512f7663643a199dcfc88f614b28abf"
EXPECTED_SOURCE_CFG_HASH = "d0d0075c0b703e5c363ef26ca5661e538264eea285a2341a38fbab6821f879d0"
EXPECTED_TRANSITION_ID = "routeb_box_jitter_v1_from_step15500"
EXPECTED_MODEL_PARAMETERS = 14373444
EXPECTED_MAX_STEPS = 180000
EXPECTED_CHECKPOINT_MILESTONES = [
    20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000
]
EXPECTED_JITTER = {
    "probabilities": [0.35, 0.40, 0.20, 0.05],
    "shift_fractions": [0.0, 0.05, 0.10, 0.15],
    "log_scale_fractions": [0.0, 0.10, 0.20, 0.30],
    "edge_fractions": [0.0, 0.03, 0.08, 0.15],
    "min_iou": 0.20,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=("preflight", "train"), required=True)
    parser.add_argument("--preflight-steps", type=int, default=1)
    return parser.parse_args()


def write_manifest(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    worktree = Path(args.worktree).resolve()
    config = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()
    if not worktree.is_dir() or not config.is_file():
        raise FileNotFoundError({"worktree": str(worktree), "config": str(config)})
    if output_root.exists():
        raise RuntimeError(f"fresh output root required: {output_root}")
    expected_parent = worktree / "data" / "outputs" / "depth_sweep"
    if expected_parent not in output_root.parents:
        raise RuntimeError(f"output must remain inside existing worktree output root: {output_root}")

    os.chdir(worktree)
    sys.path.insert(0, str(worktree))
    sys.argv = ["diffusion_train.py", "--cfg_file", str(config)]
    os.environ["ONE_SAMPLE_OUT_DIR"] = str(output_root)

    import diffusion_train as train

    from lib.utils.snake import snake_gcn_utils

    if not bool(train.cfg.resume):
        raise RuntimeError("robust-box training requires strict resume")
    if bool(getattr(train.cfg, "resume_weights_only", False)):
        raise RuntimeError("resume_weights_only must remain false")
    if bool(getattr(train.cfg, "resume_allow_partial_copy", False)):
        raise RuntimeError("resume_allow_partial_copy must remain false")
    if int(train.cfg.pure2d_expected_parameter_count) != EXPECTED_MODEL_PARAMETERS:
        raise RuntimeError("unexpected pure-2D model parameter contract")
    if int(train.cfg.train.max_steps) != EXPECTED_MAX_STEPS:
        raise RuntimeError("robust-box formal target must remain absolute step 180000")
    checkpoint_milestones = [
        int(step) for step in train.cfg.train.step_checkpoint_milestones
    ]
    if checkpoint_milestones != EXPECTED_CHECKPOINT_MILESTONES:
        raise RuntimeError(
            f"unexpected checkpoint milestones: {checkpoint_milestones}"
        )
    if int(train.cfg.train.step_checkpoint_every) != 500:
        raise RuntimeError("step checkpoint interval must remain 500")
    if int(train.cfg.train.step_checkpoint_keep) != 12:
        raise RuntimeError("rolling checkpoint retention must remain 12")
    if str(train.cfg.diffusion_init_source).strip().lower() != "bbox_octagon":
        raise RuntimeError("robust-box training requires Route-B bbox_octagon")

    jitter_config = snake_gcn_utils.resolve_routeb_box_jitter_config(train.cfg)
    if not jitter_config["enabled"]:
        raise RuntimeError("Route-B box jitter must be enabled")
    for key, expected in EXPECTED_JITTER.items():
        actual = jitter_config[key]
        if isinstance(expected, list):
            if len(actual) != len(expected) or any(
                abs(float(a) - float(b)) > 1e-12
                for a, b in zip(actual, expected)
            ):
                raise RuntimeError(f"unexpected jitter {key}: {actual}")
        elif abs(float(actual) - float(expected)) > 1e-12:
            raise RuntimeError(f"unexpected jitter {key}: {actual}")

    resume_path = Path(str(train.cfg.resume_path))
    if not resume_path.is_absolute():
        resume_path = worktree / resume_path
    resume_path = resume_path.resolve()
    if not resume_path.is_file():
        raise FileNotFoundError(f"resume checkpoint missing: {resume_path}")
    resume_sha256 = sha256_file(resume_path)
    if resume_sha256 != EXPECTED_RESUME_SHA256:
        raise RuntimeError(
            f"resume checkpoint SHA mismatch: {resume_sha256} != {EXPECTED_RESUME_SHA256}"
        )

    import torch

    checkpoint = torch.load(str(resume_path), map_location="cpu")
    source_cfg_hash = str(checkpoint.get("cfg_hash", ""))
    resume_step = int(checkpoint.get("step", checkpoint.get("global_step", -1)))
    resume_epoch = int(checkpoint.get("epoch", -1))
    resume_step_in_epoch = int(checkpoint.get("step_in_epoch", -1))
    del checkpoint
    if resume_step != 15500:
        raise RuntimeError(f"expected resume step 15500, got {resume_step}")
    if source_cfg_hash != EXPECTED_SOURCE_CFG_HASH:
        raise RuntimeError(
            f"unexpected source cfg hash: {source_cfg_hash} != {EXPECTED_SOURCE_CFG_HASH}"
        )

    os.environ["FORMAL_RESUME_EXPECTED_SOURCE_CFG_HASH"] = EXPECTED_SOURCE_CFG_HASH
    os.environ["FORMAL_RESUME_CONFIG_TRANSITION_ID"] = EXPECTED_TRANSITION_ID

    formal_max_steps = int(train.cfg.train.max_steps)
    formal_cfg_hash = train._config_hash(train.cfg)

    if args.mode == "preflight":
        if args.preflight_steps < 1 or args.preflight_steps > 100:
            raise ValueError("preflight_steps must be in [1, 100]")
        train.cfg.train.max_steps = resume_step + int(args.preflight_steps)
        train.cfg.train.num_workers = 0
        train.cfg.train.step_checkpoint_every = 1
        train.cfg.train.step_checkpoint_keep = 1

    effective_cfg_hash = train._config_hash(train.cfg)
    output_root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "diffusionsnake.pure2d_routeb_robustbox_training_launch.v1",
        "status": "STARTED",
        "mode": args.mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "worktree": str(worktree),
        "config": str(config),
        "config_sha256": sha256_file(config),
        "formal_cfg_hash": formal_cfg_hash,
        "effective_cfg_hash": effective_cfg_hash,
        "output_root": str(output_root),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "max_steps": int(train.cfg.train.max_steps),
        "formal_max_steps": formal_max_steps,
        "checkpoint_every_steps": int(train.cfg.train.step_checkpoint_every),
        "checkpoint_keep_recent": int(train.cfg.train.step_checkpoint_keep),
        "checkpoint_milestones": checkpoint_milestones,
        "resume_checkpoint": str(resume_path),
        "resume_checkpoint_sha256": resume_sha256,
        "resume_source_cfg_hash": source_cfg_hash,
        "resume_config_transition_id": EXPECTED_TRANSITION_ID,
        "resume_epoch": resume_epoch,
        "resume_step": resume_step,
        "resume_step_in_epoch": resume_step_in_epoch,
        "routeb_box_jitter": jitter_config,
        "memory_parameters_expected": 0,
        "internal_detector_parameters_expected": 0,
        "model_parameters_expected": int(train.cfg.pure2d_expected_parameter_count),
        "loss_role": "health_monitoring_only_no_plateau_early_stop",
    }
    manifest_path = output_root / "PURE2D_TRAINING_LAUNCH.json"
    write_manifest(manifest_path, manifest)
    try:
        train.main()
    except BaseException as error:
        manifest["status"] = "FAILED"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)
        write_manifest(manifest_path, manifest)
        raise
    else:
        manifest["status"] = "COMPLETED"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
