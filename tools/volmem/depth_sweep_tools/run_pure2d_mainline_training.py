#!/usr/bin/env python3
"""Stable launcher for the Memory-free pure-2D mainline training run."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys


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

    output_root.mkdir(parents=True, exist_ok=False)
    os.chdir(worktree)
    sys.path.insert(0, str(worktree))
    sys.argv = ["diffusion_train.py", "--cfg_file", str(config)]
    os.environ["ONE_SAMPLE_OUT_DIR"] = str(output_root)

    import diffusion_train as train

    if args.mode == "preflight":
        if args.preflight_steps < 1 or args.preflight_steps > 100:
            raise ValueError("preflight_steps must be in [1, 100]")
        train.cfg.train.max_steps = int(args.preflight_steps)
        train.cfg.train.num_workers = 0
        train.cfg.train.step_checkpoint_every = 1
        train.cfg.train.step_checkpoint_keep = 1

    manifest = {
        "schema": "diffusionsnake.pure2d_mainline_training_launch.v1",
        "status": "STARTED",
        "mode": args.mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "worktree": str(worktree),
        "config": str(config),
        "config_sha256": sha256_file(config),
        "output_root": str(output_root),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "max_steps": int(train.cfg.train.max_steps),
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
