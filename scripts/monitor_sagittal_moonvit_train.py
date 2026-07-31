#!/usr/bin/env python3
"""Launch or attach to a sagittal MoonViT training run and monitor its health.

The monitor is deliberately observational: it never restarts or kills a training
process.  A failed preflight produces an event and a non-zero exit status; health
thresholds are recorded while monitoring continues, leaving the operator to decide
whether to resume.
"""

from __future__ import print_function

import argparse
import csv
import datetime
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections import deque
from pathlib import Path


DEFAULT_POLL_SECONDS = 30.0
DEFAULT_STALL_SECONDS = 300.0
DEFAULT_TEMPERATURE_WARN_C = 80.0
DEFAULT_TEMPERATURE_CRITICAL_C = 84.0
DEFAULT_MEMORY_RATIO = 0.95
DEFAULT_MEMORY_FREE_MIB = 2048.0
DEFAULT_DISK_FREE_GIB = 50.0
DEFAULT_FOREGROUND_TARGET = 0.5
DEFAULT_FOREGROUND_TOLERANCE = 0.1
DEFAULT_NO_FOREGROUND_BATCHES = 200
DEFAULT_ADAPTER_ZERO_WINDOW = 200
DEFAULT_ADAPTER_ZERO_EPS = 1e-12
DEFAULT_CHECKPOINT_LAG_STEPS = 500


def utc_now():
    return datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _finite(value):
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _finite_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _row_step(row):
    try:
        return int(row.get("step"))
    except (AttributeError, TypeError, ValueError):
        return None


def _torch_tree_finite(value, path="checkpoint"):
    """Check tensors and scalar values without moving tensors to CUDA."""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("non-finite tensor at {}".format(path))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _torch_tree_finite(item, "{}.{}".format(path, key))
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _torch_tree_finite(item, "{}[{}]".format(path, index))
        return
    if not _finite(value):
        raise ValueError("non-finite scalar at {}".format(path))


def validate_checkpoint_cpu(
    checkpoint_path,
    expected_run_id=None,
    expected_cfg_hash=None,
    require_adapter=True,
):
    """Load and validate a format-v2 checkpoint entirely on CPU.

    The returned metadata is safe for the monitor to use.  This function does
    not write anything; callers must use ``atomic_update_validated_latest`` only
    after validation succeeds.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for checkpoint validation") from exc

    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a dict")

    required = (
        "format_version",
        "run_id",
        "cfg_hash",
        "saved_at",
        "epoch",
        "step",
        "step_in_epoch",
        "rng",
        "state_dict",
        "optimizer",
        "scheduler",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("checkpoint missing required fields: {}".format(missing))
    if int(payload["format_version"]) < 2:
        raise ValueError("checkpoint format_version must be >= 2")
    run_id = str(payload["run_id"])
    cfg_hash = str(payload["cfg_hash"])
    if not run_id:
        raise ValueError("checkpoint run_id is empty")
    if not cfg_hash:
        raise ValueError("checkpoint cfg_hash is empty")
    if expected_run_id is not None and run_id != str(expected_run_id):
        raise ValueError("checkpoint run_id mismatch")
    if expected_cfg_hash is not None and cfg_hash != str(expected_cfg_hash):
        raise ValueError("checkpoint cfg_hash mismatch")

    epoch = int(payload["epoch"])
    step = int(payload["step"])
    step_in_epoch = int(payload["step_in_epoch"])
    if epoch < 0 or step < 0 or step_in_epoch < 0:
        raise ValueError("checkpoint progress metadata is negative")
    if not isinstance(payload["saved_at"], str) or not payload["saved_at"]:
        raise ValueError("checkpoint saved_at is missing")
    if not isinstance(payload["state_dict"], dict):
        raise ValueError("checkpoint state_dict must be a dict")
    adapter_keys = [
        str(key)
        for key in payload["state_dict"]
        if "locate_feat_adapter" in str(key).lower()
    ]
    if require_adapter and not adapter_keys:
        raise ValueError("checkpoint has no MoonViT locate_feat_adapter key")

    _torch_tree_finite(payload)
    return {
        "path": str(checkpoint_path),
        "format_version": int(payload["format_version"]),
        "run_id": run_id,
        "cfg_hash": cfg_hash,
        "saved_at": payload["saved_at"],
        "epoch": epoch,
        "step": step,
        "step_in_epoch": step_in_epoch,
        "adapter_keys": adapter_keys,
    }


def _copy_stable_checkpoint(source_path, tmp_path):
    source_path = Path(source_path)
    tmp_path = Path(tmp_path)
    with source_path.open("rb") as source, tmp_path.open("wb") as target:
        opened = os.fstat(source.fileno())
        shutil.copyfileobj(source, target, length=1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    current = source_path.stat()
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(opened, field) != getattr(current, field) for field in identity_fields):
        raise RuntimeError("checkpoint changed during validation copy")


def atomic_update_validated_latest(source_path, destination_path):
    """Copy a previously validated checkpoint using ``os.replace``."""
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination_path.with_name(
        ".{}.tmp.{}".format(destination_path.name, os.getpid())
    )
    try:
        _copy_stable_checkpoint(source_path, tmp_path)
        os.replace(str(tmp_path), str(destination_path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return destination_path


def validate_and_update_latest(
    latest_path,
    validated_path,
    expected_run_id=None,
    expected_cfg_hash=None,
    require_adapter=True,
):
    latest_path = Path(latest_path)
    validated_path = Path(validated_path)
    validated_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = validated_path.with_name(
        ".{}.tmp.{}".format(validated_path.name, os.getpid())
    )
    try:
        _copy_stable_checkpoint(latest_path, tmp_path)
        metadata = validate_checkpoint_cpu(
            tmp_path,
            expected_run_id=expected_run_id,
            expected_cfg_hash=expected_cfg_hash,
            require_adapter=require_adapter,
        )
        os.replace(str(tmp_path), str(validated_path))
        metadata["path"] = str(latest_path)
        return metadata
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def precheck_cache(
    cache_root,
    manifest_path=None,
    expected_count=None,
    manifest_splits=None,
):
    """Check that requested MoonViT cache splits exist and are non-empty."""
    selected_splits = (
        {str(value).strip().lower() for value in manifest_splits}
        if manifest_splits is not None
        else None
    )
    result = {
        "ok": True,
        "cache_root": str(cache_root) if cache_root else None,
        "npz_count": 0,
        "expected_count": expected_count,
        "manifest_splits": sorted(selected_splits) if selected_splits else None,
        "errors": [],
        "warnings": [],
    }
    if not cache_root:
        result["ok"] = False
        result["errors"].append("cache root was not provided")
        return result

    root = Path(cache_root)
    if not root.exists() or not root.is_dir():
        result["ok"] = False
        result["errors"].append("cache root does not exist: {}".format(root))
        return result

    search_roots = [root]
    if selected_splits:
        search_roots = [root / split for split in sorted(selected_splits)]
        missing_roots = [path for path in search_roots if not path.is_dir()]
        if missing_roots:
            result["ok"] = False
            result["errors"].append(
                "cache split directories are missing: {}".format(
                    [str(path) for path in missing_roots]
                )
            )
    files = [
        path
        for search_root in search_roots
        if search_root.is_dir()
        for path in search_root.rglob("*.npz")
        if path.is_file() and path.stat().st_size > 0
    ]
    result["npz_count"] = len(files)
    if not files:
        result["ok"] = False
        result["errors"].append("cache contains no non-empty .npz files")

    if expected_count is None and manifest_path:
        manifest = Path(manifest_path)
        if manifest.exists():
            with manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = csv.DictReader(handle)
                expected_count = sum(
                    1
                    for row in rows
                    if selected_splits is None
                    or str(row.get("split", "")).strip().lower() in selected_splits
                )
            result["expected_count"] = expected_count
    if expected_count is not None and len(files) != int(expected_count):
        result["ok"] = False
        result["errors"].append(
            "cache has {} files but {} are expected".format(len(files), int(expected_count))
        )
    return result


def latest_run_id(path, min_timestamp=None):
    """Return the run_id from the latest JSONL event after ``min_timestamp``."""
    path = Path(path)
    if not path.exists():
        return None
    latest = None
    latest_timestamp = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    item = json.loads(
                        line,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
                    )
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(item, dict) or not item.get("run_id"):
                    continue
                timestamp = _parse_timestamp(item.get("timestamp"))
                if min_timestamp is not None and (
                    timestamp is None or timestamp + 1.0 < float(min_timestamp)
                ):
                    continue
                if latest_timestamp is None or timestamp is None or timestamp >= latest_timestamp:
                    latest = str(item["run_id"])
                    latest_timestamp = timestamp
    except OSError:
        return None
    return latest


def read_jsonl(path, max_rows=1000, run_id=None):
    """Read first-500 and recent valid rows, tolerating an in-progress final line."""
    recent = deque(maxlen=int(max_rows))
    early = {}
    malformed = 0
    path = Path(path)
    if not path.exists():
        return [], {"malformed": 0, "exists": False}
    selected_run_id = str(run_id) if run_id is not None else None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                malformed += 1
                continue
            if not isinstance(item, dict):
                continue
            if selected_run_id is not None and str(item.get("run_id", "")) != selected_run_id:
                continue
            recent.append(item)
            try:
                step = int(item.get("step", -1))
            except (TypeError, ValueError):
                step = -1
            if 0 <= step <= 500:
                early[step] = item
    combined = {}
    for item in early.values():
        try:
            combined[int(item.get("step", -1))] = item
        except (TypeError, ValueError):
            pass
    for item in recent:
        try:
            combined[int(item.get("step", -1))] = item
        except (TypeError, ValueError):
            continue
    rows = [combined[step] for step in sorted(combined) if step >= 0]
    return rows, {"malformed": malformed, "exists": True}


def p99(values):
    values = sorted(float(value) for value in values if _finite(value) and float(value) >= 0.0)
    if not values:
        return 0.0
    index = min(len(values) - 1, int(math.ceil(0.99 * len(values))) - 1)
    return values[index]


def _parse_timestamp(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        text = value.rstrip("Z")
        parsed = datetime.datetime.fromisoformat(text)
        return parsed.replace(tzinfo=datetime.timezone.utc).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def latest_step(rows):
    steps = []
    for row in rows:
        try:
            steps.append(int(row["step"]))
        except (KeyError, TypeError, ValueError):
            continue
    return max(steps) if steps else None


def query_nvidia_smi(gpu=None):
    command = [
        "nvidia-smi",
        "--query-gpu=index,temperature.gpu,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    records = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 6:
            continue
        if gpu is not None and str(fields[0]) != str(gpu):
            continue
        def number(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        records.append({
            "index": fields[0],
            "temperature_c": number(fields[1]),
            "memory_total_mib": number(fields[2]),
            "memory_used_mib": number(fields[3]),
            "memory_free_mib": number(fields[4]),
            "utilization_pct": number(fields[5]),
        })
    if gpu is not None and not records:
        raise RuntimeError("GPU {} was not found in nvidia-smi output".format(gpu))
    return records


def process_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def process_start_time(pid):
    """Return Linux process start time as an epoch timestamp when available."""
    try:
        stat_fields = Path("/proc").joinpath(str(int(pid)), "stat").read_text().split()
        start_ticks = int(stat_fields[21])
        clock_ticks = float(os.sysconf("SC_CLK_TCK"))
        boot_time = None
        with Path("/proc/stat").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("btime "):
                    boot_time = float(line.split()[1])
                    break
        if boot_time is None or clock_ticks <= 0.0:
            return None
        return boot_time + start_ticks / clock_ticks
    except (OSError, ValueError, IndexError):
        return None


def find_training_pid(root):
    root = Path(root).resolve()
    training_script = root / "diffusion_train.py"
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        cmdline = entry / "cmdline"
        cwd_link = entry / "cwd"
        try:
            arguments = [
                value.decode("utf-8", "replace")
                for value in cmdline.read_bytes().split(b"\x00")
                if value
            ]
            process_cwd = cwd_link.resolve()
        except OSError:
            continue
        for argument in arguments:
            argument_path = Path(argument)
            if argument_path.name != "diffusion_train.py":
                continue
            resolved = argument_path if argument_path.is_absolute() else process_cwd / argument_path
            if resolved.resolve() == training_script:
                return int(entry.name)
    return None


def disk_free_gib(path):
    usage = shutil.disk_usage(str(path))
    return float(usage.free) / (1024.0 ** 3)


def evaluate_health(
    rows,
    now=None,
    log_mtime=None,
    gpu_records=None,
    disk_free=None,
    checkpoint_step=None,
    stall_floor=DEFAULT_STALL_SECONDS,
    temperature_warn=DEFAULT_TEMPERATURE_WARN_C,
    temperature_critical=DEFAULT_TEMPERATURE_CRITICAL_C,
    memory_ratio=DEFAULT_MEMORY_RATIO,
    memory_free_mib=DEFAULT_MEMORY_FREE_MIB,
    disk_free_gib=DEFAULT_DISK_FREE_GIB,
    foreground_target=DEFAULT_FOREGROUND_TARGET,
    foreground_tolerance=DEFAULT_FOREGROUND_TOLERANCE,
    no_foreground_batches=DEFAULT_NO_FOREGROUND_BATCHES,
    adapter_zero_window=DEFAULT_ADAPTER_ZERO_WINDOW,
    adapter_zero_eps=DEFAULT_ADAPTER_ZERO_EPS,
    checkpoint_lag_steps=DEFAULT_CHECKPOINT_LAG_STEPS,
):
    """Return threshold violations; this function has no side effects."""
    now = time.time() if now is None else float(now)
    alerts = []
    valid_rows = [row for row in rows if isinstance(row, dict)]
    durations = []
    for row in valid_rows:
        try:
            duration = float(row.get("time_ms")) / 1000.0
            if _finite(duration) and duration >= 0.0:
                durations.append(duration)
        except (TypeError, ValueError):
            continue
    duration_p99 = p99(durations)
    stall_limit = max(float(stall_floor), 10.0 * duration_p99)

    last_timestamp = None
    for row in valid_rows:
        parsed = _parse_timestamp(row.get("timestamp"))
        if parsed is not None:
            last_timestamp = parsed if last_timestamp is None else max(last_timestamp, parsed)
    if last_timestamp is None and log_mtime is not None:
        last_timestamp = float(log_mtime)
    if last_timestamp is not None and now - last_timestamp > stall_limit:
        alerts.append({
            "code": "stall",
            "severity": "critical",
            "age_seconds": now - last_timestamp,
            "stall_limit_seconds": stall_limit,
            "duration_p99_seconds": duration_p99,
        })

    for gpu in gpu_records or []:
        temperature = gpu.get("temperature_c")
        if temperature is not None and temperature >= float(temperature_critical):
            alerts.append({"code": "gpu_temperature", "severity": "critical", **gpu})
        elif temperature is not None and temperature >= float(temperature_warn):
            alerts.append({"code": "gpu_temperature", "severity": "warning", **gpu})
        total = gpu.get("memory_total_mib")
        used = gpu.get("memory_used_mib")
        free = gpu.get("memory_free_mib")
        if total and used is not None and float(used) / float(total) > float(memory_ratio):
            alerts.append({"code": "gpu_memory_ratio", "severity": "critical", **gpu})
        if free is not None and float(free) < float(memory_free_mib):
            alerts.append({"code": "gpu_memory_free", "severity": "critical", **gpu})

    if disk_free is not None and float(disk_free) < float(disk_free_gib):
        alerts.append({
            "code": "disk_free",
            "severity": "critical",
            "free_gib": float(disk_free),
            "threshold_gib": float(disk_free_gib),
        })

    steps = []
    for row in valid_rows:
        step = _row_step(row)
        if step is not None:
            steps.append(step)
    last_step = max(steps) if steps else None

    foreground_window = valid_rows[-500:]
    window_steps = [_row_step(row) for row in foreground_window]
    window_steps = [step for step in window_steps if step is not None]
    window_ratios = [
        _finite_float(row.get("foreground_ratio"))
        for row in foreground_window
    ]
    window_ratios = [ratio for ratio in window_ratios if ratio is not None]
    if (
        len(foreground_window) >= 500
        and window_steps
        and len(window_ratios) == len(foreground_window)
    ):
        foreground_mean = sum(window_ratios) / len(window_ratios)
        lower = float(foreground_target) - float(foreground_tolerance)
        upper = float(foreground_target) + float(foreground_tolerance)
        if foreground_mean < lower or foreground_mean > upper:
            alerts.append({
                "code": "foreground_ratio",
                "severity": "critical",
                "mean": foreground_mean,
                "lower": lower,
                "upper": upper,
                "window_start_step": min(window_steps),
                "window_end_step": max(window_steps),
            })

    no_foreground_window = int(no_foreground_batches)
    if no_foreground_window > 0:
        recent = valid_rows[-no_foreground_window:]
        foreground_values = [
            _finite_float(row.get("foreground_count")) for row in recent
        ]
        if (
            len(recent) >= no_foreground_window
            and all(
                value is not None and value <= 0.0
                for value in foreground_values
            )
        ):
            alerts.append({
                "code": "no_foreground",
                "severity": "critical",
                "batches": no_foreground_window,
            })

    adapter_window = int(adapter_zero_window)
    if adapter_window > 0:
        adapter_recent = valid_rows[-adapter_window:]
        gradient_values = [
            _finite_float(row.get("adapter_grad_l2")) for row in adapter_recent
        ]
        update_values = [
            _finite_float(row.get("adapter_update_l2")) for row in adapter_recent
        ]
        gradients_zero = (
            len(adapter_recent) >= adapter_window
            and all(
                value is not None and value <= float(adapter_zero_eps)
                for value in gradient_values
            )
        )
        updates_zero = (
            len(adapter_recent) >= adapter_window
            and all(
                value is not None and value <= float(adapter_zero_eps)
                for value in update_values
            )
        )
        if gradients_zero:
            alerts.append({
                "code": "adapter_zero_gradient",
                "severity": "critical",
                "batches": adapter_window,
            })
        if updates_zero:
            alerts.append({
                "code": "adapter_zero_update",
                "severity": "critical",
                "batches": adapter_window,
            })

    if last_step is not None:
        if checkpoint_step is None:
            lag = int(last_step)
            checkpoint_is_missing = True
        else:
            lag = int(last_step) - int(checkpoint_step)
            checkpoint_is_missing = False
        if lag > int(checkpoint_lag_steps):
            alerts.append({
                "code": "checkpoint_lag",
                "severity": "critical",
                "last_step": int(last_step),
                "checkpoint_step": checkpoint_step,
                "lag_steps": lag,
                "checkpoint_missing": checkpoint_is_missing,
                "threshold_steps": int(checkpoint_lag_steps),
            })

    return alerts


def _write_event(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": utc_now(), **event}
    encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
    print(encoded, flush=True)


def _load_yaml_values(path):
    if not path:
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        return values if isinstance(values, dict) else {}
    except (OSError, ValueError):
        return {}


def _resolve_path(root, value):
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else Path(root) / path


def _attach_pid(value, root):
    if value is None:
        return None
    if str(value).lower() == "auto":
        return find_training_pid(root)
    return int(value)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", default=str(root))
    parser.add_argument("--cfg-file", default=os.environ.get("CFG_FILE", ""))
    parser.add_argument("--out-dir", default=os.environ.get("ONE_SAMPLE_OUT_DIR", ""))
    parser.add_argument("--cache-root", default=os.environ.get("SAGITTAL_MOONVIT_CACHE_ROOT", ""))
    parser.add_argument("--manifest", default=os.environ.get("SAGITTAL_MOONVIT_MANIFEST", ""))
    parser.add_argument("--expected-cache-count", type=int, default=None)
    parser.add_argument("--allow-incomplete-cache", action="store_true")
    parser.add_argument("--gpu", default=os.environ.get("TRAIN_GPU", "0"))
    parser.add_argument("--attach", nargs="?", const="auto", default=None, metavar="PID")
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--interval", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-cache-precheck", action="store_true")
    parser.add_argument("--no-adapter-key-check", action="store_true")
    parser.add_argument("--command", nargs=argparse.REMAINDER, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    cfg_file = Path(args.cfg_file) if args.cfg_file else None
    if cfg_file is not None and not cfg_file.is_absolute():
        cfg_file = root / cfg_file
    cfg_values = _load_yaml_values(cfg_file)
    train_values = cfg_values.get("train", {}) if isinstance(cfg_values, dict) else {}

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is None:
        configured_out = cfg_values.get("model_dir") if isinstance(cfg_values, dict) else None
        out_dir = _resolve_path(root, configured_out) or root / "data" / "outputs" / "sagittal_2d_pseudo3d_moonvit"
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor_log = out_dir / "monitor.jsonl"
    checkpoint_dir = out_dir / "checkpoints"
    latest_checkpoint = checkpoint_dir / "latest.pt"
    validated_checkpoint = checkpoint_dir / "validated_latest.pt"

    cache_root = args.cache_root or cfg_values.get("sagittal_moonvit_cache_root", "")
    manifest = args.manifest or cfg_values.get("locate_feat_manifest", "")
    cache_root = _resolve_path(root, cache_root)
    manifest = _resolve_path(root, manifest)

    if not args.no_cache_precheck:
        cache_result = precheck_cache(
            cache_root,
            manifest_path=manifest,
            expected_count=args.expected_cache_count,
            manifest_splits=("training", "validation"),
        )
        _write_event(monitor_log, {"event": "cache_precheck", **cache_result})
        if not cache_result["ok"] and not args.allow_incomplete_cache:
            _write_event(monitor_log, {"event": "monitor_aborted", "reason": "cache_precheck_failed"})
            return 2

    monitor_run_id = os.environ.get("TRAIN_RUN_ID", "").strip() or None
    monitor_cfg_hash = os.environ.get("MONITOR_CFG_HASH", "").strip() or None
    resume_requested = bool(cfg_values.get("resume", False)) if isinstance(cfg_values, dict) else False
    resume_weights_only = bool(cfg_values.get("resume_weights_only", False)) if isinstance(cfg_values, dict) else False

    attach_value = args.attach
    if args.pid is not None:
        attach_value = str(args.pid)
    attach_pid = _attach_pid(attach_value, root)
    process = None
    train_log = None
    train_log_handle = None
    monitored_process_started_at = None
    if attach_value is not None:
        if attach_pid is None:
            _write_event(monitor_log, {"event": "monitor_aborted", "reason": "training_process_not_found"})
            return 2
        monitored_process_started_at = process_start_time(attach_pid)
        _write_event(monitor_log, {
            "event": "monitor_attach",
            "pid": int(attach_pid),
            "process_started_at": monitored_process_started_at,
        })
    else:
        command = args.command
        if command and command[:1] == ["--"]:
            command = command[1:]
        is_default_command = not command
        if not command:
            command = [sys.executable, str(root / "diffusion_train.py")]
            if cfg_file is not None:
                command.extend(["--cfg_file", str(cfg_file)])

        if is_default_command and monitor_run_id is None:
            if (not resume_requested) or resume_weights_only:
                monitor_run_id = uuid.uuid4().hex
            else:
                resume_value = os.environ.get("ONE_SAMPLE_RESUME_PATH", "").strip()
                if not resume_value and isinstance(cfg_values, dict):
                    resume_value = str(cfg_values.get("resume_path", "") or "").strip()
                resume_candidate = _resolve_path(root, resume_value) if resume_value else latest_checkpoint
                if resume_candidate is not None and resume_candidate.exists():
                    try:
                        resume_metadata = validate_checkpoint_cpu(
                            resume_candidate,
                            require_adapter=False,
                        )
                        monitor_run_id = resume_metadata["run_id"]
                        if monitor_cfg_hash is None:
                            monitor_cfg_hash = resume_metadata["cfg_hash"]
                    except Exception as exc:
                        _write_event(monitor_log, {
                            "event": "resume_checkpoint_precheck_failed",
                            "path": str(resume_candidate),
                            "error": str(exc),
                        })

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["ONE_SAMPLE_OUT_DIR"] = str(out_dir)
        if cfg_file is not None:
            env["CFG_FILE"] = str(cfg_file)
        if monitor_run_id is not None:
            env["TRAIN_RUN_ID"] = str(monitor_run_id)
        train_log = out_dir / "train_monitor_{}.log".format(time.strftime("%Y%m%d_%H%M%S"))
        train_log_handle = train_log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(root),
            env=env,
            stdout=train_log_handle,
            stderr=subprocess.STDOUT,
        )
        monitored_process_started_at = time.time()
        _write_event(monitor_log, {
            "event": "train_start",
            "pid": process.pid,
            "gpu": str(args.gpu),
            "command": command,
            "train_log": str(train_log),
            "run_id": monitor_run_id,
        })

    last_malformed = None
    last_alert_signature = None
    checkpoint_signature = None
    checkpoint_metadata = None
    checkpoint_error = None
    monitor_run_id_locked = monitor_run_id is not None
    log_path = out_dir / "logs.jsonl"
    try:
        while True:
            if process is not None:
                alive = process.poll() is None
            else:
                alive = process_alive(attach_pid)

            if latest_checkpoint.exists():
                try:
                    checkpoint_stat = latest_checkpoint.stat()
                    current_signature = (
                        int(checkpoint_stat.st_ino),
                        int(checkpoint_stat.st_size),
                        int(checkpoint_stat.st_mtime_ns),
                    )
                    checkpoint_is_current = (
                        resume_requested
                        or monitored_process_started_at is None
                        or checkpoint_stat.st_mtime + 1.0 >= monitored_process_started_at
                    )
                except OSError:
                    current_signature = None
                    checkpoint_is_current = False
                if checkpoint_is_current and current_signature != checkpoint_signature:
                    checkpoint_signature = current_signature
                    try:
                        checkpoint_metadata = validate_and_update_latest(
                            latest_checkpoint,
                            validated_checkpoint,
                            expected_run_id=monitor_run_id,
                            expected_cfg_hash=monitor_cfg_hash,
                            require_adapter=not args.no_adapter_key_check,
                        )
                        monitor_run_id = checkpoint_metadata["run_id"]
                        monitor_run_id_locked = True
                        monitor_cfg_hash = checkpoint_metadata["cfg_hash"]
                        checkpoint_error = None
                    except Exception as exc:
                        checkpoint_error = str(exc)
                        _write_event(monitor_log, {
                            "event": "checkpoint_validation_failed",
                            "path": str(latest_checkpoint),
                            "error": checkpoint_error,
                        })
            elif checkpoint_metadata is None:
                checkpoint_error = "latest checkpoint does not exist yet"

            if monitor_run_id is None:
                discovered_run_id = latest_run_id(
                    log_path,
                    min_timestamp=monitored_process_started_at,
                )
                if discovered_run_id is not None:
                    monitor_run_id = discovered_run_id
                    monitor_run_id_locked = True
            rows, jsonl_status = read_jsonl(
                log_path,
                run_id=monitor_run_id if monitor_run_id_locked else None,
            )
            if jsonl_status["malformed"] != last_malformed:
                last_malformed = jsonl_status["malformed"]
                if last_malformed:
                    _write_event(monitor_log, {
                        "event": "jsonl_malformed",
                        "count": int(last_malformed),
                    })

            gpu_records = []
            try:
                gpu_records = query_nvidia_smi(args.gpu)
            except Exception as exc:
                _write_event(monitor_log, {"event": "nvidia_smi_failed", "error": str(exc)})

            checkpoint_step = checkpoint_metadata["step"] if checkpoint_metadata else None
            row_timestamps = [
                _parse_timestamp(row.get("timestamp"))
                for row in rows
                if isinstance(row, dict)
            ]
            row_timestamps = [value for value in row_timestamps if value is not None]
            if row_timestamps:
                log_mtime = max(row_timestamps)
            elif monitored_process_started_at is not None:
                log_mtime = monitored_process_started_at
            else:
                log_mtime = None
            try:
                free_gib = disk_free_gib(out_dir)
            except OSError as exc:
                free_gib = None
                _write_event(monitor_log, {"event": "disk_query_failed", "error": str(exc)})

            alerts = evaluate_health(
                rows,
                log_mtime=log_mtime,
                gpu_records=gpu_records,
                disk_free=free_gib,
                checkpoint_step=checkpoint_step,
                checkpoint_lag_steps=int(os.environ.get("MONITOR_CHECKPOINT_LAG_STEPS", DEFAULT_CHECKPOINT_LAG_STEPS)),
                adapter_zero_window=int(os.environ.get("MONITOR_ADAPTER_ZERO_WINDOW", DEFAULT_ADAPTER_ZERO_WINDOW)),
                no_foreground_batches=int(os.environ.get("MONITOR_NO_FOREGROUND_BATCHES", DEFAULT_NO_FOREGROUND_BATCHES)),
            )
            _write_event(monitor_log, {
                "event": "heartbeat",
                "alive": bool(alive),
                "pid": int(process.pid if process is not None else attach_pid),
                "run_id": monitor_run_id,
                "last_step": latest_step(rows),
                "checkpoint_step": checkpoint_step,
                "checkpoint_error": checkpoint_error,
                "jsonl_exists": bool(jsonl_status["exists"]),
                "gpu": gpu_records,
                "disk_free_gib": free_gib,
                "alerts": alerts,
            })

            alert_signature = tuple(sorted(
                (item.get("code"), item.get("severity")) for item in alerts
            ))
            if alert_signature and alert_signature != last_alert_signature:
                _write_event(monitor_log, {"event": "health_alert", "alerts": alerts})
            last_alert_signature = alert_signature

            if not alive:
                return_code = process.returncode if process is not None else None
                _write_event(monitor_log, {
                    "event": "train_exited",
                    "returncode": return_code,
                    "attached": process is None,
                    "exit_status_unknown": process is None,
                })
                return int(return_code) if return_code is not None else 0
            if args.once:
                break
            time.sleep(max(0.1, float(args.interval)))
    finally:
        if train_log_handle is not None:
            train_log_handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
