#!/usr/bin/env python3
"""Stop one verified training process at a step or wall-clock limit.

This guard is intentionally independent of the training loop so an already
running experiment can be bounded without restarting and losing optimizer
continuity. It only signals a PID whose current command line still matches the
expected training script and config.
"""

import argparse
import datetime as dt
import os
from pathlib import Path
import signal
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--expected-config", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--max-step", type=int, required=True)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--status-log", required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    return parser.parse_args()


def now():
    return dt.datetime.now().astimezone()


def append_status(path, message):
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("[{}] {}\n".format(stamp, message))


def process_command(pid):
    path = Path("/proc") / str(pid) / "cmdline"
    try:
        return path.read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except FileNotFoundError:
        return ""


def process_is_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def last_logged_step(path):
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(size - 262144, 0))
            tail = handle.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return 0
    for line in reversed(tail.splitlines()):
        marker = "[step "
        start = line.find(marker)
        if start < 0:
            continue
        end = line.find("]", start + len(marker))
        if end < 0:
            continue
        try:
            return int(line[start + len(marker):end])
        except ValueError:
            continue
    return 0


def checkpoint_ready(checkpoint_dir, step):
    path = checkpoint_dir / "step_{:06d}.pt".format(step)
    return path.is_file() and path.stat().st_size > 0


def latest_checkpoint(checkpoint_dir):
    paths = sorted(checkpoint_dir.glob("step_*.pt"))
    return paths[-1].name if paths else "none"


def main():
    args = parse_args()
    if args.max_step <= 0:
        raise ValueError("--max-step must be positive")
    if args.poll_seconds <= 0 or args.poll_seconds > 60:
        raise ValueError("--poll-seconds must be in (0, 60]")

    deadline = dt.datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        raise ValueError("--deadline must include a UTC offset")

    train_log = Path(args.train_log)
    checkpoint_dir = Path(args.checkpoint_dir)
    status_log = Path(args.status_log)
    expected_parts = (
        "tools/volmem/train_memflowdit.py",
        args.expected_config,
    )

    command = process_command(args.pid)
    if not command or any(part not in command for part in expected_parts):
        raise RuntimeError(
            "refusing to monitor pid {} with command {!r}".format(
                args.pid, command
            )
        )

    append_status(
        status_log,
        "armed pid={} max_step={} deadline={} command={}".format(
            args.pid,
            args.max_step,
            deadline.isoformat(),
            command,
        ),
    )

    stop_reason = ""
    while process_is_alive(args.pid):
        command = process_command(args.pid)
        if not command or any(part not in command for part in expected_parts):
            append_status(
                status_log,
                "pid identity changed; refusing to signal command={!r}".format(
                    command
                ),
            )
            return 2

        current_step = last_logged_step(train_log)
        if current_step >= args.max_step and checkpoint_ready(
            checkpoint_dir, args.max_step
        ):
            stop_reason = "step_limit"
            break
        if now() >= deadline:
            stop_reason = "wall_clock_limit"
            break
        time.sleep(args.poll_seconds)

    if not process_is_alive(args.pid):
        append_status(
            status_log,
            "training exited before guard fired; latest_checkpoint={}".format(
                latest_checkpoint(checkpoint_dir)
            ),
        )
        return 0

    current_step = last_logged_step(train_log)
    append_status(
        status_log,
        "stopping pid={} reason={} current_step={} latest_checkpoint={}".format(
            args.pid,
            stop_reason,
            current_step,
            latest_checkpoint(checkpoint_dir),
        ),
    )
    os.kill(args.pid, signal.SIGTERM)

    for _ in range(30):
        if not process_is_alive(args.pid):
            append_status(
                status_log,
                "training stopped; latest_checkpoint={}".format(
                    latest_checkpoint(checkpoint_dir)
                ),
            )
            return 0
        time.sleep(1.0)

    command = process_command(args.pid)
    if command and all(part in command for part in expected_parts):
        append_status(status_log, "SIGTERM timeout; sending SIGKILL")
        os.kill(args.pid, signal.SIGKILL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
