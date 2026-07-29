import fcntl
import json
import os
import pathlib
import sys
from typing import Optional, TextIO


class RunDirectoryLock:
    """Prevent two processes from mutating one experiment output directory."""

    def __init__(self, output_dir: pathlib.Path, experiment_id: str) -> None:
        self.output_dir = pathlib.Path(output_dir)
        self.experiment_id = str(experiment_id)
        self._handle: Optional[TextIO] = None

    def acquire(self) -> "RunDirectoryLock":
        if self.output_dir.name != self.experiment_id:
            raise ValueError(
                "output directory basename must equal experiment_id: {} != {}".format(
                    self.output_dir.name,
                    self.experiment_id,
                )
            )
        existing_outputs = [
            self.output_dir / "train.jsonl",
            self.output_dir / "run_manifest.json",
            self.output_dir / "checkpoints",
        ]
        if any(path.exists() for path in existing_outputs):
            raise RuntimeError(
                "experiment output already contains a run; choose a new experiment_id "
                "or use a tested resume entry point: {}".format(self.output_dir)
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_dir / ".run.lock"
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            handle.close()
            raise RuntimeError(
                "experiment output is already locked: {} ({})".format(
                    self.output_dir,
                    owner,
                )
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({
            "experiment_id": self.experiment_id,
            "pid": os.getpid(),
            "argv": sys.argv,
        }, ensure_ascii=False))
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle
        return self

    def release(self) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None

    def __enter__(self) -> "RunDirectoryLock":
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()
