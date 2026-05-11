import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
YOLOE_ROOT = ROOT / "yoloe"

if str(YOLOE_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOE_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Run a minimal BTCV RT-DETR detector trial.")
    parser.add_argument("--data", default="data/exports/btcv_yolo/btcv_detect.yaml")
    parser.add_argument(
        "--model",
        default="yoloe/ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml",
        help="RT-DETR yaml inside the vendored ultralytics tree.",
    )
    parser.add_argument("--device", default="auto", help="CUDA device index, cpu, or auto.")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--project", default="data/outputs")
    parser.add_argument("--name", default="btcv_rtdetr_trial")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--resume-summary", default="", help="Optional summary json path override.")
    return parser.parse_args()


def pick_idle_gpu():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        idx, util, mem_used, mem_total = map(int, parts)
        mem_ratio = mem_used / max(mem_total, 1)
        score = mem_ratio + util / 100.0
        rows.append((score, mem_used, util, idx, mem_total))
    if not rows:
        raise RuntimeError("No GPUs found via nvidia-smi")
    rows.sort()
    score, mem_used, util, idx, mem_total = rows[0]
    print(
        f"[auto-device] selected GPU {idx} "
        f"(score={score:.3f}, memory.used={mem_used} MiB, util={util}%, total={mem_total} MiB)"
    )
    return str(idx)


def resolve_device(device_arg: str):
    device_arg = str(device_arg).strip().lower()
    if device_arg and device_arg != "auto":
        return device_arg
    return pick_idle_gpu()


def disable_integrations():
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("COMET_DISABLE_AUTO_LOGGING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")


def ensure_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.resolve() == src.resolve():
            return
        if dst.is_dir() and not dst.is_symlink():
            raise RuntimeError(f"Refusing to replace real directory: {dst}")
        dst.unlink()
    dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def ensure_dir(path: Path):
    if path.is_symlink():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def prepare_split_symlinks(src_dir: Path, dst_dir: Path):
    ensure_dir(dst_dir)
    for src in sorted(src_dir.glob("*")):
        if src.is_file():
            ensure_symlink(src, dst_dir / src.name)


def prepare_label_symlinks(src_img_dir: Path, src_label_dir: Path, dst_label_dir: Path):
    ensure_dir(dst_label_dir)
    for src_img in sorted(src_img_dir.glob("*")):
        if not src_img.is_file():
            continue
        stem = src_img.stem
        label_stem = stem[:-6] if stem.endswith("_image") else stem
        src_label = src_label_dir / f"{label_stem}.txt"
        if src_label.exists():
            ensure_symlink(src_label, dst_label_dir / f"{stem}.txt")


def prepare_runtime_data_yaml(data_yaml: Path):
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data_root = Path(data["path"])
    if not data_root.is_absolute():
        data_root = (data_yaml.parent / data_root).resolve()

    runtime_root = ROOT / "data" / "outputs" / "btcv_rtdetr_trial" / "runtime_dataset"
    prepare_split_symlinks(data_root / "images" / "train", runtime_root / "images" / "train")
    prepare_split_symlinks(data_root / "images" / "val", runtime_root / "images" / "val")
    prepare_label_symlinks(
        data_root / "images" / "train",
        data_root / "labels" / "detect" / "train",
        runtime_root / "labels" / "train",
    )
    prepare_label_symlinks(
        data_root / "images" / "val",
        data_root / "labels" / "detect" / "val",
        runtime_root / "labels" / "val",
    )
    for cache_path in (runtime_root / "labels" / "train.cache", runtime_root / "labels" / "val.cache"):
        if cache_path.exists() or cache_path.is_symlink():
            cache_path.unlink()

    runtime_yaml = ROOT / "data" / "outputs" / "btcv_rtdetr_trial" / "runtime_btcv_detect.yaml"
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)

    runtime_data = dict(data)
    runtime_data["path"] = str(runtime_root)
    with open(runtime_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_data, f, sort_keys=False, allow_unicode=True)
    return runtime_yaml


def write_summary(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def extract_metrics(metrics):
    if metrics is None:
        return {}
    if isinstance(metrics, dict):
        return metrics

    result = {}
    for attr in ("results_dict", "keys", "fitness"):
        if hasattr(metrics, attr):
            value = getattr(metrics, attr)
            if callable(value):
                continue
            if attr == "results_dict" and isinstance(value, dict):
                result.update(value)
            else:
                result[attr] = value
    return result


def main():
    args = parse_args()
    disable_integrations()

    from ultralytics import RTDETR
    from ultralytics.models.rtdetr import RTDETRValidator
    from ultralytics.models.rtdetr.train import RTDETRTrainer
    from ultralytics.utils import SETTINGS
    from ultralytics.utils import callbacks as ul_callbacks
    from ultralytics.utils import checks

    ul_callbacks.add_integration_callbacks = lambda instance: None
    checks.check_pip_update_available = lambda *unused_args, **unused_kwargs: None

    for key in ("wandb", "clearml", "comet", "dvc", "mlflow", "neptune", "raytune", "tensorboard", "hub"):
        if key in SETTINGS:
            SETTINGS[key] = False

    class BTCVRTDETRTrainer(RTDETRTrainer):
        def read_results_csv(self):
            results = {}
            if self.csv.exists():
                with open(self.csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        for key, value in row.items():
                            try:
                                parsed = float(value)
                            except (TypeError, ValueError):
                                parsed = value
                            results.setdefault(key.strip(), []).append(parsed)
            return results

        def final_eval(self):
            if self.best.exists():
                self.validator.args.plots = self.args.plots
                self.metrics = self.validator(model=self.best)
                if isinstance(self.metrics, dict):
                    self.metrics.pop("fitness", None)

    device = resolve_device(args.device)
    data_yaml = prepare_runtime_data_yaml((ROOT / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data))
    model_yaml = (ROOT / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)

    summary_path = Path(args.resume_summary) if args.resume_summary else ROOT / args.project / args.name / "trial_summary.json"
    summary_path = summary_path.resolve()

    payload = {
        "status": "started",
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "device": device,
        "data": str(data_yaml),
        "model": str(model_yaml),
        "args": vars(args),
    }
    write_summary(summary_path, payload)
    try:
        model = RTDETR(str(model_yaml))
        metrics = model.train(
            trainer=BTCVRTDETRTrainer,
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            project=str((ROOT / args.project).resolve()),
            name=args.name,
            exist_ok=args.exist_ok,
            cache=args.cache,
            fraction=args.fraction,
            patience=args.patience,
            seed=args.seed,
            pretrained=False,
            optimizer="AdamW",
            lr0=1e-4,
            weight_decay=5e-4,
            cos_lr=False,
            close_mosaic=0,
            amp=False,
            deterministic=False,
            verbose=True,
            plots=args.plots,
            save=True,
            save_json=False,
            val=True,
        )

        run_dir = Path(model.trainer.save_dir)
        summary = {
            **payload,
            "status": "ok",
            "run_dir": str(run_dir),
            "best_checkpoint": str(model.trainer.best),
            "last_checkpoint": str(model.trainer.last),
            "metrics": extract_metrics(metrics),
        }
        write_summary(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        summary = {
            **payload,
            "status": "blocked",
            "blocker": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
        }
        write_summary(summary_path, summary)
        raise


if __name__ == "__main__":
    main()
