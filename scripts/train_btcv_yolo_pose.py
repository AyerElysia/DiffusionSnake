import argparse
import csv
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from io import BytesIO
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.networks.YOLOV8.data.build import build_yolo_dataset
from lib.networks.YOLOV8.engine.trainer import __version__, convert_optimizer_state_dict_to_fp16
from lib.networks.YOLOV8.models.yolo.pose import PoseTrainer
from lib.networks.YOLOV8.nn.tasks import PoseModel, yaml_model_load
from lib.networks.YOLOV8.utils import callbacks as yolo_callbacks
from lib.networks.YOLOV8.utils.torch_utils import de_parallel


def parse_args():
    parser = argparse.ArgumentParser(description="Train BTCV extreme-point YOLO pose model.")
    parser.add_argument("--data", default="data/exports/btcv_yolo/btcv_pose.yaml")
    parser.add_argument("--scale", choices=["n", "s", "m", "l", "x"], default="n")
    parser.add_argument("--device", default="auto", help="CUDA device index, comma list, cpu, or auto.")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--img-channels", type=int, default=3)
    parser.add_argument(
        "--pretrained",
        default="auto",
        help="Path to pretrained weights. 'auto' prefers data/pretrained/yolov8{scale}.pt if present.",
    )
    parser.add_argument("--project", default="data/outputs")
    parser.add_argument("--name", default="")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--augment-profile",
        choices=["medical-safe", "ultralytics"],
        default="medical-safe",
        help="medical-safe disables flips and heavy natural-image augmentations.",
    )
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


def model_yaml_for_scale(scale: str):
    pose_cfg_dir = ROOT / "lib" / "networks" / "YOLOV8" / "cfg" / "models" / "v8"
    return str(pose_cfg_dir / f"yolov8{scale}-pose.yaml")


def resolve_pretrained(scale: str, pretrained_arg: str):
    pretrained_arg = str(pretrained_arg).strip()
    if not pretrained_arg:
        return ""
    if pretrained_arg.lower() != "auto":
        return pretrained_arg
    candidate = ROOT / "data" / "pretrained" / f"yolov8{scale}.pt"
    if candidate.exists():
        return str(candidate)
    return ""


def disable_integrations():
    yolo_callbacks.add_integration_callbacks = lambda instance: None
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("COMET_DISABLE_AUTO_LOGGING", "1")


class BTCVPoseTrainer(PoseTrainer):
    def __init__(self, *args, model_channels=3, **kwargs):
        self.model_channels = int(model_channels)
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model_cfg = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg)
        model_cfg["ch"] = self.model_channels
        model = PoseModel(
            model_cfg,
            ch=model_cfg["ch"],
            nc=self.data["nc"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose,
        )
        if weights:
            model.load(weights)
        return model

    def save_model(self):
        results = {}
        if self.csv.exists():
            with open(self.csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key, value in row.items():
                        results.setdefault(key.strip(), []).append(float(value))

        buffer = BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": results,
                "date": datetime.now().isoformat(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()
        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)

    def final_eval(self):
        print("[btcv-pose] skipping Ultralytics final_eval(best.pt) to avoid extra pandas dependency in snake1.")


def apply_augment_profile(overrides: dict, profile: str):
    if profile != "medical-safe":
        return
    overrides.update(
        {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        }
    )


def ensure_file_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    dst.symlink_to(src.resolve())


def ensure_dir(path: Path):
    if path.is_symlink():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def prepare_pose_runtime_yaml(data_yaml_path: Path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data_root = Path(data["path"])
    if not data_root.is_absolute():
        data_root = (data_yaml_path.parent / data_root).resolve()

    generic_label_dir = data_root / "labels" / "train"
    pose_label_dir = data_root / "labels" / "pose" / "train"
    if generic_label_dir.exists():
        return data_yaml_path
    if not pose_label_dir.exists():
        raise FileNotFoundError(f"Expected pose labels at {pose_label_dir}")

    runtime_root = data_root.parent / f"{data_root.name}_pose_runtime"
    for split in ("train", "val"):
        runtime_img_dir = runtime_root / "images" / split
        runtime_label_dir = runtime_root / "labels" / split
        ensure_dir(runtime_img_dir)
        ensure_dir(runtime_label_dir)

        for src_img in sorted((data_root / "images" / split).glob("*")):
            if src_img.is_file():
                ensure_file_symlink(src_img, runtime_img_dir / src_img.name)
        for src_label in sorted((data_root / "labels" / "pose" / split).glob("*")):
            if src_label.is_file():
                dst_name = src_label.name
                if not src_label.stem.endswith("_image"):
                    dst_name = f"{src_label.stem}_image{src_label.suffix}"
                ensure_file_symlink(src_label, runtime_label_dir / dst_name)

    runtime_data = dict(data)
    runtime_data["path"] = str(runtime_root.resolve())
    runtime_yaml_path = runtime_root / "btcv_pose_runtime.yaml"
    with open(runtime_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_data, f, sort_keys=False)
    return runtime_yaml_path


def main():
    args = parse_args()
    disable_integrations()

    model_yaml = model_yaml_for_scale(args.scale)
    device = resolve_device(args.device)
    pretrained = resolve_pretrained(args.scale, args.pretrained)
    run_name = args.name or f"btcv_yolo_pose_{args.scale}"

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Pose dataset yaml not found: {data_path}")
    data_path = prepare_pose_runtime_yaml(data_path)

    overrides = {
        "task": "pose",
        "model": model_yaml,
        "data": str(data_path),
        "device": device,
        "imgsz": int(args.imgsz),
        "epochs": int(args.epochs),
        "batch": int(args.batch),
        "workers": int(args.workers),
        "patience": int(args.patience),
        "fraction": float(args.fraction),
        "project": str((ROOT / args.project).resolve()) if not Path(args.project).is_absolute() else args.project,
        "name": run_name,
        "exist_ok": bool(args.exist_ok),
        "cache": bool(args.cache),
        "seed": int(args.seed),
        "optimizer": args.optimizer,
        "lr0": float(args.lr0),
        "weight_decay": float(args.weight_decay),
        "pretrained": pretrained or False,
        "verbose": False,
        "amp": False,
        "plots": False,
        "close_mosaic": 0,
    }
    apply_augment_profile(overrides, args.augment_profile)

    print(f"[btcv-pose] data={overrides['data']}")
    print(f"[btcv-pose] model={model_yaml}")
    print(f"[btcv-pose] img_channels={args.img_channels}")
    print(f"[btcv-pose] pretrained={pretrained or 'none'}")
    print(f"[btcv-pose] device={device}")
    print(f"[btcv-pose] project={overrides['project']}")
    print(f"[btcv-pose] name={run_name}")

    trainer = BTCVPoseTrainer(overrides=overrides, model_channels=args.img_channels)
    trainer.train()


if __name__ == "__main__":
    main()
