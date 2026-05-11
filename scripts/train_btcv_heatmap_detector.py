import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train BTCV CenterNet-style heatmap detector baseline.")
    parser.add_argument("--cfg", default="configs/btcv_heatmap_resnet18_detect_only.yaml")
    parser.add_argument("--device", default="auto", help="CUDA device index or auto.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-ep", type=int, default=1)
    parser.add_argument("--heatmap-backbone", default="resnet18")
    parser.add_argument("--heatmap-pretrained", action="store_true")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--resume", action="store_true")
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


def main():
    args = parse_args()
    device = resolve_device(args.device)
    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device

    cmd = [
        sys.executable,
        "train_net.py",
        "--cfg_file",
        str(cfg_path),
        "gpus",
        "[0]",
        "resume",
        "True" if args.resume else "False",
        "train.epoch",
        str(args.epochs),
        "train.batch_size",
        str(args.batch),
        "train.num_workers",
        str(args.workers),
        "train.lr",
        str(args.lr),
        "train.save_ep",
        str(args.save_ep),
        "detector_backend",
        "heatmap_resnet18",
        "heatmap_backbone",
        str(args.heatmap_backbone),
        "heatmap_pretrained",
        "True" if args.heatmap_pretrained else "False",
    ]
    if args.model_dir:
        cmd.extend(["model_dir", args.model_dir])

    print("[train-cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
