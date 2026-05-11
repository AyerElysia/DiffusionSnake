import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

import cv2
import yaml

os.environ["CFG_FILE"] = os.environ.get("CFG_FILE", "configs/btcv_yolo_detect_only.yaml")
sys.argv = [sys.argv[0]]

from lib.utils.getedge import binary_mask_to_polygon
from lib.utils.snake import snake_voc_utils


CLASS_NAMES = [
    "spleen",
    "right_kidney",
    "left_kidney",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Export BTCV prepared masks to YOLO detect/pose labels.")
    parser.add_argument("--train-root", default="/home/medteam/Zhrch/Datasets/BTCV/btcv_png_new_snake")
    parser.add_argument("--val-root", default="/home/medteam/Zhrch/Datasets/BTCV/btcv_png_test_new_snake")
    parser.add_argument("--out-dir", default="data/exports/btcv_yolo")
    return parser.parse_args()


def read_split_images(root: str, split: str):
    candidates = ["train_list.txt", "test_list.txt", "val_list.txt"] if split == "train" else ["test_list.txt", "val_list.txt", "train_list.txt"]
    list_file = None
    for name in candidates:
        path = os.path.join(root, name)
        if os.path.exists(path):
            list_file = path
            break

    if list_file is not None:
        with open(list_file, "r", encoding="utf-8") as f:
            items = [x.strip() for x in f if x.strip()]
        return [os.path.join(root, x) if not os.path.isabs(x) else x for x in items]

    return sorted(glob.glob(os.path.join(root, "*_image.png")))


def image_stem(image_path: str):
    stem = Path(image_path).stem
    if stem.endswith("_image"):
        return stem[:-6]
    return stem


def ensure_link(src: str, dst: str):
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    try:
        dst_path.symlink_to(Path(src).resolve())
    except OSError:
        shutil.copy2(src, dst)


def norm_point(x: float, y: float, width: int, height: int):
    return x / max(float(width), 1.0), y / max(float(height), 1.0)


def export_split(root: str, split: str, out_dir: str):
    image_paths = read_split_images(root, split)
    detect_label_dir = Path(out_dir) / "labels" / "detect" / split
    pose_label_dir = Path(out_dir) / "labels" / "pose" / split
    image_dir = Path(out_dir) / "images" / split
    detect_label_dir.mkdir(parents=True, exist_ok=True)
    pose_label_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0
    object_count = 0

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]
        stem = image_stem(image_path)
        mask_paths = sorted(glob.glob(os.path.join(root, f"{stem}_mask*")))

        detect_lines = []
        pose_lines = []
        for mask_path in mask_paths:
            class_id = int(Path(mask_path).stem.split("_")[-1]) - 1
            polygons = binary_mask_to_polygon(mask_path)
            for poly in polygons:
                if isinstance(poly, (list, tuple)) and len(poly) == 1:
                    poly = poly[0]
                if poly is None or len(poly) < 3:
                    continue
                ext = snake_voc_utils.get_extreme_points(poly)
                x_min = float(poly[:, 0].min())
                y_min = float(poly[:, 1].min())
                x_max = float(poly[:, 0].max())
                y_max = float(poly[:, 1].max())
                bw = x_max - x_min
                bh = y_max - y_min
                if bw <= 1.0 or bh <= 1.0:
                    continue

                cx = (x_min + x_max) / 2.0
                cy = (y_min + y_max) / 2.0
                cx_n, cy_n = norm_point(cx, cy, width, height)
                bw_n, bh_n = bw / max(float(width), 1.0), bh / max(float(height), 1.0)

                detect_lines.append(f"{class_id} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

                keypoints = []
                for x, y in ext:
                    x_n, y_n = norm_point(float(x), float(y), width, height)
                    keypoints.append(f"{x_n:.6f} {y_n:.6f} 2")
                pose_lines.append(
                    f"{class_id} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f} " + " ".join(keypoints)
                )
                object_count += 1

        (detect_label_dir / f"{stem}.txt").write_text("\n".join(detect_lines) + ("\n" if detect_lines else ""), encoding="utf-8")
        (pose_label_dir / f"{stem}.txt").write_text("\n".join(pose_lines) + ("\n" if pose_lines else ""), encoding="utf-8")
        ensure_link(image_path, image_dir / Path(image_path).name)
        image_count += 1

    return {"split": split, "images": image_count, "objects": object_count}


def write_dataset_yaml(out_dir: str, task: str):
    data = {
        "path": str(Path(out_dir).resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    if task == "pose":
        data["kpt_shape"] = [4, 3]
        data["flip_idx"] = [0, 1, 2, 3]

    yaml_path = Path(out_dir) / f"btcv_{task}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main():
    args = parse_args()
    out_dir = args.out_dir
    summaries = [
        export_split(args.train_root, "train", out_dir),
        export_split(args.val_root, "val", out_dir),
    ]
    write_dataset_yaml(out_dir, "detect")
    write_dataset_yaml(out_dir, "pose")

    for summary in summaries:
        print(f"{summary['split']}: images={summary['images']} objects={summary['objects']}")
    print(f"export_dir={Path(out_dir).resolve()}")


if __name__ == "__main__":
    main()
