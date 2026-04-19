import os
import sys
import cv2
import torch
import random
import argparse
import datetime
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# 默认配置文件引导 (防止 lib.config 找不到默认项)
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if not os.environ.get("CFG_FILE"):
    os.environ["CFG_FILE"] = os.path.join(_REPO_ROOT, "configs", "btcv_diffusion_dit_v3_4_single_overfit.yaml")

from lib.config import cfg as global_cfg
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.datasets.collate_batch import make_collator
from lib.utils.snake import snake_config


BASE_CFG = global_cfg.clone()


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def reset_and_merge_cfg(cfg_path):
    global_cfg.defrost()
    global_cfg.merge_from_other_cfg(BASE_CFG)
    global_cfg.merge_from_file(cfg_path)
    global_cfg.use_diffusion_evolution = True
    return global_cfg


def find_checkpoint_path(cfg_path, cfg_obj):
    by_model_dir = os.path.join(_REPO_ROOT, str(cfg_obj.model_dir), "checkpoints", "latest.pt")
    if os.path.exists(by_model_dir):
        return by_model_dir

    cfg_stem = Path(cfg_path).stem
    by_stem = os.path.join(_REPO_ROOT, "data", "outputs", cfg_stem, "checkpoints", "latest.pt")
    if os.path.exists(by_stem):
        return by_stem
    return None


def load_version_model(cfg_path):
    cfg_obj = reset_and_merge_cfg(cfg_path)
    network = make_network(cfg_obj)
    trainer = make_trainer(cfg_obj, network)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = find_checkpoint_path(cfg_path, cfg_obj)
    if ckpt_path is None:
        print(f"[!] Warning: No checkpoint found for {Path(cfg_path).stem}, skipping.")
        return None, None, None

    print(f"[*] Loading {Path(cfg_path).stem} from: {ckpt_path}")
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt_obj.get("state_dict") or ckpt_obj.get("model") or ckpt_obj.get("net") or ckpt_obj

    try:
        from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
        sd = remap_legacy_state_dict(sd)
    except Exception:
        pass

    model = trainer.network.module if hasattr(trainer.network, "module") else trainer.network
    info = model.load_state_dict(sd, strict=False)
    if len(info.missing_keys) > 0 or len(info.unexpected_keys) > 0:
        print(f"[!] load_state_dict: missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")

    model = trainer.network.to(device).eval()
    return model, device, cfg_obj


def build_batch_for_cfg(cfg_obj, index):
    dataset = make_dataset(cfg_obj, cfg_obj.test.dataset, make_transforms(cfg_obj, is_train=False), is_train=False)
    if len(dataset) == 0:
        raise RuntimeError(f"Empty dataset for cfg: {cfg_obj.test.dataset}")
    idx = min(max(int(index), 0), len(dataset) - 1)
    sample = dataset[idx]
    batch_raw = make_collator(cfg_obj)([sample])
    return batch_raw, idx, len(dataset)


def run_inference(model, device, batch_raw):
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch_raw.items()}
    dr = float(snake_config.down_ratio)

    with torch.no_grad():
        ret = model(batch)
        output = ret[0] if isinstance(ret, (list, tuple)) else ret

    if not isinstance(output, dict):
        return None

    if isinstance(output.get("py"), torch.Tensor):
        pred_t = output["py"]
    elif isinstance(output.get("py_pred"), (list, tuple)) and len(output["py_pred"]) > 0:
        pred_t = output["py_pred"][-1]
    else:
        return None

    # GT / Init 使用批次中准备好的轮廓（与该 cfg 的数据处理一致）
    ct_num = 0
    if isinstance(batch.get("meta"), dict) and isinstance(batch["meta"].get("ct_num"), torch.Tensor):
        ct_num = int(batch["meta"]["ct_num"][0].item())

    gt_src = batch["i_gt_py"][0]
    init_src = batch["i_it_py"][0]
    if ct_num > 0:
        gt_src = gt_src[:ct_num]
        init_src = init_src[:ct_num]

    pred_np = to_numpy(pred_t) * dr
    init_np = to_numpy(init_src) * dr
    gt_np = to_numpy(gt_src) * dr

    img_item = batch_raw["orig_img"][0]
    img_np = to_numpy(img_item).astype(np.uint8)

    return {
        "pred": pred_np,
        "init": init_np,
        "gt": gt_np,
        "img": img_np,
    }


def draw_and_save(result, save_path):
    out_img = result["img"].copy()
    gt = result["gt"]
    init = result["init"]
    pred = result["pred"]

    for p in gt:
        cv2.polylines(out_img, [p.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)   # GT blue
    for p in init:
        cv2.polylines(out_img, [p.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=1)  # Init yellow
    for p in pred:
        cv2.polylines(out_img, [p.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)    # Pred red

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, out_img)


def default_single_overfit_cfgs():
    candidates = [
        "configs/btcv_diffusion_dit_v3_1_single_overfit.yaml",
        "configs/btcv_diffusion_dit_v3_2_single_overfit.yaml",
        "configs/btcv_diffusion_dit_v3_4_single_overfit.yaml",
        "configs/btcv_diffusion_dit_v3_5_single_overfit.yaml",
        "configs/btcv_diffusion_dit_v3_6_single_overfit.yaml",
    ]
    return [os.path.join(_REPO_ROOT, x) for x in candidates if os.path.exists(os.path.join(_REPO_ROOT, x))]


def parse_args():
    parser = argparse.ArgumentParser(description="Fair comparison inference across versions")
    parser.add_argument("--index", type=int, default=-1, help="sample index, -1 for random")
    parser.add_argument("--seed", type=int, default=42, help="random seed when index=-1")
    parser.add_argument("--out_dir", type=str, default="", help="output directory")
    parser.add_argument("--configs", nargs="*", default=None, help="optional custom config list")
    return parser.parse_args()


def main():
    args = parse_args()

    configs = [os.path.abspath(c) for c in args.configs] if args.configs else default_single_overfit_cfgs()
    if not configs:
        raise RuntimeError("No config files found for comparison.")

    # 以第一份配置确定索引范围
    first_cfg = reset_and_merge_cfg(configs[0])
    first_dataset = make_dataset(first_cfg, first_cfg.test.dataset, make_transforms(first_cfg, is_train=False), is_train=False)
    if len(first_dataset) == 0:
        raise RuntimeError(f"Empty dataset for first cfg: {configs[0]}")

    if args.index < 0:
        random.seed(args.seed)
        index = random.randint(0, len(first_dataset) - 1)
    else:
        index = min(max(args.index, 0), len(first_dataset) - 1)

    save_dir = args.out_dir or os.path.join(_REPO_ROOT, "visual", "all_version_comparison_fair")
    ts = datetime.datetime.now().strftime("%m%d_%H%M")

    print("=" * 80)
    print(f"[*] Fair comparison start, sample index: {index}")
    print(f"[*] Output dir: {save_dir}")
    print("=" * 80)

    for cfg_path in configs:
        if not os.path.exists(cfg_path):
            print(f"[!] Missing config, skip: {cfg_path}")
            continue

        model, device, cfg_obj = load_version_model(cfg_path)
        if model is None:
            continue

        batch_raw, idx_used, ds_len = build_batch_for_cfg(cfg_obj, index)
        result = run_inference(model, device, batch_raw)
        if result is None:
            print(f"[!] Inference failed to produce polygons: {Path(cfg_path).stem}")
            continue

        v_name = Path(cfg_path).stem.replace("btcv_diffusion_dit_", "")
        save_path = os.path.join(save_dir, f"{ts}_idx{idx_used}_{v_name}.png")
        draw_and_save(result, save_path)
        print(f"[#] {v_name}: dataset_size={ds_len}, idx={idx_used}, saved={save_path}")

    print("\n" + "=" * 80)
    print(f"[*] Done. Results saved in: {save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
