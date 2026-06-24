#!/usr/bin/env python3
import argparse
import copy
import os
import sys
from pathlib import Path


DEFAULT_CFG = "configs/1232_final_diffusion_dit_v4_6c_geom_bridge_scratch_gpu0.yaml"
ROOT = Path(__file__).resolve().parents[1]
GCF_RECORDS = []
GCF_CAPTURE_ENABLED = False


def parse_args():
    parser = argparse.ArgumentParser(description="M1 geom-bridge sanity overfit")
    parser.add_argument("--cfg", default=DEFAULT_CFG)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--n_samples", type=int, default=8)
    return parser.parse_args()


def resolve_cfg_path(cfg_path: str) -> str:
    path = Path(cfg_path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return str(path)


def polygon_to_mask(poly, h, w, cv2, np):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly).astype(np.int32)
    if pts.shape[0] >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask


def polygon_iou(poly_a, poly_b, h, w, cv2, np):
    mask_a = polygon_to_mask(poly_a, h, w, cv2, np)
    mask_b = polygon_to_mask(poly_b, h, w, cv2, np)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def trim_batch_instances(batch, max_instances, torch):
    max_instances = max(int(max_instances), 1)
    ct_num = batch["meta"]["ct_num"].clone().to(dtype=torch.int64)
    batch_size = int(ct_num.numel())
    keep_counts = []
    remaining = max_instances
    for i in range(batch_size):
        keep = min(int(ct_num[i].item()), remaining)
        keep_counts.append(keep)
        remaining -= keep
    if sum(keep_counts) <= 0:
        keep_counts[0] = min(int(ct_num[0].item()), 1)

    contour_fields = [
        "wh", "ct_cls", "ct_ind", "ct_01",
        "i_it_4py", "c_it_4py", "i_gt_4py", "c_gt_4py",
        "i_it_py", "c_it_py", "i_gt_py", "c_gt_py",
        "point_mask", "eagle_i_gt_4py", "eagle_4py_mask",
    ]
    for name in contour_fields:
        value = batch.get(name)
        if not torch.is_tensor(value) or value.dim() < 2 or value.size(0) != batch_size:
            continue
        for bi, keep in enumerate(keep_counts):
            if keep < value.size(1):
                value[bi, keep:] = 0

    new_ct_01 = torch.zeros_like(batch["ct_01"])
    for bi, keep in enumerate(keep_counts):
        if keep > 0:
            new_ct_01[bi, :keep] = 1
    batch["ct_01"] = new_ct_01.to(dtype=batch["ct_01"].dtype)
    batch["meta"]["ct_num"] = torch.tensor(keep_counts, dtype=batch["meta"]["ct_num"].dtype)

    if all(k in batch for k in ("bboxes", "cls", "batch_idx")):
        keep_mask = torch.zeros(batch["batch_idx"].shape[0], dtype=torch.bool)
        taken = [0 for _ in range(batch_size)]
        batch_idx = batch["batch_idx"].view(-1).to(dtype=torch.int64)
        for row_idx, sample_idx in enumerate(batch_idx.tolist()):
            if 0 <= sample_idx < batch_size and taken[sample_idx] < keep_counts[sample_idx]:
                keep_mask[row_idx] = True
                taken[sample_idx] += 1
        batch["bboxes"] = batch["bboxes"][keep_mask]
        batch["cls"] = batch["cls"][keep_mask]
        batch["batch_idx"] = batch["batch_idx"][keep_mask]

    return batch, int(sum(keep_counts))


def extract_compacted_gt(batch, snake_gcn_utils):
    gt = snake_gcn_utils.prepare_training({}, batch)["i_gt_py"]
    return gt


def extract_compacted_init(batch, snake_gcn_utils):
    init = snake_gcn_utils.prepare_training({}, batch)
    py_ind = init.get("py_ind", init.get("ind"))
    if py_ind is None:
        raise RuntimeError("prepare_training did not return py_ind/ind for eval init.")
    return {
        "i_it_py": init["i_it_py"],
        "c_it_py": init["c_it_py"],
        "py_ind": py_ind,
        "i_gt_py": init["i_gt_py"],
    }


def main():
    args = parse_args()
    cfg_path = resolve_cfg_path(args.cfg)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CFG_FILE"] = cfg_path
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(ROOT))

    import cv2
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        print("M1 FAIL: CUDA is not available")
        return 1

    from lib.config import cfg
    from lib.datasets import make_data_loader
    from lib.networks import make_network
    from lib.networks.diffusion.flow_matching_evolution import FlowMatchingEvolution
    from lib.train.trainers import make_trainer
    from lib.utils.snake import snake_config, snake_gcn_utils

    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True
    cfg.train.batch_size = max(1, min(int(args.n_samples), int(cfg.train.batch_size)))
    cfg.train.num_workers = 0
    cfg.dataloader_persistent_workers = False
    cfg.dataloader_prefetch_factor = 0

    original_get_gcn_feature = snake_gcn_utils.get_gcn_feature

    def recording_get_gcn_feature(cnn_feature, img_poly, ind, h, w):
        global GCF_RECORDS
        if GCF_CAPTURE_ENABLED and torch.is_tensor(img_poly) and img_poly.numel() > 0:
            GCF_RECORDS.append({
                "x_min": float(img_poly[..., 0].min().detach().cpu().item()),
                "x_max": float(img_poly[..., 0].max().detach().cpu().item()),
                "y_min": float(img_poly[..., 1].min().detach().cpu().item()),
                "y_max": float(img_poly[..., 1].max().detach().cpu().item()),
                "h": int(h),
                "w": int(w),
            })
        return original_get_gcn_feature(cnn_feature, img_poly, ind, h, w)

    snake_gcn_utils.get_gcn_feature = recording_get_gcn_feature

    try:
        torch.cuda.set_device(0)

        network = make_network(cfg)
        trainer = make_trainer(cfg, network)
        net = trainer.network.net
        evolution = None
        for module in net.modules():
            if isinstance(module, FlowMatchingEvolution):
                evolution = module
                break
        if evolution is None:
            raise RuntimeError("FlowMatchingEvolution module not found in network.")
        net.train()

        data_loader = make_data_loader(cfg, is_train=True)
        raw_batch = next(iter(data_loader))
        batch, kept_instances = trim_batch_instances(copy.deepcopy(raw_batch), args.n_samples, torch)
        if kept_instances <= 0:
            raise RuntimeError("No valid training instances found in first batch.")
        batch = trainer.to_cuda(batch)

        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)

        initial_loss = None
        final_loss = None

        global GCF_RECORDS
        global GCF_CAPTURE_ENABLED
        GCF_RECORDS = []

        for step in range(1, int(args.steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            GCF_CAPTURE_ENABLED = (step == 1)
            output = net(batch["inp"], batch)
            GCF_CAPTURE_ENABLED = False

            if "diff_loss" not in output:
                raise RuntimeError("Network output does not contain diff_loss during training.")
            diff_loss = output["diff_loss"]
            loss_value = float(diff_loss.detach().cpu().item())
            if initial_loss is None:
                initial_loss = loss_value
            final_loss = loss_value

            diff_loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), 40.0)
            optimizer.step()

            if step % 50 == 0 or step == 1 or step == int(args.steps):
                print(f"[train] step={step:04d} diff_loss={loss_value:.6f}")

        if not GCF_RECORDS:
            raise RuntimeError("No get_gcn_feature calls were recorded on step 1.")

        feat_x_min = min(r["x_min"] for r in GCF_RECORDS)
        feat_x_max = max(r["x_max"] for r in GCF_RECORDS)
        feat_y_min = min(r["y_min"] for r in GCF_RECORDS)
        feat_y_max = max(r["y_max"] for r in GCF_RECORDS)
        feat_min = min(feat_x_min, feat_y_min)
        feat_max = max(feat_x_max, feat_y_max)

        for rec in GCF_RECORDS:
            if rec["x_min"] < -8.0 or rec["x_max"] > rec["w"] + 8.0 or rec["y_min"] < -8.0 or rec["y_max"] > rec["h"] + 8.0:
                raise AssertionError(
                    "get_gcn_feature received out-of-grid coordinates: "
                    f"x=[{rec['x_min']:.3f},{rec['x_max']:.3f}] "
                    f"y=[{rec['y_min']:.3f},{rec['y_max']:.3f}] "
                    f"hw=({rec['h']},{rec['w']})"
                )

        init_compacted = extract_compacted_init(batch, snake_gcn_utils)
        gt_compacted = init_compacted["i_gt_py"]
        i_it_py = init_compacted["i_it_py"]
        c_it_py = init_compacted["c_it_py"]
        py_ind = init_compacted["py_ind"]

        captured = {}

        def capture_cnn_feature(module, inputs, output):
            if len(inputs) >= 2 and torch.is_tensor(inputs[1]):
                captured["cnn_feature"] = inputs[1].detach()

        hook_handle = evolution.register_forward_hook(capture_cnn_feature)
        try:
            net.eval()
            with torch.no_grad():
                net(batch["inp"], batch)
        finally:
            hook_handle.remove()

        cnn_feature = captured.get("cnn_feature")
        if cnn_feature is None:
            raise RuntimeError("Failed to capture cnn_feature from FlowMatchingEvolution forward.")

        with torch.no_grad():
            if evolution.use_iterative_refinement:
                iter_steps = int(getattr(cfg, "iterative_num_steps", 3))
                use_rich_infer_schedule = bool(getattr(cfg, "v4_9_use_rich_infer_schedule", False))
                if use_rich_infer_schedule:
                    fractions = list(getattr(cfg, "v4_9_infer_target_fractions", []))
                    if not fractions:
                        fractions = [0.3333, 0.5, 0.80, 0.97, 1.0]
                    fractions = evolution._progress_targets_to_residual_fractions(fractions)
                    iter_steps = len(fractions)
                else:
                    fractions = list(getattr(cfg, "iterative_fractions", []))
                if not fractions:
                    fractions = [1.0 / (iter_steps - i) for i in range(iter_steps)]
                iter_ode_steps = int(
                    getattr(
                        cfg,
                        "iterative_ode_steps",
                        getattr(cfg, "iterative_ddim_steps", evolution.ode_steps),
                    )
                )
                if iter_ode_steps <= 0:
                    iter_ode_steps = evolution.ode_steps
                disp = evolution.sample_disp_iterative(
                    cnn_feature,
                    i_it_py,
                    c_it_py,
                    py_ind,
                    num_iter_steps=iter_steps,
                    fractions=fractions,
                    ode_steps=iter_ode_steps,
                    batch=batch,
                )
            else:
                disp = evolution.sample_disp(
                    cnn_feature,
                    i_it_py,
                    c_it_py,
                    py_ind,
                    steps=evolution.ode_steps,
                    batch=batch,
                )
            if evolution.use_fourier_smooth > 0:
                disp = evolution.fourier_smooth(disp, evolution.use_fourier_smooth)
            py = i_it_py + disp

        pred_np = py.detach().cpu().numpy() * float(snake_config.down_ratio)
        init_np = i_it_py.detach().cpu().numpy() * float(snake_config.down_ratio)
        gt_np = gt_compacted.detach().cpu().numpy() * float(snake_config.down_ratio)
        num_eval = min(len(pred_np), len(gt_np))
        if num_eval <= 0:
            raise RuntimeError("No valid contours available for IoU evaluation.")

        image0 = batch["orig_img"][0]
        if torch.is_tensor(image0):
            image0 = image0.detach().cpu().numpy()
        image_h, image_w = image0.shape[:2]
        baseline_ious = [
            polygon_iou(init_np[i], gt_np[i], image_h, image_w, cv2, np)
            for i in range(num_eval)
        ]
        ious = [
            polygon_iou(pred_np[i], gt_np[i], image_h, image_w, cv2, np)
            for i in range(num_eval)
        ]
        baseline_mean_iou = float(np.mean(baseline_ious))
        mean_iou = float(np.mean(ious))

        loss_ratio = float(initial_loss / max(final_loss, 1e-12))

        print(
            "[summary] "
            f"initial_loss={initial_loss:.6f} "
            f"final_loss={final_loss:.6f} "
            f"loss_drop={loss_ratio:.3f}x "
            f"baseline_iou={baseline_mean_iou:.4f} "
            f"mean_iou={mean_iou:.4f} "
            f"feat_poly_min={feat_min:.3f} "
            f"feat_poly_max={feat_max:.3f}"
        )

        passed = (loss_ratio > 5.0) and (mean_iou > 0.85)
        if passed:
            print("M1 PASS")
            return 0
        print("M1 FAIL")
        return 1
    finally:
        snake_gcn_utils.get_gcn_feature = original_get_gcn_feature


if __name__ == "__main__":
    raise SystemExit(main())
