#!/usr/bin/env python3
import argparse
import copy
import os
import sys
from pathlib import Path


DEFAULT_CFG = "configs/1232_final_diffusion_dit_v4_6c_geom_bridge_scratch_noresample_gpu2.yaml"
ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Single-sample geom-bridge diagnostic")
    parser.add_argument("--cfg", default=DEFAULT_CFG)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--n_samples", type=int, default=16)
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


def extract_compacted_init(batch, snake_gcn_utils):
    init = snake_gcn_utils.prepare_training({}, batch)
    py_ind = init.get("py_ind", init.get("ind"))
    if py_ind is None:
        raise RuntimeError("prepare_training did not return py_ind/ind.")
    return {
        "i_it_py": init["i_it_py"],
        "c_it_py": init["c_it_py"],
        "py_ind": py_ind,
        "i_gt_py": init["i_gt_py"],
    }


def align_gt_to_training_target(i_init_py, i_gt_py, optimal_cyclic_align, torch):
    def _signed_area(poly):
        x, y = poly[..., 0], poly[..., 1]
        x1, y1 = torch.roll(x, -1, 1), torch.roll(y, -1, 1)
        return 0.5 * torch.sum(x * y1 - x1 * y, dim=1)

    i_gt_aligned = i_gt_py.clone()
    area_init = _signed_area(i_init_py)
    area_gt = _signed_area(i_gt_aligned)
    orient_mismatch = ((area_init >= 0) ^ (area_gt >= 0))
    if orient_mismatch.any():
        flipped_gt = torch.flip(i_gt_aligned, dims=[1])
        i_gt_aligned = torch.where(orient_mismatch.view(-1, 1, 1), flipped_gt, i_gt_aligned)

    if optimal_cyclic_align:
        n_pts = i_gt_aligned.size(1)
        batch_size = i_gt_aligned.size(0)
        with torch.no_grad():
            oct_pts = i_init_py.detach()
            gt_pts = i_gt_aligned.detach()
            shift_costs = torch.zeros(batch_size, n_pts, device=i_init_py.device, dtype=i_init_py.dtype)
            for k in range(n_pts):
                diff = oct_pts - torch.roll(gt_pts, -k, 1)
                shift_costs[:, k] = diff.pow(2).sum(dim=(1, 2))
            best_k = shift_costs.argmin(dim=1)
        i_gt_aligned = torch.stack(
            [torch.roll(i_gt_aligned[i], -int(best_k[i].item()), 0) for i in range(batch_size)],
            dim=0,
        )
    else:
        d2 = (i_init_py[:, :1, :] - i_gt_aligned).pow(2).sum(-1)
        i_gt_aligned = torch.stack(
            [torch.roll(i_gt_aligned[i], -int(d2[i].argmin().item()), 0) for i in range(i_gt_aligned.size(0))],
            dim=0,
        )

    return i_gt_aligned


def contour_range_str(poly):
    return (
        f"x=[{float(poly[..., 0].min().item()):.3f},{float(poly[..., 0].max().item()):.3f}] "
        f"y=[{float(poly[..., 1].min().item()):.3f},{float(poly[..., 1].max().item()):.3f}]"
    )


def disp_range_str(disp):
    return (
        f"x=[{float(disp[..., 0].min().item()):.6f},{float(disp[..., 0].max().item()):.6f}] "
        f"y=[{float(disp[..., 1].min().item()):.6f},{float(disp[..., 1].max().item()):.6f}]"
    )


def calc_single_iou(pred_poly, gt_poly, down_ratio, image_h, image_w, cv2, np):
    pred_np = pred_poly.detach().cpu().numpy() * down_ratio
    gt_np = gt_poly.detach().cpu().numpy() * down_ratio
    return polygon_iou(pred_np, gt_np, image_h, image_w, cv2, np)


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
        print("CUDA is not available")
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
        raise RuntimeError("FlowMatchingEvolution module not found.")

    if not bool(getattr(evolution, "_geom_bridge", False)):
        raise RuntimeError("Config is not using flow_geom_bridge=true.")
    if bool(getattr(evolution, "_resample_feat_at_xt", False)):
        raise RuntimeError("Config is not fixed-feature; expected flow_resample_feat_at_xt=false.")

    data_loader = make_data_loader(cfg, is_train=True)
    raw_batch = next(iter(data_loader))
    batch, kept_instances = trim_batch_instances(copy.deepcopy(raw_batch), args.n_samples, torch)
    if kept_instances <= 0:
        raise RuntimeError("No valid training instances found in first batch.")
    batch = trainer.to_cuda(batch)

    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    net.train()

    initial_loss = None
    final_loss = None
    last_train_output = None

    for step in range(1, int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        output = net(batch["inp"], batch)
        if "diff_loss" not in output:
            raise RuntimeError("Network output does not contain diff_loss.")
        diff_loss = output["diff_loss"]
        loss_value = float(diff_loss.detach().cpu().item())
        if initial_loss is None:
            initial_loss = loss_value
        final_loss = loss_value
        last_train_output = output

        diff_loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), 40.0)
        optimizer.step()

        if step % 50 == 0 or step == 1 or step == int(args.steps):
            print(f"[train] step={step:04d} diff_loss={loss_value:.8f}")

    init_compacted = extract_compacted_init(batch, snake_gcn_utils)
    i_it_py = init_compacted["i_it_py"]
    c_it_py = init_compacted["c_it_py"]
    py_ind = init_compacted["py_ind"]
    i_gt_raw = init_compacted["i_gt_py"]
    i_gt_aligned = align_gt_to_training_target(
        i_it_py,
        i_gt_raw,
        bool(getattr(evolution, "_optimal_cyclic_align", False)),
        torch,
    )

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

    image0 = batch["orig_img"][0]
    if torch.is_tensor(image0):
        image0 = image0.detach().cpu().numpy()
    image_h, image_w = image0.shape[:2]
    down_ratio = float(snake_config.down_ratio)

    single_init = i_it_py[:1]
    single_can = c_it_py[:1]
    single_ind = py_ind[:1]
    single_gt = i_gt_aligned[:1]
    contour_scale = evolution.compute_contour_scale(single_init)

    with torch.no_grad():
        zero_norm = torch.zeros((1, single_init.size(1), 2), device=single_init.device, dtype=single_init.dtype)
        zero_denorm = evolution.denormalize_pred_disp(zero_norm, contour_scale)

        init_iou = calc_single_iou(single_init[0], single_gt[0], down_ratio, image_h, image_w, cv2, np)

        disp_single = evolution.sample_disp(
            cnn_feature,
            single_init,
            single_can,
            single_ind,
            steps=10,
            batch=batch,
        )
        pred_single = single_init + disp_single
        single_iou = calc_single_iou(pred_single[0], single_gt[0], down_ratio, image_h, image_w, cv2, np)

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
        iter_ode_steps = int(getattr(cfg, "iterative_ode_steps", getattr(cfg, "iterative_ddim_steps", evolution.ode_steps)))
        if iter_ode_steps <= 0:
            iter_ode_steps = evolution.ode_steps

        disp_iter = evolution.sample_disp_iterative(
            cnn_feature,
            single_init,
            single_can,
            single_ind,
            num_iter_steps=iter_steps,
            fractions=fractions,
            ode_steps=iter_ode_steps,
            batch=batch,
        )
        pred_iter = single_init + disp_iter
        iter_iou = calc_single_iou(pred_iter[0], single_gt[0], down_ratio, image_h, image_w, cv2, np)

        ideal_iou = calc_single_iou(single_gt[0], single_gt[0], down_ratio, image_h, image_w, cv2, np)

        net.train()
        evolution.train()
        x1_raw = single_gt - single_init
        x1 = evolution.normalize_target_disp(x1_raw, contour_scale)
        t_scalar = 0.99
        t_tensor = torch.full((1,), t_scalar, device=single_init.device, dtype=single_init.dtype)
        x_t = x1 * t_scalar
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        sampled_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, single_init, single_ind, h, w)
        detail_feat = evolution.sample_detail_features(
            cnn_feature,
            single_init,
            single_ind,
            h,
            w,
            sampled_feat=sampled_feat,
            contour_scale=contour_scale,
        )
        locate_context = None
        if bool(getattr(evolution, "_locate_token_enabled", False)):
            locate_context = evolution.build_locate_token_context(
                batch,
                single_init,
                single_ind,
                contour_scale=contour_scale,
            )
        v_pred, _ = evolution.predict_velocity(
            cnn_feature,
            single_init,
            single_can,
            sampled_feat,
            detail_feat,
            single_ind,
            x_t,
            t_tensor,
            contour_scale=contour_scale.view(-1),
            x_self_cond=None,
            locate_context=locate_context,
        )
        x1_pred = x_t + (1.0 - t_scalar) * v_pred
        pred_disp_train = evolution.denormalize_pred_disp(x1_pred, contour_scale)
        pred_disp_train = evolution.clamp_pred_disp(pred_disp_train, single_init)
        pred_disp_train_for_contours = pred_disp_train
        if bool(getattr(evolution, "_use_disp_gate", False)) and getattr(evolution, "disp_gate_head", None) is not None:
            gate = evolution.predict_disp_gate(sampled_feat, pred_disp_train, contour_scale)
            if bool(getattr(evolution, "_disp_gate_apply_training_pred", False)):
                pred_disp_train_for_contours = pred_disp_train * gate
        pred_train = single_init + pred_disp_train_for_contours
        train_iou = calc_single_iou(pred_train[0], single_gt[0], down_ratio, image_h, image_w, cv2, np)
        net.eval()
        evolution.eval()

    print("[diag] cfg=", cfg_path)
    print(
        "[diag] losses "
        f"initial={initial_loss:.8f} "
        f"final={final_loss:.8f} "
        f"last_train_pred_disp_abs_mean={float(last_train_output['pred_disp'][:1].abs().mean().detach().cpu().item()):.8f}"
    )
    print(f"[diag] init_range {contour_range_str(single_init[0])}")
    print(f"[diag] gt_aligned_range {contour_range_str(single_gt[0])}")
    print(f"[diag] single_bridge_disp_range {disp_range_str(disp_single[0])}")
    print(
        "[diag] contour_scale="
        f"{float(contour_scale.view(-1)[0].detach().cpu().item()):.6f} "
        f"zero_denorm_absmax={float(zero_denorm.abs().max().detach().cpu().item()):.8f}"
    )
    print(
        "[diag] compare "
        f"init_iou={init_iou:.6f} "
        f"single_bridge_iou={single_iou:.6f} "
        f"iter_bridge_iou={iter_iou:.6f} "
        f"ideal_iou={ideal_iou:.6f} "
        f"train_self_iou_t099={train_iou:.6f}"
    )
    print(
        "[diag] disp_abs_mean "
        f"single_bridge={float(disp_single.abs().mean().detach().cpu().item()):.8f} "
        f"iter_bridge={float(disp_iter.abs().mean().detach().cpu().item()):.8f} "
        f"train_self={float(pred_disp_train_for_contours.abs().mean().detach().cpu().item()):.8f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
