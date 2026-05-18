#!/usr/bin/env python3
"""
Full-dataset V3.7 contour-refinement evaluation.

This script mirrors the single-sample IoU logic in scripts/infer_v3_7_iou.py,
but runs on the full cfg.test.dataset split and saves:
1. aggregate metrics JSON
2. per-sample summary JSON
3. optional per-sample overlay images
"""

import datetime
import json
import os
import random
import sys

import cv2
import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _THIS_DIR)

_DEFAULT_CFG = os.path.join(_THIS_DIR, 'configs', 'btcv_diffusion_dit_v3_7_v6c_full_noleak.yaml')
if not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = _DEFAULT_CFG

from lib.config import cfg
from lib.datasets.collate_batch import make_collator
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.utils.snake import snake_config, snake_decode, snake_gcn_utils


def apply_gpu_override():
    eval_gpu = os.environ.get('EVAL_GPU', '').strip()
    if not eval_gpu:
        return
    cfg.gpus = [int(eval_gpu)]
    os.environ['CUDA_VISIBLE_DEVICES'] = eval_gpu
    print(f'[*] Override evaluation GPU -> {eval_gpu}')


def set_eval_seed(seed):
    if seed is None or seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    print(f'[*] Evaluation seed -> {seed}')


def poly_to_mask(poly_pts, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly_pts).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def compute_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def compute_dice_from_iou(iou):
    return float(2.0 * iou / (1.0 + iou)) if iou > 0 else 0.0


def extract_boundary(mask, tolerance):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.zeros_like(mask, dtype=np.uint8)
    if contours:
        cv2.drawContours(boundary, contours, -1, 1, thickness=int(tolerance))
    return boundary


def compute_boundary_dice(mask_a, mask_b, tolerance):
    ba = extract_boundary(mask_a, tolerance)
    bb = extract_boundary(mask_b, tolerance)
    inter = np.logical_and(ba, bb).sum()
    denom = ba.sum() + bb.sum()
    return float(2.0 * inter / denom) if denom > 0 else 0.0


def compute_mboundf(mask_a, mask_b):
    vals = [compute_boundary_dice(mask_a, mask_b, tolerance=t) for t in range(1, 11)]
    return float(np.mean(vals)) if vals else 0.0


def compute_ordered_or_matched_metrics(pred_polys, gt_polys, height, width, match_by_iou=False):
    gt_masks = [poly_to_mask(poly, height, width) for poly in gt_polys]
    pred_masks = [poly_to_mask(poly, height, width) for poly in pred_polys]

    if (not match_by_iou) and len(pred_masks) == len(gt_masks):
        matched_pred = list(range(len(gt_masks)))
    else:
        matched_pred = [-1 for _ in gt_masks]
        if pred_masks and gt_masks:
            iou_mat = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
            for gi, gt_mask in enumerate(gt_masks):
                for pi, pred_mask in enumerate(pred_masks):
                    iou_mat[gi, pi] = compute_iou(pred_mask, gt_mask)
            used_gt = set()
            used_pred = set()
            while len(used_gt) < len(gt_masks) and len(used_pred) < len(pred_masks):
                gi, pi = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[gi, pi] < 0:
                    break
                matched_pred[int(gi)] = int(pi)
                used_gt.add(int(gi))
                used_pred.add(int(pi))
                iou_mat[gi, :] = -1
                iou_mat[:, pi] = -1

    per_iou = []
    per_dice = []
    per_mboundf = []
    for gi, gt_mask in enumerate(gt_masks):
        pi = matched_pred[gi]
        if pi < 0 or pi >= len(pred_masks):
            per_iou.append(0.0)
            per_dice.append(0.0)
            per_mboundf.append(0.0)
            continue
        iou = compute_iou(pred_masks[pi], gt_mask)
        per_iou.append(iou)
        per_dice.append(compute_dice_from_iou(iou))
        per_mboundf.append(compute_mboundf(pred_masks[pi], gt_mask))

    return per_iou, per_dice, per_mboundf, matched_pred


def load_model(ckpt_path=None):
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if ckpt_path is None:
        cfg_stem = os.path.splitext(os.path.basename(os.environ.get('CFG_FILE', 'default')))[0]
        ckpt_path = os.path.join(_THIS_DIR, 'data', 'outputs', cfg_stem, 'checkpoints', 'latest.pt')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    print(f'[*] Loading checkpoint: {ckpt_path}')
    ckpt_obj = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt_obj.get('state_dict') or ckpt_obj.get('model') or ckpt_obj.get('net') or ckpt_obj

    from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
    sd = remap_legacy_state_dict(sd)

    wrapper = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    info = wrapper.load_state_dict(sd, strict=False)
    n_ok = len(sd) - len(info.missing_keys)
    print(f'[✔] Loaded {n_ok} / {len(sd)} keys (missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)})')
    return trainer.network.to(device).eval(), device, ckpt_path


def get_extreme_points_torch(pts, thresh=0.02):
    n_inst, n_pts, _ = pts.shape
    device = pts.device
    left = pts[..., 0].min(dim=-1)[0]
    top = pts[..., 1].min(dim=-1)[0]
    right = pts[..., 0].max(dim=-1)[0]
    bottom = pts[..., 1].max(dim=-1)[0]
    width = right - left + 1
    height = bottom - top + 1

    results = []
    for i in range(n_inst):
        poly_i = pts[i]

        def _find_ex(dim_idx, is_min, other_dim_range):
            if is_min:
                idx = torch.argmin(poly_i[:, dim_idx])
                val = poly_i[idx, dim_idx]

                def cond(j):
                    return poly_i[j, dim_idx] - val <= thresh * other_dim_range
            else:
                idx = torch.argmax(poly_i[:, dim_idx])
                val = poly_i[idx, dim_idx]

                def cond(j):
                    return val - poly_i[j, dim_idx] <= thresh * other_dim_range

            idxs = [idx.item()]
            tmp = (idx + 1) % n_pts
            while tmp != idx and cond(tmp):
                idxs.append(tmp.item())
                tmp = (tmp + 1) % n_pts
            tmp = (idx - 1) % n_pts
            while tmp != idx and cond(tmp):
                idxs.append(tmp.item())
                tmp = (tmp - 1) % n_pts
            return torch.tensor(idxs, device=device)

        t_idxs = _find_ex(1, True, height[i])
        tt_x = (poly_i[t_idxs, 0].max() + poly_i[t_idxs, 0].min()) / 2
        tt = torch.stack([tt_x, top[i]])

        b_idxs = _find_ex(1, False, height[i])
        bb_x = (poly_i[b_idxs, 0].max() + poly_i[b_idxs, 0].min()) / 2
        bb = torch.stack([bb_x, bottom[i]])

        l_idxs = _find_ex(0, True, width[i])
        ll_y = (poly_i[l_idxs, 1].max() + poly_i[l_idxs, 1].min()) / 2
        ll = torch.stack([left[i], ll_y])

        r_idxs = _find_ex(0, False, width[i])
        rr_y = (poly_i[r_idxs, 1].max() + poly_i[r_idxs, 1].min()) / 2
        rr = torch.stack([right[i], rr_y])

        results.append(torch.stack([tt, ll, bb, rr]))
    return torch.stack(results)


def build_init_polys(batch, gt_all):
    batch_size, num_contours, num_points, _ = gt_all.shape
    if 'i_it_py' in batch and batch['i_it_py'].numel() > 0:
        return batch['i_it_py'].view(-1, num_points, 2)

    poly_flat = gt_all.view(batch_size * num_contours, num_points, 2)
    ex = get_extreme_points_torch(poly_flat)
    init_polys = snake_decode.get_octagon(ex).view(batch_size, num_contours, 12, 2)
    return snake_gcn_utils.uniform_upsample(init_polys, 128)[0]


def build_sam_gt_box_init_polys(batch, gt_all, device):
    batch_size, num_contours, num_points, _ = gt_all.shape
    dr = float(snake_config.down_ratio)
    orig_imgs = batch.get('orig_img', [])
    if not orig_imgs:
        return None

    boxes_by_image = []
    for b in range(batch_size):
        gt_b = gt_all[b].detach().cpu().numpy().astype(np.float32)
        if 'ct_01' in batch:
            keep = batch['ct_01'][b].detach().cpu().numpy().astype(bool)
            gt_b = gt_b[keep]
        if gt_b.size == 0:
            boxes_by_image.append(np.zeros((0, 4), dtype=np.float32))
            continue
        boxes = np.concatenate([gt_b.min(axis=1), gt_b.max(axis=1)], axis=1) * dr
        boxes_by_image.append(boxes.astype(np.float32))

    img0 = orig_imgs[0].detach().cpu().numpy() if torch.is_tensor(orig_imgs[0]) else np.asarray(orig_imgs[0])
    out_h = int(img0.shape[0] // snake_config.down_ratio)
    out_w = int(img0.shape[1] // snake_config.down_ratio)

    from lib.utils.snake.sam_init import build_sam_polys_from_boxes
    polys_by_image, _ = build_sam_polys_from_boxes(
        orig_imgs,
        boxes_by_image,
        device=device,
        out_h=out_h,
        out_w=out_w,
        num_points=num_points,
    )
    polys = [torch.from_numpy(p) for p in polys_by_image if p.shape[0] > 0]
    if not polys:
        return None
    return torch.cat(polys, dim=0).to(device=device, dtype=gt_all.dtype)


def save_visual(sample_dir, img, gt_polys, init_polys, pred_polys, per_contour_iou):
    os.makedirs(sample_dir, exist_ok=True)
    vis = img.copy()
    for poly in gt_polys:
        cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 255, 0), 2)
    for poly in init_polys:
        cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 255, 255), 1)
    for poly in pred_polys:
        cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 0, 255), 2)

    for idx, iou in enumerate(per_contour_iou):
        if idx >= len(pred_polys):
            continue
        cx = int(pred_polys[idx, :, 0].mean())
        cy = int(pred_polys[idx, :, 1].mean())
        cv2.putText(vis, f'{idx}:{iou * 100:.1f}%', (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    cv2.imwrite(os.path.join(sample_dir, 'overlay.png'), vis)


def eval_sample(model, device, batch, ode_steps=10, save_visuals=False, sample_dir=None):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    dr = float(snake_config.down_ratio)
    core = model.net if hasattr(model, 'net') else model
    contour_init_method = str(getattr(cfg, 'contour_init_method', 'octagon')).strip().lower()
    sam_prompt_source = str(getattr(cfg, 'sam_prompt_source', 'yolo_box')).strip().lower()
    detector_backend = str(getattr(cfg, 'detector_backend', 'yolo')).strip().lower()
    use_sam_yolo_init = (
        contour_init_method in ('sam', 'efficient_sam')
        and sam_prompt_source == 'yolo_box'
    )
    use_sam_gt_box_init = contour_init_method in ('sam', 'efficient_sam') and sam_prompt_source == 'gt_box'
    use_full_forward = (
        use_sam_yolo_init
        or detector_backend != 'yolo'
        or bool(getattr(cfg, 'eval_use_network_forward', False))
    )

    with torch.no_grad():
        gt_all = batch['i_gt_py']
        if gt_all.numel() == 0:
            raise RuntimeError('No GT polygons in batch')

        batch_size, num_contours, num_points, _ = gt_all.shape

        if use_full_forward:
            net_out = core(batch['inp'], batch)
            if 'py' not in net_out:
                raise RuntimeError('Network output has no predicted polygons')
            pred_polys = net_out['py'].detach().cpu().numpy() * dr
            init_src = net_out.get('i_it_py', torch.zeros_like(net_out['py']))
            init_polys = init_src.detach().cpu().numpy() * dr
            if 'ct_01' in batch:
                gt_polys = gt_all[batch['ct_01'].bool()].detach().cpu().numpy() * dr
            else:
                gt_polys = gt_all.view(-1, num_points, 2).detach().cpu().numpy() * dr
            match_by_iou = not bool(getattr(cfg, 'use_gt_det', False))
        else:
            if not hasattr(core, 'yolo') or core.yolo is None:
                raise RuntimeError(f'Manual eval path requires YOLO backend, got detector_backend={detector_backend}')
            yolo_out = core.yolo(batch['inp'])
            feat_list = yolo_out[1] if isinstance(yolo_out, (list, tuple)) and len(yolo_out) > 1 else None
            feat_p2 = feat_list[0] if isinstance(feat_list, (list, tuple)) else yolo_out
            if getattr(core, 'use_swin_snake_feature', False):
                if not hasattr(core, 'swin_snake_feature') or core.swin_snake_feature is None:
                    raise RuntimeError('Swin feature evaluation requested but swin_snake_feature is missing')
                cnn_feature = core.swin_snake_feature(batch['inp'])
            else:
                cnn_feature = core.cnn_proj(feat_p2)
            if (not getattr(core, 'use_swin_snake_feature', False)) and getattr(core, 'use_p3_features', False) and hasattr(core, 'cnn_proj_p3'):
                if isinstance(feat_list, (list, tuple)) and len(feat_list) > 1:
                    feat_p3 = feat_list[1]
                    feat_p3_up = torch.nn.functional.interpolate(
                        feat_p3, size=feat_p2.shape[-2:], mode='bilinear', align_corners=False
                    )
                    cnn_feature = cnn_feature + core.cnn_proj_p3(feat_p3_up)

            if use_sam_gt_box_init:
                i_it_py = build_sam_gt_box_init_polys(batch, gt_all, device)
                if i_it_py is None:
                    i_it_py = build_init_polys(batch, gt_all)
            else:
                i_it_py = build_init_polys(batch, gt_all)
            c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)
            if batch_size == 1:
                py_ind = torch.zeros(i_it_py.size(0), dtype=torch.long, device=device)
            else:
                py_ind = torch.cat([torch.full((num_contours,), i, dtype=torch.long, device=device) for i in range(batch_size)])

            if getattr(core.gcn, 'use_iterative_refinement', False):
                iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
                fractions = list(getattr(cfg, 'iterative_fractions', []))
                if not fractions:
                    fractions = [1.0 / (iter_steps - i) for i in range(iter_steps)]
                iter_ode_steps = int(
                    getattr(
                        cfg,
                        'iterative_ode_steps',
                        getattr(cfg, 'iterative_ddim_steps', ode_steps),
                    )
                )
                if iter_ode_steps <= 0:
                    iter_ode_steps = ode_steps
                disp = core.gcn.sample_disp_iterative(
                    cnn_feature,
                    i_it_py,
                    c_it_py,
                    py_ind,
                    num_iter_steps=iter_steps,
                    fractions=fractions,
                    ode_steps=iter_ode_steps,
                )
            else:
                disp = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind, steps=ode_steps)
            fk = int(getattr(cfg, 'fourier_smooth_k', 0))
            if fk > 0:
                from lib.networks.diffusion.flow_matching_evolution import FlowMatchingEvolution
                disp = FlowMatchingEvolution.fourier_smooth(disp, fk)

            pred_polys = (i_it_py + disp).cpu().numpy() * dr
            gt_polys = gt_all.view(-1, num_points, 2).cpu().numpy() * dr
            init_polys = i_it_py.cpu().numpy() * dr
            match_by_iou = False

    if 'orig_img' in batch:
        img_raw = batch['orig_img'][0]
        img = img_raw.detach().cpu().numpy() if torch.is_tensor(img_raw) else img_raw
        img = img.astype(np.uint8)
    else:
        img = np.zeros((512, 512, 3), dtype=np.uint8)

    height, width = img.shape[:2]
    per_contour_iou, per_contour_dice, per_contour_mboundf, matched_pred = compute_ordered_or_matched_metrics(
        pred_polys,
        gt_polys,
        height,
        width,
        match_by_iou=match_by_iou,
    )
    init_iou, init_dice, init_mboundf, _ = compute_ordered_or_matched_metrics(
        init_polys,
        gt_polys,
        height,
        width,
        match_by_iou=match_by_iou,
    )

    if save_visuals and sample_dir is not None:
        save_visual(sample_dir, img, gt_polys, init_polys, pred_polys, per_contour_iou)

    return {
        'mean_iou': float(np.mean(per_contour_iou)) if per_contour_iou else 0.0,
        'mean_dice': float(np.mean(per_contour_dice)) if per_contour_dice else 0.0,
        'mean_mboundf': float(np.mean(per_contour_mboundf)) if per_contour_mboundf else 0.0,
        'mean_init_iou': float(np.mean(init_iou)) if init_iou else 0.0,
        'mean_init_dice': float(np.mean(init_dice)) if init_dice else 0.0,
        'mean_init_mboundf': float(np.mean(init_mboundf)) if init_mboundf else 0.0,
        'per_contour_iou': per_contour_iou,
        'per_contour_dice': per_contour_dice,
        'per_contour_mboundf': per_contour_mboundf,
        'per_contour_init_iou': init_iou,
        'per_contour_init_dice': init_dice,
        'per_contour_init_mboundf': init_mboundf,
        'matched_pred': matched_pred,
        'num_pred_contours': int(pred_polys.shape[0]),
        'num_gt_contours': int(gt_polys.shape[0]),
    }


def main():
    apply_gpu_override()

    ckpt = os.environ.get('CKPT')
    eval_seed = int(os.environ.get('EVAL_SEED', '20260504'))
    ode_steps = int(os.environ.get('ODE_STEPS', getattr(cfg, 'flow_ode_steps', 10)))
    save_dir = os.environ.get('SAVE_DIR', os.path.join(_THIS_DIR, 'visual', 'v3_7_eval_now_full'))
    save_visuals = os.environ.get('SAVE_VISUALS', '1') != '0'
    max_samples_env = os.environ.get('MAX_SAMPLES', '')
    max_samples = int(max_samples_env) if max_samples_env else None

    os.makedirs(save_dir, exist_ok=True)
    per_sample_root = os.path.join(save_dir, 'per_sample')
    if save_visuals:
        os.makedirs(per_sample_root, exist_ok=True)

    model, device, ckpt_path = load_model(ckpt)

    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)
    set_eval_seed(eval_seed)

    dataset_size = len(dataset)
    limit = min(dataset_size, max_samples) if max_samples is not None else dataset_size

    rows = []
    sample_mean_ious = []
    sample_mean_dices = []
    sample_mean_mboundfs = []
    sample_mean_init_ious = []
    sample_mean_init_mboundfs = []
    all_contour_ious = []
    all_contour_dices = []
    all_contour_mboundfs = []
    all_contour_init_ious = []
    all_contour_init_mboundfs = []
    failed_indices = []

    print(f'[*] Evaluating {limit} / {dataset_size} samples from {cfg.test.dataset}')
    print(f'[*] ODE steps: {ode_steps} | save_visuals={save_visuals}')

    for index in range(limit):
        print(f'[{index + 1}/{limit}] sample {index}')
        sample_dir = os.path.join(per_sample_root, f'idx_{index:03d}') if save_visuals else ''
        try:
            batch = collator([dataset[index]])
            result = eval_sample(
                model,
                device,
                batch,
                ode_steps=ode_steps,
                save_visuals=save_visuals,
                sample_dir=sample_dir if save_visuals else None,
            )
            rows.append({
                'index': index,
                'ok': True,
                'mean_iou': result['mean_iou'],
                'mean_dice': result['mean_dice'],
                'mean_mboundf': result['mean_mboundf'],
                'mean_init_iou': result['mean_init_iou'],
                'mean_init_mboundf': result['mean_init_mboundf'],
                'per_contour_iou': result['per_contour_iou'],
                'per_contour_dice': result['per_contour_dice'],
                'per_contour_mboundf': result['per_contour_mboundf'],
                'per_contour_init_iou': result['per_contour_init_iou'],
                'matched_pred': result['matched_pred'],
                'num_pred_contours': result['num_pred_contours'],
                'num_gt_contours': result['num_gt_contours'],
                'dir': sample_dir,
            })
            sample_mean_ious.append(result['mean_iou'])
            sample_mean_dices.append(result['mean_dice'])
            sample_mean_mboundfs.append(result['mean_mboundf'])
            sample_mean_init_ious.append(result['mean_init_iou'])
            sample_mean_init_mboundfs.append(result['mean_init_mboundf'])
            all_contour_ious.extend(result['per_contour_iou'])
            all_contour_dices.extend(result['per_contour_dice'])
            all_contour_mboundfs.extend(result['per_contour_mboundf'])
            all_contour_init_ious.extend(result['per_contour_init_iou'])
            all_contour_init_mboundfs.extend(result['per_contour_init_mboundf'])
        except Exception as exc:
            failed_indices.append(index)
            rows.append({
                'index': index,
                'ok': False,
                'error': str(exc),
                'dir': sample_dir,
            })
            print(f'  [!] failed: {exc}')

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    summary = {
        'timestamp': ts,
        'cfg_file': os.path.relpath(os.environ.get('CFG_FILE', _DEFAULT_CFG), _THIS_DIR),
        'ckpt': os.path.relpath(ckpt_path, _THIS_DIR),
        'eval_seed': eval_seed,
        'ode_steps': ode_steps,
        'dataset': cfg.test.dataset,
        'test_img_path': getattr(cfg.test, 'img_path', ''),
        'dataset_size': dataset_size,
        'evaluated_samples': len(sample_mean_ious),
        'failed_samples': len(failed_indices),
        'failed_indices': failed_indices,
        'mean_iou_sample_avg': float(np.mean(sample_mean_ious)) if sample_mean_ious else 0.0,
        'mean_iou_contour_avg': float(np.mean(all_contour_ious)) if all_contour_ious else 0.0,
        'mean_dice_sample_avg': float(np.mean(sample_mean_dices)) if sample_mean_dices else 0.0,
        'mean_dice_contour_avg': float(np.mean(all_contour_dices)) if all_contour_dices else 0.0,
        'mean_mboundf_sample_avg': float(np.mean(sample_mean_mboundfs)) if sample_mean_mboundfs else 0.0,
        'mean_mboundf_contour_avg': float(np.mean(all_contour_mboundfs)) if all_contour_mboundfs else 0.0,
        'mean_init_iou_sample_avg': float(np.mean(sample_mean_init_ious)) if sample_mean_init_ious else 0.0,
        'mean_init_iou_contour_avg': float(np.mean(all_contour_init_ious)) if all_contour_init_ious else 0.0,
        'mean_init_mboundf_sample_avg': float(np.mean(sample_mean_init_mboundfs)) if sample_mean_init_mboundfs else 0.0,
        'mean_init_mboundf_contour_avg': float(np.mean(all_contour_init_mboundfs)) if all_contour_init_mboundfs else 0.0,
        'median_iou_sample_avg': float(np.median(sample_mean_ious)) if sample_mean_ious else 0.0,
        'std_iou_sample_avg': float(np.std(sample_mean_ious)) if sample_mean_ious else 0.0,
        'sample_mean_ious': sample_mean_ious,
        'sample_mean_dices': sample_mean_dices,
        'sample_mean_mboundfs': sample_mean_mboundfs,
        'sample_mean_init_ious': sample_mean_init_ious,
        'sample_mean_init_mboundfs': sample_mean_init_mboundfs,
    }

    summary_path = os.path.join(save_dir, f'v3_7_full_test_iou_{ts}.json')
    rows_path = os.path.join(save_dir, f'summary_rows_{ts}.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    with open(rows_path, 'w', encoding='utf-8') as f:
        json.dump({'timestamp': ts, 'rows': rows}, f, indent=2)

    print('\n' + '=' * 80)
    print(f'Saved summary: {summary_path}')
    print(f'Saved rows:    {rows_path}')
    print(f"mean_iou_sample_avg:   {summary['mean_iou_sample_avg']:.6f}")
    print(f"mean_iou_contour_avg:  {summary['mean_iou_contour_avg']:.6f}")
    print(f"mean_dice_sample_avg:  {summary['mean_dice_sample_avg']:.6f}")
    print(f"mean_mboundf_sample_avg: {summary['mean_mboundf_sample_avg']:.6f}")
    print(f"median_iou_sample_avg: {summary['median_iou_sample_avg']:.6f}")
    print(f"std_iou_sample_avg:    {summary['std_iou_sample_avg']:.6f}")
    print(f"failed_samples:        {summary['failed_samples']}")
    print('=' * 80)


if __name__ == '__main__':
    main()
