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
from lib.train.per_point_fm_policy import PerPointFMScalePolicy
from lib.train.rewards.region_reward import _calc_nsd
from lib.train.trainers import make_trainer
from lib.utils.snake import snake_config, snake_decode, snake_gcn_utils
from lib.utils.snake.viz_colors import (
    UNMATCHED_COLOR_BGR,
    draw_dashed_line,
    draw_dotted_bbox,
    draw_instance_legend,
    draw_polyline_style,
    draw_text_with_outline,
    instance_color,
    _INSTANCE_PALETTE_BGR,
)


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


def apply_ablation_mode():
    mode = os.environ.get('EVAL_ABLATION_MODE', '').strip().lower()
    if not mode:
        mode = 'config'
    cfg.eval_manual_gt_init = False
    cfg.eval_manual_gt_box_octagon = False

    if mode in ('config', 'default'):
        pass
    elif mode in ('pred_det_pred_ext', 'full_det'):
        cfg.eval_use_network_forward = True
        cfg.use_gt_det = False
        cfg.use_pred_extreme_init_for_inference = True
    elif mode in ('gt_det_pred_ext', 'gt_box_pred_ext'):
        cfg.eval_use_network_forward = True
        cfg.use_gt_det = True
        cfg.use_pred_extreme_init_for_inference = True
    elif mode in ('pred_det_bbox_init', 'pred_box_no_ext'):
        cfg.eval_use_network_forward = True
        cfg.use_gt_det = False
        cfg.use_pred_extreme_init_for_inference = False
    elif mode in ('gt_det_bbox_init', 'gt_box_no_ext'):
        cfg.eval_use_network_forward = True
        cfg.use_gt_det = True
        cfg.use_pred_extreme_init_for_inference = False
    elif mode in ('gt_init', 'gt_extreme_init', 'gt_octagon'):
        cfg.eval_use_network_forward = False
        cfg.use_gt_det = True
        cfg.use_pred_extreme_init_for_inference = False
        cfg.eval_manual_gt_init = True
    elif mode in ('gt_box_octagon', 'gt_bbox_octagon', 'gt_box_init'):
        cfg.eval_use_network_forward = False
        cfg.use_gt_det = True
        cfg.use_pred_extreme_init_for_inference = False
        cfg.eval_manual_gt_init = True
        cfg.eval_manual_gt_box_octagon = True
        cfg.diffusion_init_source = 'gt_box_octagon'
    else:
        raise ValueError(
            f"Unsupported EVAL_ABLATION_MODE={mode}. "
            "Use config, pred_det_pred_ext, gt_det_pred_ext, "
            "pred_det_bbox_init, gt_det_bbox_init, gt_box_octagon, or gt_init."
        )

    print(
        '[*] Ablation mode: '
        f'{mode} | eval_use_network_forward={bool(getattr(cfg, "eval_use_network_forward", False))} '
        f'use_gt_det={bool(getattr(cfg, "use_gt_det", False))} '
        f'use_pred_extreme_init_for_inference='
        f'{bool(getattr(cfg, "use_pred_extreme_init_for_inference", False))}'
    )
    det_conf = os.environ.get('EVAL_DET_CONF_THRESH', '').strip()
    if det_conf:
        cfg.det_conf_thresh = float(det_conf)
        print(f'[*] Override det_conf_thresh -> {cfg.det_conf_thresh}')
    det_iou = os.environ.get('EVAL_DET_IOU_THRESH', '').strip()
    if det_iou:
        cfg.det_iou_thresh = float(det_iou)
        print(f'[*] Override det_iou_thresh -> {cfg.det_iou_thresh}')
    det_max = os.environ.get('EVAL_DET_MAX_DET', '').strip()
    if det_max:
        cfg.det_max_det = int(det_max)
        print(f'[*] Override det_max_det -> {cfg.det_max_det}')
    return mode


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

    nsd_delta_px = float(os.environ.get('NSD_DELTA_PX', '2.0'))
    per_iou = []
    per_dice = []
    per_mboundf = []
    per_nsd = []
    for gi, gt_mask in enumerate(gt_masks):
        pi = matched_pred[gi]
        if pi < 0 or pi >= len(pred_masks):
            per_iou.append(0.0)
            per_dice.append(0.0)
            per_mboundf.append(0.0)
            per_nsd.append(0.0)
            continue
        iou = compute_iou(pred_masks[pi], gt_mask)
        per_iou.append(iou)
        per_dice.append(compute_dice_from_iou(iou))
        per_mboundf.append(compute_mboundf(pred_masks[pi], gt_mask))
        per_nsd.append(float(_calc_nsd(pred_masks[pi], gt_mask, delta_px=nsd_delta_px)))

    return per_iou, per_dice, per_mboundf, per_nsd, matched_pred


_UNIFIED_POLICY_OUTER_STEPS = 5


def _policy_cfg(name, default):
    nested = getattr(cfg, 'rl_v4', None)
    if nested is not None and name in nested:
        return nested[name]
    return getattr(cfg, f'rl_v4_{name}', default)


def _normalize_rollout_fractions(fractions, outer_steps):
    fractions = [float(v) for v in list(fractions)]
    if not fractions:
        fractions = [1.0 / (outer_steps - i) for i in range(outer_steps)]
    if len(fractions) < outer_steps:
        fractions.extend([1.0] * (outer_steps - len(fractions)))
    return fractions[:outer_steps]


def _resolve_active_policy_steps(checkpoint_metadata, outer_steps):
    explicit = checkpoint_metadata.get('active_policy_step_indices')
    if explicit is not None:
        active_steps = sorted({int(v) for v in explicit})
        if not active_steps or any(v < 0 or v >= outer_steps for v in active_steps):
            raise RuntimeError(
                f'Invalid active_policy_step_indices={explicit} for outer_steps={outer_steps}'
            )
        return active_steps, len(active_steps)

    train_last_n_steps = int(
        checkpoint_metadata.get(
            'policy_train_last_n_steps',
            _policy_cfg('policy_train_last_n_steps', outer_steps),
        )
    )
    if not 1 <= train_last_n_steps <= outer_steps:
        raise RuntimeError(
            f'Invalid policy_train_last_n_steps={train_last_n_steps} for outer_steps={outer_steps}'
        )
    return list(range(outer_steps - train_last_n_steps, outer_steps)), train_last_n_steps


def _build_rollout_metadata(
    requested_mode,
    effective_mode,
    checkpoint_has_per_point_policy,
    fractions,
    actual_ode_steps,
    active_step_indices,
    zero_mean_local=False,
    max_scale=None,
    train_last_n_steps=0,
    rollout_backend=None,
    deterministic=None,
):
    fractions = [float(v) for v in fractions]
    active_step_indices = [int(v) for v in active_step_indices]
    unified = bool(checkpoint_has_per_point_policy)
    if unified and len(fractions) != _UNIFIED_POLICY_OUTER_STEPS:
        raise RuntimeError(
            f'Per-point policy evaluation requires {_UNIFIED_POLICY_OUTER_STEPS} fractions, '
            f'got {fractions}'
        )
    if rollout_backend is None:
        rollout_backend = (
            'unified_per_point_5step_deterministic' if unified else 'configured_native'
        )
    if deterministic is None:
        deterministic = unified
    return {
        'requested_mode': requested_mode,
        'effective_mode': effective_mode,
        'checkpoint_has_per_point_policy': unified,
        'loaded': False,
        'deterministic': bool(deterministic),
        'rollout_backend': str(rollout_backend),
        'outer_steps': len(fractions),
        'outer_step_indices': list(range(len(fractions))),
        'actual_ode_steps': int(actual_ode_steps),
        'fractions': fractions,
        'train_last_n_steps': int(train_last_n_steps),
        'active_step_indices': active_step_indices,
        'scale_applied_step_indices': (
            active_step_indices if effective_mode == 'mean' else []
        ),
        'max_scale': None if max_scale is None else float(max_scale),
        'zero_mean_local': bool(zero_mean_local),
    }


def _configured_native_rollout_metadata(requested_mode, eval_ode_steps=None):
    if bool(getattr(cfg, 'use_curve_inference', False)):
        return _build_rollout_metadata(
            requested_mode,
            'off',
            False,
            [],
            int(getattr(cfg, 'curve_steps', 20)),
            [],
            rollout_backend='configured_curve_sampling',
        )

    if not bool(getattr(cfg, 'use_iterative_refinement', False)):
        actual_ode_steps = int(
            getattr(cfg, 'flow_ode_steps', 10)
            if eval_ode_steps is None
            else eval_ode_steps
        )
        return _build_rollout_metadata(
            requested_mode,
            'off',
            False,
            [1.0],
            actual_ode_steps,
            [],
            rollout_backend='configured_single_sampling',
        )

    iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
    if bool(getattr(cfg, 'v4_9_use_rich_infer_schedule', False)):
        targets = list(getattr(cfg, 'v4_9_infer_target_fractions', []))
        if not targets:
            targets = [0.3333, 0.5, 0.80, 0.97, 1.0]
        fractions = []
        previous = 0.0
        for target in targets:
            target = min(max(float(target), previous), 1.0)
            fractions.append((target - previous) / max(1.0 - previous, 1e-6))
            previous = target
    else:
        fractions = list(getattr(cfg, 'iterative_fractions', []))
    fractions = _normalize_rollout_fractions(fractions, iter_steps)
    actual_ode_steps = int(
        getattr(
            cfg,
            'iterative_ode_steps',
            getattr(cfg, 'iterative_ddim_steps', getattr(cfg, 'flow_ode_steps', 10)),
        )
    )
    if actual_ode_steps <= 0:
        actual_ode_steps = int(
            getattr(cfg, 'flow_ode_steps', 10)
            if eval_ode_steps is None
            else eval_ode_steps
        )
    return _build_rollout_metadata(
        requested_mode,
        'off',
        False,
        fractions,
        actual_ode_steps,
        [],
        rollout_backend='configured_iterative_sampling',
    )


def load_model(ckpt_path=None, return_policy=False, eval_ode_steps=None):
    requested_mode = os.environ.get('EVAL_FM_POLICY', 'auto').strip().lower()
    if requested_mode not in ('auto', 'off', 'mean'):
        raise ValueError(
            f'Invalid EVAL_FM_POLICY={requested_mode!r}; expected auto, off, or mean'
        )

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

    policy_sd = ckpt_obj.get('fm_velocity_policy_state_dict') if isinstance(ckpt_obj, dict) else None
    action_policy = str(_policy_cfg('action_policy', '')).strip().lower()
    has_per_point_policy = (
        isinstance(policy_sd, dict)
        and action_policy == 'per_point_fm_scale'
        and any(str(k).startswith('point_net.') for k in policy_sd)
    )
    if requested_mode == 'mean' and not has_per_point_policy:
        raise RuntimeError(
            'EVAL_FM_POLICY=mean requires a per_point_fm_scale policy checkpoint'
        )
    effective_mode = 'mean' if requested_mode != 'off' and has_per_point_policy else 'off'
    checkpoint_metadata = (
        ckpt_obj.get('experiment_metadata', {}) if isinstance(ckpt_obj, dict) else {}
    )
    if not isinstance(checkpoint_metadata, dict):
        checkpoint_metadata = {}

    policy = None
    if has_per_point_policy:
        outer_steps = int(_policy_cfg('outer_steps', _UNIFIED_POLICY_OUTER_STEPS))
        state_outer_steps = int(policy_sd['step_embed.weight'].shape[0])
        if outer_steps != _UNIFIED_POLICY_OUTER_STEPS or state_outer_steps != outer_steps:
            raise RuntimeError(
                'Unified per-point evaluation requires a 5-step policy; '
                f'config outer_steps={outer_steps}, checkpoint outer_steps={state_outer_steps}'
            )
        fractions = _normalize_rollout_fractions(
            _policy_cfg('fractions', getattr(cfg, 'iterative_fractions', [])),
            outer_steps,
        )
        actual_ode_steps = int(
            _policy_cfg(
                'ode_steps',
                getattr(cfg, 'iterative_ode_steps', getattr(cfg, 'flow_ode_steps', 10)),
            )
        )
        if actual_ode_steps <= 0:
            actual_ode_steps = int(getattr(cfg, 'flow_ode_steps', 10))
        active_step_indices, train_last_n_steps = _resolve_active_policy_steps(
            checkpoint_metadata, outer_steps
        )
        zero_mean_local = bool(
            checkpoint_metadata.get(
                'fm_velocity_zero_mean_local',
                _policy_cfg('fm_velocity_zero_mean_local', False),
            )
        )
        max_scale = float(
            checkpoint_metadata.get(
                'fm_velocity_max_scale',
                _policy_cfg('fm_velocity_max_scale', 0.25),
            )
        )
        policy_metadata = _build_rollout_metadata(
            requested_mode,
            effective_mode,
            True,
            fractions,
            actual_ode_steps,
            active_step_indices,
            zero_mean_local=zero_mean_local,
            max_scale=max_scale,
            train_last_n_steps=train_last_n_steps,
        )
        if effective_mode == 'mean':
            policy = PerPointFMScalePolicy(
                outer_steps=outer_steps,
                feature_dim=int(_policy_cfg('fm_velocity_feature_dim', 64)),
                feature_embed_dim=int(_policy_cfg('fm_velocity_feature_embed_dim', 32)),
                hidden_dim=int(_policy_cfg('fm_velocity_hidden_dim', 64)),
                init_logstd=float(_policy_cfg('fm_velocity_init_logstd', -1.0)),
                logstd_min=float(_policy_cfg('fm_velocity_logstd_min', -4.0)),
                logstd_max=float(_policy_cfg('fm_velocity_logstd_max', 1.0)),
                offset_scale=float(_policy_cfg('fm_velocity_offset_scale', 0.5)),
                max_scale=max_scale,
                zero_mean_local=zero_mean_local,
            ).to(device)
            policy.load_state_dict(policy_sd, strict=True)
            policy.eval()
            policy_metadata['loaded'] = True
    else:
        policy_metadata = _configured_native_rollout_metadata(
            requested_mode, eval_ode_steps=eval_ode_steps
        )

    print(
        '[*] FM policy: requested={} effective={} loaded={} backend={} '
        'actual_ode_steps={} fractions={} active_steps={} scale_steps={}'.format(
            requested_mode,
            policy_metadata['effective_mode'],
            policy_metadata['loaded'],
            policy_metadata['rollout_backend'],
            policy_metadata['actual_ode_steps'],
            policy_metadata['fractions'],
            policy_metadata['active_step_indices'],
            policy_metadata['scale_applied_step_indices'],
        )
    )
    result = (trainer.network.to(device).eval(), device, ckpt_path)
    if return_policy:
        return result + (policy, policy_metadata)
    return result


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
    init_source = str(getattr(cfg, 'diffusion_init_source', 'extreme')).strip().lower()
    if (
        bool(getattr(cfg, 'eval_manual_gt_box_octagon', False))
        or init_source in ('gt_box_octagon', 'gt_bbox_octagon', 'box_octagon', 'bbox_octagon')
    ):
        gt_flat = gt_all.view(batch_size * num_contours, num_points, 2)
        if 'ct_01' in batch:
            keep = batch['ct_01'].view(-1).bool()
            gt_flat = gt_flat[keep]
        return snake_gcn_utils.build_box_octagon_from_poly(gt_flat, num_points)

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


_LABEL_PALETTE_BGR = [
    (60, 180, 255),
    (80, 220, 80),
    (255, 120, 80),
    (220, 120, 255),
    (90, 220, 220),
    (255, 210, 90),
    (120, 160, 255),
    (170, 230, 120),
    (255, 150, 200),
    (210, 210, 70),
    (140, 120, 255),
    (90, 170, 180),
]


def label_color(label):
    try:
        label_int = int(label)
    except Exception:
        label_int = 0
    return _LABEL_PALETTE_BGR[label_int % len(_LABEL_PALETTE_BGR)]


def soften_color(color, alpha=0.45):
    base = np.array(color, dtype=np.float32)
    gray = np.array([190, 190, 190], dtype=np.float32)
    return tuple(int(v) for v in (alpha * base + (1.0 - alpha) * gray))


def crop_poly_view(img, polys, margin=28):
    valid = [np.asarray(poly, dtype=np.float32) for poly in polys if poly is not None and len(poly) > 0]
    if not valid:
        return img
    pts = np.concatenate(valid, axis=0)
    h, w = img.shape[:2]
    x1 = max(0, int(np.floor(pts[:, 0].min())) - margin)
    y1 = max(0, int(np.floor(pts[:, 1].min())) - margin)
    x2 = min(w, int(np.ceil(pts[:, 0].max())) + margin)
    y2 = min(h, int(np.ceil(pts[:, 1].max())) + margin)
    if x2 <= x1 or y2 <= y1:
        return img
    return img[y1:y2, x1:x2].copy()


# High-contrast role colors for the "clear" visualization style. GT is a solid
# pure-green, thick line; Pred is a pure-red/magenta thick dashed line with a
# large dash/gap so the dash pattern survives thumbnail/montage downscaling.
_CLEAR_GT_COLOR_BGR = (40, 220, 40)      # pure green
_CLEAR_PRED_COLOR_BGR = (40, 40, 235)    # pure red
_CLEAR_GT_FILL_BGR = (60, 220, 60)
_CLEAR_PRED_FILL_BGR = (60, 60, 230)


def _draw_legend_box(img, lines_with_colors, x=10, y=10):
    """Draw a legend with a semi-transparent background box for readability."""
    pad = 6
    line_h = 20
    font_scale = 0.5
    thickness = 1
    max_w = 0
    for text, _ in lines_with_colors:
        (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        max_w = max(max_w, tw)
    box_w = max_w + 3 * pad + 24
    box_h = pad * 2 + line_h * len(lines_with_colors)
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
    for i, (text, color) in enumerate(lines_with_colors):
        cy = y + pad + line_h * i + line_h // 2
        cv2.line(img, (x + pad, cy), (x + pad + 20, cy), color, 3, lineType=cv2.LINE_AA)
        draw_text_with_outline(img, text, (x + pad + 26, cy + 5), font_scale=font_scale, thickness=thickness)


def _save_visual_clear(sample_dir, img, gt_polys, pred_polys, per_contour_iou, matched_pred=None):
    """Paper-style overlay: GT solid green vs Pred dashed red, plus a
    translucent fill version where overlap reads as a blended color."""
    if matched_pred is None:
        matched_pred = list(range(min(len(gt_polys), len(pred_polys))))

    # 1) Line overlay: thick solid green GT, thick dashed red Pred with a
    # large dash/gap so it stays legible after resize.
    line_vis = img.copy()
    for poly in gt_polys:
        pts = np.round(poly).astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(line_vis, [pts], True, _CLEAR_GT_COLOR_BGR, 3, lineType=cv2.LINE_AA)
    for poly in pred_polys:
        pts = np.round(poly).astype(np.int32)
        if len(pts) >= 2:
            draw_dashed_line_poly(line_vis, pts, _CLEAR_PRED_COLOR_BGR, thickness=3, dash_len=14, gap_len=10)

    for gt_idx, iou in enumerate(per_contour_iou):
        pred_idx = int(matched_pred[gt_idx]) if gt_idx < len(matched_pred) else -1
        if pred_idx < 0 or pred_idx >= len(pred_polys):
            continue
        poly = np.asarray(pred_polys[pred_idx], dtype=np.float32)
        if poly.ndim != 2 or poly.shape[0] == 0:
            continue
        cx, cy = int(poly[:, 0].mean()), int(poly[:, 1].mean())
        draw_text_with_outline(line_vis, f'{iou * 100:.1f}%', (cx - 22, cy), font_scale=0.42, thickness=1)

    _draw_legend_box(
        line_vis,
        [('GT (solid)', _CLEAR_GT_COLOR_BGR), ('Pred (dashed)', _CLEAR_PRED_COLOR_BGR)],
    )
    cv2.imwrite(os.path.join(sample_dir, 'overlay.png'), line_vis)

    # 2) Translucent fill version: GT filled green, Pred filled red, overlap
    # blends to yellow/olive so the match region is visible at a glance.
    fill_base = img.copy()
    gt_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    pred_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for poly in gt_polys:
        pts = np.round(poly).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(gt_mask, [pts], 255)
    for poly in pred_polys:
        pts = np.round(poly).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(pred_mask, [pts], 255)

    fill_vis = fill_base.astype(np.float32)
    alpha = 0.42
    gt_only = (gt_mask > 0) & (pred_mask == 0)
    pred_only = (pred_mask > 0) & (gt_mask == 0)
    overlap = (gt_mask > 0) & (pred_mask > 0)
    color_arr = np.array(_CLEAR_GT_FILL_BGR, dtype=np.float32)
    fill_vis[gt_only] = fill_vis[gt_only] * (1 - alpha) + color_arr * alpha
    color_arr = np.array(_CLEAR_PRED_FILL_BGR, dtype=np.float32)
    fill_vis[pred_only] = fill_vis[pred_only] * (1 - alpha) + color_arr * alpha
    overlap_color = np.array((45, 215, 225), dtype=np.float32)  # yellow-ish blend
    fill_vis[overlap] = fill_vis[overlap] * (1 - alpha) + overlap_color * alpha
    fill_vis = np.clip(fill_vis, 0, 255).astype(np.uint8)

    for poly in gt_polys:
        pts = np.round(poly).astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(fill_vis, [pts], True, _CLEAR_GT_COLOR_BGR, 2, lineType=cv2.LINE_AA)
    for poly in pred_polys:
        pts = np.round(poly).astype(np.int32)
        if len(pts) >= 2:
            draw_dashed_line_poly(fill_vis, pts, _CLEAR_PRED_COLOR_BGR, thickness=2, dash_len=14, gap_len=10)

    _draw_legend_box(
        fill_vis,
        [
            ('GT only', tuple(int(c) for c in _CLEAR_GT_FILL_BGR)),
            ('Pred only', tuple(int(c) for c in _CLEAR_PRED_FILL_BGR)),
            ('Overlap', (45, 215, 225)),
        ],
    )
    cv2.imwrite(os.path.join(sample_dir, 'overlay_fill.png'), fill_vis)

    # Cropped tight views per matched GT/Pred pair, useful for zooming into
    # small structures in a montage.
    pair_dir = os.path.join(sample_dir, 'pairs')
    os.makedirs(pair_dir, exist_ok=True)
    for gt_idx, gt_poly in enumerate(gt_polys):
        pred_idx = int(matched_pred[gt_idx]) if gt_idx < len(matched_pred) else -1
        if pred_idx < 0 or pred_idx >= len(pred_polys):
            continue
        pair_vis = img.copy()
        gt_pts = np.round(gt_poly).astype(np.int32)
        pred_pts = np.round(pred_polys[pred_idx]).astype(np.int32)
        if len(gt_pts) >= 2:
            cv2.polylines(pair_vis, [gt_pts], True, _CLEAR_GT_COLOR_BGR, 2, lineType=cv2.LINE_AA)
        if len(pred_pts) >= 2:
            draw_dashed_line_poly(pair_vis, pred_pts, _CLEAR_PRED_COLOR_BGR, thickness=2, dash_len=12, gap_len=8)
        crop = crop_poly_view(pair_vis, [gt_poly, pred_polys[pred_idx]])
        iou = float(per_contour_iou[gt_idx]) if gt_idx < len(per_contour_iou) else 0.0
        out_name = f'gt_{gt_idx:03d}_iou_{iou:.3f}.png'
        cv2.imwrite(os.path.join(pair_dir, out_name), crop)


def draw_dashed_line_poly(img, pts, color, thickness=2, dash_len=14, gap_len=10):
    n = len(pts)
    if n < 2:
        return
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        draw_dashed_line(img, p1, p2, color, thickness=thickness, dash_len=dash_len, gap_len=gap_len)


def save_visual(
    sample_dir,
    img,
    gt_polys,
    init_polys,
    pred_polys,
    per_contour_iou,
    matched_pred=None,
    gt_labels=None,
    pred_labels=None,
):
    os.makedirs(sample_dir, exist_ok=True)
    vis_style = os.environ.get('VIS_STYLE', '').strip().lower()
    if vis_style in ('clear', 'clear_gt_pred', 'paper'):
        _save_visual_clear(sample_dir, img, gt_polys, pred_polys, per_contour_iou, matched_pred)
        return
    if vis_style in ('old', 'old_iou', 'legacy', 'very_old', 'old_blue_gt', 'legacy_blue_gt'):
        gt_color = (255, 0, 0) if vis_style in ('very_old', 'old_blue_gt', 'legacy_blue_gt') else (0, 255, 0)
        init_color = (0, 255, 255)
        pred_color = (0, 0, 255)
        if matched_pred is None:
            matched_pred = list(range(min(len(gt_polys), len(pred_polys))))
        show_unmatched = os.environ.get('VIS_SHOW_UNMATCHED', '1') != '0'
        if show_unmatched:
            pred_indices = list(range(len(pred_polys)))
            init_indices = list(range(len(init_polys)))
        else:
            pred_indices = sorted({int(i) for i in matched_pred if 0 <= int(i) < len(pred_polys)})
            init_indices = [i for i in pred_indices if i < len(init_polys)]
        vis = img.copy()
        for poly in gt_polys:
            pts = np.round(poly).astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(vis, [pts], True, gt_color, 2, lineType=cv2.LINE_AA)
        for init_idx in init_indices:
            poly = init_polys[init_idx]
            pts = np.round(poly).astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(vis, [pts], True, init_color, 1, lineType=cv2.LINE_AA)
        for pred_idx in pred_indices:
            poly = pred_polys[pred_idx]
            pts = np.round(poly).astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(vis, [pts], True, pred_color, 2, lineType=cv2.LINE_AA)

        for gt_idx, iou in enumerate(per_contour_iou):
            pred_idx = int(matched_pred[gt_idx]) if gt_idx < len(matched_pred) else gt_idx
            if pred_idx < 0 or pred_idx >= len(pred_polys):
                continue
            poly = pred_polys[pred_idx]
            if len(poly) == 0:
                continue
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            cv2.putText(
                vis,
                f'{iou * 100:.1f}%',
                (cx - 24, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )
        cv2.imwrite(os.path.join(sample_dir, 'overlay.png'), vis)
        return

    vis = img.copy()
    label_vis = img.copy()
    gt_vis = img.copy()
    pred_vis = img.copy()
    init_vis = img.copy()
    pred_all_debug = img.copy()
    init_all_debug = img.copy()
    show_init = os.environ.get('VIS_SHOW_INIT', '0') != '0'
    show_text = os.environ.get('VIS_SHOW_TEXT', '0') != '0'
    show_unmatched = os.environ.get('VIS_SHOW_UNMATCHED', '0') != '0'
    color_mode = os.environ.get('VIS_COLOR_MODE', 'instance').strip().lower()
    use_role_color = color_mode not in ('label', 'label_color', 'old', 'oldstyle')
    gt_role_color = (80, 255, 80)
    pred_role_color = (60, 60, 255)
    init_role_color = (255, 200, 80)

    if matched_pred is None:
        matched_pred = list(range(min(len(gt_polys), len(pred_polys))))
    if gt_labels is None or len(gt_labels) != len(gt_polys):
        gt_labels = np.zeros((len(gt_polys),), dtype=np.int32)
    if pred_labels is None or len(pred_labels) != len(pred_polys):
        pred_labels = np.full((len(pred_polys),), -1, dtype=np.int32)

    pred_to_gt = {}
    for gt_idx, pred_idx in enumerate(matched_pred):
        if 0 <= int(pred_idx) < len(pred_polys):
            pred_to_gt[int(pred_idx)] = gt_idx

    if color_mode == 'instance':
        show_det_box = os.environ.get('VIS_SHOW_DET_BOX', '1') != '0'

        def _pred_instance_color(pred_idx):
            gt_idx = pred_to_gt.get(int(pred_idx))
            return instance_color(gt_idx) if gt_idx is not None else UNMATCHED_COLOR_BGR

        def _show_pred_index(pred_idx):
            return int(pred_idx) in pred_to_gt or show_unmatched

        # Init polygons are still saved separately; overlay drawing is controlled by VIS_SHOW_INIT.
        for pred_idx, poly in enumerate(init_polys):
            color = _pred_instance_color(pred_idx)
            draw_polyline_style(init_all_debug, poly, color, thickness=1, style='dotted')
            if not _show_pred_index(pred_idx):
                continue
            draw_polyline_style(init_vis, poly, color, thickness=1, style='dotted')
            if show_det_box:
                draw_dotted_bbox(vis, poly, color, thickness=1)
                draw_dotted_bbox(label_vis, poly, color, thickness=1)
            if show_init:
                draw_polyline_style(vis, poly, color, thickness=1, style='dotted')
                draw_polyline_style(label_vis, poly, color, thickness=1, style='dotted')

        for gt_idx, poly in enumerate(gt_polys):
            color = instance_color(gt_idx)
            draw_polyline_style(vis, poly, color, thickness=2, style='solid')
            draw_polyline_style(label_vis, poly, color, thickness=2, style='solid')
            draw_polyline_style(gt_vis, poly, color, thickness=2, style='solid')

        for pred_idx, poly in enumerate(pred_polys):
            color = _pred_instance_color(pred_idx)
            draw_polyline_style(pred_all_debug, poly, color, thickness=2, style='dashed')
            if not _show_pred_index(pred_idx):
                continue
            draw_polyline_style(vis, poly, color, thickness=2, style='dashed')
            draw_polyline_style(label_vis, poly, color, thickness=2, style='dashed')
            draw_polyline_style(pred_vis, poly, color, thickness=2, style='dashed')

        if show_text:
            for gt_idx, iou in enumerate(per_contour_iou):
                pred_idx = int(matched_pred[gt_idx]) if gt_idx < len(matched_pred) else -1
                if pred_idx < 0 or pred_idx >= len(pred_polys):
                    continue
                pred_poly = np.asarray(pred_polys[pred_idx], dtype=np.float32)
                if pred_poly.ndim != 2 or pred_poly.shape[0] == 0:
                    continue
                cx = int(pred_poly[:, 0].mean())
                cy = int(pred_poly[:, 1].mean())
                draw_text_with_outline(
                    vis,
                    'g{}:{:.1f}%'.format(gt_idx, iou * 100.0),
                    (cx - 30, cy),
                    font_scale=0.32,
                    thickness=1,
                )

        draw_instance_legend(vis, range(len(gt_polys)))
        cv2.imwrite(os.path.join(sample_dir, 'overlay.png'), vis)
        cv2.imwrite(os.path.join(sample_dir, 'overlay_label_color.png'), label_vis)
        cv2.imwrite(os.path.join(sample_dir, 'gt_only.png'), gt_vis)
        cv2.imwrite(os.path.join(sample_dir, 'pred_only.png'), pred_vis)
        cv2.imwrite(os.path.join(sample_dir, 'init_only.png'), init_vis)
        if not show_unmatched:
            cv2.imwrite(os.path.join(sample_dir, 'pred_all_debug.png'), pred_all_debug)
            cv2.imwrite(os.path.join(sample_dir, 'init_all_debug.png'), init_all_debug)

        if os.environ.get('VIS_SAVE_PAIRS', '1') != '0':
            pair_dir = os.path.join(sample_dir, 'pairs')
            os.makedirs(pair_dir, exist_ok=True)
            for gt_idx, gt_poly in enumerate(gt_polys):
                pred_idx = int(matched_pred[gt_idx]) if gt_idx < len(matched_pred) else -1
                if pred_idx < 0 or pred_idx >= len(pred_polys):
                    continue
                color = instance_color(gt_idx)
                pair_vis = img.copy()
                draw_polyline_style(pair_vis, gt_poly, color, thickness=2, style='solid')
                if show_det_box and pred_idx < len(init_polys):
                    draw_dotted_bbox(pair_vis, init_polys[pred_idx], color, thickness=1)
                draw_polyline_style(pair_vis, pred_polys[pred_idx], color, thickness=2, style='dashed')
                crop_polys = [gt_poly, pred_polys[pred_idx]]
                if pred_idx < len(init_polys):
                    crop_polys.append(init_polys[pred_idx])
                crop = crop_poly_view(pair_vis, crop_polys)
                iou = float(per_contour_iou[gt_idx]) if gt_idx < len(per_contour_iou) else 0.0
                label = int(gt_labels[gt_idx]) if gt_idx < len(gt_labels) else 0
                out_name = f'gt_{gt_idx:03d}_label_{label:02d}_iou_{iou:.3f}.png'
                cv2.imwrite(os.path.join(pair_dir, out_name), crop)
        return

    # Draw init separately by default. When requested in overlay, keep it faint.
    for pred_idx, poly in enumerate(init_polys):
        gt_idx = pred_to_gt.get(pred_idx)
        label = gt_labels[gt_idx] if gt_idx is not None else pred_labels[pred_idx]
        color = soften_color(label_color(label), alpha=0.35) if gt_idx is not None else (150, 150, 150)
        draw_polyline_style(init_all_debug, poly, color, thickness=1, style='dotted')
        if gt_idx is None and not show_unmatched:
            continue
        draw_polyline_style(init_vis, poly, color, thickness=1, style='dotted')
        if show_init:
            draw_polyline_style(vis, poly, init_role_color if use_role_color else color, thickness=1, style='dotted')
            draw_polyline_style(label_vis, poly, color, thickness=1, style='dotted')

    for gt_idx, poly in enumerate(gt_polys):
        color = label_color(gt_labels[gt_idx])
        draw_polyline_style(vis, poly, gt_role_color if use_role_color else color, thickness=2 if use_role_color else 1, style='solid')
        draw_polyline_style(label_vis, poly, color, thickness=1, style='solid')
        draw_polyline_style(gt_vis, poly, gt_role_color if use_role_color else color, thickness=2 if use_role_color else 1, style='solid')

    for pred_idx, poly in enumerate(pred_polys):
        gt_idx = pred_to_gt.get(pred_idx)
        label = gt_labels[gt_idx] if gt_idx is not None else pred_labels[pred_idx]
        color = label_color(label) if gt_idx is not None else (120, 120, 120)
        draw_polyline_style(pred_all_debug, poly, color, thickness=1, style='dashed')
        if gt_idx is None and not show_unmatched:
            continue
        draw_polyline_style(vis, poly, pred_role_color if use_role_color else color, thickness=1, style='dashed')
        draw_polyline_style(label_vis, poly, color, thickness=1, style='dashed')
        draw_polyline_style(pred_vis, poly, pred_role_color if use_role_color else color, thickness=1, style='dashed')

    if show_text:
        for gt_idx, iou in enumerate(per_contour_iou):
            pred_idx = int(matched_pred[gt_idx]) if gt_idx < len(matched_pred) else -1
            if pred_idx < 0 or pred_idx >= len(pred_polys):
                continue
            cx = int(pred_polys[pred_idx, :, 0].mean())
            cy = int(pred_polys[pred_idx, :, 1].mean())
            label = int(gt_labels[gt_idx]) if gt_idx < len(gt_labels) else 0
            cv2.putText(
                vis,
                f'g{gt_idx}/l{label}:{iou * 100:.1f}%',
                (cx - 35, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

    if use_role_color:
        cv2.putText(vis, 'GT solid green', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gt_role_color, 1, lineType=cv2.LINE_AA)
        cv2.putText(vis, 'Pred dashed red', (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pred_role_color, 1, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(sample_dir, 'overlay.png'), vis)
    cv2.imwrite(os.path.join(sample_dir, 'overlay_label_color.png'), label_vis)
    cv2.imwrite(os.path.join(sample_dir, 'gt_only.png'), gt_vis)
    cv2.imwrite(os.path.join(sample_dir, 'pred_only.png'), pred_vis)
    cv2.imwrite(os.path.join(sample_dir, 'init_only.png'), init_vis)
    if not show_unmatched:
        cv2.imwrite(os.path.join(sample_dir, 'pred_all_debug.png'), pred_all_debug)
        cv2.imwrite(os.path.join(sample_dir, 'init_all_debug.png'), init_all_debug)

    if os.environ.get('VIS_SAVE_PAIRS', '1') != '0':
        pair_dir = os.path.join(sample_dir, 'pairs')
        os.makedirs(pair_dir, exist_ok=True)
        for gt_idx, gt_poly in enumerate(gt_polys):
            pred_idx = int(matched_pred[gt_idx]) if gt_idx < len(matched_pred) else -1
            if pred_idx < 0 or pred_idx >= len(pred_polys):
                continue
            label = int(gt_labels[gt_idx]) if gt_idx < len(gt_labels) else 0
            color = label_color(label)
            pair_vis = img.copy()
            draw_polyline_style(pair_vis, gt_poly, color, thickness=1, style='solid')
            draw_polyline_style(pair_vis, pred_polys[pred_idx], color, thickness=1, style='dashed')
            crop = crop_poly_view(pair_vis, [gt_poly, pred_polys[pred_idx]])
            iou = float(per_contour_iou[gt_idx]) if gt_idx < len(per_contour_iou) else 0.0
            out_name = f'gt_{gt_idx:03d}_label_{label:02d}_iou_{iou:.3f}.png'
            cv2.imwrite(os.path.join(pair_dir, out_name), crop)


def _flow_disp_from_zero_latent(
    flow, cnn_feature, current, c_cur, py_ind, steps, return_context=False
):
    steps = max(int(steps), 1)
    ctx = flow.prepare_sampling_context(cnn_feature, current, py_ind)
    x = torch.zeros_like(current)
    x_self_cond = torch.zeros_like(x) if getattr(flow, '_use_self_conditioning', False) else None
    dt = 1.0 / float(steps)
    for idx in range(steps):
        x, _, _, _, next_self_cond = flow.step_with_logprob(
            cnn_feature,
            current,
            c_cur,
            py_ind,
            x_t=x,
            t_value=idx * dt,
            step_index=idx,
            total_steps=steps,
            action_std=0.0,
            prev_sample=None,
            sampled_feat=ctx['sampled_feat'],
            detail_feat=ctx['detail_feat'],
            contour_scale=ctx['contour_scale'],
            x_self_cond=x_self_cond,
            step_mode='gaussian',
        )
        if getattr(flow, '_use_self_conditioning', False):
            x_self_cond = next_self_cond
    disp = flow.clamp_pred_disp(
        flow.denormalize_pred_disp(x, ctx['contour_scale']), current
    )
    return (disp, ctx) if return_context else (disp, ctx['sampled_feat'])


def _deterministic_unified_rollout(
    gcn,
    policy,
    cnn_feature,
    initial,
    py_ind,
    fractions,
    ode_steps,
    active_step_indices=None,
    return_states=False,
    return_step_records=False,
):
    fractions = [float(v) for v in fractions]
    if len(fractions) != _UNIFIED_POLICY_OUTER_STEPS:
        raise ValueError(
            f'Unified rollout requires {_UNIFIED_POLICY_OUTER_STEPS} fractions, got {fractions}'
        )
    current = initial
    active_steps = (
        set(range(len(fractions)))
        if active_step_indices is None
        else {int(v) for v in active_step_indices}
    )
    states = [current]
    step_records = []
    for si, frac in enumerate(fractions):
        c_cur = snake_gcn_utils.img_poly_to_can_poly(current)
        raw_disp, ctx = _flow_disp_from_zero_latent(
            gcn,
            cnn_feature,
            current,
            c_cur,
            py_ind,
            ode_steps,
            return_context=True,
        )
        mean = raw_disp * frac
        action = mean
        if policy is not None and si in active_steps:
            mu, _ = policy(si, current, c_cur, mean, ctx['sampled_feat'], frac)
            scale = torch.tanh(mu) * float(policy.max_scale)
            action = mean * (1.0 + scale.unsqueeze(-1))
        if return_step_records:
            step_records.append({
                'step_index': int(si),
                'fraction': float(frac),
                'state': current,
                'canonical_state': c_cur,
                'fm_velocity': raw_disp,
                'mean_action': mean,
                'sampled_feat': ctx['sampled_feat'],
                'detail_feat': ctx.get('detail_feat'),
                'contour_scale': ctx['contour_scale'],
            })
        current = current + action
        states.append(current)
    disp = current - initial
    if return_step_records:
        return disp, states, step_records
    return (disp, states) if return_states else disp


def _sample_identity(sample, index):
    path = sample.get('img_path') if isinstance(sample, dict) else None
    if isinstance(path, (list, tuple)):
        path = path[0] if path else None
    if isinstance(path, bytes):
        path = path.decode('utf-8', errors='replace')
    if path is not None:
        path = str(path)
    sample_id = os.path.splitext(os.path.basename(path))[0] if path else str(index)
    return {
        'sample_id': sample_id,
        'sample_path': path,
    }


@torch.no_grad()
def prepare_manual_gt_init_context(core, batch, device):
    """Build the manual GT-init feature/contour context used by evaluation."""
    gt_all = batch['i_gt_py']
    if gt_all.numel() == 0:
        raise RuntimeError('No GT polygons in batch')

    batch_size, num_contours, num_points, _ = gt_all.shape
    detector_backend = str(getattr(cfg, 'detector_backend', 'yolo')).strip().lower()
    if detector_backend == 'yolo':
        if not hasattr(core, 'yolo') or core.yolo is None:
            raise RuntimeError('Manual YOLO eval path requested but yolo module is missing')
        yolo_out = core.yolo(batch['inp'])
        feat_list = yolo_out[1] if isinstance(yolo_out, (list, tuple)) and len(yolo_out) > 1 else None
        feat_p2 = feat_list[0] if isinstance(feat_list, (list, tuple)) else yolo_out
        if getattr(core, 'use_swin_snake_feature', False):
            if not hasattr(core, 'swin_snake_feature') or core.swin_snake_feature is None:
                raise RuntimeError('Swin feature evaluation requested but swin_snake_feature is missing')
            cnn_feature = core.swin_snake_feature(batch['inp'])
        else:
            cnn_feature = core.cnn_proj(feat_p2)
        if (
            (not getattr(core, 'use_swin_snake_feature', False))
            and getattr(core, 'use_p3_features', False)
            and hasattr(core, 'cnn_proj_p3')
            and isinstance(feat_list, (list, tuple))
            and len(feat_list) > 1
        ):
            feat_p3 = feat_list[1]
            feat_p3_up = torch.nn.functional.interpolate(
                feat_p3, size=feat_p2.shape[-2:], mode='bilinear', align_corners=False
            )
            cnn_feature = cnn_feature + core.cnn_proj_p3(feat_p3_up)
    elif (
        detector_backend.startswith('heatmap_')
        or detector_backend.startswith('convnext')
        or detector_backend.startswith('moonvit')
    ):
        if not hasattr(core, 'heatmap_detector') or core.heatmap_detector is None:
            raise RuntimeError(
                f'Manual eval path requires heatmap_detector, got detector_backend={detector_backend}'
            )
        cnn_feature, _ct_hm, _wh, mask_logits = core.heatmap_detector(batch['inp'])
        if mask_logits is not None:
            mask_guidance_alpha = float(getattr(cfg, 'heatmap_mask_guidance_alpha', 0.0))
            if mask_guidance_alpha > 0.0:
                mask_guidance = torch.sigmoid(mask_logits).amax(dim=1, keepdim=True)
                cnn_feature = cnn_feature * (1.0 + mask_guidance_alpha * mask_guidance)
    else:
        raise RuntimeError(f'Manual GT-init eval does not support detector_backend={detector_backend}')

    locate_feat_stats = {}
    if hasattr(core, 'apply_locate_feature_injection'):
        cnn_feature, locate_feat_stats = core.apply_locate_feature_injection(cnn_feature, batch)
    if hasattr(core, 'apply_locate_feature_replacement'):
        cnn_feature, replace_stats = core.apply_locate_feature_replacement(cnn_feature, batch)
        locate_feat_stats.update(replace_stats)

    contour_init_method = str(getattr(cfg, 'contour_init_method', 'octagon')).strip().lower()
    sam_prompt_source = str(getattr(cfg, 'sam_prompt_source', 'yolo_box')).strip().lower()
    use_sam_gt_box_init = contour_init_method in ('sam', 'efficient_sam') and sam_prompt_source == 'gt_box'
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
        counts = (
            batch['ct_01'].bool().sum(dim=1).tolist()
            if 'ct_01' in batch
            else [num_contours] * batch_size
        )
        py_ind = torch.cat([
            torch.full((int(count),), i, dtype=torch.long, device=device)
            for i, count in enumerate(counts)
        ])

    if 'ct_01' in batch:
        gt_flat = gt_all[batch['ct_01'].bool()]
    else:
        gt_flat = gt_all.view(-1, num_points, 2)
    count = min(i_it_py.size(0), gt_flat.size(0), py_ind.size(0))
    if count <= 0:
        raise RuntimeError('Manual GT-init context contains no valid contours')
    return {
        'cnn_feature': cnn_feature,
        'i_it_py': i_it_py[:count],
        'c_it_py': c_it_py[:count],
        'i_gt_py': gt_flat[:count],
        'py_ind': py_ind[:count],
        'image_hw': (int(batch['inp'].shape[-2]), int(batch['inp'].shape[-1])),
        'locate_feat_stats': locate_feat_stats,
    }


def eval_sample(model, device, batch, policy=None, policy_metadata=None, ode_steps=10, save_visuals=False, sample_dir=None):
    for k, v in batch.items():
        if k == 'locate_feat' or str(k).startswith('locate_feat_'):
            continue
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
        not bool(getattr(cfg, 'eval_manual_gt_init', False))
    ) and (
        use_sam_yolo_init
        or detector_backend != 'yolo'
        or bool(getattr(cfg, 'eval_use_network_forward', False))
    )
    use_unified_policy_rollout = bool(
        policy_metadata
        and policy_metadata.get('checkpoint_has_per_point_policy', False)
    )
    if use_unified_policy_rollout and use_full_forward:
        raise RuntimeError(
            'Unified per-point off/mean evaluation requires the explicit manual '
            'contour rollout path; eval_use_network_forward must be false'
        )

    with torch.no_grad():
        gt_all = batch['i_gt_py']
        if gt_all.numel() == 0:
            raise RuntimeError('No GT polygons in batch')

        batch_size, num_contours, num_points, _ = gt_all.shape
        gt_labels = None
        pred_labels = None

        if use_full_forward:
            net_out = core(batch['inp'], batch)
            if 'py' not in net_out:
                raise RuntimeError('Network output has no predicted polygons')
            pred_polys = net_out['py'].detach().cpu().numpy() * dr
            init_src = net_out.get('i_it_py', torch.zeros_like(net_out['py']))
            init_polys = init_src.detach().cpu().numpy() * dr
            if 'ct_01' in batch:
                keep = batch['ct_01'].bool()
                gt_polys = gt_all[keep].detach().cpu().numpy() * dr
                if 'ct_cls' in batch:
                    gt_labels = batch['ct_cls'][keep].detach().cpu().numpy().astype(np.int32)
            else:
                gt_polys = gt_all.view(-1, num_points, 2).detach().cpu().numpy() * dr
                if 'ct_cls' in batch:
                    gt_labels = batch['ct_cls'].view(-1).detach().cpu().numpy().astype(np.int32)
            if 'detection' in net_out and torch.is_tensor(net_out['detection']):
                det = net_out['detection'].detach()
                if det.numel() > 0 and det.size(-1) >= 6:
                    det_flat = det.reshape(-1, det.size(-1))
                    det_flat = det_flat[det_flat[:, 4] > 1e-4]
                    pred_labels = det_flat[:, 5].detach().cpu().numpy().astype(np.int32)
            match_by_iou = not bool(getattr(cfg, 'use_gt_det', False))
        else:
            manual_context = prepare_manual_gt_init_context(core, batch, device)
            cnn_feature = manual_context['cnn_feature']
            i_it_py = manual_context['i_it_py']
            c_it_py = manual_context['c_it_py']
            py_ind = manual_context['py_ind']

            use_curve_path = bool(getattr(cfg, 'use_curve_inference', False))
            if use_unified_policy_rollout:
                disp = _deterministic_unified_rollout(
                    core.gcn,
                    policy,
                    cnn_feature,
                    i_it_py,
                    py_ind,
                    policy_metadata['fractions'],
                    policy_metadata['actual_ode_steps'],
                    active_step_indices=policy_metadata['active_step_indices'],
                )
            elif use_curve_path and hasattr(core.gcn, 'sample_disp_curve'):
                curve_alpha = float(getattr(cfg, 'curve_alpha', 2.0))
                curve_steps = int(getattr(cfg, 'curve_steps', 20))
                curve_s_max = float(getattr(cfg, 'curve_s_max', 0.97))
                curve_resample = bool(getattr(cfg, 'curve_resample_feat', True))
                disp = core.gcn.sample_disp_curve(
                    cnn_feature,
                    i_it_py,
                    c_it_py,
                    py_ind,
                    alpha=curve_alpha,
                    steps=curve_steps,
                    noise_scale=None,
                    batch=batch,
                    s_max=curve_s_max,
                    resample_feat=curve_resample,
                )
            elif getattr(core.gcn, 'use_iterative_refinement', False):
                iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
                use_rich_infer_schedule = bool(getattr(cfg, 'v4_9_use_rich_infer_schedule', False))
                if use_rich_infer_schedule and hasattr(core.gcn, '_progress_targets_to_residual_fractions'):
                    targets = list(getattr(cfg, 'v4_9_infer_target_fractions', []))
                    if not targets:
                        targets = [0.3333, 0.5, 0.80, 0.97, 1.0]
                    fractions = core.gcn._progress_targets_to_residual_fractions(targets)
                    iter_steps = len(fractions)
                else:
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
                if hasattr(core.gcn, 'ode_steps'):
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
                    disp = core.gcn.sample_disp_iterative(
                        cnn_feature,
                        i_it_py,
                        c_it_py,
                        py_ind,
                        num_iter_steps=iter_steps,
                        fractions=fractions,
                        ddim_steps=iter_ode_steps,
                    )
            else:
                disp = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind, steps=ode_steps)
            fk = int(getattr(cfg, 'fourier_smooth_k', 0))
            if fk > 0:
                from lib.networks.diffusion.flow_matching_evolution import FlowMatchingEvolution
                disp = FlowMatchingEvolution.fourier_smooth(disp, fk)

            pred_polys = (i_it_py + disp).cpu().numpy() * dr
            gt_polys = manual_context['i_gt_py'].cpu().numpy() * dr
            init_polys = i_it_py.cpu().numpy() * dr
            if 'ct_cls' in batch:
                labels = batch['ct_cls'][batch['ct_01'].bool()] if 'ct_01' in batch else batch['ct_cls'].view(-1)
                gt_labels = labels[:i_it_py.size(0)].detach().cpu().numpy().astype(np.int32)
                pred_labels = gt_labels.copy()
            match_by_iou = False

    if 'orig_img' in batch:
        img_raw = batch['orig_img'][0]
        img = img_raw.detach().cpu().numpy() if torch.is_tensor(img_raw) else img_raw
        img = img.astype(np.uint8)
    else:
        img = np.zeros((512, 512, 3), dtype=np.uint8)

    height, width = img.shape[:2]
    per_contour_iou, per_contour_dice, per_contour_mboundf, per_contour_nsd, matched_pred = compute_ordered_or_matched_metrics(
        pred_polys,
        gt_polys,
        height,
        width,
        match_by_iou=match_by_iou,
    )
    init_iou, init_dice, init_mboundf, init_nsd, _ = compute_ordered_or_matched_metrics(
        init_polys,
        gt_polys,
        height,
        width,
        match_by_iou=match_by_iou,
    )

    if save_visuals and sample_dir is not None:
        save_visual(
            sample_dir,
            img,
            gt_polys,
            init_polys,
            pred_polys,
            per_contour_iou,
            matched_pred=matched_pred,
            gt_labels=gt_labels,
            pred_labels=pred_labels,
        )

    return {
        'mean_iou': float(np.mean(per_contour_iou)) if per_contour_iou else 0.0,
        'mean_dice': float(np.mean(per_contour_dice)) if per_contour_dice else 0.0,
        'mean_mboundf': float(np.mean(per_contour_mboundf)) if per_contour_mboundf else 0.0,
        'mean_nsd': float(np.mean(per_contour_nsd)) if per_contour_nsd else 0.0,
        'mean_init_iou': float(np.mean(init_iou)) if init_iou else 0.0,
        'mean_init_dice': float(np.mean(init_dice)) if init_dice else 0.0,
        'mean_init_mboundf': float(np.mean(init_mboundf)) if init_mboundf else 0.0,
        'mean_init_nsd': float(np.mean(init_nsd)) if init_nsd else 0.0,
        'per_contour_iou': per_contour_iou,
        'per_contour_dice': per_contour_dice,
        'per_contour_mboundf': per_contour_mboundf,
        'per_contour_nsd': per_contour_nsd,
        'per_contour_init_iou': init_iou,
        'per_contour_init_dice': init_dice,
        'per_contour_init_mboundf': init_mboundf,
        'per_contour_init_nsd': init_nsd,
        'matched_pred': matched_pred,
        'num_pred_contours': int(pred_polys.shape[0]),
        'num_gt_contours': int(gt_polys.shape[0]),
    }


def main():
    apply_gpu_override()
    ablation_mode = apply_ablation_mode()

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

    model, device, ckpt_path, policy, policy_metadata = load_model(
        ckpt, return_policy=True, eval_ode_steps=ode_steps
    )

    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)
    set_eval_seed(eval_seed)

    dataset_size = len(dataset)
    limit = min(dataset_size, max_samples) if max_samples is not None else dataset_size

    rows = []
    sample_mean_ious = []
    sample_mean_dices = []
    sample_mean_mboundfs = []
    sample_mean_nsds = []
    sample_mean_init_ious = []
    sample_mean_init_mboundfs = []
    sample_mean_init_nsds = []
    all_contour_ious = []
    all_contour_dices = []
    all_contour_mboundfs = []
    all_contour_nsds = []
    all_contour_init_ious = []
    all_contour_init_mboundfs = []
    all_contour_init_nsds = []
    failed_indices = []

    print(f'[*] Evaluating {limit} / {dataset_size} samples from {cfg.test.dataset}')
    print(f'[*] ODE steps: {ode_steps} | save_visuals={save_visuals}')

    for index in range(limit):
        print(f'[{index + 1}/{limit}] sample {index}')
        sample_dir = os.path.join(per_sample_root, f'idx_{index:03d}') if save_visuals else ''
        identity = _sample_identity(None, index)
        try:
            sample = dataset[index]
            identity = _sample_identity(sample, index)
            batch = collator([sample])
            result = eval_sample(
                model,
                device,
                batch,
                policy=policy,
                policy_metadata=policy_metadata,
                ode_steps=ode_steps,
                save_visuals=save_visuals,
                sample_dir=sample_dir if save_visuals else None,
            )
            rows.append({
                'index': index,
                **identity,
                'ok': True,
                'mean_iou': result['mean_iou'],
                'mean_dice': result['mean_dice'],
                'mean_mboundf': result['mean_mboundf'],
                'mean_nsd': result['mean_nsd'],
                'mean_init_iou': result['mean_init_iou'],
                'mean_init_mboundf': result['mean_init_mboundf'],
                'mean_init_nsd': result['mean_init_nsd'],
                'per_contour_iou': result['per_contour_iou'],
                'per_contour_dice': result['per_contour_dice'],
                'per_contour_mboundf': result['per_contour_mboundf'],
                'per_contour_nsd': result['per_contour_nsd'],
                'per_contour_init_iou': result['per_contour_init_iou'],
                'per_contour_init_nsd': result['per_contour_init_nsd'],
                'matched_pred': result['matched_pred'],
                'num_pred_contours': result['num_pred_contours'],
                'num_gt_contours': result['num_gt_contours'],
                'dir': sample_dir,
            })
            sample_mean_ious.append(result['mean_iou'])
            sample_mean_dices.append(result['mean_dice'])
            sample_mean_mboundfs.append(result['mean_mboundf'])
            sample_mean_nsds.append(result['mean_nsd'])
            sample_mean_init_ious.append(result['mean_init_iou'])
            sample_mean_init_mboundfs.append(result['mean_init_mboundf'])
            sample_mean_init_nsds.append(result['mean_init_nsd'])
            all_contour_ious.extend(result['per_contour_iou'])
            all_contour_dices.extend(result['per_contour_dice'])
            all_contour_mboundfs.extend(result['per_contour_mboundf'])
            all_contour_nsds.extend(result['per_contour_nsd'])
            all_contour_init_ious.extend(result['per_contour_init_iou'])
            all_contour_init_mboundfs.extend(result['per_contour_init_mboundf'])
            all_contour_init_nsds.extend(result['per_contour_init_nsd'])
        except Exception as exc:
            failed_indices.append(index)
            rows.append({
                'index': index,
                **identity,
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
        'ablation_mode': ablation_mode,
        'eval_use_network_forward': bool(getattr(cfg, 'eval_use_network_forward', False)),
        'use_gt_det': bool(getattr(cfg, 'use_gt_det', False)),
        'use_pred_extreme_init_for_inference': bool(getattr(cfg, 'use_pred_extreme_init_for_inference', False)),
        'diffusion_init_source': str(getattr(cfg, 'diffusion_init_source', 'extreme')),
        'eval_seed': eval_seed,
        'ode_steps': ode_steps,
        'nsd_delta_px': float(os.environ.get('NSD_DELTA_PX', '2.0')),
        'fm_policy': policy_metadata,
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
        'mean_nsd_sample_avg': float(np.mean(sample_mean_nsds)) if sample_mean_nsds else 0.0,
        'mean_nsd_contour_avg': float(np.mean(all_contour_nsds)) if all_contour_nsds else 0.0,
        'mean_init_iou_sample_avg': float(np.mean(sample_mean_init_ious)) if sample_mean_init_ious else 0.0,
        'mean_init_iou_contour_avg': float(np.mean(all_contour_init_ious)) if all_contour_init_ious else 0.0,
        'mean_init_mboundf_sample_avg': float(np.mean(sample_mean_init_mboundfs)) if sample_mean_init_mboundfs else 0.0,
        'mean_init_mboundf_contour_avg': float(np.mean(all_contour_init_mboundfs)) if all_contour_init_mboundfs else 0.0,
        'mean_init_nsd_sample_avg': float(np.mean(sample_mean_init_nsds)) if sample_mean_init_nsds else 0.0,
        'mean_init_nsd_contour_avg': float(np.mean(all_contour_init_nsds)) if all_contour_init_nsds else 0.0,
        'median_iou_sample_avg': float(np.median(sample_mean_ious)) if sample_mean_ious else 0.0,
        'std_iou_sample_avg': float(np.std(sample_mean_ious)) if sample_mean_ious else 0.0,
        'sample_mean_ious': sample_mean_ious,
        'sample_mean_dices': sample_mean_dices,
        'sample_mean_mboundfs': sample_mean_mboundfs,
        'sample_mean_nsds': sample_mean_nsds,
        'sample_mean_init_ious': sample_mean_init_ious,
        'sample_mean_init_mboundfs': sample_mean_init_mboundfs,
        'sample_mean_init_nsds': sample_mean_init_nsds,
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
    print(f"mean_nsd_sample_avg:   {summary['mean_nsd_sample_avg']:.6f}")
    print(f"mean_nsd_contour_avg:  {summary['mean_nsd_contour_avg']:.6f}")
    print(f"mean_init_nsd_sample_avg: {summary['mean_init_nsd_sample_avg']:.6f}")
    print(f"median_iou_sample_avg: {summary['median_iou_sample_avg']:.6f}")
    print(f"std_iou_sample_avg:    {summary['std_iou_sample_avg']:.6f}")
    print(f"failed_samples:        {summary['failed_samples']}")
    print('=' * 80)


if __name__ == '__main__':
    main()
