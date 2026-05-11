import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

_custom_parser = argparse.ArgumentParser(add_help=False)
_custom_parser.add_argument('--tag', default='', type=str)
_custom_parser.add_argument('--out_dir', default='', type=str)
_custom_parser.add_argument('--index', default=0, type=int)
_custom_parser.add_argument('--annotate_pred_count', default=0, type=int)
_custom_args, _remaining_argv = _custom_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining_argv

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..'))
_DEFAULT_CFG = os.path.join(_ROOT_DIR, 'configs', 'btcv_diffusion_dit_v3.yaml')

if not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = _DEFAULT_CFG

from lib.config import cfg, args

# Allow EVAL_GPU env var to override the GPU set by config (same as eval_v37_full_iou.py)
_eval_gpu = os.environ.get('EVAL_GPU', '').strip()
if _eval_gpu:
    cfg.gpus = [int(_eval_gpu)]
    os.environ['CUDA_VISIBLE_DEVICES'] = _eval_gpu
    print(f'[*] Override infer GPU -> {_eval_gpu}')

from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.datasets.collate_batch import make_collator
from lib.utils.snake import snake_config


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def draw_poly(img, poly, color, thickness=2):
    if poly is None:
        return
    pts = np.asarray(poly, dtype=np.int32)
    if pts.size == 0:
        return
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def apply_valid_point_mask(poly_pts, point_mask_row=None):
    pts = np.asarray(poly_pts, dtype=np.float32)
    if point_mask_row is None:
        return pts
    vm = np.asarray(point_mask_row).astype(np.float32) > 0.5
    if vm.ndim != 1 or vm.shape[0] != pts.shape[0]:
        return pts
    if vm.sum() < 3:
        return pts
    return pts[vm]


def remove_extreme_points(poly_pts, passes=2, edge_factor=3.0, turn_ratio_thr=2.8, radial_z_thr=3.0):
    """Suppress isolated spike points by local interpolation on closed polygons."""
    pts = np.asarray(poly_pts, dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[0] < 6:
        return pts

    for _ in range(int(max(1, passes))):
        prev_pts = np.roll(pts, 1, axis=0)
        next_pts = np.roll(pts, -1, axis=0)

        e_prev = np.linalg.norm(pts - prev_pts, axis=1)
        e_next = np.linalg.norm(next_pts - pts, axis=1)
        med_edge = float(np.median(np.concatenate([e_prev, e_next])) + 1e-6)

        chord = np.linalg.norm(next_pts - prev_pts, axis=1) + 1e-6
        detour_ratio = (e_prev + e_next) / chord

        center = pts.mean(axis=0, keepdims=True)
        radius = np.linalg.norm(pts - center, axis=1)
        r_med = float(np.median(radius))
        r_mad = float(np.median(np.abs(radius - r_med)) + 1e-6)
        radial_z = np.abs(radius - r_med) / (1.4826 * r_mad + 1e-6)

        long_both = (e_prev > edge_factor * med_edge) & (e_next > edge_factor * med_edge)
        strong_spike = detour_ratio > turn_ratio_thr
        radial_out = radial_z > radial_z_thr
        outlier = (long_both & (strong_spike | radial_out)) | (detour_ratio > (turn_ratio_thr + 1.2))
        idx = np.where(outlier)[0]
        if idx.size == 0:
            break
        for i in idx:
            pts[i] = 0.5 * (prev_pts[i] + next_pts[i])
    return pts


def clip_poly(poly_pts, h, w):
    pts = np.asarray(poly_pts, dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[0] == 0:
        return pts
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(w - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(h - 1))
    return pts


def annotate_pred_point_counts(img, pred_np, valid_counts=None):
    for idx, poly in enumerate(pred_np):
        pts = np.asarray(poly, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0:
            continue
        center = pts.mean(axis=0)
        x = int(np.clip(center[0], 0, img.shape[1] - 1))
        y = int(np.clip(center[1], 0, img.shape[0] - 1))
        if valid_counts is not None and idx < len(valid_counts):
            n_pts = int(valid_counts[idx])
        else:
            n_pts = int(pts.shape[0])
        text = f"#{idx} pts={n_pts}"
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def fourier_low_pass(disp, k):
    """Low-pass filter on a closed contour displacement."""
    if disp.numel() == 0 or k <= 0:
        return disp
    B, N, C = disp.shape
    freq = torch.fft.rfft(disp, dim=1)
    mask = torch.zeros(freq.shape[1], device=disp.device, dtype=torch.bool)
    mask[:k + 1] = True
    freq[:, ~mask, :] = 0
    return torch.fft.irfft(freq, n=N, dim=1)


def load_model(ckpt_path):
    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True

    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    ckpt_obj = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt_obj.get('state_dict') or ckpt_obj.get('model') or ckpt_obj.get('net') or ckpt_obj

    try:
        from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
        sd = remap_legacy_state_dict(sd)
    except Exception:
        pass

    model = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    model.load_state_dict(sd, strict=False)
    return trainer.network.to(device).eval(), device, ckpt_obj


def infer_one(model, device, sample, out_path):
    batch = make_collator(cfg)([sample])
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    core = model.net if hasattr(model, 'net') else model
    with torch.no_grad():
        yolo_out = core.yolo(batch['inp'])
        feat_list = yolo_out[1] if isinstance(yolo_out, (list, tuple)) and len(yolo_out) > 1 else None
        if isinstance(feat_list, (list, tuple)):
            p2 = feat_list[0]
        else:
            p2 = feat_list
        if p2 is None:
            raise RuntimeError('YOLO feature P2 not found')
        cnn_feature = core.cnn_proj(p2)
        if getattr(core, 'use_p3_features', False) and hasattr(core, 'cnn_proj_p3'):
            if isinstance(feat_list, (list, tuple)) and len(feat_list) > 1:
                p3 = feat_list[1]
                p3_up = torch.nn.functional.interpolate(
                    p3, size=p2.shape[-2:], mode='bilinear', align_corners=False
                )
                cnn_feature = cnn_feature + core.cnn_proj_p3(p3_up)

        if 'ct_01' in batch:
            mask = batch['ct_01'][0].bool()
        elif 'meta' in batch and 'ct_num' in batch['meta']:
            ct_num = int(batch['meta']['ct_num'][0].item())
            mask = torch.zeros((batch['i_it_py'].shape[1],), dtype=torch.bool, device=device)
            mask[:ct_num] = True
        else:
            mask = torch.ones((batch['i_it_py'].shape[1],), dtype=torch.bool, device=device)

        i_it_py = batch['i_it_py'][0][mask]
        c_it_py = batch['c_it_py'][0][mask]
        i_gt_py = batch['i_gt_py'][0][mask]
        i_gt_4py = batch['i_gt_4py'][0][mask] if 'i_gt_4py' in batch else None
        point_mask = batch['point_mask'][0][mask] if 'point_mask' in batch else None
        py_ind = torch.zeros((i_it_py.size(0),), dtype=torch.long, device=device)

        # V3.5 uses its own Fourier-space forward path.
        # For all other models, keep the normal single-sample initialization
        # and only apply Fourier smoothing to the final displacement.
        if getattr(cfg, 'use_dit_v3_5', False):
            output, _, _, _ = model(batch)
            pred_py = output.get('py', i_it_py)
            init_py = output.get('i_it_py', i_it_py)
        elif i_it_py.numel() == 0:
            pred_py = i_it_py
            init_py = i_it_py
        else:
            if getattr(cfg, 'use_iterative_refinement', False):
                iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
                fractions = list(getattr(cfg, 'iterative_fractions', []))
                if not fractions:
                    fractions = [1.0 / (iter_steps - i) for i in range(iter_steps)]
                # DiffusionEvolution uses `ddim_steps`; FlowMatchingEvolution uses `ode_steps`.
                if hasattr(core.gcn, 'ode_steps'):
                    ode_steps = int(getattr(core.gcn, 'ode_steps', getattr(cfg, 'flow_ode_steps', 10)))
                    disp = core.gcn.sample_disp_iterative(
                        cnn_feature, i_it_py, c_it_py, py_ind,
                        num_iter_steps=iter_steps, fractions=fractions, ode_steps=ode_steps,
                    )
                else:
                    ddim_steps = int(getattr(cfg, 'iterative_ddim_steps', 20))
                    disp = core.gcn.sample_disp_iterative(
                        cnn_feature, i_it_py, c_it_py, py_ind,
                        num_iter_steps=iter_steps, fractions=fractions, ddim_steps=ddim_steps,
                    )
            else:
                disp = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind)
            k = int(getattr(cfg, 'fourier_smooth_k', 0))
            if k > 0:
                disp = fourier_low_pass(disp, k)
            pred_py = i_it_py + disp
            init_py = i_it_py

    dr = float(snake_config.down_ratio)
    orig_img = to_numpy(batch['orig_img'][0]).astype(np.uint8).copy()
    init_np = to_numpy(init_py) * dr
    pred_np = to_numpy(pred_py) * dr
    gt_np = to_numpy(i_gt_py) * dr
    gt4_np = to_numpy(i_gt_4py) * dr if i_gt_4py is not None else None

    point_mask_np = to_numpy(point_mask) if point_mask is not None else None
    remove_extreme = int(os.environ.get('REMOVE_EXTREME_POINTS', '1')) > 0
    extreme_passes = int(os.environ.get('EXTREME_PASSES', '2'))
    edge_factor = float(os.environ.get('EXTREME_EDGE_FACTOR', '3.0'))
    turn_ratio_thr = float(os.environ.get('EXTREME_TURN_RATIO', '2.8'))
    radial_z_thr = float(os.environ.get('EXTREME_RADIAL_Z', '3.0'))
    h_img, w_img = orig_img.shape[:2]

    gt_vis, init_vis, pred_vis = [], [], []
    for idx in range(pred_np.shape[0]):
        pm_row = point_mask_np[idx] if point_mask_np is not None and idx < point_mask_np.shape[0] else None
        gt_poly = apply_valid_point_mask(gt_np[idx], pm_row)
        init_poly = apply_valid_point_mask(init_np[idx], pm_row)
        pred_poly = apply_valid_point_mask(pred_np[idx], pm_row)
        if remove_extreme:
            pred_poly = remove_extreme_points(
                pred_poly,
                passes=extreme_passes,
                edge_factor=edge_factor,
                turn_ratio_thr=turn_ratio_thr,
                radial_z_thr=radial_z_thr,
            )
        gt_vis.append(clip_poly(gt_poly, h_img, w_img))
        init_vis.append(clip_poly(init_poly, h_img, w_img))
        pred_vis.append(clip_poly(pred_poly, h_img, w_img))

    for poly in gt_vis:
        draw_poly(orig_img, poly, (255, 0, 0), thickness=2)
    for poly in init_vis:
        draw_poly(orig_img, poly, (0, 255, 255), thickness=1)
    for poly in pred_vis:
        draw_poly(orig_img, poly, (0, 0, 255), thickness=2)
    if int(_custom_args.annotate_pred_count) > 0:
        valid_counts = None
        if point_mask is not None:
            valid_counts = to_numpy(point_mask).sum(axis=1).astype(np.int32).tolist()
        annotate_pred_point_counts(orig_img, pred_vis, valid_counts=valid_counts)
    if gt4_np is not None:
        for poly in gt4_np:
            draw_poly(orig_img, poly, (255, 0, 255), thickness=1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, orig_img)


def main():
    cfg_file = getattr(args, 'cfg_file', '') or os.environ.get('CFG_FILE', _DEFAULT_CFG)
    os.environ['CFG_FILE'] = cfg_file
    cfg.merge_from_file(cfg_file)
    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True

    ckpt_path = getattr(args, 'ckpt', '') or os.environ.get('ONE_SAMPLE_CKPT', '')
    if not ckpt_path:
        ckpt_path = os.path.join(_ROOT_DIR, 'data', 'outputs', Path(cfg_file).stem, 'checkpoints', 'latest.pt')

    model, device, ckpt_obj = load_model(ckpt_path)
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, is_train=False), is_train=False)
    if len(dataset) == 0:
        raise RuntimeError('Empty dataset')

    sample = dataset[min(max(int(_custom_args.index), 0), len(dataset) - 1)]
    out_dir = _custom_args.out_dir or os.path.join(_THIS_DIR, 'visual', 'single_sample_all_models')
    tag = _custom_args.tag or Path(cfg_file).stem
    epoch = ckpt_obj.get('epoch', -1) if isinstance(ckpt_obj, dict) else -1
    out_path = os.path.join(out_dir, f'{tag}_idx{_custom_args.index}_epoch{epoch}.png')
    infer_one(model, device, sample, out_path)
    print(out_path)


if __name__ == '__main__':
    main()
