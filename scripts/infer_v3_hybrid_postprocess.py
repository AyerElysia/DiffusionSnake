#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--cfg_file', default='', type=str)
_parser.add_argument('--ckpt', default='', type=str)
_parser.add_argument('--tag', default='', type=str)
_parser.add_argument('--out_dir', default='', type=str)
_parser.add_argument('--index', default=0, type=int)
_parser.add_argument('--keep_k', default=12, type=int)
_parser.add_argument('--outlier_z', default=2.0, type=float)
_parser.add_argument('--blend', default=0.90, type=float)
_parser.add_argument('--high_freq_scale', default=0.05, type=float)
_custom_args, _remaining_argv = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining_argv

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..'))
_DEFAULT_CFG = os.path.join(_ROOT_DIR, 'configs', 'btcv_diffusion_dit_v3_4_single_overfit.yaml')

if _custom_args.cfg_file:
    os.environ['CFG_FILE'] = _custom_args.cfg_file
elif not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = _DEFAULT_CFG

sys.path.insert(0, _ROOT_DIR)

from lib.config import cfg, args
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


def _local_ruggedness_score(contour):
    contour = np.asarray(contour, dtype=np.float32)
    prev = np.roll(contour, 1, axis=0)
    nxt = np.roll(contour, -1, axis=0)
    curvature = np.linalg.norm(prev - 2.0 * contour + nxt, axis=1)
    chord = np.linalg.norm(nxt - prev, axis=1) + 1e-6
    center = np.median(contour, axis=0)
    radial = np.linalg.norm(contour - center[None, :], axis=1)
    radial_med = np.median(radial)
    radial_mad = np.median(np.abs(radial - radial_med)) + 1e-6
    radial_z = np.abs(radial - radial_med) / radial_mad
    return curvature / chord + 0.35 * radial_z


def _repair_spikes(contour, z_thresh=2.0, max_drop_ratio=0.35):
    """Drop very unstable points and interpolate them from neighbors."""
    contour = np.asarray(contour, dtype=np.float32)
    n = contour.shape[0]
    if contour.ndim != 2 or n < 5:
        return contour

    score = _local_ruggedness_score(contour)
    med = np.median(score)
    mad = np.median(np.abs(score - med)) + 1e-6
    invalid = score > (med + float(z_thresh) * mad)

    invalid_ratio = float(invalid.mean())
    if invalid_ratio > max_drop_ratio:
        keep_cut = np.quantile(score, 1.0 - float(max_drop_ratio))
        invalid = score > keep_cut

    if not invalid.any():
        return contour
    if invalid.mean() >= 0.5:
        return contour

    valid = np.where(~invalid)[0]
    if valid.size < 2:
        return contour

    repaired = contour.copy()
    m = valid.size
    for i in range(m):
        a = int(valid[i])
        b = int(valid[(i + 1) % m])
        gap = (b - a) % n
        if gap <= 1:
            continue
        p0 = contour[a]
        p1 = contour[b]
        for t in range(1, gap):
            idx = (a + t) % n
            alpha = t / float(gap)
            repaired[idx] = (1.0 - alpha) * p0 + alpha * p1

    return repaired


def _spectral_sharpen(contour, keep_k=12, high_freq_scale=0.05, blend=0.90):
    """Keep low frequencies and softly shrink high frequencies."""
    contour = np.asarray(contour, dtype=np.float32)
    n = contour.shape[0]
    if contour.ndim != 2 or n < 5:
        return contour

    keep_k = int(max(0, keep_k))
    high_freq_scale = float(np.clip(high_freq_scale, 0.0, 1.0))
    blend = float(np.clip(blend, 0.0, 1.0))

    freq = np.fft.rfft(contour, axis=0)
    keep = min(keep_k + 1, freq.shape[0])
    if keep < freq.shape[0]:
        freq[keep:] *= high_freq_scale
    smoothed = np.fft.irfft(freq, n=n, axis=0).astype(np.float32)
    return blend * smoothed + (1.0 - blend) * contour


def hybrid_postprocess(contour, keep_k=12, outlier_z=2.0, blend=0.90, high_freq_scale=0.05):
    contour = _repair_spikes(contour, z_thresh=outlier_z)
    contour = _spectral_sharpen(
        contour,
        keep_k=keep_k,
        high_freq_scale=high_freq_scale,
        blend=blend,
    )
    return contour


def load_model(cfg_file, ckpt_path):
    cfg.merge_from_file(cfg_file)
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
        p2 = feat_list[0] if isinstance(feat_list, (list, tuple)) else feat_list
        if p2 is None:
            raise RuntimeError('YOLO feature P2 not found')
        cnn_feature = core.cnn_proj(p2)

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
        py_ind = torch.zeros((i_it_py.size(0),), dtype=torch.long, device=device)

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
                ddim_steps = int(getattr(cfg, 'iterative_ddim_steps', 20))
                disp = core.gcn.sample_disp_iterative(
                    cnn_feature,
                    i_it_py,
                    c_it_py,
                    py_ind,
                    num_iter_steps=iter_steps,
                    fractions=fractions,
                    ddim_steps=ddim_steps,
                )
            else:
                disp = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind)

            pred_py = i_it_py + disp
            init_py = i_it_py

        dr = float(snake_config.down_ratio)
        pred_np = pred_py.detach().cpu().numpy() * dr
        init_np = init_py.detach().cpu().numpy() * dr
        gt_np = i_gt_py.detach().cpu().numpy() * dr
        gt4_np = i_gt_4py.detach().cpu().numpy() * dr if i_gt_4py is not None else None

        post_np = []
        for poly in pred_np:
            post_np.append(
                hybrid_postprocess(
                    poly,
                    keep_k=int(_custom_args.keep_k),
                    outlier_z=float(_custom_args.outlier_z),
                    blend=float(_custom_args.blend),
                    high_freq_scale=float(_custom_args.high_freq_scale),
                )
            )
        post_np = np.asarray(post_np, dtype=np.float32)

    img_raw = batch['orig_img'][0]
    img = img_raw.detach().cpu().numpy() if torch.is_tensor(img_raw) else img_raw
    img = img.astype(np.uint8)

    for poly in gt_np:
        draw_poly(img, poly, (0, 255, 0), thickness=2)
    for poly in init_np:
        draw_poly(img, poly, (0, 255, 255), thickness=1)
    for poly in pred_np:
        draw_poly(img, poly, (255, 0, 0), thickness=1)
    for poly in post_np:
        draw_poly(img, poly, (0, 0, 255), thickness=2)
    if gt4_np is not None:
        for poly in gt4_np:
            draw_poly(img, poly, (255, 0, 255), thickness=1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)


def main():
    cfg_file = _custom_args.cfg_file or os.environ.get('CFG_FILE', _DEFAULT_CFG)
    os.environ['CFG_FILE'] = cfg_file

    ckpt_path = _custom_args.ckpt or os.environ.get('ONE_SAMPLE_CKPT', '')
    if not ckpt_path:
        ckpt_path = os.path.join(_ROOT_DIR, 'data', 'outputs', Path(cfg_file).stem, 'checkpoints', 'latest.pt')

    model, device, ckpt_obj = load_model(cfg_file, ckpt_path)
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, is_train=False), is_train=False)
    if len(dataset) == 0:
        raise RuntimeError('Empty dataset')

    sample = dataset[min(max(int(_custom_args.index), 0), len(dataset) - 1)]
    out_dir = _custom_args.out_dir or os.path.join(_THIS_DIR, 'visual', 'hybrid_postprocess_v3_4')
    tag = _custom_args.tag or Path(cfg_file).stem
    epoch = ckpt_obj.get('epoch', -1) if isinstance(ckpt_obj, dict) else -1
    out_path = os.path.join(out_dir, f'{tag}_idx{_custom_args.index}_epoch{epoch}.png')
    infer_one(model, device, sample, out_path)
    print(out_path)


if __name__ == '__main__':
    main()
