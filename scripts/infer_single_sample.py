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
_custom_args, _remaining_argv = _custom_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining_argv

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..'))
_DEFAULT_CFG = os.path.join(_ROOT_DIR, 'configs', 'btcv_diffusion_dit_v3.yaml')

if not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = _DEFAULT_CFG

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

        if i_it_py.numel() == 0:
            pred_py = i_it_py
        else:
            disp = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind)
            pred_py = i_it_py + disp

    dr = float(snake_config.down_ratio)
    orig_img = to_numpy(batch['orig_img'][0]).astype(np.uint8).copy()
    init_np = to_numpy(i_it_py) * dr
    pred_np = to_numpy(pred_py) * dr
    gt_np = to_numpy(i_gt_py) * dr
    gt4_np = to_numpy(i_gt_4py) * dr if i_gt_4py is not None else None

    for poly in gt_np:
        draw_poly(orig_img, poly, (255, 0, 0), thickness=2)
    for poly in init_np:
        draw_poly(orig_img, poly, (0, 255, 255), thickness=1)
    for poly in pred_np:
        draw_poly(orig_img, poly, (0, 0, 255), thickness=2)
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
