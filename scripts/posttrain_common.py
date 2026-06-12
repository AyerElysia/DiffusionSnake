#!/usr/bin/env python3
"""Shared utilities for supervised post-training experiments."""

import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

from lib.config import cfg
from lib.networks import make_network
from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
from lib.train.trainers import make_trainer
from lib.utils.snake import snake_config, snake_gcn_utils


def set_cuda_from_cfg():
    if hasattr(cfg, 'gpus') and cfg.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, cfg.gpus))


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    for key, value in list(batch.items()):
        if key in ('meta', 'orig_img', 'img_path') or key == 'locate_feat' or str(key).startswith('locate_feat_'):
            continue
        if torch.is_tensor(value):
            batch[key] = value.to(device)
        elif isinstance(value, (list, tuple)):
            moved = [item.to(device) if torch.is_tensor(item) else item for item in value]
            batch[key] = moved if isinstance(value, list) else tuple(moved)
    return batch


def first_img_path(batch: Dict) -> str:
    value = batch.get('img_path', '')
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ''
    if torch.is_tensor(value):
        raw = value.detach().cpu().numpy().tolist()
        if isinstance(raw, list) and raw:
            return str(raw[0])
        return str(raw)
    return str(value)


def load_state_dict_flexible(module: torch.nn.Module, ckpt_path: str) -> Tuple[int, int]:
    obj = torch.load(ckpt_path, map_location='cpu')
    sd = obj.get('state_dict') or obj.get('model') or obj.get('net') or obj
    sd = remap_legacy_state_dict(sd)
    info = module.load_state_dict(sd, strict=False)
    loaded = len(sd) - len(info.unexpected_keys)
    print(
        f'[*] Loaded checkpoint {ckpt_path}: '
        f'keys={len(sd)} missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}'
    )
    return loaded, len(sd)


def build_trainer_from_cfg(ckpt_path: str = ''):
    set_cuda_from_cfg()
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    if ckpt_path:
        load_state_dict_flexible(trainer.network, ckpt_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer.network.to(device)
    return trainer, device


def freeze_all(module: torch.nn.Module):
    for param in module.parameters():
        param.requires_grad = False


def enable_params_by_name(module: torch.nn.Module, names: Iterable[str]) -> int:
    enabled = 0
    name_parts = tuple(str(name) for name in names)
    for name, param in module.named_parameters():
        if any(part in name for part in name_parts):
            param.requires_grad = True
            enabled += int(param.numel())
    return enabled


def trainable_parameters(module: torch.nn.Module):
    return [p for p in module.parameters() if p.requires_grad]


def save_checkpoint(path: str, trainer, optimizer, step: int, extra: Dict = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'state_dict': trainer.network.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'step': int(step),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f'[*] saved checkpoint: {path}')


def core_network(trainer):
    wrapped = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    return wrapped.net if hasattr(wrapped, 'net') else wrapped


def extract_cnn_feature(core, batch: Dict, device: torch.device) -> torch.Tensor:
    detector_backend = str(getattr(cfg, 'detector_backend', 'yolo')).strip().lower()
    inp = batch['inp']
    if detector_backend == 'yolo':
        yolo_out = core.yolo(inp)
        feat_list = yolo_out[1] if isinstance(yolo_out, (list, tuple)) and len(yolo_out) > 1 else None
        feat_p2 = feat_list[0] if isinstance(feat_list, (list, tuple)) else yolo_out
        if getattr(core, 'use_swin_snake_feature', False):
            cnn_feature = core.swin_snake_feature(inp)
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
            feat_p3_up = F.interpolate(feat_p3, size=feat_p2.shape[-2:], mode='bilinear', align_corners=False)
            cnn_feature = cnn_feature + core.cnn_proj_p3(feat_p3_up)
    elif detector_backend.startswith('heatmap_') or detector_backend.startswith('convnext') or detector_backend.startswith('moonvit'):
        cnn_feature, _ct_hm, _wh, mask_logits = core.heatmap_detector(inp)
        if mask_logits is not None:
            alpha = float(getattr(cfg, 'heatmap_mask_guidance_alpha', 0.0))
            if alpha > 0.0:
                cnn_feature = cnn_feature * (1.0 + alpha * torch.sigmoid(mask_logits).amax(dim=1, keepdim=True))
    else:
        raise RuntimeError(f'Unsupported detector_backend for post-training: {detector_backend}')

    if hasattr(core, 'apply_locate_feature_injection'):
        cnn_feature, _ = core.apply_locate_feature_injection(cnn_feature, batch)
    if hasattr(core, 'apply_locate_feature_replacement'):
        cnn_feature, _ = core.apply_locate_feature_replacement(cnn_feature, batch)
    return cnn_feature


def build_manual_init(batch: Dict, device: torch.device):
    import scripts.eval_v37_full_iou as eval_mod

    gt_all = batch['i_gt_py']
    i_it_py = eval_mod.build_init_polys(batch, gt_all).to(device)
    c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)
    if gt_all.size(0) == 1:
        py_ind = torch.zeros(i_it_py.size(0), dtype=torch.long, device=device)
    else:
        num_contours = gt_all.size(1)
        py_ind = torch.cat([
            torch.full((num_contours,), i, dtype=torch.long, device=device)
            for i in range(gt_all.size(0))
        ])
    return i_it_py, c_it_py, py_ind


def rollout_final_contour(core, cnn_feature, i_it_py, c_it_py, py_ind, batch=None):
    gcn = core.gcn
    if getattr(gcn, 'use_iterative_refinement', False):
        iter_steps = int(getattr(cfg, 'iterative_num_steps', 3))
        fractions = list(getattr(cfg, 'iterative_fractions', []))
        if not fractions:
            fractions = [1.0 / (iter_steps - i) for i in range(iter_steps)]
        iter_ode_steps = int(getattr(cfg, 'iterative_ode_steps', getattr(cfg, 'iterative_ddim_steps', gcn.ode_steps)))
        disp = gcn.sample_disp_iterative(
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
        disp = gcn.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind, steps=gcn.ode_steps, batch=batch)
    fk = int(getattr(cfg, 'fourier_smooth_k', 0))
    if fk > 0:
        from lib.networks.diffusion.flow_matching_evolution import FlowMatchingEvolution
        disp = FlowMatchingEvolution.fourier_smooth(disp, fk)
    return i_it_py + disp


def feature_scale_from_image_scale(poly: torch.Tensor) -> torch.Tensor:
    return poly / float(snake_config.down_ratio)
