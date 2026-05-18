#!/usr/bin/env python3
"""
GT-supervised fine-tuning for DiffusionSnake FM denoiser.

Unlike pseudo-label fine-tuning, this uses ACTUAL GROUND TRUTH displacement as the
training target: x1 = i_gt_py - i_it_py. This gives perfect signal quality and
tests whether the model has any residual capacity to improve with continued training.

Usage:
    CUDA_VISIBLE_DEVICES=5 CFG_FILE=configs/btcv_v3_4_fm_rl_v5b_gpu4.yaml \
    SPLIT=train LR=1e-6 STEPS=200 BATCH_SIZE=4 ANCHOR_LAMBDA=0.1 \
    CKPT=data/outputs/.../checkpoints/latest.pt \
    SAVE_DIR=data/outputs/btcv_fm_gt_finetune_lr1e6 \
        python scripts/finetune_gt_supervised.py
"""

import os, sys, json, datetime, random
import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _THIS_DIR)

_DEFAULT_CFG = os.path.join(_THIS_DIR, 'configs', 'btcv_v3_4_fm_rl_v5b_gpu4.yaml')
if not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = _DEFAULT_CFG

from lib.config import cfg
from lib.datasets.collate_batch import make_collator
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.utils.snake import snake_config, snake_gcn_utils
import torch.nn.functional as F


def load_model(ckpt_path):
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    ckpt_obj = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt_obj.get('state_dict') or ckpt_obj.get('model') or ckpt_obj.get('net') or ckpt_obj
    from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
    sd = remap_legacy_state_dict(sd)
    wrapper = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    info = wrapper.load_state_dict(sd, strict=False)
    print(f'[✔] Loaded {len(sd) - len(info.missing_keys)}/{len(sd)} keys')
    return trainer.network.to(device), device


def save_checkpoint(model, step, save_dir):
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    wrapper = model.module if hasattr(model, 'module') else model
    sd = wrapper.state_dict()
    ckpt = {'state_dict': sd, 'step': step}
    path = os.path.join(save_dir, 'checkpoints', f'step{step:03d}.pt')
    torch.save(ckpt, path)
    torch.save(ckpt, os.path.join(save_dir, 'checkpoints', 'latest.pt'))
    return path


def finetune_step_gt(model, device, batch, optimizer, grad_clip=0.5, iter_steps=3,
                     anchor_weights=None, anchor_lambda=0.0):
    """
    One GT-supervised FM fine-tuning step.

    Uses actual ground-truth displacement: x1 = i_gt_py - i_it_py.
    This mirrors the original FM training objective exactly but on a fresh Adam
    optimizer with optional anchor regularization to prevent catastrophic forgetting.

    The multi-frac sampling (frac ∈ {0, 1/3, 2/3}) is retained to train all
    iteration positions, matching the original training distribution.
    """
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.to(device)

    core = model.net if hasattr(model, 'net') else model
    gcn = core.gcn

    # ----- CNN features (frozen — eval mode) -----
    model.eval()
    with torch.no_grad():
        yolo_out = core.yolo(batch['inp'])
        feat_list = yolo_out[1] if isinstance(yolo_out, (list, tuple)) and len(yolo_out) > 1 else None
        feat_p2 = feat_list[0] if isinstance(feat_list, (list, tuple)) else yolo_out
        cnn_feature = core.cnn_proj(feat_p2)
        if getattr(core, 'use_p3_features', False) and hasattr(core, 'cnn_proj_p3'):
            if isinstance(feat_list, (list, tuple)) and len(feat_list) > 1:
                feat_p3 = feat_list[1]
                feat_p3_up = F.interpolate(feat_p3, size=feat_p2.shape[-2:],
                                            mode='bilinear', align_corners=False)
                cnn_feature = cnn_feature + core.cnn_proj_p3(feat_p3_up)
        cnn_feature = cnn_feature.detach()

    gcn.train()

    # ----- Initial and GT polygons -----
    i_init = batch['i_it_py'].view(-1, batch['i_it_py'].shape[-2], 2).detach().float()
    i_gt   = batch['i_gt_py'].view(-1, batch['i_gt_py'].shape[-2], 2).detach().float()
    n = i_init.size(0)
    if n == 0:
        return None, None

    py_ind = torch.zeros(n, dtype=torch.long, device=device)
    full_disp = (i_gt - i_init).float()  # GT displacement: x1 = i_gt - i_init

    # ----- Multi-frac sampling: match original iterative-refinement training -----
    if iter_steps > 1:
        situations = torch.randint(0, iter_steps, (n,), device=device)
        frac = situations.float().view(n, 1, 1) / float(iter_steps)
    else:
        frac = torch.zeros(n, 1, 1, device=device, dtype=full_disp.dtype)

    # Adjust starting polygon and remaining displacement
    i_init_adj = i_init + full_disp * frac           # [N, 128, 2]
    c_init_adj = snake_gcn_utils.img_poly_to_can_poly(i_init_adj)
    pl_disp_remaining = full_disp * (1.0 - frac)     # GT remaining displacement

    # ----- FM training step -----
    with torch.no_grad():
        distill_ctx = gcn.prepare_sampling_context(cnn_feature, i_init_adj, py_ind)

    contour_scale = distill_ctx['contour_scale'].detach()
    x1 = gcn.normalize_target_disp(pl_disp_remaining, contour_scale).detach()
    t  = gcn.sample_train_t(n, device=device, dtype=x1.dtype)
    x0 = gcn.sample_train_x0(x1).detach()
    x_t      = (1.0 - t) * x0 + t * x1
    v_target = (x1 - x0).detach()

    contour_scale_flat = contour_scale.view(-1).to(dtype=x1.dtype)

    v_pred, l_reg = gcn.predict_velocity(
        cnn_feature,
        i_init_adj,
        c_init_adj,
        distill_ctx['sampled_feat'],
        distill_ctx['detail_feat'],
        py_ind,
        x_t,
        t.view(-1),
        contour_scale=contour_scale_flat,
        x_self_cond=None,
    )

    loss = F.mse_loss(v_pred, v_target, reduction='mean')
    if isinstance(l_reg, torch.Tensor):
        loss = loss + l_reg.mean() * 0.0

    if anchor_weights is not None and anchor_lambda > 0:
        anchor_loss = sum(
            (p - anchor_weights[name].to(p.device)).pow(2).mean()
            for name, p in model.named_parameters()
            if p.requires_grad and name in anchor_weights
        )
        loss = loss + anchor_lambda * anchor_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        max_norm=grad_clip,
    )
    optimizer.step()
    return float(loss.item()), float(grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm))


def main():
    ckpt_rel = os.environ.get('CKPT',
        'data/outputs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolom_gpu35_reusemax/checkpoints/latest.pt')
    lr           = float(os.environ.get('LR', '1e-6'))
    steps        = int(os.environ.get('STEPS', '200'))
    batch_size   = int(os.environ.get('BATCH_SIZE', '4'))
    save_every   = int(os.environ.get('SAVE_EVERY', '20'))
    grad_clip    = float(os.environ.get('GRAD_CLIP', '0.5'))
    iter_steps   = int(os.environ.get('ITER_STEPS', '3'))
    anchor_lambda = float(os.environ.get('ANCHOR_LAMBDA', '0.0'))
    split        = os.environ.get('SPLIT', 'train')
    save_dir_rel = os.environ.get('SAVE_DIR', 'data/outputs/btcv_fm_gt_finetune')

    save_dir = os.path.join(_THIS_DIR, save_dir_rel)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'logs.jsonl')

    ckpt_path = os.path.join(_THIS_DIR, ckpt_rel)
    model, device = load_model(ckpt_path)

    trainable = []
    for name, p in model.named_parameters():
        if 'gcn' in name and ('denoiser' in name or 'flow' in name or 'diffusion' in name):
            p.requires_grad_(True)
            trainable.append(p)
        else:
            p.requires_grad_(False)
    if not trainable:
        print('[!] No denoiser params found; training full gcn')
        for name, p in model.named_parameters():
            if 'gcn' in name:
                p.requires_grad_(True)
                trainable.append(p)
            else:
                p.requires_grad_(False)

    print(f'[*] Trainable params: {sum(p.numel() for p in trainable):,}')
    optimizer = torch.optim.Adam(trainable, lr=lr)

    anchor_weights = None
    if anchor_lambda > 0:
        anchor_weights = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        print(f'[*] Anchor regularization: lambda={anchor_lambda}')

    if split == 'test':
        dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    else:
        dataset = make_dataset(cfg, cfg.train.dataset, make_transforms(cfg, True), True)
    collator = make_collator(cfg)
    n_samples = len(dataset)
    print(f'[*] GT fine-tuning on {n_samples} {split} samples  lr={lr}  steps={steps}  '
          f'iter_steps={iter_steps}  anchor={anchor_lambda}')

    all_idxs = list(range(n_samples))
    step = 0
    losses = []

    while step < steps:
        random.shuffle(all_idxs)
        batch_idxs = all_idxs[:batch_size]

        batch_loss, batch_count = 0.0, 0
        for sample_idx in batch_idxs:
            b = collator([dataset[sample_idx]])
            loss_val, gnorm = finetune_step_gt(
                model, device, b, optimizer, grad_clip,
                iter_steps=iter_steps,
                anchor_weights=anchor_weights,
                anchor_lambda=anchor_lambda,
            )
            if loss_val is not None:
                batch_loss += loss_val
                batch_count += 1

        if batch_count == 0:
            continue

        step += 1
        mean_loss = batch_loss / batch_count
        losses.append(mean_loss)

        log_item = {'step': step, 'loss': mean_loss, 'timestamp': datetime.datetime.now().isoformat()}
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_item) + '\n')

        if step % 10 == 0 or step <= 5:
            print(f'[step {step:4d}/{steps}] loss={mean_loss:.6f}  loss_avg50={np.mean(losses[-50:]):.6f}')

        if step % save_every == 0:
            ckpt_saved = save_checkpoint(model, step, save_dir)
            print(f'[✔] Saved {ckpt_saved}')

    if step % save_every != 0:
        ckpt_saved = save_checkpoint(model, step, save_dir)
        print(f'[✔] Final checkpoint: {ckpt_saved}')

    print(f'\n[*] GT fine-tuning done. {step} steps, final_loss={losses[-1]:.6f}')
    print(f'    Checkpoints in {save_dir}/checkpoints/')


if __name__ == '__main__':
    main()
