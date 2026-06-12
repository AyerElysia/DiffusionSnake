#!/usr/bin/env python3
"""Distill a fixed ensemble-mean contour teacher into a single deterministic pass."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-file', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--out-dir', default='data/outputs/v7_6_ensemble_distill')
    parser.add_argument('--dataset', default='BtcvTrain')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-8)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--teacher-weight', type=float, default=1.0)
    parser.add_argument('--gt-weight', type=float, default=0.25)
    parser.add_argument('--loss-mode', choices=('flow', 'rollout'), default='flow')
    parser.add_argument('--save-every', type=int, default=200)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--max-samples', type=int, default=-1)
    return parser.parse_args()


def load_teacher(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    samples = obj.get('samples', {})
    if not samples:
        raise ValueError(f'No teacher samples found in {path}')
    return samples


def main():
    args = parse_args()
    os.environ['CFG_FILE'] = args.cfg_file
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    sys.argv = [sys.argv[0], '--cfg_file', args.cfg_file]

    from lib.config import cfg
    from lib.datasets.collate_batch import make_collator
    from lib.datasets.make_dataset import make_dataset
    from lib.datasets.transforms import make_transforms
    from lib.networks.diffusion.flow_matching_evolution import FlowMatchingEvolution
    from lib.utils.snake import snake_gcn_utils
    import scripts.posttrain_common as pc

    if args.gpu >= 0:
        cfg.gpus = [args.gpu]
    cfg.train.dataset = args.dataset
    cfg.train.batch_size = 1
    cfg.train.num_workers = 0
    cfg.use_grpo = False
    cfg.grpo_pure_rl_loss = False
    cfg.infer_avg_samples = 1
    cfg.infer_noise_scale = 0.0
    cfg.flow_noise_scale = 0.0
    cfg.freeze_yolo = True

    teacher = load_teacher(args.teacher)
    trainer, device = pc.build_trainer_from_cfg(args.ckpt)
    core = pc.core_network(trainer)
    trainer.network.train()

    # Keep detector/features fixed. The distillation target is for the contour
    # refiner; changing detector init would invalidate the fixed teacher.
    for name, param in trainer.network.named_parameters():
        if '.gcn.' not in name and not name.endswith('gcn'):
            param.requires_grad = False
    for attr in ('yolo', 'heatmap_detector', 'cnn_proj', 'cnn_proj_p3', 'swin_snake_feature'):
        mod = getattr(core, attr, None)
        if mod is not None and hasattr(mod, 'eval'):
            mod.eval()

    params = pc.trainable_parameters(trainer.network)
    if not params:
        raise RuntimeError('No trainable parameters selected for distillation')
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    dataset = make_dataset(cfg, args.dataset, make_transforms(cfg, is_train=False), is_train=False)
    if args.max_samples > 0:
        dataset_len = min(args.max_samples, len(dataset))
        indices = list(range(dataset_len))
        dataset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(dataset),
        collate_fn=make_collator(cfg),
        num_workers=0,
        pin_memory=True,
    )

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    running = []
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            pc.move_batch_to_device(batch, device)
            img_path = pc.first_img_path(batch)
            item = teacher.get(img_path)
            if item is None:
                continue

            teacher_py = torch.tensor(item['teacher_py'], device=device, dtype=batch['inp'].dtype)
            gt_py = torch.tensor(item['gt_py'], device=device, dtype=batch['inp'].dtype)
            if teacher_py.numel() == 0:
                continue

            with torch.no_grad():
                cnn_feature = pc.extract_cnn_feature(core, batch, device)
                cnn_feature = cnn_feature.detach()
                i_it_py, c_it_py, py_ind = pc.build_manual_init(batch, device)

            if i_it_py.size(0) != teacher_py.size(0):
                continue

            scale = FlowMatchingEvolution.compute_contour_scale(i_it_py).to(device=device, dtype=i_it_py.dtype).clamp_min(1.0)
            if args.loss_mode == 'rollout':
                pred_py = pc.rollout_final_contour(core, cnn_feature, i_it_py, c_it_py, py_ind, batch=batch)
                teacher_loss = F.smooth_l1_loss(pred_py / scale, teacher_py / scale)
                gt_loss = F.smooth_l1_loss(pred_py / scale, gt_py / scale)
            else:
                gcn = core.gcn
                teacher_disp = teacher_py - i_it_py
                gt_disp = gt_py - i_it_py
                x1_teacher = gcn.normalize_target_disp(teacher_disp, scale)
                x1_gt = gcn.normalize_target_disp(gt_disp, scale)
                t = gcn.sample_train_t(i_it_py.size(0), device=device, dtype=i_it_py.dtype)
                x0 = gcn.sample_train_x0(x1_teacher)
                x_t_teacher = (1.0 - t) * x0 + t * x1_teacher
                h, w = cnn_feature.size(2), cnn_feature.size(3)
                sampled_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_py, py_ind, h, w)
                detail_feat = gcn.sample_detail_features(
                    cnn_feature,
                    i_it_py,
                    py_ind,
                    h,
                    w,
                    sampled_feat=sampled_feat,
                    contour_scale=scale,
                )
                locate_context = None
                if getattr(gcn, '_locate_token_enabled', False):
                    locate_context = gcn.build_locate_token_context(batch, i_it_py, py_ind, contour_scale=scale)
                v_pred, reg_loss = gcn.predict_velocity(
                    cnn_feature,
                    i_it_py,
                    c_it_py,
                    sampled_feat,
                    detail_feat,
                    py_ind,
                    x_t_teacher,
                    t.view(-1),
                    contour_scale=scale.view(-1),
                    locate_context=locate_context,
                )
                teacher_loss = F.mse_loss(v_pred, x1_teacher - x0)
                gt_loss = F.mse_loss(v_pred, x1_gt - x0)
                teacher_loss = teacher_loss + reg_loss
            loss = args.teacher_weight * teacher_loss + args.gt_weight * gt_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            step += 1
            running.append((float(loss.detach().cpu()), float(teacher_loss.detach().cpu()), float(gt_loss.detach().cpu())))
            if step % args.log_every == 0:
                vals = torch.tensor(running[-args.log_every:])
                print(
                    f'[distill] step={step}/{args.steps} '
                    f'loss={vals[:, 0].mean():.6f} teacher={vals[:, 1].mean():.6f} gt={vals[:, 2].mean():.6f}',
                    flush=True,
                )
            if step % args.save_every == 0:
                pc.save_checkpoint(str(ckpt_dir / f'step{step}.pt'), trainer, optimizer, step)
                pc.save_checkpoint(str(ckpt_dir / 'latest.pt'), trainer, optimizer, step)

    pc.save_checkpoint(str(ckpt_dir / 'latest.pt'), trainer, optimizer, step)


if __name__ == '__main__':
    main()
