#!/usr/bin/env python3
"""Small supervised post-training runner for controls and displacement gate."""

import argparse
import os
import sys
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-file', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--mode', choices=('full', 'gate'), default='full')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-8)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--gate-loss-weight', type=float, default=1.0)
    parser.add_argument('--save-every', type=int, default=200)
    parser.add_argument('--log-every', type=int, default=20)
    return parser.parse_args()


def set_frozen_modules_eval(core):
    for attr in ('yolo', 'heatmap_detector', 'cnn_proj', 'cnn_proj_p3', 'swin_snake_feature'):
        mod = getattr(core, attr, None)
        if mod is not None and hasattr(mod, 'eval'):
            mod.eval()


def main():
    args = parse_args()
    os.environ['CFG_FILE'] = args.cfg_file
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    sys.argv = [sys.argv[0], '--cfg_file', args.cfg_file]

    from lib.config import cfg
    from lib.datasets import make_data_loader
    import scripts.posttrain_common as pc

    if args.gpu >= 0:
        cfg.gpus = [args.gpu]
    cfg.use_grpo = False
    cfg.use_grpo_kl = False
    cfg.grpo_pure_rl_loss = False
    cfg.train.lr = args.lr
    cfg.train.weight_decay = args.weight_decay
    cfg.train.num_workers = 0
    cfg.train.batch_size = 1
    cfg.freeze_yolo = True

    if args.mode == 'gate':
        cfg.flow_use_disp_gate = True
        cfg.flow_disp_gate_loss_weight = float(args.gate_loss_weight)
        cfg.flow_disp_gate_apply_training_pred = False
        cfg.flow_disp_gate_apply_inference = True
        cfg.flow_disp_gate_detach_input = True

    trainer, device = pc.build_trainer_from_cfg(args.ckpt)
    core = pc.core_network(trainer)

    if args.mode == 'gate':
        pc.freeze_all(trainer.network)
        enabled = pc.enable_params_by_name(trainer.network, ('disp_gate_head',))
        if enabled <= 0:
            raise RuntimeError('mode=gate selected, but no disp_gate_head parameters were found')
        print(f'[*] gate-only trainable parameters: {enabled}')

    params = pc.trainable_parameters(trainer.network)
    if not params:
        raise RuntimeError('No trainable parameters')
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    loader = make_data_loader(cfg, is_train=True)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    running = []
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            trainer.network.train()
            set_frozen_modules_eval(core)
            pc.move_batch_to_device(batch, device)

            output, loss, loss_stats, _image_stats = trainer.network(batch)
            loss = loss.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            step += 1

            row = {'loss': float(loss.detach().cpu())}
            for key, value in loss_stats.items():
                if torch.is_tensor(value):
                    row[key] = float(value.detach().float().mean().cpu())
            running.append(row)

            if step % args.log_every == 0:
                recent = running[-args.log_every:]
                mean_loss = sum(r['loss'] for r in recent) / max(len(recent), 1)
                msg = f'[sup-{args.mode}] step={step}/{args.steps} loss={mean_loss:.6f}'
                for key in ('diff_loss', 'diff_loss_gate', 'disp_gate_mean', 'disp_gate_target_mean'):
                    vals = [r[key] for r in recent if key in r]
                    if vals:
                        msg += f' {key}={sum(vals) / len(vals):.6f}'
                print(msg, flush=True)

            if step % args.save_every == 0:
                pc.save_checkpoint(str(ckpt_dir / f'step{step}.pt'), trainer, optimizer, step)
                pc.save_checkpoint(str(ckpt_dir / 'latest.pt'), trainer, optimizer, step)

            del output, loss, loss_stats, batch

    pc.save_checkpoint(str(ckpt_dir / 'latest.pt'), trainer, optimizer, step)


if __name__ == '__main__':
    main()
