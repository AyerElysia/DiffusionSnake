#!/usr/bin/env python3
import json
import os
import sys
import time
import gc

import torch

_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _THIS_DIR)

if not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = os.path.join(
        _THIS_DIR,
        'configs',
        '1232_final_v8_heatmap_extreme_diffusion_from_detext_gpu3.yaml',
    )

from lib.config import cfg
from lib.datasets import make_data_loader
from lib.networks import make_network
from lib.train.optimizer import make_optimizer
from lib.train.trainers.make_trainer import _wrapper_factory


def resolve_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def move_batch_to_device(batch, device):
    for k in list(batch.keys()):
        if k == 'locate_feat' or str(k).startswith('locate_feat_'):
            continue
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device=device, non_blocking=(device.type == 'cuda'))
    return batch


def load_weights(wrapper):
    ckpt_path = str(getattr(cfg, 'resume_path', '') or '')
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f'[WARN] resume_path not found, training from current init: {ckpt_path}', flush=True)
        return
    obj = torch.load(ckpt_path, map_location='cpu')
    sd = obj.get('state_dict') or obj.get('model') or obj.get('net') or obj
    try:
        from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
        sd = remap_legacy_state_dict(sd)
    except Exception:
        pass
    info = wrapper.load_state_dict(sd, strict=False)
    loaded = len(sd) - len(info.missing_keys)
    print(
        f'[INFO] loaded checkpoint={ckpt_path} loaded_keys={loaded}/{len(sd)} '
        f'missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}',
        flush=True,
    )


def load_training_checkpoint(wrapper, optimizer, device):
    ckpt_path = str(os.environ.get('RESUME_TRAIN_CKPT', '') or getattr(cfg, 'RESUME_TRAIN_CKPT', '') or '')
    if not ckpt_path:
        return 1
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'RESUME_TRAIN_CKPT not found: {ckpt_path}')

    obj = torch.load(ckpt_path, map_location='cpu')
    sd = obj.get('state_dict') or obj.get('model') or obj.get('net') or obj
    try:
        from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
        sd = remap_legacy_state_dict(sd)
    except Exception:
        pass
    info = wrapper.load_state_dict(sd, strict=False)
    loaded = len(sd) - len(info.missing_keys)

    weights_only_env = os.environ.get('RESUME_TRAIN_WEIGHTS_ONLY', '').strip()
    if weights_only_env:
        weights_only = weights_only_env == '1'
    else:
        weights_only = bool(getattr(cfg, 'resume_weights_only', False))
    if (not weights_only) and isinstance(obj, dict) and isinstance(obj.get('optimizer', None), dict):
        optimizer.load_state_dict(obj['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device=device, non_blocking=(device.type == 'cuda'))

    start_step = int(obj.get('step', 0)) + 1 if isinstance(obj, dict) else 1
    print(
        f'[INFO] resumed training checkpoint={ckpt_path} start_step={start_step} '
        f'loaded_keys={loaded}/{len(sd)} missing={len(info.missing_keys)} '
        f'unexpected={len(info.unexpected_keys)} weights_only={weights_only}',
        flush=True,
    )
    return max(start_step, 1)


def save_checkpoint(out_dir, wrapper, optimizer, step):
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        'state_dict': wrapper.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': int(step),
    }
    latest = os.path.join(ckpt_dir, 'latest.pt')
    torch.save(payload, latest)
    save_steps = int(os.environ.get('SAVE_STEPS', '500'))
    if save_steps > 0 and step % save_steps == 0:
        torch.save(payload, os.path.join(ckpt_dir, f'step_{int(step)}.pt'))
    del payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def scalarize_stats(loss_stats):
    row_stats = {}
    if not isinstance(loss_stats, dict):
        return row_stats
    with torch.no_grad():
        for k, v in loss_stats.items():
            if isinstance(v, torch.Tensor):
                row_stats[k] = float(v.detach().item())
            else:
                try:
                    row_stats[k] = float(v)
                except Exception:
                    pass
    return row_stats


def main():
    if 'TRAIN_BATCH_SIZE' in os.environ:
        cfg.train.batch_size = int(os.environ['TRAIN_BATCH_SIZE'])
    out_dir = str(getattr(cfg, 'model_dir', '') or os.path.join(_THIS_DIR, 'data', 'outputs', 'v8_heatmap_minimal'))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'minimal_logs.jsonl')
    run_steps = int(os.environ.get('MAX_STEPS', '2000'))
    log_every = int(os.environ.get('LOG_EVERY', '1'))

    cfg.train.num_workers = int(os.environ.get('NUM_WORKERS', '0'))
    cfg.dataloader_persistent_workers = False
    cfg.dataloader_prefetch_factor = 0

    print(f'[INFO] cfg={os.environ.get("CFG_FILE")} out_dir={out_dir}', flush=True)
    device = resolve_device()
    print(
        f'[INFO] batch_size={cfg.train.batch_size} run_steps={run_steps} '
        f'num_workers={cfg.train.num_workers} device={device}',
        flush=True,
    )

    network = make_network(cfg)
    wrapper = _wrapper_factory(cfg, network).to(device).train()
    load_weights(wrapper)

    optimizer = make_optimizer(cfg, wrapper)
    start_step = load_training_checkpoint(wrapper, optimizer, device)
    end_step = start_step + max(run_steps, 0) - 1
    print(f'[INFO] run_steps={run_steps} start_step={start_step} end_step={end_step}', flush=True)
    loader = make_data_loader(cfg, is_train=True, is_distributed=False)
    data_iter = iter(loader)
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step, end_step + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = move_batch_to_device(batch, device)

        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        output, loss, loss_stats, _ = wrapper(batch)
        loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_value_(wrapper.parameters(), 40.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        dt = time.time() - t0

        row = {
            'step': int(step),
            'loss': float(loss.detach().item()),
            'time_ms': float(dt * 1000.0),
        }
        row['loss_stats'] = scalarize_stats(loss_stats)
        del output, loss, loss_stats, batch

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')

        if step % max(log_every, 1) == 0:
            stats = row.get('loss_stats', {})
            print(
                f"[step {step}] loss={row['loss']:.4f} "
                f"det={stats.get('det_loss', 0.0):.4f} "
                f"mask={stats.get('mask_loss', 0.0):.4f} "
                f"ex={stats.get('ex_loss', 0.0):.4f} "
                f"diff={stats.get('diff_loss', 0.0):.5f} "
                f"jitter={stats.get('ex_box_jitter_count', 0.0):.0f} "
                f"pred_init={stats.get('pred_extreme_init_count', 0.0):.0f} "
                f"gt_init={stats.get('gt_extreme_init_count', 0.0):.0f} "
                f"locfeat={stats.get('locate_feat_residual_absmax', 0.0):.6f} "
                f"locw={stats.get('locate_feat_adapter_last_absmax', 0.0):.6f} "
                f"time={row['time_ms']:.1f}ms",
                flush=True,
            )

        save_steps = int(os.environ.get('SAVE_STEPS', '500'))
        if save_steps > 0 and (step == start_step or step % save_steps == 0):
            save_checkpoint(out_dir, wrapper, optimizer, step)


if __name__ == '__main__':
    main()
