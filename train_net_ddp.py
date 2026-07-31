#!/usr/bin/env python3
"""DDP entry point for EnergySnake training.

Example:
    CUDA_VISIBLE_DEVICES=0,5,6 torchrun --standalone --nproc_per_node=3 \
        train_net_ddp.py --cfg_file configs/sagittal_2d_v4_6c_moonvit_train.yaml
"""

import argparse
import os
import random
import traceback

import torch
import torch.distributed as dist


def make_network(cfg):
    from lib.networks.make_network import make_network as create_network

    return create_network(cfg)


def make_optimizer(cfg, network):
    optim_name = str(getattr(cfg.train, 'optim', 'adamw')).lower()
    if optim_name == 'adamw':
        return torch.optim.AdamW(
            network.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
    if optim_name == 'adam':
        return torch.optim.Adam(network.parameters(), lr=cfg.train.lr)
    if optim_name == 'sgd':
        return torch.optim.SGD(
            network.parameters(), lr=cfg.train.lr, momentum=0.9
        )
    raise ValueError(f'Unsupported optimizer: {optim_name}')


def make_lr_scheduler(cfg, optimizer):
    from torch.optim.lr_scheduler import MultiStepLR

    return MultiStepLR(
        optimizer,
        milestones=list(cfg.train.milestones),
        gamma=cfg.train.gamma,
    )


def make_recorder(cfg):
    from lib.train.recorder import Recorder

    return Recorder(cfg)


def make_trainer(cfg, network, local_rank=0, distributed=False):
    from lib.train.trainers import make_trainer as create_trainer

    return create_trainer(
        cfg,
        network,
        distributed=distributed,
        local_rank=local_rank,
    )


def _checkpoint_path(model_dir, target_epoch=None, explicit_path=None):
    import glob

    if explicit_path:
        explicit_path = os.path.expanduser(os.fspath(explicit_path))
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(
                f'Configured resume_path does not exist: {explicit_path}'
            )
        return explicit_path

    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    if target_epoch is not None:
        candidate = os.path.join(checkpoint_dir, f'epoch_{int(target_epoch)}.pt')
        return candidate if os.path.isfile(candidate) else None

    latest = os.path.join(checkpoint_dir, 'latest.pt')
    if os.path.isfile(latest):
        return latest

    candidates = glob.glob(os.path.join(checkpoint_dir, 'epoch_*.pt'))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _normalize_resume_exclude_prefixes(value):
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(',') if item.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    raise TypeError('resume_exclude_prefixes must be a string, list, or tuple')


def load_model(
    network,
    optimizer,
    scheduler,
    model_dir,
    resume=True,
    strict=False,
    target_epoch=None,
    weights_only=False,
    resume_path=None,
    exclude_prefixes=(),
):
    if not resume:
        return 0

    ckpt_path = _checkpoint_path(
        model_dir,
        target_epoch=target_epoch,
        explicit_path=resume_path,
    )
    if ckpt_path is None:
        print('No checkpoint found; starting from scratch', flush=True)
        return 0

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if not isinstance(checkpoint, dict):
        state_dict = checkpoint
        checkpoint = {}
    else:
        state_dict = checkpoint.get(
            'net', checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        )
    if not isinstance(state_dict, dict):
        raise RuntimeError(f'Checkpoint {ckpt_path} does not contain a state dict')

    excluded_keys = []
    if exclude_prefixes:
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if any(str(key).startswith(prefix) for prefix in exclude_prefixes):
                excluded_keys.append(str(key))
            else:
                filtered_state_dict[key] = value
        state_dict = filtered_state_dict

    missing, unexpected = network.load_state_dict(state_dict, strict=strict)
    print(
        f'Loaded checkpoint={ckpt_path} weights_only={weights_only} '
        f'excluded={len(excluded_keys)} missing={len(missing)} unexpected={len(unexpected)}',
        flush=True,
    )
    if weights_only:
        return 0

    optimizer_state = checkpoint.get('optim', checkpoint.get('optimizer'))
    scheduler_state = checkpoint.get('scheduler')
    if optimizer_state is None or scheduler_state is None:
        raise RuntimeError(
            f'Formal resume checkpoint {ckpt_path} is missing optimizer/scheduler state'
        )
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)
    return int(checkpoint.get('epoch', -1)) + 1


def save_model(network, optimizer, scheduler, epoch, model_dir):
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pt')
    temp_path = checkpoint_path + '.tmp'

    model = network.module if hasattr(network, 'module') else network
    torch.save(
        {
            'net': model.state_dict(),
            'optim': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': int(epoch),
        },
        temp_path,
    )
    os.replace(temp_path, checkpoint_path)

    latest_path = os.path.join(checkpoint_dir, 'latest.pt')
    latest_tmp = latest_path + '.tmp'
    try:
        os.unlink(latest_tmp)
    except FileNotFoundError:
        pass
    os.symlink(os.path.basename(checkpoint_path), latest_tmp)
    os.replace(latest_tmp, latest_path)
    print(f'Model saved: {checkpoint_path}', flush=True)


def _shutdown_data_loader(data_loader):
    """Explicitly stop persistent workers on PyTorch 1.11 before rank exit."""
    if data_loader is None:
        return
    iterator = getattr(data_loader, '_iterator', None)
    if iterator is None:
        return
    shutdown = getattr(iterator, '_shutdown_workers', None)
    if shutdown is not None:
        shutdown()
    data_loader._iterator = None


def _configure_cuda(cfg):
    enable_tf32 = bool(getattr(cfg, 'enable_tf32', True))
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
    torch.backends.cudnn.benchmark = bool(getattr(cfg, 'cudnn_benchmark', True))
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high' if enable_tf32 else 'highest')


def _amp_settings(cfg):
    enabled = bool(getattr(cfg, 'use_amp', False))
    name = str(getattr(cfg, 'amp_dtype', 'bfloat16')).strip().lower()
    dtype_by_name = {
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'fp16': torch.float16,
        'float16': torch.float16,
    }
    if not enabled:
        return False, torch.float16, None
    if name not in dtype_by_name:
        raise ValueError(f'Unsupported amp_dtype={name!r}')
    dtype = dtype_by_name[name]
    if dtype == torch.bfloat16 and hasattr(torch.cuda, 'is_bf16_supported'):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('AMP bfloat16 was requested but this CUDA device does not support it')
    scaler = (
        torch.cuda.amp.GradScaler(enabled=True)
        if dtype == torch.float16 else None
    )
    return True, dtype, scaler


def train_traditional(cfg, network, trainer, local_rank, world_size):
    from lib.datasets import make_data_loader

    optimizer = make_optimizer(cfg, trainer.network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)

    target_epoch = getattr(cfg, 'load_epoch', None)
    begin_epoch = load_model(
        network,
        optimizer,
        scheduler,
        cfg.model_dir,
        resume=bool(getattr(cfg, 'resume', False)),
        strict=False,
        target_epoch=target_epoch,
        weights_only=bool(getattr(cfg, 'resume_weights_only', False)),
        resume_path=getattr(cfg, 'resume_path', ''),
        exclude_prefixes=_normalize_resume_exclude_prefixes(
            getattr(cfg, 'resume_exclude_prefixes', ())
        ),
    )

    amp_enabled, amp_dtype, grad_scaler = _amp_settings(cfg)
    trainer.configure_runtime(
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        grad_scaler=grad_scaler,
        gradient_clip=float(getattr(cfg.train, 'gradient_clip', 1.0)),
        gradient_accumulation_steps=int(
            getattr(cfg.train, 'gradient_accumulation_steps', 1) or 1
        ),
        empty_cache_interval=int(getattr(cfg, 'cuda_empty_cache_interval', 0) or 0),
        rank=local_rank,
        is_main_process=(local_rank == 0),
    )

    max_steps = int(getattr(cfg.train, 'max_steps', 0) or 0)
    train_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=(world_size > 1),
        max_iter=max_steps if max_steps > 0 else -1,
    )
    save_ep = int(getattr(cfg.train, 'save_ep', 0) or 0)

    try:
        if local_rank == 0:
            print(
                f'begin_epoch={begin_epoch} epochs={cfg.train.epoch} '
                f'world_size={world_size} per_rank_batch={cfg.train.batch_size} '
                f'grad_accum={getattr(cfg.train, "gradient_accumulation_steps", 1)} '
                f'amp={amp_enabled}:{amp_dtype} steps_per_epoch={len(train_loader)}',
                flush=True,
            )

        for epoch in range(begin_epoch, cfg.train.epoch):
            if local_rank == 0:
                print(f'Epoch {epoch}', flush=True)
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            recorder.epoch = epoch
            trainer.train(epoch, train_loader, optimizer, recorder)
            scheduler.step()

            if local_rank == 0 and save_ep > 0 and (epoch + 1) % save_ep == 0:
                save_model(network, optimizer, scheduler, epoch, cfg.model_dir)
            if world_size > 1:
                dist.barrier()
    finally:
        _shutdown_data_loader(train_loader)
        close = getattr(recorder, 'close', None)
        if close is not None:
            close()

    return network


def main():
    parser = argparse.ArgumentParser(description='EnergySnake DDP training')
    parser.add_argument('--cfg_file', required=True, type=str)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    if local_rank < 0:
        local_rank = 0
    if not torch.cuda.is_available():
        raise RuntimeError('train_net_ddp.py requires CUDA')

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    normal_exit = False

    try:
        from lib.config import cfg

        cfg.merge_from_file(args.cfg_file)
        cfg.merge_from_list(args.opts or [])
        _configure_cuda(cfg)

        seed = int(cfg.random_num) + rank
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if rank == 0:
            print(
                f'Starting EnergySnake DDP: world_size={world_size} '
                f'local_rank={local_rank} config={args.cfg_file}',
                flush=True,
            )

        network = make_network(cfg).to(torch.device('cuda', local_rank))
        trainer = make_trainer(
            cfg,
            network,
            local_rank=local_rank,
            distributed=(world_size > 1),
        )
        train_traditional(cfg, network, trainer, local_rank, world_size)
        normal_exit = True

        if rank == 0:
            print('Training completed', flush=True)
    except Exception:
        if rank == 0:
            traceback.print_exc()
        raise
    finally:
        if dist.is_initialized():
            # Keep NCCL teardown ordered across ranks. PyTorch 1.11 can leave
            # rank processes waiting in communicator destruction when rank 0
            # reaches destroy_process_group first.
            if normal_exit and world_size > 1:
                dist.barrier()
            dist.destroy_process_group()
            if normal_exit:
                # PyTorch 1.11 may leave multiprocessing queue threads alive
                # after all explicit cleanup has completed. At this point the
                # writer, loader and process group are closed, so bypass those
                # stale interpreter-finalization threads on clean completion.
                os._exit(0)


if __name__ == '__main__':
    main()
