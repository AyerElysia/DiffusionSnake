import os
import time
import json
import copy

import torch
import torch.utils.data

try:
    import torch.distributed as dist
except Exception:
    dist = None

try:
    import wandb
except Exception:
    wandb = None

# IMPORTANT: lib.config 会在 import 时就解析 argv/环境变量并加载默认 cfg 文件。
_THIS_DIR = os.path.dirname(__file__)
_DEFAULT_CFG = os.path.join(_THIS_DIR, 'configs', 'diffusion_snake.yaml')
os.environ['CFG_FILE'] = os.environ.get('CFG_FILE', _DEFAULT_CFG)

from lib.config import cfg
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.datasets.make_dataset import make_data_loader
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.datasets.collate_batch import make_collator
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.networks.YOLOV8.utils.loss import v8DetectionLoss


class JsonlLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, 'a', encoding='utf-8')

    def log(self, obj: dict):
        self.f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def _is_main_process() -> bool:
    if dist is not None and dist.is_available() and dist.is_initialized():
        try:
            return dist.get_rank() == 0
        except Exception:
            return True
    return True


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    for k, v in list(batch.items()):
        if k in ('meta', 'orig_img', 'img_path'):
            continue
        if isinstance(v, (list, tuple)):
            moved = []
            for itm in v:
                if torch.is_tensor(itm):
                    moved.append(itm.to(device, non_blocking=True))
                else:
                    moved.append(itm)
            batch[k] = moved if isinstance(v, list) else tuple(moved)
        elif torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=True)
    return batch


def _set_yolo_trainable(yolo_model: torch.nn.Module, scope: str):
    scope = str(scope).strip().lower()

    for p in yolo_model.parameters():
        p.requires_grad = False

    if scope == 'full':
        for p in yolo_model.parameters():
            p.requires_grad = True
        yolo_model.train()
        return

    if scope != 'head':
        raise ValueError(f"Unsupported YOLO_TRAIN_SCOPE={scope!r}, expected 'head' or 'full'")

    try:
        detect = yolo_model.model[-1]
    except Exception:
        detect = None
    if detect is None:
        raise RuntimeError('Cannot locate YOLO Detect head at yolo.model[-1]')

    for p in detect.parameters():
        p.requires_grad = True

    try:
        for m in yolo_model.model[:-1]:
            m.eval()
    except Exception:
        pass
    detect.train()


def _build_optimizer(params, lr: float):
    opt_name = str(getattr(cfg.train, 'optim', 'adamw')).lower()
    wd = float(getattr(cfg.train, 'weight_decay', 0.0))

    if opt_name in ('adam',):
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if opt_name in ('adamw',):
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if opt_name in ('sgd',):
        momentum = float(getattr(cfg.train, 'momentum', 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def _apply_cfg_overrides():
    disable_aug = os.environ.get('YOLO_DISABLE_AUG', '').strip()
    if disable_aug:
        os.environ['SNAKE_DISABLE_AUG'] = disable_aug

    load_yolo_pretrained = os.environ.get('YOLO_LOAD_PRETRAINED', '').strip()
    if load_yolo_pretrained:
        cfg.load_yolo_pretrained = load_yolo_pretrained.lower() in ('1', 'true', 'yes', 'on')

    yolo_num_classes = os.environ.get('YOLO_NUM_CLASSES', '').strip()
    if yolo_num_classes:
        cfg.yolo_num_classes = int(yolo_num_classes)

    yolo_pretrained = os.environ.get('YOLO_PRETRAINED_WEIGHTS', '').strip()
    if yolo_pretrained:
        cfg.yolo_pretrained = yolo_pretrained

    train_dataset = os.environ.get('YOLO_TRAIN_DATASET', '').strip()
    if train_dataset:
        cfg.train.dataset = train_dataset

    train_data_path = os.environ.get('YOLO_TRAIN_DATA_PATH', '').strip()
    if train_data_path:
        cfg.train.data_path = train_data_path

    test_dataset = os.environ.get('YOLO_VAL_DATASET', '').strip()
    if test_dataset:
        cfg.test.dataset = test_dataset

    test_img_path = os.environ.get('YOLO_VAL_DATA_PATH', '').strip()
    if test_img_path:
        cfg.test.img_path = test_img_path

    epochs = os.environ.get('YOLO_MAX_EPOCHS', '').strip()
    if epochs:
        cfg.train.epoch = int(epochs)

    batch_size = os.environ.get('YOLO_BATCH_SIZE', '').strip()
    if batch_size:
        cfg.train.batch_size = int(batch_size)

    lr = os.environ.get('YOLO_LR', '').strip()
    if lr:
        cfg.train.lr = float(lr)

    num_workers = os.environ.get('YOLO_NUM_WORKERS', '').strip()
    if num_workers:
        cfg.train.num_workers = int(num_workers)

    disable_lr_flip = os.environ.get('YOLO_DISABLE_LR_FLIP', '').strip()
    if disable_lr_flip:
        os.environ['SNAKE_DISABLE_LR_FLIP'] = disable_lr_flip
    elif bool(getattr(cfg, 'disable_lr_flip', False)):
        os.environ['SNAKE_DISABLE_LR_FLIP'] = '1'


def _build_train_loader():
    transforms = make_transforms(cfg, is_train=True)
    dataset = make_dataset(cfg, cfg.train.dataset, transforms, is_train=True)

    subset_indices = None
    single_sample_index = os.environ.get('YOLO_SINGLE_SAMPLE_INDEX', '').strip()
    if single_sample_index:
        idx = int(single_sample_index)
        if idx < 0 or idx >= len(dataset):
            raise IndexError(f'YOLO_SINGLE_SAMPLE_INDEX={idx} out of range for dataset of size {len(dataset)}')
        subset_indices = [idx]

    max_samples = os.environ.get('YOLO_MAX_SAMPLES', '').strip()
    if subset_indices is None and max_samples:
        keep = min(int(max_samples), len(dataset))
        subset_indices = list(range(keep))

    if subset_indices is not None:
        dataset = torch.utils.data.Subset(dataset, subset_indices)

    sampler = torch.utils.data.RandomSampler(dataset)
    batch_sampler = torch.utils.data.BatchSampler(sampler, int(cfg.train.batch_size), False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=int(cfg.train.num_workers),
        collate_fn=make_collator(cfg),
        pin_memory=True,
    )
    return data_loader, dataset, subset_indices


def _save_checkpoint(ckpt_dir: str, trainer, optimizer, step: int, epoch: int = None):
    model_to_save = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    ckpt = {
        'state_dict': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': int(step),
    }
    if epoch is not None:
        ckpt['epoch'] = int(epoch)

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'step_{int(step)}.pt')
    torch.save(ckpt, ckpt_path)
    torch.save(ckpt, os.path.join(ckpt_dir, 'latest.pt'))


def main():
    is_main = _is_main_process()

    _apply_cfg_overrides()

    train_scope = os.environ.get('YOLO_TRAIN_SCOPE', str(getattr(cfg, 'yolo_train_scope', 'head'))).strip().lower()

    out_dir = os.environ.get('YOLO_HEAD_OUT_DIR', os.path.join(_THIS_DIR, 'data', 'outputs', 'yolo_head_only'))
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    log_path = os.path.join(out_dir, 'logs.jsonl')
    os.makedirs(ckpt_dir, exist_ok=True)

    if is_main:
        try:
            print(f"[train_yolo_head_only] cfg_file={os.environ.get('CFG_FILE','')}")
            print(f"[train_yolo_head_only] train.dataset={cfg.train.dataset}")
            if cfg.train.dataset in DatasetCatalog.dataset_attrs:
                a = DatasetCatalog.get(cfg.train.dataset)
                print(f"[train_yolo_head_only] train.data_root={a.get('data_root')} ann_file={a.get('ann_file')}")
            print(f"[train_yolo_head_only] train.data_path={getattr(cfg.train, 'data_path', '')}")
            print(f"[train_yolo_head_only] val.img_path={getattr(cfg.test, 'img_path', '')}")
            print(f"[train_yolo_head_only] train_scope={train_scope}")
            print(f"[train_yolo_head_only] disable_aug={os.environ.get('SNAKE_DISABLE_AUG', '0')}")
            print(f"[train_yolo_head_only] disable_lr_flip={os.environ.get('SNAKE_DISABLE_LR_FLIP', '0')}")
            print(f"[train_yolo_head_only] load_yolo_pretrained={getattr(cfg, 'load_yolo_pretrained', False)}")
            print(f"[train_yolo_head_only] yolo_pretrained={getattr(cfg, 'yolo_pretrained', '')}")
            print(f"[train_yolo_head_only] yolo_num_classes={getattr(cfg, 'yolo_num_classes', 0)}")
            print(f"[train_yolo_head_only] out_dir={out_dir}")
        except Exception:
            pass

    # build network + wrapper trainer (for save/load compatibility)
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)

    wrapper = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    net = getattr(wrapper, 'net', None)
    if net is None or not hasattr(net, 'yolo'):
        raise RuntimeError('Expected trainer.network to have .net.yolo')

    device = next(wrapper.parameters()).device

    _set_yolo_trainable(net.yolo, train_scope)

    # criterion
    try:
        det_crit = net.yolo.init_criterion()
    except Exception:
        det_crit = v8DetectionLoss(net.yolo)

    trainable_params = [p for p in net.yolo.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError('No trainable params found in YOLO model')

    lr = float(getattr(cfg.train, 'lr', 1e-4))
    optimizer = _build_optimizer(trainable_params, lr=lr)

    # resume
    resume_step = 0
    resume_checkpoint = None
    resume_path = os.environ.get('YOLO_HEAD_RESUME_PATH', '').strip()
    if bool(getattr(cfg, 'resume', False)):
        candidate = resume_path if resume_path else os.path.join(ckpt_dir, 'latest.pt')
        if candidate and os.path.exists(candidate):
            try:
                resume_checkpoint = torch.load(candidate, map_location='cpu')
            except Exception:
                resume_checkpoint = None

    weights_only_resume = bool(int(os.environ.get('YOLO_WEIGHTS_ONLY_RESUME', '0')))

    if isinstance(resume_checkpoint, dict):
        state_dict = resume_checkpoint.get('state_dict')
        if state_dict is None:
            state_dict = resume_checkpoint.get('model')
        if state_dict is None:
            state_dict = resume_checkpoint
        try:
            model_to_load = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
            model_to_load.load_state_dict(state_dict, strict=False)
        except Exception:
            pass
        if (not weights_only_resume) and 'optimizer' in resume_checkpoint:
            try:
                optimizer.load_state_dict(resume_checkpoint['optimizer'])
            except Exception:
                pass
        if not weights_only_resume:
            try:
                resume_step = int(resume_checkpoint.get('step', 0))
            except Exception:
                resume_step = 0

    # wandb (separate by default)
    wandb_run = None
    if is_main and wandb is not None:
        try:
            wandb_project = os.environ.get('WANDB_PROJECT', 'DiffusionSnake-YOLO-Head')
            wandb_entity = os.environ.get('WANDB_ENTITY', None)
            wandb_name = os.environ.get('WANDB_NAME', 'yolo_head_only')
            wandb_dir = os.environ.get('WANDB_DIR', out_dir)
            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_name,
                dir=wandb_dir,
                resume='allow',
            )
        except Exception:
            wandb_run = None

    jsonl_logger = JsonlLogger(log_path) if is_main else None

    data_loader, train_dataset, subset_indices = _build_train_loader()
    steps_per_epoch = len(data_loader)
    num_epochs = int(getattr(cfg.train, 'epoch', 1))
    if num_epochs <= 0:
        raise RuntimeError('cfg.train.epoch must be > 0')

    use_amp = bool(int(os.environ.get('YOLO_HEAD_AMP', '1')))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    global_step = int(resume_step)
    start_epoch = int(global_step // max(steps_per_epoch, 1))
    start_step_in_epoch = int(global_step % max(steps_per_epoch, 1))

    log_interval = int(os.environ.get('YOLO_HEAD_LOG_INTERVAL', '20'))
    save_interval = int(os.environ.get('YOLO_HEAD_SAVE_INTERVAL', '500'))
    max_steps = int(os.environ.get('YOLO_MAX_STEPS', '0') or '0')

    if train_scope == 'head':
        wrapper.train()
        net.yolo.train()
        try:
            wrapper.eval()
        except Exception:
            pass
        try:
            net.eval()
        except Exception:
            pass
        try:
            for m in net.yolo.model[:-1]:
                m.eval()
            net.yolo.model[-1].train()
        except Exception:
            net.yolo.train()
    else:
        wrapper.train()
        net.train()
        net.yolo.train()

    if is_main:
        try:
            subset_desc = 'full'
            if subset_indices is not None:
                subset_desc = f'{len(subset_indices)} sample(s)'
            print(f"[train_yolo_head_only] train_samples={len(train_dataset)} ({subset_desc})")
            print(f"[train_yolo_head_only] steps_per_epoch={steps_per_epoch} epochs={num_epochs} max_steps={max_steps}")
        except Exception:
            pass

    for epoch in range(start_epoch, num_epochs):
        for step_in_epoch, batch in enumerate(data_loader):
            if max_steps > 0 and global_step >= max_steps:
                break
            if epoch == start_epoch and step_in_epoch < start_step_in_epoch:
                continue

            t0 = time.time()
            batch = _move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = net.yolo(batch['inp'])
                det_loss, det_items = det_crit(preds, batch)
                det_loss = det_loss.mean()

            scaler.scale(det_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            dt_ms = (time.time() - t0) * 1000.0

            # parse det_items (box, cls, dfl)
            try:
                box_l = float(det_items[0].mean().detach().item())
                cls_l = float(det_items[1].mean().detach().item())
                dfl_l = float(det_items[2].mean().detach().item())
            except Exception:
                box_l, cls_l, dfl_l = None, None, None

            lr_now = float(optimizer.param_groups[0]['lr'])

            if is_main and (global_step % log_interval == 0 or step_in_epoch == steps_per_epoch - 1):
                entry = {
                    'phase': 'yolo_head_only',
                    'epoch': int(epoch),
                    'step_in_epoch': int(step_in_epoch + 1),
                    'steps_per_epoch': int(steps_per_epoch),
                    'step': int(global_step),
                    'loss': float(det_loss.detach().item()),
                    'lr': lr_now,
                    'time_ms': float(dt_ms),
                }
                if box_l is not None:
                    entry['det_box'] = box_l
                if cls_l is not None:
                    entry['det_cls'] = cls_l
                if dfl_l is not None:
                    entry['det_dfl'] = dfl_l

                if jsonl_logger is not None:
                    jsonl_logger.log(entry)

                if wandb_run is not None:
                    try:
                        wandb.log(entry, step=int(global_step))
                    except Exception:
                        pass

            if is_main and (save_interval > 0) and (global_step % save_interval == 0):
                _save_checkpoint(ckpt_dir, trainer, optimizer, step=global_step, epoch=epoch)

            del batch, preds, det_loss

        if is_main:
            _save_checkpoint(ckpt_dir, trainer, optimizer, step=global_step, epoch=epoch)
        if max_steps > 0 and global_step >= max_steps:
            break

    if jsonl_logger is not None:
        try:
            jsonl_logger.close()
        except Exception:
            pass

    if is_main and wandb_run is not None:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()
