"""
预训练程序
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 diffusion_train.py
"""

import os
import torch
import time
import copy
import sys
import json
import logging
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim.lr_scheduler import MultiStepLR, SequentialLR, LinearLR, CosineAnnealingLR

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Training constants
GRAD_CLIP_VALUE = 40.0
TIME_SCALE_FACTOR = 1000.0
DEFAULT_WARMUP_START_FACTOR = 1e-3
DEFAULT_LR_GAMMA = 0.5

try:
    import wandb
except ImportError:
    wandb = None
    logger.warning("wandb not available, logging will be local only")

try:
    from diffusers.optimization import get_scheduler, SchedulerType
except ImportError:
    get_scheduler = None
    logger.info("diffusers.optimization not available, using PyTorch schedulers")
    class SchedulerType:
        CONSTANT = "constant"
        CONSTANT_WITH_WARMUP = "constant_with_warmup"
        LINEAR = "linear"
        COSINE = "cosine"
        COSINE_WITH_RESTARTS = "cosine_with_restarts"
        POLYNOMIAL = "polynomial"


# IMPORTANT: lib.config 会在 import 时就解析 argv/环境变量并加载 cfg 文件。
# 这里不能强行覆盖用户通过环境变量/命令行指定的 cfg。
# 仅当用户没有设置 CFG_FILE 且没有传 --cfg_file 时，才回退到默认 diffusion 配置。
_THIS_DIR = os.path.dirname(__file__)
_DEFAULT_CFG = os.path.join(_THIS_DIR, 'configs', 'diffusion_snake.yaml')
_argv_lower = [a.lower() for a in sys.argv]
_has_cli_cfg = ('--cfg_file' in _argv_lower) or ('--cfg-file' in _argv_lower)
if (not _has_cli_cfg) and (not os.environ.get('CFG_FILE')):
    os.environ['CFG_FILE'] = os.environ.get('DIFFUSION_CFG_FILE', _DEFAULT_CFG)

from lib.config import cfg, args
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.train.optimizer import make_optimizer
from lib.train.recorder import make_recorder
from lib.datasets import make_data_loader
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.utils.snake import snake_config
from lib.recorder import JsonLogger, NullLogger
from lib.visualizers.diffusion_one_sample import save_affine_visualization


def safe_barrier(is_distributed: bool) -> None:
    """Safely execute distributed barrier with error handling."""
    if is_distributed:
        try:
            dist.barrier()
        except RuntimeError as e:
            logger.warning(f"Distributed barrier failed: {e}")


def main():
    is_distributed = False
    rank = 0
    world_size = 1
    local_rank = 0
    try:
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        rank = int(os.environ.get('RANK', '0'))
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse distributed environment variables: {e}")
        world_size, rank, local_rank = 1, 0, 0

    is_distributed = world_size > 1
    if is_distributed:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
    is_main_process = (rank == 0)

    # Enable Diffusion policy + trainer (遵循配置，其它开关不在此处强行覆盖)
    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True

    try:
        _cfg_file_used = getattr(args, 'cfg_file', '') or os.environ.get('CFG_FILE', '')
        _cfg_stem = Path(str(_cfg_file_used)).stem if _cfg_file_used else 'default'
    except (AttributeError, TypeError) as e:
        logger.warning(f"Failed to get config file stem: {e}")
        _cfg_stem = 'default'

    try:
        train_name = getattr(cfg.train, 'dataset', None)
        test_name = getattr(cfg.test, 'dataset', None)
        logger.info(f"cfg_file={os.environ.get('CFG_FILE','')}")
        logger.info(f"train.dataset={train_name} test.dataset={test_name}")
        if train_name in DatasetCatalog.dataset_attrs:
            a = DatasetCatalog.get(train_name)
            logger.info(f"train.data_root={a.get('data_root')} ann_file={a.get('ann_file')}")
        if test_name in DatasetCatalog.dataset_attrs:
            a = DatasetCatalog.get(test_name)
            logger.info(f"test.data_root={a.get('data_root')} ann_file={a.get('ann_file')}")
    except (AttributeError, KeyError) as e:
        logger.warning(f"Failed to log dataset info: {e}")

    # cfg.train.num_workers = 0  # 已移除：使用 yaml 中配置的 num_workers (默认4)

    # Set optimizer to adamw if not already configured
    if not hasattr(cfg.train, 'optim') or cfg.train.optim is None:
        cfg.train.optim = 'adamw'
        logger.info("Set default optimizer to adamw")

    # Build
    logger.info("Building network")
    network = make_network(cfg)
    logger.info("Building trainer")
    trainer = make_trainer(cfg, network)
    if torch.cuda.is_available():
        trainer.network.cuda()

    if bool(getattr(cfg.train, 'detail_context_only', False)):
        detail_keywords = ('detail_local_proj', 'detail_point_proj')
        total_params = 0
        trainable_params = 0
        for name, param in trainer.network.named_parameters():
            allow_train = any(keyword in name for keyword in detail_keywords)
            param.requires_grad = allow_train
            total_params += int(param.numel())
            if allow_train:
                trainable_params += int(param.numel())
        logger.info(
            "Detail-context-only fine-tune enabled: "
            f"trainable_params={trainable_params} / total_params={total_params}"
        )

    if is_distributed:
        _ddp_find_unused = os.environ.get('DDP_FIND_UNUSED_PARAMETERS', '1')
        _ddp_find_unused = str(_ddp_find_unused).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        trainer.network = DDP(
            trainer.network,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=bool(_ddp_find_unused),
        )
    optimizer = make_optimizer(cfg, trainer.network)
    recorder = make_recorder(cfg)

    if is_main_process:
        logger.info(f"use_grpo={getattr(cfg, 'use_grpo', None)}")

    # ========== 数据加载策略 ==========
    # 默认：正常训练，不合并 train/val（避免数据泄露）
    # 可选：仅在显式开启时合并 train/val，用于上限潜力实验
    train_loader = make_data_loader(cfg, is_train=True, is_distributed=is_distributed)

    merge_train_val = bool(getattr(cfg.train, 'merge_with_val', False))
    merge_env = os.environ.get('DIFFUSION_MERGE_TRAIN_VAL', '').strip().lower()
    if merge_env:
        merge_train_val = merge_env in ('1', 'true', 'yes', 'y', 'on')

    if not merge_train_val:
        logger.info("=" * 60)
        logger.info("Normal training mode: TRAIN split only (no train/val merge)")
        logger.info("=" * 60)
        data_loader = train_loader
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Batches per epoch: {len(data_loader)}")
    else:
        logger.info("=" * 60)
        logger.info("WARNING: Merging train and val datasets (data leakage mode)")
        logger.info("This mode is for capacity probing only, not normal training")
        logger.info("=" * 60)

        original_dataset = cfg.train.dataset
        try:
            if 'Train' in cfg.train.dataset:
                test_dataset_name = cfg.train.dataset.replace('Train', 'Val')
                cfg.train.dataset = test_dataset_name
                test_loader = make_data_loader(cfg, is_train=True, is_distributed=is_distributed)
                logger.info(f"Loaded val dataset for merge: {test_dataset_name}")
            else:
                test_loader = None
                logger.warning("Could not infer val dataset name; fallback to train only")
        finally:
            cfg.train.dataset = original_dataset

        if test_loader is not None:
            from torch.utils.data import ConcatDataset, DataLoader

            train_dataset = train_loader.dataset
            test_dataset = test_loader.dataset
            combined_dataset = ConcatDataset([train_dataset, test_dataset])

            inferred_batch_size = getattr(train_loader, 'batch_size', None)
            if inferred_batch_size is None:
                inferred_batch_size = getattr(getattr(train_loader, 'batch_sampler', None), 'batch_size', None)
            if inferred_batch_size is None:
                inferred_batch_size = int(getattr(cfg.train, 'batch_size', 1))

            inferred_drop_last = getattr(getattr(train_loader, 'batch_sampler', None), 'drop_last', False)
            collate_fn = train_loader.collate_fn if hasattr(train_loader, 'collate_fn') else None

            if is_distributed:
                combined_sampler = torch.utils.data.distributed.DistributedSampler(
                    combined_dataset, shuffle=True
                )
                combined_batch_sampler = torch.utils.data.sampler.BatchSampler(
                    combined_sampler, int(inferred_batch_size), bool(inferred_drop_last)
                )
                data_loader = DataLoader(
                    combined_dataset,
                    batch_sampler=combined_batch_sampler,
                    num_workers=train_loader.num_workers,
                    collate_fn=collate_fn,
                    pin_memory=True,
                )
            else:
                data_loader = DataLoader(
                    combined_dataset,
                    batch_size=int(inferred_batch_size),
                    shuffle=True,
                    num_workers=train_loader.num_workers,
                    collate_fn=collate_fn,
                    pin_memory=True,
                    drop_last=bool(inferred_drop_last),
                )

            logger.info(f"Combined dataset size: {len(combined_dataset)}")
            logger.info(f"  - Train samples: {len(train_dataset)}")
            logger.info(f"  - Val samples: {len(test_dataset)}")
            logger.info(f"  - Batches per epoch: {len(data_loader)}")
        else:
            data_loader = train_loader
            logger.info(f"Using train dataset only: {len(train_loader.dataset)} samples")

    def move_batch_to_cuda(batch):
        for k in list(batch.keys()):
            if k == 'meta':
                continue
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda(non_blocking=True)
        return batch

    # 取一个批次用于可视化，训练循环内重新取 batch
    if is_main_process:
        try:
            viz_batch = move_batch_to_cuda(copy.deepcopy(next(iter(data_loader))))
        except StopIteration:
            viz_batch = None
    else:
        viz_batch = None

    # Helper: run inference and save visualization on affine-input coordinates
    def _visualize_one_sample(tag: str):
        if not is_main_process:
            return
        batch = viz_batch
        trainer.network.eval()
        with torch.no_grad():
            output, _, _, _ = trainer.network(batch)
        save_dir = os.path.join(out_dir, 'visual', 'diffusion_one_sample')
        save_affine_visualization(output=output, batch=batch, tag=str(tag), save_dir=save_dir)

    infer_and_save = _visualize_one_sample

    # 单阶段联合训练配置
    log_interval = int(os.environ.get('ONE_SAMPLE_LOG_INTERVAL', '1'))

    # json logging setup
    out_dir_override = os.environ.get('ONE_SAMPLE_OUT_DIR', '').strip()
    if out_dir_override:
        out_dir = out_dir_override
    else:
        out_dir = os.path.join(os.path.dirname(__file__), 'data', 'outputs', _cfg_stem)
    log_path = os.path.join(out_dir, 'logs.jsonl')
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    save_ep = int(getattr(cfg.train, 'save_ep', 0))
    resume_step = 0
    resume_json_pos = None
    resume_checkpoint = None
    resume_path = os.environ.get('ONE_SAMPLE_RESUME_PATH', '').strip()

    def _safe_load_optimizer_state(optimizer_obj, opt_state_dict) -> bool:
        """Load optimizer state_dict while dropping states whose tensor shapes mismatch current params.

        This is important when resuming from a checkpoint whose model heads changed (e.g. ct_hm classes).
        """
        if not isinstance(opt_state_dict, dict):
            return False
        try:
            # Build mapping from parameter index used in optimizer state_dict to actual parameter.
            param_list = []
            for group in optimizer_obj.param_groups:
                param_list.extend(list(group.get('params', [])))
            if not param_list:
                return False

            state = opt_state_dict.get('state', {})
            if not isinstance(state, dict):
                return False

            filtered_state = {}
            for pid, st in state.items():
                if not isinstance(st, dict):
                    continue
                try:
                    pid_int = int(pid)
                except Exception:
                    continue
                if pid_int < 0 or pid_int >= len(param_list):
                    continue
                p = param_list[pid_int]
                if p is None or (not hasattr(p, 'shape')):
                    continue

                new_st = dict(st)
                ok = True
                for k in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                    v = new_st.get(k, None)
                    if v is None:
                        continue
                    if isinstance(v, torch.Tensor) and tuple(v.shape) != tuple(p.shape):
                        ok = False
                        break
                if ok:
                    filtered_state[pid_int] = new_st

            filtered = dict(opt_state_dict)
            filtered['state'] = filtered_state
            optimizer_obj.load_state_dict(filtered)
            return True
        except Exception:
            return False

    resume_weights_only = bool(getattr(cfg, 'resume_weights_only', False))

    if getattr(cfg, 'resume', False):
        candidate = resume_path if resume_path else os.path.join(ckpt_dir, 'latest.pt')
        if candidate and os.path.exists(candidate):
            try:
                resume_checkpoint = torch.load(candidate, map_location='cpu')
                logger.info(f"Loaded checkpoint from {candidate}")
            except (RuntimeError, FileNotFoundError) as e:
                logger.error(f"Failed to load checkpoint from {candidate}: {e}")
                resume_checkpoint = None

            if isinstance(resume_checkpoint, dict):
                state_dict = resume_checkpoint.get('state_dict')
                if state_dict is None:
                    state_dict = resume_checkpoint.get('model')
                if state_dict is None:
                    state_dict = resume_checkpoint

                try:
                    from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
                    state_dict = remap_legacy_state_dict(state_dict)
                except ImportError:
                    logger.debug("Legacy state dict remapping not available")

                try:
                    model_to_load = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
                    missing, unexpected = model_to_load.load_state_dict(state_dict, strict=False)

                    # Validate critical modules loaded
                    critical_missing = [k for k in missing if any(x in k for x in ['yolo', 'gcn', 'denoiser'])]
                    if critical_missing:
                        logger.error(f"Critical modules missing from checkpoint: {critical_missing[:10]}")
                        logger.warning("Training may start from partially initialized weights!")

                    if missing:
                        logger.warning(f"Missing keys ({len(missing)} total): {missing[:5]}...")
                    if unexpected:
                        logger.warning(f"Unexpected keys ({len(unexpected)} total): {unexpected[:5]}...")
                except RuntimeError as e:
                    logger.error(f"Failed to load model state dict: {e}")

                if (not resume_weights_only) and ('optimizer' in resume_checkpoint):
                    _safe_load_optimizer_state(optimizer, resume_checkpoint['optimizer'])

                if resume_weights_only:
                    resume_step = 0
                    logger.info("Weights-only resume enabled: optimizer, scheduler, and step state were reset.")
                else:
                    resume_step = int(resume_checkpoint.get('step', 0))
                    logger.info(f"Resuming from step {resume_step}")

    def _find_jsonl_truncate_pos(jsonl_path: str, keep_step: int):
        """Return byte position to truncate jsonl so that all remaining lines satisfy step <= keep_step."""
        try:
            keep_step = int(keep_step)
        except (ValueError, TypeError):
            return None
        if keep_step <= 0:
            return None
        if not jsonl_path or (not os.path.exists(jsonl_path)):
            return None
        pos = 0
        truncate_pos = None
        try:
            with open(jsonl_path, 'rb') as f:
                while True:
                    line_start = pos
                    line = f.readline()
                    if not line:
                        break
                    pos = f.tell()
                    try:
                        obj = json.loads(line.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        # Log corrupted lines for debugging
                        logger.debug(f"Skipping corrupted log line at pos {line_start}: {e}")
                        continue
                    try:
                        step_val = int(obj.get('step', -1))
                    except (ValueError, TypeError):
                        step_val = -1
                    if step_val > keep_step:
                        truncate_pos = line_start
                        break
        except IOError as e:
            logger.warning(f"Failed to read log file for truncation: {e}")
            return None
        return truncate_pos

    # If resuming, truncate logs.jsonl so new records overwrite the future part.
    # Default enabled unless ONE_SAMPLE_TRUNCATE_LOG=0
    truncate_log = str(os.environ.get('ONE_SAMPLE_TRUNCATE_LOG', '1')).strip().lower() not in ('0', 'false', 'no', 'off')

    safe_barrier(is_distributed)

    if is_main_process and truncate_log and int(resume_step) > 0:
        resume_json_pos = _find_jsonl_truncate_pos(log_path, int(resume_step))
        if resume_json_pos is not None:
            try:
                with open(log_path, 'rb+') as f:
                    f.truncate(int(resume_json_pos))
                logger.info(f"Truncated log file at position {resume_json_pos}")
            except IOError as e:
                logger.warning(f"Failed to truncate log file: {e}")

    safe_barrier(is_distributed)

    if is_main_process:
        json_logger = JsonLogger(log_path)
    else:
        json_logger = NullLogger()

    wandb_run = None
    if is_main_process and wandb is not None:
        try:
            wandb_project = os.environ.get('WANDB_PROJECT', 'DiffusionSnake')
            wandb_entity = os.environ.get('WANDB_ENTITY', None)
            wandb_name = os.environ.get('WANDB_NAME', None)
            wandb_dir = os.environ.get('WANDB_DIR', out_dir)
            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_name,
                dir=wandb_dir,
                resume='allow',
            )
            logger.info("WandB initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            wandb_run = None

    def _count_trainable_params(module: torch.nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def _set_phase(wrapper, phase: str):
        """phase in {'det', 'diff'}; toggle freeze flags and requires_grad to match."""
        # unwrap DataParallel to get DiffusionNetworkWrapper
        w = wrapper.module if hasattr(wrapper, 'module') else wrapper
        net = w.net
        if phase == 'det':
            # 仅训练YOLO
            w.freeze_yolo = False
            w.freeze_snake = True
            net.freeze_yolo = False
            net.freeze_snake = True
            # requires_grad
            for p in net.yolo.parameters():
                p.requires_grad = True
            for p in net.cnn_proj.parameters():
                p.requires_grad = False
            for p in net.gcn.parameters():
                p.requires_grad = False
            net.yolo.train(); net.cnn_proj.eval(); net.gcn.eval()
        elif phase == 'diff':
            # 仅训练Diffusion
            w.freeze_yolo = True
            w.freeze_snake = False
            net.freeze_yolo = True
            net.freeze_snake = False
            for p in net.yolo.parameters():
                p.requires_grad = False
            for p in net.cnn_proj.parameters():
                p.requires_grad = True
            for p in net.gcn.parameters():
                p.requires_grad = True
            net.yolo.eval(); net.cnn_proj.train(); net.gcn.train()
        else:
            raise ValueError(f"Unknown phase: {phase}")
        # diagnostics
        def _cnt(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        # removed verbose phase print

    def _run_phase(phase_name: str, steps: int):
        if steps <= 0:
            return
        trainer.network.train()
        # 调度器与优化器（step-wise staircase scheduler; optional warmup）
        optimizer_p = make_optimizer(cfg, trainer.network)
        scheduler_p = None
        if str(phase_name).lower() == 'det':
            try:
                ms = list(getattr(cfg.train, 'milestones', (80, 120)))
                scheduler_p = MultiStepLR(optimizer_p, milestones=ms, gamma=getattr(cfg.train, 'gamma', DEFAULT_LR_GAMMA))
            except Exception:
                scheduler_p = None
        else:
            try:
                warmup_steps = int(getattr(cfg.train, 'warmup_steps', 0))
            except Exception:
                warmup_steps = 0
            try:
                gamma = float(getattr(cfg.train, 'gamma', DEFAULT_LR_GAMMA))
            except Exception:
                gamma = 0.5
            try:
                milestones = getattr(cfg.train, 'milestones', None)
                if milestones is None:
                    milestones = [max(1, int(0.6 * steps)), max(1, int(0.85 * steps))]
                milestones = [int(m) for m in list(milestones)]
            except Exception:
                milestones = [max(1, int(0.6 * steps)), max(1, int(0.85 * steps))]

            if warmup_steps > 0:
                warmup = LinearLR(optimizer_p, start_factor=DEFAULT_WARMUP_START_FACTOR, end_factor=1.0, total_iters=int(warmup_steps))
                main_sched = MultiStepLR(optimizer_p, milestones=milestones, gamma=gamma)
                scheduler_p = SequentialLR(optimizer_p, schedulers=[warmup, main_sched], milestones=[int(warmup_steps)])
            else:
                scheduler_p = MultiStepLR(optimizer_p, milestones=milestones, gamma=gamma)
        t_start = time.time()
        loader_iter = iter(data_loader)
        for step in range(1, steps + 1):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(data_loader)
                batch = next(loader_iter)
            batch = move_batch_to_cuda(batch)
            t0 = time.time()
            output, loss, loss_stats, _ = trainer.network(batch)
            loss = loss.mean()
            optimizer_p.zero_grad(set_to_none=True)
            if loss.requires_grad:
                loss.backward()

            # gradient stats
            total_l2 = 0.0
            max_abs = 0.0
            with torch.no_grad():
                for p in trainer.network.parameters():
                    if p.grad is None:
                        continue
                    g = p.grad
                    total_l2 += float(torch.sum(g.detach() * g.detach()).item())
                    max_abs = max(max_abs, float(g.detach().abs().max().item()))
            total_l2 = float(total_l2 ** 0.5)

            torch.nn.utils.clip_grad_value_(trainer.network.parameters(), GRAD_CLIP_VALUE)
            optimizer_p.step()
            if scheduler_p is not None:
                # MultiStepLR 或 diffusers 的 scheduler 都支持每步 step
                scheduler_p.step()

            # periodic inference visualization (Diffusion phase only)
            diff_viz_interval = int(os.environ.get('ONE_SAMPLE_DIFF_VIZ_INTERVAL', '0'))
            if phase_name.lower() == 'diff' and diff_viz_interval > 0 and (step % diff_viz_interval == 0):
                safe_barrier(is_distributed)
                if is_main_process:
                    infer_and_save(f"{phase_name}_step{step}")
                    # switch back to train mode for next iteration
                    trainer.network.train()
                safe_barrier(is_distributed)

            dt = time.time() - t0
            # json logging per step
            lr = optimizer_p.param_groups[0].get('lr', None)
            lr = float(lr) if lr is not None else None

            entry = {
                'phase': str(phase_name),
                'step': int(step),
                'total_steps': int(steps),
                'loss': float(loss.item()),
                'lr': lr,
                'grad_l2': float(total_l2),
                'grad_max': float(max_abs),
                'time_ms': float(dt * TIME_SCALE_FACTOR)
            }
            # include loss stats if dict
            if isinstance(loss_stats, dict):
                safe_stats = {}
                for k, v in loss_stats.items():
                    try:
                        safe_stats[k] = float(getattr(v, 'item', lambda: v)())
                    except Exception:
                        pass
                entry['loss_stats'] = safe_stats
            json_logger.log(entry)

            # explicit cleanup
            del output, loss, loss_stats, batch
            # torch.cuda.empty_cache()
        # no prints

    def _build_scheduler(optimizer_obj, total_steps: int, warmup_steps: int = 0):
        """Step-wise (staircase) LR schedule with optional warmup.

        Defaults are chosen to be safe even if the config does not provide milestones.
        """
        if total_steps <= 0:
            return None
        try:
            gamma = float(getattr(cfg.train, 'gamma', 0.5))
        except Exception:
            gamma = 0.5
        try:
            milestones = getattr(cfg.train, 'milestones', None)
            if milestones is None:
                milestones = [max(1, int(0.6 * total_steps)), max(1, int(0.85 * total_steps))]
            milestones = [int(m) for m in list(milestones)]
        except Exception:
            milestones = [max(1, int(0.6 * total_steps)), max(1, int(0.85 * total_steps))]

        warmup_steps = int(max(0, warmup_steps))
        if warmup_steps > 0:
            warmup = LinearLR(optimizer_obj, start_factor=DEFAULT_WARMUP_START_FACTOR, end_factor=1.0, total_iters=int(warmup_steps))
            main_sched = MultiStepLR(optimizer_obj, milestones=milestones, gamma=gamma)
            return SequentialLR(optimizer_obj, schedulers=[warmup, main_sched], milestones=[int(warmup_steps)])
        return MultiStepLR(optimizer_obj, milestones=milestones, gamma=gamma)

    def _save_checkpoint(step: int, scheduler_obj, total_steps: int = None, epoch: int = None):
        if not is_main_process:
            return
        model_to_save = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
        ckpt = {
            'state_dict': model_to_save.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': int(step),
        }
        if epoch is not None:
            try:
                ckpt['epoch'] = int(epoch)
            except Exception:
                pass
        if scheduler_obj is not None:
            try:
                ckpt['scheduler'] = scheduler_obj.state_dict()
            except Exception:
                pass
        if total_steps is not None:
            ckpt['total_steps'] = int(total_steps)
        if epoch is not None:
            ckpt_path = os.path.join(ckpt_dir, f'epoch_{int(epoch)}.pt')
        else:
            ckpt_path = os.path.join(ckpt_dir, f'step_{int(step)}.pt')
        torch.save(ckpt, ckpt_path)
        torch.save(ckpt, os.path.join(ckpt_dir, 'latest.pt'))


    def _set_yolo_trainable(wrapper, trainable: bool):
        """Enable/disable YOLO head training without affecting diffusion branch."""
        w = wrapper.module if hasattr(wrapper, 'module') else wrapper
        net = getattr(w, 'net', None)
        if net is None:
            return
        try:
            w.freeze_yolo = (not bool(trainable))
        except Exception:
            pass
        try:
            net.freeze_yolo = (not bool(trainable))
        except Exception:
            pass
        try:
            for p in net.yolo.parameters():
                p.requires_grad = bool(trainable)
            if trainable:
                net.yolo.train()
            else:
                net.yolo.eval()
        except Exception:
            pass


    # 统一进入单阶段联合训练（余弦学习率调度器）。支持在指定 epoch 后冻结检测头。
    if True:
        logger.info("Starting epoch-based training loop")
        trainer.network.train()
        steps_per_epoch = 0
        try:
            steps_per_epoch = int(len(data_loader))
        except Exception:
            steps_per_epoch = 0
        if steps_per_epoch <= 0:
            raise RuntimeError('DataLoader is empty or has no __len__; cannot run epoch-based training.')

        try:
            num_epochs = int(getattr(cfg.train, 'epoch', 0))
        except Exception:
            num_epochs = 0
        if num_epochs <= 0:
            raise RuntimeError('cfg.train.epoch must be > 0 for epoch-based training.')

        total_steps = int(num_epochs * steps_per_epoch)
        global_step = int(resume_step)
        start_epoch = int(global_step // steps_per_epoch)
        start_step_in_epoch = int(global_step % steps_per_epoch)

        try:
            eta_min = float(getattr(cfg.train, 'eta_min', 0.0))
        except Exception:
            eta_min = 0.0
        try:
            freeze_yolo_after_epoch = int(getattr(cfg.train, 'freeze_yolo_after_epoch', -1))
        except Exception:
            freeze_yolo_after_epoch = -1

        scheduler = None
        if total_steps > 0:
            warmup_steps = int(getattr(cfg.train, 'warmup_steps', 0))
            if warmup_steps > 0:
                # 1. 线性预热阶段：从基准 LR 的 1e-3 开始爬升
                warmup_sched = LinearLR(
                    optimizer, 
                    start_factor=1.0e-3, 
                    end_factor=1.0, 
                    total_iters=warmup_steps
                )
                # 2. 余弦退火主阶段：处理剩余的所有步数
                main_sched = CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, int(total_steps - warmup_steps)),
                    eta_min=float(eta_min)
                )
                # 3. 组合通过 SequentialLR 实现无缝切换
                scheduler = SequentialLR(
                    optimizer, 
                    schedulers=[warmup_sched, main_sched], 
                    milestones=[warmup_steps],
                    last_epoch=int(global_step - 1)
                )
            else:
                # 若无预热步数，则维持纯余弦退火
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, int(total_steps)),
                    eta_min=float(eta_min),
                    last_epoch=int(global_step - 1),
                )
            if (not resume_weights_only) and isinstance(resume_checkpoint, dict) and 'scheduler' in resume_checkpoint:
                try:
                    scheduler.load_state_dict(resume_checkpoint['scheduler'])
                except Exception:
                    pass

        for epoch in range(start_epoch, num_epochs):
            # Freeze detection head after a chosen epoch boundary.
            if freeze_yolo_after_epoch >= 0 and epoch >= freeze_yolo_after_epoch:
                _set_yolo_trainable(trainer.network, False)
            else:
                _set_yolo_trainable(trainer.network, True)

            # distributed sampler epoch
            try:
                if hasattr(getattr(data_loader, 'batch_sampler', None), 'sampler') and hasattr(data_loader.batch_sampler.sampler, 'set_epoch'):
                    data_loader.batch_sampler.sampler.set_epoch(epoch)
            except Exception:
                pass

            # skip batches if resuming mid-epoch
            data_iter = iter(data_loader)
            if epoch == start_epoch and start_step_in_epoch > 0:
                for _ in range(start_step_in_epoch):
                    try:
                        next(data_iter)
                    except StopIteration:
                        data_iter = iter(data_loader)
                        next(data_iter)

            for step_in_epoch in range(start_step_in_epoch if epoch == start_epoch else 0, steps_per_epoch):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    batch = next(data_iter)
                batch = move_batch_to_cuda(batch)
                t0 = time.time()

                output, loss, loss_stats, _ = trainer.network(batch)
                loss = loss.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                total_l2 = 0.0
                max_abs = 0.0
                with torch.no_grad():
                    for p in trainer.network.parameters():
                        if p.grad is None:
                            continue
                        g = p.grad
                        total_l2 += float(torch.sum(g.detach() * g.detach()).item())
                        max_abs = max(max_abs, float(g.detach().abs().max().item()))
                total_l2 = float(total_l2 ** 0.5)

                torch.nn.utils.clip_grad_value_(trainer.network.parameters(), GRAD_CLIP_VALUE)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                global_step += 1
                dt = time.time() - t0

                # periodic inference visualization every N steps
                viz_interval = int(os.environ.get('ONE_SAMPLE_VIZ_INTERVAL', '0'))
                do_viz = (viz_interval > 0 and (global_step % viz_interval == 0))
                if do_viz:
                    safe_barrier(is_distributed)
                    if is_main_process:
                        if viz_batch is not None:
                            infer_and_save(f"one_stage_step{global_step}")
                            trainer.network.train()
                    safe_barrier(is_distributed)

                # json logging per step
                lr = optimizer.param_groups[0].get('lr', None)
                lr = float(lr) if lr is not None else None

                entry = {
                    'phase': 'one_stage',
                    'epoch': int(epoch),
                    'step_in_epoch': int(step_in_epoch + 1),
                    'steps_per_epoch': int(steps_per_epoch),
                    'step': int(global_step),
                    'total_steps': int(total_steps),
                    'loss': float(loss.item()),
                    'lr': lr,
                    'grad_l2': float(total_l2),
                    'grad_max': float(max_abs),
                    'time_ms': float(dt * TIME_SCALE_FACTOR)
                }
                if isinstance(loss_stats, dict):
                    safe_stats = {}
                    for k, v in loss_stats.items():
                        try:
                            safe_stats[k] = float(getattr(v, 'item', lambda: v)())
                        except Exception:
                            pass
                    entry['loss_stats'] = safe_stats
                json_logger.log(entry)

                if wandb_run is not None:
                    try:
                        wandb_payload = dict(entry)
                        loss_stats_dict = wandb_payload.pop('loss_stats', None)
                        if isinstance(loss_stats_dict, dict):
                            for k, v in loss_stats_dict.items():
                                wandb_payload[f"loss_stats/{k}"] = v
                        wandb.log(wandb_payload, step=int(global_step))
                    except Exception:
                        pass

                del output, loss, loss_stats, batch

            # epoch-based checkpoint saving
            do_save_epoch = (save_ep > 0 and ((epoch + 1) % save_ep == 0)) or (epoch == (num_epochs - 1))
            if do_save_epoch:
                safe_barrier(is_distributed)
                if is_main_process:
                    _save_checkpoint(global_step, scheduler, total_steps=total_steps, epoch=(epoch + 1))
                safe_barrier(is_distributed)

            start_step_in_epoch = 0

    # Final inference visualization on affine-input coords
    if is_main_process:
        infer_and_save('final')

    # close logger
    if hasattr(json_logger, 'close'):
        json_logger.close()

    if is_main_process and wandb_run is not None:
        try:
            wandb.finish()
            logger.info("WandB run finished")
        except Exception as e:
            logger.warning(f"Failed to finish WandB run: {e}")

    safe_barrier(is_distributed)

    if is_distributed:
        try:
            dist.destroy_process_group()
        except Exception as e:
            logger.warning(f"Failed to destroy process group: {e}")


if __name__ == '__main__':
    main()
