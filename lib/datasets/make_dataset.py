import importlib
import os

import torch
import torch.utils.data
import torch.utils.data.distributed

from . import samplers
from .collate_batch import make_collator
from .dataset_catalog import DatasetCatalog
from .transforms import make_transforms


torch.multiprocessing.set_sharing_strategy('file_system')


def _dataset_factory(data_source, task):
    if data_source != 'sagittal_2d_fixed' or task != 'snake':
        raise ValueError('the mainline only supports sagittal_2d_fixed/snake')
    return importlib.import_module(
        'lib.datasets.sagittal_2d_fixed.snake'
    ).Dataset

def make_dataset(cfg, dataset_name, transforms, is_train=True):
    args = DatasetCatalog.get(dataset_name)
    data_source = args['id']
    dataset = _dataset_factory(data_source, cfg.task)
    del args['id']
    return dataset(**args)


def make_data_sampler(
        dataset, shuffle, foreground_fraction=None, seed=0):
    if foreground_fraction is not None:
        return samplers.ForegroundBalancedSampler(
            dataset,
            foreground_fraction=foreground_fraction,
            seed=seed,
        )
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_distributed_data_sampler(
        dataset, shuffle, foreground_fraction=None, seed=0):
    if foreground_fraction is not None:
        return samplers.ForegroundBalancedSampler(
            dataset,
            foreground_fraction=foreground_fraction,
            seed=seed,
        )
    # Let DistributedSampler handle shuffling per-epoch via set_epoch.
    return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=bool(shuffle))


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter):
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, max_iter)
    return batch_sampler


def make_data_loader(cfg, is_train=True, is_distributed=False, max_iter=-1):
    train_cfg = getattr(cfg, 'train', None)
    if train_cfg is None:
        train_balance = getattr(cfg, 'balance_foreground_empty', False)
        train_merge = getattr(cfg, 'merge_with_val', False)
    else:
        train_balance = getattr(
            train_cfg,
            'balance_foreground_empty',
            getattr(cfg, 'balance_foreground_empty', False),
        )
        train_merge = getattr(
            train_cfg,
            'merge_with_val',
            getattr(cfg, 'merge_with_val', False),
        )
    balance_foreground_empty = bool(is_train and train_balance)
    merge_with_val = bool(train_merge)
    merge_env = os.environ.get('DIFFUSION_MERGE_TRAIN_VAL', '').strip().lower()
    if merge_env:
        merge_with_val = merge_env in ('1', 'true', 'yes', 'y', 'on')
    if balance_foreground_empty and merge_with_val:
        raise ValueError(
            'balance_foreground_empty is incompatible with merge_with_val=true'
        )

    if is_train:
        batch_size = cfg.train.batch_size
        shuffle = True
        drop_last = bool(getattr(cfg.train, 'drop_last', False))
    else:
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False

    dataset_name = cfg.train.dataset if is_train else cfg.test.dataset
    foreground_fraction = None
    if balance_foreground_empty:
        foreground_fraction = float(
            getattr(
                train_cfg,
                'foreground_fraction',
                getattr(cfg, 'foreground_fraction', 0.5),
            )
        )
    sampler_seed = int(getattr(cfg, 'random_num', 0))

    transforms = make_transforms(cfg, is_train)
    dataset = make_dataset(cfg, dataset_name, transforms, is_train)
    if is_distributed:
        sampler = make_distributed_data_sampler(
            dataset, shuffle, foreground_fraction, sampler_seed
        )
    else:
        sampler = make_data_sampler(
            dataset, shuffle, foreground_fraction, sampler_seed
        )
    batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter)
    num_workers = cfg.train.num_workers
    collator = make_collator(cfg)
    loader_kwargs = {
        'batch_sampler': batch_sampler,
        'num_workers': num_workers,
        'collate_fn': collator,
        'pin_memory': True,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = bool(
            getattr(cfg, 'dataloader_persistent_workers', True)
        )
        prefetch_factor = int(getattr(cfg, 'dataloader_prefetch_factor', 0) or 0)
        if prefetch_factor > 0:
            loader_kwargs['prefetch_factor'] = prefetch_factor

    data_loader = torch.utils.data.DataLoader(
        dataset,
        **loader_kwargs
    )

    return data_loader
