from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler
import numpy as np
import torch
import math
import torch.distributed as dist


class ImageSizeBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last, min_size=600, max_size=800, size_int=8):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.hmin = min_size
        self.hmax = max_size
        self.wmin = min_size
        self.wmax = max_size
        self.size_int = size_int
        self.hint = (self.hmax-self.hmin)//self.size_int+1
        self.wint = (self.wmax-self.wmin)//self.size_int+1

    def generate_height_width(self):
        hi, wi = np.random.randint(0, self.hint), np.random.randint(0, self.wint)
        h, w = self.hmin + hi * self.size_int, self.wmin + wi * self.size_int
        return h, w

    def __iter__(self):
        batch = []
        h, w = self.generate_height_width()
        for idx in self.sampler:
            batch.append((idx, h, w))
            if len(batch) == self.batch_size:
                h, w = self.generate_height_width()
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class ForegroundBalancedSampler(Sampler):
    """Sample foreground and empty records with replacement."""

    def __init__(
            self, dataset, foreground_fraction=0.5, seed=0,
            num_replicas=None, rank=None):
        if not hasattr(dataset, 'foreground_flags'):
            raise ValueError(
                'Foreground-balanced sampling requires dataset.foreground_flags'
            )
        if len(dataset.foreground_flags) != len(dataset):
            raise ValueError(
                'dataset.foreground_flags must match dataset length: {} != {}'.format(
                    len(dataset.foreground_flags), len(dataset)
                )
            )

        foreground_fraction = float(foreground_fraction)
        if not math.isfinite(foreground_fraction) or not 0.0 < foreground_fraction < 1.0:
            raise ValueError(
                'foreground_fraction must be finite and in (0, 1), got {!r}'.format(
                    foreground_fraction
                )
            )

        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        num_replicas = int(num_replicas)
        rank = int(rank)
        if num_replicas <= 0:
            raise ValueError('num_replicas must be positive, got {}'.format(num_replicas))
        if rank < 0 or rank >= num_replicas:
            raise ValueError(
                'rank must be in [0, {}), got {}'.format(num_replicas, rank)
            )

        self.dataset = dataset
        self.foreground_fraction = foreground_fraction
        self.seed = int(seed)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(dataset) / float(num_replicas)))
        self.total_size = self.num_samples * num_replicas
        self.foreground_indices = [
            index for index, flag in enumerate(dataset.foreground_flags) if bool(flag)
        ]
        self.empty_indices = [
            index for index, flag in enumerate(dataset.foreground_flags) if not bool(flag)
        ]
        if not self.foreground_indices or not self.empty_indices:
            raise ValueError(
                'Foreground-balanced sampling requires non-empty foreground and '
                'empty pools, got {} foreground and {} empty records'.format(
                    len(self.foreground_indices), len(self.empty_indices)
                )
            )

    @staticmethod
    def _sample_pool(pool, count, generator):
        positions = torch.randint(
            len(pool), (count,), generator=generator, dtype=torch.int64
        ).tolist()
        return [pool[position] for position in positions]

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        foreground_count = int(self.total_size * self.foreground_fraction + 0.5)
        foreground_count = min(max(foreground_count, 1), self.total_size - 1)
        empty_count = self.total_size - foreground_count
        global_indices = self._sample_pool(
            self.foreground_indices, foreground_count, generator
        )
        global_indices.extend(
            self._sample_pool(self.empty_indices, empty_count, generator)
        )
        permutation = torch.randperm(
            self.total_size, generator=generator, dtype=torch.int64
        ).tolist()
        global_indices = [global_indices[position] for position in permutation]

        rank_indices = global_indices[self.rank:self.total_size:self.num_replicas]
        if len(rank_indices) != self.num_samples:
            raise RuntimeError(
                'Balanced sampler rank stride produced {} samples; expected {}'.format(
                    len(rank_indices), self.num_samples
                )
            )
        return iter(rank_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


BalancedForegroundSampler = ForegroundBalancedSampler


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """
    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    @property
    def sampler(self):
        return self.batch_sampler.sampler

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations
