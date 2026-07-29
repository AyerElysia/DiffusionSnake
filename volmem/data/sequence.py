import random
from typing import Dict, Iterator, List, Sequence, Tuple


VolumeWindow = Tuple[str, Tuple[int, ...], Tuple[int, ...]]


class VolumeChunkSampler:
    """Sample parallel contiguous chunks from distinct medical volumes."""

    def __init__(
        self,
        records: Sequence[Dict[str, object]],
        chunk_length: int,
        chunks_per_step: int,
        seed: int = 0,
        steps_per_epoch: int = 0,
    ) -> None:
        self.chunk_length = int(chunk_length)
        self.chunks_per_step = int(chunks_per_step)
        self.random = random.Random(int(seed))
        if self.chunk_length < 2:
            raise ValueError("chunk_length must be at least 2")
        if self.chunks_per_step <= 0:
            raise ValueError("chunks_per_step must be positive")

        grouped: Dict[str, List[Tuple[int, int]]] = {}
        for dataset_index, record in enumerate(records):
            volume_id = str(record["case_id"])
            slice_index = int(record["slice_idx"])
            grouped.setdefault(volume_id, []).append((slice_index, dataset_index))

        self.windows_by_volume: Dict[str, List[VolumeWindow]] = {}
        total_windows = 0
        for volume_id, items in grouped.items():
            items.sort()
            windows = []
            for start in range(0, len(items) - self.chunk_length + 1):
                window = items[start:start + self.chunk_length]
                slice_indices = tuple(item[0] for item in window)
                if any(
                    right != left + 1
                    for left, right in zip(slice_indices[:-1], slice_indices[1:])
                ):
                    continue
                dataset_indices = tuple(item[1] for item in window)
                windows.append((volume_id, slice_indices, dataset_indices))
            if windows:
                self.windows_by_volume[volume_id] = windows
                total_windows += len(windows)

        if len(self.windows_by_volume) < self.chunks_per_step:
            raise ValueError(
                "not enough distinct volumes for chunks_per_step={}".format(
                    self.chunks_per_step
                )
            )
        self.volume_ids = sorted(self.windows_by_volume)
        self.steps_per_epoch = (
            int(steps_per_epoch)
            if int(steps_per_epoch) > 0
            else max(total_windows // self.chunks_per_step, 1)
        )

    def __iter__(self) -> Iterator[List[VolumeWindow]]:
        volume_order = list(self.volume_ids)
        self.random.shuffle(volume_order)
        cursor = 0
        for _ in range(self.steps_per_epoch):
            if cursor + self.chunks_per_step > len(volume_order):
                self.random.shuffle(volume_order)
                cursor = 0
            selected_ids = volume_order[cursor:cursor + self.chunks_per_step]
            cursor += self.chunks_per_step
            yield [
                self.random.choice(self.windows_by_volume[volume_id])
                for volume_id in selected_ids
            ]

    def __len__(self) -> int:
        return self.steps_per_epoch
