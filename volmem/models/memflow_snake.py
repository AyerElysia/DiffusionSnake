from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn

from .contracts import SliceSequenceMeta
from .memory_bank import SliceMemoryBank
from .memflow_dit import install_memflow_dit
from .slice_memory import SliceMemoryEncoder


class MemFlowDiTSnake(nn.Module):
    """Slice-sequential contour Flow DiT with one direct Memory path."""

    maturity = "prototype"

    def __init__(
        self,
        contour_adapter: nn.Module,
        feature_dim: int = 2304,
        memory_dim: int = 256,
        memory_capacity: int = 4,
        memory_heads: int = 8,
        mask_channels: int = 1,
        memory_pool_size: int = 8,
        dit_state_dim: int = 256,
        distance_scale: float = 4.0,
    ) -> None:
        super().__init__()
        self.contour_adapter = contour_adapter
        self.memory_encoder = SliceMemoryEncoder(
            feature_dim=feature_dim,
            memory_dim=memory_dim,
            mask_channels=mask_channels,
            pool_size=memory_pool_size,
        )
        self.memflow_controller = install_memflow_dit(
            contour_adapter=contour_adapter,
            memory_dim=memory_dim,
            state_dim=dit_state_dim,
            num_heads=memory_heads,
            distance_scale=distance_scale,
        )
        self.memory_capacity = int(memory_capacity)

    def new_banks(self, volume_ids: Sequence[str]) -> List[SliceMemoryBank]:
        banks = []
        for volume_id in volume_ids:
            bank = SliceMemoryBank(self.memory_capacity)
            bank.reset(str(volume_id))
            banks.append(bank)
        return banks

    @staticmethod
    def _raw_features(batch: Dict[str, object]) -> List[torch.Tensor]:
        raw_features = batch.get("locate_feat")
        if not isinstance(raw_features, (list, tuple)):
            raise TypeError("locate_feat must be a per-slice feature list")
        normalized = []
        for feature in raw_features:
            normalized.append(feature.unsqueeze(0) if feature.dim() == 3 else feature)
        return normalized

    def write_step(
        self,
        raw_features: Sequence[torch.Tensor],
        memory_masks: Sequence[torch.Tensor],
        metas: Sequence[SliceSequenceMeta],
        banks: Sequence[SliceMemoryBank],
    ) -> None:
        if not (
            len(raw_features) == len(memory_masks) == len(metas) == len(banks)
        ):
            raise ValueError("parallel MemFlowDiT inputs must have identical lengths")
        for feature, mask, meta, bank in zip(
            raw_features,
            memory_masks,
            metas,
            banks,
        ):
            bank.append(self.memory_encoder(feature, mask, meta))

    def _stats(
        self,
        loss: torch.Tensor,
        banks: Sequence[SliceMemoryBank],
    ) -> Dict[str, torch.Tensor]:
        return {
            "volmem_memory_size": loss.detach().new_tensor(
                float(sum(len(bank) for bank in banks)) / float(len(banks))
            ),
            "volmem_memory_read_delta": loss.detach().new_tensor(
                self.memflow_controller.mean_read_delta()
            ),
            "memflow_active_states": loss.detach().new_tensor(
                float(self.memflow_controller.active_state_count)
            ),
        }

    def forward_step(
        self,
        batch: Dict[str, object],
        metas: Sequence[SliceSequenceMeta],
        memory_masks: Sequence[torch.Tensor],
        banks: Sequence[SliceMemoryBank],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw_features = self._raw_features(batch)
        if len(raw_features) != len(banks):
            raise ValueError("locate feature count must match memory bank count")
        self.memflow_controller.set_slice_memory(banks, metas)
        loss, slice_stats = self.contour_adapter(batch)
        self.write_step(raw_features, memory_masks, metas, banks)
        stats = dict(slice_stats)
        stats.update(self._stats(loss, banks))
        return loss, stats

    def forward_2d(
        self,
        batch: Dict[str, object],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run inherited single-slice V4.6c without Memory read or write."""
        self.memflow_controller.set_slice_memory([], [])
        return self.contour_adapter(batch)

    def predict_step(
        self,
        batch: Dict[str, object],
        metas: Sequence[SliceSequenceMeta],
        banks: Sequence[SliceMemoryBank],
    ):
        raw_features = self._raw_features(batch)
        self.memflow_controller.set_slice_memory(banks, metas)
        output = self.contour_adapter.predict(batch)
        delta = raw_features[0].new_tensor(
            self.memflow_controller.mean_read_delta()
        )
        return output, raw_features, delta

    @staticmethod
    def detach_banks(
        banks: Sequence[SliceMemoryBank],
        keep_recent: int = 1,
    ) -> None:
        for bank in banks:
            bank.detach_states(keep_recent=keep_recent)
