from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn

from .contracts import SliceSequenceMeta
from .memory_bank import SliceMemoryBank
from .slice_memory import SliceMemoryAttention, SliceMemoryEncoder


class VolMemSnake(nn.Module):
    """Trainable slice-sequential volume wrapper around a 2D contour adapter."""

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
    ) -> None:
        super().__init__()
        self.contour_adapter = contour_adapter
        self.memory_attention = SliceMemoryAttention(
            feature_dim=feature_dim,
            memory_dim=memory_dim,
            num_heads=memory_heads,
        )
        self.memory_encoder = SliceMemoryEncoder(
            feature_dim=feature_dim,
            memory_dim=memory_dim,
            mask_channels=mask_channels,
            pool_size=memory_pool_size,
        )
        self.memory_capacity = int(memory_capacity)

    def new_banks(self, volume_ids: Sequence[str]) -> List[SliceMemoryBank]:
        banks = []
        for volume_id in volume_ids:
            bank = SliceMemoryBank(self.memory_capacity)
            bank.reset(str(volume_id))
            banks.append(bank)
        return banks

    def condition_step(
        self,
        batch: Dict[str, object],
        metas: Sequence[SliceSequenceMeta],
        banks: Sequence[SliceMemoryBank],
    ) -> Tuple[Dict[str, object], List[torch.Tensor], torch.Tensor]:
        if len(metas) != len(banks):
            raise ValueError("parallel VolMem inputs must have identical lengths")
        raw_features = batch.get("locate_feat")
        if not isinstance(raw_features, (list, tuple)):
            raise TypeError("locate_feat must be a per-slice feature list")
        if len(raw_features) != len(banks):
            raise ValueError("locate feature count must match memory bank count")

        conditioned_features = []
        normalized_raw = []
        read_deltas = []
        for raw_feature, meta, bank in zip(raw_features, metas, banks):
            meta.validate()
            if meta.volume_id != bank.volume_id:
                raise ValueError("memory bank volume does not match slice metadata")
            if raw_feature.dim() == 3:
                raw_feature = raw_feature.unsqueeze(0)
            conditioned = self.memory_attention(raw_feature, bank.states())
            conditioned_features.append(conditioned.squeeze(0))
            normalized_raw.append(raw_feature)
            read_deltas.append((conditioned - raw_feature).abs().mean())

        step_batch = dict(batch)
        step_batch["locate_feat"] = conditioned_features
        read_delta = torch.stack(read_deltas).mean()
        return step_batch, normalized_raw, read_delta

    def write_step(
        self,
        raw_features: Sequence[torch.Tensor],
        memory_masks: Sequence[torch.Tensor],
        metas: Sequence[SliceSequenceMeta],
        banks: Sequence[SliceMemoryBank],
    ) -> None:
        if not (len(raw_features) == len(metas) == len(memory_masks) == len(banks)):
            raise ValueError("parallel VolMem inputs must have identical lengths")
        for raw_feature, mask_evidence, meta, bank in zip(
            raw_features, memory_masks, metas, banks,
        ):
            state = self.memory_encoder(raw_feature, mask_evidence, meta)
            bank.append(state)

    def predict_step(
        self,
        batch: Dict[str, object],
        metas: Sequence[SliceSequenceMeta],
        banks: Sequence[SliceMemoryBank],
    ) -> Tuple[Dict[str, object], List[torch.Tensor], torch.Tensor]:
        step_batch, raw_features, read_delta = self.condition_step(
            batch, metas, banks
        )
        output = self.contour_adapter.predict(step_batch)
        return output, raw_features, read_delta

    def forward_step(
        self,
        batch: Dict[str, object],
        metas: Sequence[SliceSequenceMeta],
        memory_masks: Sequence[torch.Tensor],
        banks: Sequence[SliceMemoryBank],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not (len(metas) == len(memory_masks) == len(banks)):
            raise ValueError("parallel VolMem inputs must have identical lengths")
        step_batch, normalized_raw, read_delta = self.condition_step(
            batch, metas, banks
        )
        loss, slice_stats = self.contour_adapter(step_batch)
        self.write_step(normalized_raw, memory_masks, metas, banks)

        stats = dict(slice_stats)
        stats.update({
            "volmem_memory_size": loss.detach().new_tensor(
                float(sum(len(bank) for bank in banks)) / float(len(banks))
            ),
            "volmem_memory_read_delta": read_delta.detach(),
        })
        return loss, stats

    @staticmethod
    def detach_banks(
        banks: Sequence[SliceMemoryBank],
        keep_recent: int = 1,
    ) -> None:
        for bank in banks:
            bank.detach_states(keep_recent=keep_recent)
