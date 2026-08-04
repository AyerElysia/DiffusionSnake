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
        distance_mode: str = "signed",
        memory_mask_fusion_mode: str = "concat",
        memory_mask_evidence_scale: float = 0.25,
        memory_position_in_values: bool = True,
        memory_global_pool_size: int = 0,
    ) -> None:
        super().__init__()
        self.contour_adapter = contour_adapter
        self.memory_encoder = SliceMemoryEncoder(
            feature_dim=feature_dim,
            memory_dim=memory_dim,
            mask_channels=mask_channels,
            pool_size=memory_pool_size,
            fusion_mode=memory_mask_fusion_mode,
            mask_evidence_scale=memory_mask_evidence_scale,
        )
        self.memflow_controller = install_memflow_dit(
            contour_adapter=contour_adapter,
            memory_dim=memory_dim,
            state_dim=dit_state_dim,
            num_heads=memory_heads,
            distance_scale=distance_scale,
            distance_mode=distance_mode,
        )
        self.memflow_controller.set_value_position_scale(
            1.0 if bool(memory_position_in_values) else 0.0
        )
        self.memory_capacity = int(memory_capacity)
        self.memory_global_pool_size = int(memory_global_pool_size)
        self.mask_channels = int(mask_channels)
        self.memory_pool_size = int(memory_pool_size)

    def new_banks(self, volume_ids: Sequence[str]) -> List[SliceMemoryBank]:
        banks = []
        for volume_id in volume_ids:
            bank = SliceMemoryBank(
                self.memory_capacity, self.memory_global_pool_size
            )
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
        prediction_evidence_fraction: float = 0.0,
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
            "memflow_global_source_slices": loss.detach().new_tensor(
                float(sum(bank.global_count for bank in banks)) / float(len(banks))
            ),
            "volmem_prediction_evidence_fraction": loss.detach().new_tensor(
                float(prediction_evidence_fraction)
            ),
        }

    def _prediction_memory_masks(
        self,
        output: Dict[str, object],
        batch: Dict[str, object],
    ):
        contours = output.get("pred_contours")
        py_ind = output.get("py_ind")
        if not torch.is_tensor(contours) or not torch.is_tensor(py_ind):
            return None
        contours = contours.detach().to(dtype=torch.float32)
        py_ind = py_ind.detach().to(device=contours.device, dtype=torch.long)
        ct_cls = batch.get("ct_cls")
        ct_01 = batch.get("ct_01")
        if not torch.is_tensor(ct_cls) or not torch.is_tensor(ct_01):
            return None
        class_ids = ct_cls[ct_01.bool()].detach().to(
            device=contours.device,
            dtype=torch.long,
        )
        if (
            contours.dim() != 3
            or contours.size(-1) != 2
            or int(contours.size(0)) != int(py_ind.numel())
            or int(contours.size(0)) != int(class_ids.numel())
        ):
            return None

        batch_size = int(batch["inp"].size(0))
        source_h, source_w = batch["ct_hm"].shape[-2:]
        pool = self.memory_pool_size
        yy, xx = torch.meshgrid(
            (torch.arange(pool, device=contours.device, dtype=contours.dtype) + 0.5)
            * (float(source_h) / float(pool)),
            (torch.arange(pool, device=contours.device, dtype=contours.dtype) + 0.5)
            * (float(source_w) / float(pool)),
            indexing="ij",
        )
        evidence = contours.new_zeros(
            (batch_size, self.mask_channels, pool, pool)
        )
        x0 = contours[..., 0].clamp(0, max(float(source_w - 1), 0.0))
        y0 = contours[..., 1].clamp(0, max(float(source_h - 1), 0.0))
        x1 = torch.roll(x0, shifts=-1, dims=1)
        y1 = torch.roll(y0, shifts=-1, dims=1)
        denominator = y1 - y0
        denominator = torch.where(
            denominator.abs() < 1e-6,
            torch.full_like(denominator, 1e-6),
            denominator,
        )
        crosses = (
            (y0[:, :, None, None] > yy)
            != (y1[:, :, None, None] > yy)
        )
        x_at_y = (
            (x1 - x0)[:, :, None, None]
            * (yy - y0[:, :, None, None])
            / denominator[:, :, None, None]
            + x0[:, :, None, None]
        )
        inside_masks = (
            (crosses & (xx < x_at_y)).sum(dim=1) % 2 == 1
        )
        sample_indices = py_ind.detach().cpu().tolist()
        memory_classes = (
            class_ids.detach().cpu().tolist()
            if self.mask_channels > 1
            else [0] * len(sample_indices)
        )
        for inside, sample_index, class_id in zip(
            inside_masks, sample_indices, memory_classes
        ):
            sample_index = int(sample_index)
            class_id = int(class_id)
            if (
                sample_index < 0
                or sample_index >= batch_size
                or class_id < 0
                or class_id >= self.mask_channels
            ):
                continue
            evidence[sample_index, class_id] = torch.maximum(
                evidence[sample_index, class_id],
                inside.to(dtype=evidence.dtype),
            )
        return [item.unsqueeze(0) for item in evidence]

    def forward_step(
        self,
        batch: Dict[str, object],
        metas: Sequence[SliceSequenceMeta],
        memory_masks: Sequence[torch.Tensor],
        banks: Sequence[SliceMemoryBank],
        prediction_evidence_probability: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw_features = self._raw_features(batch)
        if len(raw_features) != len(banks):
            raise ValueError("locate feature count must match memory bank count")
        self.memflow_controller.set_slice_memory(banks, metas)
        output, loss, slice_stats = self.contour_adapter.forward_with_output(batch)
        write_masks = list(memory_masks)
        prediction_fraction = 0.0
        prediction_probability = min(
            max(float(prediction_evidence_probability), 0.0),
            1.0,
        )
        if prediction_probability > 0.0:
            predicted_masks = self._prediction_memory_masks(output, batch)
            if predicted_masks is not None and len(predicted_masks) == len(write_masks):
                selected = 0
                for index, predicted in enumerate(predicted_masks):
                    if float(torch.rand((), device=predicted.device).item()) < prediction_probability:
                        write_masks[index] = predicted.to(
                            device=memory_masks[index].device,
                            dtype=memory_masks[index].dtype,
                        )
                        selected += 1
                prediction_fraction = float(selected) / float(max(len(write_masks), 1))
        self.write_step(raw_features, write_masks, metas, banks)
        stats = dict(slice_stats)
        stats.update(self._stats(loss, banks, prediction_fraction))
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
