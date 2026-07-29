from typing import Dict, Sequence, Tuple

import torch
from torch import nn


class VolMemPretrainWrapper(nn.Module):
    """Aggregate the existing single-slice loss over ordered volume chunks."""

    def __init__(self, volmem_model: nn.Module, slice_loss_wrapper: nn.Module) -> None:
        super().__init__()
        self.volmem_model = volmem_model
        self.slice_loss_wrapper = slice_loss_wrapper

    def forward(
        self,
        sequence_batches,
        metas,
        memory_masks,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        outputs = self.volmem_model.forward_sequence(
            sequence_batches,
            metas,
            memory_masks,
        )
        losses = []
        diff_losses = []
        for output, batch in zip(outputs, sequence_batches):
            _, loss, stats, _ = self.slice_loss_wrapper.compute_from_output(
                output,
                batch,
            )
            losses.append(loss.mean())
            if "diff_loss" in stats:
                diff_losses.append(stats["diff_loss"].mean())
        total = torch.stack(losses).mean()
        stats = {
            "loss": total.detach(),
            "memory_size": outputs[-1]["volmem_memory_size"].detach(),
            "attention_scale": outputs[-1]["volmem_attention_scale"].detach(),
        }
        if diff_losses:
            stats["diff_loss"] = torch.stack(diff_losses).mean().detach()
        return total, stats
