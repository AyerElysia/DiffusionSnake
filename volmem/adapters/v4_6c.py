from typing import Dict, Tuple

import torch
from torch import nn


class V46cContourAdapter(nn.Module):
    """Isolation boundary around the existing single-slice V4.6c loss wrapper."""

    def __init__(self, slice_loss_wrapper: nn.Module) -> None:
        super().__init__()
        self.slice_loss_wrapper = slice_loss_wrapper

    def forward(
        self,
        batch: Dict[str, object],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _, loss, stats = self.forward_with_output(batch)
        return loss.mean(), stats

    def forward_with_output(
        self,
        batch: Dict[str, object],
    ):
        output, loss, stats, _ = self.slice_loss_wrapper(batch)
        return output, loss.mean(), stats

    def predict(self, batch: Dict[str, object]) -> Dict[str, object]:
        """Run the inherited V4.6c network without its training-loss wrapper."""
        return self.slice_loss_wrapper.net(batch["inp"], batch)
