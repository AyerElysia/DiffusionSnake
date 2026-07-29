from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class SliceMemoryState:
    """One learned memory state written after processing a medical slice."""

    volume_id: str
    slice_index: int
    slice_position: float
    position_unit: str
    key: torch.Tensor
    value: torch.Tensor
    valid_mask: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class VolMemOutput:
    """Typed output of one slice step in VolMemSnake."""

    contour_prediction: torch.Tensor
    memory_conditioned_features: torch.Tensor
    written_state: SliceMemoryState


@dataclass(frozen=True)
class SliceSequenceMeta:
    """Ordering metadata for a slice inside one medical volume."""

    volume_id: str
    slice_index: int
    slice_position: float
    position_unit: str
    sequence_direction: str = "ascending"

    def validate(self) -> None:
        if not self.volume_id:
            raise ValueError("volume_id must be non-empty")
        if self.slice_index < 0:
            raise ValueError("slice_index must be non-negative")
        if self.sequence_direction not in ("ascending", "descending"):
            raise ValueError(
                "sequence_direction must be 'ascending' or 'descending'"
            )
        if self.position_unit not in ("index", "mm"):
            raise ValueError("position_unit must be 'index' or 'mm'")


FeatureShape = Tuple[int, int, int, int]
