from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class SliceSequenceSample:
    """One ordered medical slice and its volume-scoped metadata."""

    volume_id: str
    slice_index: int
    slice_position: float
    position_unit: str
    sequence_direction: str
    payload: Dict[str, Any]

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
        required = {
            "slice_image",
            "mask_target",
            "contour_target",
            "moonvit_layer_18",
            "moonvit_layer_26",
        }
        missing = sorted(required.difference(self.payload))
        if missing:
            raise ValueError("missing VolMem sample fields: {}".format(missing))
