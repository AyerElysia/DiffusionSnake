from collections import deque
from typing import Deque, Iterable, List, Optional

from .contracts import SliceMemoryState


class SliceMemoryBank:
    """Runtime memory scoped to exactly one medical volume.

    This object stores learned states; it is not a previous-contour heuristic.
    A volume switch always clears the bank to prevent patient leakage.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._volume_id: Optional[str] = None
        self._states: Deque[SliceMemoryState] = deque(maxlen=self.capacity)

    @property
    def volume_id(self) -> Optional[str]:
        return self._volume_id

    def __len__(self) -> int:
        return len(self._states)

    def reset(self, volume_id: Optional[str] = None) -> None:
        self._states.clear()
        self._volume_id = volume_id

    def append(self, state: SliceMemoryState) -> None:
        if self._volume_id is None:
            self._volume_id = state.volume_id
        if state.volume_id != self._volume_id:
            raise ValueError(
                "SliceMemoryBank cannot mix states from different volumes; "
                "call reset(new_volume_id) first"
            )
        if self._states:
            previous = self._states[-1]
            if state.slice_index == previous.slice_index:
                raise ValueError("duplicate slice_index in SliceMemoryBank")
        self._states.append(state)

    def states(self) -> List[SliceMemoryState]:
        return list(self._states)

    def extend(self, states: Iterable[SliceMemoryState]) -> None:
        for state in states:
            self.append(state)

    def detach_states(self, keep_recent: int = 0) -> None:
        """Truncate old autograd history while keeping recent states trainable."""
        keep_recent = max(int(keep_recent), 0)
        states = list(self._states)
        detach_before = max(len(states) - keep_recent, 0)
        detached = []
        for index, state in enumerate(states):
            should_detach = index < detach_before
            detached.append(SliceMemoryState(
                volume_id=state.volume_id,
                slice_index=state.slice_index,
                slice_position=state.slice_position,
                position_unit=state.position_unit,
                key=state.key.detach() if should_detach else state.key,
                value=state.value.detach() if should_detach else state.value,
                valid_mask=(
                    state.valid_mask.detach()
                    if should_detach and state.valid_mask is not None
                    else state.valid_mask
                ),
            ))
        self._states.clear()
        self._states.extend(detached)
