from collections import deque
import random
from typing import Deque, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .contracts import SliceMemoryState, SliceSequenceMeta


def select_memory_states(
    states: Sequence[SliceMemoryState],
    target: SliceSequenceMeta,
    capacity: int,
    policy: str = "causal-nearest",
    seed: int = 0,
    stride: float = 1.0,
    target_state: Optional[SliceMemoryState] = None,
) -> List[SliceMemoryState]:
    """Select a small, volume-local memory set for one target slice.

    Selection is deliberately parameter-free.  It lets evaluation separate the
    value of temporal breadth from learned Memory quality without introducing a
    second router whose own training could confound the result.
    """
    target.validate()
    capacity = int(capacity)
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    normalized_policy = str(policy).strip().lower()
    if normalized_policy not in {
        "causal-nearest",
        "causal-strided",
        "causal-recent-key-similar",
        "causal-all",
        "bidirectional-nearest",
        "shuffled",
    }:
        raise ValueError("unsupported memory selection policy: {}".format(policy))
    stride = float(stride)
    if stride <= 0.0:
        raise ValueError("memory selection stride must be positive")

    candidates = []
    for state in states:
        if state.volume_id != target.volume_id:
            raise ValueError("memory selection cannot mix volumes")
        if state.position_unit != target.position_unit:
            raise ValueError("memory selection position units do not match")
        if state.slice_index == target.slice_index:
            continue
        candidates.append(state)

    if normalized_policy in {
        "causal-nearest",
        "causal-strided",
        "causal-recent-key-similar",
        "causal-all",
    }:
        candidates = [
            state
            for state in candidates
            if state.slice_position < target.slice_position
        ]
        candidates.sort(
            key=lambda state: (
                target.slice_position - state.slice_position,
                abs(target.slice_index - state.slice_index),
            )
        )
        if normalized_policy == "causal-all":
            # The compact bank must see states in chronological order so only
            # old states are summarized and the local deque remains recent.
            candidates.sort(
                key=lambda state: (state.slice_position, state.slice_index)
            )
            return list(candidates[-capacity:])
        if normalized_policy == "causal-recent-key-similar":
            if target_state is None:
                raise ValueError(
                    "causal-recent-key-similar requires target_state"
                )
            if (
                target_state.volume_id != target.volume_id
                or target_state.slice_index != target.slice_index
                or target_state.position_unit != target.position_unit
            ):
                raise ValueError("target_state does not match target metadata")
            if not candidates:
                return []

            # Always preserve the immediate predecessor.  Fill the remaining
            # fixed budget with the most similar historical Memory keys.  The
            # key is derived primarily from the frozen MoonViT feature map and
            # already has the exact spatial pooling used by Memory attention,
            # so this adds neither parameters nor a second learned router.
            recent = candidates[0]
            if capacity == 1 or len(candidates) == 1:
                return [recent]
            remaining = candidates[1:]
            target_descriptor = target_state.key.detach().float().mean(
                dim=(0, 2, 3)
            )
            target_descriptor = F.normalize(
                target_descriptor, dim=0, eps=1e-6
            )
            candidate_descriptors = torch.stack([
                state.key.detach().float().mean(dim=(0, 2, 3))
                for state in remaining
            ])
            candidate_descriptors = F.normalize(
                candidate_descriptors, dim=1, eps=1e-6
            )
            similarities = torch.mv(candidate_descriptors, target_descriptor)
            similarity_values = similarities.cpu().tolist()
            ranked = sorted(
                range(len(remaining)),
                key=lambda index: (
                    -similarity_values[index],
                    target.slice_position - remaining[index].slice_position,
                    abs(target.slice_index - remaining[index].slice_index),
                ),
            )
            selected = [recent]
            selected.extend(
                remaining[index] for index in ranked[: capacity - 1]
            )
            return selected
        if normalized_policy == "causal-strided":
            # Match SAM 2's bounded temporal-stride idea: keep the immediate
            # predecessor, then sample a fixed number of progressively older
            # states without growing the token budget.  `stride` is expressed
            # in the same spatial unit as SliceSequenceMeta.
            remaining = list(candidates)
            selected = []
            for slot in range(min(capacity, len(remaining))):
                desired_distance = 1.0 + float(slot) * stride
                state = min(
                    remaining,
                    key=lambda item: (
                        abs(
                            (target.slice_position - item.slice_position)
                            - desired_distance
                        ),
                        target.slice_position - item.slice_position,
                        abs(target.slice_index - item.slice_index),
                    ),
                )
                selected.append(state)
                remaining.remove(state)
            return selected
    elif normalized_policy == "bidirectional-nearest":
        candidates.sort(
            key=lambda state: (
                abs(target.slice_position - state.slice_position),
                abs(target.slice_index - state.slice_index),
                state.slice_index,
            )
        )
    else:
        rng = random.Random(int(seed) + int(target.slice_index) * 1_000_003)
        rng.shuffle(candidates)

    return list(candidates[:capacity])


class SliceMemoryBank:
    """Runtime memory scoped to exactly one medical volume.

    This object stores learned states; it is not a previous-contour heuristic.
    A volume switch always clears the bank to prevent patient leakage.
    """

    def __init__(self, capacity: int, global_pool_size: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.global_pool_size = int(global_pool_size)
        if self.global_pool_size < 0:
            raise ValueError("global_pool_size must be non-negative")
        self._volume_id: Optional[str] = None
        self._states: Deque[SliceMemoryState] = deque(maxlen=self.capacity)
        self._global_state: Optional[SliceMemoryState] = None
        self._global_count = 0
        self._global_weight: Optional[torch.Tensor] = None

    @property
    def volume_id(self) -> Optional[str]:
        return self._volume_id

    @property
    def global_count(self) -> int:
        return self._global_count

    @property
    def has_global_state(self) -> bool:
        return self._global_state is not None

    def __len__(self) -> int:
        return len(self._states)

    def reset(self, volume_id: Optional[str] = None) -> None:
        self._states.clear()
        self._volume_id = volume_id

        self._global_state = None
        self._global_count = 0
        self._global_weight = None

    def _merge_global(self, state: SliceMemoryState) -> None:
        if self.global_pool_size <= 0:
            return
        if state.is_global:
            raise ValueError("cannot recursively summarize a global state")
        size = (self.global_pool_size, self.global_pool_size)
        key = F.adaptive_avg_pool2d(state.key, size)
        value = F.adaptive_avg_pool2d(state.value, size)
        # Exclude empty slices without a host-side synchronization on every
        # update. Invalid all-zero summaries stay masked until evidence exists.
        presence = (state.value.detach().abs().sum() > 0).to(dtype=key.dtype)
        presence = presence.reshape(1, 1, 1, 1)
        old_count = self._global_count
        new_count = old_count + 1
        if self._global_state is not None:
            total_weight = self._global_weight + presence
            denominator = total_weight.clamp_min(1.0)
            key = (
                self._global_state.key * self._global_weight + key * presence
            ) / denominator
            value = (
                self._global_state.value * self._global_weight + value * presence
            ) / denominator
        else:
            total_weight = presence
            key = key * presence
            value = value * presence
        position = (
            state.slice_position
            if old_count == 0
            else (
                self._global_state.slice_position * old_count
                + state.slice_position
            ) / float(new_count)
        )
        self._global_count = new_count
        self._global_weight = total_weight
        valid = (total_weight > 0).reshape(1, 1, 1).expand(
            1, self.global_pool_size, self.global_pool_size
        )
        self._global_state = SliceMemoryState(
            volume_id=state.volume_id,
            slice_index=state.slice_index,
            slice_position=float(position),
            position_unit=state.position_unit,
            key=key,
            value=value,
            valid_mask=valid,
            is_global=True,
        )

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
        if len(self._states) == self.capacity and self.global_pool_size > 0:
            self._merge_global(self._states[0])
        self._states.append(state)

    def states(self) -> List[SliceMemoryState]:
        states = list(self._states)
        if self._global_state is not None:
            return [self._global_state] + states
        return states

    def extend(self, states: Iterable[SliceMemoryState]) -> None:
        for state in states:
            self.append(state)

    def detach_states(self, keep_recent: int = 0) -> None:
        """Truncate old autograd history while keeping recent states trainable."""
        keep_recent = max(int(keep_recent), 0)
        states = list(self._states)
        detach_before = max(len(states) - keep_recent, 0)
        detached = []
        if self._global_state is not None:
            state = self._global_state
            self._global_state = SliceMemoryState(
                volume_id=state.volume_id,
                slice_index=state.slice_index,
                slice_position=state.slice_position,
                position_unit=state.position_unit,
                key=state.key.detach(),
                value=state.value.detach(),
                valid_mask=(state.valid_mask.detach() if state.valid_mask is not None else None),
                is_global=True,
            )
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
                is_global=state.is_global,
            ))
        self._states.clear()
        self._states.extend(detached)
