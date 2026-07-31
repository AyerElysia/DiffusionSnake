"""Minimal experiment controls shared by RL-V5 training and tests."""

from __future__ import annotations

import random
from typing import List, Optional


def network_training_enabled(freeze_gcn_backbone: bool = False) -> bool:
    """Keep historical network training unless explicitly frozen."""
    return not bool(freeze_gcn_backbone)


def group_ids_for_update(
    rl_update_step: int,
    outer_steps: int,
    n_groups: int,
    seed: int,
    schedule: str = "random",
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Choose one shared rollout group for each outer step."""
    schedule = str(schedule).strip().lower()
    if schedule not in ("random", "cyclic"):
        raise ValueError(f"group schedule must be random or cyclic, got {schedule!r}")
    if int(outer_steps) < 0:
        raise ValueError(f"outer_steps must be non-negative, got {outer_steps}")
    if int(n_groups) <= 0:
        raise ValueError(f"n_groups must be positive, got {n_groups}")

    if schedule == "random":
        source = random if rng is None else rng
        return [source.randrange(int(n_groups)) for _ in range(int(outer_steps))]

    base = int(seed) + int(rl_update_step) - 1
    return [
        (base + outer_step) % int(n_groups)
        for outer_step in range(int(outer_steps))
    ]
