import importlib.util
import random
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parents[1] / "lib/train/grpo_experiment_controls.py"
_SPEC = importlib.util.spec_from_file_location("grpo_experiment_controls", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
group_ids_for_update = _MODULE.group_ids_for_update
network_training_enabled = _MODULE.network_training_enabled


def test_cyclic_schedule_covers_all_groups_and_is_reproducible():
    kwargs = {
        "outer_steps": 5,
        "n_groups": 16,
        "seed": 20260714,
        "schedule": "cyclic",
    }
    first = [group_ids_for_update(step, **kwargs) for step in range(1001, 1017)]
    second = [group_ids_for_update(step, **kwargs) for step in range(1001, 1017)]

    assert first == second
    for outer_step in range(kwargs["outer_steps"]):
        assert {ids[outer_step] for ids in first} == set(range(kwargs["n_groups"]))


def test_random_schedule_default_matches_legacy_randrange_sequence():
    expected_rng = random.Random(17)
    expected = [expected_rng.randrange(16) for _ in range(5)]

    assert group_ids_for_update(1001, 5, 16, 20260712, rng=random.Random(17)) == expected


def test_network_training_is_enabled_by_default():
    assert network_training_enabled()
    assert network_training_enabled(False)
    assert not network_training_enabled(True)


def test_invalid_group_schedule_is_rejected():
    with pytest.raises(ValueError, match="random or cyclic"):
        group_ids_for_update(1, 5, 16, 0, schedule="unknown")
