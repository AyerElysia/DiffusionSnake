import os
from pathlib import Path
import sys

import torch

_ROOT = Path(__file__).parents[1]
os.environ.setdefault(
    "CFG_FILE",
    str(_ROOT / "configs/1232_final_v5_perpoint_fmscale_v15_last2_gpu5.yaml"),
)

_TEST_RUNNER_ARGV = sys.argv[:]
try:
    sys.argv[:] = [sys.argv[0]]
    from lib.train.per_point_fm_policy import PerPointFMScalePolicy
    from scripts import eval_v37_full_iou as eval_mod
finally:
    sys.argv[:] = _TEST_RUNNER_ARGV


class _DeterministicFlow:
    def __init__(self):
        self.context_states = []
        self.step_calls = []
        self._use_self_conditioning = False

    def prepare_sampling_context(self, cnn_feature, current, py_ind):
        self.context_states.append(current.detach().clone())
        batch, points = current.shape[:2]
        sampled_feat = current.new_zeros(batch, points, 4)
        return {
            "sampled_feat": sampled_feat,
            "detail_feat": None,
            "contour_scale": current.new_ones(batch, 1, 1),
        }

    def step_with_logprob(
        self,
        cnn_feature,
        current,
        c_cur,
        py_ind,
        x_t,
        step_index,
        total_steps,
        step_mode,
        **kwargs,
    ):
        self.step_calls.append((int(step_index), int(total_steps), step_mode))
        velocity = 0.1 + current * 0.01
        x_next = x_t + velocity / float(total_steps)
        zeros = current.new_zeros(current.size(0))
        return x_next, zeros, x_next, zeros, None

    @staticmethod
    def denormalize_pred_disp(disp, contour_scale):
        return disp

    @staticmethod
    def clamp_pred_disp(disp, current):
        return disp


def _zero_mean_policy():
    return PerPointFMScalePolicy(
        outer_steps=5,
        feature_dim=4,
        feature_embed_dim=4,
        hidden_dim=8,
        max_scale=0.25,
        zero_mean_local=True,
    ).eval()


def test_zero_mean_and_off_share_every_rollout_state(monkeypatch):
    monkeypatch.setattr(
        eval_mod.snake_gcn_utils,
        "img_poly_to_can_poly",
        lambda poly: poly,
    )
    fractions = [0.2, 0.25, 0.3333, 0.5, 1.0]
    initial = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    cnn_feature = torch.zeros(1, 4, 2, 2)
    py_ind = torch.zeros(1, dtype=torch.long)
    off_flow = _DeterministicFlow()
    mean_flow = _DeterministicFlow()

    with torch.no_grad():
        off_disp, off_states = eval_mod._deterministic_unified_rollout(
            off_flow,
            None,
            cnn_feature,
            initial,
            py_ind,
            fractions,
            ode_steps=3,
            active_step_indices=[3, 4],
            return_states=True,
        )
        mean_disp, mean_states = eval_mod._deterministic_unified_rollout(
            mean_flow,
            _zero_mean_policy(),
            cnn_feature,
            initial,
            py_ind,
            fractions,
            ode_steps=3,
            active_step_indices=[3, 4],
            return_states=True,
        )

    torch.testing.assert_close(mean_disp, off_disp, rtol=0, atol=0)
    assert len(mean_states) == len(off_states) == 6
    for mean_state, off_state in zip(mean_states, off_states):
        torch.testing.assert_close(mean_state, off_state, rtol=0, atol=0)
    for mean_state, off_state in zip(mean_flow.context_states, off_flow.context_states):
        torch.testing.assert_close(mean_state, off_state, rtol=0, atol=0)
    assert mean_flow.step_calls == off_flow.step_calls
    assert len(off_flow.step_calls) == 5 * 3
    assert all(total_steps == 3 for _, total_steps, _ in off_flow.step_calls)
    assert all(step_mode == "gaussian" for _, _, step_mode in off_flow.step_calls)


def test_step_records_expose_final_action_input_state_and_full_fm_velocity(monkeypatch):
    monkeypatch.setattr(eval_mod.snake_gcn_utils, "img_poly_to_can_poly", lambda poly: poly)
    fractions = [0.2, 0.25, 0.3333, 0.5, 1.0]
    initial = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    flow = _DeterministicFlow()

    with torch.no_grad():
        _disp, states, records = eval_mod._deterministic_unified_rollout(
            flow,
            None,
            torch.zeros(1, 4, 2, 2),
            initial,
            torch.zeros(1, dtype=torch.long),
            fractions,
            ode_steps=3,
            return_states=True,
            return_step_records=True,
        )

    assert len(records) == 5
    final_record = records[4]
    torch.testing.assert_close(final_record["state"], states[4])
    torch.testing.assert_close(
        states[5] - states[4],
        final_record["fm_velocity"] * fractions[4],
    )
    assert final_record["step_index"] == 4
    assert final_record["fraction"] == 1.0
    assert final_record["sampled_feat"].shape == (1, 4, 4)


def test_unified_rollout_metadata_records_comparison_contract():
    fractions = [0.2, 0.25, 0.3333, 0.5, 1.0]
    off = eval_mod._build_rollout_metadata(
        "off", "off", True, fractions, 10, [3, 4], train_last_n_steps=2
    )
    mean = eval_mod._build_rollout_metadata(
        "mean", "mean", True, fractions, 10, [3, 4], train_last_n_steps=2
    )

    shared_keys = (
        "rollout_backend",
        "deterministic",
        "outer_steps",
        "outer_step_indices",
        "actual_ode_steps",
        "fractions",
        "active_step_indices",
    )
    assert all(off[key] == mean[key] for key in shared_keys)
    assert off["rollout_backend"] == "unified_per_point_5step_deterministic"
    assert off["actual_ode_steps"] == 10
    assert off["active_step_indices"] == [3, 4]
    assert off["scale_applied_step_indices"] == []
    assert mean["scale_applied_step_indices"] == [3, 4]


def test_sample_identity_prefers_image_path():
    identity = eval_mod._sample_identity(
        {"img_path": "/dataset/case_001_image.png"}, index=7
    )

    assert identity == {
        "sample_id": "case_001_image",
        "sample_path": "/dataset/case_001_image.png",
    }
