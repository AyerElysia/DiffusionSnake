import pathlib
import sys
import unittest

import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from volmem.models.contracts import SliceMemoryState, SliceSequenceMeta
from volmem.models.memflow_dit import (
    MemFlowDiTController,
    MemoryCrossAttention,
    RelativeSliceDistanceEncoding,
)
from volmem.models.memory_bank import SliceMemoryBank, select_memory_states
from volmem.models.slice_memory import SliceMemoryEncoder


class IdentityBlock(torch.nn.Module):
    def forward(self, x, context, t_emb):
        return x


class MemFlowDiTTests(unittest.TestCase):
    @staticmethod
    def _state(index):
        return SliceMemoryState(
            volume_id="case_a",
            slice_index=index,
            slice_position=float(index),
            position_unit="index",
            key=torch.full((1, 8, 2, 2), float(index)),
            value=torch.full((1, 8, 2, 2), float(index)),
        )

    def test_empty_memory_is_exact_2d_identity(self):
        adapter = MemoryCrossAttention(dim=8, num_heads=2)
        tokens = torch.randn(3, 5, 8)
        self.assertTrue(torch.equal(adapter(tokens), tokens))

    def test_zero_init_preserves_pretrained_output(self):
        adapter = MemoryCrossAttention(dim=8, num_heads=2)
        adapter.set_slice_memory(
            torch.randn(2, 4, 8),
            torch.randn(2, 4, 8),
            torch.ones(2, 4, dtype=torch.bool),
        )
        adapter.set_contour_indices(torch.tensor([0, 1, 1]))
        tokens = torch.randn(3, 5, 8)
        self.assertTrue(torch.equal(adapter(tokens), tokens))

    def test_memory_changes_tokens_after_output_projection_opens(self):
        adapter = MemoryCrossAttention(dim=8, num_heads=2)
        torch.nn.init.normal_(adapter.output_proj.weight, std=0.05)
        adapter.set_slice_memory(
            torch.randn(2, 4, 8),
            torch.randn(2, 4, 8),
            torch.ones(2, 4, dtype=torch.bool),
        )
        adapter.set_contour_indices(torch.tensor([0, 1, 1]))
        tokens = torch.randn(3, 5, 8)
        output = adapter(tokens)
        self.assertGreater(float((output - tokens).abs().mean()), 0.0)

    def test_memory_read_scale_controls_residual_without_new_parameters(self):
        adapter = MemoryCrossAttention(dim=8, num_heads=2)
        torch.nn.init.normal_(adapter.output_proj.weight, std=0.05)
        adapter.set_slice_memory(
            torch.randn(1, 4, 8),
            torch.randn(1, 4, 8),
            torch.ones(1, 4, dtype=torch.bool),
        )
        adapter.set_contour_indices(torch.zeros(1, dtype=torch.long))
        tokens = torch.randn(1, 5, 8)
        adapter.set_read_scale(0.0)
        self.assertTrue(torch.equal(adapter(tokens), tokens))
        adapter.set_read_scale(2.0)
        self.assertGreater(float((adapter(tokens) - tokens).abs().mean()), 0.0)

    def test_empty_slice_in_mixed_batch_stays_identity(self):
        adapter = MemoryCrossAttention(dim=8, num_heads=2)
        torch.nn.init.normal_(adapter.output_proj.weight, std=0.05)
        adapter.set_slice_memory(
            torch.randn(2, 4, 8),
            torch.randn(2, 4, 8),
            torch.tensor([
                [True, True, True, True],
                [False, False, False, False],
            ]),
        )
        adapter.set_contour_indices(torch.tensor([0, 1]))
        tokens = torch.randn(2, 5, 8)
        output = adapter(tokens)
        self.assertGreater(float((output[0] - tokens[0]).abs().mean()), 0.0)
        self.assertTrue(torch.equal(output[1], tokens[1]))

    def test_relative_slice_distance_is_distinct_from_flow_time(self):
        encoding = RelativeSliceDistanceEncoding(dim=8, distance_scale=4.0)
        distance_one = encoding(torch.tensor([1.0]))
        distance_two = encoding(torch.tensor([2.0]))
        self.assertFalse(torch.equal(distance_one, distance_two))
        self.assertFalse(hasattr(encoding, "time_embedding"))

    def test_distance_encoding_can_be_kept_out_of_memory_values(self):
        controller = MemFlowDiTController(
            dit_blocks=[IdentityBlock()],
            memory_dim=8,
            state_dim=8,
            num_heads=2,
            distance_scale=4.0,
        )
        state = SliceMemoryState(
            volume_id="case_a",
            slice_index=0,
            slice_position=0.0,
            position_unit="index",
            key=torch.zeros(1, 8, 1, 1),
            value=torch.zeros(1, 8, 1, 1),
        )
        bank = SliceMemoryBank(capacity=1)
        bank.reset("case_a")
        bank.append(state)
        target = SliceSequenceMeta("case_a", 1, 1.0, "index", "ascending")

        controller.set_value_position_scale(0.0)
        controller.set_slice_memory([bank], [target])
        self.assertEqual(
            int(torch.count_nonzero(controller.adapters[0]._cached_value)),
            0,
        )

    def test_evidence_value_keeps_feature_content_out_of_values(self):
        encoder = SliceMemoryEncoder(
            feature_dim=4,
            memory_dim=8,
            mask_channels=3,
            pool_size=2,
            fusion_mode="evidence_value",
            mask_evidence_scale=0.25,
        )
        meta = SliceSequenceMeta("case_a", 1, 2.0, "mm", "ascending")
        mask = torch.zeros(1, 3, 4, 4)
        mask[:, 1, 1:3, 1:3] = 1.0
        first = encoder(torch.randn(1, 4, 4, 4), mask, meta)
        second = encoder(torch.randn(1, 4, 4, 4), mask, meta)
        self.assertFalse(torch.equal(first.key, second.key))
        self.assertTrue(torch.equal(first.value, second.value))

    def test_evidence_value_is_zero_for_empty_mask(self):
        encoder = SliceMemoryEncoder(
            feature_dim=4,
            memory_dim=8,
            mask_channels=3,
            pool_size=2,
            fusion_mode="evidence_value",
        )
        meta = SliceSequenceMeta("case_a", 1, 2.0, "mm", "ascending")
        state = encoder(
            torch.randn(1, 4, 4, 4),
            torch.zeros(1, 3, 4, 4),
            meta,
        )
        self.assertEqual(int(torch.count_nonzero(state.value)), 0)

    def test_causal_selection_uses_only_nearest_past_states(self):
        states = [self._state(index) for index in range(7)]
        meta = SliceSequenceMeta("case_a", 4, 4.0, "index", "ascending")
        selected = select_memory_states(
            states, meta, capacity=2, policy="causal-nearest"
        )
        self.assertEqual([state.slice_index for state in selected], [3, 2])

    def test_causal_strided_selection_expands_coverage_at_fixed_capacity(self):
        states = [self._state(index) for index in range(31)]
        meta = SliceSequenceMeta("case_a", 30, 30.0, "index", "ascending")
        selected = select_memory_states(
            states,
            meta,
            capacity=4,
            policy="causal-strided",
            stride=4.0,
        )
        self.assertEqual(
            [state.slice_index for state in selected],
            [29, 25, 21, 17],
        )

    def test_key_similarity_keeps_recent_and_selects_matching_history(self):
        states = [self._state(index) for index in range(7)]
        # Replace scalar keys with orthogonal descriptors.  Slice 1 is the
        # content match, while slice 5 must remain as the immediate predecessor.
        for state in states:
            state.key.zero_()
            state.key[:, state.slice_index % 8] = 1.0
        target = states[6]
        target.key.zero_()
        target.key[:, 1] = 1.0
        meta = SliceSequenceMeta("case_a", 6, 6.0, "index", "ascending")
        selected = select_memory_states(
            states,
            meta,
            capacity=3,
            policy="causal-recent-key-similar",
            target_state=target,
        )
        self.assertEqual(selected[0].slice_index, 5)
        self.assertEqual(selected[1].slice_index, 1)
        self.assertEqual(len(selected), 3)

    def test_key_similarity_requires_matching_target_state(self):
        states = [self._state(index) for index in range(4)]
        meta = SliceSequenceMeta("case_a", 3, 3.0, "index", "ascending")
        with self.assertRaises(ValueError):
            select_memory_states(
                states,
                meta,
                capacity=2,
                policy="causal-recent-key-similar",
            )

    def test_compact_bank_summarizes_old_nonempty_states(self):
        bank = SliceMemoryBank(capacity=2, global_pool_size=1)
        bank.reset("case_a")
        for index in range(4):
            bank.append(self._state(index + 1))
        states = bank.states()
        self.assertEqual(len(states), 3)
        self.assertTrue(states[0].is_global)
        self.assertEqual(bank.global_count, 2)
        self.assertEqual(tuple(states[0].key.shape), (1, 8, 1, 1))
        self.assertEqual([state.slice_index for state in states[1:]], [3, 4])
        self.assertTrue(torch.allclose(states[0].value, torch.full((1, 8, 1, 1), 1.5)))

    def test_compact_bank_does_not_dilute_with_empty_values(self):
        bank = SliceMemoryBank(capacity=1, global_pool_size=1)
        bank.reset("case_a")
        empty = SliceMemoryState(
            volume_id="case_a",
            slice_index=0,
            slice_position=0.0,
            position_unit="index",
            key=torch.ones(1, 8, 2, 2),
            value=torch.zeros(1, 8, 2, 2),
        )
        bank.append(empty)
        bank.append(self._state(1))
        self.assertTrue(bank.has_global_state)
        self.assertEqual(bank.global_count, 1)
        global_state = bank.states()[0]
        self.assertEqual(int(torch.count_nonzero(global_state.valid_mask)), 0)
        self.assertEqual(int(torch.count_nonzero(global_state.value)), 0)

    def test_global_state_skips_fake_slice_distance(self):
        controller = MemFlowDiTController(
            dit_blocks=[IdentityBlock()],
            memory_dim=8,
            state_dim=8,
            num_heads=2,
            distance_scale=4.0,
        )
        bank = SliceMemoryBank(capacity=1)
        bank.reset("case_a")
        bank.append(SliceMemoryState(
            volume_id="case_a",
            slice_index=0,
            slice_position=0.0,
            position_unit="mm",
            key=torch.zeros(1, 8, 1, 1),
            value=torch.zeros(1, 8, 1, 1),
            is_global=True,
        ))
        target = SliceSequenceMeta("case_a", 100, 100.0, "mm", "ascending")
        controller.set_slice_memory([bank], [target])
        self.assertEqual(
            int(torch.count_nonzero(controller.adapters[0]._cached_key)), 0
        )
        self.assertEqual(
            int(torch.count_nonzero(controller.adapters[0]._cached_value)), 0
        )

    def test_bidirectional_selection_excludes_target_and_is_symmetric(self):
        states = [self._state(index) for index in range(7)]
        meta = SliceSequenceMeta("case_a", 3, 3.0, "index", "ascending")
        selected = select_memory_states(
            states, meta, capacity=4, policy="bidirectional-nearest"
        )
        self.assertEqual([state.slice_index for state in selected], [2, 4, 1, 5])

    def test_absolute_distance_controller_accepts_future_memory(self):
        block = IdentityBlock()
        controller = MemFlowDiTController(
            dit_blocks=[block],
            memory_dim=8,
            state_dim=8,
            num_heads=2,
            distance_scale=4.0,
            distance_mode="absolute",
        )
        torch.nn.init.normal_(controller.adapters[0].output_proj.weight, std=0.05)
        bank = SliceMemoryBank(capacity=2)
        bank.reset("case_a")
        bank.append(self._state(4))
        meta = SliceSequenceMeta("case_a", 3, 3.0, "index", "ascending")
        controller.set_slice_memory([bank], [meta])
        controller.set_contour_indices(torch.zeros(1, dtype=torch.long))
        tokens = torch.randn(1, 5, 8)
        output = block(tokens, torch.randn(1, 3, 8), torch.randn(1, 8))
        self.assertGreater(float((output - tokens).abs().mean()), 0.0)

    def test_controller_injects_memory_inside_block_hook(self):
        block = IdentityBlock()
        controller = MemFlowDiTController(
            dit_blocks=[block],
            memory_dim=8,
            state_dim=8,
            num_heads=2,
            distance_scale=4.0,
        )
        torch.nn.init.normal_(controller.adapters[0].output_proj.weight, std=0.05)
        bank = SliceMemoryBank(capacity=2)
        bank.reset("case_a")
        bank.append(SliceMemoryState(
            volume_id="case_a",
            slice_index=0,
            slice_position=0.0,
            position_unit="index",
            key=torch.randn(1, 8, 2, 2),
            value=torch.randn(1, 8, 2, 2),
        ))
        meta = SliceSequenceMeta("case_a", 1, 1.0, "index", "ascending")
        controller.set_slice_memory([bank], [meta])
        controller.set_contour_indices(torch.zeros(2, dtype=torch.long))
        tokens = torch.randn(2, 5, 8)
        output = block(tokens, torch.randn(2, 3, 8), torch.randn(2, 8))
        self.assertGreater(float((output - tokens).abs().mean()), 0.0)
        self.assertEqual(controller.active_state_count, 1)


if __name__ == "__main__":
    unittest.main()
