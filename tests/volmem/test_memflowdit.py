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
from volmem.models.memory_bank import SliceMemoryBank


class IdentityBlock(torch.nn.Module):
    def forward(self, x, context, t_emb):
        return x


class MemFlowDiTTests(unittest.TestCase):
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
