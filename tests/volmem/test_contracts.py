import pathlib
import sys
import unittest

import numpy as np
import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from volmem.adapters.legacy_dataset import align_mask_to_token_grid
from volmem.data.sequence import VolumeChunkSampler
from volmem.models.contracts import SliceMemoryState, SliceSequenceMeta
from volmem.models.memory_bank import SliceMemoryBank
from volmem.models.slice_memory import SliceMemoryAttention, SliceMemoryEncoder
from volmem.models.volmem_snake import VolMemSnake


def make_state(volume_id, slice_index, requires_grad=False):
    tensor = torch.zeros(1, 4, 2, 2, requires_grad=requires_grad)
    return SliceMemoryState(
        volume_id=volume_id,
        slice_index=slice_index,
        slice_position=float(slice_index),
        position_unit="index",
        key=tensor,
        value=tensor,
    )


class DummyContourAdapter(torch.nn.Module):
    def forward(self, batch):
        zero = batch["locate_feat"][0].sum() * 0.0
        return zero, {}

    def predict(self, batch):
        return {"conditioned": batch["locate_feat"][0]}


class VolMemContractTests(unittest.TestCase):
    def test_memory_bank_is_volume_scoped(self):
        bank = SliceMemoryBank(capacity=2)
        bank.append(make_state("case_a", 0))
        with self.assertRaises(ValueError):
            bank.append(make_state("case_b", 1))

    def test_memory_bank_has_bounded_capacity(self):
        bank = SliceMemoryBank(capacity=2)
        bank.append(make_state("case_a", 0))
        bank.append(make_state("case_a", 1))
        bank.append(make_state("case_a", 2))
        self.assertEqual(
            [state.slice_index for state in bank.states()],
            [1, 2],
        )

    def test_detach_keeps_only_recent_state_trainable(self):
        bank = SliceMemoryBank(capacity=3)
        bank.append(make_state("case_a", 0, True))
        bank.append(make_state("case_a", 1, True))
        bank.detach_states(keep_recent=1)
        self.assertFalse(bank.states()[0].key.requires_grad)
        self.assertTrue(bank.states()[1].key.requires_grad)

    def test_empty_memory_is_identity(self):
        attention = SliceMemoryAttention(feature_dim=4, memory_dim=4, num_heads=1)
        features = torch.randn(1, 4, 2, 2)
        self.assertTrue(torch.equal(attention(features, []), features))

    def test_memory_path_backpropagates_to_encoder(self):
        encoder = SliceMemoryEncoder(4, 4, pool_size=2)
        attention = SliceMemoryAttention(4, 4, num_heads=1)
        attention.output_proj.weight.data.normal_(mean=0.0, std=0.01)
        features = torch.randn(1, 4, 3, 3)
        mask = torch.ones(1, 1, 3, 3)
        meta = SliceSequenceMeta("case_a", 0, 0.0, "index", "ascending")
        state = encoder(features, mask, meta)
        current = torch.randn(1, 4, 3, 3, requires_grad=True)
        attention(current, [state]).sum().backward()
        self.assertIsNotNone(encoder.key_proj.weight.grad)
        self.assertGreater(float(encoder.key_proj.weight.grad.abs().sum()), 0.0)

    def test_balanced_memory_encoder_keeps_mask_as_separate_signal(self):
        encoder = SliceMemoryEncoder(
            4,
            8,
            mask_channels=2,
            pool_size=2,
            fusion_mode="balanced_add",
            mask_evidence_scale=0.25,
        )
        self.assertEqual(encoder.key_proj.in_channels, 4)
        self.assertEqual(encoder.mask_key_proj.in_channels, 2)
        features = torch.zeros(1, 4, 4, 4)
        empty_mask = torch.zeros(1, 2, 4, 4)
        sparse_mask = empty_mask.clone()
        sparse_mask[:, 1, 0, 0] = 1.0
        meta = SliceSequenceMeta("case_a", 0, 0.0, "index", "ascending")
        empty_state = encoder(features, empty_mask, meta)
        sparse_state = encoder(features, sparse_mask, meta)
        self.assertTrue(torch.equal(empty_state.key, torch.zeros_like(empty_state.key)))
        self.assertGreater(
            float((sparse_state.key - empty_state.key).abs().mean()),
            0.01,
        )

    def test_sequence_meta_rejects_non_spatial_direction(self):
        meta = SliceSequenceMeta("case_a", 0, 0.0, "index", "forward")
        with self.assertRaises(ValueError):
            meta.validate()

    def test_mask_alignment_respects_cache_padding(self):
        mask = np.zeros((4, 2), dtype=np.uint8)
        mask[:, 0] = 1
        metadata = {
            "locate_feat_resized_hw": np.asarray([4, 2]),
            "locate_feat_padded_hw": np.asarray([4, 4]),
            "locate_feat_pad": np.asarray([0, 0, 2, 0]),
            "locate_feat_grid_hw": np.asarray([2, 2]),
        }
        token_mask = align_mask_to_token_grid(mask, metadata)
        self.assertEqual(token_mask.shape, (1, 2, 2))
        self.assertGreater(float(token_mask[0, :, 0].mean()), 0.0)
        self.assertEqual(float(token_mask[0, :, 1].max()), 0.0)

    def test_mask_alignment_preserves_partial_patch_coverage(self):
        mask = np.ones((4, 3), dtype=np.uint8)
        metadata = {
            "locate_feat_resized_hw": np.asarray([4, 3]),
            "locate_feat_padded_hw": np.asarray([4, 4]),
            "locate_feat_pad": np.asarray([0, 0, 1, 0]),
            "locate_feat_grid_hw": np.asarray([2, 2]),
        }
        token_mask = align_mask_to_token_grid(mask, metadata)
        partial = float(token_mask[0, :, 1].mean())
        self.assertGreater(partial, 0.0)
        self.assertLess(partial, 1.0)

    def test_prediction_writes_memory_only_after_explicit_write(self):
        model = VolMemSnake(
            contour_adapter=DummyContourAdapter(),
            feature_dim=4,
            memory_dim=4,
            memory_capacity=2,
            memory_heads=1,
            memory_pool_size=2,
        )
        bank = model.new_banks(["case_a"])
        meta = SliceSequenceMeta("case_a", 0, 0.0, "index", "ascending")
        batch = {"locate_feat": [torch.randn(4, 3, 3)]}
        output, raw_features, _ = model.predict_step(batch, [meta], bank)
        self.assertIn("conditioned", output)
        self.assertEqual(len(bank[0]), 0)
        mask = torch.ones(1, 1, 3, 3)
        model.write_step(raw_features, [mask], [meta], bank)
        self.assertEqual(len(bank[0]), 1)

    def test_chunk_sampler_uses_distinct_volumes_per_step(self):
        records = []
        for volume_id in ("a", "b", "c"):
            for slice_index in range(5):
                records.append({"case_id": volume_id, "slice_idx": slice_index})
        sampler = VolumeChunkSampler(
            records,
            chunk_length=3,
            chunks_per_step=2,
            seed=1,
            steps_per_epoch=4,
        )
        for windows in sampler:
            volume_ids = [window[0] for window in windows]
            self.assertEqual(len(volume_ids), len(set(volume_ids)))
            for _, slice_indices, _ in windows:
                self.assertEqual(
                    list(slice_indices),
                    list(range(slice_indices[0], slice_indices[0] + 3)),
                )


if __name__ == "__main__":
    unittest.main()
