import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


_ROOT = Path(__file__).parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_OLD_CFG_FILE = os.environ.get('CFG_FILE')
_OLD_ARGV = sys.argv[:]
os.environ['CFG_FILE'] = str(_ROOT / 'configs/sagittal_2d_pseudo3d.yaml')
try:
    sys.argv[:] = [sys.argv[0]]
    from lib.config import cfg
    from lib.datasets import make_dataset
    from lib.datasets.samplers import ForegroundBalancedSampler
    from lib.datasets.sagittal_2d_fixed.snake import Dataset
finally:
    sys.argv[:] = _OLD_ARGV
    if _OLD_CFG_FILE is None:
        os.environ.pop('CFG_FILE', None)
    else:
        os.environ['CFG_FILE'] = _OLD_CFG_FILE


class _FlagDataset:
    def __init__(self, flags):
        self.foreground_flags = list(flags)

    def __len__(self):
        return len(self.foreground_flags)


class ForegroundSamplingScopeTest(unittest.TestCase):
    def test_bbox_manifest_aggregates_duplicate_slices_and_defaults_missing_to_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = os.path.join(directory, 'bbox_2d_manifest.csv')
            with open(manifest_path, 'w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=['split', 'case_id', 'slice_idx', 'mask_pixel_count'],
                )
                writer.writeheader()
                writer.writerows([
                    {
                        'split': 'training',
                        'case_id': 'case_a',
                        'slice_idx': '3',
                        'mask_pixel_count': '0',
                    },
                    {
                        'split': 'training',
                        'case_id': 'case_a',
                        'slice_idx': '3',
                        'mask_pixel_count': '7',
                    },
                    {
                        'split': 'training',
                        'case_id': 'case_a',
                        'slice_idx': '4',
                        'mask_pixel_count': '0',
                    },
                    {
                        'split': 'validation',
                        'case_id': 'case_a',
                        'slice_idx': '3',
                        'mask_pixel_count': '99',
                    },
                ])

            keys = Dataset._read_foreground_keys(manifest_path)
            self.assertEqual(keys, {('training', 'case_a', 3), ('validation', 'case_a', 3)})

            slice_manifest = os.path.join(directory, 'slice_manifest.csv')
            with open(slice_manifest, 'w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=['split', 'case_id', 'slice_idx', 'image_path', 'mask_path'],
                )
                writer.writeheader()
                writer.writerows([
                    {
                        'split': 'training',
                        'case_id': 'case_a',
                        'slice_idx': '3',
                        'image_path': 'case_a_x0003.png',
                        'mask_path': 'case_a_x0003_mask.png',
                    },
                    {
                        'split': 'training',
                        'case_id': 'case_a',
                        'slice_idx': '4',
                        'image_path': 'case_a_x0004.png',
                        'mask_path': 'case_a_x0004_mask.png',
                    },
                    {
                        'split': 'training',
                        'case_id': 'case_a',
                        'slice_idx': '5',
                        'image_path': 'case_a_x0005.png',
                        'mask_path': 'case_a_x0005_mask.png',
                    },
                ])

            had_bbox_manifest = 'bbox_2d_manifest' in cfg.train
            previous_bbox_manifest = getattr(cfg.train, 'bbox_2d_manifest', None)
            cfg.train.bbox_2d_manifest = manifest_path
            try:
                dataset = Dataset(slice_manifest, directory, 'train')
            finally:
                if had_bbox_manifest:
                    cfg.train.bbox_2d_manifest = previous_bbox_manifest
                else:
                    del cfg.train['bbox_2d_manifest']
            self.assertEqual(dataset.foreground_flags, [True, False, False])
            self.assertEqual(len(dataset.records), len(dataset.foreground_flags))

            missing_bbox_dataset = Dataset(slice_manifest, directory, 'train')
            self.assertEqual(missing_bbox_dataset.foreground_flags, [False, False, False])

    def test_balanced_sampler_is_epoch_seeded_and_ddp_rank_strided(self):
        dataset = _FlagDataset([True, True, False, False, False, False])
        rank_zero = ForegroundBalancedSampler(
            dataset, foreground_fraction=0.5, seed=17, num_replicas=2, rank=0
        )
        rank_one = ForegroundBalancedSampler(
            dataset, foreground_fraction=0.5, seed=17, num_replicas=2, rank=1
        )

        rank_zero.set_epoch(0)
        rank_one.set_epoch(0)
        samples_zero = list(rank_zero)
        samples_one = list(rank_one)
        self.assertEqual(len(samples_zero), len(samples_one))
        self.assertEqual(len(samples_zero), len(rank_zero))
        self.assertEqual(
            sum(dataset.foreground_flags[index] for index in samples_zero + samples_one),
            3,
        )

        rank_zero.set_epoch(1)
        rank_one.set_epoch(1)
        self.assertNotEqual(samples_zero + samples_one, list(rank_zero) + list(rank_one))

        rank_zero_again = ForegroundBalancedSampler(
            dataset, foreground_fraction=0.5, seed=17, num_replicas=2, rank=0
        )
        rank_zero_again.set_epoch(0)
        self.assertEqual(samples_zero, list(rank_zero_again))

    def test_balance_is_train_only_and_merge_conflicts(self):
        cfg = SimpleNamespace(
            task='snake',
            random_num=0,
            train=SimpleNamespace(
                balance_foreground_empty=True,
                foreground_fraction=0.5,
                merge_with_val=True,
                dataset='SagittalPseudo3DTrain',
                batch_size=1,
                num_workers=0,
            )
        )
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            make_dataset.make_data_loader(cfg, is_train=True)

        cfg.train.merge_with_val = False
        with mock.patch.object(make_dataset, 'make_transforms', return_value=None), \
                mock.patch.object(make_dataset, 'make_dataset', return_value=_FlagDataset([True, False])):
            loader = make_dataset.make_data_loader(cfg, is_train=True)
        self.assertIsInstance(loader.batch_sampler.sampler, ForegroundBalancedSampler)

    def test_formal_config_contains_full_moonvit_settings_and_training_overrides(self):
        import yaml

        with open(
                _ROOT / 'configs/sagittal_2d_pseudo3d_moonvit.yaml',
                'r', encoding='utf-8') as source:
            base = yaml.safe_load(source)
        with open(
                _ROOT / 'configs/sagittal_2d_pseudo3d_moonvit_train.yaml',
                'r', encoding='utf-8') as source:
            train = yaml.safe_load(source)

        self.assertEqual(set(base), set(train))
        self.assertEqual(train['train']['dataset'], 'SagittalPseudo3DTrain')
        self.assertTrue(train['train']['balance_foreground_empty'])
        self.assertEqual(train['train']['foreground_fraction'], 0.5)
        self.assertEqual(train['train']['max_steps'], 5000)
        self.assertEqual(train['train']['epoch'], 2)
        self.assertEqual(train['train']['gradient_clip'], 1.0)
        self.assertEqual(train['train']['gradient_accumulation_steps'], 1)
        self.assertFalse(train['resume'])
        self.assertFalse(train['train']['merge_with_val'])
        self.assertNotEqual(base['model_dir'], train['model_dir'])
        self.assertTrue(os.path.isabs(train['train']['bbox_2d_manifest']))


if __name__ == '__main__':
    unittest.main()
