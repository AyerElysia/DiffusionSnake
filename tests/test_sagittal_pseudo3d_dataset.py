import csv
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import torch


_ROOT = Path(__file__).parents[1]
_OLD_CFG_FILE = os.environ.get('CFG_FILE')
os.environ['CFG_FILE'] = str(_ROOT / 'configs/sagittal_2d_pseudo3d.yaml')
_OLD_ARGV = sys.argv[:]
try:
    sys.argv[:] = [sys.argv[0]]
    from lib.config import cfg
    from lib.datasets.collate_batch import snake_collator
    from lib.datasets.dataset_catalog import DatasetCatalog
    from lib.datasets.sagittal_2d_fixed.snake import Dataset
    from lib.utils.snake import snake_config, snake_voc_utils
finally:
    sys.argv[:] = _OLD_ARGV
    if _OLD_CFG_FILE is None:
        os.environ.pop('CFG_FILE', None)
    else:
        os.environ['CFG_FILE'] = _OLD_CFG_FILE


_FIELDS = [
    'split', 'case_id', 'slice_idx', 'image_path', 'mask_path', 'has_mask'
]


@pytest.fixture(autouse=True)
def _configure_pseudo3d_for_tests():
    previous = {
        'mean': cfg.pseudo3d_mean,
        'std': cfg.pseudo3d_std,
        'color_aug': cfg.pseudo3d_color_aug,
        'lr_flip': cfg.pseudo3d_lr_flip,
        'input_mode': cfg.pseudo3d_input_mode,
        'ct_hm': cfg.heads.ct_hm,
        'vis_zrc': cfg.vis_zrc,
    }
    cfg.pseudo3d_mean = 0.0
    cfg.pseudo3d_std = 1.0
    cfg.pseudo3d_color_aug = False
    cfg.pseudo3d_lr_flip = False
    cfg.pseudo3d_input_mode = 'neighbors'
    cfg.heads.ct_hm = 26
    cfg.vis_zrc = 0
    yield
    cfg.pseudo3d_mean = previous['mean']
    cfg.pseudo3d_std = previous['std']
    cfg.pseudo3d_color_aug = previous['color_aug']
    cfg.pseudo3d_lr_flip = previous['lr_flip']
    cfg.pseudo3d_input_mode = previous['input_mode']
    cfg.heads.ct_hm = previous['ct_hm']
    cfg.vis_zrc = previous['vis_zrc']


def _add_slice(root, rows, split, case_id, slice_idx, value, mask=None):
    shape = (40, 40)
    image_dir = root / split / 'images' / case_id
    mask_dir = root / split / 'masks' / case_id
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / '{}_x{:04d}.png'.format(case_id, slice_idx)
    mask_path = mask_dir / '{}_x{:04d}_mask.png'.format(case_id, slice_idx)
    image = np.full(shape, value, dtype=np.uint8)
    if mask is None:
        mask = np.zeros(shape, dtype=np.uint16)
    else:
        mask = np.asarray(mask, dtype=np.uint16)

    assert cv2.imwrite(str(image_path), image)
    assert cv2.imwrite(str(mask_path), mask)
    rows.append({
        'split': split,
        'case_id': case_id,
        'slice_idx': str(slice_idx),
        'image_path': str(image_path.relative_to(root)),
        'mask_path': str(mask_path.relative_to(root)),
        'has_mask': 'True',
    })


def _write_manifest(root, rows):
    manifest = root / 'manifests' / 'slice_manifest.csv'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _make_dataset(root, rows, split='val'):
    manifest = _write_manifest(root, rows)
    return Dataset(ann_file=str(manifest), data_root=str(root), split=split)


def _sample(dataset, case_id, slice_idx):
    index = next(
        index for index, row in enumerate(dataset.records)
        if row['case_id'] == case_id and row['slice_idx'] == slice_idx
    )
    return dataset[index]


def _center_channels(sample):
    return sample['inp'][:, 256, 256]


def test_neighbors_are_case_local_with_boundary_copy_and_channel_order(tmp_path):
    rows = []
    case_a = 'case_alpha_with_underscores'
    case_b = 'case_beta_with_underscores'
    for slice_idx, value in enumerate((10, 20, 30)):
        _add_slice(tmp_path, rows, 'validation', case_a, slice_idx, value)
    for slice_idx, value in enumerate((100, 110, 120)):
        _add_slice(tmp_path, rows, 'validation', case_b, slice_idx, value)

    dataset = _make_dataset(tmp_path, rows)
    expected = {
        (case_a, 0): ((0, 0, 1), (10, 10, 20)),
        (case_a, 1): ((0, 1, 2), (10, 20, 30)),
        (case_a, 2): ((1, 2, 2), (20, 30, 30)),
        (case_b, 0): ((0, 0, 1), (100, 100, 110)),
    }

    for key, (indices, values) in expected.items():
        sample = _sample(dataset, *key)
        assert sample['meta']['case_id'] == key[0]
        assert sample['meta']['slice_idx'] == key[1]
        np.testing.assert_array_equal(sample['meta']['neighbor_indices'], indices)
        np.testing.assert_allclose(
            _center_channels(sample), np.asarray(values) / 255.0, atol=1e-6
        )

    # The second case's first slice must copy itself, never use the last slice of case A.
    first_b = _sample(dataset, case_b, 0)
    assert float(_center_channels(first_b)[0]) == pytest.approx(100.0 / 255.0)


def test_center_repeat_uses_center_slice_in_all_input_channels(tmp_path):
    rows = []
    case_id = 'center_repeat_case'
    for slice_idx, value in enumerate((11, 77, 143)):
        _add_slice(tmp_path, rows, 'validation', case_id, slice_idx, value)

    cfg.pseudo3d_input_mode = 'center_repeat'
    dataset = _make_dataset(tmp_path, rows)
    sample = _sample(dataset, case_id, 1)

    np.testing.assert_allclose(
        _center_channels(sample), np.full(3, 77.0 / 255.0), atol=1e-6
    )
    np.testing.assert_array_equal(sample['inp'][0], sample['inp'][1])
    np.testing.assert_array_equal(sample['inp'][1], sample['inp'][2])
    np.testing.assert_array_equal(sample['meta']['neighbor_indices'], (0, 1, 2))


def test_uint16_label25_multicomponent_targets_and_center_orig_img(tmp_path):
    rows = []
    case_id = 'label_25_case'
    _add_slice(tmp_path, rows, 'validation', case_id, 0, 21)

    mask = np.zeros((40, 40), dtype=np.uint16)
    mask[4:13, 5:14] = 25
    mask[23:34, 25:36] = 25
    _add_slice(tmp_path, rows, 'validation', case_id, 1, 77, mask=mask)
    _add_slice(tmp_path, rows, 'validation', case_id, 2, 201)

    dataset = _make_dataset(tmp_path, rows)
    sample = _sample(dataset, case_id, 1)

    assert sample['ct_hm'].shape == (26, 128, 128)
    assert sample['meta']['ct_num'] == 2
    assert sample['ct_cls'] == [25, 25]
    assert sample['ct_hm'][25].max() == pytest.approx(1.0)
    assert sample['ct_hm'][:25].max() == pytest.approx(0.0)
    assert sample['cls'].shape == (2, 1)
    torch.testing.assert_close(sample['cls'], torch.full((2, 1), 24.0))
    assert sample['bboxes'].shape == (2, 4)
    assert len(sample['i_it_py']) == 2
    assert len(sample['i_gt_py']) == 2

    # Visualization input must be the affine center slice repeated, not pseudo-color neighbors.
    assert sample['orig_img'].shape == (512, 512, 3)
    np.testing.assert_array_equal(sample['orig_img'][:, :, 0], sample['orig_img'][:, :, 1])
    np.testing.assert_array_equal(sample['orig_img'][:, :, 1], sample['orig_img'][:, :, 2])
    np.testing.assert_array_equal(sample['orig_img'][256, 256], (77, 77, 77))


def test_empty_center_mask_returns_legal_targets_and_collates(tmp_path):
    rows = []
    case_id = 'empty_mask_case'
    _add_slice(tmp_path, rows, 'validation', case_id, 0, 40)

    nonempty_mask = np.zeros((40, 40), dtype=np.uint16)
    nonempty_mask[8:28, 10:30] = 3
    _add_slice(
        tmp_path, rows, 'validation', case_id, 1, 50, mask=nonempty_mask
    )

    dataset = _make_dataset(tmp_path, rows)
    empty = _sample(dataset, case_id, 0)
    nonempty = _sample(dataset, case_id, 1)

    assert empty['meta']['ct_num'] == 0
    assert empty['wh'] == []
    assert empty['ct_cls'] == []
    assert empty['ct_ind'] == []
    assert empty['bboxes'].shape == (0, 4)
    assert empty['cls'].shape == (0, 1)
    assert empty['batch_idx'].shape == (0, 1)

    empty_batch = snake_collator([empty])
    assert empty_batch['inp'].shape == (1, 3, 512, 512)
    assert empty_batch['ct_hm'].shape == (1, 26, 128, 128)
    assert empty_batch['wh'].shape == (1, 0, 2)
    assert empty_batch['i_it_py'].shape[1] == 0
    assert empty_batch['bboxes'].shape == (0, 4)
    assert empty_batch['meta']['case_id'] == [case_id]
    torch.testing.assert_close(
        empty_batch['meta']['neighbor_indices'], torch.tensor([[0, 0, 1]])
    )

    mixed_batch = snake_collator([empty, nonempty])
    torch.testing.assert_close(mixed_batch['meta']['ct_num'], torch.tensor([0, 1]))
    assert mixed_batch['bboxes'].shape == (1, 4)
    torch.testing.assert_close(mixed_batch['cls'], torch.tensor([[2.0]]))
    torch.testing.assert_close(mixed_batch['batch_idx'], torch.tensor([[1.0]]))


def test_manifest_split_mapping_and_deterministic_mini_case(tmp_path):
    rows = []
    for slice_idx in range(3):
        _add_slice(tmp_path, rows, 'training', 'first_training_case', slice_idx, 10)
    for slice_idx in range(2):
        _add_slice(tmp_path, rows, 'training', 'second_training_case', slice_idx, 20)
    _add_slice(tmp_path, rows, 'validation', 'validation_case', 0, 30)
    _add_slice(tmp_path, rows, 'test', 'test_case', 0, 40)
    manifest = _write_manifest(tmp_path, rows)

    train = Dataset(str(manifest), str(tmp_path), 'train')
    val = Dataset(str(manifest), str(tmp_path), 'val')
    mini = Dataset(str(manifest), str(tmp_path), 'mini')
    test = Dataset(str(manifest), str(tmp_path), 'test')

    assert {row['case_id'] for row in train.records} == {
        'first_training_case', 'second_training_case'
    }
    assert [row['case_id'] for row in val.records] == ['validation_case']
    assert [row['case_id'] for row in test.records] == ['test_case']
    assert [row['case_id'] for row in mini.records] == ['first_training_case'] * 3
    assert [row['slice_idx'] for row in mini.records] == [0, 1, 2]


def test_catalog_registers_all_sagittal_splits():
    expected = {
        'SagittalPseudo3DTrain': 'train',
        'SagittalPseudo3DVal': 'val',
        'SagittalPseudo3DMini': 'mini',
        'SagittalPseudo3DTest': 'test',
    }
    for name, split in expected.items():
        attrs = DatasetCatalog.get(name)
        assert attrs['id'] == 'sagittal_2d_fixed'
        assert attrs['split'] == split
        assert attrs['data_root'].endswith('/sagittal_2d_fixed')
        assert attrs['ann_file'].endswith('/manifests/slice_manifest.csv')


def test_augment_kwargs_disable_medical_color_and_lr_flip(monkeypatch):
    image = np.full((40, 40, 3), 100, dtype=np.uint8)
    calls = []

    def record_color_aug(*args, **kwargs):
        calls.append(True)

    monkeypatch.setattr(snake_voc_utils.data_utils, 'color_aug', record_color_aug)
    monkeypatch.setattr(np.random, 'random', lambda: 0.0)

    legacy = snake_voc_utils.augment(
        image.copy(), 'train',
        snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
        snake_config.mean, snake_config.std,
    )
    assert legacy[4] is True
    assert calls == [True]

    calls.clear()
    controlled = snake_voc_utils.augment(
        image.copy(), 'train',
        snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
        snake_config.mean, snake_config.std,
        color_aug=False, lr_flip=False,
    )
    assert controlled[4] is False
    assert calls == []


def test_augment_random_crop_false_keeps_training_center_and_scale(monkeypatch):
    image = np.full((40, 60, 3), 100, dtype=np.uint8)

    def unexpected_random_crop(*args, **kwargs):
        pytest.fail('random crop RNG should not be used when random_crop=False')

    monkeypatch.setattr(np.random, 'uniform', unexpected_random_crop)
    monkeypatch.setattr(np.random, 'randint', unexpected_random_crop)

    augmented = snake_voc_utils.augment(
        image, 'train',
        snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
        snake_config.mean, snake_config.std,
        color_aug=False, lr_flip=False, random_crop=False,
    )

    np.testing.assert_array_equal(augmented[5], np.asarray([30.0, 20.0]))
    np.testing.assert_array_equal(augmented[6], np.asarray([60.0, 60.0]))
    assert augmented[4] is False
