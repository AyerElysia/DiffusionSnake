import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn


_ROOT = Path(__file__).parents[1]
_OLD_CFG_FILE = os.environ.get('CFG_FILE')
os.environ['CFG_FILE'] = str(_ROOT / 'configs/sagittal_2d_pseudo3d.yaml')
_OLD_ARGV = sys.argv[:]
try:
    sys.argv[:] = [sys.argv[0]]
    from lib.datasets.dataset_catalog import DatasetCatalog
    from lib.evaluators.sagittal_2d_fixed.snake import (
        Evaluator,
        binary_iou_dice,
        configure_box_mode,
        inverse_affine_points,
        rasterize_polygons,
    )
finally:
    sys.argv[:] = _OLD_ARGV
    if _OLD_CFG_FILE is None:
        os.environ.pop('CFG_FILE', None)
    else:
        os.environ['CFG_FILE'] = _OLD_CFG_FILE


def test_binary_iou_dice_perfect_partial_and_empty():
    empty = np.zeros((4, 4), dtype=np.uint8)
    full = np.ones((4, 4), dtype=np.uint8)
    partial = np.zeros((4, 4), dtype=np.uint8)
    partial[:2, :2] = 1

    assert binary_iou_dice(full, full) == pytest.approx((1.0, 1.0))
    assert binary_iou_dice(partial, full) == pytest.approx((0.25, 0.4))
    assert binary_iou_dice(empty, empty) == pytest.approx((1.0, 1.0))
    assert binary_iou_dice(empty, full) == pytest.approx((0.0, 0.0))


def test_rasterize_polygons_supports_class_ids_and_empty_polygons():
    first = np.asarray([[1, 1], [3, 1], [3, 3], [1, 3]], dtype=np.float32)
    second = np.asarray([[5, 5], [7, 5], [7, 7], [5, 7]], dtype=np.float32)
    labels = rasterize_polygons([first, second], (10, 10), class_ids=[2, 3])

    assert labels.dtype == np.uint16
    assert labels[2, 2] == 2
    assert labels[6, 6] == 3
    assert labels[0, 0] == 0
    np.testing.assert_array_equal(
        rasterize_polygons([], (10, 10)), np.zeros((10, 10), dtype=np.uint16)
    )


def test_inverse_affine_points_applies_flip_in_original_frame():
    points = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    inv = np.asarray([[2.0, 0.0, 3.0], [0.0, 2.0, 4.0]], dtype=np.float32)

    np.testing.assert_allclose(
        inverse_affine_points(points, inv, [10, 20]),
        [[5.0, 8.0], [9.0, 9.0]],
    )
    np.testing.assert_allclose(
        inverse_affine_points(points, inv, [10, 20], flipped=True),
        [[14.0, 8.0], [10.0, 9.0]],
    )


def test_configure_box_mode_sets_eval_switches():
    config = SimpleNamespace(use_gt_det=False, use_gt_det_train_only=True)

    assert configure_box_mode(config, 'gt') is config
    assert config.sagittal_eval_box_mode == 'gt'
    assert config.use_gt_det is True
    assert config.use_gt_det_train_only is False

    configure_box_mode(config, 'predicted')
    assert config.sagittal_eval_box_mode == 'predicted'
    assert config.use_gt_det is False
    with pytest.raises(ValueError):
        configure_box_mode(config, 'invalid')


def test_evaluator_rejects_contour_detection_count_mismatch():
    evaluator = Evaluator.__new__(Evaluator)
    output = {
        'detection': torch.tensor([[[0, 0, 4, 4, 0.9, 0]]], dtype=torch.float32),
        'py': torch.zeros((2, 8, 2), dtype=torch.float32),
    }

    with pytest.raises(RuntimeError, match='Contour/detection count mismatch'):
        evaluator._prepare_predictions(output, batch_size=1)


def test_evaluator_writes_slice_and_summary_json(tmp_path, monkeypatch):
    image_path = tmp_path / 'images' / 'case_a_0000.png'
    mask_path = tmp_path / 'masks' / 'case_a_0000.png'
    image_path.parent.mkdir()
    mask_path.parent.mkdir()
    assert cv2.imwrite(str(image_path), np.zeros((20, 20), dtype=np.uint8))
    mask = np.zeros((20, 20), dtype=np.uint16)
    mask[4:13, 4:13] = 1
    assert cv2.imwrite(str(mask_path), mask)

    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['split', 'case_id', 'slice_idx', 'image_path', 'mask_path'],
        )
        writer.writeheader()
        writer.writerow({
            'split': 'validation',
            'case_id': 'case_a',
            'slice_idx': '0',
            'image_path': str(image_path.relative_to(tmp_path)),
            'mask_path': str(mask_path.relative_to(tmp_path)),
        })

    monkeypatch.setattr(
        DatasetCatalog,
        'get',
        staticmethod(lambda name: {
            'id': 'sagittal_2d_fixed',
            'data_root': str(tmp_path),
            'ann_file': str(manifest),
            'split': 'val',
        }),
    )
    config = SimpleNamespace(
        test=SimpleNamespace(dataset='SagittalPseudo3DVal'),
        sagittal_eval_box_mode='predicted',
    )
    evaluator = Evaluator(str(tmp_path / 'results'), config=config)
    batch = {
        'inp': torch.zeros((1, 3, 20, 20), dtype=torch.float32),
        'img_path': [str(image_path)],
        'meta': {
            'case_id': ['case_a'],
            'slice_idx': torch.tensor([0]),
            'inv_trans_input': torch.eye(2, 3).unsqueeze(0),
            'orig_hw': torch.tensor([[20, 20]]),
            'flipped': torch.tensor([0]),
        },
    }
    output = {
        'detection': torch.tensor([[[4, 4, 12, 12, 0.9, 0]]], dtype=torch.float32),
        'py': torch.tensor([[[1, 1], [3, 1], [3, 3], [1, 3]]], dtype=torch.float32),
        'py_cls': torch.tensor([0], dtype=torch.long),
        'py_score': torch.tensor([0.9]),
    }

    evaluator.evaluate(output, batch)
    summary = evaluator.summarize()

    assert summary['num_slices'] == 1
    assert summary['all_slice_mean_iou'] == pytest.approx(1.0)
    assert summary['all_slice_mean_dice'] == pytest.approx(1.0)
    assert summary['class_mean_iou'] == pytest.approx(1.0)
    assert summary['class_aggregate']['1']['iou'] == pytest.approx(1.0)
    assert json.loads((tmp_path / 'results' / 'slices.json').read_text()) == evaluator.results
    assert json.loads((tmp_path / 'results' / 'summary.json').read_text())['num_slices'] == 1


def test_eval_script_defaults_to_moonvit_config_and_rejects_architecture_mismatch(
    tmp_path, monkeypatch
):
    from tools import eval_sagittal_2d_fixed as eval_script

    assert eval_script._DEFAULT_CFG_FILE == _ROOT / 'configs' / 'sagittal_2d_pseudo3d_moonvit.yaml'
    monkeypatch.delenv('CFG_FILE', raising=False)
    monkeypatch.setattr(sys, 'argv', ['eval_sagittal_2d_fixed.py'])
    assert eval_script._resolve_cfg_file() == str(eval_script._DEFAULT_CFG_FILE)

    network = nn.Linear(2, 2)
    original_weight = network.weight.detach().clone()
    checkpoint = tmp_path / 'mismatch.pt'
    torch.save(
        {'state_dict': {
            'weight': torch.zeros_like(network.weight),
            'unexpected': torch.zeros(1),
        }},
        checkpoint,
    )

    with pytest.raises(RuntimeError, match='architecture mismatch'):
        eval_script.load_checkpoint(network, checkpoint)
    torch.testing.assert_close(network.weight, original_weight)


def test_eval_checkpoint_rejects_non_tensor_without_mutating_network(tmp_path):
    from tools import eval_sagittal_2d_fixed as eval_script

    network = nn.Linear(2, 2)
    original_weight = network.weight.detach().clone()
    checkpoint = tmp_path / 'non_tensor.pt'
    torch.save(
        {'state_dict': {
            'weight': torch.zeros_like(network.weight),
            'bias': 'not-a-tensor',
        }},
        checkpoint,
    )

    with pytest.raises(RuntimeError, match='value is not a tensor'):
        eval_script.load_checkpoint(network, checkpoint)
    torch.testing.assert_close(network.weight, original_weight)


def test_eval_preflight_rejects_missing_cache_before_model_start(tmp_path):
    from tools import eval_sagittal_2d_fixed as eval_script

    existing = tmp_path / 'x0000.npz'
    existing.touch()
    dataset = SimpleNamespace(
        locate_feat_enabled=True,
        records=[{'cache_path': str(existing)}, {'cache_path': str(tmp_path / 'x0001.npz')}],
        _sagittal_moonvit_cache_path=lambda row: row['cache_path'],
    )

    assert eval_script.preflight_locate_cache(dataset, max_slices=1) == 1
    with pytest.raises(FileNotFoundError, match='1 of 2 slices missing'):
        eval_script.preflight_locate_cache(dataset, max_slices=2)
