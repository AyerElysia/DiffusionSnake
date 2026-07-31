import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn


_ROOT = Path(__file__).parents[1]
_OLD_CFG_FILE = os.environ.get('CFG_FILE')
os.environ['CFG_FILE'] = str(_ROOT / 'configs/sagittal_2d_pseudo3d.yaml')
_OLD_ARGV = sys.argv[:]
_OLD_SIMPLEITK = sys.modules.get('SimpleITK')
_SIMPLEITK_WAS_ABSENT = 'SimpleITK' not in sys.modules
if _SIMPLEITK_WAS_ABSENT:
    sys.modules['SimpleITK'] = types.ModuleType('SimpleITK')
try:
    sys.argv[:] = [sys.argv[0]]
    from lib.config import cfg
    from lib.networks.snake.ct_snake import Network
finally:
    sys.argv[:] = _OLD_ARGV
    if _OLD_CFG_FILE is None:
        os.environ.pop('CFG_FILE', None)
    else:
        os.environ['CFG_FILE'] = _OLD_CFG_FILE
    if _SIMPLEITK_WAS_ABSENT:
        sys.modules.pop('SimpleITK', None)
    else:
        sys.modules['SimpleITK'] = _OLD_SIMPLEITK


@pytest.fixture(autouse=True)
def _restore_detection_config():
    keys = (
        'gt_detection_class_offset',
        'det_conf_thresh',
        'det_iou_thresh',
        'det_max_det',
        'per_class_nms',
        'use_nms_for_snake',
        'use_gt_det',
        'use_gt_det_train_only',
        'contour_init_method',
        'yolo_num_classes',
    )
    previous = {key: getattr(cfg, key) for key in keys}
    yield
    for key, value in previous.items():
        setattr(cfg, key, value)


def _bare_network(detector_backend='yolo'):
    network = Network.__new__(Network)
    nn.Module.__init__(network)
    network.down_ratio = 4.0
    network.detector_backend = detector_backend
    return network


def test_gt_detection_offset_maps_labels_1_and_25_to_yolo_0_and_24():
    cfg.gt_detection_class_offset = 1
    network = _bare_network()
    output = {'feat_hw': (4, 4)}
    batch = {
        'inp': torch.zeros((1, 3, 16, 16), dtype=torch.float32),
        'ct_01': torch.tensor([[True, True]]),
        'ct_ind': torch.tensor([[0, 15]], dtype=torch.long),
        'wh': torch.tensor([[[2.0, 2.0], [2.0, 2.0]]]),
        'ct_cls': torch.tensor([[1, 25]], dtype=torch.long),
    }

    network.use_gt_detection(output, batch)

    torch.testing.assert_close(
        output['detection'][0, :, 5], torch.tensor([0.0, 24.0])
    )


def test_train_only_gt_detection_is_skipped_in_eval_and_with_none_batch():
    calls = []

    def maybe_replace(is_training, batch):
        if Network.should_use_gt_detection(True, True, is_training, batch):
            calls.append(batch)

    maybe_replace(is_training=False, batch=None)
    maybe_replace(is_training=False, batch={'ct_01': torch.tensor([[True]])})
    assert calls == []

    training_batch = {'ct_01': torch.tensor([[True]])}
    maybe_replace(is_training=True, batch=training_batch)
    assert calls == [training_batch]

    assert Network.should_use_gt_detection(True, False, False, training_batch)
    assert not Network.should_use_gt_detection(False, False, True, training_batch)


def test_filter_detection_candidates_uses_axis_aligned_per_class_nms():
    cfg.det_conf_thresh = 0.1
    cfg.det_iou_thresh = 0.5
    cfg.det_max_det = 10
    cfg.per_class_nms = True
    cfg.use_nms_for_snake = True
    network = _bare_network()
    raw_detection = torch.tensor([
        [
            [0.0, 0.0, 10.0, 10.0, 0.90, 2.0],
            [1.0, 1.0, 11.0, 11.0, 0.80, 2.0],
            [1.0, 1.0, 11.0, 11.0, 0.70, 3.0],
        ]
    ])

    detection = network.filter_detection_candidates(raw_detection)

    assert detection.shape == (1, 2, 6)
    torch.testing.assert_close(detection[0, :, 4], torch.tensor([0.90, 0.70]))
    torch.testing.assert_close(detection[0, :, 5], torch.tensor([2.0, 3.0]))


def test_attach_py_detection_metadata_follows_batch_row_major_order():
    detection = torch.tensor([
        [
            [0.0, 0.0, 5.0, 5.0, 0.80, 24.0],
            [6.0, 6.0, 9.0, 9.0, 0.60, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.00, 0.0],
        ],
        [
            [2.0, 2.0, 8.0, 8.0, 0.90, 7.0],
            [0.0, 0.0, 0.0, 0.0, 0.00, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.00, 0.0],
        ],
    ])
    output = {
        'detection': detection,
        'py': torch.zeros((3, 128, 2)),
    }

    Network.attach_py_detection_metadata(
        output, fail_on_mismatch=True, num_classes=25
    )

    torch.testing.assert_close(output['py_score'], torch.tensor([0.80, 0.60, 0.90]))
    torch.testing.assert_close(output['py_cls'], torch.tensor([24, 0, 7]))
    assert output['py_cls'].dtype == torch.long

    mismatch = {'detection': detection, 'py': torch.zeros((2, 128, 2))}
    with pytest.raises(RuntimeError, match='Cannot associate final contours'):
        Network.attach_py_detection_metadata(mismatch, fail_on_mismatch=True)


def test_symmetric_stem_preserves_center_repeat_grayscale_response():
    weight = torch.arange(2 * 3 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3, 3)
    symmetric = Network.make_symmetric_stem_weight(weight)

    torch.testing.assert_close(symmetric[:, 0], symmetric[:, 1])
    torch.testing.assert_close(symmetric[:, 1], symmetric[:, 2])

    grayscale = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5) / 25.0
    center_repeat = grayscale.repeat(1, 3, 1, 1)
    original_response = F.conv2d(center_repeat, weight)
    symmetric_response = F.conv2d(center_repeat, symmetric)
    torch.testing.assert_close(symmetric_response, original_response)


def test_yolo_forward_restores_detection_and_attaches_contour_metadata():
    class FakeYolo(nn.Module):
        def forward(self, x):
            # One decoded xywh candidate with class 24 as the highest logit.
            prediction = x.new_full((x.size(0), 29, 1), -20.0)
            prediction[:, 0, 0] = 8.0
            prediction[:, 1, 0] = 8.0
            prediction[:, 2, 0] = 8.0
            prediction[:, 3, 0] = 8.0
            prediction[:, 28, 0] = 20.0
            feature = x.new_zeros((x.size(0), 89, 4, 4))
            return prediction, [feature]

    class DroppingGcn(nn.Module):
        def forward(self, output, cnn_feature, batch):
            # Match flow_matching_evolution, which creates a fresh return dict.
            return {'py': cnn_feature.new_zeros((1, 128, 2))}

    cfg.det_conf_thresh = 0.1
    cfg.det_iou_thresh = 0.5
    cfg.det_max_det = 10
    cfg.per_class_nms = True
    cfg.use_nms_for_snake = True
    cfg.use_gt_det = True
    cfg.use_gt_det_train_only = True
    cfg.contour_init_method = 'octagon'
    cfg.yolo_num_classes = 25

    network = _bare_network()
    network.yolo = FakeYolo()
    network.use_swin_snake_feature = False
    network.use_p3_features = False
    network.cnn_proj = nn.Conv2d(89, 64, kernel_size=1, bias=False)
    network.use_extreme_refine = False
    network.freeze_snake = False
    network.gcn = DroppingGcn()
    network.eval()

    output = network(torch.zeros((1, 3, 16, 16)), batch=None)

    assert output['detection'].shape == (1, 1, 6)
    assert output['detection'][0, 0, 5].item() == pytest.approx(24.0)
    torch.testing.assert_close(output['py_cls'], torch.tensor([24]))
    torch.testing.assert_close(output['py_score'], torch.tensor([1.0]))
    assert 'yolo_preds' in output


def test_affine_visualization_keeps_boxes_and_polygons_in_input_frame(
        monkeypatch, tmp_path):
    from lib.visualizers import diffusion_one_sample as visualizer

    captured = {}

    def capture_draw(
            orig_img_bgr, det_b, pred_poly, gt_poly, save_path,
            init_poly=None, gt4_poly=None):
        captured.update({
            'image': orig_img_bgr,
            'det': det_b,
            'pred': pred_poly,
            'gt': gt_poly,
            'init': init_poly,
            'gt4': gt4_poly,
            'save_path': save_path,
        })

    monkeypatch.setattr(visualizer, 'draw_results', capture_draw)
    output = {
        'detection': torch.tensor([[[10.0, 20.0, 30.0, 40.0, 0.9, 2.0]]]),
        'py': torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
        'it_py': torch.tensor([[[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]]),
    }
    batch = {
        'orig_img': [torch.zeros((512, 512, 3), dtype=torch.uint8)],
        'meta': {'ct_num': torch.tensor([1])},
        'i_gt_py': torch.tensor([[[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]]),
        'i_gt_4py': torch.tensor([[[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]]]),
    }

    visualizer.save_affine_visualization(
        output=output,
        batch=batch,
        tag='contract',
        save_dir=str(tmp_path),
    )

    torch.testing.assert_close(captured['det'], output['detection'][0])
    np.testing.assert_allclose(captured['pred'], output['py'].numpy() * 4.0)
    np.testing.assert_allclose(captured['init'], output['it_py'].numpy() * 4.0)
    np.testing.assert_allclose(captured['gt'], batch['i_gt_py'][0].numpy() * 4.0)
    np.testing.assert_allclose(captured['gt4'], batch['i_gt_4py'][0].numpy() * 4.0)
