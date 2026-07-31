import csv
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml


_ROOT = Path(__file__).parents[1]
_OLD_CFG_FILE = os.environ.get('CFG_FILE')
os.environ['CFG_FILE'] = str(_ROOT / 'configs/sagittal_2d_pseudo3d_moonvit.yaml')
_OLD_ARGV = sys.argv[:]
_OLD_SIMPLEITK = sys.modules.get('SimpleITK')
_SIMPLEITK_WAS_ABSENT = 'SimpleITK' not in sys.modules
if _SIMPLEITK_WAS_ABSENT:
    sys.modules['SimpleITK'] = types.ModuleType('SimpleITK')
_OLD_DIFFUSERS = sys.modules.get('diffusers')
_DIFFUSERS_WAS_ABSENT = 'diffusers' not in sys.modules
if _DIFFUSERS_WAS_ABSENT:
    try:
        import diffusers  # noqa: F401
    except Exception:
        diffusers_stub = types.ModuleType('diffusers')

        class _SchedulerStub:
            pass

        diffusers_stub.DDPMScheduler = _SchedulerStub
        diffusers_stub.DDIMScheduler = _SchedulerStub
        sys.modules['diffusers'] = diffusers_stub
try:
    sys.argv[:] = [sys.argv[0]]
    from lib.config import cfg
    from lib.datasets.sagittal_2d_fixed.snake import Dataset
    from lib.networks.snake.ct_snake import LocateFeatAdapter, Network
    from lib.networks.diffusion.flow_matching_evolution import FlowMatchingEvolution
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
    if _DIFFUSERS_WAS_ABSENT:
        sys.modules.pop('diffusers', None)
    else:
        sys.modules['diffusers'] = _OLD_DIFFUSERS


_MANIFEST_FIELDS = ['split', 'case_id', 'slice_idx', 'image_path', 'mask_path']
_FEATURE_SHAPE = (1152, 2, 3)
_CACHE_META = {
    'grid_hw': np.asarray([2, 3], dtype=np.int32),
    'orig_hw': np.asarray([20, 30], dtype=np.int32),
    'resized_hw': np.asarray([28, 42], dtype=np.int32),
    'padded_hw': np.asarray([28, 42], dtype=np.int32),
    'pad': np.asarray([0, 0, 0, 0], dtype=np.int32),
    'scale': np.asarray([1.4], dtype=np.float32),
    'patch_size': np.asarray(14, dtype=np.int32),
    'input_size': np.asarray(42, dtype=np.int32),
    'normalization': np.asarray('moonvit_pretrained_rgb_mean_std'),
    'manifest_split': np.asarray('validation'),
    'feature_key': np.asarray('layer_18'),
}


@pytest.fixture
def sagittal_cfg(monkeypatch, tmp_path):
    checkpoint = tmp_path / 'moonvit-checkpoint'
    values = {
        'locate_feat_inject': True,
        'locate_feat_replace': False,
        'locate_feat_cache_root': str(tmp_path / 'cache'),
        'locate_feat_cache_dir': str(tmp_path / 'cache'),
        'sagittal_moonvit_cache_root': str(tmp_path / 'cache'),
        'sagittal_moonvit_feature_key': 'layer_18',
        'sagittal_moonvit_fusion_mode': 'center_neighbor_mean',
        'sagittal_moonvit_expected_input_size': 42,
        'sagittal_moonvit_expected_patch_size': 14,
        'sagittal_moonvit_expected_normalization': 'moonvit_pretrained_rgb_mean_std',
        'sagittal_moonvit_expected_checkpoint': str(checkpoint),
        'pseudo3d_input_mode': 'neighbors',
        'pseudo3d_mean': 0.0,
        'pseudo3d_std': 1.0,
        'pseudo3d_color_aug': False,
        'pseudo3d_lr_flip': False,
        'pseudo3d_random_crop': False,
        'locate_feat_dim': 2304,
        'locate_feat_keys': ['layer_18'],
    }
    for key, value in values.items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    monkeypatch.setattr(cfg.heads, 'ct_hm', 26)
    return tmp_path / 'cache', checkpoint


def _write_manifest(root, case_id='case_contract'):
    manifest = root / 'manifest.csv'
    rows = []
    for slice_idx in range(3):
        rows.append({
            'split': 'validation',
            'case_id': case_id,
            'slice_idx': str(slice_idx),
            'image_path': 'images/{}_x{:04d}.png'.format(case_id, slice_idx),
            'mask_path': 'masks/{}_x{:04d}.png'.format(case_id, slice_idx),
        })
    with manifest.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest, rows


def _cache_path(cache_root, case_id, slice_idx):
    path = cache_root / 'validation' / case_id / 'x{:04d}.npz'.format(slice_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_cache(cache_root, checkpoint, cache_case_id, cache_slice_idx, feature, **overrides):
    metadata = dict(_CACHE_META)
    metadata.update(overrides)
    payload = {
        'layer_18': np.asarray(feature, dtype=np.float16),
        'case_id': np.asarray(cache_case_id),
        'slice_idx': np.asarray(cache_slice_idx, dtype=np.int32),
        'checkpoint': np.asarray(str(checkpoint)),
    }
    payload.update(metadata)
    np.savez(_cache_path(cache_root, cache_case_id, cache_slice_idx), **payload)


def _make_dataset(tmp_path, cache_root, checkpoint, features=None):
    manifest, _ = _write_manifest(tmp_path)
    if features is None:
        features = [np.zeros(_FEATURE_SHAPE, dtype=np.float16) for _ in range(3)]
    for slice_idx, feature in enumerate(features):
        _write_cache(cache_root, checkpoint, 'case_contract', slice_idx, feature)
    return Dataset(str(manifest), str(tmp_path), 'val')


def test_center_neighbor_mean_fuses_three_layer_18_caches(sagittal_cfg, tmp_path):
    cache_root, checkpoint = sagittal_cfg
    values = np.arange(np.prod(_FEATURE_SHAPE), dtype=np.float32).reshape(_FEATURE_SHAPE)
    previous = (values % 17 - 5).astype(np.float16)
    center = (values * 0.01 + 10).astype(np.float16)
    following = (values % 13 + 3).astype(np.float16)
    dataset = _make_dataset(
        tmp_path,
        cache_root,
        checkpoint,
        features=[previous, center, following],
    )

    center_row = dataset.records[1]
    fused = dataset._load_locate_feature(dataset._neighbor_rows(center_row))
    expected_neighbors = (
        (previous.astype(np.float32) + following.astype(np.float32)) * 0.5
    ).astype(np.float16)
    expected = np.concatenate((center, expected_neighbors), axis=0)

    assert fused['locate_feat'].shape == (2304, 2, 3)
    assert fused['locate_feat'].dtype == np.float16
    np.testing.assert_array_equal(fused['locate_feat'], expected)


@pytest.mark.parametrize(
    'field,bad_value,error_pattern',
    [
        ('case_id', 'wrong_case', r'case_id=.*does not match'),
        ('slice_idx', 99, r'slice_idx=99 does not match'),
        ('input_size', np.asarray(21, dtype=np.int32), r'input_size=21 does not match'),
        ('scale', np.asarray([1.0], dtype=np.float32), r'scale=.*disagrees'),
        ('resized_hw', np.asarray([14, 42], dtype=np.int32), r'resized_hw=.*disagrees'),
        ('pad', np.asarray([1, 0, 0, 0], dtype=np.int32), r'pad=.*disagrees'),
    ],
)
def test_cache_identity_and_metadata_errors_fail_fast(
    sagittal_cfg, tmp_path, field, bad_value, error_pattern
):
    cache_root, checkpoint = sagittal_cfg
    manifest, _ = _write_manifest(tmp_path)
    for slice_idx in range(3):
        feature = np.full(_FEATURE_SHAPE, slice_idx + 1, dtype=np.float16)
        overrides = {field: bad_value} if slice_idx == 1 else {}
        _write_cache(
            cache_root,
            checkpoint,
            'case_contract',
            slice_idx,
            feature,
            **overrides,
        )
    dataset = Dataset(str(manifest), str(tmp_path), 'val')

    with pytest.raises(ValueError, match=error_pattern):
        dataset._load_locate_feature(dataset._neighbor_rows(dataset.records[1]))


def test_cache_rejects_nonminimal_patch_grid(sagittal_cfg, tmp_path):
    cache_root, checkpoint = sagittal_cfg
    manifest, _ = _write_manifest(tmp_path)
    for slice_idx in range(3):
        feature = np.zeros(_FEATURE_SHAPE, dtype=np.float16)
        overrides = {}
        if slice_idx == 1:
            feature = np.zeros((1152, 3, 3), dtype=np.float16)
            overrides = {
                'grid_hw': np.asarray([3, 3], dtype=np.int32),
                'padded_hw': np.asarray([42, 42], dtype=np.int32),
                'pad': np.asarray([0, 0, 0, 14], dtype=np.int32),
            }
        _write_cache(
            cache_root,
            checkpoint,
            'case_contract',
            slice_idx,
            feature,
            **overrides,
        )
    dataset = Dataset(str(manifest), str(tmp_path), 'val')

    with pytest.raises(ValueError, match=r'grid_hw=.*not the minimal patch grid'):
        dataset._load_locate_feature(dataset._neighbor_rows(dataset.records[1]))


def _locate_feature_batch():
    return {
        'locate_feat': torch.randn((1, 2304, 2, 2)),
        'meta': {
            'inv_trans_input': torch.eye(2, 3).unsqueeze(0),
            'orig_hw': torch.tensor([[8.0, 8.0]]),
            'flipped': torch.zeros((1, 1)),
        },
        'locate_feat_scale': torch.ones((1, 1)),
        'locate_feat_grid_hw': torch.tensor([[2, 2]], dtype=torch.int32),
        'locate_feat_patch_size': torch.full((1, 1), 14, dtype=torch.int32),
        'locate_feat_pad': torch.zeros((1, 4), dtype=torch.int32),
    }


def test_locate_feature_injection_is_identity_with_zero_init_adapter():
    network = Network.__new__(Network)
    nn.Module.__init__(network)
    network.down_ratio = 4.0
    network.locate_feat_inject = True
    network.locate_feat_adapter = LocateFeatAdapter(in_channels=2304, hidden_channels=64)
    cnn_feature = torch.randn((1, 64, 2, 2))

    output, stats = network.apply_locate_feature_injection(
        cnn_feature, _locate_feature_batch()
    )

    torch.testing.assert_close(output, cnn_feature)
    assert float(stats['locate_feat_residual_absmax']) == pytest.approx(0.0)
    assert float(stats['locate_feat_adapter_last_absmax']) == pytest.approx(0.0)


class _FakeYolo(nn.Module):
    def __init__(self):
        super().__init__()
        self.yaml = {'nc': 25}

    def forward(self, x):
        prediction = x.new_full((x.size(0), 29, 1), -20.0)
        prediction[:, 0, 0] = 8.0
        prediction[:, 1, 0] = 8.0
        prediction[:, 2, 0] = 8.0
        prediction[:, 3, 0] = 8.0
        prediction[:, 28, 0] = 20.0
        p2 = x.new_ones((x.size(0), 89, 4, 4))
        p3 = x.new_full((x.size(0), 89, 2, 2), 2.0)
        return prediction, [p2, p3]


class _RecordingGcn(nn.Module):
    def __init__(self, captured):
        super().__init__()
        self.captured = captured

    def forward(self, output, cnn_feature, batch):
        self.captured['gcn_feature'] = cnn_feature.detach().clone()
        return {'py': cnn_feature.new_zeros((1, 128, 2))}


def test_yolo_branch_separates_detector_feature_from_snake_feature(monkeypatch):
    monkeypatch.setattr(cfg, 'det_conf_thresh', 0.1)
    monkeypatch.setattr(cfg, 'det_iou_thresh', 0.5)
    monkeypatch.setattr(cfg, 'det_max_det', 10)
    monkeypatch.setattr(cfg, 'per_class_nms', True)
    monkeypatch.setattr(cfg, 'use_nms_for_snake', True)
    monkeypatch.setattr(cfg, 'use_gt_det', False)
    monkeypatch.setattr(cfg, 'use_gt_det_train_only', True)
    monkeypatch.setattr(cfg, 'contour_init_method', 'octagon')
    monkeypatch.setattr(cfg, 'yolo_num_classes', 25)

    captured = {}
    network = Network.__new__(Network)
    nn.Module.__init__(network)
    network.down_ratio = 4.0
    network.detector_backend = 'yolo'
    network.yolo = _FakeYolo()
    network.use_swin_snake_feature = False
    network.use_p3_features = False
    network.cnn_proj = nn.Conv2d(89, 64, kernel_size=1, bias=False)
    network.locate_feat_inject = False
    network.use_extreme_refine = False
    network.freeze_snake = False
    network.skip_diffusion_forward = False
    network.gcn = _RecordingGcn(captured)

    def fake_injection(self, detector_feature, batch=None):
        captured['detector_feature'] = detector_feature.detach().clone()
        return detector_feature + 3.0, {}

    def fake_attach(self, output, detector_feature, batch=None):
        captured['attach_feature'] = detector_feature.detach().clone()
        return output

    monkeypatch.setattr(
        network,
        'apply_locate_feature_injection',
        types.MethodType(fake_injection, network),
    )
    monkeypatch.setattr(
        network,
        'attach_extreme_prediction',
        types.MethodType(fake_attach, network),
    )
    network.eval()

    output = network(torch.zeros((1, 3, 16, 16)), batch=None)

    torch.testing.assert_close(captured['attach_feature'], captured['detector_feature'])
    torch.testing.assert_close(
        captured['gcn_feature'], captured['detector_feature'] + 3.0
    )
    torch.testing.assert_close(output['cnn_feature'], captured['gcn_feature'])
    assert not torch.equal(captured['attach_feature'], captured['gcn_feature'])
    assert output['detection'].shape == (1, 1, 6)
    assert 'yolo_preds' in output


def test_moonvit_yaml_selects_standard_no_detail_and_no_ffn_moe():
    config_path = _ROOT / 'configs/sagittal_2d_pseudo3d_moonvit.yaml'
    with config_path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    assert config['train']['dataset'] == 'SagittalPseudo3DMini'
    assert config['v4_1_final_head_type'] == 'standard'
    assert config['v4_1_use_detail_context'] is False
    assert config['v4_1_use_p3_features'] is True
    assert config['v4_10_use_dit_ffn_moe'] is False
    assert not any(key.startswith('v4_6_moe_') for key in config)


def test_flow_matching_v41_switches_standard_detail_and_moe(monkeypatch):
    from lib.networks.diffusion.flow_matching_evolution import FlowMatchingEvolution

    branch_flags = (
        'use_dit_v3_7',
        'use_dit_v3_1',
        'use_dit_v4_2',
        'use_dit_v4',
        'use_dit_v3_4',
        'use_dit_v3_6',
    )
    for key in branch_flags:
        monkeypatch.setattr(cfg, key, False, raising=False)
    monkeypatch.setattr(cfg, 'use_dit_v4_1', True, raising=False)
    monkeypatch.setattr(cfg, 'flow_2d_s_conditioning', False, raising=False)
    monkeypatch.setattr(cfg, 'use_iterative_refinement', False, raising=False)
    monkeypatch.setattr(cfg, 'v3_4_use_detail_context', False, raising=False)
    monkeypatch.setattr(cfg, 'v3_7_use_detail_context', False, raising=False)
    monkeypatch.setattr(cfg, 'v4_use_detail_context', False, raising=False)
    monkeypatch.setattr(cfg, 'v4_2_use_detail_context', False, raising=False)
    monkeypatch.setattr(cfg, 'v4_1_use_detail_context', False, raising=False)
    monkeypatch.setattr(cfg, 'v4_1_use_per_point_delta', False, raising=False)
    monkeypatch.setattr(cfg, 'v4_10_use_dit_ffn_moe', False, raising=False)
    monkeypatch.setattr(cfg, 'v4_1_final_head_type', 'standard', raising=False)

    standard = FlowMatchingEvolution(
        state_dim=32,
        feature_dim=8,
        num_points=4,
        dit_num_layers=1,
        dit_num_heads=2,
        dit_state_dim=32,
        ode_steps=1,
    )
    standard_denoiser = standard.denoiser
    assert standard_denoiser.final_head_type == 'standard'
    assert not standard_denoiser.use_moe_final_head
    assert not standard_denoiser.use_detail_context
    assert not hasattr(standard_denoiser, 'detail_local_proj')
    assert all(not layer.use_ffn_moe for layer in standard_denoiser.dit_layers)

    monkeypatch.setattr(cfg, 'v4_1_final_head_type', 'moe')
    moe = FlowMatchingEvolution(
        state_dim=32,
        feature_dim=8,
        num_points=4,
        dit_num_layers=1,
        dit_num_heads=2,
        dit_state_dim=32,
        ode_steps=1,
    )
    assert moe.denoiser.use_moe_final_head
    assert moe.denoiser.final_layer.__class__.__name__ == 'MoEFinalHead'

    monkeypatch.setattr(cfg, 'v4_1_final_head_type', 'standard')
    monkeypatch.setattr(cfg, 'v4_1_use_detail_context', True)
    monkeypatch.setattr(cfg, 'v4_1_detail_context_mode', 'normal')
    detail = FlowMatchingEvolution(
        state_dim=32,
        feature_dim=8,
        num_points=4,
        dit_num_layers=1,
        dit_num_heads=2,
        dit_state_dim=32,
        ode_steps=1,
    )
    assert detail.denoiser.use_detail_context
    assert detail.denoiser.detail_feature_dim == 24
    assert hasattr(detail.denoiser, 'detail_local_proj')
