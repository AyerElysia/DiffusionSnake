import csv
import os

import cv2
import numpy as np
import torch
import torch.utils.data as data

from lib.config import cfg
from lib.datasets.voc.snake import Dataset as VocDataset
from lib.utils.snake import snake_config, snake_voc_utils


_MANIFEST_SPLITS = {
    'train': 'training',
    'val': 'validation',
    'mini': 'training',
    'test': 'test',
}
_CACHE_SPLITS = dict(_MANIFEST_SPLITS)
_REQUIRED_COLUMNS = {'split', 'case_id', 'slice_idx', 'image_path', 'mask_path'}
_MOONVIT_FEATURE_KEY = 'layer_18'
_MOONVIT_LAYER_CHANNELS = 1152
_MOONVIT_FUSED_CHANNELS = 2304
_MOONVIT_INPUT_SIZE = 448
_MOONVIT_PATCH_SIZE = 14
_MOONVIT_NORMALIZATION = 'moonvit_pretrained_rgb_mean_std'


class Dataset(VocDataset):
    """Sagittal pseudo-3D slices with center-slice contour supervision."""

    @staticmethod
    def _config_value(names, default=None):
        for name in names:
            value = getattr(cfg, name, None)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return default

    @staticmethod
    def _canonical_int(value, description):
        if value is None:
            return None
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(
                '{} must contain one integer, got shape {}'.format(
                    description, array.shape
                )
            )
        scalar = array.reshape(-1)[0]
        try:
            parsed = int(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('{} must be an integer, got {!r}'.format(
                description, scalar
            )) from exc
        try:
            numeric = float(scalar)
        except (TypeError, ValueError, OverflowError):
            numeric = float(parsed)
        if not np.isfinite(numeric) or numeric != float(parsed) or parsed <= 0:
            raise ValueError('{} must be a positive integer, got {!r}'.format(
                description, scalar
            ))
        return parsed

    @classmethod
    def _canonical_size(cls, value, description):
        if value is None:
            return None
        array = np.asarray(value)
        if array.size == 2:
            values = [
                cls._canonical_int(item, description) for item in array.reshape(-1)
            ]
            if values[0] != values[1]:
                raise ValueError(
                    '{} must be square, got {}'.format(description, values)
                )
            return values[0]
        return cls._canonical_int(value, description)

    @staticmethod
    def _canonical_text(value, description, lower=False):
        if value is None:
            return None
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(
                '{} must contain one string, got shape {}'.format(
                    description, array.shape
                )
            )
        scalar = array.reshape(-1)[0]
        if isinstance(scalar, bytes):
            try:
                scalar = scalar.decode('utf-8')
            except UnicodeDecodeError as exc:
                raise ValueError('{} is not valid UTF-8'.format(description)) from exc
        text = str(scalar).strip()
        if not text:
            raise ValueError('{} must not be empty'.format(description))
        return text.lower() if lower else text

    @classmethod
    def _canonical_normalization(cls, value, description):
        text = cls._canonical_text(value, description, lower=True)
        aliases = {
            'moonvit_rgb_mean_0.5_std_0.5': _MOONVIT_NORMALIZATION,
        }
        return aliases.get(text, text)

    @classmethod
    def _canonical_checkpoint(cls, value, description):
        text = cls._canonical_text(value, description)
        if text is None:
            return None
        return os.path.realpath(os.path.abspath(os.path.expanduser(text)))

    def __init__(self, ann_file, data_root, split):
        data.Dataset.__init__(self)
        if split not in _MANIFEST_SPLITS:
            raise ValueError(
                'Unsupported catalog split {!r}; expected one of {}'.format(
                    split, sorted(_MANIFEST_SPLITS)
                )
            )

        self.ann_file = os.path.abspath(os.fspath(ann_file))
        self.data_root = os.path.abspath(os.fspath(data_root))
        self.split = split
        self.manifest_split = _MANIFEST_SPLITS[split]
        self.locate_feat_enabled = bool(
            getattr(cfg, 'locate_feat_inject', False)
            or getattr(cfg, 'locate_feat_replace', False)
        )
        if self.locate_feat_enabled:
            cache_root = self._config_value(
                ('sagittal_moonvit_cache_root', 'locate_feat_cache_root'),
                None,
            )
            if cache_root is None:
                raise ValueError(
                    'MoonViT sagittal cache is enabled, but cfg.sagittal_moonvit_cache_root '
                    'and cfg.locate_feat_cache_root are both empty'
                )
            self.locate_feat_cache_root = os.path.abspath(os.fspath(cache_root))
            self.sagittal_moonvit_cache_root = self.locate_feat_cache_root
            self.locate_feat_split_map = dict(_CACHE_SPLITS)
            self.locate_feat_split = self.locate_feat_split_map[split]
            self.locate_feat_cache_split = self.locate_feat_split
            self.sagittal_moonvit_cache_split = self.locate_feat_split
            self.sagittal_moonvit_cache_split_map = dict(self.locate_feat_split_map)

            feature_key = str(self._config_value(
                (
                    'sagittal_moonvit_feature_key',
                    'sagittal_moonvit_feat_key',
                    'locate_feat_feature_key',
                    'locate_feat_key',
                ),
                _MOONVIT_FEATURE_KEY,
            )).strip()
            # Accept 'layer_18' (single) or 'layer_18,layer_26' (multi, comma-separated).
            _VALID_BASE_KEY = _MOONVIT_FEATURE_KEY
            parsed_keys = [k.strip() for k in feature_key.split(',') if k.strip()]
            if not parsed_keys or parsed_keys[0] != _VALID_BASE_KEY:
                raise ValueError(
                    'Sagittal MoonViT feature keys must start with {!r}; got {!r}'.format(
                        _VALID_BASE_KEY, feature_key
                    )
                )
            self.locate_feat_key = parsed_keys[0]     # primary key (for compat checks)
            self.locate_feat_keys = parsed_keys        # all keys to load and concatenate
            self.sagittal_moonvit_feature_key = parsed_keys[0]

            fusion_mode = str(self._config_value(
                (
                    'sagittal_moonvit_fusion_mode',
                    'sagittal_moonvit_fusion',
                    'locate_feat_fusion_mode',
                ),
                'center_neighbor_mean',
            )).strip().lower()
            _VALID_FUSION_MODES = ('center_neighbor_mean', 'center_only')
            if fusion_mode not in _VALID_FUSION_MODES:
                raise ValueError(
                    'Unsupported sagittal MoonViT fusion mode {!r}; expected one of {}'.format(
                        fusion_mode, _VALID_FUSION_MODES
                    )
                )
            self.locate_feat_fusion_mode = fusion_mode
            self.sagittal_moonvit_fusion_mode = fusion_mode

            expected_input_size = self._config_value(
                (
                    'sagittal_moonvit_expected_input_size',
                    'sagittal_moonvit_input_size',
                    'locate_feat_expected_input_size',
                    'locate_feat_input_size',
                ),
                _MOONVIT_INPUT_SIZE,
            )
            expected_normalization = self._config_value(
                (
                    'sagittal_moonvit_expected_normalization',
                    'sagittal_moonvit_normalization',
                    'sagittal_moonvit_norm',
                    'locate_feat_expected_normalization',
                    'locate_feat_normalization',
                ),
                _MOONVIT_NORMALIZATION,
            )
            expected_patch_size = self._config_value(
                (
                    'sagittal_moonvit_expected_patch_size',
                    'sagittal_moonvit_patch_size',
                    'locate_feat_expected_patch_size',
                    'locate_feat_patch_size',
                ),
                _MOONVIT_PATCH_SIZE,
            )
            expected_checkpoint = self._config_value(
                (
                    'sagittal_moonvit_expected_checkpoint',
                    'sagittal_moonvit_checkpoint',
                    'sagittal_moonvit_checkpoint_path',
                    'locate_feat_expected_checkpoint',
                    'locate_feat_checkpoint',
                ),
                None,
            )
            self.locate_feat_expected_input_size = self._canonical_size(
                expected_input_size, 'expected input_size'
            )
            self.locate_feat_expected_normalization = self._canonical_normalization(
                expected_normalization, 'expected normalization'
            )
            self.locate_feat_expected_patch_size = self._canonical_int(
                expected_patch_size, 'expected patch_size'
            )
            self.locate_feat_expected_checkpoint = self._canonical_checkpoint(
                expected_checkpoint, 'expected checkpoint'
            )
            self.sagittal_moonvit_expected_input_size = self.locate_feat_expected_input_size
            self.sagittal_moonvit_expected_normalization = self.locate_feat_expected_normalization
            self.sagittal_moonvit_expected_patch_size = self.locate_feat_expected_patch_size
            self.sagittal_moonvit_expected_checkpoint = self.locate_feat_expected_checkpoint
            self.sagittal_moonvit_input_size = self.locate_feat_expected_input_size
            self.sagittal_moonvit_normalization = self.locate_feat_expected_normalization
            self.sagittal_moonvit_patch_size = self.locate_feat_expected_patch_size
            self.sagittal_moonvit_checkpoint = self.locate_feat_expected_checkpoint
            self.locate_feat_input_size = self.locate_feat_expected_input_size
            self.locate_feat_normalization = self.locate_feat_expected_normalization
            self.locate_feat_patch_size = self.locate_feat_expected_patch_size
            self.locate_feat_checkpoint = self.locate_feat_expected_checkpoint

        self.input_mode = str(getattr(cfg, 'pseudo3d_input_mode', 'neighbors')).strip().lower()
        if self.input_mode not in ('neighbors', 'center_repeat'):
            raise ValueError(
                "cfg.pseudo3d_input_mode must be 'neighbors' or 'center_repeat', got {!r}".format(
                    self.input_mode
                )
            )

        mean = float(getattr(cfg, 'pseudo3d_mean', 0.5))
        std = float(getattr(cfg, 'pseudo3d_std', 0.5))
        if std <= 0:
            raise ValueError('cfg.pseudo3d_std must be positive')
        self.mean = np.full((1, 1, 3), mean, dtype=np.float32)
        self.std = np.full((1, 1, 3), std, dtype=np.float32)
        self.color_aug = bool(getattr(cfg, 'pseudo3d_color_aug', False))
        self.lr_flip = bool(getattr(cfg, 'pseudo3d_lr_flip', False))
        self.random_crop = bool(getattr(cfg, 'pseudo3d_random_crop', True))

        if int(cfg.heads.ct_hm) != 26:
            raise ValueError(
                'Sagittal pseudo-3D requires cfg.heads.ct_hm=26, got {}'.format(
                    cfg.heads.ct_hm
                )
            )

        rows = self._read_manifest()
        self._case_rows = self._group_case_rows(rows)
        self.records = [row for case_rows in self._case_rows.values() for row in case_rows]
        if split == 'mini' and self.records:
            first_case = next(iter(self._case_rows))
            self.records = self._case_rows[first_case][:100]
        if not self.records:
            raise ValueError(
                'No rows for manifest split {!r} in {}'.format(
                    self.manifest_split, self.ann_file
                )
            )

        self.bbox_2d_manifest = self._configured_bbox_manifest()
        foreground_keys = self._read_foreground_keys(self.bbox_2d_manifest)
        self.foreground_flags = [
            (self.manifest_split, row['case_id'], row['slice_idx']) in foreground_keys
            for row in self.records
        ]

        # Prev-contour initialization: during training, with this probability the
        # adjacent slice's GT contour is used as the Snake init instead of the bbox
        # octagon.  Falls back to octagon when the adjacent slice has no matching
        # class or is a boundary (same slice repeated).
        self.prev_contour_init_prob = float(
            getattr(cfg, 'prev_contour_init_prob', 0.0)
        )

    def _resolve_path(self, path):
        path = os.fspath(path)
        if os.path.isabs(path):
            return os.path.normpath(path)
        return os.path.normpath(os.path.join(self.data_root, path))

    @staticmethod
    def _configured_bbox_manifest():
        train_cfg = getattr(cfg, 'train', None)
        for config_node in (train_cfg, cfg):
            if config_node is None:
                continue
            for name in ('bbox_2d_manifest', 'bbox_manifest'):
                value = getattr(config_node, name, None)
                if value is None:
                    continue
                path = os.fspath(value).strip()
                if path:
                    return os.path.abspath(os.path.expanduser(path))
        return None

    @staticmethod
    def _read_foreground_keys(manifest_path):
        if manifest_path is None:
            return set()
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                '2D bbox manifest not found: {}'.format(manifest_path)
            )

        required_columns = {'split', 'case_id', 'slice_idx', 'mask_pixel_count'}
        foreground_keys = set()
        with open(manifest_path, 'r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            missing = required_columns.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    '2D bbox manifest is missing columns: {}'.format(sorted(missing))
                )
            for row_number, raw in enumerate(reader, start=2):
                split = raw['split']
                case_id = raw['case_id']
                if not split or not case_id:
                    raise ValueError(
                        'Empty split or case_id at 2D bbox manifest row {}'.format(
                            row_number
                        )
                    )
                try:
                    slice_idx = int(raw['slice_idx'])
                    count_text = str(raw['mask_pixel_count'] or '').strip()
                    mask_pixel_count = int(count_text) if count_text else 0
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        'Invalid slice_idx or mask_pixel_count at 2D bbox manifest '
                        'row {}: slice_idx={!r}, mask_pixel_count={!r}'.format(
                            row_number, raw['slice_idx'], raw['mask_pixel_count']
                        )
                    ) from exc
                if mask_pixel_count > 0:
                    foreground_keys.add((split, case_id, slice_idx))
        return foreground_keys

    def _read_manifest(self):
        if not os.path.isfile(self.ann_file):
            raise FileNotFoundError('Slice manifest not found: {}'.format(self.ann_file))

        rows = []
        with open(self.ann_file, 'r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            missing = _REQUIRED_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    'Slice manifest is missing columns: {}'.format(sorted(missing))
                )
            for row_number, raw in enumerate(reader, start=2):
                if raw['split'] != self.manifest_split:
                    continue
                case_id = raw['case_id']
                if not case_id:
                    raise ValueError('Empty case_id at manifest row {}'.format(row_number))
                try:
                    slice_idx = int(raw['slice_idx'])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        'Invalid slice_idx at manifest row {}: {!r}'.format(
                            row_number, raw['slice_idx']
                        )
                    ) from exc
                rows.append({
                    'case_id': case_id,
                    'slice_idx': slice_idx,
                    'image_path': self._resolve_path(raw['image_path']),
                    'mask_path': self._resolve_path(raw['mask_path']),
                    'row_number': row_number,
                })
        return rows

    @staticmethod
    def _group_case_rows(rows):
        grouped = {}
        for row in rows:
            grouped.setdefault(row['case_id'], []).append(row)

        for case_id, case_rows in grouped.items():
            case_rows.sort(key=lambda row: row['slice_idx'])
            indices = [row['slice_idx'] for row in case_rows]
            if len(indices) != len(set(indices)):
                raise ValueError(
                    'Duplicate slice_idx values for case {!r}'.format(case_id)
                )
        return grouped

    def _neighbor_rows(self, center_row):
        case_rows = self._case_rows[center_row['case_id']]
        by_index = {row['slice_idx']: row for row in case_rows}
        center_idx = center_row['slice_idx']
        first_idx = case_rows[0]['slice_idx']
        last_idx = case_rows[-1]['slice_idx']

        if center_idx == first_idx:
            previous = center_row
        else:
            previous = by_index.get(center_idx - 1)
            if previous is None:
                raise ValueError(
                    'Missing slice {} before case {!r} slice {}'.format(
                        center_idx - 1, center_row['case_id'], center_idx
                    )
                )

        if center_idx == last_idx:
            following = center_row
        else:
            following = by_index.get(center_idx + 1)
            if following is None:
                raise ValueError(
                    'Missing slice {} after case {!r} slice {}'.format(
                        center_idx + 1, center_row['case_id'], center_idx
                    )
                )
        return previous, center_row, following

    def _sagittal_moonvit_cache_path(self, row):
        case_id = str(row['case_id'])
        if (
            not case_id
            or case_id in ('.', '..')
            or os.path.basename(case_id) != case_id
            or '/' in case_id
            or '\\' in case_id
        ):
            raise ValueError(
                'Unsafe case_id={!r}; MoonViT cache case IDs must be one path component'.format(
                    case_id
                )
            )
        slice_idx = int(row['slice_idx'])
        if slice_idx < 0:
            raise ValueError(
                'Invalid slice_idx={}; MoonViT cache filenames require non-negative indices'.format(
                    slice_idx
                )
            )
        return os.path.join(
            self.locate_feat_cache_root,
            self.locate_feat_split,
            case_id,
            'x{:04d}.npz'.format(slice_idx),
        )

    @staticmethod
    def _cache_required(npz, path, key):
        if key not in npz.files:
            raise ValueError(
                'MoonViT cache {} is missing required metadata {!r}; available keys={}'.format(
                    path, key, list(npz.files)
                )
            )
        return npz[key]

    @classmethod
    def _cache_integer(cls, value, path, key, minimum=0):
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(
                'MoonViT cache {} metadata {} must be scalar, got shape {}'.format(
                    path, key, array.shape
                )
            )
        scalar = array.reshape(-1)[0]
        try:
            parsed = int(scalar)
            numeric = float(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                'MoonViT cache {} metadata {} must be an integer, got {!r}'.format(
                    path, key, scalar
                )
            ) from exc
        if not np.isfinite(numeric) or numeric != float(parsed) or parsed < minimum:
            raise ValueError(
                'MoonViT cache {} metadata {} must be an integer >= {}, got {!r}'.format(
                    path, key, minimum, scalar
                )
            )
        return parsed

    @classmethod
    def _cache_integer_vector(cls, npz, path, key, size, minimum=0):
        value = np.asarray(cls._cache_required(npz, path, key))
        if value.size != size:
            raise ValueError(
                'MoonViT cache {} metadata {} must contain {} integers, got shape {}'.format(
                    path, key, size, value.shape
                )
            )
        parsed = [
            cls._cache_integer(item, path, key, minimum=minimum)
            for item in value.reshape(-1)
        ]
        return np.asarray(parsed, dtype=np.int32)

    @staticmethod
    def _cache_text(npz, path, key, lower=False):
        value = np.asarray(Dataset._cache_required(npz, path, key))
        if value.size != 1:
            raise ValueError(
                'MoonViT cache {} metadata {} must be one string, got shape {}'.format(
                    path, key, value.shape
                )
            )
        scalar = value.reshape(-1)[0]
        if isinstance(scalar, bytes):
            try:
                scalar = scalar.decode('utf-8')
            except UnicodeDecodeError as exc:
                raise ValueError(
                    'MoonViT cache {} metadata {} is not valid UTF-8'.format(path, key)
                ) from exc
        text = str(scalar).strip()
        if not text:
            raise ValueError(
                'MoonViT cache {} metadata {} must not be empty'.format(path, key)
            )
        return text.lower() if lower else text

    @classmethod
    def _cache_normalization(cls, npz, path):
        value = cls._cache_text(npz, path, 'normalization', lower=True)
        aliases = {
            'moonvit_rgb_mean_0.5_std_0.5': _MOONVIT_NORMALIZATION,
        }
        return aliases.get(value, value)

    @classmethod
    def _cache_checkpoint(cls, npz, path):
        checkpoint = cls._cache_text(npz, path, 'checkpoint')
        return os.path.realpath(os.path.abspath(os.path.expanduser(checkpoint)))

    @staticmethod
    def _cache_scale(npz, path):
        value = np.asarray(Dataset._cache_required(npz, path, 'scale'))
        if value.size != 1:
            raise ValueError(
                'MoonViT cache {} metadata scale must be scalar, got shape {}'.format(
                    path, value.shape
                )
            )
        try:
            scale = float(value.reshape(-1)[0])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                'MoonViT cache {} metadata scale must be finite and positive, got {!r}'.format(
                    path, value
                )
            ) from exc
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                'MoonViT cache {} metadata scale must be finite and positive, got {!r}'.format(
                    path, scale
                )
            )
        return np.asarray([scale], dtype=np.float32)

    def _load_sagittal_moonvit_file(self, row):
        path = self._sagittal_moonvit_cache_path(row)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                'MoonViT cache missing for case_id={!r}, slice_idx={}; expected {}'.format(
                    row['case_id'], row['slice_idx'], path
                )
            )

        with np.load(path, allow_pickle=False) as npz:
            # Load and concatenate all configured layer keys along the channel dim.
            # For a single key the output is [C, H, W]; for two keys [2C, H, W].
            layer_tensors = []
            for feature_key in self.locate_feat_keys:
                if feature_key not in npz.files:
                    raise ValueError(
                        'MoonViT cache {} is missing feature key {!r}; available keys={}'.format(
                            path, feature_key, list(npz.files)
                        )
                    )
                layer = np.asarray(npz[feature_key])
                if layer.dtype != np.float16:
                    raise ValueError(
                        'MoonViT cache {} feature {!r} has dtype {}; expected float16'.format(
                            path, feature_key, layer.dtype
                        )
                    )
                if layer.ndim != 3:
                    raise ValueError(
                        'MoonViT cache {} feature {!r} must be [C,H,W], got shape {}'.format(
                            path, feature_key, layer.shape
                        )
                    )
                if layer.shape[0] != _MOONVIT_LAYER_CHANNELS:
                    raise ValueError(
                        'MoonViT cache {} feature {!r} must have {} channels, got shape {}'.format(
                            path, feature_key, _MOONVIT_LAYER_CHANNELS, layer.shape
                        )
                    )
                layer_tensors.append(layer)
            feature = np.concatenate(layer_tensors, axis=0) if len(layer_tensors) > 1 else layer_tensors[0]
            feature_key = self.locate_feat_key  # primary key (for grid metadata checks)

            grid_hw = self._cache_integer_vector(npz, path, 'grid_hw', 2, minimum=1)
            if tuple(layer_tensors[0].shape[-2:]) != tuple(grid_hw.tolist()):
                raise ValueError(
                    'MoonViT cache {} feature {!r} spatial shape {} disagrees with grid_hw={}'.format(
                        path, feature_key, tuple(layer_tensors[0].shape[-2:]), grid_hw.tolist()
                    )
                )
            orig_hw = self._cache_integer_vector(npz, path, 'orig_hw', 2, minimum=1)
            resized_hw = self._cache_integer_vector(npz, path, 'resized_hw', 2, minimum=1)
            padded_hw = self._cache_integer_vector(npz, path, 'padded_hw', 2, minimum=1)
            pad = self._cache_integer_vector(npz, path, 'pad', 4, minimum=0)
            patch_size = self._cache_integer(
                self._cache_required(npz, path, 'patch_size'), path, 'patch_size', minimum=1
            )
            input_size = self._cache_integer(
                self._cache_required(npz, path, 'input_size'), path, 'input_size', minimum=1
            )
            scale = self._cache_scale(npz, path)
            normalization = self._cache_normalization(npz, path)
            checkpoint = self._cache_checkpoint(npz, path)
            case_id = self._cache_text(npz, path, 'case_id')
            slice_idx = self._cache_integer(
                self._cache_required(npz, path, 'slice_idx'), path, 'slice_idx', minimum=0
            )

            if case_id != str(row['case_id']):
                raise ValueError(
                    'MoonViT cache {} case_id={!r} does not match manifest case_id={!r}'.format(
                        path, case_id, row['case_id']
                    )
                )
            if slice_idx != int(row['slice_idx']):
                raise ValueError(
                    'MoonViT cache {} slice_idx={} does not match manifest slice_idx={}'.format(
                        path, slice_idx, row['slice_idx']
                    )
                )
            if 'manifest_split' in npz.files:
                manifest_split = self._cache_text(npz, path, 'manifest_split')
                if manifest_split != self.manifest_split:
                    raise ValueError(
                        'MoonViT cache {} manifest_split={!r} does not match {!r}'.format(
                            path, manifest_split, self.manifest_split
                        )
                    )
            if 'feature_key' in npz.files:
                stored_key = self._cache_text(npz, path, 'feature_key')
                if stored_key != feature_key:
                    raise ValueError(
                        'MoonViT cache {} feature_key={!r} does not match {!r}'.format(
                            path, stored_key, feature_key
                        )
                    )

            if input_size != self.locate_feat_expected_input_size:
                raise ValueError(
                    'MoonViT cache {} input_size={} does not match expected input_size={}'.format(
                        path, input_size, self.locate_feat_expected_input_size
                    )
                )
            if patch_size != self.locate_feat_expected_patch_size:
                raise ValueError(
                    'MoonViT cache {} patch_size={} does not match expected patch_size={}'.format(
                        path, patch_size, self.locate_feat_expected_patch_size
                    )
                )
            if normalization != self.locate_feat_expected_normalization:
                raise ValueError(
                    'MoonViT cache {} normalization={!r} does not match expected normalization={!r}'.format(
                        path, normalization, self.locate_feat_expected_normalization
                    )
                )
            if (
                self.locate_feat_expected_checkpoint is not None
                and checkpoint != self.locate_feat_expected_checkpoint
            ):
                raise ValueError(
                    'MoonViT cache {} checkpoint={!r} does not match expected checkpoint={!r}'.format(
                        path, checkpoint, self.locate_feat_expected_checkpoint
                    )
                )

            expected_scale = float(input_size) / float(max(orig_hw.tolist()))
            if not np.isclose(float(scale[0]), expected_scale, rtol=1e-6, atol=1e-6):
                raise ValueError(
                    'MoonViT cache {} scale={} disagrees with orig_hw={} and input_size={}; expected {}'.format(
                        path, float(scale[0]), orig_hw.tolist(), input_size, expected_scale
                    )
                )
            expected_resized = np.maximum(
                np.rint(orig_hw.astype(np.float64) * expected_scale).astype(np.int32),
                1,
            )
            if not np.array_equal(resized_hw, expected_resized):
                raise ValueError(
                    'MoonViT cache {} resized_hw={} disagrees with orig_hw={}, scale={}; expected {}'.format(
                        path,
                        resized_hw.tolist(),
                        orig_hw.tolist(),
                        expected_scale,
                        expected_resized.tolist(),
                    )
                )

            expected_grid = np.ceil(
                resized_hw.astype(np.float64) / float(patch_size)
            ).astype(np.int32)
            if not np.array_equal(grid_hw, expected_grid):
                raise ValueError(
                    'MoonViT cache {} grid_hw={} is not the minimal patch grid for resized_hw={} and patch_size={}; expected {}'.format(
                        path,
                        grid_hw.tolist(),
                        resized_hw.tolist(),
                        patch_size,
                        expected_grid.tolist(),
                    )
                )
            expected_padded = expected_grid.astype(np.int64) * int(patch_size)
            if not np.array_equal(padded_hw, expected_padded.astype(np.int32)):
                raise ValueError(
                    'MoonViT cache {} padded_hw={} does not equal minimal grid*patch_size={}'.format(
                        path, padded_hw.tolist(), expected_padded.tolist()
                    )
                )
            expected_pad = np.asarray(
                [
                    0,
                    0,
                    int(padded_hw[1]) - int(resized_hw[1]),
                    int(padded_hw[0]) - int(resized_hw[0]),
                ],
                dtype=np.int32,
            )
            if np.any(expected_pad < 0) or not np.array_equal(pad, expected_pad):
                raise ValueError(
                    'MoonViT cache {} pad={} disagrees with resized_hw={} and padded_hw={}; expected {}'.format(
                        path,
                        pad.tolist(),
                        resized_hw.tolist(),
                        padded_hw.tolist(),
                        expected_pad.tolist(),
                    )
                )

            return {
                'feature': np.ascontiguousarray(feature),
                'grid_hw': np.asarray(grid_hw, dtype=np.int32),
                'orig_hw': np.asarray(orig_hw, dtype=np.int32),
                'resized_hw': np.asarray(resized_hw, dtype=np.int32),
                'padded_hw': np.asarray(padded_hw, dtype=np.int32),
                'pad': np.asarray(pad, dtype=np.int32),
                'scale': scale,
                'patch_size': np.asarray([patch_size], dtype=np.int32),
                'input_size': input_size,
                'normalization': normalization,
                'checkpoint': checkpoint,
                'path': path,
            }

    def _load_locate_feature(self, neighbor_rows, expected_orig_hw=None):
        if not self.locate_feat_enabled:
            return {}
        if len(neighbor_rows) != 3:
            raise ValueError(
                'Sagittal MoonViT fusion requires prev/center/next rows, got {}'.format(
                    len(neighbor_rows)
                )
            )
        center_row = neighbor_rows[1]
        case_id = str(center_row['case_id'])
        for row in neighbor_rows:
            if str(row['case_id']) != case_id:
                raise ValueError(
                    'Sagittal MoonViT neighbor rows must be case-local; got case IDs {}'.format(
                        [item['case_id'] for item in neighbor_rows]
                    )
                )

        # center_only: load and validate only the center slice, skip neighbor fusion.
        if self.locate_feat_fusion_mode == 'center_only':
            center_entry = self._load_sagittal_moonvit_file(center_row)
            if expected_orig_hw is not None:
                expected_orig = tuple(int(value) for value in expected_orig_hw)
                if tuple(center_entry['orig_hw'].tolist()) != expected_orig:
                    raise ValueError(
                        'MoonViT cache {} orig_hw={} does not match center image shape={}'.format(
                            center_entry['path'], center_entry['orig_hw'].tolist(), expected_orig
                        )
                    )
            feature = center_entry['feature'].astype(np.float16, copy=False)
            return {
                'locate_feat': np.ascontiguousarray(feature),
                'locate_feat_grid_hw': center_entry['grid_hw'].copy(),
                'locate_feat_orig_hw': center_entry['orig_hw'].copy(),
                'locate_feat_resized_hw': center_entry['resized_hw'].copy(),
                'locate_feat_padded_hw': center_entry['padded_hw'].copy(),
                'locate_feat_pad': center_entry['pad'].copy(),
                'locate_feat_scale': center_entry['scale'].copy(),
                'locate_feat_patch_size': center_entry['patch_size'].copy(),
                'locate_feat_path': center_entry['path'],
            }

        entries = [self._load_sagittal_moonvit_file(row) for row in neighbor_rows]
        center_entry = entries[1]
        if expected_orig_hw is not None:
            expected_orig = tuple(int(value) for value in expected_orig_hw)
            if tuple(center_entry['orig_hw'].tolist()) != expected_orig:
                raise ValueError(
                    'MoonViT cache {} orig_hw={} does not match center image shape={}'.format(
                        center_entry['path'], center_entry['orig_hw'].tolist(), expected_orig
                    )
                )

        for entry in entries:
            if tuple(entry['feature'].shape[-2:]) != tuple(center_entry['feature'].shape[-2:]):
                raise ValueError(
                    'MoonViT neighbor feature spatial sizes differ for case {!r}: {} has {}, '
                    'center {} has {}'.format(
                        case_id,
                        entry['path'],
                        tuple(entry['feature'].shape[-2:]),
                        center_entry['path'],
                        tuple(center_entry['feature'].shape[-2:]),
                    )
                )
            for key in ('input_size', 'patch_size', 'normalization', 'checkpoint'):
                if entry[key] != center_entry[key]:
                    raise ValueError(
                        'MoonViT neighbor metadata {} mismatch for case {!r}: {} has {!r}, '
                        'center {} has {!r}'.format(
                            key,
                            case_id,
                            entry['path'],
                            entry[key],
                            center_entry['path'],
                            center_entry[key],
                        )
                    )
            for key in ('grid_hw', 'orig_hw', 'resized_hw', 'padded_hw', 'pad'):
                if not np.array_equal(entry[key], center_entry[key]):
                    raise ValueError(
                        'MoonViT neighbor metadata {} mismatch for case {!r}: {} has {}, '
                        'center {} has {}'.format(
                            key,
                            case_id,
                            entry['path'],
                            entry[key].tolist(),
                            center_entry['path'],
                            center_entry[key].tolist(),
                        )
                    )
            if not np.allclose(entry['scale'], center_entry['scale'], rtol=0.0, atol=1e-7):
                raise ValueError(
                    'MoonViT neighbor metadata scale mismatch for case {!r}: {} has {}, '
                    'center {} has {}'.format(
                        case_id,
                        entry['path'],
                        entry['scale'].tolist(),
                        center_entry['path'],
                        center_entry['scale'].tolist(),
                    )
                )

        previous_feature = entries[0]['feature'].astype(np.float32)
        following_feature = entries[2]['feature'].astype(np.float32)
        neighbor_mean = ((previous_feature + following_feature) * 0.5).astype(np.float16)
        feature = np.concatenate(
            (entries[1]['feature'], neighbor_mean), axis=0
        ).astype(np.float16, copy=False)
        if feature.shape[0] != _MOONVIT_FUSED_CHANNELS:
            raise ValueError(
                'Sagittal MoonViT fusion produced {} channels, expected {}'.format(
                    feature.shape[0], _MOONVIT_FUSED_CHANNELS
                )
            )

        return {
            'locate_feat': np.ascontiguousarray(feature),
            'locate_feat_grid_hw': center_entry['grid_hw'].copy(),
            'locate_feat_orig_hw': center_entry['orig_hw'].copy(),
            'locate_feat_resized_hw': center_entry['resized_hw'].copy(),
            'locate_feat_padded_hw': center_entry['padded_hw'].copy(),
            'locate_feat_pad': center_entry['pad'].copy(),
            'locate_feat_scale': center_entry['scale'].copy(),
            'locate_feat_patch_size': center_entry['patch_size'].copy(),
            'locate_feat_path': center_entry['path'],
        }

    @staticmethod
    def _read_grayscale_image(path):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError('Image not found or cannot be read: {}'.format(path))
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        elif image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        if image.ndim != 2:
            raise ValueError('Expected a 2D grayscale image at {}, got {}'.format(path, image.shape))
        if image.dtype != np.uint8:
            raise ValueError('Expected uint8 image at {}, got {}'.format(path, image.dtype))
        return image

    @staticmethod
    def _read_mask(path, expected_shape):
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError('Mask not found or cannot be read: {}'.format(path))
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        if mask.ndim != 2:
            raise ValueError('Expected a 2D label mask at {}, got {}'.format(path, mask.shape))
        if mask.dtype != np.uint16:
            raise ValueError('Expected uint16 label mask at {}, got {}'.format(path, mask.dtype))
        if mask.shape != expected_shape:
            raise ValueError(
                'Image/mask shape mismatch for {}: image {}, mask {}'.format(
                    path, expected_shape, mask.shape
                )
            )
        labels = np.unique(mask)
        invalid = labels[(labels < 0) | (labels > 25)]
        if invalid.size:
            raise ValueError(
                'Mask labels must be in [0, 25] at {}; got {}'.format(
                    path, invalid.tolist()
                )
            )
        return mask

    @staticmethod
    def _mask_to_instances(mask):
        component_mode = str(
            getattr(cfg, 'sagittal_component_mode', 'largest')
        ).strip().lower()
        if component_mode not in ('largest', 'significant'):
            raise ValueError(
                "cfg.sagittal_component_mode must be 'largest' or "
                "'significant', got {!r}".format(component_mode)
            )
        max_components_per_class = max(
            int(getattr(cfg, 'sagittal_max_components_per_class', 4)), 1
        )
        max_instances_per_slice = max(
            int(getattr(cfg, 'sagittal_max_instances_per_slice', 32)), 1
        )
        min_component_area = max(
            float(getattr(cfg, 'sagittal_min_component_area_raw', 2.0)), 0.0
        )
        candidates = []
        for label in np.unique(mask):
            label = int(label)
            if label == 0:
                continue
            binary = np.asarray(mask == label, dtype=np.uint8)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            valid_contours = [c for c in contours if len(c.reshape(-1, 2)) >= 4]
            if not valid_contours:
                continue
            valid_contours.sort(key=cv2.contourArea, reverse=True)
            if component_mode == 'largest':
                valid_contours = valid_contours[:1]
            else:
                valid_contours = [
                    contour
                    for contour in valid_contours[:max_components_per_class]
                    if float(cv2.contourArea(contour)) >= min_component_area
                ]
            for contour in valid_contours:
                candidates.append((
                    float(cv2.contourArea(contour)),
                    label,
                    contour.reshape(-1, 2).astype(np.float32),
                ))

        # A single polygon cannot represent disconnected anatomy.  Treat each
        # retained component as an instance of the same vertebra class.  The
        # global cap prevents noisy masks from recreating the historical
        # 51-contour OOM case.  Full-dataset audit for top-4/area>=2/global-32:
        # 99.52% train and 99.73% validation foreground-pixel coverage.
        candidates.sort(key=lambda item: (-item[0], item[1]))
        candidates = candidates[:max_instances_per_slice]
        candidates.sort(key=lambda item: (item[1], -item[0]))
        instance_polys = []
        cls_ids = []
        for _, label, poly in candidates:
            instance_polys.append([poly.astype(np.float32)])
            cls_ids.append(label)
        return instance_polys, cls_ids

    def get_valid_polys(self, instance_polys, inp_out_hw):
        """Same as VocDataset.get_valid_polys but with a configurable area floor.

        The inherited implementation hard-codes ``Polygon(poly).area > 5`` in the
        1/4-resolution output space.  Sagittal vertebrae are small: a vertebra
        spanning ~100 px^2 in a 616x473 slice shrinks to ~3 px^2 at 128x128, so
        the default threshold silently drops entire slices worth of GT (measured:
        26 of 209 foreground val slices lose every instance).  Lowering the floor
        keeps them.
        """
        from shapely.geometry import Polygon

        min_area = float(getattr(cfg, 'min_poly_area_output', 5.0))
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            for poly in instance:
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
            polys = [poly for poly in instance if Polygon(poly).area > min_area]
            polys = snake_voc_utils.get_cw_polys(polys)
            polys = [
                poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])]
                for poly in polys
            ]
            instance_polys_.append(polys)
        return instance_polys_

    def __getitem__(self, index):
        row = self.records[index]
        neighbor_rows = self._neighbor_rows(row)
        neighbor_images = [
            self._read_grayscale_image(neighbor['image_path'])
            for neighbor in neighbor_rows
        ]
        center_image = neighbor_images[1]
        for neighbor, image in zip(neighbor_rows, neighbor_images):
            if image.shape != center_image.shape:
                raise ValueError(
                    'Inconsistent image shape in case {!r}: slice {} is {}, center is {}'.format(
                        row['case_id'], neighbor['slice_idx'], image.shape, center_image.shape
                    )
                )

        locate_feat = {}
        if self.locate_feat_enabled:
            locate_feat = self._load_locate_feature(
                neighbor_rows, expected_orig_hw=center_image.shape
            )

        if self.input_mode == 'center_repeat':
            image = np.repeat(center_image[:, :, None], 3, axis=2)
        else:
            image = np.stack(neighbor_images, axis=2)

        height, width = center_image.shape
        mask = self._read_mask(row['mask_path'], center_image.shape)
        instance_polys, cls_ids = self._mask_to_instances(mask)

        # --- prev-contour init: collect adjacent-slice raw polys (pre-transform) ---
        use_prev_contour = (
            self.split == 'train'
            and self.prev_contour_init_prob > 0.0
            and np.random.rand() < self.prev_contour_init_prob
        )
        adj_raw_polys_by_class = {}  # {cls_id (int): np.ndarray [N,2]}
        if use_prev_contour:
            center_slice_idx = int(row['slice_idx'])
            for adj_row in [neighbor_rows[0], neighbor_rows[2]]:
                if int(adj_row['slice_idx']) != center_slice_idx:
                    adj_mask = self._read_mask(adj_row['mask_path'], center_image.shape)
                    for label in np.unique(adj_mask):
                        label = int(label)
                        if label == 0:
                            continue
                        binary = np.asarray(adj_mask == label, dtype=np.uint8)
                        contours, _ = cv2.findContours(
                            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                        )
                        if contours:
                            # keep largest contour per class
                            largest = max(contours, key=cv2.contourArea)
                            poly = largest.reshape(-1, 2).astype(np.float32)
                            if len(poly) >= 4:
                                adj_raw_polys_by_class[label] = poly
                    break  # found a valid adjacent slice, stop
            if not adj_raw_polys_by_class:
                use_prev_contour = False

        augmented, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            snake_voc_utils.augment(
                image, self.split,
                snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
                self.mean, self.std, instance_polys,
                color_aug=self.color_aug,
                lr_flip=self.lr_flip,
                random_crop=self.random_crop,
            )
        orig_img = np.repeat(augmented[:, :, 1:2], 3, axis=2)
        instance_polys = self.transform_original_data(
            instance_polys, flipped, width, trans_output, inp_out_hw
        )
        instance_polys = self.get_valid_polys(instance_polys, inp_out_hw)
        extreme_points = self.get_extreme_points(instance_polys)

        # Apply the same spatial transform to adjacent-slice polys.
        adj_polys_by_class = {}  # {cls_id: transformed poly [N,2]}
        if use_prev_contour:
            for cls_id, raw_poly in adj_raw_polys_by_class.items():
                transformed = self.transform_original_data(
                    [[raw_poly]], flipped, width, trans_output, inp_out_hw
                )
                transformed = self.get_valid_polys(transformed, inp_out_hw)
                if transformed and transformed[0]:
                    adj_polys_by_class[cls_id] = transformed[0][0]
            if not adj_polys_by_class:
                use_prev_contour = False

        output_h, output_w = inp_out_hw[2:]
        ct_hm = np.zeros([26, output_h, output_w], dtype=np.float32)
        wh = []
        ct_cls = []
        ct_ind = []
        yolo_xywh = []
        yolo_cls = []

        i_it_4pys = []
        c_it_4pys = []
        i_gt_4pys = []
        c_gt_4pys = []
        i_it_pys = []
        c_it_pys = []
        i_gt_pys = []
        c_gt_pys = []

        for cls_id, instance, instance_points in zip(
                cls_ids, instance_polys, extreme_points):
            for poly, extreme_point in zip(instance, instance_points):
                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue
                bbox = [x_min, y_min, x_max, y_max]

                self.prepare_detection(
                    bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind
                )
                input_h, input_w = inp_out_hw[:2]
                scale_x = float(input_w) / float(output_w)
                scale_y = float(input_h) / float(output_h)
                yolo_xywh.append([
                    ((x_min + x_max) / 2.0 * scale_x) / input_w,
                    ((y_min + y_max) / 2.0 * scale_y) / input_h,
                    ((x_max - x_min) * scale_x) / input_w,
                    ((y_max - y_min) * scale_y) / input_h,
                ])
                yolo_cls.append(float(cls_id - 1))
                self.prepare_init(
                    bbox, extreme_point,
                    i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys,
                    output_h, output_w,
                )
                # Use adjacent-slice contour as init when available, else octagon.
                if use_prev_contour and cls_id in adj_polys_by_class:
                    adj_poly = adj_polys_by_class[cls_id]
                    ep_x_min = float(np.min(extreme_point[:, 0]))
                    ep_y_min = float(np.min(extreme_point[:, 1]))
                    ep_x_max = float(np.max(extreme_point[:, 0]))
                    ep_y_max = float(np.max(extreme_point[:, 1]))
                    num_points = self.compute_adaptive_points(
                        [ep_x_min, ep_y_min, ep_x_max, ep_y_max]
                    )
                    img_init_poly = snake_voc_utils.uniformsample(adj_poly, num_points)
                    can_init_poly = snake_voc_utils.img_poly_to_can_poly(
                        img_init_poly, ep_x_min, ep_y_min, ep_x_max, ep_y_max
                    )
                    img_gt_poly = snake_voc_utils.uniformsample(poly, len(poly) * num_points)
                    tt_idx = np.argmin(
                        np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1)
                    )
                    img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]
                    can_gt_poly = snake_voc_utils.img_poly_to_can_poly(
                        img_gt_poly, ep_x_min, ep_y_min, ep_x_max, ep_y_max
                    )
                    i_it_pys.append(img_init_poly)
                    c_it_pys.append(can_init_poly)
                    i_gt_pys.append(img_gt_poly)
                    c_gt_pys.append(can_gt_poly)
                else:
                    self.prepare_evolution(
                        poly, extreme_point, i_it_4pys[-1],
                        i_it_pys, c_it_pys, i_gt_pys, c_gt_pys,
                    )

        ret = {
            'inp': inp,
            'orig_img': orig_img,
            'img_path': row['image_path'],
            'ct_hm': ct_hm,
            'wh': wh,
            'ct_cls': ct_cls,
            'ct_ind': ct_ind,
            'i_it_4py': i_it_4pys,
            'c_it_4py': c_it_4pys,
            'i_gt_4py': i_gt_4pys,
            'c_gt_4py': c_gt_4pys,
            'i_it_py': i_it_pys,
            'c_it_py': c_it_pys,
            'i_gt_py': i_gt_pys,
            'c_gt_py': c_gt_pys,
        }
        if yolo_xywh:
            ret['bboxes'] = torch.tensor(yolo_xywh, dtype=torch.float32)
            ret['cls'] = torch.tensor(yolo_cls, dtype=torch.float32).unsqueeze(1)
            ret['batch_idx'] = torch.zeros((len(yolo_xywh), 1), dtype=torch.float32)
        else:
            ret['bboxes'] = torch.zeros((0, 4), dtype=torch.float32)
            ret['cls'] = torch.zeros((0, 1), dtype=torch.float32)
            ret['batch_idx'] = torch.zeros((0, 1), dtype=torch.float32)
        if locate_feat:
            ret.update(locate_feat)

        inv_trans_input = cv2.invertAffineTransform(trans_input).astype(np.float32)
        ret['meta'] = {
            'center': center,
            'scale': scale,
            'ct_num': len(ct_ind),
            'trans_input': trans_input.astype(np.float32),
            'inv_trans_input': inv_trans_input,
            'flipped': np.asarray([1 if flipped else 0], dtype=np.float32),
            'orig_hw': np.asarray([height, width], dtype=np.float32),
            'inp_out_hw': np.asarray(inp_out_hw, dtype=np.float32),
            'img_id': int(index),
            'case_id': row['case_id'],
            'slice_idx': int(row['slice_idx']),
            'neighbor_indices': np.asarray(
                [neighbor['slice_idx'] for neighbor in neighbor_rows],
                dtype=np.int64,
            ),
        }
        return ret

    def __len__(self):
        return len(self.records)
