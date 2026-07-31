import csv
import json
import os

import cv2
import numpy as np

from lib.config import cfg as global_cfg
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.utils.snake import snake_config


_REQUIRED_COLUMNS = {'split', 'case_id', 'slice_idx', 'image_path', 'mask_path'}
_SCORE_THRESHOLD = 1e-4
_NUM_CLASSES = 25


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _flatten_polygons(value):
    """Return contours from the common nested polygon representations."""
    if value is None:
        return []
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray) and value.dtype != object:
        if value.size == 0:
            return []
        if value.ndim == 2 and value.shape[-1] == 2:
            return [value]
        if value.ndim >= 3 and value.shape[-1] == 2:
            return [item for item in value.reshape((-1, value.shape[-2], 2))]
    if isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            array = None
        if array is not None and array.dtype != object:
            return _flatten_polygons(array)
        contours = []
        for item in value:
            contours.extend(_flatten_polygons(item))
        return contours
    return []


def _polygon_groups(value):
    """Keep top-level groups so class_ids can match nested instance data."""
    if value is None:
        return []
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray) and value.dtype != object:
        if value.size == 0:
            return []
        if value.ndim == 2 and value.shape[-1] == 2:
            return [[value]]
        if value.ndim >= 3 and value.shape[-1] == 2:
            return [[item] for item in value.reshape((-1, value.shape[-2], 2))]
    if isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            array = None
        if array is not None and array.dtype != object:
            return _polygon_groups(array)
        groups = []
        for item in value:
            contours = _flatten_polygons(item)
            if contours:
                groups.append(contours)
        return groups
    return []


def rasterize_polygons(polygons, shape, class_ids=None):
    """Rasterize contours into a binary or class-label mask.

    ``class_ids`` are mask labels, so a model class 0 should be passed as 1.
    Empty contours and empty polygon collections produce an all-zero mask.
    """
    if len(tuple(shape)) != 2:
        raise ValueError('shape must be (height, width), got {!r}'.format(shape))
    height, width = int(shape[0]), int(shape[1])
    if height < 0 or width < 0:
        raise ValueError('shape must be non-negative, got {!r}'.format(shape))

    groups = _polygon_groups(polygons)
    contours = [contour for group in groups for contour in group]
    if class_ids is None:
        labels = [1] * len(contours)
    else:
        class_array = _to_numpy(class_ids).reshape(-1)
        if class_array.size == 1 and len(contours) != 1:
            labels = [int(class_array[0])] * len(contours)
        elif class_array.size == len(contours):
            labels = [int(label) for label in class_array]
        else:
            group_labels = class_array
            if group_labels.size != len(groups):
                raise ValueError(
                    'class_ids count {} does not match {} polygons'.format(
                        class_array.size, len(contours)
                    )
                )
            labels = [int(group_labels[group_idx]) for group_idx, group in enumerate(groups) for _ in group]

    output = np.zeros((height, width), dtype=np.uint16)
    for contour, label in zip(contours, labels):
        points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] < 3:
            continue
        if not np.isfinite(points).all():
            raise ValueError('Polygon contains non-finite coordinates')
        points = np.rint(points).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, max(width - 1, 0))
        points[:, 1] = np.clip(points[:, 1], 0, max(height - 1, 0))
        cv2.fillPoly(output, [points], int(label))
    return output


def binary_iou_dice(pred, gt):
    """Return ``(IoU, Dice)`` for two binary masks."""
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError('pred and gt shapes differ: {} vs {}'.format(pred.shape, gt.shape))
    intersection = int(np.logical_and(pred, gt).sum())
    pred_count = int(pred.sum())
    gt_count = int(gt.sum())
    union = pred_count + gt_count - intersection
    if union == 0:
        return 1.0, 1.0
    iou = float(intersection) / float(union)
    denominator = pred_count + gt_count
    dice = float(2 * intersection) / float(denominator) if denominator else 1.0
    return iou, dice


def inverse_affine_points(points, inv_trans_input, orig_hw, flipped=False):
    """Map affine-input points back to original image coordinates."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape[-1:] != (2,):
        raise ValueError('points must end in dimension 2, got {}'.format(points.shape))
    matrix = np.asarray(inv_trans_input, dtype=np.float32)
    if matrix.shape != (2, 3):
        raise ValueError('inv_trans_input must have shape (2, 3), got {}'.format(matrix.shape))
    hw = np.asarray(orig_hw).reshape(-1)
    if hw.size != 2:
        raise ValueError('orig_hw must contain [height, width], got {}'.format(orig_hw))
    height, width = int(hw[0]), int(hw[1])
    if height <= 0 or width <= 0:
        raise ValueError('orig_hw must be positive, got {}'.format(orig_hw))

    original_shape = points.shape
    flat = points.reshape(-1, 2)
    if flat.size:
        homogeneous = np.concatenate(
            [flat, np.ones((flat.shape[0], 1), dtype=np.float32)], axis=1
        )
        restored = np.matmul(homogeneous, matrix.T)
    else:
        restored = flat.copy()
    if bool(np.asarray(flipped).reshape(-1)[0]) if np.asarray(flipped).size else False:
        restored[:, 0] = float(width) - restored[:, 0] - 1.0
    restored[:, 0] = np.clip(restored[:, 0], 0.0, float(width - 1))
    restored[:, 1] = np.clip(restored[:, 1], 0.0, float(height - 1))
    return restored.reshape(original_shape)


def configure_box_mode(config, mode):
    """Configure whether evaluation uses GT or predicted detector boxes."""
    normalized = str(mode).strip().lower()
    if normalized not in ('gt', 'predicted'):
        raise ValueError("box mode must be 'gt' or 'predicted', got {!r}".format(mode))
    config.sagittal_eval_box_mode = normalized
    config.use_gt_det = normalized == 'gt'
    config.use_gt_det_train_only = False
    return config


def _path_key(path, data_root):
    path = os.fspath(path)
    if not os.path.isabs(path):
        path = os.path.join(data_root, path)
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))


def _matches_split(row_split, catalog_split):
    row_split = str(row_split).strip().lower()
    catalog_split = str(catalog_split).strip().lower()
    accepted = {
        'train': {'train', 'training'},
        'val': {'val', 'validation'},
        'mini': {'mini', 'train', 'training'},
        'test': {'test'},
    }
    return row_split in accepted.get(catalog_split, {catalog_split})


def _as_scalar(value):
    array = _to_numpy(value)
    if array is None or array.size == 0:
        return None
    return array.reshape(-1)[0].item()


def _batch_size(batch):
    if not isinstance(batch, dict):
        return 1
    inp = batch.get('inp')
    if inp is not None and hasattr(inp, 'shape'):
        if len(inp.shape) >= 4:
            return int(inp.shape[0])
        if len(inp.shape) == 3:
            return 1
    image_paths = batch.get('img_path')
    if isinstance(image_paths, str):
        return 1
    if isinstance(image_paths, (list, tuple)) and image_paths:
        return len(image_paths)

    meta = batch.get('meta', {})
    if not isinstance(meta, dict):
        return 1
    case_ids = meta.get('case_id')
    if isinstance(case_ids, str):
        return 1
    if isinstance(case_ids, (list, tuple)) and case_ids:
        return len(case_ids)
    for key in ('slice_idx', 'orig_hw', 'inv_trans_input'):
        value = _to_numpy(meta.get(key))
        if value is None or value.ndim == 0:
            continue
        if key == 'orig_hw' and value.shape == (2,):
            continue
        if key == 'inv_trans_input' and value.shape == (2, 3):
            continue
        return int(value.shape[0])
    return 1


def _select_meta_value(value, index, batch_size, key):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    array = _to_numpy(value)
    if array is not None:
        if key == 'inv_trans_input':
            if array.ndim == 3:
                return array[index]
            if array.shape == (2, 3):
                return array
        if key == 'orig_hw':
            if array.ndim >= 2:
                return array[index]
            if array.size == 2:
                return array
        if key == 'flipped':
            if array.ndim >= 2:
                return array[index]
            if array.size == batch_size and batch_size > 1:
                return array[index]
            return array.reshape(-1)[0]
        if array.ndim >= 1 and array.shape[0] == batch_size:
            return array[index]
        if batch_size == 1:
            return array
    if isinstance(value, (list, tuple)):
        if len(value) == batch_size:
            return value[index]
        if batch_size == 1:
            return value
    return value


def _select_batch_value(value, index, batch_size):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) == batch_size:
            return value[index]
        if batch_size == 1 and len(value) == 1:
            return value[0]
    array = _to_numpy(value)
    if array is not None:
        if array.ndim >= 1 and array.shape[0] == batch_size:
            return array[index]
        if batch_size == 1:
            return array.reshape(-1)[0] if array.ndim == 1 and array.size == 1 else array
    return value


def _last_stage(value):
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _last_stage(value[-1])
    return value


def _contour_array(value):
    if value is None:
        return np.zeros((0, 0, 2), dtype=np.float32), None
    array = _to_numpy(value).astype(np.float32, copy=False)
    if array.size == 0:
        if array.ndim == 4:
            return np.zeros((0, array.shape[-2], 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        if array.ndim == 3:
            return array.reshape((0, array.shape[-2], 2)), np.zeros((0,), dtype=np.int64)
        return np.zeros((0, 0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    if array.ndim == 2 and array.shape[-1] == 2:
        return array[None], np.zeros((1,), dtype=np.int64)
    if array.ndim == 3 and array.shape[-1] == 2:
        return array, None
    if array.ndim == 4 and array.shape[-1] == 2:
        batch, count = array.shape[:2]
        indices = np.repeat(np.arange(batch, dtype=np.int64), count)
        return array.reshape((-1, array.shape[-2], 2)), indices
    raise RuntimeError('py must have shape [N,P,2] or [B,N,P,2], got {}'.format(array.shape))


def _metadata_array(output, key):
    value = output.get(key) if isinstance(output, dict) else None
    if value is None:
        return None
    return _to_numpy(value).reshape(-1)


class Evaluator:
    def __init__(self, result_dir, config=None):
        self.cfg = config if config is not None else global_cfg
        self.result_dir = os.path.abspath(os.fspath(result_dir))
        os.makedirs(self.result_dir, exist_ok=True)
        self.results = []
        self._class_stats = {}

        dataset_name = self.cfg.test.dataset
        attrs = DatasetCatalog.get(dataset_name)
        self.data_root = os.path.abspath(os.fspath(attrs.get('data_root', '.')))
        self.ann_file = os.path.abspath(os.fspath(attrs['ann_file']))
        self.catalog_split = attrs.get('split', 'val')
        default_box_mode = 'gt' if (
            bool(getattr(self.cfg, 'use_gt_det', False))
            and not bool(getattr(self.cfg, 'use_gt_det_train_only', False))
        ) else 'predicted'
        self.box_mode = str(
            getattr(self.cfg, 'sagittal_eval_box_mode', default_box_mode)
        ).strip().lower()
        if self.box_mode not in ('gt', 'predicted'):
            raise ValueError('Invalid sagittal_eval_box_mode: {!r}'.format(self.box_mode))
        self._records_by_image = {}
        self._read_manifest()

    def _read_manifest(self):
        if not os.path.isfile(self.ann_file):
            raise FileNotFoundError('Slice manifest not found: {}'.format(self.ann_file))
        with open(self.ann_file, 'r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            missing = _REQUIRED_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError('Slice manifest is missing columns: {}'.format(sorted(missing)))
            for row_number, row in enumerate(reader, start=2):
                if not _matches_split(row.get('split', ''), self.catalog_split):
                    continue
                try:
                    slice_idx = int(row['slice_idx'])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        'Invalid slice_idx at manifest row {}: {!r}'.format(
                            row_number, row.get('slice_idx')
                        )
                    ) from exc
                if not row.get('case_id'):
                    raise ValueError('Empty case_id at manifest row {}'.format(row_number))
                image_path = _path_key(row['image_path'], self.data_root)
                mask_path = _path_key(row['mask_path'], self.data_root)
                if image_path in self._records_by_image:
                    raise ValueError('Duplicate image_path in manifest: {}'.format(image_path))
                self._read_mask(mask_path)
                record = {
                    'case_id': str(row['case_id']),
                    'slice_idx': slice_idx,
                    'image_path': image_path,
                    'mask_path': mask_path,
                    'row_number': row_number,
                }
                self._records_by_image[image_path] = record

    @staticmethod
    def _read_mask(path):
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError('Mask not found or cannot be read: {}'.format(path))
        if mask.ndim != 2:
            raise ValueError('Expected a 2D uint16 mask at {}, got {}'.format(path, mask.shape))
        if mask.dtype != np.uint16:
            raise ValueError('Expected a uint16 mask at {}, got {}'.format(path, mask.dtype))
        labels = np.unique(mask)
        if np.any((labels < 0) | (labels > _NUM_CLASSES)):
            raise ValueError(
                'Mask labels must be in [0, 25] at {}; got {}'.format(path, labels.tolist())
            )
        return mask

    def _record_for_path(self, image_path):
        key = _path_key(image_path, self.data_root)
        try:
            return self._records_by_image[key]
        except KeyError as exc:
            raise KeyError(
                'Image path is not present in the evaluator manifest: {}'.format(image_path)
            ) from exc

    def _sample_metadata(self, batch, index, batch_size, record, mask_shape):
        meta = batch.get('meta', {}) if isinstance(batch, dict) else {}
        if not isinstance(meta, dict):
            meta = {}
        image_path = _select_batch_value(batch.get('img_path'), index, batch_size)
        if image_path is None:
            image_path = record['image_path']
        inv_trans = _select_meta_value(meta.get('inv_trans_input'), index, batch_size, 'inv_trans_input')
        if inv_trans is None:
            inv_trans = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        orig_hw = _select_meta_value(meta.get('orig_hw'), index, batch_size, 'orig_hw')
        if orig_hw is None:
            orig_hw = mask_shape
        flipped = _select_meta_value(meta.get('flipped'), index, batch_size, 'flipped')
        return image_path, np.asarray(inv_trans, dtype=np.float32), np.asarray(orig_hw), flipped

    def _prepare_predictions(self, output, batch_size):
        detection = output.get('detection') if isinstance(output, dict) else None
        if detection is None:
            raise RuntimeError("Evaluator output is missing 'detection'")
        detection = _to_numpy(detection)
        if detection.ndim == 2:
            if batch_size != 1 or detection.shape[-1] != 6:
                raise RuntimeError(
                    'detection with shape {} requires batch_size=1 and six columns'.format(
                        detection.shape
                    )
                )
            detection = detection[None]
        if detection.ndim != 3 or detection.shape[-1] != 6 or detection.shape[0] != batch_size:
            raise RuntimeError(
                'detection must have shape [B,N,6], got {} for batch size {}'.format(
                    detection.shape, batch_size
                )
            )
        valid_detections = []
        for sample_detection in detection:
            scores = sample_detection[:, 4]
            valid = np.isfinite(scores) & (scores > _SCORE_THRESHOLD)
            rows = sample_detection[valid]
            labels = np.rint(rows[:, 5]).astype(np.int64) if rows.size else np.zeros((0,), dtype=np.int64)
            if np.any((labels < 0) | (labels >= _NUM_CLASSES)):
                raise RuntimeError('Detection class ids must be in [0, 24], got {}'.format(labels.tolist()))
            valid_detections.append({'rows': rows, 'labels': labels})
        total_valid = sum(item['rows'].shape[0] for item in valid_detections)

        py_value = _last_stage(output.get('py'))
        contours, py_batch_indices = _contour_array(py_value)
        contour_count = contours.shape[0]
        py_cls = _metadata_array(output, 'py_cls')
        py_score = _metadata_array(output, 'py_score')
        for name, values in (('py_cls', py_cls), ('py_score', py_score)):
            if values is not None and values.size != contour_count:
                raise RuntimeError(
                    '{} count {} does not match contour count {}'.format(
                        name, values.size, contour_count
                    )
                )
        if contour_count != total_valid:
            raise RuntimeError(
                'Contour/detection count mismatch: {} contours for {} valid detections'.format(
                    contour_count, total_valid
                )
            )
        if py_cls is not None:
            py_cls = np.rint(py_cls).astype(np.int64)
            if np.any((py_cls < 0) | (py_cls >= _NUM_CLASSES)):
                raise RuntimeError('py_cls values must be in [0, 24], got {}'.format(py_cls.tolist()))
        if py_score is not None and np.any(~np.isfinite(py_score)):
            raise RuntimeError('py_score contains non-finite values')

        if py_batch_indices is None:
            py_batch_indices = np.concatenate([
                np.full(item['rows'].shape[0], batch_index, dtype=np.int64)
                for batch_index, item in enumerate(valid_detections)
            ]) if total_valid else np.zeros((0,), dtype=np.int64)
        elif py_batch_indices.size != contour_count:
            raise RuntimeError('py batch-index count does not match contour count')
        if np.any((py_batch_indices < 0) | (py_batch_indices >= batch_size)):
            raise RuntimeError('py batch indices are outside the batch')
        expected_counts = np.asarray(
            [item['rows'].shape[0] for item in valid_detections], dtype=np.int64
        )
        contour_counts = np.bincount(py_batch_indices, minlength=batch_size)
        if not np.array_equal(contour_counts, expected_counts):
            raise RuntimeError(
                'Contour/detection count mismatch by batch: {} vs {}'.format(
                    contour_counts.tolist(), expected_counts.tolist()
                )
            )

        fallback_labels = np.concatenate(
            [item['labels'] for item in valid_detections], axis=0
        ) if total_valid else np.zeros((0,), dtype=np.int64)
        fallback_scores = np.concatenate(
            [item['rows'][:, 4] for item in valid_detections], axis=0
        ) if total_valid else np.zeros((0,), dtype=np.float32)
        if py_cls is None:
            labels = fallback_labels
        else:
            labels = py_cls
        if labels.size != contour_count:
            raise RuntimeError('Prediction class count does not match contour count')
        scores = fallback_scores if py_score is None else py_score.astype(np.float32, copy=False)
        if scores.size != contour_count:
            raise RuntimeError('Prediction score count does not match contour count')

        predictions = [[] for _ in range(batch_size)]
        for contour, sample_index, label, score in zip(
                contours, py_batch_indices, labels, scores):
            predictions[int(sample_index)].append((contour, int(label), float(score)))
        return predictions

    def evaluate(self, output, batch):
        batch_size = _batch_size(batch)
        predictions = self._prepare_predictions(output, batch_size)
        for sample_index in range(batch_size):
            meta = batch.get('meta', {}) if isinstance(batch, dict) else {}
            meta_case = _select_meta_value(
                meta.get('case_id') if isinstance(meta, dict) else None,
                sample_index,
                batch_size,
                'case_id',
            )
            meta_slice = _select_meta_value(
                meta.get('slice_idx') if isinstance(meta, dict) else None,
                sample_index,
                batch_size,
                'slice_idx',
            )
            image_hint = _select_batch_value(batch.get('img_path'), sample_index, batch_size)
            if image_hint is None:
                raise RuntimeError("Batch is missing 'img_path'")
            record = self._record_for_path(image_hint)
            if meta_case is not None and str(meta_case) != record['case_id']:
                raise RuntimeError(
                    'Batch case_id {!r} does not match manifest {!r}'.format(
                        meta_case, record['case_id']
                    )
                )
            if meta_slice is not None and int(_as_scalar(meta_slice)) != record['slice_idx']:
                raise RuntimeError(
                    'Batch slice_idx {!r} does not match manifest {!r}'.format(
                        meta_slice, record['slice_idx']
                    )
                )

            gt_mask = self._read_mask(record['mask_path'])
            _, inv_trans_input, orig_hw, flipped = self._sample_metadata(
                batch, sample_index, batch_size, record, gt_mask.shape
            )
            requested_hw = np.asarray(orig_hw).reshape(-1)
            if requested_hw.size != 2:
                raise RuntimeError('meta.orig_hw must contain [height, width]')
            if tuple(int(x) for x in requested_hw) != tuple(gt_mask.shape):
                raise RuntimeError(
                    'meta.orig_hw {} does not match mask shape {}'.format(
                        tuple(int(x) for x in requested_hw), gt_mask.shape
                    )
                )

            contours = []
            labels = []
            scores = []
            for contour, class_id, score in predictions[sample_index]:
                restored = inverse_affine_points(
                    contour * float(snake_config.down_ratio),
                    inv_trans_input,
                    requested_hw,
                    flipped=flipped,
                )
                contours.append(restored)
                labels.append(class_id + 1)
                scores.append(float(score))

            pred_foreground = rasterize_polygons(contours, gt_mask.shape) > 0
            foreground_iou, foreground_dice = binary_iou_dice(pred_foreground, gt_mask > 0)
            class_iou = {}
            class_dice = {}
            for class_id in sorted(int(x) for x in np.unique(gt_mask) if int(x) > 0):
                pred_class = np.zeros(gt_mask.shape, dtype=bool)
                for contour, label in zip(contours, labels):
                    if label == class_id:
                        pred_class |= rasterize_polygons([contour], gt_mask.shape) > 0
                iou, dice = binary_iou_dice(pred_class, gt_mask == class_id)
                class_key = str(class_id)
                class_iou[class_key] = iou
                class_dice[class_key] = dice
                gt_count = int((gt_mask == class_id).sum())
                pred_count = int(pred_class.sum())
                intersection = int(np.logical_and(pred_class, gt_mask == class_id).sum())
                stats = self._class_stats.setdefault(
                    class_key, {'gt': 0, 'pred': 0, 'intersection': 0, 'num_slices': 0}
                )
                stats['gt'] += gt_count
                stats['pred'] += pred_count
                stats['intersection'] += intersection
                stats['num_slices'] += 1

            self.results.append({
                'case_id': record['case_id'],
                'slice_idx': int(record['slice_idx']),
                'image_path': record['image_path'],
                'n_pred': int(len(contours)),
                'pred_scores': scores,
                'gt_foreground_pixels': int((gt_mask > 0).sum()),
                'pred_foreground_pixels': int(pred_foreground.sum()),
                'foreground_iou': foreground_iou,
                'foreground_dice': foreground_dice,
                'class_iou': class_iou,
                'class_dice': class_dice,
                'box_mode': self.box_mode,
            })

    @staticmethod
    def _mean(values):
        return float(np.mean(values)) if values else 0.0

    def summarize(self):
        all_iou = [float(item['foreground_iou']) for item in self.results]
        all_dice = [float(item['foreground_dice']) for item in self.results]
        foreground = [item for item in self.results if item['gt_foreground_pixels'] > 0]
        slices_with_predictions = [item for item in self.results if item['n_pred'] > 0]
        foreground_with_predictions = [
            item for item in foreground if item['n_pred'] > 0
        ]
        total_predicted_boxes = sum(int(item['n_pred']) for item in self.results)
        class_iou_values = [
            float(value)
            for item in self.results
            for value in item['class_iou'].values()
        ]
        class_dice_values = [
            float(value)
            for item in self.results
            for value in item['class_dice'].values()
        ]
        class_aggregate = {}
        for class_key, stats in sorted(self._class_stats.items(), key=lambda item: int(item[0])):
            union = stats['gt'] + stats['pred'] - stats['intersection']
            denominator = stats['gt'] + stats['pred']
            if union == 0:
                aggregate_iou = 1.0
            else:
                aggregate_iou = float(stats['intersection']) / float(union)
            aggregate_dice = (
                float(2 * stats['intersection']) / float(denominator)
                if denominator else 1.0
            )
            class_aggregate[class_key] = {
                'iou': aggregate_iou,
                'dice': aggregate_dice,
                'num_slices': int(stats['num_slices']),
            }

        summary = {
            'num_slices': int(len(self.results)),
            'num_foreground_slices': int(len(foreground)),
            'num_slices_with_predictions': int(len(slices_with_predictions)),
            'num_foreground_slices_with_predictions': int(
                len(foreground_with_predictions)
            ),
            'total_predicted_boxes': int(total_predicted_boxes),
            'all_slice_mean_iou': self._mean(all_iou),
            'all_slice_mean_dice': self._mean(all_dice),
            'foreground_slice_mean_iou': self._mean(
                [float(item['foreground_iou']) for item in foreground]
            ),
            'foreground_slice_mean_dice': self._mean(
                [float(item['foreground_dice']) for item in foreground]
            ),
            'class_mean_iou': self._mean(class_iou_values),
            'class_mean_dice': self._mean(class_dice_values),
            'class_aggregate': class_aggregate,
            'box_mode': self.box_mode,
            'empty_validation': not bool(self.results),
        }
        with open(os.path.join(self.result_dir, 'slices.json'), 'w', encoding='utf-8') as handle:
            json.dump(self.results, handle, indent=2, sort_keys=True, allow_nan=False)
        with open(os.path.join(self.result_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        return summary
