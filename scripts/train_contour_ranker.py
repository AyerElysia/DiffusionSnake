#!/usr/bin/env python3
"""
Train a lightweight contour-quality ranker from offline best-of-K candidates.

Environment:
  DATA_PATH default data/stats/ranker_dataset_train_full.npz
  EVAL_GPU  CUDA_VISIBLE_DEVICES override
  EPOCHS    default 100
  OUT_PATH  default data/stats/contour_ranker_best.pt
"""

import json
import math
import os
import sys
import time

if str(os.environ.get('EVAL_GPU', '')).strip():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(os.environ.get('EVAL_GPU', '')).strip()

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)


DEFAULT_DATA_PATH = os.path.join('data', 'stats', 'ranker_dataset_train_full.npz')
DEFAULT_OUT_PATH = os.path.join('data', 'stats', 'contour_ranker_best.pt')
SCALAR_DIM = 6
SCALAR_NAMES = (
    'consensus',
    'edge',
    'perimeter_area_ratio',
    'curvature_mean',
    'curvature_max',
    'curvature_std',
)


def resolve_path(path, default_value):
    value = str(path or '').strip()
    if not value:
        value = default_value
    if os.path.isabs(value):
        return value
    return os.path.join(_REPO_ROOT, value)


def choose_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def polygon_area(poly):
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_perimeter(poly):
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 2:
        return 0.0
    diff = np.roll(pts, -1, axis=0) - pts
    return float(np.linalg.norm(diff, axis=1).sum())


def normalize_poly(poly):
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return pts.astype(np.float32)
    center = pts.mean(axis=0, keepdims=True)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    diag = float(np.linalg.norm(max_xy - min_xy))
    if diag < 1e-6:
        diag = 1.0
    return ((pts - center) / diag).astype(np.float32)


def curvature_stats(norm_poly):
    pts = np.asarray(norm_poly, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return 0.0, 0.0, 0.0
    second = np.roll(pts, -1, axis=0) - 2.0 * pts + np.roll(pts, 1, axis=0)
    vals = np.linalg.norm(second, axis=1)
    if vals.size == 0:
        return 0.0, 0.0, 0.0
    return float(vals.mean()), float(vals.max()), float(vals.std())


def compute_geometry_features(poly_group, consensus_group, edge_group):
    """Return normalized flattened polygons and raw scalar features for one K-candidate group."""
    poly_group = np.asarray(poly_group, dtype=np.float32)
    consensus_group = np.asarray(consensus_group, dtype=np.float32).reshape(-1)
    edge_group = np.asarray(edge_group, dtype=np.float32).reshape(-1)
    flat_polys = []
    scalars = []

    for idx in range(poly_group.shape[0]):
        poly = poly_group[idx]
        norm = normalize_poly(poly)
        perimeter = polygon_perimeter(poly)
        area = polygon_area(poly)
        ratio = perimeter / max(area, 1.0)
        curv_mean, curv_max, curv_std = curvature_stats(norm)
        flat_polys.append(norm.reshape(-1))
        scalars.append([
            float(consensus_group[idx]),
            float(edge_group[idx]),
            float(ratio),
            float(curv_mean),
            float(curv_max),
            float(curv_std),
        ])

    return (
        np.asarray(flat_polys, dtype=np.float32),
        np.asarray(scalars, dtype=np.float32),
    )


def bbox_for_poly(poly, height, width):
    if poly is None:
        return np.zeros((4,), dtype=np.float32)
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return np.zeros((4,), dtype=np.float32)
    x1 = float(np.clip(pts[:, 0].min(), 0.0, max(float(width), 0.0)))
    y1 = float(np.clip(pts[:, 1].min(), 0.0, max(float(height), 0.0)))
    x2 = float(np.clip(pts[:, 0].max(), 0.0, max(float(width), 0.0)))
    y2 = float(np.clip(pts[:, 1].max(), 0.0, max(float(height), 0.0)))
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def to_gray_image(image):
    if image is None:
        return None
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return None


def crop_single_patch(gray, bbox, patch_size=64, expand=0.2):
    if gray is None or gray.size == 0:
        return np.zeros((patch_size, patch_size), dtype=np.float32)

    height, width = gray.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox, dtype=np.float32).reshape(4)]
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1.0 or bh <= 1.0:
        return np.zeros((patch_size, patch_size), dtype=np.float32)

    pad_x = bw * float(expand)
    pad_y = bh * float(expand)
    ix1 = max(0, int(math.floor(x1 - pad_x)))
    iy1 = max(0, int(math.floor(y1 - pad_y)))
    ix2 = min(int(width), int(math.ceil(x2 + pad_x)) + 1)
    iy2 = min(int(height), int(math.ceil(y2 + pad_y)) + 1)
    if ix2 <= ix1 or iy2 <= iy1:
        return np.zeros((patch_size, patch_size), dtype=np.float32)

    patch = gray[iy1:iy2, ix1:ix2]
    if patch.size == 0:
        return np.zeros((patch_size, patch_size), dtype=np.float32)
    patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    return patch.astype(np.float32) / 255.0


def crop_candidate_patches(image, bboxes, patch_size=64):
    gray = to_gray_image(image)
    patches = []
    for bbox in np.asarray(bboxes, dtype=np.float32).reshape(-1, 4):
        patches.append(crop_single_patch(gray, bbox, patch_size=patch_size))
    return np.asarray(patches, dtype=np.float32)[:, None, :, :]


def apply_scalar_norm(scalars, scalar_mean, scalar_std):
    scalars = np.asarray(scalars, dtype=np.float32)
    if scalar_mean is None or scalar_std is None:
        return scalars
    mean = np.asarray(scalar_mean, dtype=np.float32).reshape(1, -1)
    std = np.asarray(scalar_std, dtype=np.float32).reshape(1, -1)
    return ((scalars - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def build_ranker_inputs(poly_group, consensus_group, edge_group, bboxes, image,
                        scalar_mean=None, scalar_std=None):
    poly_flat, scalars = compute_geometry_features(poly_group, consensus_group, edge_group)
    scalars = apply_scalar_norm(scalars, scalar_mean, scalar_std)
    patches = crop_candidate_patches(image, bboxes, patch_size=64)
    return poly_flat.astype(np.float32), scalars.astype(np.float32), patches.astype(np.float32)


class ContourRanker(nn.Module):
    def __init__(self, num_points, scalar_dim=SCALAR_DIM):
        super(ContourRanker, self).__init__()
        self.num_points = int(num_points)
        self.scalar_dim = int(scalar_dim)
        poly_dim = self.num_points * 2

        self.poly_mlp = nn.Sequential(
            nn.Linear(poly_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.patch_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + self.scalar_dim + 64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, poly_flat, scalars, patches):
        batch_size, num_candidates = poly_flat.shape[:2]
        poly_flat = poly_flat.reshape(batch_size * num_candidates, -1)
        scalars = scalars.reshape(batch_size * num_candidates, -1)
        patches = patches.reshape(batch_size * num_candidates, 1, 64, 64)

        poly_feat = self.poly_mlp(poly_flat)
        patch_feat = self.patch_cnn(patches).reshape(batch_size * num_candidates, -1)
        feat = torch.cat([poly_feat, scalars, patch_feat], dim=1)
        scores = self.head(feat).reshape(batch_size, num_candidates)
        return scores


class RankerGroupDataset(Dataset):
    def __init__(self, arrays, group_indices, scalar_mean, scalar_std, repo_root):
        self.poly = arrays['poly']
        self.gt_iou = arrays['gt_iou']
        self.consensus = arrays['consensus']
        self.edge = arrays['edge']
        self.local_patch_meta = arrays['local_patch_meta']
        self.sample_index = arrays['sample_index']
        self.sample_img_paths = arrays['sample_img_paths']
        self.sample_indices = arrays.get('sample_indices', None)
        self.group_indices = np.asarray(group_indices, dtype=np.int64)
        self.scalar_mean = np.asarray(scalar_mean, dtype=np.float32)
        self.scalar_std = np.asarray(scalar_std, dtype=np.float32)
        self.repo_root = repo_root
        self.path_by_sample = self._build_path_map()
        self._image_cache = {}
        self._cache_keys = []

    def _build_path_map(self):
        path_by_sample = {}
        if self.sample_indices is not None:
            for idx, path in zip(self.sample_indices.tolist(), self.sample_img_paths.tolist()):
                path_by_sample[int(idx)] = str(path)
        else:
            for idx, path in enumerate(self.sample_img_paths.tolist()):
                path_by_sample[int(idx)] = str(path)
        return path_by_sample

    def __len__(self):
        return int(self.group_indices.shape[0])

    def _resolve_image_path(self, sample_index):
        path = self.path_by_sample.get(int(sample_index), '')
        if not path and int(sample_index) < len(self.sample_img_paths):
            path = str(self.sample_img_paths[int(sample_index)])
        if path and not os.path.isabs(path):
            path = os.path.join(self.repo_root, path)
        return path

    def _read_gray(self, sample_index):
        path = self._resolve_image_path(sample_index)
        cached = self._image_cache.get(path)
        if cached is not None:
            return cached
        gray = None
        if path:
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = np.zeros((64, 64), dtype=np.uint8)
        self._image_cache[path] = gray
        self._cache_keys.append(path)
        if len(self._cache_keys) > 32:
            old = self._cache_keys.pop(0)
            self._image_cache.pop(old, None)
        return gray

    def __getitem__(self, pos):
        group_index = int(self.group_indices[int(pos)])
        sample_index = int(self.sample_index[group_index])
        image = self._read_gray(sample_index)
        poly_flat, scalars, patches = build_ranker_inputs(
            self.poly[group_index],
            self.consensus[group_index],
            self.edge[group_index],
            self.local_patch_meta[group_index],
            image,
            scalar_mean=self.scalar_mean,
            scalar_std=self.scalar_std,
        )
        return {
            'poly_flat': torch.from_numpy(poly_flat),
            'scalars': torch.from_numpy(scalars),
            'patches': torch.from_numpy(patches),
            'gt_iou': torch.from_numpy(self.gt_iou[group_index].astype(np.float32)),
        }


def load_npz_arrays(data_path):
    data = np.load(data_path, allow_pickle=False)
    arrays = {}
    for key in data.files:
        arrays[key] = data[key]
    return arrays


def split_groups_by_sample(sample_index, train_fraction=0.9, seed=12345):
    unique_samples = np.unique(sample_index.astype(np.int64))
    if unique_samples.size < 2:
        all_indices = np.arange(sample_index.shape[0], dtype=np.int64)
        return all_indices, all_indices

    rng = np.random.RandomState(seed)
    shuffled = unique_samples.copy()
    rng.shuffle(shuffled)
    num_train = int(round(float(unique_samples.size) * float(train_fraction)))
    num_train = max(1, min(num_train, unique_samples.size - 1))
    train_samples = set([int(v) for v in shuffled[:num_train].tolist()])

    train_indices = []
    val_indices = []
    for idx, sample in enumerate(sample_index.tolist()):
        if int(sample) in train_samples:
            train_indices.append(idx)
        else:
            val_indices.append(idx)
    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(val_indices, dtype=np.int64),
    )


def compute_scalar_stats(arrays, group_indices):
    scalars = []
    for group_index in np.asarray(group_indices, dtype=np.int64).tolist():
        _poly_flat, group_scalars = compute_geometry_features(
            arrays['poly'][group_index],
            arrays['consensus'][group_index],
            arrays['edge'][group_index],
        )
        scalars.append(group_scalars)
    if not scalars:
        return np.zeros((SCALAR_DIM,), dtype=np.float32), np.ones((SCALAR_DIM,), dtype=np.float32)
    all_scalars = np.concatenate(scalars, axis=0).astype(np.float32)
    mean = all_scalars.mean(axis=0).astype(np.float32)
    std = all_scalars.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean, std


def move_batch_to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def ranker_loss(scores, gt_iou, min_iou_diff=0.005, margin=0.1, listwise_weight=0.3, temperature=0.05):
    gt_diff = gt_iou.unsqueeze(2) - gt_iou.unsqueeze(1)
    mask = gt_diff > float(min_iou_diff)
    if bool(mask.any().item()):
        score_i = scores.unsqueeze(2).expand_as(gt_diff)
        score_j = scores.unsqueeze(1).expand_as(gt_diff)
        target = torch.ones_like(score_i[mask])
        pair_loss = F.margin_ranking_loss(
            score_i[mask],
            score_j[mask],
            target,
            margin=float(margin),
            reduction='mean',
        )
    else:
        pair_loss = scores.sum() * 0.0

    target_prob = F.softmax(gt_iou / float(temperature), dim=1)
    log_prob = F.log_softmax(scores, dim=1)
    list_loss = -(target_prob * log_prob).sum(dim=1).mean()
    return pair_loss + float(listwise_weight) * list_loss


def run_train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_groups = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        scores = model(batch['poly_flat'], batch['scalars'], batch['patches'])
        loss = ranker_loss(scores, batch['gt_iou'])
        loss.backward()
        optimizer.step()

        batch_size = int(batch['gt_iou'].shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_groups += batch_size
    return total_loss / float(max(total_groups, 1))


def evaluate_ranker(model, loader, device):
    model.eval()
    selected_ious = []
    oracle_ious = []
    random_ious = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            scores = model(batch['poly_flat'], batch['scalars'], batch['patches'])
            gt = batch['gt_iou']
            selected_idx = torch.argmax(scores, dim=1)
            row_idx = torch.arange(gt.shape[0], device=gt.device)
            selected_ious.append(gt[row_idx, selected_idx].detach().cpu())
            oracle_ious.append(torch.max(gt, dim=1)[0].detach().cpu())
            random_ious.append(gt[:, 0].detach().cpu())

    if not selected_ious:
        return 0.0, 0.0, 0.0
    selected = torch.cat(selected_ious).float().mean().item()
    oracle = torch.cat(oracle_ious).float().mean().item()
    random = torch.cat(random_ious).float().mean().item()
    return float(selected), float(oracle), float(random)


def save_checkpoint(path, model, optimizer, epoch, best_metric, num_points,
                    scalar_mean, scalar_std, data_path, train_groups, val_groups):
    payload = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': int(epoch),
        'best_val_selected_mean_iou': float(best_metric),
        'num_points': int(num_points),
        'scalar_dim': int(SCALAR_DIM),
        'scalar_names': list(SCALAR_NAMES),
        'scalar_mean': np.asarray(scalar_mean, dtype=np.float32),
        'scalar_std': np.asarray(scalar_std, dtype=np.float32),
        'meta': {
            'data_path': data_path,
            'train_groups': int(train_groups),
            'val_groups': int(val_groups),
            'saved_at': time.strftime('%Y%m%d_%H%M%S'),
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_contour_ranker(path, device=None):
    if device is None:
        device = choose_device()
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
    num_points = int(ckpt.get('num_points', 128))
    scalar_dim = int(ckpt.get('scalar_dim', SCALAR_DIM))
    model = ContourRanker(num_points=num_points, scalar_dim=scalar_dim)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, ckpt


def print_dataset_summary(data_path, arrays, train_indices, val_indices, scalar_mean, scalar_std):
    poly_shape = arrays['poly'].shape
    gt_shape = arrays['gt_iou'].shape
    sample_count = int(np.unique(arrays['sample_index']).size)
    print('[*] DATA_PATH: {}'.format(data_path))
    print('[*] poly shape: {} | gt_iou shape: {}'.format(poly_shape, gt_shape))
    print('[*] samples: {} | train groups: {} | val groups: {}'.format(
        sample_count,
        int(train_indices.shape[0]),
        int(val_indices.shape[0]),
    ))
    print('[*] scalar mean: {}'.format(json.dumps([float(v) for v in scalar_mean.tolist()])))
    print('[*] scalar std:  {}'.format(json.dumps([float(v) for v in scalar_std.tolist()])))


def main():
    data_path = resolve_path(os.environ.get('DATA_PATH', ''), DEFAULT_DATA_PATH)
    out_path = resolve_path(os.environ.get('OUT_PATH', ''), DEFAULT_OUT_PATH)
    epochs = int(os.environ.get('EPOCHS', '100'))
    batch_size = int(os.environ.get('BATCH_SIZE', '32'))
    num_workers = int(os.environ.get('NUM_WORKERS', '4'))
    patience = int(os.environ.get('PATIENCE', '15'))

    arrays = load_npz_arrays(data_path)
    if arrays['poly'].ndim != 4:
        raise ValueError('Expected poly shape [G,K,N,2], got {}'.format(arrays['poly'].shape))
    if arrays['gt_iou'].ndim != 2:
        raise ValueError('Expected gt_iou shape [G,K], got {}'.format(arrays['gt_iou'].shape))

    num_points = int(arrays['poly'].shape[2])
    train_indices, val_indices = split_groups_by_sample(arrays['sample_index'])
    scalar_mean, scalar_std = compute_scalar_stats(arrays, train_indices)
    print_dataset_summary(data_path, arrays, train_indices, val_indices, scalar_mean, scalar_std)

    train_set = RankerGroupDataset(arrays, train_indices, scalar_mean, scalar_std, _REPO_ROOT)
    val_set = RankerGroupDataset(arrays, val_indices, scalar_mean, scalar_std, _REPO_ROOT)
    device = choose_device()
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = ContourRanker(num_points=num_points, scalar_dim=SCALAR_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(epochs), 1),
        eta_min=1e-5,
    )

    print('[*] device: {} | epochs: {} | batch: {} | workers: {}'.format(
        str(device),
        int(epochs),
        int(batch_size),
        int(num_workers),
    ))
    print('[*] saving best checkpoint to {}'.format(out_path))

    best_metric = -1.0
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        train_loss = run_train_epoch(model, train_loader, optimizer, device)
        val_selected, val_oracle, val_random = evaluate_ranker(model, val_loader, device)
        scheduler.step()

        print(
            'epoch {}/{} | train_loss={:.6f} | val_top1_mean_iou={:.6f} '
            '| val_oracle_mean_iou={:.6f} | val_random_mean_iou={:.6f}'.format(
                int(epoch),
                int(epochs),
                float(train_loss),
                float(val_selected),
                float(val_oracle),
                float(val_random),
            )
        )

        if val_selected > best_metric:
            best_metric = float(val_selected)
            bad_epochs = 0
            save_checkpoint(
                out_path,
                model,
                optimizer,
                epoch,
                best_metric,
                num_points,
                scalar_mean,
                scalar_std,
                data_path,
                train_indices.shape[0],
                val_indices.shape[0],
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print('[*] early stop at epoch {} with best val_top1_mean_iou={:.6f}'.format(
                    int(epoch),
                    float(best_metric),
                ))
                break

    print('[*] best val_top1_mean_iou={:.6f}'.format(float(best_metric)))
    print('[*] best checkpoint: {}'.format(out_path))


if __name__ == '__main__':
    main()
