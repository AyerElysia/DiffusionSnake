from __future__ import annotations

import cv2
import numpy as np
import torch


def _poly_to_mask_np(poly: np.ndarray, H: int, W: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.round(poly).astype(np.int32)
    if pts.ndim == 2 and pts.shape[0] > 2:
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 1)
    return mask


def _boundary_mask(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.zeros_like(mask, dtype=np.uint8)
    if contours:
        cv2.drawContours(boundary, contours, -1, 1, thickness=max(int(thickness), 1))
    return boundary.astype(bool)


def _distance_to_boundary(boundary: np.ndarray) -> np.ndarray:
    if boundary.sum() == 0:
        return np.full(boundary.shape, 1e3, dtype=np.float32)
    return cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 3)


def _sample_image_at_points(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    H, W = image.shape[:2]
    xy = np.round(pts).astype(np.int32)
    xs = np.clip(xy[:, 0], 0, W - 1)
    ys = np.clip(xy[:, 1], 0, H - 1)
    return image[ys, xs].astype(np.float32)


def _curvature(poly: np.ndarray) -> np.ndarray:
    if poly.shape[0] < 3:
        return np.zeros((poly.shape[0],), dtype=np.float32)
    return np.linalg.norm(np.roll(poly, 1, axis=0) - 2.0 * poly + np.roll(poly, -1, axis=0), axis=1)


def _normalize(x: np.ndarray, q: float = 95.0) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    denom = float(np.percentile(x, np.clip(q, 1.0, 100.0)))
    return np.clip(x / max(denom, 1e-6), 0.0, 2.0).astype(np.float32)


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(x * w) / max(float(np.sum(w)), 1e-6))


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    cdf = np.cumsum(ws)
    cutoff = np.clip(float(q) / 100.0, 0.0, 1.0) * max(float(cdf[-1]), 1e-6)
    return float(xs[min(int(np.searchsorted(cdf, cutoff, side="left")), xs.size - 1)])


def _boundary_iou(pred_boundary: np.ndarray, gt_boundary: np.ndarray, radius: int) -> float:
    k = 2 * max(int(radius), 1) + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    pred_band = cv2.dilate(pred_boundary.astype(np.uint8), kernel, iterations=1).astype(bool)
    gt_band = cv2.dilate(gt_boundary.astype(np.uint8), kernel, iterations=1).astype(bool)
    union = np.logical_or(pred_band, gt_band).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(pred_band, gt_band).sum() / union)


def _area_drift(pred_mask: np.ndarray, gt_mask: np.ndarray, max_frac: float) -> float:
    gt_area = float(gt_mask.sum())
    pred_area = float(pred_mask.sum())
    if gt_area <= 0:
        return 1.0
    drift = abs(pred_area - gt_area) / gt_area
    return float(np.clip(drift / max(float(max_frac), 1e-6), 0.0, 2.0))


def _single_detail_score(
    pred: np.ndarray,
    gt: np.ndarray,
    H: int,
    W: int,
    *,
    corner_dist_max_px: float,
    corner_dist_quantile: float,
    corner_dist_quantile_weight: float,
    curvature_max_px: float,
    burr_margin_px: float,
    burr_max_px: float,
    burr_quantile: float,
    local_band_radius_px: int,
    area_max_frac: float,
    w_corner_dist: float,
    w_curv_match: float,
    w_local_biou: float,
    w_burr: float,
    w_area: float,
) -> tuple[float, dict[str, float]]:
    pred_mask = _poly_to_mask_np(pred, H, W)
    gt_mask = _poly_to_mask_np(gt, H, W)
    pred_boundary = _boundary_mask(pred_mask, 1)
    gt_boundary = _boundary_mask(gt_mask, 1)

    pred_curv = _curvature(pred)
    gt_curv = _curvature(gt)
    point_err = np.linalg.norm(pred - gt, axis=1) if pred.shape == gt.shape else np.zeros_like(gt_curv)
    weights = 1.0 + _normalize(gt_curv) + 0.5 * _normalize(point_err)
    weights = weights.astype(np.float32)

    pred_dt = _distance_to_boundary(pred_boundary)
    gt_dt = _distance_to_boundary(gt_boundary)
    gt_to_pred = _sample_image_at_points(pred_dt, gt)
    pred_to_gt = _sample_image_at_points(gt_dt, pred)
    if gt_to_pred.shape != pred_to_gt.shape:
        n = min(gt_to_pred.shape[0], pred_to_gt.shape[0], weights.shape[0])
        gt_to_pred = gt_to_pred[:n]
        pred_to_gt = pred_to_gt[:n]
        weights = weights[:n]
    dist = 0.5 * (gt_to_pred + pred_to_gt)
    dist_mean = _weighted_mean(dist, weights)
    dist_q = _weighted_quantile(dist, weights, corner_dist_quantile)
    dist_mix = (dist_mean + float(corner_dist_quantile_weight) * dist_q) / (1.0 + float(corner_dist_quantile_weight))
    corner_dist = float(1.0 - np.clip(dist_mix / max(float(corner_dist_max_px), 1e-6), 0.0, 1.0))

    if pred_curv.shape != gt_curv.shape:
        n = min(pred_curv.shape[0], gt_curv.shape[0], weights.shape[0])
        pred_curv = pred_curv[:n]
        gt_curv = gt_curv[:n]
        weights = weights[:n]
    curv_err = np.abs(pred_curv - gt_curv)
    curv_match = float(1.0 - np.clip(_weighted_mean(curv_err, weights) / max(float(curvature_max_px), 1e-6), 0.0, 1.0))

    excess = np.maximum(pred_curv - gt_curv - float(burr_margin_px), 0.0)
    burr_raw = _weighted_quantile(excess, weights, burr_quantile)
    burr_penalty = float(np.clip(burr_raw / max(float(burr_max_px), 1e-6), 0.0, 2.0))

    local_biou = _boundary_iou(pred_boundary, gt_boundary, local_band_radius_px)
    area_penalty = _area_drift(pred_mask, gt_mask, area_max_frac)

    total = (
        float(w_corner_dist) * corner_dist
        + float(w_curv_match) * curv_match
        + float(w_local_biou) * local_biou
        - float(w_burr) * burr_penalty
        - float(w_area) * area_penalty
    )
    return total, {
        "corner_dist": corner_dist,
        "curv_match": curv_match,
        "local_biou": local_biou,
        "burr_penalty": burr_penalty,
        "burr_raw_px": float(burr_raw),
        "area_penalty": area_penalty,
    }


def compute_curvature_detail_score(
    poly_py: torch.Tensor,
    gt_py: torch.Tensor,
    H: int,
    W: int,
    *,
    coord_scale: float = 1.0,
    corner_dist_max_px: float = 6.0,
    corner_dist_quantile: float = 95.0,
    corner_dist_quantile_weight: float = 0.7,
    curvature_max_px: float = 4.0,
    burr_margin_px: float = 0.5,
    burr_max_px: float = 1.5,
    burr_quantile: float = 95.0,
    local_band_radius_px: int = 2,
    area_max_frac: float = 0.15,
    w_corner_dist: float = 0.35,
    w_curv_match: float = 0.20,
    w_local_biou: float = 0.10,
    w_burr: float = 0.07,
    w_area: float = 0.03,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = poly_py.device
    pred_np = poly_py.detach().float().cpu().numpy() * float(coord_scale)
    gt_np = gt_py.detach().float().cpu().numpy() * float(coord_scale)

    scores = []
    comps: dict[str, list[float]] = {
        "corner_dist": [],
        "curv_match": [],
        "local_biou": [],
        "burr_penalty": [],
        "burr_raw_px": [],
        "area_penalty": [],
    }
    for i in range(pred_np.shape[0]):
        score, comp = _single_detail_score(
            pred_np[i],
            gt_np[i],
            int(H),
            int(W),
            corner_dist_max_px=corner_dist_max_px,
            corner_dist_quantile=corner_dist_quantile,
            corner_dist_quantile_weight=corner_dist_quantile_weight,
            curvature_max_px=curvature_max_px,
            burr_margin_px=burr_margin_px,
            burr_max_px=burr_max_px,
            burr_quantile=burr_quantile,
            local_band_radius_px=local_band_radius_px,
            area_max_frac=area_max_frac,
            w_corner_dist=w_corner_dist,
            w_curv_match=w_curv_match,
            w_local_biou=w_local_biou,
            w_burr=w_burr,
            w_area=w_area,
        )
        scores.append(score)
        for key, value in comp.items():
            comps[key].append(value)

    score_t = torch.tensor(scores, device=device, dtype=torch.float32)
    if not return_components:
        return score_t
    comp_t = {key: torch.tensor(vals, device=device, dtype=torch.float32) for key, vals in comps.items()}
    return score_t, comp_t
