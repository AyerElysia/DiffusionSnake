import numpy as np
import cv2
import torch
def _poly_to_mask_np(poly: np.ndarray, H: int, W: int) -> np.ndarray:
    """将多边形轮廓转换为二值掩码。"""
    m = np.zeros((H, W), dtype=np.uint8)
    p = np.round(poly).astype(np.int32)
    if p.ndim == 2 and p.shape[0] > 2:
        cv2.fillPoly(m, [p.reshape(-1, 1, 2)], 1)
    return m


def _extract_contour(mask: np.ndarray, tolerance: int = 1) -> np.ndarray:
    """从二值掩码中提取边界（通过描绘轮廓实现，可调厚度）"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(mask, dtype=np.uint8)
    if len(contours) > 0:
        cv2.drawContours(contour_mask, contours, -1, 1, thickness=tolerance)
    return contour_mask.astype(np.float32)


def _boundary_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance: int) -> float:
    pred_contour = _extract_contour(pred_mask, tolerance)
    gt_contour = _extract_contour(gt_mask, tolerance)
    intersection = (pred_contour * gt_contour).sum()
    union = pred_contour.sum() + gt_contour.sum()
    return float(2.0 * intersection / union) if union > 0 else 0.0


def _calc_mboundf(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """复用 test_medical 中逻辑：多容忍度下的边界Dice平均"""
    tolerances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    vals = []
    for tol in tolerances:
        vals.append(_boundary_dice(pred_mask, gt_mask, tol))
    return float(np.mean(vals)) if len(vals) > 0 else 0.0


def _calc_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(inter / union) if union > 0 else 0.0


def _calc_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = np.logical_and(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    return float(2.0 * inter / denom) if denom > 0 else 0.0


def compute_region_score(poly_py: torch.Tensor,
                         gt_py: torch.Tensor,
                         H: int,
                         W: int,
                         w_boundary: float = 1.0,
                         w_dice: float = 0.0,
                         w_iou: float = 0.0,
                         coord_scale: float = 1.0) -> torch.Tensor:
    """Compute absolute contour quality from boundary F-score, Dice, and optional IoU."""
    device = poly_py.device
    pred_np = poly_py.detach().float().cpu().numpy() * float(coord_scale)
    gt_np = gt_py.detach().float().cpu().numpy() * float(coord_scale)

    scores = []
    for i in range(pred_np.shape[0]):
        m_pred = _poly_to_mask_np(pred_np[i], H, W)
        m_gt = _poly_to_mask_np(gt_np[i], H, W)
        score = float(w_boundary) * _calc_mboundf(m_pred, m_gt)
        if w_dice != 0:
            score += float(w_dice) * _calc_dice(m_pred, m_gt)
        if w_iou != 0:
            score += float(w_iou) * _calc_iou(m_pred, m_gt)
        scores.append(score)

    return torch.tensor(scores, device=device, dtype=torch.float32)


def compute_region_reward(i_it_py: torch.Tensor,
                          disp: torch.Tensor,
                          gt_py: torch.Tensor,
                          H: int,
                          W: int,
                          w1: float = 1.0,
                          w_dice: float = 0.0,
                          w_iou: float = 0.0,
                          coord_scale: float = 1.0) -> torch.Tensor:
    """Compute contour reward from boundary quality, Dice, and optional mask IoU."""
    return compute_region_score(
        i_it_py + disp,
        gt_py,
        H=H,
        W=W,
        w_boundary=w1,
        w_dice=w_dice,
        w_iou=w_iou,
        coord_scale=coord_scale,
    )
