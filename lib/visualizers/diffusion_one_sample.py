import os
import cv2
import numpy as np
import torch

from lib.utils import data_utils
from lib.utils.snake import snake_config


def draw_results(orig_img_bgr, det_b, pred_poly, gt_poly, save_path,
                 init_poly=None, gt4_poly=None):
    img = orig_img_bgr.copy()

    # draw detection boxes (green)
    if det_b is not None and det_b.size(0) > 0:
        for i in range(det_b.shape[0]):
            x1, y1, x2, y2, score, cls_id = det_b[i, :6]
            # Convert tensors to Python floats if needed
            x1 = float(x1.item()) if hasattr(x1, 'item') else float(x1)
            y1 = float(y1.item()) if hasattr(y1, 'item') else float(y1)
            x2 = float(x2.item()) if hasattr(x2, 'item') else float(x2)
            y2 = float(y2.item()) if hasattr(y2, 'item') else float(y2)
            score = float(score.item()) if hasattr(score, 'item') else float(score)
            cls_id = int(cls_id.item()) if hasattr(cls_id, 'item') else int(cls_id)

            if score <= 0:
                continue
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 1)
            cv2.putText(img, f"{cls_id}:{score:.2f}", (p1[0], max(0, p1[1]-2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    # draw predicted poly (red)
    if pred_poly is not None and pred_poly.shape[0] > 0:
        for k in range(pred_poly.shape[0]):
            poly = pred_poly[k]
            poly = np.concatenate([poly, poly[:1]], axis=0)
            cv2.polylines(img, [poly.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

    # draw GT poly (blue)
    if gt_poly is not None and gt_poly.shape[0] > 0:
        for k in range(gt_poly.shape[0]):
            poly = gt_poly[k]
            poly = np.concatenate([poly, poly[:1]], axis=0)
            cv2.polylines(img, [poly.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

    # draw init poly (yellow)
    if init_poly is not None and init_poly.shape[0] > 0:
        for k in range(init_poly.shape[0]):
            poly = init_poly[k]
            poly = np.concatenate([poly, poly[:1]], axis=0)
            cv2.polylines(img, [poly.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=1)

    # draw GT 4-point poly (magenta with points)
    if gt4_poly is not None and gt4_poly.shape[0] > 0:
        for k in range(gt4_poly.shape[0]):
            pts = gt4_poly[k].astype(np.int32)
            # draw lines
            loop = np.concatenate([pts, pts[:1]], axis=0)
            cv2.polylines(img, [loop], isClosed=True, color=(255, 0, 255), thickness=1)
            # draw points
            for p in pts:
                cv2.circle(img, (int(p[0]), int(p[1])), 2, (255, 0, 255), -1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def save_affine_visualization(*, output: dict, batch: dict, tag: str, save_dir: str):
    if batch is None:
        return

    # Prepare affine input background image
    inp_bgr = batch['orig_img'][0]
    if isinstance(inp_bgr, torch.Tensor):
        inp_np = inp_bgr.detach().float().cpu().numpy()
        if inp_np.max() <= 255.0 and inp_np.ndim == 3 and inp_np.shape[-1] == 3:
            if inp_np.dtype != np.uint8:
                inp_np = np.clip(inp_np, 0, 255).astype(np.uint8)
        else:
            inp_np = inp_np - inp_np.min()
            if inp_np.max() > 0:
                inp_np = inp_np / inp_np.max()
            inp_np = (inp_np * 255.0).astype(np.uint8)
        inp_img = inp_np
    else:
        inp_img = np.array(inp_bgr)

    center = _to_numpy(batch['meta']['center'][0])
    scale = _to_numpy(batch['meta']['scale'][0])
    input_w, input_h = int(snake_config.voc_input_w), int(snake_config.voc_input_h)
    output_w = int(input_w // snake_config.down_ratio)
    output_h = int(input_h // snake_config.down_ratio)
    trans_input_inv = data_utils.get_affine_transform(
        center, scale, 0, [input_w, input_h], inv=1
    ).astype(np.float32)
    trans_output_inv = data_utils.get_affine_transform(
        center, scale, 0, [output_w, output_h], inv=1
    ).astype(np.float32)

    def apply_affine_pts(pts, M):
        return data_utils.affine_transform(pts.reshape(-1, 2), M).reshape(pts.shape)

    # detections in original-image coords
    det = output.get('detection', None)
    det_orig = None
    if det is not None and isinstance(det, torch.Tensor) and det.size(0) > 0:
        det_b = det[0].detach().float().cpu()
        det_raw = det_b.numpy().copy()
        for i in range(det_raw.shape[0]):
            x1, y1, x2, y2 = det_raw[i, :4]
            p = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            p_orig = apply_affine_pts(p, trans_input_inv)
            det_raw[i, 0:2] = p_orig[0]
            det_raw[i, 2:4] = p_orig[1]
        det_orig = torch.from_numpy(det_raw)

    # polys in original-image coords
    pred_py = output.get('py', None)
    pred_orig = None
    if pred_py is not None:
        last = pred_py[-1] if isinstance(pred_py, list) else pred_py
        if isinstance(last, torch.Tensor) and last.numel() > 0:
            pred_raw = last.detach().float().cpu().numpy()
            pred_orig = apply_affine_pts(pred_raw, trans_output_inv)

    init_py = output.get('it_py', None)
    init_orig = None
    if isinstance(init_py, torch.Tensor) and init_py.numel() > 0:
        init_raw = init_py.detach().float().cpu().numpy()
        init_orig = apply_affine_pts(init_raw, trans_output_inv)

    gt_orig = None
    if 'i_gt_py' in batch and 'meta' in batch and 'ct_num' in batch['meta']:
        ct_meta = batch['meta']['ct_num']
        ct_num = int(ct_meta[0].item()) if isinstance(ct_meta, torch.Tensor) else int(ct_meta)
        gt = batch['i_gt_py'][0][:ct_num]
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().float().cpu().numpy()
        gt_orig = apply_affine_pts(gt, trans_output_inv)

    gt4_orig = None
    if 'i_gt_4py' in batch and 'meta' in batch and 'ct_num' in batch['meta']:
        ct_meta = batch['meta']['ct_num']
        ct_num = int(ct_meta[0].item()) if isinstance(ct_meta, torch.Tensor) else int(ct_meta)
        gt4 = batch['i_gt_4py'][0][:ct_num]
        if isinstance(gt4, torch.Tensor):
            gt4 = gt4.detach().float().cpu().numpy()
        gt4_orig = apply_affine_pts(gt4, trans_output_inv)

    os.makedirs(save_dir, exist_ok=True)
    save_path_aff = os.path.join(save_dir, f'vis_affine_{tag}.png')
    draw_results(inp_img, det_orig, pred_orig, gt_orig, save_path_aff, init_poly=init_orig, gt4_poly=gt4_orig)
