import os
import cv2
import torch
import numpy as np

from lib.config import cfg
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.datasets import make_data_loader
from lib.utils.snake import snake_config, snake_decode, snake_gcn_utils
from lib.utils import data_utils
import datetime


def _filter_valid_polys(polys):
    valid = []
    if polys is None:
        return valid
    polys_np = np.asarray(polys)
    for poly in polys_np:
        if poly is None or poly.size == 0:
            continue
        if np.allclose(poly, 0):
            continue
        x_span = float(np.max(poly[:, 0]) - np.min(poly[:, 0]))
        y_span = float(np.max(poly[:, 1]) - np.min(poly[:, 1]))
        if x_span < 1 or y_span < 1:
            continue
        valid.append(poly)
    return valid


def draw_results(orig_img_bgr, det_b, pred_poly, save_path,
                 init_poly=None, gt_poly=None):
    img = orig_img_bgr.copy()

    if det_b is not None and isinstance(det_b, torch.Tensor) and det_b.size(0) > 0:
        for i in range(det_b.shape[0]):
            x1, y1, x2, y2, score, cls_id = det_b[i, :6]
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

    if init_poly is not None and getattr(init_poly, 'shape', [0])[0] > 0:
        for k in range(init_poly.shape[0]):
            poly = init_poly[k]
            poly = np.concatenate([poly, poly[:1]], axis=0)
            cv2.polylines(img, [poly.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=1)

    pred_valid = _filter_valid_polys(pred_poly)
    if len(pred_valid) > 0:
        for poly in pred_valid:
            poly = np.concatenate([poly, poly[:1]], axis=0)
            cv2.polylines(img, [poly.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

    gt_valid = _filter_valid_polys(gt_poly)
    if len(gt_valid) > 0:
        for poly in gt_valid:
            poly = np.concatenate([poly, poly[:1]], axis=0)
            cv2.polylines(img, [poly.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def main():
    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True

    # Build network and trainer wrapper to match training graph
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)

    # Load checkpoint
    default_ckpt = os.path.join(os.path.dirname(__file__), 'data', 'outputs', 'one_sample', 'model_grpo_20251204_233802.pth')
    ckpt_path = os.environ.get('ONE_SAMPLE_CKPT', default_ckpt)
    ckpt_obj = torch.load(ckpt_path, map_location='cpu')

    def _extract_state_dict(obj):
        if isinstance(obj, dict):
            for k in ('state_dict', 'model', 'net', 'network'):
                if k in obj and isinstance(obj[k], dict):
                    return obj[k]
            return obj
        return obj

    sd = _extract_state_dict(ckpt_obj)
    w = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    missing, unexpected = w.load_state_dict(sd, strict=False)
    if missing:
        print('[infer_one_sample] Missing keys after loading checkpoint (showing up to 8):', missing[:8])
    if unexpected:
        print('[infer_one_sample] Unexpected keys after loading checkpoint (showing up to 8):', unexpected[:8])

    # unwrap to core net for direct access
    core = w.net if hasattr(w, 'net') else w

    # Backward compatibility for displacement normalization.
    # If cfg enables disp norm but checkpoint lacks disp stats buffers, it likely was trained without normalization.
    # In that case, denormalizing would incorrectly amplify predicted displacement.
    try:
        disp_norm_cfg = bool(getattr(cfg, 'diffusion_disp_norm', False))
        if disp_norm_cfg and isinstance(missing, (list, tuple)):
            miss_set = set([str(k) for k in missing])
            if any(k.endswith('._disp_min') or k.endswith('._disp_max') for k in miss_set):
                print('[infer_one_sample] WARNING: cfg.diffusion_disp_norm=True but checkpoint has no _disp_min/_disp_max buffers.')
                print('[infer_one_sample] WARNING: This usually means the checkpoint was trained WITHOUT displacement normalization.')
                print('[infer_one_sample] WARNING: Disabling displacement normalization for this inference run to avoid double-scaling.')
                try:
                    cfg.diffusion_disp_norm = False
                except Exception:
                    pass
                try:
                    if hasattr(core, 'gcn') and hasattr(core.gcn, '_disp_norm_enabled'):
                        core.gcn._disp_norm_enabled = False
                except Exception:
                    pass
    except Exception:
        pass

    # Data loader (eval split) and take the first batch
    data_loader = make_data_loader(cfg, is_train=False)
    batch = next(iter(data_loader))

    # Move tensors to CUDA (except meta)
    for k in list(batch.keys()):
        if k == 'meta':
            continue
        v = batch[k]
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda(non_blocking=True)

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

    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    center = _to_numpy(batch['meta']['center'][0])
    scale = _to_numpy(batch['meta']['scale'][0])
    input_w, input_h = int(snake_config.voc_input_w), int(snake_config.voc_input_h)
    trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h]).astype(np.float32)

    def apply_affine_pts(pts, M):
        return data_utils.affine_transform(pts.reshape(-1, 2), M).reshape(pts.shape)

    # ========= Round 0: YOLO detection -> init poly =========
    core.eval()
    with torch.no_grad():
        # run YOLO head to get yolo_y and features
        yolo_out = core.yolo(batch['inp'])
        if isinstance(yolo_out, tuple) and len(yolo_out) >= 2:
            yolo_y, yolo_feats = yolo_out[0], yolo_out[1]
        else:
            yolo_y, yolo_feats = yolo_out, []
        p2 = yolo_feats[0] if isinstance(yolo_feats, (list, tuple)) and len(yolo_feats) > 0 else None
        if p2 is None:
            raise RuntimeError('YOLO P2 feature not found')
        cnn_feature = core.cnn_proj(p2)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        dr = float(snake_config.down_ratio)
        h_img, w_img = int(round(h * dr)), int(round(w * dr))
        raw_det = core.decode_detection_from_yolo(yolo_y, h_img, w_img)
        # optional NMS consistent with core.forward
        use_nms = getattr(cfg, 'use_nms_for_snake', True)
        if use_nms:
            from lib.networks.YOLOV8.utils.ops import non_max_suppression
            y = yolo_y.permute(0, 2, 1).contiguous()
            xywh = y[..., :4]
            cls_logits = y[..., 4:]
            cls_prob = cls_logits.sigmoid()
            x_c, y_c, w_box, h_box = xywh.unbind(-1)
            x1 = x_c - w_box / 2
            y1 = y_c - h_box / 2
            x2 = x_c + w_box / 2
            y2 = y_c + h_box / 2
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            boxes = data_utils.clip_to_image(boxes, h_img, w_img)
            pred = torch.cat([boxes, cls_prob], dim=-1).permute(0, 2, 1).contiguous()
            nms_out = non_max_suppression(
                pred,
                conf_thres=float(getattr(cfg, 'det_conf_thresh', 0.20)),
                iou_thres=float(getattr(cfg, 'det_iou_thresh', 0.30)),
                classes=None,
                agnostic=not bool(getattr(cfg, 'per_class_nms', True)),
                multi_label=False,
                labels=(),
                max_det=int(getattr(cfg, 'det_max_det', 300)),
                nc=cls_prob.shape[-1],
                in_place=True,
                rotated=True,
            )
            max_len = max((d.size(0) for d in nms_out), default=0)
            if max_len == 0:
                detection = raw_det.new_zeros((raw_det.size(0), 0, 6))
            else:
                detection = raw_det.new_zeros((raw_det.size(0), max_len, 6))
                for b, det_b in enumerate(nms_out):
                    if det_b is not None and det_b.size(0) > 0:
                        detection[b, :det_b.size(0)] = det_b[:, :6]
        else:
            detection = raw_det

        # map detection to affine-input for visualization only
        det_aff = None
        if detection is not None and detection.numel() > 0:
            det_b = detection[0].detach().float().cpu().numpy().copy()
            for i in range(det_b.shape[0]):
                x1, y1, x2, y2 = det_b[i, :4]
                p = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                p_aff = apply_affine_pts(p, trans_input)
                det_b[i, 0:2] = p_aff[0]
                det_b[i, 2:4] = p_aff[1]
            det_aff = torch.from_numpy(det_b)

        # build init poly from boxes on feature-map coords
        rect4_all = snake_decode.get_box(detection[..., :4])  # [B, M, 4, 2]
        rect4_feat = rect4_all / dr
        det_score = detection[..., 4]
        mask = det_score > 1e-4
        rect4_sel = rect4_feat[mask]

        if rect4_sel.size(0) > 0:
            i_it_py = snake_gcn_utils.uniform_upsample(rect4_sel.unsqueeze(0), snake_config.poly_num)[0]
            c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)
        else:
            i_it_py = torch.zeros([0, snake_config.poly_num, 2], device=cnn_feature.device)
            c_it_py = torch.zeros_like(i_it_py)

        # build contour-to-image index
        img_inds = []
        B = mask.size(0)
        for i in range(B):
            cnt = int(mask[i].sum().item())
            if cnt > 0:
                img_inds.append(torch.full((cnt,), i, device=cnn_feature.device, dtype=torch.long))
        ind = torch.cat(img_inds, dim=0) if len(img_inds) else torch.zeros((0,), device=cnn_feature.device, dtype=torch.long)

        # ========= Round 1 evolution (only) =========
        pred_polys = [None]
        if i_it_py.size(0) > 0:
            disp1 = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, ind, steps=50)
            py1 = i_it_py + disp1
            pred_polys[0] = py1.detach().float().cpu().numpy() * float(snake_config.down_ratio)

    gt_aff = None
    if 'i_gt_py' in batch:
        mask = None
        if 'ct_01' in batch:
            mask = batch['ct_01'][0].bool()
        elif 'meta' in batch and 'ct_num' in batch['meta']:
            ct_meta = batch['meta']['ct_num']
            ct_num = int(ct_meta[0].item()) if isinstance(ct_meta, torch.Tensor) else int(ct_meta)
            mask = torch.zeros(
                (batch['i_gt_py'].shape[1],),
                dtype=torch.bool,
                device=batch['i_gt_py'].device,
            )
            mask[:ct_num] = True
        gt = batch['i_gt_py'][0][mask] if mask is not None else batch['i_gt_py'][0]
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().float().cpu().numpy()
        gt_aff = gt * float(snake_config.down_ratio)

    # also visualize init poly in affine coords
    init_aff = None
    if 'i_it_py' in locals() and isinstance(i_it_py, torch.Tensor) and i_it_py.numel() > 0:
        init_aff = i_it_py.detach().float().cpu().numpy() * float(snake_config.down_ratio)

    save_dir = os.path.join(os.path.dirname(__file__), 'visual', 'diffusion_one_sample')
    os.makedirs(save_dir, exist_ok=True)
    img_basename = None
    if 'img_path' in batch:
        try:
            img_path0 = batch['img_path'][0]
            img_basename = os.path.splitext(os.path.basename(str(img_path0)))[0]
        except Exception:
            img_basename = None
    ds_tag = str(getattr(cfg, 'test', {}).dataset) if hasattr(cfg, 'test') and hasattr(cfg.test, 'dataset') else 'dataset'
    tag = f"{ds_tag}" if not img_basename else f"{ds_tag}_{img_basename}"
    ts_prefix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path_r1 = os.path.join(save_dir, f'{ts_prefix}_{tag}_r1.png')

    draw_results(inp_img, det_aff, pred_polys[0], save_path_r1, init_poly=init_aff, gt_poly=gt_aff)


if __name__ == '__main__':
    main()
