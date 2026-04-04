


import os
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# IMPORTANT: lib.config 会在 import 时就解析 argv/环境变量并加载默认 cfg 文件。
# 这里先设置 CFG_FILE 的默认值，避免默认回落到 configs/sbd_snake.yaml。
_THIS_DIR = os.path.dirname(__file__)
_DEFAULT_CFG = os.path.join(_THIS_DIR, 'configs', 'diffusion_snake.yaml')
os.environ.setdefault('CFG_FILE', _DEFAULT_CFG)

from lib.config import cfg
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.datasets import make_data_loader
from lib.utils.snake import snake_config, snake_decode, snake_gcn_utils
from lib.utils import data_utils


def _filter_valid_polys(polys):
    """过滤掉全零或面积极小的无效多边形。"""
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
            cv2.rectangle(img, p1, p2, (0, 255, 0), 1)
            cv2.putText(img, f"{cls_id}:{score:.2f}", (p1[0], max(0, p1[1]-2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

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

def draw_quiver_like(orig_img_bgr, base_pts, vec, save_path, gt_poly=None):
    img = orig_img_bgr.copy()
    if base_pts is None:
        cv2.imwrite(save_path, img)
        return
    K = base_pts.shape[0]
    if K == 0:
        cv2.imwrite(save_path, img)
        return
    P = base_pts.shape[1]
    stride = max(1, P // 32)
    Xs, Ys, Us, Vs = [], [], [], []
    for k in range(K):
        pts = base_pts[k]
        vv = vec[k]
        Xs.append(pts[::stride, 0])
        Ys.append(pts[::stride, 1])
        Us.append(vv[::stride, 0])
        Vs.append(vv[::stride, 1])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    U = np.concatenate(Us, axis=0)
    V = np.concatenate(Vs, axis=0)
    M = np.sqrt(U * U + V * V)
    eps = 1e-6
    Un = np.where(M > eps, U / (M + eps), 0.0)
    Vn = np.where(M > eps, V / (M + eps), 0.0)
    H, W = img.shape[0], img.shape[1]
    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if M.size > 0:
        fixed_len = max(H, W) / 40.0
        plt.quiver(X, Y, Un * fixed_len, Vn * fixed_len, M, angles='xy', scale_units='xy', scale=1.0, cmap='coolwarm', width=0.003)
    if gt_poly is not None and getattr(gt_poly, 'shape', [0])[0] > 0:
        for k in range(gt_poly.shape[0]):
            poly = gt_poly[k]
            poly = np.concatenate([poly, poly[:1]], axis=0)
            plt.plot(poly[:, 0], poly[:, 1], color=(0/255.0, 0/255.0, 255/255.0), linewidth=1.5)
    plt.gca().invert_yaxis()
    plt.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def draw_quiver_pure_single(base_pts, vec, save_path, gt_poly=None, stride=None, fixed_len=None, width=0.002):
    if base_pts is None or vec is None:
        fig = plt.figure(figsize=(4, 4), dpi=120)
        plt.axis('off')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return
    pts = np.asarray(base_pts)
    vv = np.asarray(vec)
    P = pts.shape[0]
    if stride is None:
        stride = max(1, P // 48)
    xs = pts[::stride, 0]
    ys = pts[::stride, 1]
    us = vv[::stride, 0]
    vs = vv[::stride, 1]
    m = np.sqrt(us * us + vs * vs)
    eps = 1e-6
    un = np.where(m > eps, us / (m + eps), 0.0)
    vn = np.where(m > eps, vs / (m + eps), 0.0)
    xmin, xmax = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
    ymin, ymax = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
    dx, dy = xmax - xmin, ymax - ymin
    dx = dx if dx > 1 else 1.0
    dy = dy if dy > 1 else 1.0
    if fixed_len is None:
        fixed_len = max(dx, dy) / 60.0
    fig_w = max(dx, 200.0) / 100.0
    fig_h = max(dy, 200.0) / 100.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)
    if m.size > 0:
        plt.quiver(xs, ys, un * fixed_len, vn * fixed_len, m, angles='xy', scale_units='xy', scale=1.0, cmap='coolwarm', width=width)
    if gt_poly is not None and getattr(gt_poly, 'shape', [0])[0] > 0:
        g = gt_poly[0] if gt_poly.ndim == 3 else gt_poly
        loop = np.concatenate([g, g[:1]], axis=0)
        plt.plot(loop[:, 0], loop[:, 1], color=(0, 0, 1), linewidth=1.0)
    plt.xlim(xmin - 10, xmax + 10)
    plt.ylim(ymin - 10, ymax + 10)
    plt.gca().invert_yaxis()
    plt.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def draw_quiver_pure_combined(base_pts_list, vec_list, save_path, gt_poly=None, stride=None, fixed_len=None, width=0.001):
    items = []
    for base_pts, vec in zip(base_pts_list, vec_list):
        if base_pts is None or vec is None:
            continue
        pts = np.asarray(base_pts)
        vv = np.asarray(vec)
        P = pts.shape[0]
        s = stride if stride is not None else max(1, P // 48)
        xs = pts[::s, 0]; ys = pts[::s, 1]
        us = vv[::s, 0]; vs = vv[::s, 1]
        items.append((xs, ys, us, vs))
    if len(items) == 0:
        fig = plt.figure(figsize=(4, 4), dpi=120)
        plt.axis('off')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return
    xs_all = np.concatenate([it[0] for it in items], axis=0)
    ys_all = np.concatenate([it[1] for it in items], axis=0)
    xmin, xmax = float(np.min(xs_all)), float(np.max(xs_all))
    ymin, ymax = float(np.min(ys_all)), float(np.max(ys_all))
    dx, dy = xmax - xmin, ymax - ymin
    dx = dx if dx > 1 else 1.0
    dy = dy if dy > 1 else 1.0
    fl = fixed_len if fixed_len is not None else max(dx, dy) / 70.0
    fig_w = max(dx, 200.0) / 100.0
    fig_h = max(dy, 200.0) / 100.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)
    for xs, ys, us, vs in items:
        m = np.sqrt(us * us + vs * vs)
        eps = 1e-6
        un = np.where(m > eps, us / (m + eps), 0.0)
        vn = np.where(m > eps, vs / (m + eps), 0.0)
        plt.quiver(xs, ys, un * fl, vn * fl, m, angles='xy', scale_units='xy', scale=1.0, cmap='coolwarm', width=width)
    if gt_poly is not None and getattr(gt_poly, 'shape', [0])[0] > 0:
        g = gt_poly[0] if gt_poly.ndim == 3 else gt_poly
        loop = np.concatenate([g, g[:1]], axis=0)
        plt.plot(loop[:, 0], loop[:, 1], color=(0, 0, 1), linewidth=0.8)
    plt.xlim(xmin - 10, xmax + 10)
    plt.ylim(ymin - 10, ymax + 10)
    plt.gca().invert_yaxis()
    plt.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def main():
    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True

    network = make_network(cfg)
    trainer = make_trainer(cfg, network)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        trainer.network.to(device)
    except Exception:
        pass

    #default_ckpt = os.path.join(os.path.dirname(__file__), 'data', 'outputs', 'one_sample', 'model_final.pth')
    #default_ckpt = os.path.join(os.path.dirname(__file__), 'data/one_sample/model_final.pth')
    default_ckpt = os.path.join(os.path.dirname(__file__), 'data', 'outputs', 'one_sample', 'checkpoints', 'latest.pt')
    #default_ckpt = os.path.join(os.path.dirname(__file__), 'data', 'one_sampleWithOneModifying','model_final.pth')
    ckpt_path = os.environ.get('ONE_SAMPLE_CKPT', default_ckpt)

    def _extract_state_dict(ckpt_obj):
        if isinstance(ckpt_obj, dict):
            for k in ('state_dict', 'model', 'net', 'network'):
                if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                    return ckpt_obj[k]
        if isinstance(ckpt_obj, dict):
            # 有些保存方式直接就是 state_dict
            return ckpt_obj
        return ckpt_obj

    ckpt_obj = torch.load(ckpt_path, map_location='cpu')
    sd = _extract_state_dict(ckpt_obj)
    w = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    try:
        w.load_state_dict(sd, strict=False)
    except Exception:
        # 兼容部分权重名不一致的情况
        w.load_state_dict(sd, strict=False)
    
    core = w.net if hasattr(w, 'net') else w

    data_loader = make_data_loader(cfg, is_train=False)
    batch = next(iter(data_loader))

    for k in list(batch.keys()):
        if k == 'meta':
            continue
        v = batch[k]
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)

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

    core.eval()
    with torch.no_grad():
        # YOLO
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
        h_img, w_img = h * 4, w * 4

        # decode + optional NMS (一致)
        raw_det = core.decode_detection_from_yolo(yolo_y, h_img, w_img)
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

        # det 可视化坐标
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

        # Use dynamic initialization (handles octagon if specified in config)
        rect4_all = snake_decode.get_init(detection[..., :4])  # [B, M, V, 2]
        dr = float(snake_config.down_ratio)
        rect4_feat = rect4_all / dr
        det_score = detection[..., 4]
        mask = det_score > 1e-4
        rect4_sel = rect4_feat[mask]

        if rect4_sel.size(0) > 0:
            i_it_py = snake_gcn_utils.uniform_upsample(rect4_sel.unsqueeze(0), snake_config.poly_num)[0]
        else:
            i_it_py = torch.zeros([0, snake_config.poly_num, 2], device=cnn_feature.device)

        # contour-to-image mapping
        img_inds = []
        B = mask.size(0)
        for i in range(B):
            cnt = int(mask[i].sum().item())
            if cnt > 0:
                img_inds.append(torch.full((cnt,), i, device=cnn_feature.device, dtype=torch.long))
        ind = torch.cat(img_inds, dim=0) if len(img_inds) else torch.zeros((0,), device=cnn_feature.device, dtype=torch.long)

        pred_polys = [None, None, None, None, None, None]
        if i_it_py.size(0) > 0:
            c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)
            disp1 = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, ind, steps=50)
            print(disp1)
            print(disp1.shape)
            py1 = i_it_py + disp1
            c_it_py2 = snake_gcn_utils.img_poly_to_can_poly(py1)
            disp2 = core.gcn.sample_disp(cnn_feature, py1, c_it_py2, ind, steps=50)
            py2 = py1 + disp2 / 2.0
            c_it_py3 = snake_gcn_utils.img_poly_to_can_poly(py2)
            disp3 = core.gcn.sample_disp(cnn_feature, py2, c_it_py3, ind, steps=50)
            py3 = py2 + disp3

            pred_polys[0] = py1.detach().float().cpu().numpy() * dr
            pred_polys[1] = py2.detach().float().cpu().numpy() * dr
            pred_polys[2] = py3.detach().float().cpu().numpy() * dr
            c_it_py4 = snake_gcn_utils.img_poly_to_can_poly(py3)
            disp4 = core.gcn.sample_disp(cnn_feature, py3, c_it_py4, ind, steps=50)
            py4 = py3 + disp4
            pred_polys[3] = py4.detach().float().cpu().numpy() * dr
            c_it_py5 = snake_gcn_utils.img_poly_to_can_poly(py4)
            disp5 = core.gcn.sample_disp(cnn_feature, py4, c_it_py5, ind, steps=50)
            py5 = py4 + disp5
            pred_polys[4] = py5.detach().float().cpu().numpy() * dr
            c_it_py6 = snake_gcn_utils.img_poly_to_can_poly(py5)
            disp6 = core.gcn.sample_disp(cnn_feature, py5, c_it_py6, ind, steps=50)
            py6 = py5 + disp6
            pred_polys[5] = py6.detach().float().cpu().numpy() * dr

        init_aff = i_it_py.detach().float().cpu().numpy() * dr if i_it_py.numel() > 0 else None

    # optional: GT for visualization
    gt_aff = None
    if 'i_gt_py' in batch:
        # 优先使用 ct_01 掩码，保证取到所有非空 GT 多边形
        mask = None
        if 'ct_01' in batch:
            mask = batch['ct_01'][0].bool()
        elif 'meta' in batch and 'ct_num' in batch['meta']:
            ct_meta = batch['meta']['ct_num']
            ct_num = int(ct_meta[0].item()) if isinstance(ct_meta, torch.Tensor) else int(ct_meta)
            mask = torch.zeros((batch['i_gt_py'].shape[1],), dtype=torch.bool)
            mask[:ct_num] = True
        gt = batch['i_gt_py'][0][mask] if mask is not None else batch['i_gt_py'][0]
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().float().cpu().numpy()
        gt_aff = gt * float(snake_config.down_ratio)

    # save three images separately
    save_dir = os.path.join(os.path.dirname(__file__), 'visual', 'diffusion_one_sample')
    os.makedirs(save_dir, exist_ok=True)
    save_path_r1 = os.path.join(save_dir, 'vis_affine_infer_frac_r1.png')
    save_path_r2 = os.path.join(save_dir, 'vis_affine_infer_frac_r2.png')
    save_path_r3 = os.path.join(save_dir, 'vis_affine_infer_frac_r3.png')
    save_path_r4 = os.path.join(save_dir, 'vis_affine_infer_frac_r4.png')
    save_path_r5 = os.path.join(save_dir, 'vis_affine_infer_frac_r5.png')
    save_path_r6 = os.path.join(save_dir, 'vis_affine_infer_frac_r6.png')

    draw_results(inp_img, det_aff, pred_polys[0], save_path_r1, init_poly=init_aff, gt_poly=gt_aff)
    draw_results(inp_img, det_aff, pred_polys[1], save_path_r2, init_poly=init_aff, gt_poly=gt_aff)
    draw_results(inp_img, det_aff, pred_polys[2], save_path_r3, init_poly=init_aff, gt_poly=gt_aff)
    if pred_polys[3] is not None:
        draw_results(inp_img, det_aff, pred_polys[3], save_path_r4, init_poly=init_aff, gt_poly=gt_aff)
    if pred_polys[4] is not None:
        draw_results(inp_img, det_aff, pred_polys[4], save_path_r5, init_poly=init_aff, gt_poly=gt_aff)
    if pred_polys[5] is not None:
        draw_results(inp_img, det_aff, pred_polys[5], save_path_r6, init_poly=init_aff, gt_poly=gt_aff)

    save_path_r1_vec = os.path.join(save_dir, 'vis_affine_infer_frac_r1_vec.png')
    save_path_r2_vec = os.path.join(save_dir, 'vis_affine_infer_frac_r2_vec.png')
    save_path_r3_vec = os.path.join(save_dir, 'vis_affine_infer_frac_r3_vec.png')
    if pred_polys[0] is not None:
        base1 = (i_it_py.detach().float().cpu().numpy() * dr) if i_it_py.numel() > 0 else None
        vec1 = (disp1.detach().float().cpu().numpy() * dr)
        draw_quiver_like(inp_img, base1, vec1, save_path_r1_vec, gt_poly=gt_aff)
    if pred_polys[1] is not None:
        base2 = pred_polys[0]
        vec2 = (disp2.detach().float().cpu().numpy() * dr)
        draw_quiver_like(inp_img, base2, vec2, save_path_r2_vec, gt_poly=gt_aff)
    if pred_polys[2] is not None:
        base3 = pred_polys[1]
        vec3 = (disp3.detach().float().cpu().numpy() * dr)
        draw_quiver_like(inp_img, base3, vec3, save_path_r3_vec, gt_poly=gt_aff)

    save_path_r1_vec_single = os.path.join(save_dir, 'vis_affine_infer_frac_r1_vec_single.png')
    save_path_r2_vec_single = os.path.join(save_dir, 'vis_affine_infer_frac_r2_vec_single.png')
    save_path_r3_vec_single = os.path.join(save_dir, 'vis_affine_infer_frac_r3_vec_single.png')
    save_path_combined_single = os.path.join(save_dir, 'vis_affine_infer_frac_vec_combined_single.png')
    if i_it_py.size(0) > 0:
        b1s = (i_it_py[0].detach().float().cpu().numpy() * dr)
        v1s = (disp1[0].detach().float().cpu().numpy() * dr)
        b2s = pred_polys[0][0] if pred_polys[0] is not None and len(pred_polys[0]) > 0 else None
        v2s = (disp2[0].detach().float().cpu().numpy() * dr)
        b3s = pred_polys[1][0] if pred_polys[1] is not None and len(pred_polys[1]) > 0 else None
        v3s = (disp3[0].detach().float().cpu().numpy() * dr)
        def _fl(b):
            if b is None or b.size == 0:
                return None
            xmin, xmax = float(np.min(b[:, 0])), float(np.max(b[:, 0]))
            ymin, ymax = float(np.min(b[:, 1])), float(np.max(b[:, 1]))
            dx, dy = xmax - xmin, ymax - ymin
            dx = dx if dx > 1 else 1.0
            dy = dy if dy > 1 else 1.0
            return max(dx, dy) / 80.0
        fl1 = _fl(b1s)
        fl2 = _fl(b2s)
        fl3 = _fl(b3s)
        # if b1s is not None:
        #     draw_quiver_pure_single(b1s, v1s, save_path_r1_vec_single, gt_poly=gt_aff, fixed_len=fl1, width=0.001)
        # if b2s is not None:
        #     draw_quiver_pure_single(b2s, v2s, save_path_r2_vec_single, gt_poly=gt_aff, fixed_len=fl2, width=0.001)
        # if b3s is not None:
        #     draw_quiver_pure_single(b3s, v3s, save_path_r3_vec_single, gt_poly=gt_aff, fixed_len=fl3, width=0.001)
        # flc = min([x for x in (fl1, fl2, fl3) if x is not None]) if any(x is not None for x in (fl1, fl2, fl3)) else None
        # draw_quiver_pure_combined(
        #     [b1s, b2s, b3s],
        #     [v1s, v2s, v3s],
        #     save_path_combined_single,
        #     gt_poly=gt_aff,
        #     fixed_len=flc,
        #     width=0.0008,
        # )

    # only draw red predicted contours (no det boxes, no init, no GT)
    save_path_r1_only = os.path.join(save_dir, 'vis_affine_infer_frac_r1_only_pred.png')
    save_path_r2_only = os.path.join(save_dir, 'vis_affine_infer_frac_r2_only_pred.png')
    save_path_r3_only = os.path.join(save_dir, 'vis_affine_infer_frac_r3_only_pred.png')

    draw_results(inp_img, None, pred_polys[0], save_path_r1_only, init_poly=None, gt_poly=None)
    draw_results(inp_img, None, pred_polys[1], save_path_r2_only, init_poly=None, gt_poly=None)
    draw_results(inp_img, None, pred_polys[2], save_path_r3_only, init_poly=None, gt_poly=None)


if __name__ == '__main__':
    main()
