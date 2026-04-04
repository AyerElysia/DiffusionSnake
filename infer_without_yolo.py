"""
简化的推理脚本 - 从 GT 框生成轮廓预测
"""
import os
import sys
import cv2
import torch
import numpy as np
import random
import datetime
from pathlib import Path

# 默认配置文件
_THIS_DIR = os.path.dirname(__file__)
_DEFAULT_CFG = os.path.join(_THIS_DIR, 'configs', 'diffusion_snake.yaml')
_argv_lower = [a.lower() for a in sys.argv]
_has_cli_cfg = ('--cfg_file' in _argv_lower) or ('--cfg-file' in _argv_lower)
if (not _has_cli_cfg) and (not os.environ.get('CFG_FILE')):
    os.environ['CFG_FILE'] = os.environ.get('INFER_CFG_FILE', _DEFAULT_CFG)

from lib.config import cfg, args
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.datasets.collate_batch import make_collator
from lib.utils.snake import snake_config, snake_decode, snake_gcn_utils
from lib.utils import data_utils


# ======== 工具函数 ========

def to_numpy(x):
    """Tensor 转 numpy"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def filter_valid_polys(polys):
    """过滤无效多边形"""
    if polys is None:
        return []
    polys_np = np.asarray(polys)
    valid = []
    for poly in polys_np:
        if poly is None or poly.size == 0 or np.allclose(poly, 0):
            continue
        x_span = float(np.max(poly[:, 0]) - np.min(poly[:, 0]))
        y_span = float(np.max(poly[:, 1]) - np.min(poly[:, 1]))
        if x_span >= 1 and y_span >= 1:
            valid.append(poly)
    return valid


def draw_poly(img, poly, color, thickness=2, closed=True):
    """在图像上画多边形"""
    poly = np.concatenate([poly, poly[:1]], axis=0) if closed else poly
    cv2.polylines(img, [poly.astype(np.int32)], isClosed=closed, color=color, thickness=thickness)


def draw_results(img, pred_poly, init_poly=None, gt_poly=None, save_path=None):
    """可视化结果"""
    img = img.copy()

    # 初始轮廓 (黄色)
    if init_poly is not None and init_poly.shape[0] > 0:
        for k in range(init_poly.shape[0]):
            draw_poly(img, init_poly[k], (0, 255, 255), thickness=1)

    # 预测轮廓 (红色)
    for poly in filter_valid_polys(pred_poly):
        draw_poly(img, poly, (0, 0, 255), thickness=2)

    # GT 轮廓 (蓝色)
    for poly in filter_valid_polys(gt_poly):
        draw_poly(img, poly, (255, 0, 0), thickness=2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)


# ======== 加载模型 ========

def load_model():
    """加载模型和 checkpoint"""
    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True

    network = make_network(cfg)
    trainer = make_trainer(cfg, network)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        trainer.network.to(device)
    except Exception:
        pass

    # 获取 checkpoint 路径
    cfg_stem = Path(str(getattr(args, 'cfg_file', '') or os.environ.get('CFG_FILE', ''))).stem or 'default'
    default_ckpt = os.path.join(_THIS_DIR, 'data', 'outputs', cfg_stem, 'checkpoints', 'latest.pt')
    ckpt_path = getattr(args, 'ckpt', '') or os.environ.get('ONE_SAMPLE_CKPT', '') or default_ckpt

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 加载权重 (容错模式)
    ckpt_obj = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt_obj.get('state_dict') or ckpt_obj.get('model') or ckpt_obj.get('net') or ckpt_obj
    model = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network

    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_sd and isinstance(v, torch.Tensor) and v.shape == model_sd[k].shape}
    model.load_state_dict(filtered, strict=False)

    return model.to(device), device


# ======== GT -> Detection ========

def gt_to_detection(batch, down_ratio):
    """从 GT 多边形构造检测框"""
    gt_all = batch['i_gt_py']
    B, M = gt_all.size(0), gt_all.size(1)

    # 获取掩码
    if 'ct_01' in batch:
        masks = [batch['ct_01'][b].bool() for b in range(B)]
    elif 'ct_num' in batch['meta']:
        masks = []
        for b in range(B):
            ct_num = int(batch['meta']['ct_num'][b].item())
            m = torch.zeros((M,), dtype=torch.bool, device=gt_all.device)
            m[:ct_num] = True
            masks.append(m)
    else:
        masks = [torch.ones((M,), dtype=torch.bool, device=gt_all.device) for _ in range(B)]

    # GT -> bbox
    det_list, max_det = [], 0
    for b in range(B):
        m = masks[b]
        if not m.any():
            det_list.append(torch.zeros((0, 6), device=gt_all.device))
            continue

        gt_b = gt_all[b][m] * down_ratio  # 特征坐标 -> 图像坐标
        x1, y1 = gt_b[..., 0].min(dim=1).values, gt_b[..., 1].min(dim=1).values
        x2, y2 = gt_b[..., 0].max(dim=1).values, gt_b[..., 1].max(dim=1).values

        # 确保 (x1,y1) 是左上角
        x1, x2 = torch.minimum(x1, x2), torch.maximum(x1, x2)
        y1, y2 = torch.minimum(y1, y2), torch.maximum(y1, y2)

        det_b = torch.stack([x1, y1, x2, y2, torch.ones_like(x1), torch.zeros_like(x1)], dim=-1)
        det_list.append(det_b)
        max_det = max(max_det, det_b.size(0))

    # Pad 到相同长度
    detection = torch.zeros((B, max_det, 6), device=gt_all.device)
    for b, det_b in enumerate(det_list):
        if det_b.numel() > 0:
            detection[b, :det_b.size(0)] = det_b

    return detection


# ======== 推理 ========

def infer(model, device):
    """单样本推理"""
    # 加载数据
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, is_train=False), is_train=False)
    sample = dataset[random.randint(0, len(dataset) - 1)]
    batch = make_collator(cfg)([sample])

    # 转到 GPU
    for k, v in batch.items():
        if k != 'meta' and isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    # 获取原始图像
    inp_np = to_numpy(batch['orig_img'][0])
    if inp_np.max() <= 255.0:
        inp_np = np.clip(inp_np, 0, 255).astype(np.uint8)
    else:
        inp_np = ((inp_np - inp_np.min()) / max(inp_np.max(), 1e-5) * 255).astype(np.uint8)

    # 准备变换矩阵
    center, scale = to_numpy(batch['meta']['center'][0]), to_numpy(batch['meta']['scale'][0])
    input_w, input_h = int(snake_config.voc_input_w), int(snake_config.voc_input_h)
    trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h]).astype(np.float32)

    def apply_affine(pts):
        return data_utils.affine_transform(pts.reshape(-1, 2), trans_input).reshape(pts.shape)

    # 推理
    model.eval()
    with torch.no_grad():
        core = model.net if hasattr(model, 'net') else model

        # 提取特征
        yolo_out = core.yolo(batch['inp'])
        yolo_feats = yolo_out[1] if isinstance(yolo_out, tuple) and len(yolo_out) >= 2 else []
        p2 = yolo_feats[0] if yolo_feats else None
        if p2 is None:
            raise RuntimeError('YOLO P2 feature not found')

        cnn_feature = core.cnn_proj(p2)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        dr = float(snake_config.down_ratio)

        # GT -> detection
        detection = gt_to_detection(batch, dr)

        # Detection -> 初始轮廓
        rect4_all = snake_decode.get_box(detection[..., :4]) / dr
        mask = detection[..., 4] > 1e-4
        rect4_sel = rect4_all[mask]

        if rect4_sel.size(0) > 0:
            i_it_py = snake_gcn_utils.uniform_upsample(rect4_sel.unsqueeze(0), snake_config.poly_num)[0]
        else:
            i_it_py = torch.zeros([0, snake_config.poly_num, 2], device=cnn_feature.device)

        # 构造 img_inds
        img_inds = [torch.full((int(mask[i].sum().item()),), i, device=cnn_feature.device, dtype=torch.long)
                    for i in range(mask.size(0)) if mask[i].any()]
        ind = torch.cat(img_inds) if img_inds else torch.zeros((0,), device=cnn_feature.device, dtype=torch.long)

        # 扩散采样
        pred_polys = [None]
        if i_it_py.size(0) > 0:
            c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)
            disp = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, ind, steps=50)
            pred_polys[0] = (i_it_py + disp).detach().float().cpu().numpy() * dr

    # 准备可视化数据
    init_poly = i_it_py.detach().float().cpu().numpy() * dr if i_it_py.numel() > 0 else None

    # GT
    if 'ct_01' in batch:
        mask = batch['ct_01'][0].bool()
    elif 'ct_num' in batch['meta']:
        ct_num = int(batch['meta']['ct_num'][0].item())
        mask = torch.zeros((batch['i_gt_py'].shape[1],), dtype=torch.bool, device=batch['i_gt_py'].device)
        mask[:ct_num] = True
    else:
        mask = None

    gt_poly = to_numpy(batch['i_gt_py'][0][mask]) * dr if mask is not None else to_numpy(batch['i_gt_py'][0]) * dr

    # 保存
    save_dir = os.path.join(_THIS_DIR, 'visual', 'diffusion_one_sample')
    os.makedirs(save_dir, exist_ok=True)

    img_basename = os.path.splitext(os.path.basename(str(batch['img_path'][0])))[0] if 'img_path' in batch else None
    ds_tag = str(getattr(cfg, 'test', {}).get('dataset', 'dataset'))
    tag = f"{ds_tag}_{img_basename}_rand{random.randint(0, 10000)}" if img_basename else f"{ds_tag}_rand{random.randint(0, 10000)}"
    ts_prefix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'{ts_prefix}_{tag}.png')

    draw_results(inp_np, pred_polys[0], init_poly, gt_poly, save_path)
    print(f"Saved to {save_path}")


# ======== 主函数 ========

def main():
    model, device = load_model()
    infer(model, device)


if __name__ == '__main__':
    main()
