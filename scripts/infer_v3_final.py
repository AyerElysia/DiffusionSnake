import sys, os
import torch
import numpy as np
import cv2
import datetime
import json
from pathlib import Path

# 环境与配置初始化
_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CFG = os.path.join(_THIS_DIR, 'configs', 'btcv_diffusion_dit_v3.yaml')
if not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = _DEFAULT_CFG
sys.path.insert(0, _THIS_DIR)

from lib.config import cfg, args
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.datasets.make_dataset import make_dataset
from lib.datasets.collate_batch import make_collator
from lib.datasets.transforms import make_transforms
from lib.utils.snake import snake_config, snake_decode, snake_gcn_utils

def load_v3_model(cfg_file=None):
    if cfg_file:
        cfg.merge_from_file(cfg_file)
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 获取权重路径
    ckpt_path = getattr(args, 'ckpt', '') or os.path.join(
        _THIS_DIR,
        'data/outputs',
        cfg.model_dir.split('/')[-1],
        'checkpoints',
        'latest.pt',
    )
    print(f"[*] Loading Weights: {ckpt_path}")

    if os.path.exists(ckpt_path):
        ckpt_obj = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt_obj.get('state_dict') or ckpt_obj.get('model') or ckpt_obj.get('net') or ckpt_obj

        # Remap legacy keys (keep net. prefix for NetworkWrapper)
        from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict
        sd = remap_legacy_state_dict(sd)
        wrapper = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network

        info = wrapper.load_state_dict(sd, strict=False)
        print(f"[✔] Success: {len(sd) - len(info.missing_keys)} layers matched.")
        if info.missing_keys:
            print(f"[!] Missing keys ({len(info.missing_keys)}): {info.missing_keys[:5]}...")
        if info.unexpected_keys:
            print(f"[!] Unexpected keys ({len(info.unexpected_keys)}): {info.unexpected_keys[:5]}...")
    else:
        print(f"[!] Error: Checkpoint NOT found at {ckpt_path}!")
        sys.exit(1)

    return trainer.network.to(device).eval(), device


def get_extreme_points_torch(pts, thresh=0.02):
    """
    PyTorch 版本的极值点提取，与训练时 snake_voc_utils.get_extreme_points() 完全一致。

    Args:
        pts: (N, P, 2) - 多边形点坐标
        thresh: 阈值比例，默认 0.02

    Returns:
        ex: (N, 4, 2) - 极值点 [Top, Left, Bottom, Right]
    """
    N, P, _ = pts.shape
    device = pts.device

    # 计算 bounding box
    l = pts[..., 0].min(dim=-1)[0]  # (N,)
    t = pts[..., 1].min(dim=-1)[0]
    r = pts[..., 0].max(dim=-1)[0]
    b = pts[..., 1].max(dim=-1)[0]

    w = r - l + 1
    h = b - t + 1

    results = []

    for i in range(N):
        poly_i = pts[i]  # (P, 2)

        # Top point (Y 最小)
        t_idx = torch.argmin(poly_i[:, 1])
        t_val = poly_i[t_idx, 1]
        # 向前后扩展找阈值内的点
        t_idxs = [t_idx.item()]
        tmp = (t_idx + 1) % P
        while tmp != t_idx and poly_i[tmp, 1] - t_val <= thresh * h[i]:
            t_idxs.append(tmp.item())
            tmp = (tmp + 1) % P
        tmp = (t_idx - 1) % P
        while tmp != t_idx and poly_i[tmp, 1] - t_val <= thresh * h[i]:
            t_idxs.append(tmp.item())
            tmp = (tmp - 1) % P
        # 取 X 中间值
        t_idxs_tensor = torch.tensor(t_idxs, device=device)
        tt_x = (poly_i[t_idxs_tensor, 0].max() + poly_i[t_idxs_tensor, 0].min()) / 2
        tt = torch.stack([tt_x, t[i]])

        # Bottom point (Y 最大)
        b_idx = torch.argmax(poly_i[:, 1])
        b_val = poly_i[b_idx, 1]
        b_idxs = [b_idx.item()]
        tmp = (b_idx + 1) % P
        while tmp != b_idx and b_val - poly_i[tmp, 1] <= thresh * h[i]:
            b_idxs.append(tmp.item())
            tmp = (tmp + 1) % P
        tmp = (b_idx - 1) % P
        while tmp != b_idx and b_val - poly_i[tmp, 1] <= thresh * h[i]:
            b_idxs.append(tmp.item())
            tmp = (tmp - 1) % P
        b_idxs_tensor = torch.tensor(b_idxs, device=device)
        bb_x = (poly_i[b_idxs_tensor, 0].max() + poly_i[b_idxs_tensor, 0].min()) / 2
        bb = torch.stack([bb_x, b[i]])

        # Left point (X 最小)
        l_idx = torch.argmin(poly_i[:, 0])
        l_val = poly_i[l_idx, 0]
        l_idxs = [l_idx.item()]
        tmp = (l_idx + 1) % P
        while tmp != l_idx and poly_i[tmp, 0] - l_val <= thresh * w[i]:
            l_idxs.append(tmp.item())
            tmp = (tmp + 1) % P
        tmp = (l_idx - 1) % P
        while tmp != l_idx and poly_i[tmp, 0] - l_val <= thresh * w[i]:
            l_idxs.append(tmp.item())
            tmp = (tmp - 1) % P
        l_idxs_tensor = torch.tensor(l_idxs, device=device)
        ll_y = (poly_i[l_idxs_tensor, 1].max() + poly_i[l_idxs_tensor, 1].min()) / 2
        ll = torch.stack([l[i], ll_y])

        # Right point (X 最大)
        r_idx = torch.argmax(poly_i[:, 0])
        r_val = poly_i[r_idx, 0]
        r_idxs = [r_idx.item()]
        tmp = (r_idx + 1) % P
        while tmp != r_idx and r_val - poly_i[tmp, 0] <= thresh * w[i]:
            r_idxs.append(tmp.item())
            tmp = (tmp + 1) % P
        tmp = (r_idx - 1) % P
        while tmp != r_idx and r_val - poly_i[tmp, 0] <= thresh * w[i]:
            r_idxs.append(tmp.item())
            tmp = (tmp - 1) % P
        r_idxs_tensor = torch.tensor(r_idxs, device=device)
        rr_y = (poly_i[r_idxs_tensor, 1].max() + poly_i[r_idxs_tensor, 1].min()) / 2
        rr = torch.stack([r[i], rr_y])

        results.append(torch.stack([tt, ll, bb, rr]))

    return torch.stack(results)  # (N, 4, 2)


def run_inference(model, device, batch, save_dir, ver_tag, index):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor): batch[k] = v.to(device)

    dr = float(snake_config.down_ratio)
    core = model.net if hasattr(model, 'net') else model

    with torch.no_grad():
        # 1. 网络前向
        yolo_out = core.yolo(batch['inp'])
        if isinstance(yolo_out, (list, tuple)):
            feat_p2 = yolo_out[1][0] if len(yolo_out) > 1 else yolo_out[0]
        else:
            feat_p2 = yolo_out
        cnn_feature = core.cnn_proj(feat_p2)

        # 2. 初始化轮廓 - 方案A: 使用训练数据集中预准备的八边形（与训练完全一致）
        # 注意：batch['i_it_py'] 已经是训练时 prepare_evolution() 生成的八边形初始化
        # 它经过了：阈值扩展极值点 -> get_octagon -> uniformsample -> GT起点对齐
        gt_all = batch['i_gt_py']
        if gt_all.numel() == 0:
            print("  [!] No GT polygons, skipping...")
            return

        B, M, P, _ = gt_all.shape

        # 方案A: 直接使用训练数据集中的八边形初始化（最推荐，与训练完全一致）
        if 'i_it_py' in batch and batch['i_it_py'].numel() > 0:
            print("  [Init] Using training dataset octagon initialization (fully consistent with training)")
            i_it_py = batch['i_it_py'].view(-1, P, 2)  # (B*M, P, 2)
        else:
            # 方案B: 如果没有预准备的初始化，使用与训练一致的极值点提取方式
            print("  [Init] Reconstructing octagon from GT with consistent extreme point extraction")
            poly_flat = gt_all.view(B * M, P, 2)

            # 使用与训练一致的极值点提取（阈值扩展 + 取中间值）
            ex = get_extreme_points_torch(poly_flat)

            # 生成八边形
            init_polys = snake_decode.get_octagon(ex).view(B, M, 12, 2)
            i_it_py = snake_gcn_utils.uniform_upsample(init_polys, 128)[0]

        c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)

        # 构建 py_ind（多图批处理支持）
        if B == 1:
            py_ind = torch.zeros(i_it_py.size(0), dtype=torch.long, device=device)
        else:
            py_ind = torch.cat([torch.full((M,), i, dtype=torch.long, device=device) for i in range(B)])

        # 3. 核心：调用原生的 sample_disp
        disp = core.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind, steps=50)

        # 检查位移统计
        print(f"  [Stats] Disp Min: {disp.min().item():.3f}, Max: {disp.max().item():.3f}, Mean: {disp.abs().mean().item():.3f}")

        # 4. 生成预测多边形
        pred_polys = (i_it_py + disp).cpu().numpy() * dr
        init_np = i_it_py.cpu().numpy() * dr
        gt_np = gt_all.cpu().numpy() * dr

    # 5. 渲染
    if 'orig_img' in batch:
        img_raw = batch['orig_img'][0]
        img = img_raw.detach().cpu().numpy() if torch.is_tensor(img_raw) else img_raw
        img = img.astype(np.uint8)
    else:
        print("  [!] Warning: 'orig_img' missing, using black background.")
        img = np.zeros((512, 512, 3), dtype=np.uint8)

    # 画图：OpenCV 顺序是 BGR
    for poly in gt_np[0]:
        cv2.polylines(img, [poly.astype(np.int32)], True, (0, 255, 0), 2)  # GT: 绿
    for poly in init_np:
        cv2.polylines(img, [poly.astype(np.int32)], True, (0, 255, 255), 1)  # Init: 黄
    for poly in pred_polys:
        cv2.polylines(img, [poly.astype(np.int32)], True, (0, 0, 255), 2)  # Pred: 红

    save_path = os.path.join(save_dir, f"CLEAN_{ver_tag}_idx{index}_{datetime.datetime.now().strftime('%H%M%S')}.png")
    cv2.imwrite(save_path, img)
    print(f"[*] Saved: {save_path}")

def main():
    tag = os.environ.get('TAG', 'v3')
    index = int(os.environ.get('INDEX', 0))

    model, device = load_v3_model(None)
    dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, False), False)
    collator = make_collator(cfg)

    save_dir = os.path.join(_THIS_DIR, 'visual/v3_clean_eval')
    os.makedirs(save_dir, exist_ok=True)

    print(f"[*] Starting CLEAN inference (Tag: {tag}, Index: {index})...")
    batch = collator([dataset[index]])
    run_inference(model, device, batch, save_dir, tag, index)

if __name__ == '__main__':
    main()
