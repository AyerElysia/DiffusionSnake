#!/usr/bin/env python3
"""
计算基于八边形初始化的 displacement 统计数据
用于 DiT V3 训练
"""
import os
import sys

# 设置配置文件为 V3（使用八边形初始化）
os.environ['CFG_FILE'] = 'configs/btcv_diffusion_dit_v3.yaml'

# 导入必要的模块
import json
import math
import torch
from lib.config import cfg
from lib.datasets import make_data_loader
from lib.utils.snake import snake_gcn_utils


@torch.no_grad()
def _signed_area(poly: torch.Tensor) -> torch.Tensor:
    """计算多边形的有向面积"""
    x = poly[..., 0]
    y = poly[..., 1]
    x1 = torch.roll(x, shifts=-1, dims=1)
    y1 = torch.roll(y, shifts=-1, dims=1)
    return 0.5 * torch.sum(x * y1 - x1 * y, dim=1)


@torch.no_grad()
def compute_octagon_disp_stats() -> dict:
    """
    计算基于八边形初始化的 displacement 统计数据

    Returns:
        dict: 包含 dx_min, dx_max, dy_min, dy_max 的字典
    """
    print(f"[INFO] 使用配置: {os.environ.get('CFG_FILE', 'default')}")
    print(f"[INFO] use_dit_v3: {getattr(cfg, 'use_dit_v3', False)}")

    # 创建数据加载器
    loader = make_data_loader(cfg, is_train=True, is_distributed=False)
    print(f"[INFO] 数据集大小: {len(loader.dataset)}")

    dx_min = math.inf
    dx_max = -math.inf
    dy_min = math.inf
    dy_max = -math.inf

    total_contours = 0

    for batch_idx, batch in enumerate(loader):
        # 使用 snake_gcn_utils.prepare_training 获取初始化
        # 这会自动使用八边形初始化（因为 cfg.use_dit_v3=True）
        init = snake_gcn_utils.prepare_training({}, batch)

        i_gt_py = init['i_gt_py']
        i_init_train_py = init['i_it_py']

        if not isinstance(i_gt_py, torch.Tensor) or i_gt_py.numel() == 0:
            continue

        device = i_gt_py.device
        i_init_train_py = i_init_train_py.to(device)

        # 方向对齐 + 起点对齐（必须与训练代码一致）
        area_init = _signed_area(i_init_train_py)
        area_gt = _signed_area(i_gt_py)
        orient_mismatch = ((area_init >= 0) ^ (area_gt >= 0))
        if orient_mismatch.any():
            i_gt_py = i_gt_py.clone()
            i_gt_py[orient_mismatch] = torch.flip(i_gt_py[orient_mismatch], dims=[1])

        # 起点对齐
        d2 = (i_init_train_py[:, :1, :] - i_gt_py).pow(2).sum(-1)
        nearest = torch.argmin(d2, dim=1)
        if i_gt_py.size(0) > 0:
            rolled = []
            for i in range(i_gt_py.size(0)):
                s = int(nearest[i].item())
                if s != 0:
                    rolled.append(torch.roll(i_gt_py[i], shifts=-s, dims=0))
                else:
                    rolled.append(i_gt_py[i])
            i_gt_py = torch.stack(rolled, dim=0)

        # 计算 displacement
        disp = i_gt_py - i_init_train_py  # [N,P,2]

        disp_cpu = disp.detach().float().cpu()
        dx_min = min(dx_min, float(disp_cpu[..., 0].min().item()))
        dx_max = max(dx_max, float(disp_cpu[..., 0].max().item()))
        dy_min = min(dy_min, float(disp_cpu[..., 1].min().item()))
        dy_max = max(dy_max, float(disp_cpu[..., 1].max().item()))

        total_contours += i_gt_py.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"[INFO] 已处理 {batch_idx + 1}/{len(loader)} 批次, "
                  f"累计轮廓数: {total_contours}")

    if not (math.isfinite(dx_min) and math.isfinite(dx_max) and
            math.isfinite(dy_min) and math.isfinite(dy_max)):
        raise RuntimeError(
            'Failed to compute disp stats: got non-finite min/max. '
            'Check your dataset and cfg.'
        )

    print(f"\n[INFO] 统计完成！总轮廓数: {total_contours}")

    return {
        'dx_min': dx_min,
        'dx_max': dx_max,
        'dy_min': dy_min,
        'dy_max': dy_max,
    }


def main():
    # 输出文件路径
    output_path = 'data/stats/btcv_disp_stats_octagon.json'

    print("=" * 60)
    print("计算基于八边形初始化的 Displacement 统计数据")
    print("=" * 60)

    # 计算统计数据
    stats = compute_octagon_disp_stats()

    # 保存到文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print(f"[SUCCESS] 统计数据已保存到: {output_path}")
    print("=" * 60)
    print("\n统计结果:")
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")

    # 对比旧的统计数据
    old_stats_path = 'data/stats/btcv_disp_stats.json'
    if os.path.exists(old_stats_path):
        with open(old_stats_path, 'r') as f:
            old_stats = json.load(f)

        print("\n对比旧统计数据 (bbox 初始化):")
        print(f"  {'指标':<10} {'旧值 (bbox)':<15} {'新值 (octagon)':<15} {'差异':<10}")
        print("  " + "-" * 55)
        for key in ['dx_min', 'dx_max', 'dy_min', 'dy_max']:
            old_val = old_stats[key]
            new_val = stats[key]
            diff = new_val - old_val
            print(f"  {key:<10} {old_val:<15.6f} {new_val:<15.6f} {diff:+.6f}")

    print("\n[NEXT] 更新 V3 配置文件:")
    print(f"  diffusion_disp_stats: \"{output_path}\"")


if __name__ == '__main__':
    main()
