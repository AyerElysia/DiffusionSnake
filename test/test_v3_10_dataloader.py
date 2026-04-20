#!/usr/bin/env python3
"""
V3.10测试：验证基于bbox的自适应点数数据加载
"""
import os
import sys
import torch
import numpy as np

# 设置环境
os.environ['CFG_FILE'] = 'configs/btcv_diffusion_dit_v3_10.yaml'
sys.path.insert(0, '/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30')

from lib.config import cfg
from lib.datasets import make_data_loader
from lib.utils.snake import snake_config

def test_adaptive_dataloader():
    """测试自适应点数数据加载"""
    print("=" * 60)
    print("V3.10 自适应点数数据加载测试")
    print("=" * 60)

    # 检查配置
    print(f"\n配置检查:")
    print(f"  adaptive_points_enabled: {snake_config.adaptive_points_enabled}")
    print(f"  min_points: {snake_config.min_points}")
    print(f"  max_points: {snake_config.max_points}")
    print(f"  target_density: {snake_config.target_density}")
    print(f"  point_strategy: {snake_config.point_strategy}")

    # 创建数据加载器
    print(f"\n创建数据加载器...")
    train_loader = make_data_loader(cfg, is_train=True)

    # 加载一个batch
    print(f"\n加载第一个batch...")
    batch = next(iter(train_loader))

    # 检查batch内容
    print(f"\nBatch内容:")
    print(f"  inp shape: {batch['inp'].shape}")
    print(f"  i_it_py shape: {batch['i_it_py'].shape}")
    print(f"  i_gt_py shape: {batch['i_gt_py'].shape}")

    if 'point_mask' in batch:
        print(f"  point_mask shape: {batch['point_mask'].shape}")
        print(f"  ✓ point_mask存在")

        # 分析点数分布
        point_mask = batch['point_mask']
        B, ct_num, P = point_mask.shape

        print(f"\n点数分布分析:")
        print(f"  Batch size: {B}")
        print(f"  Max contours per image: {ct_num}")
        print(f"  Max points: {P}")

        # 统计每个轮廓的实际点数
        point_counts = []
        for b in range(B):
            for c in range(ct_num):
                mask = point_mask[b, c]
                if mask.sum() > 0:
                    n_pts = int(mask.sum().item())
                    point_counts.append(n_pts)

        if point_counts:
            point_counts = np.array(point_counts)
            print(f"\n  总轮廓数: {len(point_counts)}")
            print(f"  点数范围: [{point_counts.min()}, {point_counts.max()}]")
            print(f"  平均点数: {point_counts.mean():.1f}")
            print(f"  点数分布:")
            unique, counts = np.unique(point_counts, return_counts=True)
            for pts, cnt in zip(unique, counts):
                print(f"    {pts}点: {cnt}个轮廓")

        # 验证mask正确性
        print(f"\n验证mask正确性:")
        for b in range(min(2, B)):  # 只检查前2个样本
            for c in range(min(3, ct_num)):  # 只检查前3个轮廓
                mask = point_mask[b, c]
                if mask.sum() > 0:
                    n_pts = int(mask.sum().item())
                    poly = batch['i_it_py'][b, c]

                    # 检查有效点是否非零
                    valid_poly = poly[:n_pts]
                    invalid_poly = poly[n_pts:]

                    valid_nonzero = (valid_poly.abs().sum() > 0).item()
                    invalid_zero = (invalid_poly.abs().sum() == 0).item()

                    status = "✓" if (valid_nonzero and invalid_zero) else "✗"
                    print(f"  样本{b}轮廓{c}: {n_pts}点 {status}")
    else:
        print(f"  ✗ point_mask不存在（使用固定点数）")

    print(f"\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == '__main__':
    test_adaptive_dataloader()
