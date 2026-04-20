#!/usr/bin/env python3
"""
V3.10推理测试：使用V3.6的checkpoint测试自适应点数推理
"""
import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

os.environ['CFG_FILE'] = 'configs/btcv_diffusion_dit_v3_10.yaml'
sys.path.insert(0, '/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30')

from lib.config import cfg
from lib.datasets import make_data_loader
from lib.networks import make_network
from lib.utils.snake import snake_config

def test_inference():
    print("=" * 60)
    print("V3.10 推理测试")
    print("=" * 60)

    # 检查自适应点数配置
    print(f"\n自适应点数配置:")
    print(f"  enabled: {snake_config.adaptive_points_enabled}")
    print(f"  min_points: {snake_config.min_points}")
    print(f"  max_points: {snake_config.max_points}")
    print(f"  target_density: {snake_config.target_density}")

    # 创建数据加载器
    print(f"\n创建测试数据加载器...")
    test_loader = make_data_loader(cfg, is_train=False)

    # 创建网络
    print("创建网络...")
    network = make_network(cfg)
    network = network.cuda()
    network.eval()

    # 加载checkpoint（使用V3.6的权重）
    ckpt_path = '/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30/data/outputs/btcv_diffusion_dit_v3_6_single_overfit/checkpoints/epoch_9500.pt'
    print(f"\n加载checkpoint: {ckpt_path}")

    try:
        checkpoint = torch.load(ckpt_path, map_location='cuda:0')
        if 'state_dict' in checkpoint:
            network.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f"  ✓ Checkpoint加载成功（epoch {checkpoint.get('epoch', '?')}）")
        elif 'net' in checkpoint:
            network.load_state_dict(checkpoint['net'], strict=False)
            print("  ✓ Checkpoint加载成功")
        else:
            print(f"  ✗ Checkpoint格式不正确，keys: {list(checkpoint.keys())}")
            return
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return

    # 推理一个样本
    print(f"\n推理测试样本...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 1:  # 只测试1个样本
                break

            # 移到GPU
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].cuda()
                elif isinstance(batch[k], dict):
                    for kk in batch[k]:
                        if isinstance(batch[k][kk], torch.Tensor):
                            batch[k][kk] = batch[k][kk].cuda()

            print(f"\n样本 {i+1}:")
            print(f"  inp shape: {batch['inp'].shape}")

            if 'point_mask' in batch:
                mask = batch['point_mask']
                print(f"  point_mask shape: {mask.shape}")
                valid_counts = mask.sum(dim=-1)
                if valid_counts.sum() > 0:
                    print(f"  点数范围: [{valid_counts[valid_counts>0].min():.0f}, {valid_counts[valid_counts>0].max():.0f}]")

            try:
                # 前向传播
                output = network(batch['inp'], batch)
                print(f"  ✓ 推理成功")

                # 检查输出
                if 'py_pred' in output:
                    py_pred = output['py_pred']
                    if isinstance(py_pred, list) and len(py_pred) > 0:
                        pred_contours = py_pred[-1]  # 最后一次迭代的结果
                        print(f"  预测轮廓数: {pred_contours.shape[0]}")
                        print(f"  轮廓点数: {pred_contours.shape[1]}")

                        # 可视化第一个轮廓
                        if pred_contours.shape[0] > 0:
                            contour = pred_contours[0].cpu().numpy()
                            print(f"  第一个轮廓范围: x=[{contour[:, 0].min():.1f}, {contour[:, 0].max():.1f}], y=[{contour[:, 1].min():.1f}, {contour[:, 1].max():.1f}]")

            except Exception as e:
                print(f"  ✗ 推理失败: {e}")
                import traceback
                traceback.print_exc()
                return

    print("\n" + "=" * 60)
    print("✓ V3.10 推理测试完成！")
    print("=" * 60)

if __name__ == '__main__':
    test_inference()
