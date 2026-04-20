#!/usr/bin/env python3
"""
V3.10快速测试：训练几个iteration验证pipeline
"""
import os
import sys
import torch

os.environ['CFG_FILE'] = 'configs/btcv_diffusion_dit_v3_10.yaml'
sys.path.insert(0, '/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30')

from lib.config import cfg
from lib.datasets import make_data_loader
from lib.networks import make_network
from lib.train import make_trainer

def quick_test():
    print("=" * 60)
    print("V3.10 快速训练测试")
    print("=" * 60)

    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader = make_data_loader(cfg, is_train=True)

    # 创建网络
    print("创建网络...")
    network = make_network(cfg)
    network = network.cuda()

    # 创建trainer
    print("创建trainer...")
    trainer = make_trainer(cfg, network)

    # 训练3个batch
    print("\n开始训练3个batch...")
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break

        print(f"\nBatch {i+1}/3:")
        print(f"  inp shape: {batch['inp'].shape}")
        print(f"  i_it_py shape: {batch['i_it_py'].shape}")

        if 'point_mask' in batch:
            mask = batch['point_mask']
            print(f"  point_mask shape: {mask.shape}")
            # 统计点数
            valid_counts = mask.sum(dim=-1)
            print(f"  点数范围: [{valid_counts[valid_counts>0].min():.0f}, {valid_counts[valid_counts>0].max():.0f}]")

        try:
            # 将batch移到GPU
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].cuda()
                elif isinstance(batch[k], dict):
                    for kk in batch[k]:
                        if isinstance(batch[k][kk], torch.Tensor):
                            batch[k][kk] = batch[k][kk].cuda()

            # 前向传播
            output, loss, loss_stats, _ = trainer.network(batch)
            print(f"  ✓ 前向传播成功")
            print(f"  Loss: {loss.item():.4f}")

            # 反向传播（简化版，不需要optimizer）
            loss.backward()
            print(f"  ✓ 反向传播成功")

        except Exception as e:
            print(f"  ✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "=" * 60)
    print("✓ V3.10 pipeline测试通过！")
    print("=" * 60)
    return True

if __name__ == '__main__':
    success = quick_test()
    sys.exit(0 if success else 1)
