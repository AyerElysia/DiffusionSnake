#!/usr/bin/env python
"""
调试推理输出
"""
import os
import sys
import torch
import numpy as np

os.chdir('/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30')
sys.path.insert(0, '/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30')

os.environ['CFG_FILE'] = 'configs/btcv_diffusion_dit_v3_4_single_overfit.yaml'

# 清除缓存
for mod in list(sys.modules.keys()):
    if mod.startswith('lib.'):
        del sys.modules[mod]

from lib.config import cfg
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.utils.snake import snake_config

print("加载模型...")
network = make_network(cfg)
trainer = make_trainer(cfg, network)

checkpoint = torch.load('data/outputs/btcv_diffusion_dit_v3_4_single_overfit/checkpoints/epoch_10000.pt', map_location='cpu')
state_dict = checkpoint.get('net') or checkpoint.get('state_dict') or checkpoint
model = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
model.load_state_dict(state_dict, strict=False)
model = model.cuda().eval()

print("加载数据...")
dataset = make_dataset(cfg, cfg.test.dataset, make_transforms(cfg, is_train=False), is_train=False)
data = dataset[0]

print("\n数据keys:", data.keys())
print("inp shape:", data['inp'].shape if 'inp' in data else "N/A")

batch = {k: torch.tensor(v).unsqueeze(0).cuda() if isinstance(v, np.ndarray) else v
         for k, v in data.items()}

print("\n推理...")
with torch.no_grad():
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].cuda()

    output = model(batch)

print("\nOutput type:", type(output))
if isinstance(output, tuple):
    print(f"Output is tuple, len={len(output)}")
    for i, item in enumerate(output):
        print(f"  output[{i}]: type={type(item)}")
        if isinstance(item, dict):
            print(f"    keys: {item.keys()}")
        elif isinstance(item, torch.Tensor):
            print(f"    shape: {item.shape}")
    output = output[0] if len(output) > 0 and isinstance(output[0], dict) else {}

print("\nOutput keys:", output.keys() if isinstance(output, dict) else "N/A")
for k in output.keys():
    if isinstance(output[k], torch.Tensor):
        print(f"  {k}: shape={output[k].shape}, device={output[k].device}")
    elif isinstance(output[k], (list, tuple)):
        print(f"  {k}: type={type(output[k])}, len={len(output[k])}")
    else:
        print(f"  {k}: type={type(output[k])}")

if 'py' in output:
    print(f"\npy shape: {output['py'].shape}")
    print(f"py[0] shape: {output['py'][0].shape}")
    print(f"py[0] 前5个点:\n{output['py'][0][:5]}")
