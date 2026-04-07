import torch
import sys
import os
import numpy as np

os.chdir('/home/medteam/Zhrch/DiffusionSnake-12-30')
sys.path.insert(0, '.')

# 测试 V3.0
print("=" * 60)
print("Testing V3.0")
print("=" * 60)

os.environ['CFG_FILE'] = 'configs/btcv_diffusion_dit_v3.yaml'
from lib.config import cfg
cfg.merge_from_file('configs/btcv_diffusion_dit_v3.yaml')

from lib.networks import make_network
from lib.train.trainers import make_trainer

network_v3 = make_network(cfg)
trainer_v3 = make_trainer(cfg, network_v3)

ckpt_v3 = torch.load('data/outputs/btcv_diffusion_dit_v3/checkpoints/latest.pt', map_location='cpu')
sd_v3 = ckpt_v3['state_dict']

wrapper_v3 = trainer_v3.network.module if hasattr(trainer_v3.network, 'module') else trainer_v3.network
info_v3 = wrapper_v3.load_state_dict(sd_v3, strict=False)
print(f"V3.0 Loaded: {len(sd_v3) - len(info_v3.missing_keys)} layers matched")
print(f"Missing: {len(info_v3.missing_keys)}, Unexpected: {len(info_v3.unexpected_keys)}")

# 测试前向传播
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wrapper_v3 = wrapper_v3.to(device).eval()

# 创建测试输入
B, C, H, W = 1, 64, 128, 128
cnn_feature = torch.randn(B, C, H, W, device=device)
N, P = 4, 128
x_t = torch.randn(N, P, 2, device=device)
t = torch.randint(0, 1000, (N,), device=device)
sampled_feat = torch.randn(N, C, P, device=device)
py_ind = torch.zeros(N, dtype=torch.long, device=device)

with torch.no_grad():
    core = wrapper_v3.net
    gcn_output, _ = core.gcn.denoiser(cnn_feature, sampled_feat, x_t, t, None, None, py_ind)
    print(f"V3.0 Output shape: {gcn_output.shape}")
    print(f"V3.0 Output stats: min={gcn_output.min().item():.4f}, max={gcn_output.max().item():.4f}, mean={gcn_output.mean().item():.4f}")

# 重新导入以测试 V3.1
print("\n" + "=" * 60)
print("Testing V3.1")
print("=" * 60)

# 清除之前的配置
import importlib
import lib.config.config as config_module
importlib.reload(config_module)
from lib.config import cfg as cfg_new
cfg_new.merge_from_file('configs/btcv_diffusion_dit_v3_1.yaml')

network_v31 = make_network(cfg_new)
trainer_v31 = make_trainer(cfg_new, network_v31)

ckpt_v31 = torch.load('data/outputs/btcv_diffusion_dit_v3_1/checkpoints/latest.pt', map_location='cpu')
sd_v31 = ckpt_v31['state_dict']

wrapper_v31 = trainer_v31.network.module if hasattr(trainer_v31.network, 'module') else trainer_v31.network
info_v31 = wrapper_v31.load_state_dict(sd_v31, strict=False)
print(f"V3.1 Loaded: {len(sd_v31) - len(info_v31.missing_keys)} layers matched")
print(f"Missing: {len(info_v31.missing_keys)}, Unexpected: {len(info_v31.unexpected_keys)}")

wrapper_v31 = wrapper_v31.to(device).eval()

with torch.no_grad():
    core_v31 = wrapper_v31.net
    gcn_output_v31, _ = core_v31.gcn.denoiser(cnn_feature, sampled_feat, x_t, t, None, None, py_ind)
    print(f"V3.1 Output shape: {gcn_output_v31.shape}")
    print(f"V3.1 Output stats: min={gcn_output_v31.min().item():.4f}, max={gcn_output_v31.max().item():.4f}, mean={gcn_output_v31.mean().item():.4f}")