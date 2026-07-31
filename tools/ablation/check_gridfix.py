"""独立进程验证 gcn_sample_mode / padding_mode 是否生效。
用法: python tools/ablation/check_gridfix.py <cfg_file>
"""
import sys

cfg_path = sys.argv[1]
sys.argv = ['check', '--cfg_file', cfg_path]

import torch
from lib.config import cfg
from lib.utils.snake import snake_gcn_utils

print('cfg:', cfg_path)
print('  gcn_sample_mode        =', cfg.get('gcn_sample_mode', '<unset>'))
print('  gcn_sample_padding_mode=', cfg.get('gcn_sample_padding_mode', '<unset>'))
print('  resolved               =', snake_gcn_utils._gcn_sample_cfg())

H = W = 8
# 特征图 = x 坐标本身，便于反解采样位置
feat = torch.arange(W, dtype=torch.float32).view(1, 1, 1, W).repeat(1, 1, H, 1)
# 取像素中心 index 0..7，以及越界点 -2 和 9
xs = torch.tensor([0., 1., 2., 3., 4., 5., 6., 7., -2., 9.])
poly = torch.stack([xs, torch.full_like(xs, 4.)], dim=-1).view(1, -1, 2)
out = snake_gcn_utils.get_gcn_feature(feat, poly, torch.zeros(1, dtype=torch.long), H, W)
print('  sampled =', [round(v, 3) for v in out[0, 0].tolist()])
