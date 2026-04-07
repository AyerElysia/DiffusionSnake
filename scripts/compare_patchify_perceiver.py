"""Compare Patchify vs Perceiver output ranges"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'lib')

from networks.diffusion.dit_blocks import PerceiverCompressor
from networks.diffusion.dit_blocks_v3_1 import PatchifyEmbedding

# Create dummy input: typical P2 feature map [B, 64, 128, 128]
B = 1
C = 64
H = 128
W = 128
torch.manual_seed(42)
cnn_feature = torch.randn(B, C, H, W)  # Normal distribution, std=1

print('=== Input feature stats ===')
print(f'Shape: {cnn_feature.shape}')
print(f'Mean: {cnn_feature.mean():.4f}, Std: {cnn_feature.std():.4f}')

# Perceiver (V3 style)
perceiver = PerceiverCompressor(in_dim=64, out_dim=256, num_queries=256)
perceiver_out = perceiver(cnn_feature)
print('\n=== Perceiver output (V3) ===')
print(f'Shape: {perceiver_out.shape}')
print(f'Mean: {perceiver_out.mean():.4f}, Std: {perceiver_out.std():.4f}')
print(f'Min: {perceiver_out.min():.4f}, Max: {perceiver_out.max():.4f}')

# Patchify (V3.1 style)
patchify = PatchifyEmbedding(in_channels=64, patch_size=8, out_dim=256)
patchify_out = patchify(cnn_feature)
print('\n=== Patchify output (V3.1) ===')
print(f'Shape: {patchify_out.shape}')
print(f'Mean: {patchify_out.mean():.4f}, Std: {patchify_out.std():.4f}')
print(f'Min: {patchify_out.min():.4f}, Max: {patchify_out.max():.4f}')

# Compare the output magnitude difference
print('\n=== Output magnitude comparison ===')
print(f'Perceiver mean magnitude: {perceiver_out.abs().mean():.4f}')
print(f'Patchify mean magnitude: {patchify_out.abs().mean():.4f}')
print(f'Ratio (Patchify/Perceiver): {patchify_out.abs().mean() / perceiver_out.abs().mean():.2f}x')