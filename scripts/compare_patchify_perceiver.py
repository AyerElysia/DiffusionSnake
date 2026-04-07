"""Compare Patchify vs Perceiver output ranges"""
import torch
import torch.nn as nn

# Direct implementation to avoid config import issues

# PerceiverCompressor (from dit_blocks.py)
class PerceiverCompressor(nn.Module):
    def __init__(self, in_dim: int = 64, out_dim: int = 256, num_queries: int = 256):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, out_dim)
        self.queries = nn.Embedding(num_queries, out_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=8,
            batch_first=True
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, image_feat: torch.Tensor) -> torch.Tensor:
        N, C, H, W = image_feat.shape
        img = image_feat.flatten(2).transpose(1, 2)  # (N, H*W, 64)
        img = self.input_proj(img)  # (N, H*W, out_dim)
        queries = self.queries.weight.unsqueeze(0).expand(N, -1, -1)  # (N, 256, out_dim)
        compressed, _ = self.cross_attn(query=queries, key=img, value=img)
        compressed = self.norm(compressed + queries)
        return compressed

# PatchifyEmbedding (from dit_blocks_v3_1.py)
class PatchifyEmbedding(nn.Module):
    def __init__(self, in_channels: int = 64, patch_size: int = 8, out_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.proj = nn.Conv2d(in_channels, out_dim, kernel_size=patch_size, stride=patch_size)
        max_grid = 16
        self.pos_embed = nn.Parameter(torch.zeros(1, max_grid * max_grid, out_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]
        return x

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