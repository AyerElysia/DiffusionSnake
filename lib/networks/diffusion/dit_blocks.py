"""
DiT Blocks for Diffusion Snake

包含：
1. SinusoidalTimeEmbedding - 正弦时间嵌入
2. TimeFiLM - 时间 FiLM 调制
3. PerceiverCompressor - 特征压缩模块
4. DiTBlock - DiT 基础模块

作者: DiffSnake Team
日期: 2026-03-11
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Apply scale and shift modulation to the input tensor.
    Formula: x * (1 + scale) + shift
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SnakePosEncoding(nn.Module):
    """
    1D Sine/Cosine Positional Encoding for Snake points.
    Encodes the cyclic order of 128 points.
    """
    def __init__(self, dim: int = 256, num_points: int = 128):
        super().__init__()
        self.dim = dim
        self.num_points = num_points
        
        # Create positional encoding: [1, P, dim]
        pe = torch.zeros(num_points, dim)
        position = torch.arange(0, num_points, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        # Using cyclic features if P is fixed
        # For Snake, we want 0 and P-1 to be close
        pe[:, 0::2] = torch.sin(position * div_term * (2.0 * math.pi / num_points))
        pe[:, 1::2] = torch.cos(position * div_term * (2.0 * math.pi / num_points))
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (N, P, dim)
        Returns:
            x + pe
        """
        return x + self.pe
class SinusoidalTimeEmbedding(nn.Module):
    """
    正弦时间嵌入

    完全复用现有 snake_denoiser.py 的实现
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间步，形状 (N,)

        Returns:
            时间嵌入，形状 (N, dim)
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)  # [N, dim//2]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [N, dim]
        if self.dim % 2 == 1:  # 如果 dim 是奇数，用 0 填充
            emb = F.pad(emb, (0, 1), "constant", 0)
        return emb


class TimeFiLM(nn.Module):
    """
    时间 FiLM 调制层

    完全复用现有 snake_denoiser.py 的实现
    """
    def __init__(self, time_dim: int, channels: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2)
        )
        # 初始化为恒等变换
        self.time_mlp[-1].weight.data.zero_()
        self.time_mlp[-1].bias.data.zero_()

    def forward(self, t_emb: torch.Tensor):
        """
        Args:
            t_emb: 时间嵌入，形状 (N, time_dim)

        Returns:
            gamma: 缩放因子，形状 (N, channels)
            beta: 平移因子，形状 (N, channels)
        """
        emb = self.time_mlp(t_emb)  # [N, 2*channels]
        gamma, beta = torch.chunk(emb, 2, dim=1)  # [N, channels], [N, channels]
        return gamma, beta


class PerceiverCompressor(nn.Module):
    """
    特征压缩模块

    基于 Perceiver IO (ICLR 2022)
    将 16384 个图像 token 压缩到 256 个 token

    计算量对比:
    - 无压缩: 128 × 16384 = 2,097,152
    - 有压缩: 256 × 16384 (只算一次) + 128 × 256 = 32,768
    - 减少: 约 64 倍
    """
    def __init__(self, in_dim: int = 64, out_dim: int = 256, num_queries: int = 256):
        super().__init__()

        # 输入投影
        self.input_proj = nn.Linear(in_dim, out_dim)

        # 可学习的 queries
        self.queries = nn.Embedding(num_queries, out_dim)

        # Cross-Attention 压缩
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=8,
            batch_first=True
        )

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, image_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_feat: P2 特征，形状 (N, 64, H, W)

        Returns:
            compressed: 压缩后的特征，形状 (N, 256, out_dim)
        """
        N, C, H, W = image_feat.shape

        # 展平并投影
        img = image_feat.flatten(2).transpose(1, 2)  # (N, H*W, 64)
        img = self.input_proj(img)  # (N, H*W, out_dim)

        # 可学习 queries 压缩
        queries = self.queries.weight.unsqueeze(0).expand(N, -1, -1)  # (N, 256, out_dim)
        compressed, _ = self.cross_attn(
            query=queries,
            key=img,
            value=img
        )

        # 残差 + 归一化
        compressed = self.norm(compressed + queries)

        return compressed  # (N, 256, out_dim)


class DiTBlock(nn.Module):
    """
    Advanced DiT Block with adaLN-Zero conditioning.
    
    Structure:
    1. adaLN-Zero Modulation (Time step guide)
    2. Self-Attention (Point interaction)
    3. Cross-Attention (Point-Image interaction)
    4. FFN (Feature refinement)
    """
    def __init__(self, dim: int = 256, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()

        # Self-Attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-Attention
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # FFN
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

        # adaLN-Zero Modulation: predicts 6 parameters (scale1, shift1, gate1, scale2, shift2, gate2)
        # alpha/beta/gamma for attention and FFN
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        
        # Initialize modulation weights to zero (Zero-init DiT stable training)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, image_context: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Points features, shape (N, P, dim)
            image_context: Compressed image features, shape (N, L, dim)
            t_emb: Time embeddings, shape (N, dim)
        """
        # 1. Predict modulation parameters
        mod = self.adaLN_modulation(t_emb)  # (N, 6 * dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)

        # 2. Self-Attention with adaLN
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.self_attn(x_norm, x_norm, x_norm)[0]

        # 3. Cross-Attention
        # Note: image_context doesn't usually need modulation because it's a fixed condition
        x_norm = self.norm2(x)
        x = x + self.cross_attn(x_norm, image_context, image_context)[0]

        # 4. FFN with adaLN
        x_norm = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x
