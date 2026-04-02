import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Re-use robust foundational components from V2
from .dit_blocks_v2 import (
    RMSNorm, 
    modulate, 
    SwiGLU, 
    CyclicRoPE1D,
    SeparatePointEmbedding
)

# ---------------------------------------------------------------------------
# V2.2: Patchify Embedding for Image Modality (Strict DiT style)
# ---------------------------------------------------------------------------
class PatchifyEmbedding(nn.Module):
    """
    Standard ViT-style Patchify for image features.
    No overlapping patches. Converts HxW into (H/p * W/p) tokens.
    Adds absolute 2D sine-cosine positional embedding.
    """
    def __init__(self, in_channels: int = 64, patch_size: int = 8, out_dim: int = 256, max_grid: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.out_dim = out_dim
        # Projection layer: equivalent to non-overlapping convolutions
        self.proj = nn.Conv2d(in_channels, out_dim, kernel_size=patch_size, stride=patch_size)
        
        # Absolute positional embedding (max sequence length = max_grid * max_grid = 256)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_grid * max_grid, out_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        Returns: [B, num_patches, out_dim]
        """
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, out_dim, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, (H*W)/(p*p), out_dim]
        
        # Truncate or pad pos_embed dynamically if necessary, but we assume 16x16=256
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]
        return x


# ---------------------------------------------------------------------------
# V2.2: Joint DiT Block (Stable Diffusion 3 / MM-DiT Paradigm)
# ---------------------------------------------------------------------------
class JointDiTBlock(nn.Module):
    """
    MM-DiT Dual Stream Block.
    Contour points (Modality 1) and Image Patches (Modality 2) maintain
    separate modulation, normalization, and projection weights, but attend
    to each other in a unified Joint Attention context.
    
    Sequence length memory = N_points(128) + N_patches(256) = 384
    """
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_points: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # ------------------ CONTOUR BRANCH ------------------
        self.norm1_c = RMSNorm(dim)
        self.q_proj_c = nn.Linear(dim, dim, bias=False)
        self.k_proj_c = nn.Linear(dim, dim, bias=False)
        self.v_proj_c = nn.Linear(dim, dim, bias=False)
        self.rope_c = CyclicRoPE1D(head_dim=self.head_dim, num_points=num_points)
        
        self.out_proj_c = nn.Linear(dim, dim, bias=False)
        self.norm2_c = RMSNorm(dim)
        self.mlp_c = SwiGLU(dim=dim, dropout=dropout)

        self.adaLN_mod_c = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_mod_c[-1].weight, 0)
        nn.init.constant_(self.adaLN_mod_c[-1].bias, 0)

        # ------------------- IMAGE BRANCH -------------------
        self.norm1_i = RMSNorm(dim)
        self.q_proj_i = nn.Linear(dim, dim, bias=False)
        self.k_proj_i = nn.Linear(dim, dim, bias=False)
        self.v_proj_i = nn.Linear(dim, dim, bias=False)
        
        self.out_proj_i = nn.Linear(dim, dim, bias=False)
        self.norm2_i = RMSNorm(dim)
        self.mlp_i = SwiGLU(dim=dim, dropout=dropout)

        self.adaLN_mod_i = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_mod_i[-1].weight, 0)
        nn.init.constant_(self.adaLN_mod_i[-1].bias, 0)
        
        # ------------------ SHARED QK-NORM ------------------
        # Applies normalized stability across the joint attention
        self.qk_norm_q = RMSNorm(self.head_dim)
        self.qk_norm_k = RMSNorm(self.head_dim)

    def forward(
        self,
        x_c: torch.Tensor, # [B, 128, dim]
        x_i: torch.Tensor, # [B, 256, dim]
        t_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, N_c, D = x_c.shape
        _, N_i, _ = x_i.shape
        H = self.num_heads
        hd = self.head_dim

        # 1. Modulation
        mod_c = self.adaLN_mod_c(t_emb).chunk(6, dim=1) # shift_a, scale_a, gate_a, shift_m, scale_m, gate_m
        mod_i = self.adaLN_mod_i(t_emb).chunk(6, dim=1)

        # 2. Pre-Norm & Projection
        norm_x_c = modulate(self.norm1_c(x_c), mod_c[0], mod_c[1])
        norm_x_i = modulate(self.norm1_i(x_i), mod_i[0], mod_i[1])

        q_c, k_c, v_c = self.q_proj_c(norm_x_c), self.k_proj_c(norm_x_c), self.v_proj_c(norm_x_c)
        q_i, k_i, v_i = self.q_proj_i(norm_x_i), self.k_proj_i(norm_x_i), self.v_proj_i(norm_x_i)

        # Reshape to [B, Seq, H, hd]
        q_c = q_c.view(B, N_c, H, hd)
        k_c = k_c.view(B, N_c, H, hd)
        v_c = v_c.view(B, N_c, H, hd)
        
        q_i = q_i.view(B, N_i, H, hd)
        k_i = k_i.view(B, N_i, H, hd)
        v_i = v_i.view(B, N_i, H, hd)

        # QK-Norm
        q_c, k_c = self.qk_norm_q(q_c), self.qk_norm_k(k_c)
        q_i, k_i = self.qk_norm_q(q_i), self.qk_norm_k(k_i)

        # Apply Cyclic RoPE ONLY to contour branch, image uses absolute PE directly
        q_c = self.rope_c.apply_rotary(q_c.transpose(1, 2)).transpose(1, 2)
        k_c = self.rope_c.apply_rotary(k_c.transpose(1, 2)).transpose(1, 2)

        # 3. Concatenate into Joint Sequence
        q = torch.cat([q_c, q_i], dim=1).transpose(1, 2) # [B, H, N_c+N_i, hd]
        k = torch.cat([k_c, k_i], dim=1).transpose(1, 2)
        v = torch.cat([v_c, v_i], dim=1).transpose(1, 2)

        # 4. Joint Attention Flash/Standard
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N_c + N_i, D)

        # 5. Split and Output Projections with Gates
        out_c = out[:, :N_c, :]
        out_i = out[:, N_c:, :]

        out_c = self.out_proj_c(out_c)
        out_i = self.out_proj_i(out_i)

        x_c = x_c + mod_c[2].unsqueeze(1) * out_c
        x_i = x_i + mod_i[2].unsqueeze(1) * out_i

        # 6. FFNs
        norm_mlp_c = modulate(self.norm2_c(x_c), mod_c[3], mod_c[4])
        norm_mlp_i = modulate(self.norm2_i(x_i), mod_i[3], mod_i[4])

        x_c = x_c + mod_c[5].unsqueeze(1) * self.mlp_c(norm_mlp_c)
        x_i = x_i + mod_i[5].unsqueeze(1) * self.mlp_i(norm_mlp_i)

        return x_c, x_i
