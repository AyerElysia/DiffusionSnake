import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit_blocks_v2 import (
    PatchifyEmbedding, 
    SeparatePointEmbedding, 
    JointDiTBlock, 
    FinalLayer, 
    TimestepEmbedder
)

class DiTFlowMatchingHybrid(nn.Module):
    """
    V2.3_Hybrid: MM-DiT Flow Matching with Odd-Even (Global-Local) Injection.
    - Global: 256 Image Patches (from Patchify)
    - Local: 128 Local Points (from GCN Sampling)
    - Goal: Combine spatial awareness of V2.1 with full-joint attention of V2.2.
    """
    def __init__(
        self,
        state_dim=256,
        feature_dim=64,
        num_layers=6,
        num_heads=8,
        num_points=128,
        patch_size=8,
        in_channels=128,
    ):
        super().__init__()
        self.num_layers = num_layers
        
        # 1. 嵌入层 (Image & Points)
        self.patch_embed = PatchifyEmbedding(
            patch_size=patch_size, 
            in_channels=in_channels, 
            embed_dim=state_dim
        )
        self.point_embed = SeparatePointEmbedding(
            state_dim=state_dim, 
            feature_dim=feature_dim
        )
        
        # 2. 局部特征投影 (将采样特征投影到 256 维 Token 空间)
        self.local_y_proj = nn.Sequential(
            nn.Linear(feature_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim)
        )
        
        # 3. 核心 Backbone
        self.blocks = nn.ModuleList([
            JointDiTBlock(state_dim, num_heads) for _ in range(num_layers)
        ])
        
        # 4. 时间步嵌入与最终输出
        self.t_embedder = TimestepEmbedder(state_dim)
        self.final_layer = FinalLayer(state_dim, num_points)

    def forward(self, cnn_feature, sampled_feat, x_t, t_scaled, adj, polys=None, py_ind=None):
        # 1. 基础嵌入
        t_emb = self.t_embedder(t_scaled)  # (N, state_dim)
        
        # x: 点流 (Target Stream) [N, 128, 256]
        x = self.point_embed(x_t, sampled_feat)
        
        # y_global: 全局图像流 (Global Context Stream) [B, 256, 256]
        y_global = self.patch_embed(cnn_feature)
        if py_ind is not None:
            y_global = y_global[py_ind]  # 对齐到点流 Batch [N, 256, 256]
            
        # y_local: 局部上下文流 (Local Context Stream) [N, 128, 256]
        y_local = self.local_y_proj(sampled_feat.transpose(1, 2))
        
        # 2. 奇偶混合动力 Backbone
        for i, block in enumerate(self.blocks):
            # 关键逻辑：分层选择交互对象
            # 偶数层：点流 与 全局图像 Patch 对话
            # 奇数层：点流 与 局部点特征纹理 对话
            y_context = y_global if (i % 2 == 0) else y_local
            
            # 使用 JointDiTBlock 让 x 和 y 深度交互
            x, _ = block(x, y_context, t_emb)
            
        # 3. 最终预测
        v_pred = self.final_layer(x, t_emb)
        L = torch.zeros(1, device=x.device)
        return v_pred, L
