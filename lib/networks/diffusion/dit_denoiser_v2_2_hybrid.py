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

class DiTDenoiserV2_2Hybrid(nn.Module):
    """
    V2.2_Hybrid: MM-DiT Diffusion with Odd-Even (Global-Local) Injection.
    - Global: 256 Image Patches (from Patchify)
    - Local: 128 Local Points (from GCN Sampling)
    - Architecture: Joint Attention per Layer
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
        
        # 1. Embeddings
        self.patch_embed = PatchifyEmbedding(
            patch_size=patch_size, 
            in_channels=in_channels, 
            embed_dim=state_dim
        )
        self.point_embed = SeparatePointEmbedding(
            state_dim=state_dim, 
            feature_dim=feature_dim
        )
        
        # 2. Local feature projection
        self.local_y_proj = nn.Sequential(
            nn.Linear(feature_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim)
        )
        
        # 3. Hybrid Backbone
        self.blocks = nn.ModuleList([
            JointDiTBlock(state_dim, num_heads) for _ in range(num_layers)
        ])
        
        # 4. Conditioning & Output
        self.t_embedder = TimestepEmbedder(state_dim)
        self.final_layer = FinalLayer(state_dim, num_points)

    def forward(self, cnn_feature, sampled_feat, x_t, t_scaled, adj, polys=None, py_ind=None):
        # Time conditioning
        t_emb = self.t_embedder(t_scaled)
        
        # Initial point tokens [N, 128, 256]
        x = self.point_embed(x_t, sampled_feat)
        
        # Initial image tokens [N, 256, 256]
        y_global = self.patch_embed(cnn_feature)
        if py_ind is not None:
            y_global = y_global[py_ind]
            
        # Local point tokens [N, 128, 256]
        y_local = self.local_y_proj(sampled_feat.transpose(1, 2))
        
        # 2. Hybrid Injection (Alternate across layers)
        for i, block in enumerate(self.blocks):
            # Select context for this layer
            y_context = y_global if (i % 2 == 0) else y_local
            
            # Joint attention between Points and current Context
            x, _ = block(x, y_context, t_emb)
            
        # 3. Final prediction
        pred = self.final_layer(x, t_emb)
        L = torch.zeros(1, device=x.device)
        return pred, L
