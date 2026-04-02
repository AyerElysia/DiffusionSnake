import torch
import torch.nn as nn

from .dit_blocks import SinusoidalTimeEmbedding
from .dit_blocks_v2 import SeparatePointEmbedding, FinalLayer
from .dit_blocks_v2_2 import PatchifyEmbedding, JointDiTBlock

class DiTFlowMatchingV2_3(nn.Module):
    """
    MM-DiT Flow Matching Network V2.3 (Patchify + Joint Attention).
    Architecture inspired by SD3, but adapted for Flow Matching (Predicting V_t).
    """

    def __init__(
        self,
        state_dim: int = 256,
        feature_dim: int = 64,
        num_layers: int = 6,
        num_heads: int = 8,
        time_dim: int = 256,
        num_points: int = 128,
        patch_size: int = 8,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        
        # 1. Time Embedding
        self.time_emb_net = nn.Sequential(
            SinusoidalTimeEmbedding(dim=state_dim // 4),
            nn.Linear(state_dim // 4, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )

        # 2. Modality 2 (Image) Embedding: Patchify 
        # For 128x128 feature map and p=8 -> 16x16=256 patches
        self.image_embed = PatchifyEmbedding(
            in_channels=feature_dim,
            patch_size=patch_size,
            out_dim=state_dim,
            max_grid=(128 // patch_size)
        )

        # 3. Modality 1 (Contour) Embedding
        self.point_embed = SeparatePointEmbedding(
            state_dim=state_dim,
            feature_dim=feature_dim
        )

        # 4. Joint DiT Blocks
        self.dit_layers = nn.ModuleList([
            JointDiTBlock(
                dim=state_dim,
                num_heads=num_heads,
                num_points=num_points,
                dropout=0.0
            ) for _ in range(num_layers)
        ])

        # 5. Final Layer (Only applies to Modality 1 - Contour)
        self.final_layer = FinalLayer(dim=state_dim, out_dim=2)

    def forward(
        self,
        cnn_feature: torch.Tensor,
        sampled_feat: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        adj: torch.Tensor = None,
        polys=None,
        py_ind: torch.Tensor = None,
    ):
        """
        Forward pass for Flow Matching MM-DiT.
        Matches V1/V2 signature for training compatibility.
        """
        # Batch Alignment
        if py_ind is not None and cnn_feature.shape[0] != x_t.shape[0]:
            cnn_feature = cnn_feature[py_ind]

        # 1. Timestep processing [B, dim]
        t_emb = self.time_emb_net(t)

        # 2. Extract Modality 1 & 2 sequences
        x_c = self.point_embed(x_t, sampled_feat)  # Contour: [B, 128, dim]
        x_i = self.image_embed(cnn_feature)        # Image: [B, 256, dim]

        # 3. Joint Diffusion Process
        for layer in self.dit_layers:
            x_c, x_i = layer(x_c, x_i, t_emb)

        # 4. Discard Image Modality, project Contour Modality to displacement
        out = self.final_layer(x_c, t_emb)
        L = torch.zeros(1, device=x_t.device, dtype=x_t.dtype)
        return out, L
