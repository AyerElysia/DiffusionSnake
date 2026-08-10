import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .dit_blocks import SinusoidalTimeEmbedding

class TimeFiLM(nn.Module):
    """
    Time-dependent Feature-wise Linear Modulation (FiLM) layer.
    """
    def __init__(self, time_dim: int, channels: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2)
        )
        self.time_mlp[-1].weight.data.zero_()
        self.time_mlp[-1].bias.data.zero_()

    def forward(self, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t_emb: Time embedding of shape [B, time_dim]
            
        Returns:
            gamma: Scaling factor of shape [B, C]
            beta: Shifting factor of shape [B, C]
        """
        emb = self.time_mlp(t_emb)  # [B, 2*C]
        gamma, beta = torch.chunk(emb, 2, dim=1)  # [B, C], [B, C]
        return gamma, beta

class SnakeDenoiser(nn.Module):
    """
    Snake-based denoiser with FiLM conditioning on time steps.
    """
    def __init__(self, state_dim: int = 128, use_vm2: bool = True,
                 cx: int = 256, cf: int = 128, n_adj: int = 4, time_dim: int = 128,
                 feature_dim: int = 64, res_layers: int = 7, fusion_dim: int = 256):
        super().__init__()
        self.Cx = cx
        self.Cf = cf
        self.n_adj = n_adj
        self.time_dim = time_dim

        # Time embedding and FiLM
        self.time_emb = SinusoidalTimeEmbedding(dim=time_dim)
        self.concat_channels = feature_dim + 2 + 2  # gcn_feat + x_t(2) + can_ch(2)
        self.time_film = TimeFiLM(time_dim=time_dim, channels=self.concat_channels)

        # Snake backbone
        from lib.networks.snake.snake import Snake
        self.snake = Snake(
            state_dim=state_dim,
            feature_dim=self.concat_channels,
            conv_type='vm2' if use_vm2 else 'dgrid',
            res_layers=res_layers,
            fusion_dim=fusion_dim
        )

    def forward(
        self, 
        gcn_feat: torch.Tensor,   # [N, 64, P]
        can_coords: torch.Tensor, # [N, P, 2]
        x_t: torch.Tensor,        # [N, P, 2]
        t: torch.Tensor,          # [N]
        adj: torch.Tensor,        # [P, k]
        polys=None,               # Optional polygon parameters
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, P, _ = x_t.shape
        device = x_t.device

        # Time embedding
        t_emb = self.time_emb(t)  # [N, time_dim]

        # Prepare input sequence: [N, 68, P]
        x_t_ch = x_t.permute(0, 2, 1)  # [N, 2, P]
        can_ch = can_coords.permute(0, 2, 1)  # [N, 2, P]
        seq = torch.cat([x_t_ch, gcn_feat, can_ch], dim=1)  # [N, 68, P]

        # Apply FiLM conditioning
        gamma_t, beta_t = self.time_film(t_emb)  # [N, 68], [N, 68]
        gamma_t = gamma_t.unsqueeze(-1).expand(-1, -1, P)  # [N, 68, P]
        beta_t = beta_t.unsqueeze(-1).expand(-1, -1, P)    # [N, 68, P]
        seq_mod = seq * (1.0 + gamma_t) + beta_t  # [N, 68, P]

        # Process through Snake network
        eps_pred, L = self.snake(seq_mod, adj, polys)  # eps_pred: [N, 2, P]
        
        # Reshape output
        eps_pred = eps_pred.permute(0, 2, 1).contiguous()  # [N, P, 2]
        return eps_pred, L
