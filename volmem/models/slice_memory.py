from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .contracts import SliceMemoryState, SliceSequenceMeta


class SliceMemoryEncoder(nn.Module):
    """Encode current-slice evidence into compact spatial memory tokens."""

    def __init__(
        self,
        feature_dim: int,
        memory_dim: int,
        mask_channels: int = 1,
        pool_size: int = 8,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.memory_dim = int(memory_dim)
        self.mask_channels = int(mask_channels)
        self.pool_size = int(pool_size)
        self.key_proj = nn.Conv2d(
            self.feature_dim + self.mask_channels,
            self.memory_dim,
            kernel_size=1,
            bias=False,
        )
        normalization_groups = min(8, self.memory_dim)
        while self.memory_dim % normalization_groups != 0:
            normalization_groups -= 1
        self.value_proj = nn.Sequential(
            nn.Conv2d(
                self.feature_dim + self.mask_channels,
                self.memory_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(normalization_groups, self.memory_dim),
            nn.GELU(),
        )

    def forward(
        self,
        slice_features: torch.Tensor,
        mask_evidence: torch.Tensor,
        meta: SliceSequenceMeta,
    ) -> SliceMemoryState:
        meta.validate()
        if slice_features.dim() != 4 or slice_features.size(0) != 1:
            raise ValueError("SliceMemoryEncoder expects one slice [1,C,H,W]")
        if mask_evidence.dim() != 4 or mask_evidence.size(0) != 1:
            raise ValueError("mask_evidence must be [1,C,H,W]")
        pooled_features = F.adaptive_avg_pool2d(
            slice_features,
            (self.pool_size, self.pool_size),
        )
        mask_evidence = F.interpolate(
            mask_evidence.to(
                device=slice_features.device,
                dtype=slice_features.dtype,
            ),
            size=(self.pool_size, self.pool_size),
            mode="bilinear",
            align_corners=False,
        )
        if mask_evidence.size(1) != self.mask_channels:
            raise ValueError(
                "mask_evidence channels {} != configured {}".format(
                    mask_evidence.size(1), self.mask_channels
                )
            )
        fused = torch.cat([pooled_features, mask_evidence], dim=1)
        key = self.key_proj(fused)
        value = self.value_proj(fused)
        return SliceMemoryState(
            volume_id=meta.volume_id,
            slice_index=meta.slice_index,
            slice_position=meta.slice_position,
            position_unit=meta.position_unit,
            key=key,
            value=value,
            valid_mask=None,
        )


class SliceMemoryAttention(nn.Module):
    """Cross-attend current MoonViT features to prior slice memory tokens."""

    def __init__(
        self,
        feature_dim: int,
        memory_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.memory_dim = int(memory_dim)
        self.query_proj = nn.Conv2d(
            self.feature_dim,
            self.memory_dim,
            kernel_size=1,
            bias=False,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=self.memory_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
        )
        self.output_proj = nn.Conv2d(
            self.memory_dim,
            self.feature_dim,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        self.query_norm = nn.LayerNorm(self.memory_dim)
        self.memory_norm = nn.LayerNorm(self.memory_dim)

    def forward(
        self,
        slice_features: torch.Tensor,
        memory_states: Sequence[SliceMemoryState],
    ) -> torch.Tensor:
        if len(memory_states) == 0:
            return slice_features
        if slice_features.dim() != 4 or slice_features.size(0) != 1:
            raise ValueError("SliceMemoryAttention expects one slice [1,C,H,W]")

        batch_size, _, height, width = slice_features.shape
        query_map = self.query_proj(slice_features)
        query = query_map.flatten(2).permute(2, 0, 1)
        query = self.query_norm(query)

        memory_keys = []
        memory_values = []
        for state in memory_states:
            if state.key.size(0) != batch_size or state.value.size(0) != batch_size:
                raise ValueError("memory batch size must match current slice")
            memory_keys.append(state.key.flatten(2).permute(2, 0, 1))
            memory_values.append(state.value.flatten(2).permute(2, 0, 1))
        key = self.memory_norm(torch.cat(memory_keys, dim=0))
        value = self.memory_norm(torch.cat(memory_values, dim=0))
        attended, _ = self.attention(
            query,
            key,
            value,
            need_weights=False,
        )
        attended = attended.permute(1, 2, 0).reshape(
            batch_size,
            self.memory_dim,
            height,
            width,
        )
        residual = self.output_proj(attended)
        return slice_features + residual
