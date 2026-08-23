"""SAM 2.1 Hiera context for high-resolution pure-2D contour evolution.

The encoder is the segmentation-pretrained Hiera-Tiny image backbone exposed by
``timm``.  Four native feature levels (strides 4/8/16/32) are projected and
fused at stride 4 so the contour model receives both fine boundary detail and
large-receptive-field semantics.  The encoder is normally frozen during the
alignment phase; only the small FPN-style fusion is learned.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Pure2DSAM2HieraFPN(nn.Module):
    """High-resolution multi-scale context from a SAM 2.1 Hiera encoder."""

    _SUPPORTED_BACKBONES = {
        "sam2_hiera_tiny.fb_r896_2pt1": (96, 192, 384, 768),
    }

    def __init__(
        self,
        backbone_name: str = "sam2_hiera_tiny.fb_r896_2pt1",
        out_channels: int = 256,
        pyramid_channels: int = 64,
        pretrained_path: str = "",
        freeze_backbone: bool = True,
        source_mean: float = 0.5,
        source_std: float = 0.5,
    ):
        super().__init__()
        try:
            import timm
            from timm.models import load_checkpoint
        except ImportError as error:
            raise RuntimeError("Pure2DSAM2HieraFPN requires timm") from error

        backbone_name = str(backbone_name).strip()
        if backbone_name not in self._SUPPORTED_BACKBONES:
            raise ValueError(
                f"unsupported SAM2 Hiera backbone {backbone_name!r}; "
                f"expected one of {sorted(self._SUPPORTED_BACKBONES)}"
            )
        out_channels = int(out_channels)
        pyramid_channels = int(pyramid_channels)
        source_mean = float(source_mean)
        source_std = float(source_std)
        if out_channels <= 0 or pyramid_channels <= 0:
            raise ValueError("out_channels and pyramid_channels must be positive")
        if not torch.isfinite(torch.tensor(source_mean)) or source_std <= 0.0:
            raise ValueError("source_mean/source_std must be finite and source_std > 0")

        self.backbone_name = backbone_name
        self.freeze_backbone = bool(freeze_backbone)
        self.source_mean = source_mean
        self.source_std = source_std
        # Build the complete model so the official safetensors checkpoint can
        # be loaded with strict=True, including its final normalization layer.
        self.backbone = timm.create_model(backbone_name, pretrained=False)
        if str(pretrained_path or "").strip():
            incompatible = load_checkpoint(
                self.backbone,
                str(pretrained_path),
                strict=True,
                weights_only=True,
            )
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    "strict Hiera checkpoint load returned incompatible keys: "
                    f"missing={incompatible.missing_keys} "
                    f"unexpected={incompatible.unexpected_keys}"
                )
            print(
                f"[Pure2D-Hiera] strict pretrained load PASS "
                f"backbone={backbone_name} path={pretrained_path}",
                flush=True,
            )

        channels = self._SUPPORTED_BACKBONES[backbone_name]
        self.projections = nn.ModuleList(
            nn.Conv2d(channel, pyramid_channels, kernel_size=1, bias=False)
            for channel in channels
        )
        merged_channels = len(channels) * pyramid_channels
        groups = 16 if out_channels % 16 == 0 else 8
        self.fuse = nn.Sequential(
            nn.Conv2d(merged_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
        )
        self.global_proj = nn.Sequential(
            nn.Conv2d(channels[-1], out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )
        self.out_norm = nn.GroupNorm(groups, out_channels)

        # The sagittal data path supplies (x - 0.5) / 0.5.  Undo that
        # normalization first, then apply the official pretrained-image stats.
        self.register_buffer(
            "encoder_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "encoder_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )

        if self.freeze_backbone:
            self._freeze_encoder()

    @property
    def encoder_modules(self) -> Tuple[nn.Module, ...]:
        return (self.backbone,)

    def _freeze_encoder(self) -> None:
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _normalize_for_encoder(self, image: torch.Tensor) -> torch.Tensor:
        image_01 = image.mul(self.source_std).add(self.source_mean).clamp_(0.0, 1.0)
        mean = self.encoder_mean.to(device=image.device, dtype=image.dtype)
        std = self.encoder_std.to(device=image.device, dtype=image.dtype)
        return (image_01 - mean) / std

    def _encode(self, image: torch.Tensor):
        return self.backbone.forward_intermediates(
            image,
            indices=[0, 1, 2, 3],
            norm=False,
            stop_early=True,
            output_fmt="NCHW",
            intermediates_only=True,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4 or image.size(1) != 3:
            raise ValueError(f"Expected image [B,3,H,W], got {tuple(image.shape)}")
        normalized = self._normalize_for_encoder(image)
        if self.freeze_backbone:
            with torch.no_grad():
                features = self._encode(normalized)
        else:
            features = self._encode(normalized)
        if len(features) != 4:
            raise RuntimeError(f"expected four Hiera feature levels, got {len(features)}")

        target_hw = features[0].shape[-2:]
        pyramid = []
        for feature, projection in zip(features, self.projections):
            projected = projection(feature)
            if projected.shape[-2:] != target_hw:
                projected = F.interpolate(
                    projected,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            pyramid.append(projected)

        fused = self.fuse(torch.cat(pyramid, dim=1))
        global_context = self.global_proj(
            F.adaptive_avg_pool2d(features[-1], output_size=1)
        )
        return F.gelu(self.out_norm(fused + global_context))
