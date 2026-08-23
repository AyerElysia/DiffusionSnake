"""Learned 2-D CNN context for detector-free contour evolution.

This module deliberately contains no hand-crafted image operators.  A
ResNet-18/34 encoder learns local-to-global visual features directly from the
input image, and a small FPN-style decoder returns one stride-4 feature map for
the contour model.
"""

from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


def _build_resnet(name: str) -> nn.Module:
    name = str(name).strip().lower()
    if name not in ("resnet18", "resnet34"):
        raise ValueError(f"pure2d CNN backbone must be resnet18/resnet34, got {name!r}")
    constructor = getattr(tv_models, name)
    try:
        return constructor(weights=None)
    except TypeError:
        return constructor(pretrained=False)


def _unwrap_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dictionary, got {type(checkpoint).__name__}")
    for key in ("model", "state_dict", "net"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    return checkpoint


def _strip_prefixes(key: str) -> str:
    prefixes = (
        "module.",
        "model.",
        "backbone.",
        "encoder.",
        "net.pure2d_cnn_context.",
        "pure2d_cnn_context.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
                break
    return key


class Pure2DResNetFPN(nn.Module):
    """High-resolution local and low-resolution global learned context.

    C2 (stride 4) supplies fine spatial detail.  C3/C4/C5 supply progressively
    larger receptive fields.  All four maps are projected and fused at stride
    4, while pooled C5 adds an explicit image-level semantic vector.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        out_channels: int = 256,
        pyramid_channels: int = 64,
        pretrained_path: str = "",
        freeze_backbone: bool = False,
    ):
        super().__init__()
        out_channels = int(out_channels)
        pyramid_channels = int(pyramid_channels)
        if out_channels <= 0 or pyramid_channels <= 0:
            raise ValueError("out_channels and pyramid_channels must be positive")

        backbone = _build_resnet(backbone_name)
        self.backbone_name = str(backbone_name).strip().lower()
        self.freeze_backbone = bool(freeze_backbone)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.projections = nn.ModuleList(
            nn.Conv2d(channels, pyramid_channels, kernel_size=1, bias=False)
            for channels in (64, 128, 256, 512)
        )
        merged_channels = 4 * pyramid_channels
        groups = 16 if out_channels % 16 == 0 else 8
        self.fuse = nn.Sequential(
            nn.Conv2d(merged_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
        )
        self.global_proj = nn.Sequential(
            nn.Conv2d(512, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )
        self.out_norm = nn.GroupNorm(groups, out_channels)

        if str(pretrained_path or "").strip():
            self.load_backbone_checkpoint(str(pretrained_path))
        if self.freeze_backbone:
            self._freeze_encoder()

    @property
    def encoder_modules(self) -> Tuple[nn.Module, ...]:
        return self.stem, self.layer1, self.layer2, self.layer3, self.layer4

    def _freeze_encoder(self) -> None:
        for module in self.encoder_modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            for module in self.encoder_modules:
                module.eval()
        return self

    def load_backbone_checkpoint(self, path: str) -> None:
        raw = torch.load(path, map_location="cpu")
        raw_state = _unwrap_state_dict(raw)
        target = OrderedDict()
        for prefix, module in (
            ("stem.0.", self.stem[0]),
            ("stem.1.", self.stem[1]),
            ("layer1.", self.layer1),
            ("layer2.", self.layer2),
            ("layer3.", self.layer3),
            ("layer4.", self.layer4),
        ):
            for key, value in module.state_dict().items():
                target[prefix + key] = value

        aliases = {"conv1.": "stem.0.", "bn1.": "stem.1."}
        reusable = {}
        for key, value in raw_state.items():
            clean = _strip_prefixes(str(key))
            for source, destination in aliases.items():
                if clean.startswith(source):
                    clean = destination + clean[len(source):]
                    break
            if clean in target and tuple(value.shape) == tuple(target[clean].shape):
                reusable[clean] = value

        if not reusable:
            raise RuntimeError(f"No compatible ResNet encoder weights found in {path}")
        coverage = len(reusable) / max(len(target), 1)
        if coverage < 0.80:
            raise RuntimeError(
                f"ResNet encoder checkpoint coverage is too low: {len(reusable)}/{len(target)} "
                f"({coverage:.1%}) from {path}"
            )

        current = self.state_dict()
        current.update(reusable)
        self.load_state_dict(current, strict=True)
        print(
            f"[Pure2D-CNN] loaded backbone={self.backbone_name} "
            f"weights={len(reusable)}/{len(target)} path={path}",
            flush=True,
        )

    def _encode(self, image: torch.Tensor):
        x = self.stem(image)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5

    def forward(self, image: torch.Tensor):
        if image.dim() != 4 or image.size(1) != 3:
            raise ValueError(f"Expected image [B,3,H,W], got {tuple(image.shape)}")
        if self.freeze_backbone:
            with torch.no_grad():
                features = self._encode(image)
        else:
            features = self._encode(image)

        target_hw = features[0].shape[-2:]
        pyramid = []
        for feature, projection in zip(features, self.projections):
            projected = projection(feature)
            if projected.shape[-2:] != target_hw:
                projected = F.interpolate(projected, size=target_hw, mode="bilinear", align_corners=False)
            pyramid.append(projected)

        fused = self.fuse(torch.cat(pyramid, dim=1))
        global_context = self.global_proj(F.adaptive_avg_pool2d(features[-1], output_size=1))
        return F.gelu(self.out_norm(fused + global_context))

