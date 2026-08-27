"""Supervised Flow loss wrapper for the MoonViT mainline."""

from __future__ import annotations

import torch
import torch.nn as nn

from lib.config import cfg


class DiffusionPretrainNetworkWrapper(nn.Module):
    """Add the supervised Flow objective without constructing detector losses."""

    _DIAGNOSTIC_KEYS = (
        "routeb_box_jitter_count",
        "routeb_box_jitter_clean_count",
        "routeb_box_jitter_mean_iou",
        "routeb_box_jitter_min_iou",
        "routeb_box_jitter_severity_0_count",
        "routeb_box_jitter_severity_1_count",
        "routeb_box_jitter_severity_2_count",
        "routeb_box_jitter_severity_3_count",
        "locate_feat_replace_absmax",
    )

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        if str(cfg.detector_backend).strip().lower() != "flow_box_only":
            raise ValueError("the mainline trainer only supports flow_box_only")
        self.diffusion_weight = float(cfg.diffusion_loss_weight)
        if self.diffusion_weight <= 0.0:
            raise ValueError("diffusion loss weight must be positive")

    def forward(self, batch: dict):
        output = self.net(batch["inp"], batch)
        if "diff_loss" not in output:
            raise RuntimeError("Flow forward pass did not return diff_loss")
        diffusion_loss = output["diff_loss"]
        if not torch.isfinite(diffusion_loss).all():
            raise FloatingPointError("non-finite supervised Flow loss")
        loss = self.diffusion_weight * diffusion_loss
        scalar_stats = {
            "diff_loss": diffusion_loss.detach(),
            "diff_loss_scaled": loss.detach(),
            "loss": loss.detach(),
        }
        for key, value in output.items():
            if isinstance(key, str) and key.startswith("diff_loss"):
                scalar_stats[key] = value.detach() if torch.is_tensor(value) else value
            if isinstance(key, str) and key.startswith("moe_"):
                scalar_stats[key] = value.detach() if torch.is_tensor(value) else value
        for key in self._DIAGNOSTIC_KEYS:
            if key not in output:
                continue
            value = output[key]
            scalar_stats[key] = (
                value.detach()
                if torch.is_tensor(value)
                else torch.tensor(float(value), device=loss.device)
            )

        if self.training:
            output = {}
        return output, loss, scalar_stats, {}
