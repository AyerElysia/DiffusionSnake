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
        if float(cfg.loss_scales.get("det", 0.0)) != 0.0:
            raise ValueError("detector loss must be zero for the detector-free mainline")
        self.diffusion_weight = float(cfg.diffusion_loss_weight) * float(
            cfg.loss_scales.get("diff", 1.0)
        )
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
        zero = loss.detach() * 0.0
        scalar_stats = {
            "det_loss": zero,
            "mask_loss": zero,
            "ex_loss": zero,
            "eagle_ex_loss": zero,
            "L_loss": zero,
            "smooth_loss": zero,
            "curv_loss": zero,
            "diff_loss": diffusion_loss.detach(),
            "diff_loss_scaled": loss.detach(),
            "det_plus_diff_loss": diffusion_loss.detach(),
            "loss": loss.detach(),
        }
        for key, value in output.items():
            if isinstance(key, str) and key.startswith("diff_loss"):
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

        if self.training and not bool(cfg.volmem_return_training_prediction):
            output = {}
        elif self.training:
            output = {
                key: output[key]
                for key in ("pred_contours", "py_ind")
                if key in output
            }
        return output, loss, scalar_stats, {}
