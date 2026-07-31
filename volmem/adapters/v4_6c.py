from typing import Dict, Optional, Tuple

import torch
from torch import nn

from volmem.detection import DetectionPolicy, LocateAnythingCache, build_detection_tensor


def build_detection_provider(config: object):
    """Construct the optional cached-box provider from config-like attributes."""
    source = str(getattr(config, "box_source", "detector") or "detector").strip().lower()
    if source in {"detector", "gt", "toy"}:
        return None, None
    if source != "locany_cached":
        raise ValueError(f"unsupported box_source: {source}")
    cache_path = str(getattr(config, "locany_cache_path", "") or "").strip()
    if not cache_path:
        raise ValueError("locany_cache_path is required when box_source=locany_cached")
    cache = LocateAnythingCache.from_path(cache_path)
    policy = DetectionPolicy(
        min_score=getattr(config, "locany_min_score", 1e-4),
        min_box_side=getattr(config, "locany_min_box_side", 1.0),
        min_box_area=getattr(config, "locany_min_box_area", 4.0),
        nms_iou=getattr(config, "locany_nms_iou", 0.5),
        max_detections=getattr(config, "locany_max_detections", 32),
        class_aware_nms=bool(getattr(config, "locany_class_aware_nms", True)),
        missing=str(getattr(config, "locany_missing", "error")),
    )
    policy.validate()
    return cache, policy


class V46cContourAdapter(nn.Module):
    """Isolation boundary around the inherited single-slice V4.6c wrapper."""

    def __init__(
        self,
        slice_loss_wrapper: nn.Module,
        detection_cache: Optional[LocateAnythingCache] = None,
        detection_policy: Optional[DetectionPolicy] = None,
    ) -> None:
        super().__init__()
        self.slice_loss_wrapper = slice_loss_wrapper
        self.detection_cache = detection_cache
        self.detection_policy = detection_policy or DetectionPolicy()

    def _prepare_batch(
        self,
        batch: Dict[str, object],
        for_training: bool,
    ) -> Dict[str, object]:
        if self.detection_cache is None:
            return batch
        if for_training:
            raise RuntimeError(
                "locany_cached is evaluation-only until class-aware one-to-one "
                "training initialization is implemented"
            )
        step_batch = dict(batch)
        step_batch["external_detection"] = build_detection_tensor(
            self.detection_cache,
            batch,
            self.detection_policy,
        )
        step_batch["external_detection_source"] = "locany_cached"
        return step_batch

    def forward(
        self,
        batch: Dict[str, object],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _, loss, stats = self.forward_with_output(batch)
        return loss.mean(), stats

    def forward_with_output(
        self,
        batch: Dict[str, object],
    ):
        output, loss, stats, _ = self.slice_loss_wrapper(
            self._prepare_batch(batch, for_training=True)
        )
        return output, loss.mean(), stats

    def predict(self, batch: Dict[str, object]) -> Dict[str, object]:
        """Run the inherited V4.6c network without its training-loss wrapper."""
        step_batch = self._prepare_batch(batch, for_training=False)
        return self.slice_loss_wrapper.net(step_batch["inp"], step_batch)
