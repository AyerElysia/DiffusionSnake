"""GRPO V2 helper utilities.

Stand-alone module for V2 RL post-training. Contains:
- EMA running mean for reward baseline.
- FrozenRefPolicy: clones the base flow policy with frozen weights for KL.
- compute_eval_metrics: deterministic eval on a fixed batch (IoU, Dice, mBoundF).
"""
from __future__ import annotations
import copy
import math
from pathlib import Path
import torch
import numpy as np

from lib.train.rewards.region_reward import (
    _poly_to_mask_np, _calc_iou, _calc_dice, _calc_mboundf,
)


class EMA:
    """Scalar exponential moving average."""
    def __init__(self, decay: float = 0.99):
        self.decay = float(decay)
        self.value: float = 0.0
        self.initialized = False

    def update(self, x: float) -> float:
        x = float(x)
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * x
        return self.value


def freeze_ref_flow(net) -> torch.nn.Module:
    """Deep-copy the flow GCN and freeze it (used as reference policy for KL)."""
    ref = copy.deepcopy(net.gcn).cuda()
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()
    return ref


@torch.no_grad()
def deterministic_predict(net, batch) -> torch.Tensor:
    """Run a full deterministic flow-ODE rollout (action_std=0) and return predicted polygons.

    Returns `py` of shape (M, P, 2) at feature scale.
    """
    net.eval()
    out = net(batch['inp'], batch)
    py_list = out.get('py', None)
    if py_list is None:
        return None
    py = py_list[-1] if isinstance(py_list, list) else py_list
    return py.detach()


@torch.no_grad()
def compute_eval_metrics(net, batch, down_ratio: int) -> dict:
    """Run a deterministic forward pass and compute IoU/Dice/mBoundF vs GT.

    Returns dict with mean values. Empty dict if no GT/contours.
    Forces train()-mode forward (no grad) so GT-processing branch fires, but BN
    stays frozen (eval mode). Restores network mode at end.
    """
    was_training = net.training
    net.train()
    freeze_bn_running_stats(net)  # keep BN frozen even in train() mode
    try:
        out = net(batch['inp'], batch)
    finally:
        if not was_training:
            net.eval()
        else:
            freeze_bn_running_stats(net)
    py_list = out.get('py', None)
    if py_list is None:
        return {}
    py = py_list[-1] if isinstance(py_list, list) else py_list
    gt = out.get('i_gt_py', None)
    if not isinstance(py, torch.Tensor) or not isinstance(gt, torch.Tensor):
        return {}
    n = min(py.size(0), gt.size(0))
    if n == 0:
        return {}

    H = int(batch['inp'].shape[-2])
    W = int(batch['inp'].shape[-1])
    pred_np = py[:n].detach().float().cpu().numpy() * float(down_ratio)
    gt_np = gt[:n].detach().float().cpu().numpy() * float(down_ratio)

    ious, dices, mbfs = [], [], []
    for i in range(n):
        mp = _poly_to_mask_np(pred_np[i], H, W)
        mg = _poly_to_mask_np(gt_np[i], H, W)
        ious.append(_calc_iou(mp, mg))
        dices.append(_calc_dice(mp, mg))
        mbfs.append(_calc_mboundf(mp, mg))
    return {
        'eval_iou': float(np.mean(ious)),
        'eval_dice': float(np.mean(dices)),
        'eval_mboundf': float(np.mean(mbfs)),
        'eval_n': int(n),
    }


def freeze_bn_running_stats(model: torch.nn.Module) -> int:
    """Set all BN layers to eval() so running mean/var are frozen during RL.

    Returns count of BN layers frozen. Keeps params trainable but stats frozen.
    Crucial for RL fine-tuning to avoid policy drift from BN stat updates.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                          torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)):
            m.eval()
            n += 1
    return n


def percentiles(xs: torch.Tensor) -> dict:
    """Return p10/p50/p90 of a 1-D tensor (or empty if too small)."""
    if xs.numel() == 0:
        return {'p10': 0.0, 'p50': 0.0, 'p90': 0.0}
    q = torch.quantile(xs.float(), torch.tensor([0.10, 0.50, 0.90], device=xs.device))
    return {'p10': float(q[0]), 'p50': float(q[1]), 'p90': float(q[2])}
