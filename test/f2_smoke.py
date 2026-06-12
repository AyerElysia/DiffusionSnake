#!/usr/bin/env python3
"""CPU smoke checks for F2 Locate feature replacement."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CFG_FILE", str(ROOT / "configs" / "e3_v8_2_boxjitter_mixinit_gpu7.yaml"))
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
sys.argv = [sys.argv[0]]

from lib.config import cfg  # noqa: E402
from lib.networks.snake.ct_snake import LocateFeatReplacer, Network  # noqa: E402


class MiniReplaceNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_ratio = 4.0
        self.locate_feat_inject = False
        self.locate_feat_replace = True
        self.locate_feat_replacer = LocateFeatReplacer(in_channels=2304)

    _batch_meta_tensor = staticmethod(Network._batch_meta_tensor)
    _batch_tensor = staticmethod(Network._batch_tensor)
    _build_locate_feature_grid = Network._build_locate_feature_grid
    apply_locate_feature_replacement = Network.apply_locate_feature_replacement

    def forward(self, cnn_feature, batch):
        return self.apply_locate_feature_replacement(cnn_feature, batch)


def make_npz(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        layer_18=rng.standard_normal((1152, 4, 4)).astype(np.float16),
        layer_26=rng.standard_normal((1152, 4, 4)).astype(np.float16),
        grid_hw=np.asarray([4, 4], dtype=np.int32),
        orig_hw=np.asarray([48, 48], dtype=np.int32),
        resized_hw=np.asarray([84, 84], dtype=np.int32),
        padded_hw=np.asarray([84, 84], dtype=np.int32),
        input_hw=np.asarray([84, 84], dtype=np.int32),
        pad=np.asarray([0, 0, 0, 0], dtype=np.int32),
        scale=np.asarray([1.75], dtype=np.float32),
        layers=np.asarray([18, 26], dtype=np.int32),
        patch_size=np.asarray([14], dtype=np.int32),
        input_size=np.asarray([896], dtype=np.int32),
        long_side=np.asarray([896], dtype=np.int32),
        image_path=np.asarray(str(path)),
    )


def load_batch(paths: list[Path]) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | list[str]]:
    feats = []
    grid_hw = []
    orig_hw = []
    resized_hw = []
    padded_hw = []
    pads = []
    scales = []
    patch_sizes = []
    for path in paths:
        with np.load(path) as npz:
            feats.append(np.concatenate([npz["layer_18"], npz["layer_26"]], axis=0))
            grid_hw.append(npz["grid_hw"])
            orig_hw.append(npz["orig_hw"])
            resized_hw.append(npz["resized_hw"])
            padded_hw.append(npz["padded_hw"])
            pads.append(npz["pad"])
            scales.append(npz["scale"])
            patch_sizes.append(npz["patch_size"])

    batch_size = len(paths)
    return {
        "locate_feat": torch.as_tensor(np.stack(feats, axis=0), dtype=torch.float16),
        "locate_feat_grid_hw": torch.as_tensor(np.stack(grid_hw, axis=0), dtype=torch.int32),
        "locate_feat_orig_hw": torch.as_tensor(np.stack(orig_hw, axis=0), dtype=torch.int32),
        "locate_feat_resized_hw": torch.as_tensor(np.stack(resized_hw, axis=0), dtype=torch.int32),
        "locate_feat_padded_hw": torch.as_tensor(np.stack(padded_hw, axis=0), dtype=torch.int32),
        "locate_feat_pad": torch.as_tensor(np.stack(pads, axis=0), dtype=torch.int32),
        "locate_feat_scale": torch.as_tensor(np.stack(scales, axis=0), dtype=torch.float32),
        "locate_feat_patch_size": torch.as_tensor(np.stack(patch_sizes, axis=0), dtype=torch.int32),
        "locate_feat_path": [str(p) for p in paths],
        "meta": {
            "inv_trans_input": torch.eye(2, 3, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1),
            "orig_hw": torch.tensor([[48, 48]] * batch_size, dtype=torch.float32),
            "flipped": torch.zeros((batch_size, 1), dtype=torch.float32),
        },
    }


def check_default_no_replacer() -> None:
    cfg.locate_feat_inject = False
    cfg.locate_feat_replace = False
    net = Network(34, {"ct_hm": 1, "wh": 2}, head_conv=16)
    assert not hasattr(net, "locate_feat_replacer") or net.locate_feat_replacer is None


def check_mutual_exclusion() -> None:
    old_inject = bool(getattr(cfg, "locate_feat_inject", False))
    old_replace = bool(getattr(cfg, "locate_feat_replace", False))
    cfg.locate_feat_inject = True
    cfg.locate_feat_replace = True
    try:
        try:
            Network(34, {"ct_hm": 1, "wh": 2}, head_conv=16)
        except ValueError:
            return
        raise AssertionError("expected mutually exclusive locate flags to raise ValueError")
    finally:
        cfg.locate_feat_inject = old_inject
        cfg.locate_feat_replace = old_replace


def check_forward_backward() -> None:
    with tempfile.TemporaryDirectory(prefix="f2_smoke_") as tmp:
        paths = [Path(tmp) / f"{idx}_image.npz" for idx in range(2)]
        for idx, path in enumerate(paths):
            make_npz(path, seed=idx)
        batch = load_batch(paths)

    model = MiniReplaceNet()
    params = sum(p.numel() for p in model.locate_feat_replacer.parameters())
    print(f"LocateFeatReplacer params={params} ({params / 1e6:.3f}M)")
    assert params < 2_000_000

    cnn_feature = torch.zeros((2, 64, 8, 8), dtype=torch.float32)
    out, stats = model(cnn_feature, batch)
    assert tuple(out.shape) == (2, 64, 8, 8)
    loss = out.square().mean()
    loss.backward()
    grad_norm = float(model.locate_feat_replacer.proj[0].weight.grad.abs().sum().item())
    assert grad_norm > 0.0
    assert "locate_feat_replace_absmax" in stats
    print(f"forward_backward_ok loss={float(loss.item()):.6f} grad_sum={grad_norm:.6f}")


def check_real_network_forward_backward() -> None:
    old_values = {
        "locate_feat_inject": bool(getattr(cfg, "locate_feat_inject", False)),
        "locate_feat_replace": bool(getattr(cfg, "locate_feat_replace", False)),
        "use_gt_det": bool(getattr(cfg, "use_gt_det", False)),
        "use_extreme_refine": bool(getattr(cfg, "use_extreme_refine", False)),
        "skip_diffusion_forward": bool(getattr(cfg, "skip_diffusion_forward", False)),
        "detector_only_warmup": bool(getattr(cfg, "detector_only_warmup", False)),
        "use_heatmap_mask_head": bool(getattr(cfg, "use_heatmap_mask_head", False)),
        "det_max_det": int(getattr(cfg, "det_max_det", 100)),
    }
    cfg.locate_feat_inject = False
    cfg.locate_feat_replace = True
    cfg.use_gt_det = False
    cfg.use_extreme_refine = False
    cfg.skip_diffusion_forward = True
    cfg.detector_only_warmup = True
    cfg.use_heatmap_mask_head = False
    cfg.det_max_det = 1
    try:
        with tempfile.TemporaryDirectory(prefix="f2_network_smoke_") as tmp:
            paths = [Path(tmp) / f"{idx}_image.npz" for idx in range(2)]
            for idx, path in enumerate(paths):
                make_npz(path, seed=10 + idx)
            batch = load_batch(paths)
            batch["inp"] = torch.randn((2, 3, 48, 48), dtype=torch.float32)

        net = Network(34, {"ct_hm": 52, "wh": 2}, head_conv=16).train()
        out = net(batch["inp"], batch)
        loss = out["cnn_feature"].square().mean()
        loss.backward()
        grad_sum = float(net.locate_feat_replacer.proj[0].weight.grad.abs().sum().item())
        assert grad_sum > 0.0
        print(f"real_network_forward_backward_ok loss={float(loss.item()):.6f} grad_sum={grad_sum:.6f}")
    finally:
        for key, value in old_values.items():
            setattr(cfg, key, value)


def main() -> None:
    check_default_no_replacer()
    check_mutual_exclusion()
    check_forward_backward()
    check_real_network_forward_backward()
    print("F2 smoke passed")


if __name__ == "__main__":
    main()
