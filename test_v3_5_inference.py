"""Test V3.5 inference: run single-sample inference and visualize results."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.config import cfg


def main():
    device = torch.device('cuda:0')

    # Build network + load checkpoint
    from lib.networks import make_network
    net = make_network(cfg).to(device)

    ckpt_path = os.path.join(cfg.model_dir, 'checkpoints', 'latest.pt')
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    cleaned = {(k[4:] if k.startswith('net.') else k): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    net.eval()

    # Load data
    from lib.datasets import make_data_loader
    dl = make_data_loader(cfg, is_train=False)
    batch = next(iter(dl))
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
        elif isinstance(batch[k], list):
            batch[k] = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch[k]]

    img_np = batch['inp'][0].cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    # Inference
    with torch.no_grad():
        output = net(batch['inp'], batch)

    print(f"Output keys: {list(output.keys())}")

    py = output.get('py', None)
    if isinstance(py, torch.Tensor) and py.numel() > 0:
        if py.dim() == 2:
            py = py.unsqueeze(0)
        py_np = py[0].cpu().numpy()

        # Smoothness metric
        diff1 = py_np[1:] - py_np[:-1]
        diff2 = diff1[1:] - diff1[:-1]
        smoothness = np.sqrt((diff2**2).sum(-1)).sum()
        print(f"V3.5 contour: {py.shape}, smoothness={smoothness:.1f}")

        # GT
        gt = output.get('i_gt_py', None)
        if isinstance(gt, list) and len(gt) > 0:
            gt = gt[0]
        gt_np = None
        if isinstance(gt, torch.Tensor) and gt.numel() > 0:
            gt_np = gt[0].cpu().numpy() if gt.dim() == 3 else gt.cpu().numpy()

        # Save
        os.makedirs('visual/v3_5_results', exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        ax = axes[0]
        ax.imshow(img_np)
        ax.plot(py_np[:, 0], py_np[:, 1], 'c-', linewidth=2, label='V3.5 Pred')
        ax.plot([py_np[-1, 0], py_np[0, 0]], [py_np[-1, 1], py_np[0, 1]], 'c-', linewidth=2)
        if gt_np is not None:
            ax.plot(gt_np[:, 0], gt_np[:, 1], 'r--', linewidth=1.5, alpha=0.7, label='GT')
            ax.plot([gt_np[-1, 0], gt_np[0, 0]], [gt_np[-1, 1], gt_np[0, 1]], 'r--', linewidth=1.5, alpha=0.7)
        ax.set_title(f'V3.5 Fourier-Space (K=16) 10k epochs\nSmoothness: {smoothness:.1f}')
        ax.legend()

        ax = axes[1]
        ax.plot(py_np[:, 0], py_np[:, 1], 'b-o', markersize=3, linewidth=1.5, label='V3.5')
        ax.plot([py_np[-1, 0], py_np[0, 0]], [py_np[-1, 1], py_np[0, 1]], 'b-o', markersize=3, linewidth=1.5)
        if gt_np is not None:
            ax.plot(gt_np[:, 0], gt_np[:, 1], 'r--o', markersize=2, linewidth=1.0, alpha=0.7, label='GT')
            ax.plot([gt_np[-1, 0], gt_np[0, 0]], [gt_np[-1, 1], gt_np[0, 1]], 'r--', linewidth=1.0, alpha=0.7)
        ax.set_title('Contour Detail')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.legend()

        plt.tight_layout()
        plt.savefig('visual/v3_5_results/v3_5_epoch10000.png', dpi=150)
        plt.close()
        print("Saved: visual/v3_5_results/v3_5_epoch10000.png")
    else:
        print(f"No valid contour. py type: {type(py)}")
        if isinstance(py, torch.Tensor):
            print(f"  py shape: {py.shape}, numel: {py.numel()}")


if __name__ == '__main__':
    main()
