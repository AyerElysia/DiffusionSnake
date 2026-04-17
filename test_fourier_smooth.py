"""Test script: Fourier smoothing on V3.0 single-sample overfit checkpoint.
Loads the trained model, runs inference with and without Fourier smoothing,
saves visual comparison and metrics.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.config import cfg


def smoothness_metric(contour):
    """Sum of squared differences between consecutive points."""
    diff = contour[1:] - contour[:-1]
    return (diff ** 2).sum().item()


def main():
    # Load config
    cfg.merge_from_file('configs/btcv_diffusion_dit_v3_5_test_smooth.yaml')
    # Override GPU to 0 (CUDA_VISIBLE_DEVICES remaps physical GPU)
    cfg.defrost()
    cfg.gpus = [0]
    cfg.freeze()

    device = torch.device('cuda:0')  # Use cuda:0 since CUDA_VISIBLE_DEVICES remaps

    # Load full network
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    from lib.datasets import make_data_loader

    network = make_network(cfg).to(device)

    # Load V3.0 10k checkpoint (diffusion_train format: .pt with 'state_dict')
    ckpt_path = os.path.join(cfg.model_dir, 'checkpoints', 'latest.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(cfg.model_dir, 'checkpoints', 'epoch_10000.pt')
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
    # Strip 'net.' prefix if present (diffusion_train saves under DiffusionTrainer.net)
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith('net.'):
            cleaned[k[4:]] = v
        else:
            cleaned[k] = v
    missing, unexpected = network.load_state_dict(cleaned, strict=False)
    print(f"  Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print(f"  Missing keys (first 5): {missing[:5]}")
    network.eval()

    # Load data
    data_loader = make_data_loader(cfg, is_train=False)
    batch = next(iter(data_loader))
    for k in batch:
        if k != 'meta':
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
            elif isinstance(batch[k], list):
                batch[k] = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch[k]]

    # Get image for visualization
    img_np = batch['inp'][0].cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    # --- Run inference with different K values ---
    results = {}

    # K=0 (no smoothing)
    from lib.config import cfg as global_cfg
    # We can't modify frozen cfg, so use a workaround via the network directly
    from lib.networks.diffusion.pretrain_evolution import DiffusionEvolution

    # Access the diffusion module
    diff_evo = None
    for module in network.modules():
        if isinstance(module, DiffusionEvolution):
            diff_evo = module
            break

    if diff_evo is None:
        print("ERROR: Could not find DiffusionEvolution module in network!")
        return

    print("Found DiffusionEvolution module. Running inference tests...")

    # We'll run the internal forward manually for more control
    # First, get the CNN features and initial contours through the network's head
    with torch.no_grad():
        output = network(batch['inp'], batch)

    # The output already has 'py' (predicted contour) with fourier_smooth_k=12 applied
    # Let's also get raw (no smoothing) and other K values
    # We need to re-run sample_disp with different K values

    # Get the contour points from the output
    if 'py' in output:
        smoothed_py = output['py']  # This has K=12 smoothing
        print(f"Output 'py' shape: {smoothed_py.shape if isinstance(smoothed_py, torch.Tensor) else [p.shape for p in smoothed_py]}")

    # Also get i_it_py (initial contour) for reference
    if 'i_it_py' in output:
        init_py = output['i_it_py']
    elif 'i_it_4py' in output:
        init_py = output['i_it_4py']

    # Get GT contours
    if 'i_gt_py' in batch:
        gt_py = batch['i_gt_py']

    # Now let's run inference multiple times with different K values
    # We need to temporarily modify the global config
    # Since cfg is frozen, we use object.__setattr__
    k_values = [0, 8, 12, 16, 24]

    for k in k_values:
        print(f"\n--- Testing K={k} ---")
        object.__setattr__(global_cfg, 'fourier_smooth_k', k)
        with torch.no_grad():
            output_k = network(batch['inp'], batch)

        if 'py' in output_k:
            py = output_k['py']
            if isinstance(py, list):
                py = py[0]
            results[k] = py.cpu().numpy()
            # Compute smoothness for each contour
            total_smooth = 0
            for i in range(py.shape[0]):
                s = smoothness_metric(py[i].cpu().numpy())
                total_smooth += s
            avg_smooth = total_smooth / max(py.shape[0], 1)
            print(f"  K={k}: avg_smoothness={avg_smooth:.2f}, num_contours={py.shape[0]}")

    # Restore original K
    object.__setattr__(global_cfg, 'fourier_smooth_k', 12)

    # --- Visualization ---
    out_dir = 'visual/fourier_smooth_test'
    os.makedirs(out_dir, exist_ok=True)

    if results:
        # Pick the first contour from each result for visualization
        fig, axes = plt.subplots(1, len(k_values), figsize=(5 * len(k_values), 5))
        if len(k_values) == 1:
            axes = [axes]

        for ax_idx, k in enumerate(k_values):
            ax = axes[ax_idx]
            ax.imshow(img_np)
            py_k = results[k]
            n_contours = py_k.shape[0]
            colors = plt.cm.tab10(np.linspace(0, 1, min(n_contours, 10)))
            for ci in range(min(n_contours, 20)):
                contour = py_k[ci]
                # Close the contour for visualization
                contour_closed = np.concatenate([contour, contour[:1]], axis=0)
                c = colors[ci % len(colors)]
                ax.plot(contour_closed[:, 0], contour_closed[:, 1], '-', color=c, linewidth=1.5)
            label = f'K={k}' if k > 0 else 'No smooth'
            ax.set_title(label, fontsize=14)
            ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(out_dir, 'fourier_smooth_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved comparison to {save_path}")

        # Also save individual high-res images
        for k in k_values:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(img_np)
            py_k = results[k]
            n_contours = py_k.shape[0]
            colors = plt.cm.tab10(np.linspace(0, 1, min(n_contours, 10)))
            for ci in range(min(n_contours, 20)):
                contour = py_k[ci]
                contour_closed = np.concatenate([contour, contour[:1]], axis=0)
                c = colors[ci % len(colors)]
                ax.plot(contour_closed[:, 0], contour_closed[:, 1], '-', color=c, linewidth=1.5)
            label = f'K={k}' if k > 0 else 'No smooth'
            ax.set_title(label, fontsize=16)
            ax.axis('off')
            save_path = os.path.join(out_dir, f'fourier_k{k}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved {save_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("FOURIER SMOOTHING TEST SUMMARY")
    print("=" * 60)
    for k in k_values:
        if k in results:
            py_k = results[k]
            total_smooth = 0
            for i in range(py_k.shape[0]):
                total_smooth += smoothness_metric(py_k[i])
            avg = total_smooth / max(py_k.shape[0], 1)
            improvement = ""
            if k > 0 and 0 in results:
                raw_smooth = sum(smoothness_metric(results[0][i]) for i in range(results[0].shape[0])) / max(results[0].shape[0], 1)
                ratio = raw_smooth / max(avg, 1e-10)
                improvement = f"  ({ratio:.1f}x smoother)"
            print(f"  K={k:2d}: smoothness={avg:.2f}{improvement}")


if __name__ == '__main__':
    main()
