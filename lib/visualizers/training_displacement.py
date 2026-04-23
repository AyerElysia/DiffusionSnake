import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from lib.utils.snake import snake_config


def _to_numpy(array_like):
    if isinstance(array_like, torch.Tensor):
        array_like = array_like.detach().cpu().numpy()
    return np.asarray(array_like, dtype=np.float32)


def _to_display_image(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)

    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    image = image.astype(np.float32)
    if image.max() <= 1.5:
        image = image * 255.0
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    if image.ndim == 3 and image.shape[2] == 3:
        image = image[..., ::-1]
    return image


def _close_poly(poly):
    return np.concatenate([poly, poly[:1]], axis=0)


def _compute_limits(init_poly, gt_poly, pad=12.0):
    both = np.concatenate([init_poly, gt_poly], axis=0)
    x_min = float(np.min(both[:, 0]) - pad)
    x_max = float(np.max(both[:, 0]) + pad)
    y_min = float(np.min(both[:, 1]) - pad)
    y_max = float(np.max(both[:, 1]) + pad)
    return x_min, x_max, y_min, y_max


def _compute_stride(num_points, vector_stride=None):
    if vector_stride is not None and vector_stride > 0:
        return int(vector_stride)
    return max(1, num_points // 32)


def _draw_poly(ax, poly, color, label, linewidth, linestyle='-'):
    closed = _close_poly(poly)
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=linewidth, linestyle=linestyle, label=label)


def _draw_overlay(ax, image, init_poly, gt_poly, disp, stride):
    ax.imshow(image)
    _draw_poly(ax, init_poly, color='#ffb703', label='Init contour', linewidth=2.0, linestyle='--')
    _draw_poly(ax, gt_poly, color='#219ebc', label='GT contour', linewidth=2.2)

    mag = np.linalg.norm(disp, axis=1)
    sample_idx = np.arange(0, len(init_poly), stride, dtype=np.int32)
    q = ax.quiver(
        init_poly[sample_idx, 0],
        init_poly[sample_idx, 1],
        disp[sample_idx, 0],
        disp[sample_idx, 1],
        mag[sample_idx],
        cmap='viridis',
        angles='xy',
        scale_units='xy',
        scale=1.0,
        width=0.004,
        alpha=0.95,
    )

    ax.scatter(init_poly[0, 0], init_poly[0, 1], s=42, color='#d00000', edgecolors='white', linewidths=0.8, zorder=5, label='Init start')
    ax.scatter(gt_poly[0, 0], gt_poly[0, 1], s=34, color='#3a86ff', edgecolors='white', linewidths=0.8, zorder=5, label='GT start')
    ax.set_title('Image Overlay', fontsize=14, fontweight='bold')
    ax.set_axis_off()
    ax.legend(loc='upper right', fontsize=9, frameon=True)
    return q


def _draw_canvas(ax, init_poly, gt_poly, disp, stride):
    x_min, x_max, y_min, y_max = _compute_limits(init_poly, gt_poly)
    _draw_poly(ax, init_poly, color='#fb8500', label='Init contour', linewidth=2.0, linestyle='--')
    _draw_poly(ax, gt_poly, color='#023047', label='GT contour', linewidth=2.4)

    mag = np.linalg.norm(disp, axis=1)
    sample_idx = np.arange(0, len(init_poly), stride, dtype=np.int32)
    ax.quiver(
        init_poly[sample_idx, 0],
        init_poly[sample_idx, 1],
        disp[sample_idx, 0],
        disp[sample_idx, 1],
        mag[sample_idx],
        cmap='viridis',
        angles='xy',
        scale_units='xy',
        scale=1.0,
        width=0.005,
        alpha=0.95,
    )

    ax.scatter(init_poly[:, 0], init_poly[:, 1], s=10, color='#fb8500', alpha=0.55)
    ax.scatter(gt_poly[:, 0], gt_poly[:, 1], s=10, color='#023047', alpha=0.55)
    ax.scatter(init_poly[0, 0], init_poly[0, 1], s=52, color='#d00000', edgecolors='white', linewidths=0.9, zorder=5)
    ax.scatter(gt_poly[0, 0], gt_poly[0, 1], s=42, color='#3a86ff', edgecolors='white', linewidths=0.9, zorder=5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect('equal')
    ax.grid(alpha=0.18, linestyle=':')
    ax.set_title('Displacement Field', fontsize=14, fontweight='bold')


def _draw_curves(ax, disp):
    idx = np.arange(len(disp), dtype=np.int32)
    mag = np.linalg.norm(disp, axis=1)

    ax.plot(idx, disp[:, 0], color='#e63946', linewidth=1.8, label='dx')
    ax.plot(idx, disp[:, 1], color='#457b9d', linewidth=1.8, label='dy')
    ax.plot(idx, mag, color='#2a9d8f', linewidth=2.1, label='|disp|')
    ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.4)
    ax.set_title('Per-Point Displacement', fontsize=14, fontweight='bold')
    ax.set_xlabel('Point index')
    ax.set_ylabel('Pixels')
    ax.grid(alpha=0.22, linestyle=':')
    ax.legend(loc='upper right', fontsize=9)


def _stats_text(init_poly, gt_poly, disp):
    mag = np.linalg.norm(disp, axis=1)
    bbox_init = np.array([
        np.min(init_poly[:, 0]), np.min(init_poly[:, 1]),
        np.max(init_poly[:, 0]), np.max(init_poly[:, 1]),
    ], dtype=np.float32)
    bbox_gt = np.array([
        np.min(gt_poly[:, 0]), np.min(gt_poly[:, 1]),
        np.max(gt_poly[:, 0]), np.max(gt_poly[:, 1]),
    ], dtype=np.float32)
    return '\n'.join([
        f'Init mode: {snake_config.evolve_init}',
        f'Points: {len(init_poly)}',
        f'dx mean/std: {disp[:, 0].mean():.2f} / {disp[:, 0].std():.2f}',
        f'dy mean/std: {disp[:, 1].mean():.2f} / {disp[:, 1].std():.2f}',
        f'|disp| min/mean/max: {mag.min():.2f} / {mag.mean():.2f} / {mag.max():.2f}',
        f'Init bbox: [{bbox_init[0]:.1f}, {bbox_init[1]:.1f}] -> [{bbox_init[2]:.1f}, {bbox_init[3]:.1f}]',
        f'GT bbox: [{bbox_gt[0]:.1f}, {bbox_gt[1]:.1f}] -> [{bbox_gt[2]:.1f}, {bbox_gt[3]:.1f}]',
    ])


def save_training_displacement_figure(image, init_poly, gt_poly, save_path, title='', subtitle='', vector_stride=None):
    init_poly = _to_numpy(init_poly)
    gt_poly = _to_numpy(gt_poly)
    if init_poly.shape != gt_poly.shape:
        raise ValueError(f'init_poly and gt_poly must share shape, got {init_poly.shape} vs {gt_poly.shape}')

    disp = gt_poly - init_poly
    stride = _compute_stride(len(init_poly), vector_stride)
    image = _to_display_image(image)

    fig = plt.figure(figsize=(16, 10), dpi=180, constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 0.78])
    ax_overlay = fig.add_subplot(grid[:, 0])
    ax_canvas = fig.add_subplot(grid[0, 1])
    ax_curve = fig.add_subplot(grid[1, 1])

    quiver_artist = _draw_overlay(ax_overlay, image, init_poly, gt_poly, disp, stride)
    _draw_canvas(ax_canvas, init_poly, gt_poly, disp, stride)
    _draw_curves(ax_curve, disp)
    ax_curve.text(
        0.02,
        0.98,
        _stats_text(init_poly, gt_poly, disp),
        transform=ax_curve.transAxes,
        va='top',
        ha='left',
        fontsize=9,
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.88, 'edgecolor': '#cccccc'},
    )

    cbar = fig.colorbar(quiver_artist, ax=[ax_overlay, ax_canvas], fraction=0.03, pad=0.02)
    cbar.set_label('Displacement magnitude (px)')

    header = title or 'Training Displacement Visualization'
    if subtitle:
        header = f'{header}\n{subtitle}'
    fig.suptitle(header, fontsize=17, fontweight='bold')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def save_sample_training_displacements(sample, save_dir, sample_index, contour_indices=None, max_contours=4, vector_stride=None):
    image = sample.get('orig_img', sample.get('inp'))
    init_polys = sample.get('i_it_py', [])
    gt_polys = sample.get('i_gt_py', [])
    if len(init_polys) == 0 or len(gt_polys) == 0:
        raise ValueError('sample does not contain training contours')

    contour_count = min(len(init_polys), len(gt_polys))
    if contour_indices is None:
        contour_indices = list(range(min(contour_count, max_contours)))
    else:
        contour_indices = [idx for idx in contour_indices if 0 <= idx < contour_count]

    if not contour_indices:
        raise ValueError('no valid contour indices selected')

    img_name = os.path.basename(str(sample.get('img_path', 'sample')))
    saved_paths = []
    for contour_idx in contour_indices:
        init_poly = init_polys[contour_idx]
        gt_poly = gt_polys[contour_idx]
        save_path = os.path.join(
            save_dir,
            f'sample_{int(sample_index):05d}_contour_{int(contour_idx):02d}.png',
        )
        save_training_displacement_figure(
            image=image,
            init_poly=init_poly,
            gt_poly=gt_poly,
            save_path=save_path,
            title=f'Sample {int(sample_index):05d} | Contour {int(contour_idx):02d}',
            subtitle=f'{img_name} | init={snake_config.evolve_init}',
            vector_stride=vector_stride,
        )
        saved_paths.append(save_path)
    return saved_paths
