"""Self-test for instance-color visualization helpers (no torch/cfg needed)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np

from lib.utils.snake.viz_colors import (
    UNMATCHED_COLOR_BGR,
    draw_dotted_bbox,
    draw_instance_legend,
    draw_polyline_style,
    instance_color,
)


def make_circle(cx, cy, r, n=64):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1).astype(np.float32)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'viz_instance_colors_out')
    os.makedirs(out_dir, exist_ok=True)

    img = np.full((512, 512, 3), 90, dtype=np.uint8)

    gt_polys = [make_circle(140, 150, 70), make_circle(360, 160, 55), make_circle(250, 380, 80)]
    # preds slightly offset from their GTs; pred 3 is unmatched
    pred_polys = [make_circle(146, 156, 66), make_circle(354, 166, 58), make_circle(256, 374, 76),
                  make_circle(440, 440, 30)]
    init_polys = [make_circle(150, 160, 75), make_circle(350, 170, 62), make_circle(260, 370, 85),
                  make_circle(440, 440, 35)]
    matched_pred = [0, 1, 2]

    pred_to_gt = {int(p): g for g, p in enumerate(matched_pred)}

    vis = img.copy()
    for pred_idx, poly in enumerate(init_polys):
        gt_idx = pred_to_gt.get(pred_idx)
        color = instance_color(gt_idx) if gt_idx is not None else UNMATCHED_COLOR_BGR
        if gt_idx is None:
            continue  # default: skip unmatched
        draw_dotted_bbox(vis, poly, color, thickness=1)
    for gt_idx, poly in enumerate(gt_polys):
        draw_polyline_style(vis, poly, instance_color(gt_idx), thickness=2, style='solid')
    for pred_idx, poly in enumerate(pred_polys):
        gt_idx = pred_to_gt.get(pred_idx)
        if gt_idx is None:
            continue
        draw_polyline_style(vis, poly, instance_color(gt_idx), thickness=2, style='dashed')
    draw_instance_legend(vis, range(len(gt_polys)))

    out_path = os.path.join(out_dir, 'overlay.png')
    ok = cv2.imwrite(out_path, vis)
    assert ok and os.path.exists(out_path), 'overlay.png not written'

    # variant with unmatched pred shown in gray
    vis2 = img.copy()
    for gt_idx, poly in enumerate(gt_polys):
        draw_polyline_style(vis2, poly, instance_color(gt_idx), thickness=2, style='solid')
    for pred_idx, poly in enumerate(pred_polys):
        gt_idx = pred_to_gt.get(pred_idx)
        color = instance_color(gt_idx) if gt_idx is not None else UNMATCHED_COLOR_BGR
        draw_polyline_style(vis2, poly, color, thickness=2, style='dashed')
        draw_dotted_bbox(vis2, init_polys[pred_idx], color, thickness=1)
    draw_instance_legend(vis2, range(len(gt_polys)))
    cv2.imwrite(os.path.join(out_dir, 'overlay_show_unmatched.png'), vis2)

    print('[OK] wrote', out_path)


if __name__ == '__main__':
    main()
