import cv2
import numpy as np


_INSTANCE_PALETTE_BGR = [
    (0, 200, 255),    # amber
    (80, 240, 80),    # green
    (255, 140, 40),   # blue
    (220, 80, 255),   # magenta
    (40, 230, 230),   # yellow
    (255, 90, 170),   # violet
    (80, 170, 255),   # orange
    (60, 255, 170),   # lime
    (255, 220, 70),   # cyan-blue
    (170, 120, 255),  # pink
    (120, 255, 255),  # pale yellow
    (255, 180, 180),  # light blue
    (0, 140, 255),    # vivid orange
    (150, 255, 90),   # spring green
    (255, 100, 100),  # bright blue
    (100, 180, 255),  # peach
]

UNMATCHED_COLOR_BGR = (130, 130, 130)


def instance_color(idx):
    try:
        color_idx = int(idx)
    except Exception:
        color_idx = 0
    return _INSTANCE_PALETTE_BGR[color_idx % len(_INSTANCE_PALETTE_BGR)]


def draw_dashed_line(img, p1, p2, color, thickness=1, dash_len=8, gap_len=5):
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    vec = p2 - p1
    dist = float(np.linalg.norm(vec))
    if dist < 1e-3:
        return
    direction = vec / dist
    pos = 0.0
    while pos < dist:
        start = p1 + direction * pos
        end = p1 + direction * min(pos + dash_len, dist)
        cv2.line(
            img,
            tuple(np.round(start).astype(np.int32)),
            tuple(np.round(end).astype(np.int32)),
            color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        pos += dash_len + gap_len


def draw_polyline_style(img, poly, color, closed=True, thickness=1, style='solid'):
    pts = np.round(poly).astype(np.int32)
    if len(pts) < 2:
        return
    if style == 'solid':
        cv2.polylines(img, [pts], closed, color, thickness, lineType=cv2.LINE_AA)
        return
    n = len(pts)
    edge_count = n if closed else n - 1
    for i in range(edge_count):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        if style == 'dotted':
            draw_dashed_line(img, p1, p2, color, thickness=thickness, dash_len=2, gap_len=6)
        else:
            draw_dashed_line(img, p1, p2, color, thickness=thickness, dash_len=9, gap_len=6)


def draw_dotted_bbox(img, poly, color, thickness=1):
    pts = np.asarray(poly, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))
    corners = np.asarray(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )
    for i in range(4):
        draw_dashed_line(
            img,
            corners[i],
            corners[(i + 1) % 4],
            color,
            thickness=thickness,
            dash_len=2,
            gap_len=5,
        )


def draw_text_with_outline(
    img,
    text,
    org,
    font_scale=0.42,
    color=(255, 255, 255),
    outline_color=(0, 0, 0),
    thickness=1,
):
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        outline_color,
        thickness + 2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_instance_legend(img, instance_indices, x=8, y=18):
    indices = [int(i) for i in instance_indices]

    row_h = 18
    cols = 4
    text = 'solid=GT  dashed=Pred  dotted box=Det'
    rows = 1 + int(np.ceil(len(indices) / float(cols)))

    draw_text_with_outline(img, text, (x, y), font_scale=0.4, thickness=1)
    for pos, gt_idx in enumerate(indices):
        row = pos // cols
        col = pos % cols
        item_x = x + col * 82
        item_y = y + row_h * (row + 1)
        color = instance_color(gt_idx)
        cv2.line(img, (item_x, item_y - 4), (item_x + 22, item_y - 4), color, 2, lineType=cv2.LINE_AA)
        draw_text_with_outline(
            img,
            '#{}'.format(gt_idx),
            (item_x + 28, item_y),
            font_scale=0.38,
            thickness=1,
        )
    return y + rows * row_h + 8
