import importlib.util
from pathlib import Path

import torch

_MODULE_PATH = (
    Path(__file__).parents[1] / "lib/train/continuous_boundary_credit.py"
)
_SPEC = importlib.util.spec_from_file_location("continuous_boundary_credit", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
continuous_boundary_quality_delta = _MODULE.continuous_boundary_quality_delta
point_to_closed_polyline_distance = _MODULE.point_to_closed_polyline_distance


def test_point_on_segment_has_zero_distance():
    points = torch.tensor([[[1.0, 0.0]]])
    polyline = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]]])

    distance = point_to_closed_polyline_distance(points, polyline)

    torch.testing.assert_close(distance, torch.zeros_like(distance))


def test_known_perpendicular_distance():
    points = torch.tensor([[[1.0, 2.0]]])
    polyline = torch.tensor([
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [4.0, -2.0], [0.0, -2.0]]
    ])

    distance = point_to_closed_polyline_distance(points, polyline)

    torch.testing.assert_close(distance, torch.tensor([[2.0]]))


def test_closing_segment_is_included():
    points = torch.tensor([[[0.0, 1.0]]])
    polyline = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]])

    distance = point_to_closed_polyline_distance(points, polyline)

    torch.testing.assert_close(distance, torch.zeros_like(distance))


def test_batched_distance_shape():
    points = torch.tensor([
        [
            [[0.5, 0.5], [3.0, 0.0], [1.0, 1.0]],
            [[10.0, 10.0], [11.0, 10.0], [10.5, 10.5]],
        ],
        [
            [[1.0, 0.0], [2.0, 2.0], [0.0, 1.0]],
            [[10.0, 9.0], [12.0, 9.0], [10.0, 11.0]],
        ],
    ])
    polylines = torch.tensor([
        [
            [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            [[10.0, 9.0], [12.0, 9.0], [10.0, 11.0]],
        ],
        [
            [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            [[10.0, 9.0], [12.0, 9.0], [10.0, 11.0]],
        ],
    ])

    distance = point_to_closed_polyline_distance(points, polylines)

    assert distance.shape == (2, 2, 3)


def test_quality_delta_sign():
    gt = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]])
    pure = torch.tensor([[[1.0, 3.0], [1.0, 2.5]]])
    sampled = torch.tensor([[[1.0, 2.5], [1.0, 3.0]]])

    quality = continuous_boundary_quality_delta(
        sampled, pure, gt, coord_scale=4.0, dist_max_px=8.0
    )

    assert quality.shape == (1, 2)
    assert quality[0, 0] > 0
    assert quality[0, 1] < 0
