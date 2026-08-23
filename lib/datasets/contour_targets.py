"""Contour-target construction shared by the VerSe slice dataset.

This is the small, dataset-agnostic part of the original VOC loader that the
mainline still needs.  Keeping it here avoids importing the retired natural-
image dataset and its COCO-specific initialization path.
"""

from __future__ import annotations

import math

import numpy as np

from lib.config import cfg
from lib.utils import data_utils
from lib.utils.snake import snake_config, snake_voc_utils


class ContourTargetMixin:
    """Build detector, initialization, and evolution targets for one polygon."""

    def transform_original_data(
        self, instance_polys, flipped, width, trans_output, inp_out_hw
    ):
        output_h, output_w = inp_out_hw[2:]
        transformed_instances = []
        for instance in instance_polys:
            polygons = [polygon.reshape(-1, 2) for polygon in instance]
            if flipped:
                flipped_polygons = []
                for polygon in polygons:
                    polygon[:, 0] = width - np.asarray(polygon[:, 0]) - 1
                    flipped_polygons.append(polygon.copy())
                polygons = flipped_polygons
            transformed_instances.append(
                snake_voc_utils.transform_polys(
                    polygons, trans_output, output_h, output_w
                )
            )
        return transformed_instances

    def get_valid_polys(self, instance_polys, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        valid_instances = []
        for instance in instance_polys:
            polygons = [polygon for polygon in instance if len(polygon) >= 4]
            for polygon in polygons:
                polygon[:, 0] = np.clip(polygon[:, 0], 0, output_w - 1)
                polygon[:, 1] = np.clip(polygon[:, 1], 0, output_h - 1)
            polygons = snake_voc_utils.filter_tiny_polys(polygons)
            polygons = snake_voc_utils.get_cw_polys(polygons)
            polygons = [
                polygon[
                    np.sort(np.unique(polygon, axis=0, return_index=True)[1])
                ]
                for polygon in polygons
            ]
            valid_instances.append(polygons)
        return valid_instances

    @staticmethod
    def get_extreme_points(instance_polys):
        return [
            [snake_voc_utils.get_extreme_points(polygon) for polygon in instance]
            for instance in instance_polys
        ]

    @staticmethod
    def prepare_detection(box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind):
        del poly
        class_heatmap = ct_hm[cls_id]
        ct_cls.append(cls_id)
        x_min, y_min, x_max, y_max = box
        center = np.asarray(
            [(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32
        )
        center = np.round(center).astype(np.int32)
        height, width = y_max - y_min, x_max - x_min
        radius = max(
            0,
            int(
                data_utils.gaussian_radius(
                    (math.ceil(height), math.ceil(width))
                )
            ),
        )
        data_utils.draw_umich_gaussian(class_heatmap, center, radius)
        wh.append([width, height])
        ct_ind.append(center[1] * class_heatmap.shape[1] + center[0])
        return [
            center[0] - width / 2,
            center[1] - height / 2,
            center[0] + width / 2,
            center[1] + height / 2,
        ]

    @staticmethod
    def prepare_init(
        box,
        extreme_point,
        image_init_extremes,
        canonical_init_extremes,
        image_gt_extremes,
        canonical_gt_extremes,
        height,
        width,
    ):
        del height, width
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])
        image_init = snake_voc_utils.uniformsample(
            snake_voc_utils.get_init(box), snake_config.init_poly_num
        )
        canonical_init = snake_voc_utils.img_poly_to_can_poly(
            image_init, x_min, y_min, x_max, y_max
        )
        image_gt = extreme_point
        canonical_gt = snake_voc_utils.img_poly_to_can_poly(
            image_gt, x_min, y_min, x_max, y_max
        )
        image_init_extremes.append(image_init)
        canonical_init_extremes.append(canonical_init)
        image_gt_extremes.append(image_gt)
        canonical_gt_extremes.append(canonical_gt)

    def prepare_evolution(
        self,
        polygon,
        extreme_point,
        image_init_extremes,
        image_init_polys,
        canonical_init_polys,
        image_gt_polys,
        canonical_gt_polys,
    ):
        del image_init_extremes
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])
        box = [x_min, y_min, x_max, y_max]
        point_count = self.compute_adaptive_points(box)
        base_init = snake_voc_utils.get_evolution_init(extreme_point, box)
        image_init = snake_voc_utils.uniformsample(base_init, point_count)
        canonical_init = snake_voc_utils.img_poly_to_can_poly(
            image_init, x_min, y_min, x_max, y_max
        )
        image_gt = snake_voc_utils.uniformsample(
            polygon, len(polygon) * point_count
        )
        start_index = np.argmin(
            np.power(image_gt - image_init[0], 2).sum(axis=1)
        )
        image_gt = np.roll(image_gt, -start_index, axis=0)[:: len(polygon)]
        canonical_gt = snake_voc_utils.img_poly_to_can_poly(
            image_gt, x_min, y_min, x_max, y_max
        )
        image_init_polys.append(image_init)
        canonical_init_polys.append(canonical_init)
        image_gt_polys.append(image_gt)
        canonical_gt_polys.append(canonical_gt)

    @staticmethod
    def compute_adaptive_points(box):
        if not bool(getattr(snake_config, "adaptive_points_enabled", False)):
            return int(snake_config.poly_num)
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
        strategy = str(
            getattr(snake_config, "point_strategy", "perimeter")
        ).strip().lower()
        if bool(getattr(snake_config, "adaptive_use_area_threshold", False)) or strategy in {
            "area_threshold",
            "threshold",
        }:
            threshold = float(
                getattr(snake_config, "adaptive_area_threshold", 4096.0)
            )
            return int(
                getattr(
                    snake_config,
                    "adaptive_small_points"
                    if max(width, 0.0) * max(height, 0.0) < threshold
                    else "adaptive_large_points",
                    64 if max(width, 0.0) * max(height, 0.0) < threshold else 128,
                )
            )
        density = max(float(getattr(snake_config, "target_density", 2.5)), 1e-6)
        minimum = int(getattr(snake_config, "min_points", 32))
        maximum = int(getattr(snake_config, "max_points", 512))
        multiple = max(1, int(getattr(snake_config, "round_to_multiple", 8)))
        perimeter = 2.0 * (width + height) / density
        area = np.sqrt(max(width * height, 0.0)) * 1.5
        if strategy == "area":
            value = area
        elif strategy == "mixed":
            value = 0.6 * perimeter + 0.4 * area
        else:
            value = perimeter
        point_count = int(np.clip(value, minimum, maximum))
        point_count = ((point_count + multiple - 1) // multiple) * multiple
        return int(min(max(point_count, minimum), maximum))
