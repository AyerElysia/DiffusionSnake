"""Contour-target construction shared by the VerSe slice dataset.

This is the small, dataset-agnostic part of the original VOC loader that the
mainline still needs.  Keeping it here avoids importing the retired natural-
image dataset and its COCO-specific initialization path.
"""

from __future__ import annotations

import numpy as np

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
    def prepare_box_target(box, cls_id, output_width, wh, ct_cls, ct_ind):
        """Store the GT box/class fields consumed by the Route-B mainline."""
        ct_cls.append(cls_id)
        x_min, y_min, x_max, y_max = box
        center = np.asarray(
            [(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32
        )
        center = np.round(center).astype(np.int32)
        height, width = y_max - y_min, x_max - x_min
        wh.append([width, height])
        ct_ind.append(center[1] * int(output_width) + center[0])

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
        point_count = snake_config.poly_num
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
