import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from volmem.detection import (
    CachedDetection,
    DetectionPolicy,
    LocateAnythingCache,
    build_detection_tensor,
    filter_detections,
    parse_raw_response,
    transform_detection,
)
from volmem.detection.locany_cache import flip_detection


NORMALIZER_PATH = PROJECT_ROOT / "tools" / "volmem" / "normalize_locany_cache.py"
NORMALIZER_SPEC = importlib.util.spec_from_file_location(
    "normalize_locany_cache", NORMALIZER_PATH
)
NORMALIZER = importlib.util.module_from_spec(NORMALIZER_SPEC)
NORMALIZER_SPEC.loader.exec_module(NORMALIZER)


class NormalizeLocateAnythingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary_directory.name)
        self.image_path = self.root / "slice.png"
        Image.new("L", (200, 100)).save(self.image_path)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def row(self, raw_response):
        return {
            "image": self.image_path.name,
            "query": "Locate each vertebra",
            "raw_response": raw_response,
            "stats": {"latency": 1.0},
        }

    def test_missing_image_fails(self):
        row = self.row("")
        del row["image"]
        with self.assertRaisesRegex(ValueError, "missing image"):
            NORMALIZER.normalize_row(row, self.root, strict=False)

    def test_non_string_image_fails(self):
        for image in (None, 7, True, [self.image_path.name]):
            row = self.row("")
            row["image"] = image
            with self.subTest(image=image), self.assertRaisesRegex(
                    TypeError, "image must be a string"):
                NORMALIZER.normalize_row(row, self.root, strict=False)

    def test_empty_image_fails(self):
        row = self.row("")
        row["image"] = ""
        with self.assertRaisesRegex(ValueError, "image must be non-empty"):
            NORMALIZER.normalize_row(row, self.root, strict=False)

    def test_string_image_is_preserved(self):
        row = self.row("")
        normalized = NORMALIZER.normalize_row(row, self.root, strict=False)
        self.assertEqual(normalized["image_rel"], self.image_path.name)

    def test_missing_query_fails_regardless_of_strict(self):
        row = self.row("")
        del row["query"]
        for strict in (False, True):
            with self.subTest(strict=strict), self.assertRaisesRegex(
                    ValueError, "missing query"):
                NORMALIZER.normalize_row(row, self.root, strict=strict)

    def test_non_string_query_fails_regardless_of_strict(self):
        for query in (None, 7, {}, []):
            for strict in (False, True):
                row = self.row("")
                row["query"] = query
                with self.subTest(query=query, strict=strict), self.assertRaisesRegex(
                        TypeError, "query must be a string"):
                    NORMALIZER.normalize_row(row, self.root, strict=strict)

    def test_missing_stats_fails_regardless_of_strict(self):
        row = self.row("")
        del row["stats"]
        for strict in (False, True):
            with self.subTest(strict=strict), self.assertRaisesRegex(
                    ValueError, "missing stats"):
                NORMALIZER.normalize_row(row, self.root, strict=strict)

    def test_non_object_stats_fails_regardless_of_strict(self):
        for stats in (None, 7, "", []):
            for strict in (False, True):
                row = self.row("")
                row["stats"] = stats
                with self.subTest(stats=stats, strict=strict), self.assertRaisesRegex(
                        TypeError, "stats must be an object"):
                    NORMALIZER.normalize_row(row, self.root, strict=strict)

    def test_empty_query_and_stats_are_accepted(self):
        raw_response = "<ref>C3 vertebra</ref><box><100><200><600><800></box>"
        row = self.row(raw_response)
        row["query"] = ""
        row["stats"] = {}
        normalized = NORMALIZER.normalize_row(row, self.root, strict=True)
        self.assertEqual(normalized["query"], "")
        self.assertNotIn("stats", normalized)
        self.assertEqual(len(normalized["instances"]), 1)

    def test_missing_raw_response_fails_even_when_not_strict(self):
        row = self.row("")
        del row["raw_response"]
        row["pred"] = "<ref>C1 vertebra</ref><box><1><1><9><9></box>"
        with self.assertRaisesRegex(ValueError, "missing raw_response"):
            NORMALIZER.normalize_row(row, self.root, strict=False)

    def test_non_string_raw_response_fails_even_when_not_strict(self):
        for raw_response in (None, 7, {}, []):
            with self.subTest(raw_response=raw_response), self.assertRaisesRegex(
                    TypeError, "raw_response must be a string"):
                NORMALIZER.normalize_row(
                    self.row(raw_response), self.root, strict=False
                )

    def test_explicit_empty_raw_response_obeys_only_strict_zero_box_gate(self):
        normalized = NORMALIZER.normalize_row(
            self.row(""), self.root, strict=False
        )
        self.assertEqual(normalized["raw_response"], "")
        self.assertEqual(normalized["instances"], [])
        with self.assertRaisesRegex(ValueError, "no valid box"):
            NORMALIZER.normalize_row(self.row(""), self.root, strict=True)

    def test_real_raw_response_is_normalized_to_canonical_instance(self):
        raw_response = "<ref>C3 vertebra</ref><box><100><200><600><800></box>"
        normalized = NORMALIZER.normalize_row(
            self.row(raw_response), self.root, strict=True
        )
        self.assertEqual(normalized["query"], "Locate each vertebra")
        self.assertEqual(normalized["raw_response"], raw_response)
        self.assertEqual(normalized["instances"], [{
            "bbox": [20.0, 20.0, 120.0, 80.0],
            "score": 1.0,
            "label": "C3 vertebra",
            "class_id": 2,
            "source": "LocateAnything",
        }])


class LocateAnythingCacheTests(unittest.TestCase):
    def test_parses_ref_boxes_from_normalized_tokens(self):
        parsed = parse_raw_response(
            "<ref>class_03</ref><box><100><200><600><800></box>",
            image_width=200,
            image_height=100,
        )
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].class_id, 2)
        self.assertEqual(parsed[0].bbox, (20.0, 20.0, 120.0, 80.0))

    def test_maps_vertebra_names_to_zero_based_detection_classes(self):
        text = (
            "<ref>C1 vertebra</ref><box><1><1><10><10></box>"
            "<ref>C7 vertebra</ref><box><11><11><20><20></box>"
            "<ref>T12 vertebra</ref><box><21><21><30><30></box>"
            "<ref>L3 vertebra</ref><box><31><31><40><40></box>"
            "<ref>L6 vertebra</ref><box><41><41><50><50></box>"
        )
        parsed = parse_raw_response(text, image_width=100, image_height=100)
        self.assertEqual([item.class_id for item in parsed], [0, 6, 18, 21, 24])

    def test_bare_box_fails_closed_without_absolute_class(self):
        parsed = parse_raw_response(
            "result <box>100, 200, 600, 800</box>",
            image_width=200,
            image_height=100,
            default_class_id=7,
        )
        self.assertEqual(parsed, [])

    def test_generic_vertebra_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported detection label"):
            parse_raw_response(
                "<ref>vertebra</ref><box><100><200><600><800></box>",
                image_width=200,
                image_height=100,
            )

    def test_structured_jsonl_is_loaded_by_image_alias(self):
        row = {
            "image": "/data/case_1_image.png",
            "width": 100,
            "height": 80,
            "instances": [{"bbox": [1, 2, 30, 40], "score": 0.8, "class_id": 2}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "cache.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            cache = LocateAnythingCache.from_path(str(path))
        detections = cache.lookup("case_1_image.png")
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_id, 2)

    def test_missing_policy_can_return_empty(self):
        cache = LocateAnythingCache([{
            "image": "known.png",
            "width": 10,
            "height": 10,
            "instances": [],
        }])
        self.assertEqual(cache.lookup("unknown.png", missing="empty"), ())
        with self.assertRaises(KeyError):
            cache.lookup("unknown.png", missing="error")

    def test_absolute_lookup_does_not_fall_back_to_basename_or_stem(self):
        cache = LocateAnythingCache([{
            "image": "/a/x.png",
            "width": 10,
            "height": 10,
            "instances": [{"bbox": [1, 1, 9, 9], "class_id": 0}],
        }])
        self.assertEqual(cache.lookup("/b/x.png", missing="empty"), ())
        with self.assertRaises(KeyError):
            cache.lookup("/b/x.png", missing="error")

    def test_explicit_empty_instances_are_authoritative(self):
        cache = LocateAnythingCache([{
            "image": "known.png",
            "width": 10,
            "height": 10,
            "instances": [],
            "raw_response": "<ref>C1 vertebra</ref><box><1><1><9><9></box>",
        }])
        self.assertEqual(cache.lookup("known.png"), ())

    def test_legacy_one_based_label_id_becomes_zero_based_class(self):
        cache = LocateAnythingCache([{
            "image": "known.png",
            "width": 10,
            "height": 10,
            "instances": [{"bbox": [1, 1, 9, 9], "label_id": 25}],
        }])
        self.assertEqual(cache.lookup("known.png")[0].class_id, 24)

    def test_real_mixed_legacy_schema_uses_one_based_class_label(self):
        cache = LocateAnythingCache([{
            "image": "known.png",
            "width": 10,
            "height": 10,
            "instances": [{
                "bbox": [1, 1, 9, 9],
                "label": "class_01",
                "label_id": 1,
                "cls_id": 1,
                "category_id": 1,
            }],
        }])
        self.assertEqual(cache.lookup("known.png")[0].class_id, 0)

    def test_canonical_class_id_range_is_enforced_at_load(self):
        with self.assertRaises(ValueError):
            LocateAnythingCache([{
                "image": "known.png",
                "width": 10,
                "height": 10,
                "instances": [{"bbox": [1, 1, 9, 9], "class_id": 25}],
            }])

    def test_canonical_class_id_accepts_integer_types_and_integer_strings(self):
        for value in (2, np.int64(2), "2", " +2 "):
            with self.subTest(value=value):
                cache = LocateAnythingCache([{
                    "image": "known.png",
                    "width": 10,
                    "height": 10,
                    "instances": [{"bbox": [1, 1, 9, 9], "class_id": value}],
                }])
                self.assertEqual(cache.lookup("known.png")[0].class_id, 2)

    def test_canonical_class_id_rejects_non_integer_nonfinite_and_out_of_range(self):
        for value in (2.9, float("nan"), float("inf"), -1, 25):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LocateAnythingCache([{
                    "image": "known.png",
                    "width": 10,
                    "height": 10,
                    "instances": [{"bbox": [1, 1, 9, 9], "class_id": value}],
                }])

    def test_legacy_label_id_rejects_fractional_and_nonfinite_values(self):
        for value in (2.9, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LocateAnythingCache([{
                    "image": "known.png",
                    "width": 10,
                    "height": 10,
                    "instances": [{"bbox": [1, 1, 9, 9], "label_id": value}],
                }])

    def test_bbox_coordinates_must_all_be_finite(self):
        for index in range(4):
            for invalid in (float("nan"), float("inf")):
                bbox = [1.0, 1.0, 9.0, 9.0]
                bbox[index] = invalid
                with self.subTest(index=index, invalid=invalid), self.assertRaises(ValueError):
                    LocateAnythingCache([{
                        "image": "known.png",
                        "width": 10,
                        "height": 10,
                        "instances": [{"bbox": bbox, "class_id": 0}],
                    }])

    def test_score_must_be_finite(self):
        for score in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(score=score), self.assertRaises(ValueError):
                LocateAnythingCache([{
                    "image": "known.png",
                    "width": 10,
                    "height": 10,
                    "instances": [{"bbox": [1, 1, 9, 9], "score": score, "class_id": 0}],
                }])

    def test_duplicate_alias_is_rejected_even_when_content_matches(self):
        records = [
            {"image": "/a/shared.png", "width": 10, "height": 10, "instances": []},
            {"image": "/b/shared.png", "width": 10, "height": 10, "instances": []},
        ]
        with self.assertRaises(ValueError):
            LocateAnythingCache(records)

    def test_class_aware_nms_keeps_different_classes(self):
        detections = [
            CachedDetection((0, 0, 10, 10), 0.9, 1),
            CachedDetection((1, 1, 9, 9), 0.8, 1),
            CachedDetection((1, 1, 9, 9), 0.7, 2),
        ]
        kept = filter_detections(detections, DetectionPolicy(nms_iou=0.5))
        self.assertEqual([(item.score, item.class_id) for item in kept], [(0.9, 1), (0.7, 2)])

    def test_quality_gates_remove_tiny_and_low_score_boxes(self):
        detections = [
            CachedDetection((0, 0, 1, 1), 0.9, 1),
            CachedDetection((0, 0, 10, 10), 0.1, 1),
            CachedDetection((0, 0, 10, 10), 0.9, 1),
        ]
        kept = filter_detections(
            detections,
            DetectionPolicy(min_score=0.5, min_box_side=2, min_box_area=4),
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].score, 0.9)

    def test_policy_requires_finite_float_parameters(self):
        names = ("min_score", "min_box_side", "min_box_area", "nms_iou")
        for name in names:
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(
                        ValueError, "finite"):
                    DetectionPolicy(**{name: value}).validate()

    def test_policy_requires_strict_non_negative_integer_max_detections(self):
        for value in (True, False, 2.5, float("nan"), float("inf"), "2", -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DetectionPolicy(max_detections=value).validate()
        for value in (0, 2, np.int64(2)):
            with self.subTest(value=value):
                DetectionPolicy(max_detections=value).validate()

    def test_cache_score_sentinel_precedes_configured_minimum(self):
        detections = [
            CachedDetection((0, 0, 10, 10), 0.0, 0),
            CachedDetection((20, 0, 30, 10), 1e-4, 1),
            CachedDetection((40, 0, 50, 10), np.nextafter(1e-4, 1.0), 2),
        ]
        kept = filter_detections(detections, DetectionPolicy(min_score=0.0))
        self.assertEqual([(item.score, item.class_id) for item in kept], [
            (np.nextafter(1e-4, 1.0), 2),
        ])

    def test_affine_transform_uses_all_four_corners(self):
        detection = CachedDetection((1, 2, 5, 6), 0.9, 1)
        transform = np.asarray([[0, -1, 10], [1, 0, 0]], dtype=np.float32)
        mapped = transform_detection(detection, transform, 20, 20)
        self.assertIsNotNone(mapped)
        self.assertEqual(mapped.bbox, (4.0, 1.0, 8.0, 5.0))

    def test_builds_padded_batched_detection_contract(self):
        cache = LocateAnythingCache([
            {
                "image": "/data/a.png",
                "width": 100,
                "height": 100,
                "instances": [{"bbox": [10, 20, 30, 40], "score": 0.9, "class_id": 2}],
            },
            {
                "image": "/data/b.png",
                "width": 100,
                "height": 100,
                "instances": [],
            },
        ])
        batch = {
            "inp": torch.zeros(2, 3, 64, 64),
            "img_path": ["/data/a.png", "/data/b.png"],
            "meta": {
                "trans_input": torch.tensor([
                    [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
                    [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
                ])
            },
        }
        detection = build_detection_tensor(cache, batch, DetectionPolicy())
        self.assertEqual(tuple(detection.shape), (2, 1, 6))
        self.assertTrue(torch.allclose(detection[0, 0], torch.tensor([5, 10, 15, 20, 0.9, 2])))
        self.assertTrue(torch.equal(detection[1, 0], torch.zeros(6)))

    def test_structured_low_scores_do_not_enter_dense_padding_contract(self):
        cache = LocateAnythingCache([{
            "image": "/data/a.png",
            "width": 100,
            "height": 100,
            "instances": [
                {"bbox": [0, 0, 10, 10], "score": 0.0, "class_id": 0},
                {"bbox": [20, 0, 30, 10], "score": 1e-4, "class_id": 1},
                {"bbox": [40, 0, 50, 10], "score": 1.0001e-4, "class_id": 2},
            ],
        }])
        batch = {
            "inp": torch.zeros(1, 3, 100, 100),
            "img_path": ["/data/a.png"],
            "meta": {
                "trans_input": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
            },
        }
        detection = build_detection_tensor(
            cache, batch, DetectionPolicy(min_score=0.0)
        )
        self.assertEqual(tuple(detection.shape), (1, 1, 6))
        self.assertAlmostEqual(float(detection[0, 0, 4]), 1.0001e-4)
        self.assertEqual(float(detection[0, 0, 5]), 2.0)

    def test_flip_detection_preserves_full_width_continuous_box(self):
        detection = CachedDetection((0.0, 2.0, 100.0, 8.0), 0.9, 1)
        flipped = flip_detection(detection, original_width=100)
        self.assertEqual(flipped.bbox, (0.0, 2.0, 100.0, 8.0))

    def test_flipped_sample_mirrors_box_before_affine_transform(self):
        cache = LocateAnythingCache([{
            "image": "/data/a.png",
            "width": 100,
            "height": 80,
            "instances": [{"bbox": [10, 20, 30, 40], "class_id": 0}],
        }])
        batch = {
            "inp": torch.zeros(1, 3, 80, 100),
            "img_path": ["/data/a.png"],
            "meta": {
                "trans_input": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
                "flipped": torch.tensor([[1.0]]),
                "orig_hw": torch.tensor([[80.0, 100.0]]),
            },
        }
        detection = build_detection_tensor(cache, batch, DetectionPolicy())
        self.assertTrue(torch.equal(
            detection[0, 0],
            torch.tensor([70.0, 20.0, 90.0, 40.0, 1.0, 0.0]),
        ))


if __name__ == "__main__":
    unittest.main()
