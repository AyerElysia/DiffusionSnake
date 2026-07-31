import os
import pathlib
import sys
import unittest
from unittest import mock
from types import SimpleNamespace

import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_PATH = PROJECT_ROOT / "configs" / "1232_final_diffusion_dit_v4_6c_geom_bridge_ddp_resample_sched_4gpu.yaml"
os.environ.setdefault("CFG_FILE", str(CONFIG_PATH))
sys.argv = [sys.argv[0]]

from volmem.adapters.v4_6c import V46cContourAdapter, build_detection_provider
from volmem.detection import DetectionPolicy, LocateAnythingCache


class DummyNetwork(torch.nn.Module):
    def forward(self, inp, batch):
        return {"detection": batch.get("external_detection")}


class DummyLossWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = DummyNetwork()
        self.last_batch = None

    def forward(self, batch):
        self.last_batch = batch
        zero = batch["inp"].sum() * 0.0
        return {}, zero, {}, {}


class ExternalDetectionAdapterTests(unittest.TestCase):
    def setUp(self):
        self.cache = LocateAnythingCache([{
            "image": "/data/a.png",
            "width": 100,
            "height": 100,
            "instances": [{"bbox": [10, 20, 30, 40], "score": 0.9, "class_id": 2}],
        }])
        self.batch = {
            "inp": torch.zeros(1, 3, 64, 64),
            "img_path": ["/data/a.png"],
            "meta": {"trans_input": torch.tensor([[[0.5, 0, 0], [0, 0.5, 0]]])},
        }

    def test_prediction_injects_standard_external_detection(self):
        wrapper = DummyLossWrapper()
        adapter = V46cContourAdapter(wrapper, self.cache, DetectionPolicy())
        output = adapter.predict(self.batch)
        self.assertEqual(tuple(output["detection"].shape), (1, 1, 6))
        self.assertEqual(float(output["detection"][0, 0, 5]), 2.0)

    def test_cached_detection_training_fails_closed(self):
        wrapper = DummyLossWrapper()
        adapter = V46cContourAdapter(wrapper, self.cache, DetectionPolicy())
        with self.assertRaises(RuntimeError):
            adapter(self.batch)
        self.assertNotIn("external_detection", self.batch)
        self.assertIsNone(wrapper.last_batch)

    def test_disabled_cache_is_exact_legacy_batch_identity(self):
        wrapper = DummyLossWrapper()
        adapter = V46cContourAdapter(wrapper)
        adapter(self.batch)
        self.assertIs(wrapper.last_batch, self.batch)

    def test_detector_source_does_not_load_cache(self):
        cache, policy = build_detection_provider(SimpleNamespace(box_source="detector"))
        self.assertIsNone(cache)
        self.assertIsNone(policy)

    def test_network_layer_contains_training_fail_closed_contract(self):
        source_path = PROJECT_ROOT / "lib" / "networks" / "snake" / "ct_snake.py"
        source = source_path.read_text(encoding="utf-8")
        start = source.index("    def apply_external_detection(")
        end = source.index("\n    def forward(", start)
        method_source = source[start:end]
        self.assertIn("if self.training:", method_source)
        self.assertIn("external_detection is evaluation-only", method_source)

    def test_normalizer_writes_canonical_class_id(self):
        source_path = PROJECT_ROOT / "tools" / "volmem" / "normalize_locany_cache.py"
        source = source_path.read_text(encoding="utf-8")
        self.assertIn('"class_id": detection.class_id', source)
        self.assertNotIn('"label_id": detection.class_id', source)

    def test_cached_source_requires_path(self):
        with self.assertRaises(ValueError):
            build_detection_provider(SimpleNamespace(box_source="locany_cached", locany_cache_path=""))

    def test_provider_does_not_coerce_invalid_policy_values(self):
        invalid_values = (
            ("locany_min_score", "0.5", "min_score"),
            ("locany_min_box_side", "1", "min_box_side"),
            ("locany_min_box_area", "4", "min_box_area"),
            ("locany_nms_iou", "0.5", "nms_iou"),
            ("locany_max_detections", 2.5, "max_detections"),
        )
        for attribute, value, message in invalid_values:
            config = SimpleNamespace(
                box_source="locany_cached",
                locany_cache_path="cache.json",
                **{attribute: value},
            )
            with self.subTest(attribute=attribute), mock.patch(
                "volmem.adapters.v4_6c.LocateAnythingCache.from_path",
                return_value=self.cache,
            ), self.assertRaisesRegex(ValueError, message):
                build_detection_provider(config)


class PrepareTestingInitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = PROJECT_ROOT / "configs" / "1232_final_diffusion_dit_v4_6c_geom_bridge_ddp_resample_sched_4gpu.yaml"
        with mock.patch.dict(os.environ, {"CFG_FILE": str(config_path)}), mock.patch.object(
            sys, "argv", [sys.argv[0]],
        ):
            from lib.utils.snake import snake_gcn_utils
        cls.snake_gcn_utils = snake_gcn_utils

    def test_dense_padding_is_filtered_before_polygon_initialization(self):
        detection = torch.tensor([
            [
                [4.0, 8.0, 20.0, 24.0, 0.9, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1e-4, 0.0],
                [12.0, 16.0, 36.0, 48.0, 0.2, 2.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1e-4, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 1e-4, 0.0],
                [40.0, 44.0, 60.0, 72.0, 0.8, 3.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ], dtype=torch.float32)
        expected_box = detection[[0, 0, 2], [0, 2, 1], :4].unsqueeze(0)
        original_get_init = self.snake_gcn_utils.snake_decode.get_init

        with mock.patch.object(
            self.snake_gcn_utils.snake_decode,
            "get_init",
            wraps=original_get_init,
        ) as get_init:
            init = self.snake_gcn_utils.prepare_testing({"detection": detection})

        self.assertEqual(init["i_it_4py"].shape[0], 3)
        self.assertEqual(init["i_it_py"].shape[0], 3)
        self.assertEqual(init["ind"].tolist(), [0, 0, 2])
        self.assertEqual(init["py_ind"].tolist(), [0, 0, 2])
        self.assertTrue(torch.isfinite(init["i_it_4py"]).all().item())
        self.assertTrue(torch.isfinite(init["c_it_4py"]).all().item())
        self.assertTrue(torch.isfinite(init["i_it_py"]).all().item())
        get_init.assert_called_once()
        torch.testing.assert_close(get_init.call_args.args[0], expected_box)

    def test_all_padding_returns_shaped_empty_tensors_without_polygon_initialization(self):
        detection = torch.zeros((3, 3, 6), dtype=torch.float32)

        with mock.patch.object(self.snake_gcn_utils.snake_decode, "get_init") as get_init:
            init = self.snake_gcn_utils.prepare_testing({"detection": detection})

        init_poly_num = self.snake_gcn_utils.snake_config.init_poly_num
        poly_num = self.snake_gcn_utils.snake_config.poly_num
        self.assertEqual(tuple(init["i_it_4py"].shape), (0, init_poly_num, 2))
        self.assertEqual(tuple(init["c_it_4py"].shape), (0, init_poly_num, 2))
        self.assertEqual(tuple(init["i_it_py"].shape), (0, poly_num, 2))
        self.assertEqual(tuple(init["c_it_py"].shape), (0, poly_num, 2))
        self.assertEqual(tuple(init["ind"].shape), (0,))
        self.assertEqual(tuple(init["py_ind"].shape), (0,))
        self.assertEqual(init["i_it_4py"].dtype, detection.dtype)
        self.assertEqual(init["i_it_4py"].device, detection.device)
        self.assertEqual(init["ind"].dtype, torch.long)
        self.assertEqual(init["ind"].device, detection.device)
        get_init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
