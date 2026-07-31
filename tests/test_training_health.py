import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

import torch

from lib.recorder.json_logger import JsonLogger


_ROOT = Path(__file__).parents[1]
_MONITOR_PATH = _ROOT / "scripts" / "monitor_sagittal_moonvit_train.py"
_SPEC = importlib.util.spec_from_file_location("sagittal_health_monitor", str(_MONITOR_PATH))
_MONITOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MONITOR)


def _checkpoint_payload(nonfinite=False):
    adapter_weight = torch.tensor([float("nan") if nonfinite else 1.0])
    return {
        "format_version": 2,
        "run_id": "run-test",
        "cfg_hash": "cfg-test",
        "saved_at": "2026-07-21T00:00:00.000Z",
        "epoch": 4,
        "step": 500,
        "step_in_epoch": 4,
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        },
        "state_dict": {"locate_feat_adapter.proj.0.weight": adapter_weight},
        "optimizer": {"state": {}, "param_groups": []},
        "scheduler": None,
    }


class TrainingHealthTest(unittest.TestCase):
    def test_json_logger_rejects_nan_and_propagates_flush_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = JsonLogger(str(path))
            with self.assertRaises(ValueError):
                logger.log({"value": float("nan")})
            logger.close()

        class BrokenFile:
            def write(self, value):
                return len(value)

            def flush(self):
                raise OSError("flush failed")

        broken = JsonLogger.__new__(JsonLogger)
        broken._f = BrokenFile()
        with self.assertRaisesRegex(OSError, "flush failed"):
            broken.log({"value": 1.0})

    def test_monitor_thresholds_cover_required_health_failures(self):
        rows = [
            {
                "step": step,
                "timestamp": "1970-01-01T00:00:00.000Z",
                "time_ms": 10.0,
                "foreground_count": 1,
                "foreground_ratio": 0.9,
                "adapter_grad_l2": 0.0,
                "adapter_update_l2": 0.0,
            }
            for step in range(1, 501)
        ]
        alerts = _MONITOR.evaluate_health(
            rows,
            now=1000.0,
            gpu_records=[
                {
                    "index": "0",
                    "temperature_c": 85.0,
                    "memory_total_mib": 100000.0,
                    "memory_used_mib": 96000.0,
                    "memory_free_mib": 1000.0,
                }
            ],
            disk_free=40.0,
            checkpoint_step=-1,
        )
        codes = {item["code"] for item in alerts}
        self.assertTrue({
            "stall",
            "gpu_temperature",
            "gpu_memory_ratio",
            "gpu_memory_free",
            "disk_free",
            "foreground_ratio",
            "adapter_zero_gradient",
            "adapter_zero_update",
            "checkpoint_lag",
        }.issubset(codes))

    def test_monitor_detects_200_batches_without_foreground(self):
        rows = [
            {
                "step": index + 1,
                "timestamp": "1970-01-01T00:16:39.900Z",
                "time_ms": 1.0,
                "foreground_count": 0,
                "foreground_ratio": 0.0,
                "adapter_grad_l2": 1.0,
                "adapter_update_l2": 1.0,
            }
            for index in range(200)
        ]
        alerts = _MONITOR.evaluate_health(
            rows,
            now=1000.0,
            no_foreground_batches=200,
            adapter_zero_window=200,
            disk_free=100.0,
        )
        self.assertTrue(any(item["code"] == "no_foreground" for item in alerts))

    def test_monitor_missing_health_fields_do_not_create_zero_alerts(self):
        rows = [
            {
                "step": index + 1,
                "timestamp": "1970-01-01T00:16:39.900Z",
                "time_ms": 1.0,
            }
            for index in range(200)
        ]
        alerts = _MONITOR.evaluate_health(
            rows,
            now=1000.0,
            checkpoint_step=200,
            no_foreground_batches=200,
            adapter_zero_window=200,
            disk_free=100.0,
        )
        codes = {item["code"] for item in alerts}
        self.assertFalse({
            "no_foreground",
            "adapter_zero_gradient",
            "adapter_zero_update",
        } & codes)

    def test_monitor_jsonl_selects_only_the_current_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logs.jsonl"
            path.write_text(
                '{"run_id":"old","step":1,"timestamp":"1970-01-01T00:00:00.000Z"}\n'
                '{"step":2,"timestamp":"1970-01-01T00:16:40.000Z"}\n'
                '{"run_id":"new","step":3,"timestamp":"1970-01-01T00:16:41.000Z"}\n',
                encoding="utf-8",
            )
            self.assertEqual(_MONITOR.latest_run_id(path, min_timestamp=1000.0), "new")
            rows, status = _MONITOR.read_jsonl(path, run_id="new")
            self.assertTrue(status["exists"])
            self.assertEqual([item["step"] for item in rows], [3])

    def test_cache_precheck_uses_manifest_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "cache"
            cache_root.mkdir()
            (cache_root / "one.npz").write_bytes(b"not-empty")
            manifest = root / "manifest.csv"
            manifest.write_text(
                "split,case_id,slice_idx,image_path,mask_path\n"
                "train,c,0,i,m\n",
                encoding="utf-8",
            )
            result = _MONITOR.precheck_cache(cache_root, manifest_path=manifest)
            self.assertTrue(result["ok"])
            self.assertEqual(result["npz_count"], 1)
            self.assertEqual(result["expected_count"], 1)

    def test_checkpoint_cpu_validation_and_atomic_validated_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest = root / "latest.pt"
            validated = root / "validated_latest.pt"
            torch.save(_checkpoint_payload(), str(latest))

            metadata = _MONITOR.validate_and_update_latest(
                latest,
                validated,
                expected_run_id="run-test",
                expected_cfg_hash="cfg-test",
            )
            self.assertEqual(metadata["step"], 500)
            self.assertTrue(validated.exists())
            loaded = torch.load(str(validated), map_location="cpu")
            self.assertEqual(loaded["run_id"], "run-test")
            self.assertFalse(list(root.glob(".*.tmp.*")))

            torch.save(_checkpoint_payload(nonfinite=True), str(latest))
            with self.assertRaisesRegex(ValueError, "non-finite tensor"):
                _MONITOR.validate_checkpoint_cpu(latest)

    def test_diffusion_training_health_contract_is_present(self):
        source = (_ROOT / "diffusion_train.py").read_text(encoding="utf-8")
        self.assertIn("gradient_accumulation_steps must be exactly 1", source)
        self.assertIn("clip_grad_norm_", source)
        self.assertIn("error_if_nonfinite=True", source)
        self.assertNotIn("clip_grad_value_", source)
        for field in (
            "run_id",
            "foreground_count",
            "foreground_ratio",
            "contour_count",
            "moonvit_adapter_grad_l2",
            "moonvit_adapter_update_l2",
            "lr_min",
            "scheduler_last_epoch",
            "cuda_allocated_bytes",
            "cuda_reserved_bytes",
            "cuda_max_allocated_bytes",
        ):
            self.assertIn(field, source)
        logger_source = (_ROOT / "lib" / "recorder" / "json_logger.py").read_text(encoding="utf-8")
        self.assertIn("allow_nan=False", logger_source)


if __name__ == "__main__":
    unittest.main()
