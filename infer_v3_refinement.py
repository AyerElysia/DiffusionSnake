#!/usr/bin/env python3
import os
import runpy
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CFG = ROOT_DIR / "configs" / "btcv_diffusion_dit_v3.yaml"


if not os.environ.get("CFG_FILE"):
    os.environ["CFG_FILE"] = str(DEFAULT_CFG)


if __name__ == "__main__":
    runpy.run_path(str(ROOT_DIR / "scripts" / "infer_v3_final.py"), run_name="__main__")
