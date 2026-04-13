#!/bin/bash
export CFG_FILE=configs/btcv_diffusion_dit_v3.yaml
python3 infer_v3_refinement.py "$@"
