#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate snake1

export CFG_FILE=configs/btcv_v3_4_fm_rl_v17_mid3_stepgrpo_gpu2.yaml
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export GRPO_V2_GPU="${GRPO_V2_GPU:-${CUDA_VISIBLE_DEVICES%%,*}}"
export PYTHONUNBUFFERED=1

python -u grpo_train_v2.py --cfg_file "$CFG_FILE"
