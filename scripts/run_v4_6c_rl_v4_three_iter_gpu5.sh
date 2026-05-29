#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate snake1

export CFG_FILE=configs/btcv_v4_6c_rl_v4_three_iter_gpu5.yaml
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export RL_V4_GPU="${RL_V4_GPU:-${CUDA_VISIBLE_DEVICES%%,*}}"

python grpo_train_v4_three_iter.py --cfg_file "$CFG_FILE"
