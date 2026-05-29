#!/usr/bin/env bash
set -eo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30
source /home/medteam/miniconda3/etc/profile.d/conda.sh
conda activate snake1

export CFG_FILE=configs/btcv_select_v4_6c_rl_v5b_geom_action_small_gpu7.yaml
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export RL_V4_GPU="${RL_V4_GPU:-${CUDA_VISIBLE_DEVICES%%,*}}"

python grpo_train_v5_geom_action.py --cfg_file "$CFG_FILE"
