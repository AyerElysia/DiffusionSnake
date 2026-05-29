#!/usr/bin/env bash
set -eo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30
source /home/medteam/miniconda3/etc/profile.d/conda.sh
conda activate snake1

export CFG_FILE=configs/btcv_select_v4_6c_rl_v4c_three_iter_spike_gpu2.yaml
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export RL_V4_GPU="${RL_V4_GPU:-${CUDA_VISIBLE_DEVICES%%,*}}"

python grpo_train_v4c_three_iter_spike.py --cfg_file "$CFG_FILE"
