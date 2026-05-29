#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate snake1

export CFG_FILE=configs/btcv_select_v4_6c_rl_v4d_structured_sumlogprob_gpu0.yaml
export CUDA_VISIBLE_DEVICES=0

python grpo_train_v4d_point_noise_policy.py --cfg_file "$CFG_FILE"
