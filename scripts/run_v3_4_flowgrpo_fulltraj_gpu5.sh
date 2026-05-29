#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate snake1

export CFG_FILE=configs/btcv_v3_4_fm_rl_v16_flowgrpo_fulltraj_gpu5.yaml
export CUDA_VISIBLE_DEVICES=5

python grpo_train_v2.py --cfg_file "$CFG_FILE"
