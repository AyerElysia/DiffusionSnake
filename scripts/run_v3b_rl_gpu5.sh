#!/usr/bin/env bash
set -eo pipefail

source /home/medteam/miniconda3/etc/profile.d/conda.sh
conda activate snake1
cd /home/medteam/Zhrch/DiffusionSnake-12-30

export CFG_FILE=/home/medteam/Zhrch/DiffusionSnake-12-30/configs/btcv_v4_6c_rl_v3b_midstate_structured_gpu5.yaml
export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1

/home/medteam/miniconda3/envs/snake1/bin/python -u grpo_train_v2.py
