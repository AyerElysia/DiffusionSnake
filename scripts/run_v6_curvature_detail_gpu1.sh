#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30
source /home/medteam/miniconda3/etc/profile.d/conda.sh
conda activate snake1

export CFG_FILE=/home/medteam/Zhrch/DiffusionSnake-12-30/configs/btcv_select_v4_6c_rl_v6_curvature_detail_gpu1.yaml
export CUDA_VISIBLE_DEVICES=1

python grpo_train_v5_geom_action.py --cfg_file "$CFG_FILE"
