#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30
source /home/medteam/miniconda3/etc/profile.d/conda.sh
conda activate snake1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CFG_FILE="configs/1232_final_p5a_point_explorer_gpu0.yaml"

python grpo_train_v5_geom_action.py --cfg_file "$CFG_FILE"
