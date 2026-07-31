#!/usr/bin/env bash
set +e

cd /home/medteam/Zhrch/DiffusionSnake-12-30
source /home/medteam/miniconda3/etc/profile.d/conda.sh
conda activate snake1
echo "[SH] conda activated: $(which python)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export CFG_FILE="configs/1232_final_v5_geom8_baseline_bs6_gpu2.yaml"

export RL_V4_CREDIT_DIAG=1
export RL_V4_CREDIT_DIAG_STOP=1
export RL_V4_STEPS=20

python -u grpo_train_v5_geom_action.py --cfg_file "$CFG_FILE"