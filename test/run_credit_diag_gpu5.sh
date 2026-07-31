#!/usr/bin/env bash
# Credit-diag baseline: terminal-only advantage (per_step_reward_weight=0.0)
# GPU 5, output: data/outputs/credit_diag_baseline_gpu5/
set +e
cd /home/medteam/Zhrch/DiffusionSnake-12-30
source /home/medteam/miniconda3/etc/profile.d/conda.sh
conda activate snake1
echo "[SH] python: $(which python)"

export CUDA_VISIBLE_DEVICES=5
export CFG_FILE="configs/credit_diag_baseline_gpu5.yaml"
export RL_V4_CREDIT_DIAG=1
export RL_V4_CREDIT_DIAG_STOP=1
export RL_V4_STEPS=20

python -u grpo_train_v5_geom_action.py --cfg_file "$CFG_FILE"
