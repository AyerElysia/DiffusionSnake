#!/usr/bin/env bash
set +e
cd /home/medteam/Zhrch/DiffusionSnake-12-30
source /home/medteam/miniconda3/etc/profile.d/conda.sh
conda activate snake1
echo "[SH] python: $(which python)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export CFG_FILE="configs/1232_final_v5_flowgrpo_extrap_bs6_gpu4.yaml"
python -u grpo_train_v5_geom_action.py --cfg_file "$CFG_FILE"
