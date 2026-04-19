#!/bin/bash

set -euo pipefail

GPU="${1:-0}"
ROOT="/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30"
CFG="configs/btcv_diffusion_dit_v3_9_single_overfit.yaml"
LOG="logs/v3_9_single_overfit_gpu${GPU}.log"
SESSION="v3_9_gpu${GPU}"

cd "${ROOT}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate snake1

mkdir -p logs

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" \
    "source ~/miniconda3/etc/profile.d/conda.sh; conda activate snake1; cd ${ROOT}; CUDA_VISIBLE_DEVICES=${GPU} python -u diffusion_train.py --cfg_file ${CFG} gpus '[${GPU}]' > ${LOG} 2>&1"

echo "Started V3.9 single-sample training on GPU ${GPU}"
echo "tmux session: ${SESSION}"
echo "log file: ${LOG}"
