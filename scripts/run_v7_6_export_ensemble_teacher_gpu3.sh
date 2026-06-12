#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate snake1
set -u

CFG=${CFG:-configs/1232_final_v7_6_conditional_residual_explorer_long_gpu3.yaml}
CKPT=${CKPT:-data/outputs/1232_final_v7_6_conditional_residual_explorer_long_gpu3/checkpoints/best_iou.pt}
OUT=${OUT:-data/teachers/v7_6_train_noise05_seed8_teacher.json}
GPU=${GPU:-3}
SEEDS=${SEEDS:-101,202,303,404,505,606,707,808}
MAX_SAMPLES=${MAX_SAMPLES:--1}

python scripts/export_ensemble_teacher.py \
  --cfg-file "$CFG" \
  --ckpt "$CKPT" \
  --out "$OUT" \
  --dataset BtcvTrain \
  --seeds "$SEEDS" \
  --noise-scale 0.5 \
  --ode-steps 10 \
  --max-samples "$MAX_SAMPLES" \
  --gpu "$GPU"
