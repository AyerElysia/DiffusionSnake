#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate snake1
set -u

CFG=${CFG:-configs/1232_final_v7_6_conditional_residual_explorer_long_gpu3.yaml}
CKPT=${CKPT:-data/outputs/1232_final_v7_6_conditional_residual_explorer_long_gpu3/checkpoints/best_iou.pt}
TEACHER=${TEACHER:-data/teachers/v7_6_train_noise05_seed8_teacher.json}
OUT_DIR=${OUT_DIR:-data/outputs/v7_6_ensemble_distill_noise05_seed8_gpu3}
GPU=${GPU:-3}
STEPS=${STEPS:-2000}

python scripts/train_ensemble_distill.py \
  --cfg-file "$CFG" \
  --ckpt "$CKPT" \
  --teacher "$TEACHER" \
  --out-dir "$OUT_DIR" \
  --dataset BtcvTrain \
  --gpu "$GPU" \
  --steps "$STEPS" \
  --lr "${LR:-1e-8}" \
  --teacher-weight "${TEACHER_WEIGHT:-1.0}" \
  --gt-weight "${GT_WEIGHT:-0.25}" \
  --save-every "${SAVE_EVERY:-200}" \
  --log-every "${LOG_EVERY:-20}"
