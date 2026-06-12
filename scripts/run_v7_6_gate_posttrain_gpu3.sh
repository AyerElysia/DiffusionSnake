#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate snake1
set -u

CFG=${CFG:-configs/1232_final_v7_6_conditional_residual_explorer_long_gpu3.yaml}
CKPT=${CKPT:-data/outputs/1232_final_v7_6_conditional_residual_explorer_long_gpu3/checkpoints/best_iou.pt}
OUT_DIR=${OUT_DIR:-data/outputs/v7_6_disp_gate_head_gpu3}
GPU=${GPU:-3}
STEPS=${STEPS:-1000}

python scripts/train_supervised_posttrain.py \
  --cfg-file "$CFG" \
  --ckpt "$CKPT" \
  --out-dir "$OUT_DIR" \
  --mode gate \
  --gpu "$GPU" \
  --steps "$STEPS" \
  --lr "${LR:-3e-5}" \
  --gate-loss-weight "${GATE_LOSS_WEIGHT:-1.0}" \
  --save-every "${SAVE_EVERY:-100}"
