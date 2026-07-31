#!/usr/bin/env bash
# Serial follow-up eval: wait for the B ep7 eval to finish, then wait for
# B's epoch_11.pt and evaluate it too. Keeps exactly one eval on GPU0 so we
# never exceed 2 trainings + 1 eval (disk is the throughput bottleneck).
#
# B (dual-layer u2) converges slower than A0, so judging it on ep7 alone is
# unfair -- A0 has ep3/ep7/ep11 points and B needs the same treatment.
set -u
cd /home/medteam/Zhrch/DiffusionSnake-12-30

PY=/home/medteam/miniconda3/envs/snake1/bin/python
CFG=configs/ablation/abl_b_u2_dual.yaml
CKPT=data/outputs/abl_b_u2_dual/checkpoints/epoch_11.pt
OUT=data/outputs/abl_b_u2_dual/eval_gt400_ep11
EP7_SUMMARY=data/outputs/abl_b_u2_dual/eval_gt400_ep7/summary.json

# 1) wait for the ep7 eval to land its summary (up to 60 min)
for i in $(seq 1 180); do
  [ -f "$EP7_SUMMARY" ] && { echo "[eval-B11] ep7 eval done $(date +%H:%M:%S)"; break; }
  sleep 20
done
[ -f "$EP7_SUMMARY" ] || { echo "[eval-B11] ep7 eval never finished, abort"; exit 1; }

# 2) wait for epoch_11.pt (up to 60 min)
for i in $(seq 1 180); do
  [ -f "$CKPT" ] && { echo "[eval-B11] ckpt found $(date +%H:%M:%S)"; break; }
  sleep 20
done
[ -f "$CKPT" ] || { echo "[eval-B11] epoch_11.pt never appeared, abort"; exit 1; }

if [ -f "$OUT/summary.json" ]; then
  echo "[eval-B11] already evaluated, nothing to do"; exit 0
fi

sleep 20   # let the writer flush the checkpoint fully
mkdir -p "$OUT"
echo "[eval-B11] starting eval on GPU0 $(date +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=0 "$PY" tools/eval_sagittal_2d_fixed.py \
  --cfg_file "$CFG" --ckpt "$CKPT" \
  --box-mode gt --split val --max-slices 400 \
  --result-dir "$OUT" --device cuda > "$OUT/eval.log" 2>&1
echo "[eval-B11] exit=$? $(date +%H:%M:%S)"
