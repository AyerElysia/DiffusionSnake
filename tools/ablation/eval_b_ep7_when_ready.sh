#!/bin/bash
cd /home/medteam/Zhrch/DiffusionSnake-12-30
PY=/home/medteam/miniconda3/envs/snake1/bin/python
CK=data/outputs/abl_b_u2_dual/checkpoints/epoch_7.pt
echo "[eval-B] waiting for $CK"
for i in $(seq 1 200); do
  [ -f "$CK" ] && break
  sleep 20
done
if [ ! -f "$CK" ]; then echo "[eval-B] timeout, ckpt missing"; exit 1; fi
sleep 20
echo "[eval-B] ckpt found $(date +%H:%M:%S), starting eval on GPU0"
OUT=data/outputs/abl_b_u2_dual/eval_gt400_ep7
mkdir -p $OUT
CUDA_VISIBLE_DEVICES=0 $PY tools/eval_sagittal_2d_fixed.py \
  --cfg_file configs/ablation/abl_b_u2_dual.yaml \
  --ckpt $CK --box-mode gt --split val --max-slices 400 \
  --result-dir $OUT --device cuda > $OUT/eval.log 2>&1
echo "[eval-B] eval done $(date +%H:%M:%S) rc=$?"
$PY tools/ablation/score_dual.py --table b_u2_dual_ep7=$OUT/slices.json
