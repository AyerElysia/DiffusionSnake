#!/bin/bash
cd /home/medteam/Zhrch/DiffusionSnake-12-30
PY=/home/medteam/miniconda3/envs/snake1/bin/python
CK=data/outputs/abl_c_u4_single/checkpoints/epoch_11.pt
for i in $(seq 1 200); do [ -f "$CK" ] && break; sleep 15; done
[ -f "$CK" ] || { echo TIMEOUT; exit 1; }
sleep 30
OUT=data/outputs/abl_c_u4_single/eval_gt400_ep11
mkdir -p $OUT
env CUDA_VISIBLE_DEVICES=7 $PY tools/eval_sagittal_2d_fixed.py \
  --cfg_file configs/ablation/abl_c_u4_single.yaml --ckpt $CK \
  --box-mode gt --split val --max-slices 400 \
  --result-dir $OUT --device cuda > $OUT/eval.log 2>&1
echo "C ep11 eval done"
