#!/usr/bin/env bash
set -e
cd /home/medteam/Zhrch/DiffusionSnake-12-30
PY=/home/medteam/miniconda3/envs/snake1/bin/python

echo "=== V4.1 eval ==="
CFG_FILE=configs/btcv_diffusion_dit_v4_1_fm_detail_curv_gpu4.yaml \
EVAL_GPU=3 CUDA_VISIBLE_DEVICES=3 \
SAVE_DIR=visual/v4_1_fm_eval \
$PY scripts/eval_v37_full_iou.py
echo "V4.1 done"

echo "=== V4.2 eval ==="
CFG_FILE=configs/btcv_diffusion_dit_v4_2_fm_yolom_gpu5.yaml \
EVAL_GPU=3 CUDA_VISIBLE_DEVICES=3 \
SAVE_DIR=visual/v4_2_fm_eval \
$PY scripts/eval_v37_full_iou.py
echo "V4.2 done"

echo "=== V4.1 visualize ==="
for idx in 0 10 25; do
  CFG_FILE=configs/btcv_diffusion_dit_v4_1_fm_detail_curv_gpu4.yaml \
  EVAL_GPU=3 CUDA_VISIBLE_DEVICES=3 \
  $PY scripts/infer_single_sample.py --index $idx --tag v4_1 --out_dir visual/v4_1_fm_eval
done

echo "=== V4.2 visualize ==="
for idx in 0 10 25; do
  CFG_FILE=configs/btcv_diffusion_dit_v4_2_fm_yolom_gpu5.yaml \
  EVAL_GPU=3 CUDA_VISIBLE_DEVICES=3 \
  $PY scripts/infer_single_sample.py --index $idx --tag v4_2 --out_dir visual/v4_2_fm_eval
done

echo "ALL DONE"
