#!/bin/bash
# 全量评估脚本 - 对比 geom 路线四个变体
# GPU4 跑 baseline + full_extrap w=0.5
# GPU7 跑 full_extrap w=1.0 + seq_delta
set -e
PY=/home/medteam/miniconda3/envs/snake1/bin/python
LOGDIR=/home/medteam/Zhrch/DiffusionSnake-12-30/NOHUP_LOGS
mkdir -p $LOGDIR

echo "========================================================"
echo "[EVAL] geom baseline (GPU4)"
echo "========================================================"
CUDA_VISIBLE_DEVICES=4 $PY run.py \
    --type test_medical \
    --cfg_file configs/1232_final_v5_geom8_perstep00_bs6_gpu5.yaml \
    test.visual_save_root data/eval_vis/geom_baseline \
    test.epoch -1 \
    2>&1 | tee $LOGDIR/eval_geom_baseline.log

echo "========================================================"
echo "[EVAL] geom + full_extrap w=0.5 (GPU4)"
echo "========================================================"
CUDA_VISIBLE_DEVICES=4 $PY run.py \
    --type test_medical \
    --cfg_file configs/1232_final_v5_geom8_perstep05_extrap_bs6_gpu0.yaml \
    test.visual_save_root data/eval_vis/geom_extrap05 \
    test.epoch -1 \
    2>&1 | tee $LOGDIR/eval_geom_extrap05.log

echo "========================================================"
echo "[EVAL] geom + full_extrap w=1.0 (GPU4)"
echo "========================================================"
CUDA_VISIBLE_DEVICES=4 $PY run.py \
    --type test_medical \
    --cfg_file configs/1232_final_v5_geom8_extrap1p0_gpu6.yaml \
    test.visual_save_root data/eval_vis/geom_extrap1p0 \
    test.epoch -1 \
    2>&1 | tee $LOGDIR/eval_geom_extrap1p0.log

echo "========================================================"
echo "[EVAL] geom + seq_delta (GPU4)"
echo "========================================================"
CUDA_VISIBLE_DEVICES=4 $PY run.py \
    --type test_medical \
    --cfg_file configs/1232_final_v5_geom8_seqdelta_gpu1.yaml \
    test.visual_save_root data/eval_vis/geom_seqdelta \
    test.epoch -1 \
    2>&1 | tee $LOGDIR/eval_geom_seqdelta.log

echo "========================================================"
echo "[EVAL] All done. 汇总结果："
grep -h "Overall mIoU\|Overall mDice\|mBoundF\|加载权重" \
    $LOGDIR/eval_geom_baseline.log \
    $LOGDIR/eval_geom_extrap05.log \
    $LOGDIR/eval_geom_extrap1p0.log \
    $LOGDIR/eval_geom_seqdelta.log 2>/dev/null || true
echo "========================================================"
