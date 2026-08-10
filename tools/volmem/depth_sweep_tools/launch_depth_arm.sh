#!/bin/bash
# launch_depth_arm.sh <arm_config_name> <gpu_id>
# Launches one depth-sweep arm. Identical protocol across arms; only the
# config's dit_num_layers differs.
set -euo pipefail

WT=/home/medteam/Zhrch/DiffusionSnake-12-30-depth-sweep-20260809
NAME="$1"
GPU="$2"

cd "$WT"

CFG="configs/volmem/depth_sweep/${NAME}.yaml"
if [ ! -f "$CFG" ]; then
    echo "MISSING CFG: $CFG"
    exit 1
fi

OUT="$WT/data/outputs/depth_sweep/${NAME}"
if [ -f "$OUT/train.log" ]; then
    echo "REFUSE: $OUT/train.log already exists (would clobber)"
    exit 1
fi
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES="$GPU" nohup python tools/volmem/train_memflowdit.py \
    --cfg_file "$CFG" \
    --init-memflow-ckpt data/model/volmem_frozen/h1_distilled_full.pt \
    --seed 20260731 \
    --max_steps 2000 \
    --save_every 250 \
    > "$OUT/train.log" 2>&1 &

echo "launched ${NAME} gpu=${GPU} pid=$!"
