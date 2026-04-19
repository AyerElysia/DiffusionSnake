#!/bin/bash
# V3.7 Anti-Burr single-sample overfit training on GPU 2

cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30

source ~/miniconda3/etc/profile.d/conda.sh
conda activate snake1

mkdir -p logs

echo "=== Starting V3.7 Anti-Burr Training on GPU 2 ==="
echo "Config: configs/btcv_diffusion_dit_v3_7_single_overfit.yaml"
echo "Log: logs/v3_7_single_overfit_gpu2.log"

export CFG_FILE=configs/btcv_diffusion_dit_v3_7_single_overfit.yaml
CUDA_VISIBLE_DEVICES=2 python -u diffusion_train.py > logs/v3_7_single_overfit_gpu2.log 2>&1

echo "=== V3.7 Training Complete ==="
