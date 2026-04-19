#!/bin/bash
# 单独启动脚本 - GPU 5 上的 V3 训练任务

cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30
source ~/miniconda3/etc/profile.d/conda.sh
conda activate snake1

mkdir -p logs

# V3.1
echo "Starting V3.1 on GPU 5..."
export CFG_FILE=configs/btcv_diffusion_dit_v3_1_single_overfit.yaml
CUDA_VISIBLE_DEVICES=5 nohup python diffusion_train.py > logs/v3_1_single_overfit_gpu5.log 2>&1 &
echo "V3.1 started, PID: $!"

# V3.3a
echo "Starting V3.3a on GPU 5..."
export CFG_FILE=configs/btcv_diffusion_dit_v3_3a_single_overfit.yaml
CUDA_VISIBLE_DEVICES=5 nohup python diffusion_train.py > logs/v3_3a_single_overfit_gpu5.log 2>&1 &
echo "V3.3a started, PID: $!"

echo "All GPU 5 V3 jobs started"
