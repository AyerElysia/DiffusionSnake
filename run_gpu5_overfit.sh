#!/bin/bash
# 单独启动脚本 - GPU 5 上的训练任务

cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30
source ~/miniconda3/etc/profile.d/conda.sh
conda activate snake1

# V2.0
echo "Starting V2.0 on GPU 5..."
export CFG_FILE=configs/btcv_diffusion_dit_v2_single_overfit.yaml
CUDA_VISIBLE_DEVICES=5 nohup python diffusion_train.py > logs/v2_0_single_overfit_gpu5.log 2>&1 &
echo "V2.0 started, PID: $!"

# V2.1
echo "Starting V2.1 on GPU 5..."
export CFG_FILE=configs/btcv_diffusion_dit_v2_1_single_overfit.yaml
CUDA_VISIBLE_DEVICES=5 nohup python diffusion_train.py > logs/v2_1_single_overfit_gpu5.log 2>&1 &
echo "V2.1 started, PID: $!"

# V2.2
echo "Starting V2.2 on GPU 5..."
export CFG_FILE=configs/btcv_diffusion_dit_v2_2_single_overfit.yaml
CUDA_VISIBLE_DEVICES=5 nohup python diffusion_train.py > logs/v2_2_single_overfit_gpu5.log 2>&1 &
echo "V2.2 started, PID: $!"

echo "All GPU 5 jobs started"
