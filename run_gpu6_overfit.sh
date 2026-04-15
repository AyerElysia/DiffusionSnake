#!/bin/bash
# 单独启动脚本 - GPU 6 上的训练任务

cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30
source ~/miniconda3/etc/profile.d/conda.sh
conda activate snake1

# V2.3
echo "Starting V2.3 on GPU 6..."
export CFG_FILE=configs/btcv_diffusion_dit_v2_3_single_overfit_gpu6.yaml
CUDA_VISIBLE_DEVICES=6 nohup python diffusion_train.py > logs/v2_3_single_overfit_gpu6.log 2>&1 &
echo "V2.3 started, PID: $!"

# V3.1
echo "Starting V3.1 on GPU 6..."
export CFG_FILE=configs/btcv_diffusion_dit_v3_1_single_overfit.yaml
CUDA_VISIBLE_DEVICES=6 nohup python diffusion_train.py > logs/v3_1_single_overfit_gpu6.log 2>&1 &
echo "V3.1 started, PID: $!"

# V3.2
echo "Starting V3.2 on GPU 6..."
export CFG_FILE=configs/btcv_diffusion_dit_v3_2_single_overfit.yaml
CUDA_VISIBLE_DEVICES=6 nohup python diffusion_train.py > logs/v3_2_single_overfit_gpu6.log 2>&1 &
echo "V3.2 started, PID: $!"

echo "All GPU 6 jobs started"
