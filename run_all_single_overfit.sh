#!/bin/bash
# 批量启动 V3 系列的单样本过拟合训练
# GPU 分配: GPU5(V3.1), GPU6(V3.2), GPU7(V3.4), GPU0(V3.6)

cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30

source ~/miniconda3/etc/profile.d/conda.sh
conda activate snake1

mkdir -p logs

echo "=== Starting V3.1 on GPU 5 ==="
export CFG_FILE=configs/btcv_diffusion_dit_v3_1_single_overfit.yaml
CUDA_VISIBLE_DEVICES=5 nohup python diffusion_train.py > logs/v3_1_single_overfit_gpu5.log 2>&1 &
echo "V3.1 started, PID: $!"
sleep 3

echo "=== Starting V3.2 on GPU 6 ==="
export CFG_FILE=configs/btcv_diffusion_dit_v3_2_single_overfit.yaml
CUDA_VISIBLE_DEVICES=6 nohup python diffusion_train.py > logs/v3_2_single_overfit_gpu6.log 2>&1 &
echo "V3.2 started, PID: $!"
sleep 3

echo "=== Starting V3.4 on GPU 7 ==="
export CFG_FILE=configs/btcv_diffusion_dit_v3_4_single_overfit.yaml
CUDA_VISIBLE_DEVICES=7 nohup python diffusion_train.py > logs/v3_4_single_overfit_gpu7.log 2>&1 &
echo "V3.4 started, PID: $!"
sleep 3

echo "=== Starting V3.6 on GPU 0 ==="
export CFG_FILE=configs/btcv_diffusion_dit_v3_6_single_overfit.yaml
CUDA_VISIBLE_DEVICES=0 nohup python diffusion_train.py > logs/v3_6_single_overfit_gpu0.log 2>&1 &
echo "V3.6 started, PID: $!"

echo ""
echo "=== All V3 training jobs started ==="
echo "Check logs in logs/ directory"
echo "Monitor with: nvidia-smi"
