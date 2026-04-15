#!/bin/bash
# 批量启动所有版本的单样本过拟合训练
# GPU分配: GPU5(V2.0-2.2), GPU6(V2.3,V3.1,V3.2), GPU7(V3.0已运行)

cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate snake1

# GPU 5: V2.0, V2.1, V2.2 (串行)
echo "=== Starting V2.0 on GPU 5 ==="
export CFG_FILE=configs/btcv_diffusion_dit_v2_single_overfit.yaml
CUDA_VISIBLE_DEVICES=5 nohup python diffusion_train.py > logs/v2_0_single_overfit_gpu5.log 2>&1 &
echo "V2.0 started, PID: $!"

sleep 5

echo "=== Starting V2.1 on GPU 5 ==="
export CFG_FILE=configs/btcv_diffusion_dit_v2_1_single_overfit.yaml
CUDA_VISIBLE_DEVICES=5 nohup python diffusion_train.py > logs/v2_1_single_overfit_gpu5.log 2>&1 &
echo "V2.1 started, PID: $!"

sleep 5

echo "=== Starting V2.2 on GPU 5 ==="
export CFG_FILE=configs/btcv_diffusion_dit_v2_2_single_overfit.yaml
CUDA_VISIBLE_DEVICES=5 nohup python diffusion_train.py > logs/v2_2_single_overfit_gpu5.log 2>&1 &
echo "V2.2 started, PID: $!"

sleep 5

# GPU 6: V2.3, V3.1, V3.2 (串行)
echo "=== Starting V2.3 on GPU 6 ==="
export CFG_FILE=configs/btcv_diffusion_dit_v2_3_single_overfit_gpu6.yaml
CUDA_VISIBLE_DEVICES=6 nohup python diffusion_train.py > logs/v2_3_single_overfit_gpu6.log 2>&1 &
echo "V2.3 started, PID: $!"

sleep 5

echo "=== Starting V3.1 on GPU 6 ==="
export CFG_FILE=configs/btcv_diffusion_dit_v3_1_single_overfit.yaml
CUDA_VISIBLE_DEVICES=6 nohup python diffusion_train.py > logs/v3_1_single_overfit_gpu6.log 2>&1 &
echo "V3.1 started, PID: $!"

sleep 5

echo "=== Starting V3.2 on GPU 6 ==="
export CFG_FILE=configs/btcv_diffusion_dit_v3_2_single_overfit.yaml
CUDA_VISIBLE_DEVICES=6 nohup python diffusion_train.py > logs/v3_2_single_overfit_gpu6.log 2>&1 &
echo "V3.2 started, PID: $!"

echo ""
echo "=== All training jobs started ==="
echo "GPU 5: V2.0, V2.1, V2.2"
echo "GPU 6: V2.3, V3.1, V3.2"
echo "GPU 7: V3.0 (already running)"
echo ""
echo "Check logs in logs/ directory"
echo "Monitor with: nvidia-smi"
