#!/bin/bash
# GPU0 single-sample overfit training for V3.6

cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30

source ~/miniconda3/etc/profile.d/conda.sh
conda activate snake1

mkdir -p logs

echo "=== Starting V3.6 on GPU 0 ==="
tmux kill-session -t v3_6 2>/dev/null || true
tmux new-session -d -s v3_6 'source ~/miniconda3/etc/profile.d/conda.sh; conda activate snake1; cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30; export CFG_FILE=configs/btcv_diffusion_dit_v3_6_single_overfit.yaml; CUDA_VISIBLE_DEVICES=0 python -u diffusion_train.py > /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30/logs/v3_6_single_overfit_gpu0.log 2>&1'
echo "V3.6 started in tmux session: v3_6"

echo ""
echo "Check log: logs/v3_6_single_overfit_gpu0.log"
