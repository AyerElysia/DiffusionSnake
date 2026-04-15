#!/bin/bash
# Resume unfinished single-sample overfit runs on GPU 5.

cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30

source ~/miniconda3/etc/profile.d/conda.sh
conda activate snake1

mkdir -p logs

run_one() {
    local cfg="$1"
    local log_file="$2"
    echo "=== Resuming ${cfg} on GPU 5 ==="
    export CFG_FILE="${cfg}"
    CUDA_VISIBLE_DEVICES=5 python -u diffusion_train.py --cfg_file "${cfg}" resume True > "${log_file}" 2>&1
}

run_one "configs/btcv_diffusion_dit_v2_single_overfit.yaml" "logs/v2_0_single_overfit_resume_gpu5.log"
run_one "configs/btcv_diffusion_dit_v2_1_single_overfit.yaml" "logs/v2_1_single_overfit_resume_gpu5.log"
run_one "configs/btcv_diffusion_dit_v2_2_single_overfit.yaml" "logs/v2_2_single_overfit_resume_gpu5.log"

echo "All GPU 5 resume jobs finished."
