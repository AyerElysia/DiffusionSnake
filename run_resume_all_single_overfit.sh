#!/bin/bash
# Resume all unfinished single-sample runs in parallel across GPU 5/6/7.

cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30

source ~/miniconda3/etc/profile.d/conda.sh
conda activate snake1

mkdir -p logs

launch() {
    local session="$1"
    local gpu="$2"
    local cfg="$3"
    local log_file="$4"
    tmux kill-session -t "${session}" 2>/dev/null || true
    tmux new-session -d -s "${session}" \
        "source ~/miniconda3/etc/profile.d/conda.sh; conda activate snake1; cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30; export CFG_FILE=${cfg}; CUDA_VISIBLE_DEVICES=${gpu} python -u diffusion_train.py --cfg_file ${cfg} resume True > ${log_file} 2>&1"
    echo "Started ${session} on GPU ${gpu}: ${cfg}"
}

launch "v3_1_resume_gpu5" 5 "configs/btcv_diffusion_dit_v3_1_single_overfit.yaml" "logs/v3_1_single_overfit_resume_gpu5.log"
launch "v3_2_resume_gpu6" 6 "configs/btcv_diffusion_dit_v3_2_single_overfit.yaml" "logs/v3_2_single_overfit_resume_gpu6.log"
launch "v3_4_resume_gpu7" 7 "configs/btcv_diffusion_dit_v3_4_single_overfit.yaml" "logs/v3_4_single_overfit_resume_gpu7.log"
launch "v3_6_resume_gpu0" 0 "configs/btcv_diffusion_dit_v3_6_single_overfit.yaml" "logs/v3_6_single_overfit_resume_gpu0.log"

echo "All unfinished single-sample runs have been launched."
