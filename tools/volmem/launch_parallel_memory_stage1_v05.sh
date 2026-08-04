#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

CFG=configs/volmem/verse_memflowdit_v0_5_minimal_gpu6.yaml
CKPT=/dev/shm/memflowdit_checkpoints/v05_step_002300.pt
CACHE=/dev/shm/memflowdit_moonvit_cache
ROOT=data/outputs/volmem/diagnostics/parallel_memory_stage1_v05_step2300
SEED=20260731

start_eval() {
    local gpu="$1"
    local name="$2"
    local mode="$3"
    local result_dir="${ROOT}/${name}"
    mkdir -p "${result_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" nohup python tools/volmem/eval_memflowdit_parallel.py \
        --cfg_file "${CFG}" \
        --ckpt "${CKPT}" \
        --split val \
        --memory-mode "${mode}" \
        --memory-capacity 4 \
        --memory-pool-size 8 \
        --parallel-batch-size 8 \
        --locate-feat-cache-root "${CACHE}" \
        --box-mode gt \
        --max-volumes 3 \
        --log-every 50 \
        --seed "${SEED}" \
        --result-dir "${result_dir}" \
        </dev/null >"${result_dir}/run.log" 2>&1 &
    local pid=$!
    echo "${pid}" >"${result_dir}/run.pid"
    printf '%s\tGPU=%s\tPID=%s\tmode=%s\n' "${name}" "${gpu}" "${pid}" "${mode}"
}

start_eval 0 autoregressive autoregressive
start_eval 1 frozen_feature_bidir frozen-feature-bidirectional
start_eval 4 frozen_causal frozen-causal
start_eval 5 frozen_predicted_bidir frozen-bidirectional
