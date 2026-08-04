#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

CFG=configs/volmem/verse_memflowdit_v0_5_minimal_gpu6.yaml
CKPT=data/outputs/volmem/verse_memflowdit_v0_5_minimal_gpu6/checkpoints/step_002300.pt
ROOT=data/outputs/volmem/diagnostics/parallel_memory_stage0_v05_step2300
SEED=20260731

start_eval() {
    local gpu="$1"
    local name="$2"
    local mode="$3"
    shift 3
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
        --locate-feat-cache-root /dev/shm/memflowdit_moonvit_cache \
        --box-mode gt \
        --max-volumes 3 \
        --log-every 50 \
        --seed "${SEED}" \
        --result-dir "${result_dir}" \
        "$@" \
        </dev/null >"${result_dir}/run.log" 2>&1 &
    local pid=$!
    echo "${pid}" >"${result_dir}/run.pid"
    printf '%s\tGPU=%s\tPID=%s\tmode=%s\n' "${name}" "${gpu}" "${pid}" "${mode}"
}

# GPU 6/7 are intentionally untouched.  Each free GPU runs a serial queue so
# checkpoint loading and evaluation outputs remain isolated and reproducible.
start_eval 0 off off
start_eval 1 oracle_causal oracle
start_eval 5 frozen_oracle_bidir frozen-oracle-bidirectional
