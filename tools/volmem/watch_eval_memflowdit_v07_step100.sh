#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

GPU=4
CFG=configs/volmem/verse_memflowdit_v0_7_balanced_memory_gpu1.yaml
SOURCE_CKPT=data/outputs/volmem/verse_memflowdit_v0_7_balanced_memory_gpu1/checkpoints/step_000100.pt
STAGED_CKPT=/dev/shm/memflowdit_checkpoints/v07_balanced_step_000100.pt
ROOT=data/outputs/volmem/diagnostics/memflowdit_v07_step000100_gt_causal_gate
CACHE_ROOT=/dev/shm/memflowdit_moonvit_cache
SEED=20260731
DEADLINE_EPOCH=$(date -d '2026-08-02 20:00:00 +0800' +%s)

mkdir -p "${ROOT}"
exec >>"${ROOT}/watch.log" 2>&1
printf '%s\n' "$$" >"${ROOT}/watch.pid"

timestamp() {
    date --iso-8601=seconds
}

before_deadline() {
    [[ "$(date +%s)" -lt "${DEADLINE_EPOCH}" ]]
}

echo "[$(timestamp)] waiting for ${SOURCE_CKPT}"
while before_deadline; do
    if [[ -s "${SOURCE_CKPT}" ]]; then
        size_before=$(stat -c %s "${SOURCE_CKPT}")
        sleep 20
        size_after=$(stat -c %s "${SOURCE_CKPT}")
        if [[ "${size_before}" = "${size_after}" ]]; then
            break
        fi
    fi
    sleep 40
done

if [[ ! -s "${SOURCE_CKPT}" ]]; then
    echo "[$(timestamp)] deadline reached before checkpoint appeared"
    exit 2
fi

echo "[$(timestamp)] waiting for GPU ${GPU} to be idle"
while before_deadline; do
    used_mib=$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [[ "${used_mib}" -lt 1024 ]]; then
        break
    fi
    sleep 60
done

if ! before_deadline; then
    echo "[$(timestamp)] deadline reached before GPU ${GPU} became idle"
    exit 3
fi

mkdir -p "$(dirname "${STAGED_CKPT}")"
cp "${SOURCE_CKPT}" "${STAGED_CKPT}"
echo "[$(timestamp)] staged checkpoint on local RAM disk"

run_eval() {
    local name="$1"
    local mode="$2"
    local result_dir="${ROOT}/${name}"
    mkdir -p "${result_dir}"
    echo "[$(timestamp)] starting ${name} mode=${mode} GPU=${GPU}"
    CUDA_VISIBLE_DEVICES="${GPU}" python tools/volmem/eval_memflowdit_parallel.py \
        --cfg_file "${CFG}" \
        --ckpt "${STAGED_CKPT}" \
        --split val \
        --memory-mode "${mode}" \
        --memory-capacity 4 \
        --memory-pool-size 8 \
        --parallel-batch-size 1 \
        --locate-feat-cache-root "${CACHE_ROOT}" \
        --box-mode gt \
        --max-volumes 3 \
        --log-every 50 \
        --seed "${SEED}" \
        --result-dir "${result_dir}" \
        >"${result_dir}/run.log" 2>&1
    echo "[$(timestamp)] completed ${name}"
}

# Both modes are single-pass and consume the same Flow noise in the same slice
# order.  The only causal difference is whether prior GT Memory is readable.
run_eval off off
run_eval oracle_causal oracle

python tools/volmem/summarize_parallel_memory_stage.py "${ROOT}" --baseline off \
    >"${ROOT}/comparison_stdout.json"
echo "[$(timestamp)] step100 causal gate complete"
