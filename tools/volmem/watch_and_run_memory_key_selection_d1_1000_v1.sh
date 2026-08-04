#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

cfg=configs/volmem/verse_memflowdit_output_head_d1_dense_residual_gpu5.yaml
ckpt=data/outputs/volmem/verse_memflowdit_output_head_confirm_d1_1000_gpu5/checkpoints/step_001000.pt
result_root=data/outputs/volmem/diagnostics/memory_causal_d1_step1000_20260803
base_comparison=${result_root}/comparison.json
cache_root=/dev/shm/memflowdit_moonvit_cache
mkdir -p "${result_root}"

# This is versioned separately because the base causal watcher was already
# running when the parameter-free selector was added.  Never overwrite a live
# shell script in place.
while [[ ! -s "${ckpt}" || ! -s "${base_comparison}" ]]; do
  date '+[%F %T] waiting for base Memory causal audit'
  sleep 30
done

run_eval() {
  local label="$1"
  local capacity="$2"
  local volumes="$3"
  local result_dir="${result_root}/${label}"
  if [[ -s "${result_dir}/summary.json" ]]; then
    return
  fi
  mkdir -p "${result_dir}"
  CUDA_VISIBLE_DEVICES=7 /usr/bin/python3.8 \
    tools/volmem/eval_memflowdit_parallel.py \
    --cfg_file "${cfg}" \
    --ckpt "${ckpt}" \
    --memory-mode frozen-key-similar \
    --memory-capacity "${capacity}" \
    --parallel-batch-size 8 \
    --box-mode gt \
    --max-volumes "${volumes}" \
    --seed 20260731 \
    --locate-feat-cache-root "${cache_root}" \
    --result-dir "${result_dir}" \
    --log-every 100 \
    >"${result_dir}/run.log" 2>&1
}

run_eval quick_key_similar_k4 4 1
run_eval quick_key_similar_k8 8 1
run_eval quick_key_similar_k16 16 1
run_eval full_key_similar_k8 8 3
run_eval full_key_similar_k16 16 3

/usr/bin/python3.8 tools/volmem/summarize_memory_causal_d1_1000.py \
  --result-root "${result_root}" \
  --head-confirm-root data/outputs/volmem/diagnostics/output_head_moe_2026_confirm_1000_20260803 \
  >"${result_root}/summary_with_key_selection.txt"
date '+[%F %T] parameter-free Memory-key selection audit complete'
