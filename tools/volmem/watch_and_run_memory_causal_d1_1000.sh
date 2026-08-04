#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

cfg=configs/volmem/verse_memflowdit_output_head_d1_dense_residual_gpu5.yaml
ckpt=data/outputs/volmem/verse_memflowdit_output_head_confirm_d1_1000_gpu5/checkpoints/step_001000.pt
dense_ckpt=data/outputs/volmem/verse_memflowdit_dit_ablation_dense6_d1_gpu7/checkpoints/step_001000.pt
head_comparison=data/outputs/volmem/diagnostics/output_head_moe_2026_confirm_1000_20260803/comparison.json
result_root=data/outputs/volmem/diagnostics/memory_causal_d1_step1000_20260803
cache_root=/dev/shm/memflowdit_moonvit_cache
mkdir -p "${result_root}"

# GPU7 first trains Dense-6.  The output-head watcher owns GPU6 until its
# latency test is complete.  Waiting for both keeps the causal audit isolated.
while [[ ! -s "${ckpt}" || ! -s "${dense_ckpt}" || ! -s "${head_comparison}" ]]; do
  date '+[%F %T] waiting for D1, Dense-6, and output-head confirmation'
  sleep 30
done

run_eval() {
  local label="$1"
  local mode="$2"
  local capacity="$3"
  local batch="$4"
  local volumes="$5"
  local result_dir="${result_root}/${label}"
  if [[ -s "${result_dir}/summary.json" ]]; then
    return
  fi
  mkdir -p "${result_dir}"
  CUDA_VISIBLE_DEVICES=7 /usr/bin/python3.8 \
    tools/volmem/eval_memflowdit_parallel.py \
    --cfg_file "${cfg}" \
    --ckpt "${ckpt}" \
    --memory-mode "${mode}" \
    --memory-capacity "${capacity}" \
    --parallel-batch-size "${batch}" \
    --box-mode gt \
    --max-volumes "${volumes}" \
    --seed 20260731 \
    --locate-feat-cache-root "${cache_root}" \
    --result-dir "${result_dir}" \
    --log-every 100 \
    >"${result_dir}/run.log" 2>&1
}

# One-volume mechanism matrix.  AR K1 is also the non-redundant equivalent of
# repeating only the nearest state: duplicate identical K/V tokens cannot add
# information to softmax attention.
run_eval quick_off parallel-off 4 8 1
for capacity in 1 2 4 8 16; do
  run_eval "quick_ar_k${capacity}" autoregressive "${capacity}" 1 1
done
for capacity in 4 8 16; do
  run_eval "quick_frozen_causal_k${capacity}" frozen-causal "${capacity}" 8 1
  run_eval "quick_frozen_bidirectional_k${capacity}" frozen-bidirectional "${capacity}" 8 1
  run_eval "quick_frozen_shuffled_k${capacity}" frozen-shuffled "${capacity}" 8 1
done
run_eval quick_frozen_all_k256 frozen-causal 256 8 1
run_eval quick_feature_k4 frozen-feature-causal 4 8 1
run_eval quick_oracle_k4 frozen-oracle-causal 4 8 1

# Three-volume causal confirmation.  The matching off and AR-K4 results are
# reused from the output-head confirmation; the frozen modes are new here.
run_eval full_frozen_causal_k4 frozen-causal 4 8 3
run_eval full_frozen_bidirectional_k4 frozen-bidirectional 4 8 3
run_eval full_frozen_shuffled_k4 frozen-shuffled 4 8 3
run_eval full_frozen_all_k256 frozen-causal 256 8 3
run_eval full_oracle_k4 frozen-oracle-causal 4 8 3

/usr/bin/python3.8 tools/volmem/summarize_memory_causal_d1_1000.py \
  --result-root "${result_root}" \
  --head-confirm-root data/outputs/volmem/diagnostics/output_head_moe_2026_confirm_1000_20260803 \
  >"${result_root}/summary.txt"
date '+[%F %T] D1 step-1000 Memory causal audit complete'
