#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

d1_cfg=configs/volmem/verse_memflowdit_output_head_d1_dense_residual_gpu5.yaml
l0_cfg=configs/volmem/verse_memflowdit_output_head_l0_legacy_gpu4_queued.yaml
d1_ckpt=data/outputs/volmem/verse_memflowdit_output_head_confirm_d1_1000_gpu5/checkpoints/step_001000.pt
l0_ckpt=data/outputs/volmem/verse_memflowdit_output_head_confirm_l0_1000_gpu4/checkpoints/step_001000.pt
result_root=data/outputs/volmem/diagnostics/output_head_moe_2026_confirm_1000_20260803
cache_root=/dev/shm/memflowdit_moonvit_cache
mkdir -p "${result_root}"

while [[ ! -s "${d1_ckpt}" || ! -s "${l0_ckpt}" ]]; do
  date '+[%F %T] waiting for D1/L0 step-1000 checkpoints'
  sleep 30
done

run_eval() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local mode="$4"
  local batch="$5"
  local result_dir="${result_root}/${label}_${mode}_batch${batch}"
  if [[ -s "${result_dir}/summary.json" ]]; then
    echo "${label}: already evaluated"
    return
  fi
  mkdir -p "${result_dir}"
  CUDA_VISIBLE_DEVICES=6 /usr/bin/python3.8 \
    tools/volmem/eval_memflowdit_parallel.py \
    --cfg_file "${config}" \
    --ckpt "${checkpoint}" \
    --memory-mode "${mode}" \
    --parallel-batch-size "${batch}" \
    --box-mode gt \
    --max-volumes 3 \
    --seed 20260731 \
    --locate-feat-cache-root "${cache_root}" \
    --result-dir "${result_dir}" \
    --log-every 100 \
    >"${result_dir}/run.log" 2>&1
}

# Two rounds per batch with reversed order on the same physical GPU.  This
# removes cold-start/order effects from the final latency comparison.
run_eval round1_d1 "${d1_cfg}" "${d1_ckpt}" parallel-off 1
run_eval round1_l0 "${l0_cfg}" "${l0_ckpt}" parallel-off 1
run_eval round2_l0 "${l0_cfg}" "${l0_ckpt}" parallel-off 1
run_eval round2_d1 "${d1_cfg}" "${d1_ckpt}" parallel-off 1

run_eval round1_d1 "${d1_cfg}" "${d1_ckpt}" parallel-off 8
run_eval round1_l0 "${l0_cfg}" "${l0_ckpt}" parallel-off 8
run_eval round2_l0 "${l0_cfg}" "${l0_ckpt}" parallel-off 8
run_eval round2_d1 "${d1_cfg}" "${d1_ckpt}" parallel-off 8

run_eval d1 "${d1_cfg}" "${d1_ckpt}" autoregressive 1
run_eval l0 "${l0_cfg}" "${l0_ckpt}" autoregressive 1

/usr/bin/python3.8 tools/volmem/summarize_output_head_confirm_1000.py \
  --result-root "${result_root}" \
  >"${result_root}/summary.txt"

date '+[%F %T] D1 vs L0 step-1000 confirmation complete'
