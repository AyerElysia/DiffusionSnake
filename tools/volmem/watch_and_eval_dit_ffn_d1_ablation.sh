#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

result_root=data/outputs/volmem/diagnostics/dit_ffn_d1_ablation_20260803
cache_root=/dev/shm/memflowdit_moonvit_cache
mkdir -p "${result_root}"

dense_cfg=configs/volmem/verse_memflowdit_dit_ablation_dense6_d1_gpu7.yaml
odd_cfg=configs/volmem/verse_memflowdit_output_head_d1_dense_residual_gpu5.yaml
all_cfg=configs/volmem/verse_memflowdit_dit_ablation_all6_d1_gpu4.yaml
shared_cfg=configs/volmem/verse_memflowdit_dit_ablation_sharedodd3_d1_gpu5.yaml
dense_ckpt=data/outputs/volmem/verse_memflowdit_dit_ablation_dense6_d1_gpu7/checkpoints/step_001000.pt
odd_ckpt=data/outputs/volmem/verse_memflowdit_output_head_confirm_d1_1000_gpu5/checkpoints/step_001000.pt
all_ckpt=data/outputs/volmem/verse_memflowdit_dit_ablation_all6_d1_gpu4/checkpoints/step_001000.pt
shared_ckpt=data/outputs/volmem/verse_memflowdit_dit_ablation_sharedodd3_d1_gpu5/checkpoints/step_001000.pt

while [[ ! -s "${dense_ckpt}" || ! -s "${odd_ckpt}" || ! -s "${all_ckpt}" || ! -s "${shared_ckpt}" ]]; do
  date '+[%F %T] waiting for four DiT-FFN step-1000 checkpoints'
  sleep 30
done

# The output-head confirmation owns GPU6 first.  Waiting for its final JSON
# keeps its latency comparison isolated from these longer evaluations.
while [[ ! -s data/outputs/volmem/diagnostics/output_head_moe_2026_confirm_1000_20260803/comparison.json ]]; do
  date '+[%F %T] waiting for output-head confirmation evaluation'
  sleep 30
done

run_eval() {
  local round="$1"
  local label="$2"
  local config="$3"
  local checkpoint="$4"
  local batch="$5"
  local result_dir="${result_root}/${round}_${label}_parallel-off_batch${batch}"
  if [[ -s "${result_dir}/summary.json" ]]; then
    return
  fi
  mkdir -p "${result_dir}"
  CUDA_VISIBLE_DEVICES=6 /usr/bin/python3.8 \
    tools/volmem/eval_memflowdit_parallel.py \
    --cfg_file "${config}" \
    --ckpt "${checkpoint}" \
    --memory-mode parallel-off \
    --parallel-batch-size "${batch}" \
    --box-mode gt \
    --max-volumes 3 \
    --seed 20260731 \
    --locate-feat-cache-root "${cache_root}" \
    --result-dir "${result_dir}" \
    --log-every 100 \
    >"${result_dir}/run.log" 2>&1
}

run_order() {
  local round="$1"
  local batch="$2"
  shift 2
  for label in "$@"; do
    case "${label}" in
      dense6) run_eval "${round}" dense6 "${dense_cfg}" "${dense_ckpt}" "${batch}" ;;
      odd3) run_eval "${round}" odd3 "${odd_cfg}" "${odd_ckpt}" "${batch}" ;;
      all6) run_eval "${round}" all6 "${all_cfg}" "${all_ckpt}" "${batch}" ;;
      sharedodd3) run_eval "${round}" sharedodd3 "${shared_cfg}" "${shared_ckpt}" "${batch}" ;;
      *) echo "unknown label ${label}" >&2; exit 2 ;;
    esac
  done
}

run_order round1 1 dense6 odd3 all6 sharedodd3
run_order round2 1 sharedodd3 all6 odd3 dense6
run_order round1 8 dense6 odd3 all6 sharedodd3
run_order round2 8 sharedodd3 all6 odd3 dense6

/usr/bin/python3.8 tools/volmem/summarize_dit_ffn_d1_ablation.py \
  --result-root "${result_root}" \
  >"${result_root}/summary.txt"
date '+[%F %T] four-way DiT-FFN D1 ablation complete'
