#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

control_cfg=configs/volmem/verse_memflowdit_moe_shared_ablation_control_gpu0.yaml
shared_cfg=configs/volmem/verse_memflowdit_moe_shared_ablation_promoe_gpu1.yaml
control_ckpt=data/outputs/volmem/verse_memflowdit_moe_shared_ablation_control_gpu0/checkpoints/step_001000.pt
shared_ckpt=data/outputs/volmem/verse_memflowdit_moe_shared_ablation_promoe_gpu1/checkpoints/step_001000.pt
result_root=data/outputs/volmem/diagnostics/moe_shared_ablation_20260802
cache_root=/dev/shm/memflowdit_moonvit_cache

mkdir -p "${result_root}"
while [[ ! -s "${control_ckpt}" || ! -s "${shared_ckpt}" ]]; do
  date '+[%F %T] waiting for both step-1000 checkpoints'
  sleep 30
done

run_eval() {
  local label="$1"
  local gpu="$2"
  local config="$3"
  local checkpoint="$4"
  local batch_size="$5"
  local result_dir="${result_root}/${label}"
  mkdir -p "${result_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" /usr/bin/python3.8 \
    tools/volmem/eval_memflowdit_parallel.py \
    --cfg_file "${config}" \
    --ckpt "${checkpoint}" \
    --memory-mode parallel-off \
    --parallel-batch-size "${batch_size}" \
    --box-mode gt \
    --max-volumes 3 \
    --seed 20260731 \
    --locate-feat-cache-root "${cache_root}" \
    --result-dir "${result_dir}" \
    --log-every 100 \
    >"${result_dir}/run.log" 2>&1
}

run_round() {
  local round="$1"
  local control_gpu_b1="$2"
  local shared_gpu_b1="$3"
  local control_gpu_b8="$4"
  local shared_gpu_b8="$5"

  run_eval "${round}_batch1_control" "${control_gpu_b1}" "${control_cfg}" "${control_ckpt}" 1 &
  local p1=$!
  run_eval "${round}_batch1_shared" "${shared_gpu_b1}" "${shared_cfg}" "${shared_ckpt}" 1 &
  local p2=$!
  run_eval "${round}_batch8_control" "${control_gpu_b8}" "${control_cfg}" "${control_ckpt}" 8 &
  local p3=$!
  run_eval "${round}_batch8_shared" "${shared_gpu_b8}" "${shared_cfg}" "${shared_ckpt}" 8 &
  local p4=$!
  wait "${p1}" "${p2}" "${p3}" "${p4}"
}

# Cross over physical GPUs in the second round so device variance does not
# masquerade as an architecture speed difference.
run_round round1 4 5 6 7
run_round round2 5 4 7 6

/usr/bin/python3.8 tools/volmem/summarize_moe_shared_ablation.py \
  --result-root "${result_root}" \
  --control-train data/outputs/volmem/verse_memflowdit_moe_shared_ablation_control_gpu0/train.jsonl \
  --shared-train data/outputs/volmem/verse_memflowdit_moe_shared_ablation_promoe_gpu1/train.jsonl \
  >"${result_root}/summary.txt"

date '+[%F %T] evaluation and summary complete'
