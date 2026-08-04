#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

launch_one() {
  local label="$1"
  local gpu="$2"
  local config="$3"
  local output_dir="$4"
  local log_file="${output_dir}/train_1000.log"

  if pgrep -af "train_memflowdit.py.*${config}" >/dev/null; then
    echo "${label}: already running"
    return
  fi
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup setsid /usr/bin/python3.8 \
    tools/volmem/train_memflowdit.py \
    --cfg_file "${config}" \
    --max_steps 1000 \
    --save_every 100 \
    --chunks_per_step 4 \
    --seed 20260802 \
    >"${log_file}" 2>&1 </dev/null &
  local pid=$!
  echo "${pid}" >"${output_dir}/launcher.pid"
  echo "${label}: pid=${pid} gpu=${gpu} log=${log_file}"
}

launch_one \
  control \
  0 \
  configs/volmem/verse_memflowdit_moe_shared_ablation_control_gpu0.yaml \
  data/outputs/volmem/verse_memflowdit_moe_shared_ablation_control_gpu0

launch_one \
  shared \
  1 \
  configs/volmem/verse_memflowdit_moe_shared_ablation_promoe_gpu1.yaml \
  data/outputs/volmem/verse_memflowdit_moe_shared_ablation_promoe_gpu1
