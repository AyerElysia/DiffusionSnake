#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

max_steps=300
seed=20260802

launch_one() {
  local label="$1"
  local gpu="$2"
  local config="$3"
  local output_dir="$4"
  local log_file="${output_dir}/train_0300.log"

  if [[ -s "${output_dir}/checkpoints/step_000300.pt" ]]; then
    echo "${label}: complete"
    return
  fi
  if pgrep -af "train_memflowdit.py.*${config}" >/dev/null; then
    echo "${label}: already running"
    return
  fi
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup setsid /usr/bin/python3.8 \
    tools/volmem/train_memflowdit.py \
    --cfg_file "${config}" \
    --max_steps "${max_steps}" \
    --save_every 100 \
    --chunks_per_step 4 \
    --seed "${seed}" \
    >"${log_file}" 2>&1 </dev/null &
  local pid=$!
  echo "${pid}" >"${output_dir}/launcher.pid"
  echo "${label}: pid=${pid} gpu=${gpu} log=${log_file}"
}

launch_one \
  d0_dense \
  4 \
  configs/volmem/verse_memflowdit_output_head_d0_dense_gpu4.yaml \
  data/outputs/volmem/verse_memflowdit_output_head_d0_dense_gpu4

launch_one \
  d1_dense_residual \
  5 \
  configs/volmem/verse_memflowdit_output_head_d1_dense_residual_gpu5.yaml \
  data/outputs/volmem/verse_memflowdit_output_head_d1_dense_residual_gpu5

launch_one \
  m1_modern_k2 \
  6 \
  configs/volmem/verse_memflowdit_output_head_m1_modern_k2_gpu6.yaml \
  data/outputs/volmem/verse_memflowdit_output_head_m1_modern_k2_gpu6

launch_one \
  m1_modern_k1 \
  7 \
  configs/volmem/verse_memflowdit_output_head_m1_modern_k1_gpu7.yaml \
  data/outputs/volmem/verse_memflowdit_output_head_m1_modern_k1_gpu7

result_root=data/outputs/volmem/diagnostics/output_head_moe_2026_screen_20260802
mkdir -p "${result_root}"
if ! pgrep -af "watch_output_head_moe_2026_screen.sh" >/dev/null; then
  nohup setsid tools/volmem/watch_output_head_moe_2026_screen.sh \
    >"${result_root}/watcher.log" 2>&1 </dev/null &
  watcher_pid=$!
  echo "${watcher_pid}" >"${result_root}/watcher.pid"
  echo "watcher: pid=${watcher_pid}"
fi
