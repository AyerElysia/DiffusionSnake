#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

launch_one() {
  local label="$1"
  local gpu="$2"
  local config="$3"
  local output_dir="$4"
  local checkpoint="${output_dir}/checkpoints/step_001000.pt"

  if [[ -s "${checkpoint}" ]]; then
    echo "${label}: complete"
    return
  fi
  if pgrep -af "train_memflowdit.py.*--output-dir-override ${output_dir}" >/dev/null; then
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
    --output-dir-override "${output_dir}" \
    >"${output_dir}/train_1000.log" 2>&1 </dev/null &
  local pid=$!
  echo "${pid}" >"${output_dir}/launcher.pid"
  echo "${label}: pid=${pid} gpu=${gpu}"
}

d1_dir=data/outputs/volmem/verse_memflowdit_output_head_confirm_d1_1000_gpu5
l0_dir=data/outputs/volmem/verse_memflowdit_output_head_confirm_l0_1000_gpu4

launch_one \
  d1_dense_residual_1000 \
  5 \
  configs/volmem/verse_memflowdit_output_head_d1_dense_residual_gpu5.yaml \
  "${d1_dir}"

launch_one \
  l0_legacy_1000 \
  4 \
  configs/volmem/verse_memflowdit_output_head_l0_legacy_gpu4_queued.yaml \
  "${l0_dir}"

result_root=data/outputs/volmem/diagnostics/output_head_moe_2026_confirm_1000_20260803
mkdir -p "${result_root}"
if ! pgrep -af "watch_output_head_confirm_d1_vs_l0_1000.sh" >/dev/null; then
  nohup setsid tools/volmem/watch_output_head_confirm_d1_vs_l0_1000.sh \
    >"${result_root}/watcher.log" 2>&1 </dev/null &
  watcher_pid=$!
  echo "${watcher_pid}" >"${result_root}/watcher.pid"
  echo "watcher: pid=${watcher_pid}"
fi
