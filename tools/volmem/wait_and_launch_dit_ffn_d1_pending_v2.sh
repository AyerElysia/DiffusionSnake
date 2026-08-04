#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

head_comparison=data/outputs/volmem/diagnostics/output_head_moe_2026_confirm_1000_20260803/comparison.json
while [[ ! -s "${head_comparison}" ]]; do
  date '+[%F %T] waiting for isolated output-head quality and latency evaluation'
  sleep 30
done

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
  if pgrep -af "train_memflowdit.py.*${output_dir}" >/dev/null; then
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

launch_one \
  all6_d1 \
  4 \
  configs/volmem/verse_memflowdit_dit_ablation_all6_d1_gpu4.yaml \
  data/outputs/volmem/verse_memflowdit_dit_ablation_all6_d1_gpu4

launch_one \
  sharedodd3_d1 \
  5 \
  configs/volmem/verse_memflowdit_dit_ablation_sharedodd3_d1_gpu5.yaml \
  data/outputs/volmem/verse_memflowdit_dit_ablation_sharedodd3_d1_gpu5
