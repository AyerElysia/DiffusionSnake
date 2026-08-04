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
  dense6_d1 \
  7 \
  configs/volmem/verse_memflowdit_dit_ablation_dense6_d1_gpu7.yaml \
  data/outputs/volmem/verse_memflowdit_dit_ablation_dense6_d1_gpu7

result_root=data/outputs/volmem/diagnostics/dit_ffn_d1_ablation_20260803
mkdir -p "${result_root}"

if ! pgrep -af '[w]ait_and_launch_dit_ffn_d1_pending_v2.sh' >/dev/null; then
  nohup setsid tools/volmem/wait_and_launch_dit_ffn_d1_pending_v2.sh \
    >"${result_root}/pending_launcher_v2.log" 2>&1 </dev/null &
  echo $! >"${result_root}/pending_launcher_v2.pid"
fi

if ! pgrep -af '[w]atch_and_eval_dit_ffn_d1_ablation.sh' >/dev/null; then
  nohup setsid tools/volmem/watch_and_eval_dit_ffn_d1_ablation.sh \
    >"${result_root}/watcher.log" 2>&1 </dev/null &
  echo $! >"${result_root}/watcher.pid"
fi
