#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

d1_confirm=data/outputs/volmem/verse_memflowdit_output_head_confirm_d1_1000_gpu5/checkpoints/step_001000.pt
l0_confirm=data/outputs/volmem/verse_memflowdit_output_head_confirm_l0_1000_gpu4/checkpoints/step_001000.pt
while [[ ! -s "${d1_confirm}" || ! -s "${l0_confirm}" ]]; do
  date '+[%F %T] waiting for output-head confirmation checkpoints'
  sleep 30
done

# Let the checkpoint writers and their training CUDA contexts exit before the
# new jobs claim GPU4 and GPU5.
sleep 30

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
  echo $! >"${output_dir}/launcher.pid"
  echo "${label}: pid=$! gpu=${gpu}"
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
