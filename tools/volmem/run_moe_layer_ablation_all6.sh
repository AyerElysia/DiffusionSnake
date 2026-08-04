#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30
config=configs/volmem/verse_memflowdit_moe_layer_ablation_all6_gpu0.yaml
output_dir=data/outputs/volmem/verse_memflowdit_moe_layer_ablation_all6_gpu0

if pgrep -af "train_memflowdit.py.*${config}" >/dev/null; then
  echo "all6 layer ablation already running"
  exit 0
fi

mkdir -p "${output_dir}"
CUDA_VISIBLE_DEVICES=0 nohup setsid /usr/bin/python3.8 \
  tools/volmem/train_memflowdit.py \
  --cfg_file "${config}" \
  --max_steps 1000 \
  --save_every 100 \
  --chunks_per_step 4 \
  --seed 20260802 \
  >"${output_dir}/train_1000.log" 2>&1 </dev/null &
pid=$!
echo "${pid}" >"${output_dir}/launcher.pid"
echo "all6: pid=${pid} gpu=0"

result_root=data/outputs/volmem/diagnostics/moe_layer_ablation_odd3_vs_all6_20260802
mkdir -p "${result_root}"
if ! pgrep -af "watch_and_eval_moe_layer_ablation.sh" >/dev/null; then
  nohup setsid tools/volmem/watch_and_eval_moe_layer_ablation.sh \
    >"${result_root}/watcher.log" 2>&1 </dev/null &
  watcher_pid=$!
  echo "${watcher_pid}" >"${result_root}/watcher.pid"
  echo "watcher: pid=${watcher_pid}"
fi
