#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

result_root=data/outputs/volmem/diagnostics/output_head_moe_2026_screen_20260802
cache_root=/dev/shm/memflowdit_moonvit_cache
mkdir -p "${result_root}"

d0_cfg=configs/volmem/verse_memflowdit_output_head_d0_dense_gpu4.yaml
d1_cfg=configs/volmem/verse_memflowdit_output_head_d1_dense_residual_gpu5.yaml
l0_cfg=configs/volmem/verse_memflowdit_output_head_l0_legacy_gpu4_queued.yaml
m1k2_cfg=configs/volmem/verse_memflowdit_output_head_m1_modern_k2_gpu6.yaml
m1k1_cfg=configs/volmem/verse_memflowdit_output_head_m1_modern_k1_gpu7.yaml

d0_dir=data/outputs/volmem/verse_memflowdit_output_head_d0_dense_gpu4
d1_dir=data/outputs/volmem/verse_memflowdit_output_head_d1_dense_residual_gpu5
l0_dir=data/outputs/volmem/verse_memflowdit_output_head_l0_legacy_gpu4_queued
m1k2_dir=data/outputs/volmem/verse_memflowdit_output_head_m1_modern_k2_gpu6
m1k1_dir=data/outputs/volmem/verse_memflowdit_output_head_m1_modern_k1_gpu7

d0_ckpt=${d0_dir}/checkpoints/step_000300.pt
d1_ckpt=${d1_dir}/checkpoints/step_000300.pt
l0_ckpt=${l0_dir}/checkpoints/step_000300.pt
m1k2_ckpt=${m1k2_dir}/checkpoints/step_000300.pt
m1k1_ckpt=${m1k1_dir}/checkpoints/step_000300.pt

# GPU4 hosts D0 first.  The legacy control is queued on the same GPU so the
# two already-running layer-coverage experiments on GPU0/1 remain untouched.
while [[ ! -s "${d0_ckpt}" ]]; do
  date '+[%F %T] waiting for D0 step-300 before launching queued L0'
  sleep 30
done

if [[ ! -s "${l0_ckpt}" ]] && ! pgrep -af "train_memflowdit.py.*${l0_cfg}" >/dev/null; then
  mkdir -p "${l0_dir}"
  CUDA_VISIBLE_DEVICES=4 nohup setsid /usr/bin/python3.8 \
    tools/volmem/train_memflowdit.py \
    --cfg_file "${l0_cfg}" \
    --max_steps 300 \
    --save_every 100 \
    --chunks_per_step 4 \
    --seed 20260802 \
    >"${l0_dir}/train_0300.log" 2>&1 </dev/null &
  l0_pid=$!
  echo "${l0_pid}" >"${l0_dir}/launcher.pid"
  echo "queued L0 launched: pid=${l0_pid} gpu=4"
fi

while [[ ! -s "${d0_ckpt}" || ! -s "${d1_ckpt}" || ! -s "${l0_ckpt}" || ! -s "${m1k2_ckpt}" || ! -s "${m1k1_ckpt}" ]]; do
  date '+[%F %T] waiting for all five step-300 checkpoints'
  sleep 30
done

run_eval() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local memory_mode="$4"
  local batch_size="$5"
  local result_dir="${result_root}/${label}_${memory_mode}_batch${batch_size}"
  if [[ -s "${result_dir}/summary.json" ]]; then
    echo "${label} ${memory_mode} batch${batch_size}: already evaluated"
    return
  fi
  mkdir -p "${result_dir}"
  CUDA_VISIBLE_DEVICES=4 /usr/bin/python3.8 \
    tools/volmem/eval_memflowdit_parallel.py \
    --cfg_file "${config}" \
    --ckpt "${checkpoint}" \
    --memory-mode "${memory_mode}" \
    --parallel-batch-size "${batch_size}" \
    --box-mode gt \
    --max-volumes 3 \
    --seed 20260731 \
    --locate-feat-cache-root "${cache_root}" \
    --result-dir "${result_dir}" \
    --log-every 100 \
    >"${result_dir}/run.log" 2>&1
}

labels=(d0_dense d1_dense_residual l0_legacy m1_modern_k2 m1_modern_k1)
configs=("${d0_cfg}" "${d1_cfg}" "${l0_cfg}" "${m1k2_cfg}" "${m1k1_cfg}")
checkpoints=("${d0_ckpt}" "${d1_ckpt}" "${l0_ckpt}" "${m1k2_ckpt}" "${m1k1_ckpt}")

# Run speed measurements in isolation on one physical GPU.  This is slower
# than parallel evaluation but prevents cross-device and concurrent-I/O noise
# from deciding a small output-head latency difference.
for index in 0 1 2 3 4; do
  run_eval "${labels[$index]}" "${configs[$index]}" "${checkpoints[$index]}" parallel-off 1
  run_eval "${labels[$index]}" "${configs[$index]}" "${checkpoints[$index]}" parallel-off 8
done

# Quality check with the real causal memory path.  The output-head decision is
# still based primarily on memory-off causality, but a candidate may not break
# autoregressive use.
for index in 0 1 2 3 4; do
  run_eval "${labels[$index]}" "${configs[$index]}" "${checkpoints[$index]}" autoregressive 1
done

/usr/bin/python3.8 tools/volmem/summarize_output_head_moe_2026.py \
  --result-root "${result_root}" \
  >"${result_root}/summary.txt"

date '+[%F %T] output-head screening evaluation complete'
