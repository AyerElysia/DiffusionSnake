#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

CFG=configs/volmem/verse_memflowdit_v0_8_evidence_value_mm_gpu5.yaml
OUT=data/outputs/volmem/verse_memflowdit_v0_8_evidence_value_mm_gpu5
LOG=${OUT}/train_2day_gpu5.log
DEADLINE=2026-08-03T07:00:00+08:00
MAX_STEP=2000

mkdir -p "${OUT}"

CUDA_VISIBLE_DEVICES=5 nohup python tools/volmem/train_memflowdit.py \
    --cfg_file "${CFG}" \
    --device cuda:0 \
    --max_steps "${MAX_STEP}" \
    --save_every 100 \
    --seed 20260731 \
    --chunks_per_step 12 \
    --chunk_length 8 \
    --init-memflow-ckpt /dev/shm/memflowdit_checkpoints/v05_step_002300.pt \
    --reset-memory-read \
    </dev/null >"${LOG}" 2>&1 &
TRAIN_PID=$!
echo "${TRAIN_PID}" >"${OUT}/train.pid"

nohup python tools/volmem/stop_training_at_limit.py \
    --pid "${TRAIN_PID}" \
    --expected-config "${CFG}" \
    --deadline "${DEADLINE}" \
    --max-step "${MAX_STEP}" \
    --train-log "${LOG}" \
    --checkpoint-dir "${OUT}/checkpoints" \
    --status-log "${OUT}/training_limit_watchdog.log" \
    --poll-seconds 20 \
    </dev/null >"${OUT}/watchdog_launcher.log" 2>&1 &
WATCHDOG_PID=$!
echo "${WATCHDOG_PID}" >"${OUT}/watchdog.pid"

printf 'train_pid=%s\nwatchdog_pid=%s\ndeadline=%s\nmax_step=%s\n' \
    "${TRAIN_PID}" "${WATCHDOG_PID}" "${DEADLINE}" "${MAX_STEP}"
