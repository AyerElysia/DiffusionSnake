#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/medteam/Zhrch/DiffusionSnake-12-30"
CFG_FILE="configs/volmem/verse_memflowdit_v0_5_minimal_gpu6.yaml"
OUTPUT_DIR="data/outputs/volmem/verse_memflowdit_v0_5_minimal_gpu6"
LOG_FILE="${OUTPUT_DIR}/train_20260731_gpu6.log"
# Two-day-budget default. The active 2026-07-31 run is additionally guarded
# by stop_training_at_limit.py, so wall-clock time remains the hard limit.
MAX_STEPS="${MAX_STEPS:-6800}"
SAVE_EVERY=100
CHUNKS_PER_STEP=12
KEEP_RECENT=20
MILESTONE_EVERY=1000

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

echo "[Launch] MemFlowDiT v0.5 minimal training" | tee -a "$LOG_FILE"
echo "[Launch] max_steps=$MAX_STEPS chunks=$CHUNKS_PER_STEP save_every=$SAVE_EVERY" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=6 nohup setsid /usr/bin/python3.8 tools/volmem/train_memflowdit.py \
  --cfg_file "$CFG_FILE" \
  --device cuda:0 \
  --max_steps "$MAX_STEPS" \
  --save_every "$SAVE_EVERY" \
  --chunks_per_step "$CHUNKS_PER_STEP" \
  >> "$LOG_FILE" 2>&1 < /dev/null &

TRAIN_PID=$!
echo "$TRAIN_PID" > "${OUTPUT_DIR}/train.pid"
echo "[Launch] training_pid=$TRAIN_PID" | tee -a "$LOG_FILE"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 300
  CURRENT_STEP="$(
    grep -oP '\[step \K[0-9]+' "$LOG_FILE" | tail -1 || true
  )"
  if [ -n "$CURRENT_STEP" ]; then
    /usr/bin/python3.8 tools/volmem/memflowdit_checkpoint_manager.py \
      "${OUTPUT_DIR}/checkpoints" \
      --keep_recent "$KEEP_RECENT" \
      --milestone_every "$MILESTONE_EVERY" \
      --current_step "$CURRENT_STEP" \
      >> "$LOG_FILE" 2>&1
  fi
done

set +e
wait "$TRAIN_PID"
STATUS=$?
set -e
echo "[Launch] training exited pid=$TRAIN_PID status=$STATUS" | tee -a "$LOG_FILE"
rm -f "${OUTPUT_DIR}/train.pid"
exit "$STATUS"
