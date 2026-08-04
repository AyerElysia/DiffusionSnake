#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/medteam/Zhrch/DiffusionSnake-12-30"
CFG_FILE="configs/volmem/verse_memflowdit_v0_6_moe2026_combined_gpu7.yaml"
OUTPUT_DIR="data/outputs/volmem/verse_memflowdit_v0_6_moe2026_combined_gpu7"
LOG_FILE="${OUTPUT_DIR}/train_2day_gpu7.log"
QUEUE_LOG="${OUTPUT_DIR}/idle_queue_gpu7.log"
PHYSICAL_GPU=7
MAX_STEPS="${MAX_STEPS:-6800}"
MAX_WALL_MINUTES="${MAX_WALL_MINUTES:-2870}"
SAVE_EVERY=100
CHUNKS_PER_STEP=12
KEEP_RECENT=20
MILESTONE_EVERY=1000
IDLE_MEMORY_MIB=512
IDLE_UTIL_PERCENT=5
IDLE_CONFIRMATIONS=3
IDLE_POLL_SECONDS=30

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"
echo "$$" > "${OUTPUT_DIR}/launcher.pid"

timestamp() {
  date '+%F %T %z'
}

log_queue() {
  echo "[$(timestamp)] $*" | tee -a "$QUEUE_LOG"
}

if ps -eo args | grep -F "tools/volmem/train_memflowdit.py" \
  | grep -F "$CFG_FILE" | grep -v grep >/dev/null; then
  log_queue "refusing duplicate v0.6 training process"
  exit 2
fi

log_queue "waiting for physical GPU${PHYSICAL_GPU} to become genuinely idle"
idle_count=0
while [ "$idle_count" -lt "$IDLE_CONFIRMATIONS" ]; do
  IFS=',' read -r used_mib util_percent <<<"$(
    nvidia-smi -i "$PHYSICAL_GPU" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )"
  used_mib="${used_mib//[[:space:]]/}"
  util_percent="${util_percent//[[:space:]]/}"
  compute_pids="$(
    nvidia-smi -i "$PHYSICAL_GPU" \
      --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null \
      | sed '/^[[:space:]]*$/d' || true
  )"
  if (
    [ "$used_mib" -le "$IDLE_MEMORY_MIB" ] &&
    [ "$util_percent" -le "$IDLE_UTIL_PERCENT" ] &&
    [ -z "$compute_pids" ]
  ); then
    idle_count=$((idle_count + 1))
    log_queue "idle confirmation ${idle_count}/${IDLE_CONFIRMATIONS}: memory=${used_mib}MiB util=${util_percent}%"
  else
    if [ "$idle_count" -ne 0 ]; then
      log_queue "idle confirmation reset: memory=${used_mib}MiB util=${util_percent}% pids=${compute_pids:-none}"
    fi
    idle_count=0
  fi
  if [ "$idle_count" -lt "$IDLE_CONFIRMATIONS" ]; then
    sleep "$IDLE_POLL_SECONDS"
  fi
done

DEADLINE="$(
  date -d "+${MAX_WALL_MINUTES} minutes" --iso-8601=seconds
)"
log_queue "launching v0.6 on GPU${PHYSICAL_GPU}: max_steps=${MAX_STEPS} deadline=${DEADLINE}"
echo "[Launch] MemFlowDiT v0.6 combined two-day training" | tee -a "$LOG_FILE"
echo "[Launch] max_steps=$MAX_STEPS chunks=$CHUNKS_PER_STEP save_every=$SAVE_EVERY deadline=$DEADLINE" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" nohup setsid /usr/bin/python3.8 \
  tools/volmem/train_memflowdit.py \
  --cfg_file "$CFG_FILE" \
  --device cuda:0 \
  --max_steps "$MAX_STEPS" \
  --save_every "$SAVE_EVERY" \
  --chunks_per_step "$CHUNKS_PER_STEP" \
  >> "$LOG_FILE" 2>&1 < /dev/null &

TRAIN_PID=$!
echo "$TRAIN_PID" > "${OUTPUT_DIR}/train.pid"
log_queue "training_pid=${TRAIN_PID}"

nohup setsid /usr/bin/python3.8 tools/volmem/stop_training_at_limit.py \
  --pid "$TRAIN_PID" \
  --expected-config "$CFG_FILE" \
  --deadline "$DEADLINE" \
  --max-step "$MAX_STEPS" \
  --train-log "$LOG_FILE" \
  --checkpoint-dir "${OUTPUT_DIR}/checkpoints" \
  --status-log "${OUTPUT_DIR}/training_limit_watchdog.log" \
  --poll-seconds 20 \
  > "${OUTPUT_DIR}/training_limit_watchdog.nohup.log" 2>&1 < /dev/null &
WATCHDOG_PID=$!
echo "$WATCHDOG_PID" > "${OUTPUT_DIR}/training_limit_watchdog.pid"
log_queue "watchdog_pid=${WATCHDOG_PID}"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 300
  CURRENT_STEP="$(
    grep -oP '\[step \K[0-9]+' "$LOG_FILE" | tail -1 || true
  )"
  if [ -n "$CURRENT_STEP" ] && [ -d "${OUTPUT_DIR}/checkpoints" ]; then
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
log_queue "training exited pid=${TRAIN_PID} status=${STATUS}"
rm -f "${OUTPUT_DIR}/train.pid"
exit "$STATUS"
