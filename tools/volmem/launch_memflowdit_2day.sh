#!/bin/bash
# MemFlowDiT 2-day formal training with automatic checkpoint management
set -e

PROJECT_ROOT="/home/medteam/Zhrch/DiffusionSnake-12-30"
cd "$PROJECT_ROOT"

CFG_FILE="configs/volmem/verse_memflowdit_v0_3_2day_gpu6.yaml"
OUTPUT_DIR="data/outputs/volmem/verse_memflowdit_v0_3_2day_gpu6"
LOG_FILE="${OUTPUT_DIR}/train_20260730_065712_gpu6.log"

# 2 days @ 9 sec/step = 19200 steps
# Save every 50 steps for frequent monitoring
# Keep recent 20 + milestone every 1000 steps
MAX_STEPS=19200
SAVE_EVERY=50
KEEP_RECENT=20
MILESTONE_EVERY=1000

echo "[Launch] Starting MemFlowDiT 2-day training" | tee -a "$LOG_FILE"
echo "[Launch] MAX_STEPS=$MAX_STEPS SAVE_EVERY=$SAVE_EVERY" | tee -a "$LOG_FILE"
echo "[Launch] Checkpoint policy: keep_recent=$KEEP_RECENT milestone_every=$MILESTONE_EVERY" | tee -a "$LOG_FILE"

# Launch training in background
nohup python tools/volmem/train_memflowdit.py   --cfg_file "$CFG_FILE"   --max_steps $MAX_STEPS   --save_every $SAVE_EVERY   --chunks_per_step 24   >> "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "[Launch] Training started: PID=$TRAIN_PID" | tee -a "$LOG_FILE"
echo $TRAIN_PID > "${OUTPUT_DIR}/train.pid"

# Checkpoint manager loop (runs every 5 minutes)
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 300  # 5 minutes
    
    # Extract current step from latest log line
    CURRENT_STEP=$(tail -1 "$LOG_FILE" | grep -oP '\[step \K[0-9]+' || echo "")
    
    if [ -n "$CURRENT_STEP" ]; then
        python tools/volmem/memflowdit_checkpoint_manager.py           "${OUTPUT_DIR}/checkpoints"           --keep_recent $KEEP_RECENT           --milestone_every $MILESTONE_EVERY           --current_step $CURRENT_STEP           >> "$LOG_FILE" 2>&1
    fi
done

echo "[Launch] Training finished: PID=$TRAIN_PID exit_code=$?" | tee -a "$LOG_FILE"
rm -f "${OUTPUT_DIR}/train.pid"
