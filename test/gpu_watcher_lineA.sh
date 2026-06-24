#!/bin/bash
# GPU-watcher: waits for GPU0/1 to free up, then auto-restarts Line A (MoonViT bs4).
# Reason: Line A was squeezed out by user's locany finetune (PID on GPU0,1).
# We do NOT kill locany; we wait for it to finish, then resume Line A from ep8 ckpt.

cd /home/medteam/Zhrch/DiffusionSnake-12-30
SNAKE_PY=/home/medteam/miniconda3/envs/snake1/bin/python
CFG_A=configs/1232_final_diffusion_dit_v4_6c_geom_bridge_lineA_moonvit_gpu0.yaml
LINEA_DIR=data/outputs/1232_final_diffusion_dit_v4_6c_geom_bridge_lineA_moonvit_gpu0
WATCH_LOG=test/gpu_watcher_lineA.log
INTERVAL=${1:-300}   # check every 5 min

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$WATCH_LOG"; }

log "=== GPU-watcher started: waiting for GPU0/1 to free up for Line A ==="
log "Will resume Line A (bs4+accum2+lr5e-5) from ep8 ckpt once GPU0 has >=30GB free."

LAUNCHED=0
while [ "$LAUNCHED" -eq 0 ]; do
    # free memory on GPU0 and GPU1
    FREE0=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    FREE1=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    # locany PIDs still alive?
    LOCANY=$(ps -ef | grep locany_finetune_magi_stream | grep -v grep | wc -l)

    log "probe: GPU0_free=${FREE0}MiB GPU1_free=${FREE1}MiB locany_procs=$LOCANY"

    # require GPU0 >= 30GB free AND locany gone (avoid racing with a half-finished locany)
    if [ "${FREE0:-0}" -ge 30000 ] && [ "${LOCANY:-0}" -eq 0 ]; then
        log "GPU0 free=${FREE0}MiB and locany gone. Launching Line A resume..."
        LOG_A="$LINEA_DIR/train_lr5e5_bs4_accum2_autoresume_$(date +%Y%m%d_%H%M).log"
        CUDA_VISIBLE_DEVICES=0 \
        PYTHONUNBUFFERED=1 \
        PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128:expandable_segments:True \
        CFG_FILE=$CFG_A \
        nohup $SNAKE_PY -u diffusion_train.py > "$LOG_A" 2>&1 &
        APID=$!
        log "Line A launched, PID=$APID, log=$LOG_A"
        sleep 45
        # verify it's actually running (passed first step)
        APROCS=$(ps -ef | grep diffusion_train.py | grep -v grep | while read -r line; do
            pid=$(echo "$line" | awk '{print $2}')
            grep -q "CFG_FILE=$CFG_A" /proc/$pid/environ 2>/dev/null && echo "$pid"
        done | wc -l)
        if [ "${APROCS:-0}" -ge 1 ]; then
            log "Line A confirmed running (procs=$APROCS). Watcher exiting."
            LAUNCHED=1
        else
            log "ALERT: Line A launched but died within 45s. Check $LOG_A. Watcher will retry next cycle."
            tail -20 "$LOG_A" 2>/dev/null | tee -a "$WATCH_LOG"
        fi
    else
        log "not yet: GPU0_free=${FREE0}MiB (<30000) or locany still running ($LOCANY). retry in ${INTERVAL}s."
    fi
    sleep "$INTERVAL"
done
log "=== GPU-watcher done ==="
