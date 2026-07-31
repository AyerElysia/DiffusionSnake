#!/bin/bash
# Phoenix watcher for Line D: auto-restart on death when GPU0 is free.
# Survives repeated GPU preemption. save_ep=1 caps loss at 1 epoch per death.

cd /home/medteam/Zhrch/DiffusionSnake-12-30
SNAKE_PY=/home/medteam/miniconda3/envs/snake1/bin/python
CFG_A=configs/1232_final_diffusion_dit_v4_6c_geom_bridge_lineD_schedon_gpu7.yaml
LINEA_DIR=data/outputs/1232_final_diffusion_dit_v4_6c_geom_bridge_lineD_schedon_gpu7
PHOENIX_LOG=test/phoenix_lineD.log
INTERVAL=${1:-180}
GPU=${2:-0}

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$PHOENIX_LOG"; }

lineA_alive() {
    ps -ef | grep diffusion_train.py | grep -v grep | while read -r line; do
        pid=$(echo "$line" | awk '{print $2}')
        grep -q "CFG_FILE=$CFG_A" /proc/$pid/environ 2>/dev/null && echo "$pid"
    done | grep -q .
}

gpu_free() {
    local need=${1:-20000}
    local free=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ "${free:-0}" -ge "$need" ]
}

launch() {
    local LOG_A="$LINEA_DIR/train_pureFM_phoenix_$(date +%Y%m%d_%H%M).log"
    CUDA_VISIBLE_DEVICES=$GPU \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128:expandable_segments:True:garbage_collection_threshold:0.6 \
    CFG_FILE=$CFG_A \
    nohup $SNAKE_PY -u diffusion_train.py > "$LOG_A" 2>&1 &
    local pid=$!
    log "LAUNCHED Line D PID=$pid GPU$GPU log=$LOG_A"
    sleep 45
    if lineA_alive; then
        log "CONFIRMED running."
    else
        log "ALERT: died within 45s. Will retry next cycle. Tail:"
        tail -10 "$LOG_A" 2>/dev/null | tee -a "$PHOENIX_LOG"
    fi
}

log "=== Phoenix watcher started: GPU$GPU, interval=${INTERVAL}s ==="
log "Will auto-restart Line D (pure FM, bs3+accum3) whenever it dies AND GPU$GPU has >=20GB free."

while true; do
    if lineA_alive; then
        : # healthy, do nothing
    else
        LAST_EP=$(tail -1 "$LINEA_DIR/logs.jsonl" 2>/dev/null | python3 -c "import sys,json
try: d=json.loads(sys.stdin.read()); print(d.get('epoch',-1))
except: print(-1)" 2>/dev/null)
        if gpu_free 20000; then
            log "Line D DEAD (last ep=$LAST_EP). GPU$GPU free. Restarting..."
            launch
        else
            log "Line D DEAD (last ep=$LAST_EP) but GPU$GPU not free, waiting..."
        fi
    fi
    sleep "$INTERVAL"
done
