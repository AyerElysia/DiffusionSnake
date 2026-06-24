#!/bin/bash
# Health sentinel for Line A (MoonViT) + Line B (YOLO) geom-bridge lr=5e-5 runs.
# Self-loop: checks every interval, logs to sentinel.log, alerts on anomalies.
# Does NOT auto-restart (avoid flapping); records state for Claude to act on.

set -u
cd /home/medteam/Zhrch/DiffusionSnake-12-30
SNAKE_PY=/home/medteam/miniconda3/envs/snake1/bin/python
SENT_LOG=test/sentinel_lr5e5.log
INTERVAL=${1:-120}   # seconds between checks (default 120s)

LINEA_CFG=configs/1232_final_diffusion_dit_v4_6c_geom_bridge_lineA_moonvit_gpu0.yaml
LINEB_CFG=configs/1232_final_diffusion_dit_v4_6c_geom_bridge_lineB_yolo_ddp3_gpu236.yaml
LINEA_DIR=data/outputs/1232_final_diffusion_dit_v4_6c_geom_bridge_lineA_moonvit_gpu0
LINEB_DIR=data/outputs/1232_final_diffusion_dit_v4_6c_geom_bridge_lineB_yolo_ddp3_gpu236

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$SENT_LOG"; }

# track last-seen step to detect stalls
LAST_A_STEP=-1
LAST_B_STEP=-1
STALL_COUNT_A=0
STALL_COUNT_B=0

log "=== Sentinel started, interval=${INTERVAL}s ==="
log "Line A: $LINEA_CFG (GPU0, bs3+accum3, lr5e-5, pure FM no sched_sampling)"
log "Line B: $LINEB_CFG (GPU2,3,6 DDP, bs16, lr5e-5 resume ep180)"

while true; do
    NOW=$(date +%s)
    # ---------- LINE A (single GPU0) ----------
    # find Line A python process by CFG_FILE env
    A_PID=$(ps -ef | grep diffusion_train.py | grep -v grep | xargs -I{} sh -c 'cat /proc/{}/environ 2>/dev/null | tr "\0" "\n" | grep -q "CFG_FILE='"$LINEA_CFG"'" && echo {}' 2>/dev/null | head -1)
    A_PROCS=$(ps -ef | grep diffusion_train.py | grep -v grep | while read -r line; do
        pid=$(echo "$line" | awk '{print $2}')
        if grep -q "CFG_FILE=$LINEA_CFG" /proc/$pid/environ 2>/dev/null; then echo "$pid"; fi
    done)
    A_NPROC=$(echo "$A_PROCS" | grep -c . 2>/dev/null)

    # Line A latest step from logs.jsonl (last line)
    A_LAST=$(tail -1 "$LINEA_DIR/logs.jsonl" 2>/dev/null)
    A_STEP=$(echo "$A_LAST" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('step',-1))" 2>/dev/null)
    A_EPOCH=$(echo "$A_LAST" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('epoch',-1))" 2>/dev/null)
    A_LOSS=$(echo "$A_LAST" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(round(d.get('loss',0),4))" 2>/dev/null)
    A_LR=$(echo "$A_LAST" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(f\"{d.get('lr',0):.2e}\")" 2>/dev/null)
    A_LOGMT=$(stat -c %Y "$LINEA_DIR/logs.jsonl" 2>/dev/null || echo 0)
    A_AGE=$(( NOW - A_LOGMT ))

    # Line A log tail for OOM/NaN/error
    A_STDLOG=$(ls -t "$LINEA_DIR"/train_lr5e5*.log 2>/dev/null | head -1)
    A_ERR=$(tail -30 "$A_STDLOG" 2>/dev/null | grep -iE "out of memory|runtimeerror|nan|traceback|cuda error|killed" | tail -3)

    # ---------- LINE B (DDP 3 GPU) ----------
    B_PROCS=$(ps -ef | grep diffusion_train.py | grep -v grep | while read -r line; do
        pid=$(echo "$line" | awk '{print $2}')
        if grep -q "CFG_FILE=$LINEB_CFG" /proc/$pid/environ 2>/dev/null; then echo "$pid"; fi
    done)
    B_NPROC=$(echo "$B_PROCS" | grep -c . 2>/dev/null)
    B_LAUNCHER=$(ps -ef | grep "torch.distributed.run.*29520" | grep -v grep | awk '{print $2}' | head -1)

    B_LAST=$(tail -1 "$LINEB_DIR/logs.jsonl" 2>/dev/null)
    B_STEP=$(echo "$B_LAST" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('step',-1))" 2>/dev/null)
    B_EPOCH=$(echo "$B_LAST" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('epoch',-1))" 2>/dev/null)
    B_LOSS=$(echo "$B_LAST" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(round(d.get('loss',0),4))" 2>/dev/null)
    B_LR=$(echo "$B_LAST" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(f\"{d.get('lr',0):.2e}\")" 2>/dev/null)
    B_LOGMT=$(stat -c %Y "$LINEB_DIR/logs.jsonl" 2>/dev/null || echo 0)
    B_AGE=$(( NOW - B_LOGMT ))

    B_STDLOG=$(ls -t "$LINEB_DIR"/train_lr5e5*.log 2>/dev/null | head -1)
    B_ERR=$(tail -30 "$B_STDLOG" 2>/dev/null | grep -iE "out of memory|runtimeerror|nan|traceback|cuda error|killed" | tail -3)

    # ---------- GPU ----------
    GPU0=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    GPU236=$(nvidia-smi -i 2,3,6 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' '|')

    # ---------- anomaly detection ----------
    STATUS="OK"
    # Line A dead?
    if [ "$A_NPROC" -lt 1 ]; then
        log "ALERT_LINEA_DEAD: procs=$A_NPROC (expect >=1). Last ep=$A_EPOCH step=$A_STEP loss=$A_LOSS. err: ${A_ERR:-none}"
        STATUS="ALERT"
    fi
    # Line A stall (log not updated > 5 min)?
    if [ -n "$A_LOGMT" ] && [ "$A_AGE" -gt 300 ]; then
        STALL_COUNT_A=$((STALL_COUNT_A+1))
        log "WARN_LINEA_STALL: log idle ${A_AGE}s (ep=$A_EPOCH step=$A_STEP). count=$STALL_COUNT_A"
        STATUS="WARN"
    else
        STALL_COUNT_A=0
    fi
    [ -n "$A_ERR" ] && { log "ERR_LINEA: $A_ERR"; STATUS="ALERT"; }
    # Line A step going backwards or stuck same step twice
    if [ -n "$A_STEP" ] && [ "$A_STEP" = "$LAST_A_STEP" ] && [ "$A_NPROC" -ge 1 ]; then
        STALL_COUNT_A=$((STALL_COUNT_A+1))
        [ "$STALL_COUNT_A" -ge 2 ] && { log "WARN_LINEA_STEP_FROZEN: step=$A_STEP unchanged. count=$STALL_COUNT_A"; STATUS="WARN"; }
    else
        STALL_COUNT_A=0
    fi
    LAST_A_STEP=$A_STEP

    # Line B dead? (expect >=3 workers + 1 launcher)
    if [ "$B_NPROC" -lt 3 ]; then
        log "ALERT_LINEB_DEAD: workers=$B_NPROC (expect 3). Last ep=$B_EPOCH step=$B_STEP loss=$B_LOSS. err: ${B_ERR:-none}"
        STATUS="ALERT"
    fi
    if [ -n "$B_LOGMT" ] && [ "$B_AGE" -gt 300 ]; then
        STALL_COUNT_B=$((STALL_COUNT_B+1))
        log "WARN_LINEB_STALL: log idle ${B_AGE}s (ep=$B_EPOCH step=$B_STEP). count=$STALL_COUNT_B"
        STATUS="WARN"
    else
        STALL_COUNT_B=0
    fi
    [ -n "$B_ERR" ] && { log "ERR_LINEB: $B_ERR"; STATUS="ALERT"; }
    if [ -n "$B_STEP" ] && [ "$B_STEP" = "$LAST_B_STEP" ] && [ "$B_NPROC" -ge 3 ]; then
        STALL_COUNT_B=$((STALL_COUNT_B+1))
        [ "$STALL_COUNT_B" -ge 2 ] && { log "WARN_LINEB_STEP_FROZEN: step=$B_STEP unchanged. count=$STALL_COUNT_B"; STATUS="WARN"; }
    else
        STALL_COUNT_B=0
    fi
    LAST_B_STEP=$B_STEP

    # ---------- periodic heartbeat (every loop) ----------
    log "HB A{ep=$A_EPOCH st=$A_STEP loss=$A_LOSS lr=$A_LR procs=$A_NPROC gpu0=${GPU0:-?} age=${A_AGE}s} B{ep=$B_EPOCH st=$B_STEP loss=$B_LOSS lr=$B_LR procs=$B_NPROC gpus=${GPU236:-?} age=${B_AGE}s} $STATUS"

    sleep "$INTERVAL"
done
