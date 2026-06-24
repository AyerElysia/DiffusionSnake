#!/bin/bash
# Watcher: when Line A reaches ep15 (pure FM ablate), auto-trigger eval on GPU7.
# Eval is bs1 (~7GB), training uses ~27GB on GPU7, together ~34GB < 48GB, OK to share.

cd /home/medteam/Zhrch/DiffusionSnake-12-30
SNAKE_PY=/home/medteam/miniconda3/envs/snake1/bin/python
CFG_A=configs/1232_final_diffusion_dit_v4_6c_geom_bridge_lineA_moonvit_gpu0.yaml
LINEA_DIR=data/outputs/1232_final_diffusion_dit_v4_6c_geom_bridge_lineA_moonvit_gpu0
WATCH_LOG=test/lineA_ep15_eval_watcher.log
TARGET_EP=${1:-15}

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$WATCH_LOG"; }

log "=== LineA ep${TARGET_EP} eval-watcher started ==="

FIRED=0
while [ "$FIRED" -eq 0 ]; do
    # read current epoch from logs.jsonl
    CUR_EP=$(tail -1 "$LINEA_DIR/logs.jsonl" 2>/dev/null | python3 -c "import sys,json
try:
    d=json.loads(sys.stdin.read()); print(d.get('epoch',-1))
except: print(-1)" 2>/dev/null)
    log "probe: LineA current epoch=${CUR_EP} (target=${TARGET_EP})"

    if [ "${CUR_EP:-0}" -ge "$TARGET_EP" ]; then
        # confirm a checkpoint at >= target epoch exists
        CKPT="$LINEA_DIR/checkpoints/latest.pt"
        if [ -f "$CKPT" ]; then
            log "LineA reached ep${CUR_EP}. Triggering eval on GPU7 (shares with training)..."
            EVAL_LOG="$LINEA_DIR/eval_pureFM_ep${CUR_EP}_$(date +%Y%m%d_%H%M).log"
            CUDA_VISIBLE_DEVICES=7 \
            EVAL_GPU=7 \
            CFG_FILE=$CFG_A \
            nohup $SNAKE_PY -u scripts/eval_v37_full_iou.py > "$EVAL_LOG" 2>&1 &
            EPID=$!
            log "eval PID=$EPID, log=$EVAL_LOG"
            # wait for eval to finish (poll for result)
            for i in $(seq 1 80); do
                if grep -q "mean_iou_sample_avg\|Saved summary" "$EVAL_LOG" 2>/dev/null; then
                    log "=== EVAL DONE ==="
                    grep -iE "mean_iou|mean_dice|Saved summary" "$EVAL_LOG" | tee -a test/lineA_pureFM_eval_result.log
                    break
                fi
                if ! ps -p $EPID > /dev/null 2>&1; then
                    log "eval process exited"
                    tail -15 "$EVAL_LOG" | tee -a test/lineA_pureFM_eval_result.log
                    break
                fi
                sleep 30
            done
            FIRED=1
        else
            log "reached target but latest.pt missing, retry next cycle"
        fi
    fi
    sleep 180
done
log "=== watcher done ==="
