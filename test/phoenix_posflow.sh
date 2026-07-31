#!/bin/bash
cd /home/medteam/Zhrch/DiffusionSnake-12-30
SNAKE_PY=/home/medteam/miniconda3/envs/snake1/bin/python
CFG=configs/posflow_verify_gpu0.yaml
DIR=data/outputs/posflow_verify_gpu0
LOG=test/phoenix_posflow.log
GPU=0
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }
alive() { ps -ef | grep diffusion_train.py | grep -v grep | while read -r l; do p=$(echo "$l"|awk '{print $2}'); grep -q "posflow_verify" /proc/$p/environ 2>/dev/null && echo 1; done | grep -q .; }
log "=== Phoenix posflow GPU$GPU started ==="
while true; do
  if alive; then :; else
    L="$DIR/train_phoenix_$(date +%m%d_%H%M).log"
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128:expandable_segments:True:garbage_collection_threshold:0.6 \
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 CFG_FILE=$CFG \
    nohup $SNAKE_PY -u diffusion_train.py > "$L" 2>&1 &
    log "LAUNCHED PID=$! log=$L"
    sleep 45; alive && log CONFIRMED || log "ALERT died 45s"
  fi
  sleep 180
done
