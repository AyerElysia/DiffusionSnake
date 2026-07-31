#!/bin/bash
# Restart arm F (dual-layer u2 + gridfix) once the GPU0 eval queue has finished.
#
# F was stopped mid-epoch-0 because four concurrent trainings starved the disk
# (iowait ~40%, dataloader workers stuck in D state) and the two arms that were
# nearly done (B, C) needed the bandwidth. Nothing is lost: F had only reached
# step 20, and its output dir is reused so the run simply starts over.
set -u
cd /home/medteam/Zhrch/DiffusionSnake-12-30
PY=/home/medteam/miniconda3/envs/snake1/bin/python
QUEUE_PID=2054621

echo "[relaunch-F] waiting for eval queue pid $QUEUE_PID"
for i in $(seq 1 400); do
  ps -p $QUEUE_PID > /dev/null 2>&1 || break
  sleep 15
done
if ps -p $QUEUE_PID > /dev/null 2>&1; then
  echo "[relaunch-F] queue still alive after 100min, abort"
  exit 1
fi
echo "[relaunch-F] queue finished"

# Pick a card with < 2 GB in use. GPU 2/3/5/6 belong to other people's jobs, so
# only the cards this ablation has been using are considered.
GPU=""
for i in $(seq 1 40); do
  for cand in 7 1 0 4; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $cand 2>/dev/null | tr -d ' ')
    [ -z "$used" ] && continue
    if [ "$used" -lt 2000 ]; then GPU=$cand; break; fi
  done
  [ -n "$GPU" ] && break
  sleep 30
done
if [ -z "$GPU" ]; then echo "[relaunch-F] no free GPU found"; exit 1; fi
echo "[relaunch-F] using GPU $GPU"

LOG=data/outputs/abl_f_u2_dual_gridfix/train_restart_$(date +%Y%m%d_%H%M%S).log
mkdir -p data/outputs/abl_f_u2_dual_gridfix
nohup env CUDA_VISIBLE_DEVICES=$GPU $PY \
  -m torch.distributed.run --master_addr=127.0.0.1 --master_port=29911 \
  --nnodes=1 --nproc_per_node=1 train_net_ddp.py \
  --cfg_file configs/ablation/abl_f_u2_dual_gridfix.yaml \
  > "$LOG" 2>&1 &
echo "[relaunch-F] launched pid $! log $LOG"
