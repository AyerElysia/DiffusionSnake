#!/usr/bin/env bash
# Serialize G training behind B: B occupies GPU4 until epoch 11 finishes.
# Disk IO is the bottleneck (sdb2 96% full, iowait ~26%), so we keep the
# concurrent load at 2 trainings + 1 eval by waiting for B to exit.
set -u
cd /home/medteam/Zhrch/DiffusionSnake-12-30
PY=/home/medteam/miniconda3/envs/snake1/bin/python
B_PID=2001042

for i in $(seq 1 240); do
  if ! kill -0 "$B_PID" 2>/dev/null; then echo "B exited at $(date +%H:%M:%S)"; break; fi
  sleep 30
done

if kill -0 "$B_PID" 2>/dev/null; then echo "TIMEOUT: B still running, not starting G"; exit 1; fi

sleep 20
mkdir -p data/outputs/abl_g_u2_dual_lnorm
LOG="data/outputs/abl_g_u2_dual_lnorm/train_$(date +%Y%m%d_%H%M%S).log"
echo "starting G -> $LOG"
CUDA_VISIBLE_DEVICES=4 nohup "$PY" -m torch.distributed.run \
  --master_addr=127.0.0.1 --master_port=29941 \
  --nnodes=1 --nproc_per_node=1 \
  train_net_ddp.py --cfg_file configs/ablation/abl_g_u2_dual_lnorm.yaml > "$LOG" 2>&1 &
echo "G pid $!"
sleep 120
tail -20 "$LOG"
