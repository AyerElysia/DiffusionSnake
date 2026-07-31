#!/usr/bin/env bash
# Start arm I only after H's ep7 eval is done, so the two never share the disk.
#
# Disk is the hard bottleneck here: an unrelated v4_6c training (19 procs, other
# session) already saturates /dev/sdb2. Two of our jobs at once measured a
# NEGATIVE sum throughput earlier (1.45 step/s alone vs 0.33+0.50 together),
# because the dataloader workers scatter random reads across the npz cache.
set -u
cd /home/medteam/Zhrch/DiffusionSnake-12-30
PY=/home/medteam/miniconda3/envs/snake1/bin/python
H_EVAL=data/outputs/abl_h_areafix/eval_gt400_ep7/summary.json

echo "[serial-i] waiting for H ep7 eval summary ..."
for i in $(seq 1 300); do          # up to 150 min
  [ -f "$H_EVAL" ] && { echo "[serial-i] H ep7 eval done at $(date +%H:%M:%S)"; break; }
  sleep 30
done
[ -f "$H_EVAL" ] || { echo "[serial-i] TIMEOUT waiting H ep7 eval"; exit 1; }

sleep 20
mkdir -p data/outputs/abl_i_areafix_gridfix
ILOG="data/outputs/abl_i_areafix_gridfix/train_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES=7 nohup "$PY" -m torch.distributed.run \
  --master_addr=127.0.0.1 --master_port=29971 --nnodes=1 --nproc_per_node=1 \
  train_net_ddp.py --cfg_file configs/ablation/abl_i_areafix_gridfix.yaml \
  > "$ILOG" 2>&1 &
echo "[serial-i] I started pid $! -> $ILOG"
