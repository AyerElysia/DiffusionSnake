#!/bin/bash
# Wait for arm C's training process to exit (frees physical GPU0), then start arm D there.
cd /home/medteam/Zhrch/DiffusionSnake-12-30
PY=/home/medteam/miniconda3/envs/snake1/bin/python

# C's trainer pid
CPID=2018993
for i in $(seq 1 240); do
  kill -0 $CPID 2>/dev/null || break
  sleep 15
done
if kill -0 $CPID 2>/dev/null; then echo "TIMEOUT: C still running"; exit 1; fi
echo "C exited, waiting 45s for GPU0 to drain"
sleep 45

LOG=data/outputs/abl_d_u4_dual/train_$(date +%Y%m%d_%H%M%S).log
mkdir -p data/outputs/abl_d_u4_dual
nohup env CUDA_VISIBLE_DEVICES=0 $PY -m torch.distributed.run \
  --master_addr=127.0.0.1 --master_port=29907 --nnodes=1 --nproc_per_node=1 \
  train_net_ddp.py --cfg_file configs/ablation/abl_d_u4_dual.yaml \
  > $LOG 2>&1 &
echo "D launched pid $! log $LOG"
