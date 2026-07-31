#!/usr/bin/env bash
# Serial pipeline: F(ep7) -> stop F -> start G -> eval F ep7.
#
# Disk is the hard bottleneck: an unrelated v4_6c training (19 procs, another
# session) already saturates /dev/sdb2 at ~30% iowait. Running F and G together
# measured NEGATIVE sum throughput (F alone 1.45 step/s; F+G 0.33+0.50=0.83),
# because 16 dataloader workers scatter random reads over the same npz cache.
# So we serialize: one training + one eval at a time.
#
# F ep11 is skipped for the same reason B ep11 was: A0 peaks at ep7 and drops
# 0.0106 by ep11, so ep11 is a low-value data point.
set -u
cd /home/medteam/Zhrch/DiffusionSnake-12-30
PY=/home/medteam/miniconda3/envs/snake1/bin/python
F_CKPT=data/outputs/abl_f_u2_dual_gridfix/checkpoints/epoch_7.pt
LOG_PREFIX="[serial]"

echo "$LOG_PREFIX waiting for F epoch_7.pt ..."
for i in $(seq 1 240); do          # up to 120 min
  [ -f "$F_CKPT" ] && { echo "$LOG_PREFIX F ep7 ckpt at $(date +%H:%M:%S)"; break; }
  sleep 30
done
[ -f "$F_CKPT" ] || { echo "$LOG_PREFIX TIMEOUT waiting F ep7"; exit 1; }

sleep 30                            # let the checkpoint finish flushing

# Stop F by explicit pid (never a pgrep pattern: a pattern once matched this
# script's own shell and killed the executor).
for pid in 2067460 2067564 2068751 2068807 2068864 2068920; do
  kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null && echo "$LOG_PREFIX killed F $pid"
done
sleep 20
for pid in 2068751 2068807 2068864 2068920; do
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
done
sleep 10

# G gets the freed IO. train_net_ddp.py must be launched through torchrun,
# otherwise dist.init_process_group raises on the missing RANK env var.
mkdir -p data/outputs/abl_g_u2_dual_lnorm
GLOG="data/outputs/abl_g_u2_dual_lnorm/train_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES=4 nohup "$PY" -m torch.distributed.run \
  --master_addr=127.0.0.1 --master_port=29951 --nnodes=1 --nproc_per_node=1 \
  train_net_ddp.py --cfg_file configs/ablation/abl_g_u2_dual_lnorm.yaml \
  > "$GLOG" 2>&1 &
echo "$LOG_PREFIX G started pid $! -> $GLOG"
sleep 60

# Eval F ep7 on GPU0. --cfg_file must be F's own config or the strict load
# raises on the replacer shape.
RES=data/outputs/abl_f_u2_dual_gridfix/eval_gt400_ep7
if [ -f "$RES/summary.json" ]; then
  echo "$LOG_PREFIX F ep7 eval already done"
else
  mkdir -p "$RES"
  echo "$LOG_PREFIX eval F ep7 start $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=0 "$PY" tools/eval_sagittal_2d_fixed.py \
    --cfg_file configs/ablation/abl_f_u2_dual_gridfix.yaml \
    --ckpt "$F_CKPT" --box-mode gt --split val --max-slices 400 \
    --result-dir "$RES" --device cuda > "$RES/eval.log" 2>&1
  echo "$LOG_PREFIX F ep7 eval done $(date +%H:%M:%S)"
fi
"$PY" tools/ablation/summarize_all.py
