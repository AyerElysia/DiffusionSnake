#!/usr/bin/env bash
# Launch one MoonViT point-feature ablation arm on a single dedicated GPU.
#
# Usage: run_ablation_arm.sh <arm> <gpu> <master_port>
#   arm         basename of configs/ablation/<arm>.yaml
#   gpu         physical GPU index (must be otherwise idle)
#   master_port unique per concurrent run
#
# Each arm resets locate_feat_replacer (its shape changes between arms) and
# warm-starts every other weight from the shared snapshot, so the arms differ
# only in the point-feature front end.
set -euo pipefail

ARM="${1:?arm name required}"
GPU="${2:?gpu index required}"
PORT="${3:?master port required}"

cd /home/medteam/Zhrch/DiffusionSnake-12-30

CFG="configs/ablation/${ARM}.yaml"
[[ -f "$CFG" ]] || { echo "missing config: $CFG" >&2; exit 1; }

PY=/home/medteam/miniconda3/envs/snake1/bin/python
LOG_DIR="data/outputs/${ARM}"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "[launch] arm=${ARM} gpu=${GPU} port=${PORT} log=${LOG}"

CUDA_VISIBLE_DEVICES="${GPU}" nohup "$PY" -m torch.distributed.run \
    --master_addr=127.0.0.1 --master_port="${PORT}" \
    --nnodes=1 --nproc_per_node=1 \
    train_net_ddp.py --cfg_file "$CFG" \
    > "$LOG" 2>&1 &

echo "[launch] pid=$! log=${LOG}"
