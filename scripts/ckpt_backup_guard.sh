#!/bin/bash
# Persistent checkpoint backup guard. Copies the latest checkpoint of each
# actively-running RL training to ckpt_backup_safe/ every 30s, so an external
# wipe of the training output dir does not lose progress irrecoverably.
set -u
BACKUP_DIR="/home/medteam/Zhrch/ckpt_backup_safe"
mkdir -p "$BACKUP_DIR"

declare -A SRC=(
  [gpu4_curvmatch]="/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/1232_final_v5_geom8_curvmatch_bs6_gpu4/checkpoints/latest.pt"
  [gpu5_perpoint]="/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/1232_final_v5_perpoint_fmscale_bs6_gpu5/checkpoints/latest.pt"
  [gpu6_extrap1p0]="/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/1232_final_v5_geom8_extrap1p0_bs6_gpu6/checkpoints/latest.pt"
  [gpu7_delta_nsd]="/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/1232_final_v5_geom8_delta_nsd_bs6_gpu7/checkpoints/latest.pt"
)

echo "[ckpt_backup_guard] started at $(date)" >> "$BACKUP_DIR/guard.log"
while true; do
  for name in "${!SRC[@]}"; do
    src="${SRC[$name]}"
    if [ -f "$src" ]; then
      dst="$BACKUP_DIR/${name}_latest_backup.pt"
      tmp="$dst.tmp"
      cp -f "$src" "$tmp" 2>>"$BACKUP_DIR/guard.log" && mv -f "$tmp" "$dst"
    fi
  done
  sleep 30
done
