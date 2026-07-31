#!/usr/bin/env bash
# Wait for H(abl_h_areafix) epoch_7.pt, stop H, then evaluate it on the freed GPU.
#
# H is the single-variable test of the GT-instance area floor: it differs from A0
# only by min_poly_area_output 5.0 -> 0.5. filter_tiny_polys (snake_voc_utils.py:233)
# applies `area > 5` in the 128x128 *output* space, which silently discards
# 354/1582 = 22.4% of val GT instances and empties 26/209 foreground slices.
#
# Serialized on purpose: an unrelated v4_6c training saturates /dev/sdb2, and
# running two of our jobs at once measured negative sum throughput.
set -u
cd /home/medteam/Zhrch/DiffusionSnake-12-30
PY=/home/medteam/miniconda3/envs/snake1/bin/python
CKPT=data/outputs/abl_h_areafix/checkpoints/epoch_7.pt
CFG=configs/ablation/abl_h_areafix.yaml
RES=data/outputs/abl_h_areafix/eval_gt400_ep7
LOG_PREFIX="[serial-h]"

echo "$LOG_PREFIX waiting for H epoch_7.pt ..."
for i in $(seq 1 360); do            # up to 180 min
  [ -f "$CKPT" ] && { echo "$LOG_PREFIX H ep7 ckpt at $(date +%H:%M:%S)"; break; }
  sleep 30
done
[ -f "$CKPT" ] || { echo "$LOG_PREFIX TIMEOUT waiting H ep7"; exit 1; }

sleep 30                             # let the checkpoint finish flushing

# Stop H by explicit pid. Never a pgrep pattern: a pattern once matched this
# script's own shell and killed the executor.
for pid in 2130463 2130484 2131013 2131070 2131128 2131185 2131241 2131298 2131360 2131416; do
  kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null && echo "$LOG_PREFIX killed H $pid"
done
sleep 20
for pid in 2131013 2131070 2131128 2131185 2131241 2131298 2131360 2131416; do
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
done
sleep 10

# --cfg_file must be H's own config, otherwise the strict load raises on the
# replacer shape. GPU1 is the card H just released.
mkdir -p "$RES"
echo "$LOG_PREFIX eval H ep7 start $(date +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=1 "$PY" tools/eval_sagittal_2d_fixed.py \
  --cfg_file "$CFG" --ckpt "$CKPT" \
  --box-mode gt --split val --max-slices 400 \
  --result-dir "$RES" --device cuda > "$RES/eval.log" 2>&1
echo "$LOG_PREFIX eval H ep7 done $(date +%H:%M:%S) rc=$?"
"$PY" tools/ablation/summarize_all.py
