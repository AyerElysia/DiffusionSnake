#!/usr/bin/env bash
# Evaluate one MoonViT point-feature ablation arm on the val split with GT boxes.
#
# Usage: eval_ablation_arm.sh <arm> <gpu> [ckpt]
#   arm   basename of configs/ablation/<arm>.yaml
#   gpu   physical GPU index to run inference on
#   ckpt  optional checkpoint path (default: data/outputs/<arm>/checkpoints/latest.pt)
#
# The headline numbers are foreground_slice_mean_dice / foreground_slice_mean_iou,
# i.e. slices that actually contain a vertebra. all_slice_mean_* counts empty
# slices as perfect and must not be compared against the 0.601/0.573 baseline.
set -euo pipefail

ARM="${1:?arm name required}"
GPU="${2:?gpu index required}"

cd /home/medteam/Zhrch/DiffusionSnake-12-30

CFG="configs/ablation/${ARM}.yaml"
[[ -f "$CFG" ]] || { echo "missing config: $CFG" >&2; exit 1; }

CKPT="${3:-data/outputs/${ARM}/checkpoints/latest.pt}"
[[ -f "$CKPT" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 1; }

PY=/home/medteam/miniconda3/envs/snake1/bin/python
OUT="data/outputs/${ARM}_eval_gt"
mkdir -p "$OUT"
SUMMARY="${OUT}/summary.json"

echo "[eval] arm=${ARM} gpu=${GPU} ckpt=${CKPT}"

CUDA_VISIBLE_DEVICES="${GPU}" "$PY" tools/eval_sagittal_2d_fixed.py \
    --cfg_file "$CFG" \
    --ckpt "$CKPT" \
    --box-mode gt \
    --split val \
    --result-dir "$OUT" \
    --device cuda \
    | tee "${OUT}/eval_stdout.log" \
    | tail -1 > "$SUMMARY"

"$PY" - "$ARM" "$SUMMARY" <<'PYEOF'
import json
import sys

arm, path = sys.argv[1], sys.argv[2]
with open(path) as handle:
    summary = json.load(handle)
print('[eval] arm={} fg_dice={:.4f} fg_iou={:.4f} fg_slices={} all_dice={:.4f}'.format(
    arm,
    summary['foreground_slice_mean_dice'],
    summary['foreground_slice_mean_iou'],
    summary['num_foreground_slices'],
    summary['all_slice_mean_dice'],
))
PYEOF
