#!/usr/bin/env bash
# Evaluate MoonViT point-feature ablation arms on one fixed val subset with GT boxes.
#
# Usage:
#   bash tools/ablation/eval_arms.sh <gpu> <max_slices> <arm> [arm ...]
#   bash tools/ablation/eval_arms.sh 5 400 abl_a0_u2_single abl_b_u2_dual
#
# Each arm reads configs/ablation/<arm>.yaml and data/outputs/<arm>/checkpoints/latest.pt,
# and writes data/outputs/<arm>/eval_gt_<max_slices>/summary.json.
set -uo pipefail

PY=/home/medteam/miniconda3/envs/snake1/bin/python
ROOT=/home/medteam/Zhrch/DiffusionSnake-12-30
cd "$ROOT" || exit 1

GPU=${1:?gpu index required}
MAX=${2:?max_slices required}
shift 2
[ "$#" -ge 1 ] || { echo "no arms given" >&2; exit 1; }

for arm in "$@"; do
  cfg="configs/ablation/${arm}.yaml"
  ckpt="data/outputs/${arm}/checkpoints/latest.pt"
  out="data/outputs/${arm}/eval_gt_${MAX}"
  if [ ! -f "$cfg" ]; then echo "[skip] missing cfg $cfg"; continue; fi
  if [ ! -f "$ckpt" ]; then echo "[skip] missing ckpt $ckpt"; continue; fi
  mkdir -p "$out"
  echo "=== $arm -> $out"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" tools/eval_sagittal_2d_fixed.py \
    --cfg_file "$cfg" \
    --ckpt "$ckpt" \
    --box-mode gt \
    --split val \
    --max-slices "$MAX" \
    --result-dir "$out" \
    --device cuda > "$out/eval.log" 2>&1
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "[fail] $arm exit=$status; tail:"; tail -5 "$out/eval.log"
  else
    "$PY" - "$out/summary.json" <<'EOF'
import json, sys
with open(sys.argv[1]) as fh:
    s = json.load(fh)
keys = ('foreground_slice_mean_dice', 'foreground_slice_mean_iou',
        'all_slice_mean_dice', 'all_slice_mean_iou',
        'class_mean_dice', 'class_mean_iou',
        'num_foreground_slices', 'num_foreground_slices_with_predictions')
print('  ' + '  '.join('{}={}'.format(k, round(s[k], 4) if isinstance(s.get(k), float) else s.get(k)) for k in keys))
EOF
  fi
done
