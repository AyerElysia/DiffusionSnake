#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/medteam/Zhrch/DiffusionSnake-12-30"
cd "$ROOT"

source /home/medteam/miniconda3/etc/profile.d/conda.sh
conda activate snake1

RUNS=(
  "btcv_ablate_v41_02_no_delta"
  "btcv_ablate_v41_03_no_curv"
  "btcv_ablate_v41_04_no_small_disp"
  "btcv_ablate_v41_05_delta_only"
  "btcv_ablate_v41_06_curv_only"
  "btcv_ablate_v41_07_small_only"
)

TARGET_STEP=69000
EVAL_ROOT="$ROOT/data/outputs/v4_1_mechanism_ablation/eval_final"
mkdir -p "$EVAL_ROOT"

is_done() {
  local stem="$1"
  local log="$ROOT/data/outputs/$stem/logs.jsonl"
  local ckpt="$ROOT/data/outputs/$stem/checkpoints/latest.pt"
  if [[ ! -f "$log" || ! -f "$ckpt" ]]; then
    return 1
  fi
  python - "$log" "$TARGET_STEP" <<'PY'
import json
import sys

log_path = sys.argv[1]
target = int(sys.argv[2])
last = None
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            last = json.loads(line)
        except json.JSONDecodeError:
            pass
step = int(last.get("step", -1)) if isinstance(last, dict) else -1
sys.exit(0 if step >= target else 1)
PY
}

while true; do
  all_done=1
  for stem in "${RUNS[@]}"; do
    if ! is_done "$stem"; then
      all_done=0
      break
    fi
  done
  if [[ "$all_done" -eq 1 ]]; then
    break
  fi
  sleep 300
done

for stem in "${RUNS[@]}"; do
  cfg="$ROOT/configs/$stem.yaml"
  save_dir="$EVAL_ROOT/$stem"
  mkdir -p "$save_dir"
  CFG_FILE="$cfg" SAVE_VISUALS=0 SAVE_DIR="$save_dir" python scripts/eval_v37_full_iou.py \
    > "$save_dir/eval.log" 2>&1
done

python scripts/summarize_v41_ablation.py > "$EVAL_ROOT/summary.md"
