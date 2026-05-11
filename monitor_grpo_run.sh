#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k"
LOG="$OUT_DIR/train_resume_20260507_1706.log"
JSONL="$OUT_DIR/posttrain_grpo/logs.jsonl"
CKPT="$OUT_DIR/checkpoints/latest.pt"
TARGET=10000
last_step=-1
last_jsonl_mtime=0
while true; do
  pid_line=$(pgrep -af 'python -u grpo_train.py' | grep 'GRPO_TRAIN_STEPS=10000' || true)
  proc_status='missing'
  if [[ -n "$pid_line" ]]; then proc_status='running'; fi
  latest_step=$(python - <<'PY'
import json, pathlib
p = pathlib.Path('data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k/posttrain_grpo/logs.jsonl')
step = -1
if p.exists():
    for line in p.read_text().splitlines()[-20:]:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if 'step' in obj:
            step = max(step, int(obj['step']))
print(step)
PY
)
  jsonl_mtime=$(stat -c %Y "$JSONL" 2>/dev/null || echo 0)
  ckpt_mtime=$(stat -c %y "$CKPT" 2>/dev/null || echo 'missing')
  gpu=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep '^2681728,' || true)
  now=$(date -Is)
  echo "[$now] status=$proc_status step=$latest_step jsonl_mtime=$jsonl_mtime ckpt_mtime=$ckpt_mtime gpu='$gpu'"
  if [[ "$latest_step" -ge "$TARGET" ]]; then
    echo "[$now] target step reached"
    exit 0
  fi
  if [[ "$proc_status" == 'missing' ]]; then
    echo "[$now] process missing; exiting monitor"
    exit 0
  fi
  sleep 120
done
