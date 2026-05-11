#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k"
JSONL="$OUT_DIR/posttrain_grpo/logs.jsonl"
CKPT="$OUT_DIR/checkpoints/latest.pt"
TARGET=10000
while true; do
  if ! pgrep -af 'python -u grpo_train.py' | grep -q 'GRPO_TRAIN_STEPS=10000'; then
    now=$(date -Is)
    echo "[$now] status=missing"
    exit 0
  fi
  step=$(python - <<'PY'
import json, pathlib
p = pathlib.Path('data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k/posttrain_grpo/logs.jsonl')
step = -1
if p.exists():
    for line in p.read_text().splitlines()[-50:]:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if 'step' in obj:
            step = max(step, int(obj['step']))
print(step)
PY
)
  gpu=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep '^2681728,' || true)
  ckpt_mtime=$(stat -c '%y' "$CKPT" 2>/dev/null || echo 'missing')
  jsonl_mtime=$(stat -c '%y' "$JSONL" 2>/dev/null || echo 'missing')
  now=$(date -Is)
  echo "[$now] status=running step=$step jsonl=$jsonl_mtime ckpt=$ckpt_mtime gpu='$gpu'"
  if [[ "$step" -ge "$TARGET" ]]; then
    echo "[$now] target step reached"
    exit 0
  fi
  sleep 900
done
