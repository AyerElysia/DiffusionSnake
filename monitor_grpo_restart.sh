#!/usr/bin/env bash
set -u
OUT_DIR="data/outputs/btcv_diffusion_dit_v3_4_fm_yolom_grpo_k8_purerl_mbf_kl_w1_long10k"
JSONL="$OUT_DIR/posttrain_grpo/logs.jsonl"
CKPT="$OUT_DIR/checkpoints/latest.pt"
TARGET=10000
while true; do
  if ! pgrep -af 'python -u grpo_train.py' | grep -q 'grpo_train.py'; then
    echo "[$(date -Is)] status=missing"
    exit 0
  fi
  step=$(python -c "import json,pathlib; p=pathlib.Path('$JSONL'); step=-1\nfor line in p.read_text().splitlines()[-20:]:\n    try: obj=json.loads(line)\n    except Exception: continue\n    step=max(step,int(obj.get('step',-1)))\nprint(step)")
  ckpt_mtime=$(stat -c '%y' "$CKPT" 2>/dev/null || echo 'missing')
  jsonl_mtime=$(stat -c '%y' "$JSONL" 2>/dev/null || echo 'missing')
  echo "[$(date -Is)] status=running step=$step jsonl=$jsonl_mtime ckpt=$ckpt_mtime"
  if [[ "$step" -ge "$TARGET" ]]; then
    echo "[$(date -Is)] target step reached"
    exit 0
  fi
  sleep 900
done
