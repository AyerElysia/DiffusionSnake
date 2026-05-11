#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/medteam/Zhrch/DiffusionSnake-12-30"
cd "$ROOT"

source /home/medteam/miniconda3/etc/profile.d/conda.sh
conda activate snake1

TARGET_STEP=69000
FREE_MEM_MIN=35000
BASE_CKPT="data/outputs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolos_detail_gpu4_reusemax/checkpoints/latest.pt"
RUN_ROOT="data/outputs/v4_1_mechanism_ablation"
mkdir -p "$RUN_ROOT"

RUN_SPECS=(
  "btcv_ablate_v41_02_no_delta|6"
  "btcv_ablate_v41_03_no_curv|7"
  "btcv_ablate_v41_04_no_small_disp|0"
  "btcv_ablate_v41_05_delta_only|1"
  "btcv_ablate_v41_06_curv_only|3"
  "btcv_ablate_v41_07_small_only|5"
)

now() {
  date '+%F %T'
}

gpu_free_mb() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F, -v target="$gpu" '{gsub(/ /, "", $1); gsub(/ /, "", $2); if ($1 == target) print $2}'
}

gpu_is_free() {
  local gpu="$1"
  local free
  free="$(gpu_free_mb "$gpu")"
  [[ -n "$free" && "$free" -ge "$FREE_MEM_MIN" ]]
}

first_free_gpu() {
  local gpu
  for gpu in 0 1 2 3 4 5 6 7; do
    if gpu_is_free "$gpu"; then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

is_running() {
  local stem="$1"
  ps -eo cmd | grep -F "diffusion_train.py" | grep -F "${stem}.yaml" | grep -v grep >/dev/null
}

last_step() {
  local stem="$1"
  local log="data/outputs/${stem}/logs.jsonl"
  if [[ ! -f "$log" ]]; then
    echo "-1"
    return
  fi
  python - "$log" <<'PY'
import json
import sys

last = None
with open(sys.argv[1], "rb") as f:
    for raw in f:
        line = raw.strip().replace(b"\x00", b"")
        if not line:
            continue
        try:
            last = json.loads(line.decode("utf-8"))
        except Exception:
            pass
if isinstance(last, dict):
    print(int(last.get("step", -1)))
else:
    print("-1")
PY
}

is_done() {
  local stem="$1"
  local step
  step="$(last_step "$stem")"
  [[ "$step" -ge "$TARGET_STEP" ]]
}

start_job() {
  local stem="$1"
  local gpu="$2"
  local cfg="configs/${stem}.yaml"
  local out_dir="data/outputs/${stem}"
  local own_ckpt="${out_dir}/checkpoints/latest.pt"
  local ts
  ts="$(date '+%Y%m%d_%H%M%S')"
  local log="${RUN_ROOT}/${stem}_restart_${ts}_gpu${gpu}.log"

  if is_done "$stem"; then
    echo "[$(now)] skip done ${stem} step=$(last_step "$stem")"
    return
  fi
  if is_running "$stem"; then
    echo "[$(now)] skip running ${stem}"
    return
  fi

  if [[ ! -f "$cfg" ]]; then
    echo "[$(now)] missing cfg ${cfg}" >&2
    return 1
  fi

  if [[ -f "$own_ckpt" ]]; then
    echo "[$(now)] start ${stem} on gpu${gpu}, resume ${own_ckpt}, log=${log}"
    nohup env PYTHONUNBUFFERED=1 WANDB_MODE=offline CFG_FILE="$ROOT/$cfg" \
      python diffusion_train.py --cfg_file "$cfg" \
      gpus "[${gpu}]" \
      resume_weights_only False resume_path "$own_ckpt" \
      > "$log" 2>&1 &
  else
    echo "[$(now)] start ${stem} on gpu${gpu}, init ${BASE_CKPT}, log=${log}"
    nohup env PYTHONUNBUFFERED=1 WANDB_MODE=offline CFG_FILE="$ROOT/$cfg" \
      python diffusion_train.py --cfg_file "$cfg" \
      gpus "[${gpu}]" \
      resume_weights_only True resume_path "$BASE_CKPT" \
      > "$log" 2>&1 &
  fi
}

while true; do
  pending=0
  started=0
  for spec in "${RUN_SPECS[@]}"; do
    stem="${spec%%|*}"
    gpu="${spec##*|}"
    if is_done "$stem" || is_running "$stem"; then
      continue
    fi
    pending=1
    if gpu_is_free "$gpu"; then
      start_job "$stem" "$gpu"
      started=1
      sleep 20
    else
      alt_gpu="$(first_free_gpu || true)"
      if [[ -n "$alt_gpu" ]]; then
        echo "[$(now)] assigned gpu${gpu} unavailable for ${stem}; using gpu${alt_gpu}"
        start_job "$stem" "$alt_gpu"
        started=1
        sleep 20
      else
        echo "[$(now)] wait any free gpu for ${stem}; assigned gpu${gpu} free=$(gpu_free_mb "$gpu")MiB"
      fi
    fi
  done
  if [[ "$pending" -eq 0 ]]; then
    break
  fi
  if [[ "$started" -eq 0 ]]; then
    sleep 300
  fi
done

echo "[$(now)] all train jobs are running or done; waiting for completion"
while true; do
  all_done=1
  for spec in "${RUN_SPECS[@]}"; do
    stem="${spec%%|*}"
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

echo "[$(now)] training done; running eval"
bash scripts/run_v41_ablation_eval_when_done.sh
