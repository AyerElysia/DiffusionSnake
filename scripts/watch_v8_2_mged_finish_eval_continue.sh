#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/medteam/Zhrch/DiffusionSnake-12-30"
CFG="configs/1232_final_v8_2_mged_final_gpu6.yaml"
OUT="data/outputs/1232_final_v8_2_mged_final_gpu6"
WATCH_LOG="${OUT}/watch_finish_eval_continue_20260605.log"
CURRENT_PID="${CURRENT_PID:-848546}"
TRAIN_GPU="${TRAIN_GPU:-6}"
EVAL_GPU="${EVAL_GPU:-7}"
AUTO_EVAL_SAMPLES="${AUTO_EVAL_SAMPLES:-64}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LOG_EVERY="${LOG_EVERY:-20}"
NUM_WORKERS="${NUM_WORKERS:-0}"

cd "${ROOT}"
mkdir -p "${OUT}"
exec > >(tee -a "${WATCH_LOG}") 2>&1

source /home/medteam/miniconda3/etc/profile.d/conda.sh
conda activate snake1
set -u

echo "[watch] started $(date)"
echo "[watch] cfg=${CFG} train_gpu=${TRAIN_GPU} eval_gpu=${EVAL_GPU} samples=${AUTO_EVAL_SAMPLES}"

wait_for_checkpoint() {
    local ckpt="$1"
    local expected_step="$2"
    echo "[watch] waiting for ${ckpt} step=${expected_step}"
    while true; do
        if python - "${ckpt}" "${expected_step}" <<'PY'
import os
import sys
import torch

ckpt, expected = sys.argv[1], int(sys.argv[2])
if not os.path.exists(ckpt):
    raise SystemExit(1)
obj = torch.load(ckpt, map_location='cpu')
step = int(obj.get('step', -1)) if isinstance(obj, dict) else -1
if step != expected:
    raise SystemExit(2)
print(f"[watch] checkpoint ready: {ckpt} step={step}")
PY
        then
            break
        fi
        tail -n 1 "${OUT}/minimal_logs.jsonl" || true
        sleep 300
    done
}

wait_for_pid_exit() {
    local pid="$1"
    if [[ -z "${pid}" ]]; then
        return
    fi
    while ps -p "${pid}" > /dev/null 2>&1; do
        echo "[watch] waiting for current training pid=${pid} to exit"
        sleep 30
    done
}

run_eval() {
    local ckpt="$1"
    local tag="$2"
    local save_dir="visual/${tag}_pred_det_pred_ext_${AUTO_EVAL_SAMPLES}"
    echo "[watch] eval ${tag} ckpt=${ckpt} save_dir=${save_dir}"
    env \
        CFG_FILE="${CFG}" \
        CKPT="${ckpt}" \
        EVAL_GPU="${EVAL_GPU}" \
        EVAL_ABLATION_MODE=pred_det_pred_ext \
        MAX_SAMPLES="${AUTO_EVAL_SAMPLES}" \
        SAVE_VISUALS=1 \
        SAVE_DIR="${save_dir}" \
        EVAL_DET_CONF_THRESH=0.01 \
        EVAL_DET_IOU_THRESH=0.45 \
        EVAL_DET_MAX_DET=100 \
        python -u scripts/eval_v37_full_iou.py
}

CKPT_20000="${OUT}/checkpoints/step_20000.pt"
CKPT_40000="${OUT}/checkpoints/step_40000.pt"

wait_for_checkpoint "${CKPT_20000}" 20000
wait_for_pid_exit "${CURRENT_PID}"
run_eval "${CKPT_20000}" "v8_2_mged_step20000"

echo "[watch] continue training 20001 -> 40000 on GPU ${TRAIN_GPU}"
env \
    CFG_FILE="${CFG}" \
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" \
    RESUME_TRAIN_CKPT="${CKPT_20000}" \
    MAX_STEPS=40000 \
    LOG_EVERY="${LOG_EVERY}" \
    SAVE_STEPS="${SAVE_STEPS}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    python -u scripts/train_v8_heatmap_minimal.py \
    > "${OUT}/train_auto_continue_20000_to_40000_20260605_gpu${TRAIN_GPU}.log" 2>&1

wait_for_checkpoint "${CKPT_40000}" 40000
run_eval "${CKPT_40000}" "v8_2_mged_step40000"
echo "[watch] finished $(date)"
