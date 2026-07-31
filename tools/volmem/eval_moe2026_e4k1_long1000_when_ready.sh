#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/medteam/Zhrch/DiffusionSnake-12-30
TRAIN_PID=1652072
GPU=1
CFG=configs/volmem/verse_memflowdit_moe2026_dataproto_phi_e4k1_odd_gpu0.yaml
RUN=data/outputs/volmem/verse_memflowdit_moe2026_dataproto_e4k1_odd_long1000_gpu1
CKPT=${RUN}/checkpoints/step_001000.pt
OFF_RESULT=data/outputs/volmem/diagnostics/moe2026_eval_e4k1_long1000_gt_off_3vol
AUTO_RESULT=data/outputs/volmem/diagnostics/moe2026_eval_e4k1_long1000_gt_autoregressive_3vol

cd "${ROOT}"
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    sleep 30
done

if [[ ! -f "${CKPT}" ]]; then
    echo "[watcher] training exited without ${CKPT}" >&2
    exit 1
fi

mkdir -p "${OFF_RESULT}" "${AUTO_RESULT}"
env CUDA_VISIBLE_DEVICES="${GPU}" /usr/bin/python3.8 \
    tools/volmem/eval_memflowdit_v03.py \
    --cfg_file "${CFG}" \
    --ckpt "${CKPT}" \
    --split val \
    --memory-mode off \
    --box-mode gt \
    --result-dir "${OFF_RESULT}" \
    --device cuda:0 \
    --max-volumes 3 \
    --log-every 100 \
    > "${OFF_RESULT}/eval.log" 2>&1

env CUDA_VISIBLE_DEVICES="${GPU}" /usr/bin/python3.8 \
    tools/volmem/eval_memflowdit_v03.py \
    --cfg_file "${CFG}" \
    --ckpt "${CKPT}" \
    --split val \
    --memory-mode autoregressive \
    --box-mode gt \
    --result-dir "${AUTO_RESULT}" \
    --device cuda:0 \
    --max-volumes 3 \
    --log-every 100 \
    > "${AUTO_RESULT}/eval.log" 2>&1

/usr/bin/python3.8 - <<'PY'
import json
from pathlib import Path

paths = [
    Path("data/outputs/volmem/diagnostics/moe2026_eval_e4k1_long1000_gt_off_3vol/summary.json"),
    Path("data/outputs/volmem/diagnostics/moe2026_eval_e4k1_long1000_gt_autoregressive_3vol/summary.json"),
]
for path in paths:
    data = json.loads(path.read_text())
    print(
        path.parent.name,
        "dice={:.6f}".format(data["volume_mean_dice"]),
        "iou={:.6f}".format(data["volume_mean_iou"]),
        "memory_delta={:.6f}".format(data["mean_memory_read_delta"]),
    )
PY
