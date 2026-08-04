#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

SRC=data/outputs/volmem/verse_memflowdit_v0_8_evidence_value_mm_gpu5/checkpoints/step_000100.pt
STAGED=/dev/shm/memflowdit_checkpoints/v08_evidence_value_mm_step_000100.pt
ROOT=data/outputs/volmem/diagnostics/memflowdit_v08_step000100_gt_gate
CFG=configs/volmem/verse_memflowdit_v0_8_evidence_value_mm_gpu5.yaml
mkdir -p "${ROOT}/off" "${ROOT}/oracle_recent_k7"

deadline=$((SECONDS + 10800))
previous_size=-1
stable_checks=0
while (( SECONDS < deadline )); do
    if [[ -f "${SRC}" ]]; then
        current_size=$(stat -c %s "${SRC}")
        if [[ "${current_size}" -gt 0 && "${current_size}" -eq "${previous_size}" ]]; then
            stable_checks=$((stable_checks + 1))
        else
            stable_checks=0
        fi
        previous_size=${current_size}
        if (( stable_checks >= 2 )); then
            break
        fi
    fi
    sleep 20
done
if [[ ! -f "${SRC}" || ${stable_checks} -lt 2 ]]; then
    echo "checkpoint wait timed out: ${SRC}" >&2
    exit 1
fi

cp "${SRC}" "${STAGED}"
sha256sum "${SRC}" "${STAGED}"

while (( SECONDS < deadline )); do
    used0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    used4=$(nvidia-smi -i 4 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if (( used0 < 1024 && used4 < 1024 )); then
        break
    fi
    sleep 20
done

env CUDA_VISIBLE_DEVICES=0 python tools/volmem/eval_memflowdit_parallel.py \
    --cfg_file "${CFG}" \
    --ckpt "${STAGED}" \
    --split val \
    --memory-mode parallel-off \
    --box-mode gt \
    --result-dir "${ROOT}/off" \
    --device cuda:0 \
    --max-volumes 1 \
    --seed 20260731 \
    --memory-capacity 7 \
    --parallel-batch-size 1 \
    --locate-feat-cache-root /dev/shm/memflowdit_moonvit_cache \
    >"${ROOT}/off/run.log" 2>&1 &
off_pid=$!

env CUDA_VISIBLE_DEVICES=4 python tools/volmem/eval_memflowdit_parallel.py \
    --cfg_file "${CFG}" \
    --ckpt "${STAGED}" \
    --split val \
    --memory-mode frozen-oracle-causal \
    --box-mode gt \
    --result-dir "${ROOT}/oracle_recent_k7" \
    --device cuda:0 \
    --max-volumes 1 \
    --seed 20260731 \
    --memory-capacity 7 \
    --parallel-batch-size 1 \
    --locate-feat-cache-root /dev/shm/memflowdit_moonvit_cache \
    >"${ROOT}/oracle_recent_k7/run.log" 2>&1 &
oracle_pid=$!

wait "${off_pid}"
wait "${oracle_pid}"

python - "${ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
off = json.loads((root / "off" / "summary.json").read_text())
oracle = json.loads((root / "oracle_recent_k7" / "summary.json").read_text())
comparison = {
    "checkpoint_step": 100,
    "volume": "sub-verse010",
    "off_volume_dice": off["volume_mean_dice"],
    "oracle_volume_dice": oracle["volume_mean_dice"],
    "oracle_delta_vs_off": oracle["volume_mean_dice"] - off["volume_mean_dice"],
    "off_foreground_slice_dice": off["foreground_slice_mean_dice"],
    "oracle_foreground_slice_dice": oracle["foreground_slice_mean_dice"],
    "oracle_read_delta": oracle["mean_memory_read_delta"],
}
(root / "comparison.json").write_text(
    json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(comparison, sort_keys=True))
PY
