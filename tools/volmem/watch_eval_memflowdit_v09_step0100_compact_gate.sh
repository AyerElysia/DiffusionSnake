#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

SRC=data/outputs/volmem/verse_memflowdit_v0_9_compact_global_gpu6/checkpoints/step_000100.pt
STAGED=/dev/shm/memflowdit_checkpoints/v09_compact_global_step_000100.pt
ROOT=data/outputs/volmem/diagnostics/memflowdit_v09_step000100_compact_gate
CFG=configs/volmem/verse_memflowdit_v0_9_compact_global_gpu6.yaml
CACHE=/dev/shm/memflowdit_moonvit_cache
mkdir -p "${ROOT}/off" "${ROOT}/oracle_local_k4" "${ROOT}/oracle_local_k4_global16"

deadline=$((SECONDS + 21600))
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
    used1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    used4=$(nvidia-smi -i 4 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if (( used0 < 1024 && used1 < 1024 && used4 < 1024 )); then
        break
    fi
    sleep 20
done

env CUDA_VISIBLE_DEVICES=0 python tools/volmem/eval_memflowdit_parallel.py \
    --cfg_file "${CFG}" --ckpt "${STAGED}" --split val \
    --memory-mode parallel-off --box-mode gt --result-dir "${ROOT}/off" \
    --device cuda:0 --max-volumes 1 --seed 20260731 \
    --memory-capacity 4 --memory-global-pool-size 0 --parallel-batch-size 1 \
    --locate-feat-cache-root "${CACHE}" >"${ROOT}/off/run.log" 2>&1 &
off_pid=$!

env CUDA_VISIBLE_DEVICES=1 python tools/volmem/eval_memflowdit_parallel.py \
    --cfg_file "${CFG}" --ckpt "${STAGED}" --split val \
    --memory-mode frozen-oracle-causal --box-mode gt \
    --result-dir "${ROOT}/oracle_local_k4" --device cuda:0 --max-volumes 1 \
    --seed 20260731 --memory-capacity 4 --memory-global-pool-size 0 \
    --parallel-batch-size 1 --locate-feat-cache-root "${CACHE}" \
    >"${ROOT}/oracle_local_k4/run.log" 2>&1 &
local_pid=$!

env CUDA_VISIBLE_DEVICES=4 python tools/volmem/eval_memflowdit_parallel.py \
    --cfg_file "${CFG}" --ckpt "${STAGED}" --split val \
    --memory-mode frozen-oracle-compact --box-mode gt \
    --result-dir "${ROOT}/oracle_local_k4_global16" --device cuda:0 \
    --max-volumes 1 --seed 20260731 --memory-capacity 4 \
    --memory-global-pool-size 4 --parallel-batch-size 1 \
    --locate-feat-cache-root "${CACHE}" \
    >"${ROOT}/oracle_local_k4_global16/run.log" 2>&1 &
compact_pid=$!

wait "${off_pid}"
wait "${local_pid}"
wait "${compact_pid}"

python - "${ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
off = json.loads((root / "off" / "summary.json").read_text())
local = json.loads((root / "oracle_local_k4" / "summary.json").read_text())
compact = json.loads(
    (root / "oracle_local_k4_global16" / "summary.json").read_text()
)

def metrics(run):
    return {
        "volume_dice": run["volume_mean_dice"],
        "foreground_slice_dice": run["foreground_slice_mean_dice"],
        "class_dice": run["class_mean_dice"],
        "read_delta": run["mean_memory_read_delta"],
        "slices_per_second": run["slices_per_second"],
        "peak_cuda_memory_gb": run["peak_cuda_memory_gb"],
    }

comparison = {
    "checkpoint_step": 100,
    "volume": "sub-verse010",
    "off": metrics(off),
    "oracle_local_k4": metrics(local),
    "oracle_local_k4_global16": metrics(compact),
    "local_delta_vs_off": local["volume_mean_dice"] - off["volume_mean_dice"],
    "compact_delta_vs_off": compact["volume_mean_dice"] - off["volume_mean_dice"],
    "compact_delta_vs_local": compact["volume_mean_dice"] - local["volume_mean_dice"],
    "compact_foreground_delta_vs_off": (
        compact["foreground_slice_mean_dice"] - off["foreground_slice_mean_dice"]
    ),
    "compact_speed_change_pct_vs_off": (
        compact["slices_per_second"] / off["slices_per_second"] - 1.0
    ) * 100.0,
}
comparison["passes_gate"] = bool(
    comparison["compact_delta_vs_off"] >= 0.001
    and comparison["compact_foreground_delta_vs_off"] > 0.0
    and comparison["compact_speed_change_pct_vs_off"] >= -10.0
)
(root / "comparison.json").write_text(
    json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(comparison, sort_keys=True))
PY
