#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

ROOT=data/outputs/volmem/diagnostics/memflowdit_v09_step000500_compact_gate_3vol
COMPARISON=${ROOT}/comparison.json
STATUS=${ROOT}/decision.log
RUN=data/outputs/volmem/verse_memflowdit_v0_9_compact_global_gpu6
EXPECTED_CFG=configs/volmem/verse_memflowdit_v0_9_compact_global_gpu6.yaml
deadline=$((SECONDS + 21600))

log_status() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$1" >>"${STATUS}"
}

log_status "waiting for ${COMPARISON}"
while (( SECONDS < deadline )); do
    [[ -s "${COMPARISON}" ]] && break
    sleep 30
done
if [[ ! -s "${COMPARISON}" ]]; then
    log_status "timed out without comparison; no signal sent"
    exit 1
fi

passes=$(python - "${COMPARISON}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
print("true" if payload.get("passes_gate") is True else "false")
PY
)
if [[ "${passes}" == "true" ]]; then
    log_status "gate passed; training left running"
    exit 0
fi

pid=$(tr -d '[:space:]' <"${RUN}/train.pid")
if [[ ! "${pid}" =~ ^[0-9]+$ || ! -r "/proc/${pid}/cmdline" ]]; then
    log_status "gate failed; training is already absent or pid is invalid (${pid})"
    exit 0
fi
command=$(tr '\0' ' ' <"/proc/${pid}/cmdline")
if [[ "${command}" != *"tools/volmem/train_memflowdit.py"* || \
      "${command}" != *"${EXPECTED_CFG}"* ]]; then
    log_status "gate failed; pid identity mismatch, refusing to signal: ${command}"
    exit 2
fi

log_status "gate failed; sending SIGTERM to verified training pid=${pid}"
kill -TERM "${pid}"
for _ in $(seq 1 30); do
    if ! kill -0 "${pid}" 2>/dev/null; then
        log_status "verified training stopped"
        exit 0
    fi
    sleep 1
done

if [[ -r "/proc/${pid}/cmdline" ]]; then
    command=$(tr '\0' ' ' <"/proc/${pid}/cmdline")
    if [[ "${command}" == *"tools/volmem/train_memflowdit.py"* && \
          "${command}" == *"${EXPECTED_CFG}"* ]]; then
        log_status "SIGTERM timeout; sending SIGKILL to verified pid=${pid}"
        kill -KILL "${pid}"
    fi
fi
