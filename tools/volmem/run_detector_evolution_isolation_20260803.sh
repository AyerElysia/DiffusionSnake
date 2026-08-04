#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/medteam/Zhrch/DiffusionSnake-12-30
PY=/home/medteam/miniconda3/envs/qy_esnake/bin/python
CFG=configs/volmem/verse_memflowdit_v0_5_minimal_gpu6.yaml
CKPT=data/outputs/volmem/rl3d/ckpt_backup/v05_step_002300.pt
OUT=data/outputs/volmem/diagnostics/detector_evolution_isolation_v05_step2300_20260803

cd "$ROOT"
test -f "$CKPT"
test -f "$OUT/cache_iou01.json"
test -f "$OUT/cache_iou03.json"

launch() {
    local gpu="$1"
    local name="$2"
    local cache="$3"
    local result="$OUT/$name"
    local log="$OUT/$name.log"
    local pid_file="$OUT/$name.pid"
    test ! -e "$result"
    test ! -e "$log"
    test ! -e "$pid_file"
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" -u tools/volmem/eval_memflowdit_v03.py \
        --cfg_file "$CFG" \
        --ckpt "$CKPT" \
        --split val \
        --memory-mode off \
        --box-mode predicted \
        --box-source locany_cached \
        --locany-cache-path "$cache" \
        --result-dir "$result" \
        --device cuda \
        --max-volumes 3 \
        --log-every 50 \
        --seed 20260731 \
        >"$log" 2>&1 < /dev/null &
    printf '%s\n' "$!" >"$pid_file"
    printf '%s gpu=%s pid=%s\n' "$name" "$gpu" "$!"
}

launch 1 iou01 "$OUT/cache_iou01.json"
launch 4 iou03 "$OUT/cache_iou03.json"
