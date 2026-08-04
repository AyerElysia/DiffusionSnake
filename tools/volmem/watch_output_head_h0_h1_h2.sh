#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/medteam/Zhrch/DiffusionSnake-12-30
EXP="$ROOT/data/outputs/volmem/output_head_h0_h1_h2_20260803"
CACHE="$EXP/cache/train_h0_step2300_stride8"
TEACHER="$ROOT/data/outputs/volmem/rl3d/ckpt_backup/v05_step_002300.pt"
DISTILL="$EXP/distilled"
LOG="$EXP/watcher.log"

cd "$ROOT"
mkdir -p "$DISTILL" "$EXP/distilled_eval"
exec >>"$LOG" 2>&1

echo "[watcher] waiting for teacher cache"
while [[ ! -f "$CACHE/manifest.json" ]]; do
    if ! pgrep -f "final-head-cache-dir data/outputs/volmem/output_head_h0_h1_h2_20260803/cache/train_h0_step2300_stride8" >/dev/null; then
        echo "[watcher] cache process exited without a manifest"
        tail -80 "$EXP/cache_teacher_stride8.log"
        exit 1
    fi
    sleep 30
done

echo "[watcher] cache complete"
if [[ ! -f "$DISTILL/distillation_summary.json" ]]; then
    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python tools/volmem/distill_output_head.py \
        --cfg_file configs/volmem/verse_memflowdit_v0_5_minimal_gpu6.yaml \
        --cache-dir "$CACHE" \
        --teacher-ckpt "$TEACHER" \
        --output-dir "$DISTILL" \
        --device cuda \
        --seed 20260803 \
        --h1-steps 5000 \
        --h2-steps 3000 \
        --batch-contours 64 \
        --learning-rate 0.0005 \
        >"$EXP/distill.log" 2>&1
fi

wait_for_gpu() {
    local gpu="$1"
    while true; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
        if [[ "$used" -lt 2000 ]]; then
            return
        fi
        sleep 30
    done
}

run_eval() {
    local gpu="$1"
    local cfg_file="$2"
    local checkpoint="$3"
    local batch_size="$4"
    local name="$5"
    local result="$EXP/distilled_eval/$name"
    if [[ -f "$result/summary.json" ]]; then
        echo "[watcher] skip completed $name"
        return
    fi
    wait_for_gpu "$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python tools/volmem/eval_memflowdit_parallel.py \
        --cfg_file "$cfg_file" \
        --ckpt "$checkpoint" \
        --split val \
        --memory-mode parallel-off \
        --parallel-batch-size "$batch_size" \
        --box-mode gt \
        --max-volumes 3 \
        --seed 20260731 \
        --locate-feat-cache-root /dev/shm/memflowdit_moonvit_cache \
        --result-dir "$result" \
        --log-every 100 \
        >"$EXP/${name}.log" 2>&1
}

echo "[watcher] launching strict H1/H2 validation"
run_eval 0 configs/volmem/verse_memflowdit_output_head_h1_distilled_dense_gpu0.yaml \
    "$DISTILL/h1_distilled_full.pt" 8 h1_parallel_off_batch8 &
pid_h1_b8=$!
run_eval 1 configs/volmem/verse_memflowdit_output_head_h2_shared_sparse_gpu1.yaml \
    "$DISTILL/h2_distilled_full.pt" 8 h2_parallel_off_batch8 &
pid_h2_b8=$!
run_eval 4 configs/volmem/verse_memflowdit_output_head_h1_distilled_dense_gpu0.yaml \
    "$DISTILL/h1_distilled_full.pt" 1 h1_parallel_off_batch1 &
pid_h1_b1=$!
run_eval 5 configs/volmem/verse_memflowdit_output_head_h2_shared_sparse_gpu1.yaml \
    "$DISTILL/h2_distilled_full.pt" 1 h2_parallel_off_batch1 &
pid_h2_b1=$!

wait "$pid_h1_b8" "$pid_h2_b8" "$pid_h1_b1" "$pid_h2_b1"
echo "[watcher] all strict evaluations complete"
