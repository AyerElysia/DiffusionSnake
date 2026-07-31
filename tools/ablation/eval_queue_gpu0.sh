#!/bin/bash
# Sequential eval queue on GPU0.
# GPU0 is freed once arm C's training process exits; after that we evaluate
# C ep11 and B ep7 one at a time (each eval needs ~13 GB and a lot of disk IO,
# so running them concurrently on one card just thrashes).
set -u
cd /home/medteam/Zhrch/DiffusionSnake-12-30
PY=/home/medteam/miniconda3/envs/snake1/bin/python
C_PID=2018993

echo "[queue] waiting for C training pid $C_PID to exit"
for i in $(seq 1 240); do
  ps -p $C_PID > /dev/null 2>&1 || break
  sleep 15
done
ps -p $C_PID > /dev/null 2>&1 && { echo "[queue] C still running after 60min, abort"; exit 1; }
echo "[queue] GPU0 free"
sleep 30

run_eval () {
  local cfg=$1 ckpt=$2 out=$3 name=$4
  if [ -f "$out/summary.json" ]; then echo "[queue] $name already done"; return 0; fi
  if [ ! -f "$ckpt" ]; then echo "[queue] $name ckpt missing: $ckpt"; return 1; fi
  mkdir -p "$out"
  echo "[queue] eval $name -> $out"
  env CUDA_VISIBLE_DEVICES=0 $PY tools/eval_sagittal_2d_fixed.py \
    --cfg_file "$cfg" --ckpt "$ckpt" \
    --box-mode gt --split val --max-slices 400 \
    --result-dir "$out" --device cuda > "$out/eval.log" 2>&1
  echo "[queue] $name exit=$?"
  [ -f "$out/summary.json" ] && $PY -c "import json;d=json.load(open('$out/summary.json'));print('$name',{k:round(v,4) for k,v in d.items() if isinstance(v,float)})"
}

run_eval configs/ablation/abl_c_u4_single.yaml \
  data/outputs/abl_c_u4_single/checkpoints/epoch_11.pt \
  data/outputs/abl_c_u4_single/eval_gt400_ep11 c_u4_ep11

echo "[queue] waiting for B epoch_7.pt"
for i in $(seq 1 160); do
  [ -f data/outputs/abl_b_u2_dual/checkpoints/epoch_7.pt ] && break
  sleep 15
done
sleep 20
run_eval configs/ablation/abl_b_u2_dual.yaml \
  data/outputs/abl_b_u2_dual/checkpoints/epoch_7.pt \
  data/outputs/abl_b_u2_dual/eval_gt400_ep7 b_u2_dual_ep7

echo "[queue] done"
