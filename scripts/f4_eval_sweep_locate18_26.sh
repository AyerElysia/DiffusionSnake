#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

RUNS=(
  "lr1e5:data/outputs/f4_locate18_26_from58000_lr1e5_gpu0:1e-5"
  "lr5e6:data/outputs/f4_locate18_26_from58000_lr5e6_gpu2:5e-6"
)
STEPS=(60000 62000 64000 66000 68000 70000)

for step in "${STEPS[@]}"; do
  while true; do
    ready=1
    for item in "${RUNS[@]}"; do
      IFS=: read -r name out_dir lr <<<"${item}"
      ckpt="${out_dir}/checkpoints/step_${step}.pt"
      if [[ ! -f "${ckpt}" ]]; then
        ready=0
      fi
    done
    if [[ "${ready}" == "1" ]]; then
      break
    fi
    echo "[wait-f4] $(date +%F_%T) waiting for step_${step} checkpoints"
    sleep 300
  done

  for item in "${RUNS[@]}"; do
    IFS=: read -r name out_dir lr <<<"${item}"
    ckpt="${out_dir}/checkpoints/step_${step}.pt"
    save_dir="visual/f4_locate18_26_${name}_step${step}_gtinit_eval"
    log="/tmp/f4_locate18_26_${name}_step${step}_gtinit_eval.log"
    if compgen -G "${save_dir}/v3_7_full_test_iou_*.json" >/dev/null; then
      echo "[skip] ${name} step=${step} already evaluated in ${save_dir}"
      continue
    fi
    echo "[eval] ${name} step=${step} ckpt=${ckpt}"
    CKPT="${ckpt}" EVAL_GPU=7 EVAL_ABLATION_MODE=gt_init SAVE_VISUALS=0 \
      SAVE_DIR="${save_dir}" ODE_STEPS=10 \
      /home/medteam/miniconda3/envs/snake1/bin/python -u scripts/eval_v37_full_iou.py \
      --cfg_file configs/f2_locate_feat_replace_gpu0.yaml \
      model_dir "${out_dir}" train.lr "${lr}" \
      2>&1 | tee "${log}"
  done
done

echo "[eval-done-f4]"
