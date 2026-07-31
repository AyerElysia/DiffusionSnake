#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30
while kill -0 1771937 2>/dev/null; do
    sleep 20
done

export CUDA_VISIBLE_DEVICES=7
seed=20260731
control_dir="data/outputs/volmem/verse_memflowdit_outputmoe_step100_control_gpu7"
hardphi_dir="data/outputs/volmem/verse_memflowdit_outputmoe_step100_hardphi_gpu7"

/usr/bin/python3.8 tools/volmem/eval_memflowdit_v03.py \
    --cfg_file configs/volmem/verse_memflowdit_outputmoe_step100_control_gpu7.yaml \
    --ckpt "${control_dir}/checkpoints/step_000100.pt" \
    --split val \
    --memory-mode off \
    --box-mode gt \
    --result-dir "${control_dir}/eval_gt_off_1vol_seed${seed}" \
    --device cuda:0 \
    --max-volumes 1 \
    --seed "${seed}" \
    --log-every 100

/usr/bin/python3.8 tools/volmem/eval_memflowdit_v03.py \
    --cfg_file configs/volmem/verse_memflowdit_outputmoe_step100_hardphi_gpu7.yaml \
    --ckpt "${hardphi_dir}/checkpoints/step_000100.pt" \
    --split val \
    --memory-mode off \
    --box-mode gt \
    --result-dir "${hardphi_dir}/eval_gt_off_1vol_seed${seed}" \
    --device cuda:0 \
    --max-volumes 1 \
    --seed "${seed}" \
    --log-every 100
