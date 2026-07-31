#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

export CUDA_VISIBLE_DEVICES=7
control_dir="data/outputs/volmem/verse_memflowdit_outputmoe_step100_control_gpu7"
hardphi_dir="data/outputs/volmem/verse_memflowdit_outputmoe_step100_hardphi_gpu7"

/usr/bin/python3.8 tools/volmem/train_memflowdit.py \
    --cfg_file configs/volmem/verse_memflowdit_outputmoe_step100_control_gpu7.yaml \
    --device cuda:0 \
    --max_steps 100 \
    --save_every 50 \
    --chunks_per_step 4 \
    --output-dir-override "${control_dir}"

/usr/bin/python3.8 tools/volmem/train_memflowdit.py \
    --cfg_file configs/volmem/verse_memflowdit_outputmoe_step100_hardphi_gpu7.yaml \
    --device cuda:0 \
    --max_steps 100 \
    --save_every 50 \
    --chunks_per_step 4 \
    --output-dir-override "${hardphi_dir}"

/usr/bin/python3.8 tools/volmem/eval_memflowdit_v03.py \
    --cfg_file configs/volmem/verse_memflowdit_outputmoe_step100_control_gpu7.yaml \
    --ckpt "${control_dir}/checkpoints/step_000100.pt" \
    --split val \
    --memory-mode off \
    --box-mode gt \
    --result-dir "${control_dir}/eval_gt_off_1vol" \
    --device cuda:0 \
    --max-volumes 1 \
    --log-every 100

/usr/bin/python3.8 tools/volmem/eval_memflowdit_v03.py \
    --cfg_file configs/volmem/verse_memflowdit_outputmoe_step100_hardphi_gpu7.yaml \
    --ckpt "${hardphi_dir}/checkpoints/step_000100.pt" \
    --split val \
    --memory-mode off \
    --box-mode gt \
    --result-dir "${hardphi_dir}/eval_gt_off_1vol" \
    --device cuda:0 \
    --max-volumes 1 \
    --log-every 100
