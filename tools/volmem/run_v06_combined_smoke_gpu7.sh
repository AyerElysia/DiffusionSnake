#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30
export CUDA_VISIBLE_DEVICES=7

combined_dir="data/outputs/volmem/verse_memflowdit_v0_6_moe2026_combined_smoke50_gpu7"
seed=20260731

/usr/bin/python3.8 tools/volmem/train_memflowdit.py \
    --cfg_file configs/volmem/verse_memflowdit_v0_6_moe2026_combined_gpu7.yaml \
    --device cuda:0 \
    --max_steps 50 \
    --save_every 50 \
    --chunks_per_step 4 \
    --output-dir-override "${combined_dir}"

/usr/bin/python3.8 tools/volmem/eval_memflowdit_v03.py \
    --cfg_file configs/volmem/verse_memflowdit_moe2026_dataproto_phi_e4k1_odd_gpu0.yaml \
    --ckpt data/outputs/volmem/verse_memflowdit_moe2026_dataproto_phi_e4k1_odd_gpu0/checkpoints/step_000050.pt \
    --split val \
    --memory-mode off \
    --box-mode gt \
    --result-dir data/outputs/volmem/verse_memflowdit_moe2026_dataproto_phi_e4k1_odd_gpu0/eval_gt_off_1vol_seed20260731 \
    --device cuda:0 \
    --max-volumes 1 \
    --seed "${seed}" \
    --log-every 100

/usr/bin/python3.8 tools/volmem/eval_memflowdit_v03.py \
    --cfg_file configs/volmem/verse_memflowdit_v0_6_moe2026_combined_gpu7.yaml \
    --ckpt "${combined_dir}/checkpoints/step_000050.pt" \
    --split val \
    --memory-mode off \
    --box-mode gt \
    --result-dir "${combined_dir}/eval_gt_off_1vol_seed${seed}" \
    --device cuda:0 \
    --max-volumes 1 \
    --seed "${seed}" \
    --log-every 100
