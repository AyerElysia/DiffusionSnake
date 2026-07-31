#!/usr/bin/env bash
set -euo pipefail

# Override GPU_LIST when another job occupies one of the default devices.
GPU_LIST="${GPU_LIST:-0,5,6}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"
CFG_FILE_PATH="${CFG_FILE_PATH:-configs/sagittal_2d_v4_6c_moonvit_train.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/medteam/miniconda3/envs/snake1/bin/python}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29611}"

if [[ "${#GPU_IDS[@]}" -ne "${NPROC_PER_NODE}" ]]; then
    echo "NPROC_PER_NODE must match GPU_LIST (${GPU_LIST})" >&2
    exit 2
fi

# PyTorch 1.11 supports max_split_size_mb but not newer allocator knobs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"

if [[ "$#" -eq 0 ]]; then
    set -- --cfg_file "${CFG_FILE_PATH}"
fi

exec env CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
    "${PYTHON_BIN}" -m torch.distributed.run \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    train_net_ddp.py "$@"
