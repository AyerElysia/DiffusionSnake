#!/usr/bin/env bash
set -euo pipefail

cd /home/medteam/Zhrch/DiffusionSnake-12-30

SOURCE_ROOT=data/sagittal_moonvit_cache/validation
TARGET_ROOT=/dev/shm/memflowdit_moonvit_cache/validation
mkdir -p "${TARGET_ROOT}"
mkdir -p /dev/shm/memflowdit_checkpoints

# The stage-0 gate is intentionally fixed to the first three sorted validation
# volumes.  Copy once to node-local tmpfs so parallel evaluations do not turn
# the shared filesystem into the experiment bottleneck.
for volume_id in sub-verse010 sub-verse011 sub-verse013; do
    if [[ ! -d "${SOURCE_ROOT}/${volume_id}" ]]; then
        echo "missing source cache: ${SOURCE_ROOT}/${volume_id}" >&2
        exit 1
    fi
    mkdir -p "${TARGET_ROOT}/${volume_id}"
    cp -a "${SOURCE_ROOT}/${volume_id}/." "${TARGET_ROOT}/${volume_id}/"
    printf '%s\t%s\n' "${volume_id}" "$(du -sh "${TARGET_ROOT}/${volume_id}" | cut -f1)"
done

cp -a \
    data/outputs/volmem/verse_memflowdit_v0_5_minimal_gpu6/checkpoints/step_002300.pt \
    /dev/shm/memflowdit_checkpoints/v05_step_002300.pt

echo "staged_root=/dev/shm/memflowdit_moonvit_cache"
echo "staged_checkpoint=/dev/shm/memflowdit_checkpoints/v05_step_002300.pt"
