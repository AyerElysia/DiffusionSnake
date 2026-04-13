#!/bin/bash
# DiffusionSnake 环境配置脚本

set -e

echo "=== Installing PyTorch with CUDA support ==="
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

echo "=== Installing core dependencies ==="
pip install numpy scipy matplotlib opencv-python Pillow
pip install tqdm pyyaml easydict tensorboard

echo "=== Installing deep learning libraries ==="
pip install timm einops
pip install diffusers transformers accelerate

echo "=== Installing YOLO dependencies ==="
pip install ultralytics

echo "=== Installing other utilities ==="
pip install pycocotools scikit-image scikit-learn
pip install wandb

echo "=== Installing additional dependencies ==="
pip install termcolor ninja yacs cython pycocotools

echo "=== Compiling custom CUDA extensions (if any) ==="
cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30
if [ -d "lib/csrc/extreme_utils" ]; then
    cd lib/csrc/extreme_utils
    python setup.py build_ext --inplace || echo "Warning: extreme_utils compilation failed"
    cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30
fi

echo "=== Environment setup complete! ==="
echo "Activate with: conda activate diffsnake"
echo "Test with: python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
