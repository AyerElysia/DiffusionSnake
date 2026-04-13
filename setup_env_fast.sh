#!/bin/bash
# DiffusionSnake 快速环境配置脚本（使用清华镜像源）

set -e

echo "=== 配置 pip 使用清华镜像源 ==="
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== Installing PyTorch with CUDA support (清华源) ==="
pip install torch==1.13.1 torchvision==0.14.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== Installing core dependencies ==="
pip install numpy scipy matplotlib opencv-python Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tqdm pyyaml easydict tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== Installing deep learning libraries ==="
pip install timm einops -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install diffusers transformers accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== Installing YOLO dependencies ==="
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== Installing other utilities ==="
pip install pycocotools scikit-image scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install wandb termcolor ninja yacs cython -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== Environment setup complete! ==="
echo "Activate with: conda activate diffsnake"
echo "Test with: python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
