#!/bin/bash
# 检查 diffsnake 环境状态

echo "========================================="
echo "DiffusionSnake 环境状态检查"
echo "========================================="
echo ""

echo "1. 环境位置："
conda env list | grep diffsnake
echo ""

echo "2. Python 版本："
conda run -n diffsnake python --version
echo ""

echo "3. 已安装的包："
conda run -n diffsnake pip list | head -20
echo "... (显示前20个包)"
echo ""

echo "4. PyTorch 状态："
conda run -n diffsnake python -c "
try:
    import torch
    print(f'✓ PyTorch {torch.__version__} 已安装')
    print(f'✓ CUDA 可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✓ CUDA 版本: {torch.version.cuda}')
        print(f'✓ GPU 数量: {torch.cuda.device_count()}')
except Exception as e:
    print(f'✗ PyTorch 未正确安装: {e}')
" 2>&1
echo ""

echo "5. 关键依赖检查："
for pkg in numpy scipy matplotlib opencv-python tqdm pyyaml ultralytics wandb; do
    conda run -n diffsnake python -c "import ${pkg%%[*} 2>/dev/null && echo '✓ $pkg' || echo '✗ $pkg'"
done
echo ""

echo "========================================="
echo "如何使用："
echo "  conda activate diffsnake"
echo "  cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30"
echo "  python diffusion_train.py"
echo "========================================="
