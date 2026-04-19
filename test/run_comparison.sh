#!/bin/bash
# V3.4 vs V3.8 简化对比脚本
# 使用现有的推理脚本进行对比

set -e

echo "=========================================="
echo "V3.4 vs V3.8 对比验证"
echo "=========================================="

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate snake1

cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30

# 创建输出目录
mkdir -p visual/comparison_v34_v38/{v3.4,v3.8}

echo ""
echo "步骤1: 推理 V3.4 (128点)..."
export CFG_FILE=configs/btcv_diffusion_dit_v3_4_single_overfit.yaml
python scripts/infer_all_versions.py \
    --ckpt data/outputs/btcv_diffusion_dit_v3_4_single_overfit/checkpoints/epoch_10000.pt \
    --output_dir visual/comparison_v34_v38/v3.4 \
    2>&1 | tee visual/comparison_v34_v38/v3.4/infer.log

echo ""
echo "步骤2: 推理 V3.8 (64点)..."
export CFG_FILE=configs/btcv_diffusion_dit_v3_8_single_overfit.yaml
python scripts/infer_all_versions.py \
    --ckpt data/outputs/btcv_diffusion_dit_v3_8_single_overfit/checkpoints/latest.pt \
    --output_dir visual/comparison_v34_v38/v3.8 \
    2>&1 | tee visual/comparison_v34_v38/v3.8/infer.log

echo ""
echo "步骤3: 分析对比结果..."
python test/analyze_comparison.py

echo ""
echo "=========================================="
echo "对比完成！结果保存在 visual/comparison_v34_v38/"
echo "=========================================="
