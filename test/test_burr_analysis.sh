#!/bin/bash

# 快速测试毛刺分析脚本

cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate snake1

# 设置配置文件
export CFG_FILE=configs/btcv_diffusion_dit_v3_4.yaml

echo "=========================================="
echo "测试毛刺分析脚本 - 样本0"
echo "=========================================="

INDEX=0 python analyze_burr_scatter.py

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo "查看结果："
echo "  ls -lh visual/burr_analysis/"
