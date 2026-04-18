#!/bin/bash

# 毛刺分析脚本 - 使用V3.4模型

cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate snake1

# 设置配置文件
export CFG_FILE=configs/btcv_diffusion_dit_v3_4.yaml

# 分析多个样本
for idx in 0 1 2 3 4; do
    echo "=========================================="
    echo "分析样本 $idx"
    echo "=========================================="
    INDEX=$idx python analyze_burr_scatter.py
    echo ""
done

echo "所有分析完成！结果保存在 visual/burr_analysis/"
