# DiT V2.1: Spatial Anchor Pooling Architecture

## 📋 概述
DiT V2.1 是对 V2.0 (Perceiver 版) 的关键修复。其核心目的是解决 Transformer 模块在全局特征提取时的“空间盲视 (Spatial Blindness)”问题。

## 🛠️ 技术改进
- **SpatialAnchorCompressor**: 放弃了 V2.0 中带有不可解释性的 Learnable Queries。
- **固定锚点网格**: 将 128x128 的特征图划分为固定的 16x16 锚点阵列，通过 Adaptive Average Pooling 提取局部特征。
- **2D 绝对位置编码**: 在每个锚点 Token 中显式注入 `(x, y)` 坐标的正弦位置编码，让网络明确感知图像的全局拓扑结构。

## 📈 收益
- **对齐精度提升**: 显著改善了轮廓点在复杂解剖结构（如重叠器管边缘）上的对齐鲁棒性。
- **训练收敛加速**: 物理坐标的引入降低了网络从零开始学习空间映射的负担。
