# DiT V2.2: MM-DiT Dual-Stream Joint Attention

## 📋 概述
DiT V2.2 彻底重构了扩散演化逻辑，参考了 **Stable Diffusion 3 (SD3)** 的设计理念，引入了多模态双流交互机制。

## 🛠️ 技术改进
- **Patchify Embedding**: 将图像特征图切分为 8x8 的 Patch，类似于 ViT 的处理方式，保留更精细的局部纹理特征。
- **MM-DiT 双流架构**: 
  - **Stream A**: 轮廓点坐标坐标向量。
  - **Stream B**: 图像 Patch 特征向量。
- **JointDiTBlock**: 每个 Block 中，两个流进行并行的 Self-Attention，并交叉进行 Cross-Attention（Joint Attention），使轮廓点能实时感知当前像素位置的特征强度。

## 📈 收益
- **图像感知的深度化**: 解决了 V2.1 中特征压缩过猛导致的信息丢失，轮廓点现在具有“极佳的视力”，能够精准吸附到细小的纹理边缘。
