# DiT-Snake v2.2 架构升级：多模态双流联合注意力 (MM-DiT / Patchify)

## 📌 核心理论支撑 (Literature References)
本方案旨在将 2024 年最前沿的扩散 Transformer 架构引入医学图像轮廓演化，参考以下顶级工作：
1. **DiT (Peebles & Xie, ICCV 2023)**: 确立了 `Patchify` 作为视觉扩散模型进入 Transformer 的标准入口。
2. **Stable Diffusion 3 / MM-DiT (Esser et al., CVPR 2024)**: 提出了 **Joint Attention (联合注意力)** 机制，让不同模态的 Token 在同一个计算通道里进行全双工通信。

---

## 🔍 从 V2.1 到 V2.2 的质变
- **V2.1 (当前)**: 依然是“单向”的。图像特征被压缩成固定 Context 后，由轮廓点去通过 Cross-Attention “查找”。图像特征本身在 Transformer 块之间是不进化的。
- **V2.2 (目标)**: 进化为**“双流交互”**。图像被 Patch 化为序列，轮廓点也被处理为序列。两者融合成一个超长序列进入 **Joint Transformer Block**。

### 带来的科研优势：
- **互适应性 (Mutual Adaptation)**: 图像 Patch 能感知到轮廓点的位置，反之亦然。
- **全局一致性**: 图像 Patch 互相对话（Self-Attention），能够从更高维度理顺解剖结构，纠正轮廓点的局部错误。

---

## 💡 v2.2 核心模块设计

### 1. 图像端：纯正 Patchify 模块
- **输入**: $128 \times 128 \times 64$ (YOLO P2 特征)。
- **操作**: 使用 $8 \times 8$ 无重叠卷积核。
- **输出**: $16 \times 16 = 256$ 个 Token，每个维度 $D=256$。
- **PE**: 注入标准的 **2D Sine-Cosine Positional Embedding**。

### 2. 接头层：联合调制 (Joint Modulation)
- 仿照 SD3，为轮廓流 (Contour Stream) 和图像流 (Image Stream) 设置各自独立的 **adaLN-Zero** 调制参数。

### 3. 主干层：JointDiTBlock (MM-DiT)
每一层 Block 包含：
1. **Joint Attention**:
   - 将 `(128 Contour Tokens)` 和 `(256 Image Tokens)` 拼接为 `384` 长度的序列。
   - 使用统一的 Attention 矩阵进行运算。
   - 实现 **Contour-to-Contour**, **Image-to-Image**, **Contour-to-Image** 的全连接交互。
2. **Dual MLP (FFN)**:
   - 交互完成后，将序列分拆。
   - 轮廓序列进入 `FFN_contour`，图像序列进入 `FFN_image`。

---

## 🛠️ 实施路线图 (Task List)

- [ ] **[New File]** 创建 `lib/networks/diffusion/dit_blocks_v2_2.py`
    - 实现 `PatchifyCompressor`
    - 实现 `JointDiTBlock` (双投影、长序列 Attention、双 FFN)
- [ ] **[New File]** 创建 `lib/networks/diffusion/dit_denoiser_v2_2.py`
    - 组建双流骨干网络
- [ ] **[Update]** 更新 `pretrain_evolution.py` 以注册 V2.2 选择开关
- [ ] **[New Config]** 创建 `configs/btcv_diffusion_dit_v2_2.yaml`

---

## 🚨 关键决策 & 风险提示 (User Review Required)

> [!IMPORTANT]
> 1. **参数爆炸**: V2.2 的双流设计意味着权重翻倍（两个 FFN），模型参数量会从 13M 提升至约 **26M**。
> 2. **显存消耗**: 拼接后的序列长度为 384，显存占用会略有上升，对于 24GB 的 4090 D 来说仍有巨大余裕。
> 3. **训练策略**: 此版本彻底改变了特征流动，属于**全新架构实验**。

**本方案不仅具备极高的工业性能潜力，更具备极强的学术创新价值。如果您确认要探索这个“最先进”分支，请批准此计划！**
