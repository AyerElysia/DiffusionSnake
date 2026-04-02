# DiT V2.2: MM-DiT Patchify Dual-Stream Architecture

## 1. 概述 (Overview)
DiT V2.2 采用了类似 **Stable Diffusion 3** 的 **Multi-Modal Diffusion Transformer (MM-DiT)** 架构。它彻底改变了“特征提取-去噪”的串行模式，转而让图像 Patch 与轮廓点在 Transformer 中进行对等的联合交互。

## 2. 核心组件 (Key Components)

| 模块 | 实现类/方法 | 功能描述 | 核心收益 |
|:---|:---|:---|:---|
| **Patchify 层** | `PatchifyEmbedding` (dit_blocks_v2_2.py) | 将图像特征切成 8x8 Patch 并展开。 | 消除了复杂的池化层，保留原始像素级的细节。 |
| **联合交互块** | `JointDiTBlock` | 实现双流 (Point-Image) 并行的 Self-Attn 与 Cross-Attn。 | 支持点流向背景学习，同时也支持背景根据点流自适应增强（联合表征）。 |
| **SwiGLU FFN** | `SwiGLU` | 每层后的门控线性单元激活。 | 相比 V2.1 极大增强了非线性表达能力。 |

## 3. ASCII 网络结构图 (Architecture Flow)

```text
       图像特征图 (P2)                     轮廓点位移 (x_t)
    [B, 64, 128, 128]                   [N, P, 2]
           │                               │
    ┌──────┴──────┐                 ┌──────┴──────┐
    │  Patchify    │                 │   Pt Embedding  │
    │ (8x8 stride) │                 │  (Separate)     │
    └──────┬──────┘                 └──────┬──────┘
           │                               │
    ┌──────┴──────┐                 ┌──────┴──────┐
    │  Image Tokens│                 │ Contour Tokens│
    │ [N, 256, 256]│                 │ [N, 128, 256] │
    └──────┬──────┘                 └──────┬──────┘
           │                               │
           │      ┏━━━━━━━━━━━━━━━━━━━━━━━━┓     ┌─────────────┐
           │      ┃ JointDiTBlock (MM-DiT) ┃     │ 时间步 t     │
           │      ┣━━━━━━━━━━━━━━━━━━━━━━━━┫     └──────┬──────┘
           ├─────►┃ Joint Self-Attention   ┃◄─────┤      │
           │      ┃ (Point <-> Image)      ┃      │      ▼
           │      ┣────────────────────────┫      │    adaLN-6
           ├─────►┃ Dual Feed-Forward      ┃◄─────┤  (6参数调制)
           │      ┗━━━━━━━━━━━━━┳━━━━━━━━━━┛      └─────────────┘
           │                    ┃
           ▼                    ▼
     (丢弃图像流)        预测速度场/噪声 (Δx)
```

## 4. 数据流动分析 (Data Flow)
1. **输入处理**：特征图不再被池化，而是通过卷积变为 Patch Tokens。
2. **MM-DiT 交互**：在每个 Block 内，轮廓点和 Image Patch 被 concat 到一起做 Self-Attention。这意味着：
   - 轮廓点可以看到每一个 Patch。
   - Patch 也能感知当前轮廓点的分布（类似于感知掩码）。
3. **这种互感机制**：使得 V2.2 在处理极其模糊的边界时，能通过图像流的对比度自适应增强来辅助轮廓收敛。

## 5. 配置参数 (Config)
```yaml
use_dit_v2_2: true
use_dit_v2_1: true # 继承点嵌入逻辑
dit_num_layers: 6
dit_num_heads: 8
dit_state_dim: 256
```
