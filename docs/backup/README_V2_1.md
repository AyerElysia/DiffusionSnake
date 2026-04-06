# DiT V2.1: Spatial Anchor Pooling Architecture

## 1. 概述 (Overview)
DiT V2.1 是针对 V2.0 (Perceiver) 的关键演进版本。V2.0 使用的 Perceiver 在提取特征时缺乏明确的空间引导，容易导致“空间盲视”。V2.1 引入了 **Spatial Anchor Compressor**，通过固定的空间网格进行特征池化，确保 Transformer 能够感知图像的全局几何拓扑。

## 2. 核心组件 (Key Components)

| 模块 | 实现类/方法 | 功能描述 | 核心收益 |
|:---|:---|:---|:---|
| **锚点压缩器** | `SpatialAnchorCompressor` (dit_denoiser_v2.py) | 将 128x128 特征图划分为 16x16 锚点阵列进行 Adaptive Pooling。 | 建立固定的特征投影坐标系，消除空间歧义。 |
| **2D 坐标注入** | `SineCosinePositionalEncoding` | 给每个锚点 Token 加入显式的 (x, y) 坐标编码。 | 让自注意力机制感知 Token 之间的物理距离。 |
| **调制网络** | `adaLN-Zero` (modified) | 使用时间步 t 动态控制每层的残差缩放。 | 确保扩散模型在不同噪声水平下具有不同的感受野。 |

## 3. ASCII 网络结构图 (Architecture Flow)

```text
┌──────────────────────────┐      ┌──────────────────────────────┐
│  CNN 特征图 (H x W x C)   │      │   轮廓点 x_t (Points, 2)      │
└────────────┬─────────────┘      └──────────────┬───────────────┘
             │                                   │
      ┌──────┴──────┐                     ┌──────┴──────┐
      │  划分锚点网格 │                     │   点特征采样 │
      │  (16x16 Grid)│                     │ (GCN Sample) │
      └──────┬──────┘                     └──────┬──────┘
             │                                   │
      ┌──────┴──────┐                     ┌──────┴──────┐
      │ 自适应平均池化 │                     │  分离点嵌入  │
      │ (Adaptive Pool)│                    │ (Pt Embed)   │
      └──────┬──────┘                     └──────┬──────┘
             │                                   │
      ┌──────┴──────┐                     ┌──────┴──────┐
      │ 2D 正余弦位置 │                     │  1D Cyclic-  │
      │   编码 (SPE)  │                     │  RoPE 编码   │
      └──────┬──────┘                     └──────┬──────┘
             │                                   │
      ┌──────┴──────┐                     ┌──────┴──────┐
      │  Visual Tokens  │                 │ Contour Tokens │
      │ (256, State_D)  │                 │ (128, State_D) │
      └──────┬──────┘                     └──────┬──────┘
             │                                   │
             │       ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓     ┌─────────────┐
             │       ┃    DiT V2.1 Transformer   ┃     │ 时间步 t     │
             │       ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫     └──────┬──────┘
             └───────┨ Cross-Attention(v_tok)    ┠───────────┤
                     ┣───────────────────────────┫           │
                     ┃ Self-Attention(c_tok)     ┃           ▼
                     ┣───────────────────────────┫     ┌─────────────┐
                     ┃       SwiGLU FFN          ┃     │  adaLN-Zero  │
                     ┗━━━━━━━━━━━━━┳━━━━━━━━━━━━━┛     │  (动态调制)  │
                                   ┃                   └─────────────┘
                                   ▼
                         预测位移 (Predicted Delta)
```

## 4. 数据流动分析 (Data Flow)
1. **输入阶段**：CNN 输出 $128 \times 128$ 特征图，`SpatialAnchorCompressor` 将其打散为 $256$ 个 $8 \times 8$ 的区域并取均值，得到全局 Token 序列。
2. **位置感官**：对比 V2.0 的纯黑盒 Perceiver，V2.1 的 Token 序列与图像空间是一一对应的。
3. **注入逻辑**：Transformer 层内，轮廓点作为 Query，全局锚点作为 Key/Value 进行 Cross-Attention，定位全局结构。

## 7. 详细架构拓扑 (Detailed Architecture)

```text
       ┌──────────┐        ┌──────────┐        ┌──────────┐
       │ 图像分支  │        │ 轮廓分支  │        │ 时间步 t │
       └────┬─────┘        └────┬─────┘        └────┬─────┘
            │                   │                   │
     Spatial Anchor Pool   Point Embedding        t_emb
     (16x16 Grid Token)    (Separate Coord)       [256]
     [B, 256, 256]         [N, 128, 256]            │
            │                   │                   │
            │           ┌───────┴───────┐           │
            ▼           ▼               ▼           ▼
      ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
      ┃           DiT V2.1 Backbone (6 Blocks)        ┃
      ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
      ┃  Layer 0: x ←[Cross-Attn]→ Global (16x16锚点) ┃
      ┃  Layer 1: x ←[Cross-Attn]→ Local  (局部边点)  ┃
      ┃  Layer 2: x ←[Cross-Attn]→ Global (16x16锚点) ┃
      ┃  Layer 3: x ←[Cross-Attn]→ Local  (局部边点)  ┃
      ┃  Layer 4: x ←[Cross-Attn]→ Global (16x16锚点) ┃
      ┃  Layer 5: x ←[Cross-Attn]→ Local  (局部边点)  ┃
      ┗━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┛
                            ┃
                    Final Projection
                            ▼
                Predicted Contour Delta (Δx)
```


