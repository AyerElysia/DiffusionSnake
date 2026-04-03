# DiT V2.2 Hybrid: 混合动力 MM-DiT 扩散模型

## 1. 概述 (Overview)
DiT V2.2 Hybrid 是 **V2.1 (奇偶注入)** 与 **V2.2 (MM-DiT)** 的深度集成。它旨在解决 MM-DiT 在统一注意力空间下可能出现的“注意力涣散”问题。通过分层切换注意力对手（全图 Patch vs 局部特征），模型在保留双流博弈优势的同时，强行建立了“全局对齐”与“局部修缘”的训练节拍。

## 2. 核心架构逻辑 (Architecture Philosophy)

| 维度 | 处理方式 | 物理含义 |
|:---|:---|:---|
| **偶数层 (0, 2, 4)** | `JointAttention(X, Y_Global)` | **远焦模式**：点流向 256 个全图 Patch 寻求空间对齐和宏观定位。 |
| **奇数层 (1, 3, 5)** | `JointAttention(X, Y_Local)` | **近焦模式**：点流向 128 个采样点寻求局部高频边缘纹理。 |
| **交互范式** | MM-DiT (Joint Stream) | X 与 Y 不再是单向注入，而是对等的联合交互更新。 |

## 3. 数据流拓扑图 (Detailed Topology)

```text
       ┌──────────┐        ┌──────────┐        ┌──────────┐
       │ 图像 Patch│        │ 局部采样  │        │ 时间步 t │
       └────┬─────┘        └────┬─────┘        └────┬─────┘
            │                   │                   │
         Global-Y            Local-Y             Point-X
     (256 Image Toks)    (128 GCN Samps)      (Noisy Disp)
            │                   │                   │
            ▼                   ▼                   ▼
      ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
      ┃        Hybrid MM-DiT Blocks (6 Blocks)      ┃
      ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
      ┃  Layer 0: Joint-Attn(X, Global-Y) -> 全局定位 ┃
      ┃  Layer 1: Joint-Attn(X, Local-Y)  -> 细节精修 ┃
      ┃  Layer 2: Joint-Attn(X, Global-Y) -> 空间校准 ┃
      ┃  Layer 3: Joint-Attn(X, Local-Y)  -> 边缘增强 ┃
      ┃  Layer 4: Joint-Attn(X, Global-Y) -> 拓扑对齐 ┃
      ┃  Layer 5: Joint-Attn(X, Local-Y)  -> 最终收尾 ┃
      ┗━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┛
                            ┃
                    Final LN-Modulation
                            ▼
                Predicted Noise Epsilon (ε)
```

## 4. 数据流动分析 (Data Flow)
1. **输入阶段**：同时生成 256 个全局 Patch 和 128 个局部点的特征采样作为两套 Context。
2. **时分复用**：在 6 层 Backbone 中，偶数层将点流拼入全局 Patch 流，奇数层将点流拼入局部采样流。
3. **注意力机制**：虽然 $Y$ 序列的长度在层间发生变化 ($256 \leftrightarrow 128$)，但由于 Transformer 的自注意力是位置对齐的且映射维度 $D$ 恒定，计算过程无缝衔接。

## 5. 配置参数 (Config)
```yaml
use_dit_v2_2: true
use_hybrid: true       # 开启奇偶注入模式
dit_num_layers: 6
dit_num_heads: 8
dit_state_dim: 256
```
