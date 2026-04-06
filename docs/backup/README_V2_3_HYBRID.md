# DiT V2.3 Hybrid: 混合动力 Flow Matching 演化模型

## 1. 概述 (Overview)
DiT V2.3 Hybrid 是 **Flow Matching (V2.3)** 范式与 **Hybrid (奇偶分层注入)** 架构的终极融合尝试。它利用 Rectified Flow 带来的直线采样优势，通过在不同层级交替赋予模型“全局定位意识”与“局部边缘触觉”，强制模型在 10 步 ODE 求解过程中，能够根据解剖学大背景（Global Patches）实时纠正如乱麻般的随机噪声位移。

## 2. 核心架构逻辑 (Architecture Philosophy)

| 阶段 | 实现方式 | 物理内涵 |
|:---|:---|:---|
| **预测目标** | Velocity Field (V_t) | 直线推力，从各向同性噪声推向精确解剖边界。 |
| **层级交织** | Global-Local Hybrid | 偶数层解决“我是谁、我在哪”；奇数层解决“边缘在哪”。 |
| **求解器** | ODE Euler Solver | 通过 10 步积分实现从 Gaussian 分布到 Delta 分布的映射。 |

## 3. 数据流拓扑图 (Detailed Topology)

```text
       ┌──────────┐        ┌──────────┐        ┌──────────┐
       │ 全局场景  │        │ 局部特征  │        │ 归一化 t │
       └────┬─────┘        └────┬─────┘        └────┬─────┘
            │                   │                   │
         Global-Y            Local-Y             Point-X
     (256 Image Toks)    (128 GCN Samps)    (Random Noise x0)
            │                   │                   │
            ▼                   ▼                   ▼
      ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
      ┃        Hybrid MM-DiT Blocks (6 Blocks)      ┃
      ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
      ┃  Layer 0: Joint-Attn(X, Global-Y) -> 本地化校准 ┃
      ┃  Layer 1: Joint-Attn(X, Local-Y)  -> 边缘探测   ┃
      ┃ [循环往复, 共 6 层迭代]                          ┃
      ┗━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┛
                            ┃
                 Predict Velocity Field (Vt)
                            ▼
      ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
      ┃    ODE Solver Loop (10 Euler-Steps)           ┃
      ┃    X_{t+dt} = X_{t} + V_{t} * dt              ┃
      ┗━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┛
                            ▼
                Refined Contour (X1) at t=1.0
```

## 4. 关键特性 (Key Features)
- **多模态博弈升级 (V2.3+)**：与之前的奇偶注入（V2.1）不同，本版本使用了 `JointDiTBlock`。即使是在奇数层与局部特征交互，图像 Token 本身也会由于残差连接而携带前一层的“全局位置信息”，从而实现了信息流的深度交织。
- **直线收敛**：强制的奇偶分层有助于 Flow Matching 模型在更少的 ODE 步数（如 10 步）内，获得比纯 Joint 注意力更加鲁棒的直线速度预测轨迹。

## 5. 配置参数 (Config)
```yaml
use_dit_v2_3: true
use_hybrid: true       # 开启混合动力注入
use_flow_matching: true
flow_ode_steps: 10
dit_num_layers: 6
dit_num_heads: 8
dit_state_dim: 256
```
