# DiT Denoiser V2 架构详解

DiT Denoiser V2 是 DiffusionSnake 的去噪骨干网络，全面升级自 V1 纯 Transformer 架构，融合了 2024-2025 年扩散模型与大型语言模型的最新设计范式。

---

## 整体架构

```
输入: cnn_feature (B,64,H,W), sampled_feat (N,64,P), x_t (N,P,2), t (N,)
  │
  ├─ 时间嵌入: t → Sinusoidal + MLP → t_emb (N, 256)
  │
  ├─ 全局上下文: cnn_feature → Perceiver → global_ctx (B,256,256) → [py_ind扩展] → (N,256,256)
  ├─ 局部上下文: sampled_feat → MLP投影 → local_ctx (N,P,256)
  │
  ├─ 点嵌入: x_t + sampled_feat → 分离MLP → x (N,P,256)
  │
  └─ DiTBlockV2 × 6 层
       ├─ Layer 0 (偶数): x ↔ global_ctx  交叉注意力 ← 全局语义
       ├─ Layer 1 (奇数): x ↔ local_ctx   交叉注意力 ← 局部边界
       ├─ Layer 2 (偶数): x ↔ global_ctx
       ├─ Layer 3 (奇数): x ↔ local_ctx
       ├─ Layer 4 (偶数): x ↔ global_ctx
       └─ Layer 5 (奇数): x ↔ local_ctx
  │
  └─ FinalLayer: x + t_emb → adaLN调制 → Linear → 输出 (N,P,2)
```

---

## 简化版架构图（适合截图）

```
┌────────────────────────────────────────────────────────────────┐
│                   DiT Denoiser V2 架构                          │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入                                                           │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌──────┐               │
│  │cnn_feat  │ │sampled   │ │  x_t    │ │  t   │               │
│  │(B,64,H,W)│ │(N,64,P)  │ │(N,P,2)  │ │ (N)  │               │
│  └────┬─────┘ └────┬─────┘ └────┬────┘ └──┬───┘               │
│       │            │            │         │                    │
│       ▼            ▼            ▼         ▼                    │
│  ┌────────────────────────────────────────────────┐            │
│  │ [M1] 多粒度上下文   [M2] 分离嵌入   时间嵌入    │            │
│  │                                                │            │
│  │  Perceiver          coord_MLP    Sinusoidal   │            │
│  │     │               feat_MLP         │        │            │
│  │     ▼                  │             ▼        │            │
│  │ global_ctx  local_ctx │         t_emb        │            │
│  │ (N,256,256) (N,P,256) ▼        (N,256)       │            │
│  │                      x(N,P,256)     │         │            │
│  └────────────────────────────────────────────────┘            │
│       │                │            │         │                │
│       └────────────────┴────────────┴─────────┘                │
│                          │                                      │
│                          ▼                                      │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              DiTBlockV2 × 6 层                          │    │
│  │                                                         │    │
│  │  每层结构:                                               │    │
│  │  ┌──────────────────────────────────────────────┐      │    │
│  │  │ t_emb ──► adaLN ──► [shift, scale, gate] × 3│      │    │
│  │  └──────────────────────────────────────────────┘      │    │
│  │                   │                                     │    │
│  │         ┌─────────┼─────────┐                          │    │
│  │         ▼         ▼         ▼                          │    │
│  │    ┌────────┐ ┌────────┐ ┌──────┐                     │    │
│  │    │Self-Att│ │Cross-Att│ │SwiGLU│                     │    │
│  │    │[M3+M4] │ │[M4.5]  │ │[M4.3]│                     │    │
│  │    │RoPE    │ │        │ │      │                     │    │
│  │    │QK-Norm │ │QK-Norm │ │      │                     │    │
│  │    └────┬───┘ └────┬───┘ └──┬───┘                     │    │
│  │         │          │        │                          │    │
│  │         ▼          ▼        ▼                          │    │
│  │    gate_sa×out gate_ca×out gate_ff×out                 │    │
│  │         │          │        │                          │    │
│  │         └──────────┴────────┘                          │    │
│  │                   │                                     │    │
│  │                   ▼                                     │    │
│  │            x = x + 残差                                 │    │
│  │  └─────────────────────────────────────────────────────┘    │
│  │                                                         │    │
│  │  Layer 0,2,4 ──► global_ctx (全局语义)                  │    │
│  │  Layer 1,3,5 ──► local_ctx  (局部边界)                  │    │
│  └─────────────────────────────────────────────────────────────┘
│                          │                                      │
│                          ▼                                      │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              [M6] FinalLayer                           │    │
│  │                                                         │    │
│  │  x + t_emb ──► adaLN ──► RMSNorm ──► Linear           │    │
│  │                                           │             │    │
│  │                                           ▼             │    │
│  │                                    eps_pred (N,P,2)     │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                        关键设计                                 │
├────────────────────────────────────────────────────────────────┤
│  [M1] 多粒度注入   偶数层global/奇数层local交替                 │
│  [M2] 分离嵌入     坐标64维 + 特征192维 独立MLP                │
│  [M3] Cyclic-RoPE  闭合轮廓旋转位置编码                        │
│  [M4] DiTBlockV2   RMSNorm + QK-Norm + SwiGLU + 9-param adaLN │
│  [M6] Final adaLN  输出层时间调制                              │
├────────────────────────────────────────────────────────────────┤
│  门控残差连接: x_out = x_in + gate(t) × SubLayer              │
│  Zero-Init: 训练初期所有gate=0，Block退化为恒等映射            │
└────────────────────────────────────────────────────────────────┘
```

---

## 网络架构 ASCII 可视化图

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                          DiT Denoiser V2 完整数据流                                    │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│   输入数据                                                                            │
│   ┌─────────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────┐                       │
│   │ cnn_feature │  │ sampled_feat │  │   x_t    │  │    t    │                       │
│   │ (B,64,H,W)  │  │  (N,64,P)    │  │ (N,P,2)  │  │   (N)   │                       │
│   └──────────┬──┘  └───────┬──────┘  └────┬─────┘  └────┬────┘                       │
│              │             │              │             │                             │
│   ┌──────────▼─────────────▼──────────────▼─────────────▼──────────┐                 │
│   │                     特征提取与嵌入阶段                            │                 │
│   ├─────────────────────────────────────────────────────────────────┤                 │
│   │                                                                  │                 │
│   │  ┌─────────────────┐                                             │                 │
│   │  │  时间嵌入网络    │                                             │                 │
│   │  │  t ──► Sinusoidal──► Linear──► SiLU──► Linear                │                 │
│   │  │       (64)        (256)       (256)                           │                 │
│   │  │       t_emb (N, 256) ──────────────────────────────────────► │ 所有 adaLN     │
│   │  └─────────────────┘                                             │                 │
│   │                                                                  │                 │
│   │  ┌───────────────────────────────────────────────────────────┐  │                 │
│   │  │ [M1] 多粒度视觉上下文                                        │  │                 │
│   │  │  cnn_feat ──► Perceiver ──► global_ctx (N,256,256)         │  │                 │
│   │  │  sampled   ──► MLP       ──► local_ctx  (N,P,256)          │  │                 │
│   │  │            (偶数层用global，奇数层用local)                  │  │                 │
│   │  └───────────────────────────────────────────────────────────┘  │                 │
│   │                                                                  │                 │
│   │  ┌───────────────────────────────────────────────────────────┐  │                 │
│   │  │ [M2] 分离点嵌入                                              │  │                 │
│   │  │  x_t(2)     ──► coord_MLP ──► (N,P,64)                     │  │                 │
│   │  │  feat(64)   ──► feat_MLP  ──► (N,P,192)                    │  │                 │
│   │  │  cat ──► x (N,P,256)                                       │  │                 │
│   │  └───────────────────────────────────────────────────────────┘  │                 │
│   └──────────────────────────────────────────────────────────────────┘                 │
│                                                                                       │
│   ┌──────────────────────────────────────────────────────────────────────────────┐   │
│   │                     DiTBlockV2 × 6 层处理阶段                                  │   │
│   ├──────────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                               │   │
│   │   x (N,P,256) + t_emb (N,256)                                                │   │
│   │         │                                                                     │   │
│   │         ▼                                                                     │   │
│   │   ┌─────────────────────────────────────────────────────────────────────┐   │   │
│   │   │  DiTBlockV2 Layer 0 [M3+M4] ─── 使用 global_ctx (N,256,256)          │   │   │
│   │   │                                                                      │   │   │
│   │   │  t_emb ──► SiLU ──► Linear(256→2304) ──► chunk(9)                    │   │   │
│   │   │                            │                                         │   │   │
│   │   │         [shift_sa, scale_sa, gate_sa]                                │   │   │
│   │   │         [shift_ca, scale_ca, gate_ca]                                │   │   │
│   │   │         [shift_ff, scale_ff, gate_ff]                                │   │   │
│   │   │                                                                      │   │   │
│   │   │  ┌─────────────────────────────────────────────────────────────┐    │   │   │
│   │   │  │ Self-Attention [M3: CyclicRoPE + M4: QK-Norm]                │    │   │   │
│   │   │  │                                                               │    │   │   │
│   │   │  │  x ──► RMSNorm ──► modulate(shift_sa, scale_sa) ──► x_sa     │    │   │   │
│   │   │  │  x_sa ──► QKV投影 ──► QK-Norm ──► RoPE旋转                   │    │   │   │
│   │   │  │       ──► Attention ──► sa_out (N,P,256)                     │    │   │   │
│   │   │  │  x = x + gate_sa × sa_out  ← [M4.8] 门控残差                 │    │   │   │
│   │   │  └─────────────────────────────────────────────────────────────┘    │   │   │
│   │   │                                                                      │   │   │
│   │   │  ┌─────────────────────────────────────────────────────────────┐    │   │   │
│   │   │  │ Cross-Attention [使用 global_ctx 或 local_ctx]              │    │   │   │
│   │   │  │                                                               │    │   │   │
│   │   │  │  x ──► RMSNorm ──► modulate(shift_ca, scale_ca) ──► x_ca     │    │   │   │
│   │   │  │  x_ca ──► CrossAttn(context) ──► ca_out (N,P,256)           │    │   │   │
│   │   │  │  x = x + gate_ca × ca_out  ← [M4.5+M4.8] 时间门控残差(V2新)  │    │   │   │
│   │   │  └─────────────────────────────────────────────────────────────┘    │   │   │
│   │   │                                                                      │   │   │
│   │   │  ┌─────────────────────────────────────────────────────────────┐    │   │   │
│   │   │  │ SwiGLU FFN [M4.3]                                            │    │   │   │
│   │   │  │                                                               │    │   │   │
│   │   │  │  x ──► RMSNorm ──► modulate(shift_ff, scale_ff) ──► x_ff     │    │   │   │
│   │   │  │  x_ff ──► W1(⊙)Swish(V) ──► W2 ──► ffn_out (N,P,256)       │    │   │   │
│   │   │  │  x = x + gate_ff × ffn_out  ← [M4.8] 门控残差                │    │   │   │
│   │   │  └─────────────────────────────────────────────────────────────┘    │   │   │
│   │   │                                                                      │   │   │
│   │   │  输出 x (N,P,256) ──► 下一层                                         │   │   │
│   │   └─────────────────────────────────────────────────────────────────────┘   │   │
│   │         │                                                                     │   │
│   │         ▼                                                                     │   │
│   │   Layer 1 (local_ctx) ──► Layer 2 (global) ──► Layer 3 (local)              │   │
│   │         │                                                                     │   │
│   │         ▼                                                                     │   │
│   │   Layer 4 (global) ──► Layer 5 (local) ──► x_out (N,P,256)                  │   │
│   │                                                                               │   │
│   └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                       │
│   ┌──────────────────────────────────────────────────────────────────────────────┐   │
│   │                     [M6] FinalLayer 输出阶段                                   │   │
│   ├──────────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                               │   │
│   │   x_out (N,P,256) + t_emb (N,256)                                            │   │
│   │         │                                                                     │   │
│   │         ▼                                                                     │   │
│   │   t_emb ──► SiLU ──► Linear(256→512) ──► [shift, scale]                      │   │
│   │   x_out ──► RMSNorm ──► modulate(shift, scale) ──► x_mod                     │   │
│   │   x_mod ──► Linear(256→2) ──► eps_pred (N,128,2)                             │   │
│   │                                                                               │   │
│   └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              关键设计标注                                             │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  [M1] 多粒度注入      奇偶层交替使用 global/local context                           │
│  [M2] 分离点嵌入      坐标64维 + 特征192维 独立MLP                                  │
│  [M3] Cyclic-RoPE     闭合轮廓循环旋转位置编码 (θ_i = 2π×i/P)                      │
│  [M4.1] RMSNorm       替代LayerNorm，更快更稳定                                     │
│  [M4.2] QK-Norm       防止注意力logits爆炸                                         │
│  [M4.3] SwiGLU        门控前馈网络，隐藏维704 (dim×8/3)                            │
│  [M4.4] 9-param adaLN shift/scale/gate × 3子模块 = 9参数                          │
│  [M4.5] CA时间门控    V2新增，交叉注意力时间条件控制                               │
│  [M4.8] 门控残差      gate(t) × SubLayer输出                                       │
│  [M6] Final adaLN     输出层时间调制                                               │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                              维度说明                                                │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  B: 图像批大小         N: 轮廓批大小 (可能 N > B)                                  │
│  P: 轮廓点数 = 128     H,W: 特征图尺寸 (如 128×128)                                │
│  dim/state_dim: 256    head_dim: 256/8 = 32                                       │
│  hidden_dim: 256×8/3 ≈ 704 (GPU对齐到64倍数)                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────┐
│                          残差连接详细流程 (单个 DiTBlockV2)                           │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│   输入: x (N,P,256), t_emb (N,256), context (N,L,256)                               │
│                                                                                       │
│   ┌───────────────────────────────────────────────────────────────────────────┐     │
│   │  adaLN 调制参数生成                                                        │     │
│   │                                                                            │     │
│   │  t_emb (N,256) ──► SiLU ──► Linear(256→2304) ──► chunk(9)                 │     │
│   │                                   │                                        │     │
│   │                                   ▼                                        │     │
│   │         ┌────────────────────────────────────────────────┐                 │     │
│   │         │  shift_sa (N,256)  scale_sa (N,256)  gate_sa   │                 │     │
│   │         │  shift_ca (N,256)  scale_ca (N,256)  gate_ca   │                 │     │
│   │         │  shift_ff (N,256)  scale_ff (N,256)  gate_ff   │                 │     │
│   │         └────────────────────────────────────────────────┘                 │     │
│   └───────────────────────────────────────────────────────────────────────────┘     │
│                                                                                       │
│   ┌───────────────────────────────────────────────────────────────────────────┐     │
│   │  Self-Attention 分支                                                       │     │
│   │                                                                            │     │
│   │  x ──► RMSNorm ──► x × (1 + scale_sa) + shift_sa ──► x_sa                 │     │
│   │                                         │                                  │     │
│   │                                         ▼                                  │     │
│   │                              Self-Attention(x_sa)                          │     │
│   │                              (含QK-Norm + RoPE)                            │     │
│   │                                         │                                  │     │
│   │                                         ▼                                  │     │
│   │                              sa_out (N,P,256)                              │     │
│   │                                         │                                  │     │
│   │                                         ▼                                  │     │
│   │         ┌───────────────────────────────────────────────────┐              │     │
│   │         │  门控残差: gate_sa.unsqueeze(1) × sa_out          │              │     │
│   │         │           (N,1,256)    ×    (N,P,256)             │              │     │
│   │         │                    ↓                              │              │     │
│   │         │              gated_sa (N,P,256)                   │              │     │
│   │         └───────────────────────────────────────────────────┘              │     │
│   │                                         │                                  │     │
│   │                                         ▼                                  │     │
│   │         ┌───────────────────────────────────────────────────┐              │     │
│   │         │  残差加法: x = x + gated_sa                        │              │     │
│   │         │           (N,P,256) + (N,P,256)                   │              │     │
│   │         └───────────────────────────────────────────────────┘              │     │
│   └───────────────────────────────────────────────────────────────────────────┘     │
│                                                                                       │
│   ┌───────────────────────────────────────────────────────────────────────────┐     │
│   │  Cross-Attention 分支 (同理)                                               │     │
│   │                                                                            │     │
│   │  x ──► RMSNorm ──► modulate ──► CrossAttn(context) ──► ca_out             │     │
│   │  x = x + gate_ca.unsqueeze(1) × ca_out                                     │     │
│   └───────────────────────────────────────────────────────────────────────────┘     │
│                                                                                       │
│   ┌───────────────────────────────────────────────────────────────────────────┐     │
│   │  SwiGLU FFN 分支 (同理)                                                    │     │
│   │                                                                            │     │
│   │  x ──► RMSNorm ──► modulate ──► SwiGLU ──► ffn_out                        │     │
│   │  x = x + gate_ff.unsqueeze(1) × ffn_out                                    │     │
│   └───────────────────────────────────────────────────────────────────────────┘     │
│                                                                                       │
│   输出: x (N,P,256)                                                                  │
│                                                                                       │
│   最终效果:                                                                           │
│   x_out = x_in + gate_sa×SA + gate_ca×CA + gate_ff×FFN                             │
│   每个gate由时间步t控制，实现时间条件的动态残差贡献                                     │
│                                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 设计一：多粒度视觉上下文注入（M1）

### 问题

V1 中仅使用 Perceiver 压缩的全局特征作为交叉注意力的 context。全局特征擅长理解器官/结构的整体语义（"这是椎体"），但在局部边界细节上（"这个边缘在哪里"）精度不足。

### 方案

同时保留两条视觉特征路径，在骨干网的不同层交替注入：

```python
# 全局路径：Perceiver 压缩 → 形状级语义
self.global_compressor = PerceiverCompressor(
    in_dim=64, out_dim=256, num_queries=256
)
# 输入: (B, 64, H, W) → 输出: (B, 256, 256)

# 局部路径：逐点采样特征投影 → 边界级细节
self.local_proj = nn.Sequential(
    nn.Linear(64, 256), nn.SiLU(), nn.Linear(256, 256)
)
# 输入: (N, 64, P) → 输出: (N, P, 256)
```

### 交替注入策略

```python
for i, dit_layer in enumerate(self.dit_layers):
    context = global_ctx if (i % 2 == 0) else local_ctx
    x = dit_layer(x, context, t_emb)
```

| 层 | Context 类型 | 作用 |
|----|-------------|------|
| 0, 2, 4 | 全局上下文 (256 tokens) | 理解整图语义，建立轮廓与解剖结构的全局对应 |
| 1, 3, 5 | 局部上下文 (P=128 tokens) | 感知局部梯度，精确拟合组织边界 |

### 批次对齐

全局特征来自图像批次 (B)，而轮廓批次 (N) 可能不同（一张图对应多个轮廓）。通过 `py_ind` 映射实现精确对齐：

```python
global_ctx = global_ctx[py_ind]  # (B, 256, 256) → (N, 256, 256)
```

### 为什么有效

- **早期层**接收全局语义，帮助轮廓"知道自己在哪类结构上"
- **中间层**接收局部细节，帮助轮廓"找到精确的边界位置"
- **交替模式**让信息在宏观与微观之间反复校准，类似人类"先看整体、再看细节"的标注过程

---

## 设计二：分离点嵌入（M2）

### 问题

V1 将坐标（2 维）和特征（64 维）直接拼接后通过一个线性层投影：

```python
# V1: cat([x_t(2), feat(64)], dim=-1) → Linear(66, 256)
```

2 维坐标信息在 66 维输入中被严重稀释，网络难以充分利用精确的几何位置信息。

### 方案

为坐标和特征分别设计独立的 MLP，按 1:3 比例分配维度容量：

```python
coord_dim = state_dim // 4   # 256 / 4 = 64
feat_dim = state_dim - coord_dim  # 256 - 64 = 192

self.coord_embed = nn.Sequential(
    nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64),
)
self.feat_embed = nn.Sequential(
    nn.Linear(64, 192), nn.SiLU(), nn.Linear(192, 192),
)
```

### 数据流

```
x_t (N, P, 2)     → coord_embed → (N, P, 64)
sampled_feat (N, 64, P) → feat_embed → (N, P, 192)
拼接 → (N, P, 256)
```

### 为什么有效

- 坐标获得 **64 维专属空间**（而非 V1 中线性混合后的模糊表示）
- 特征获得 **192 维专属空间**，有足够的容量编码纹理和语义
- 两层 MLP（而非单层 Linear）提供非线性变换能力，更好地提取各自模态的信息
- 参考 ContourFormer (CVPR 2025) 的分离编码策略

---

## 设计三：Cyclic-RoPE 1D 循环旋转位置编码（M3）

### 问题

V1 使用可加式的 SnakePosEncoding（固定正弦余弦位置编码），存在两个缺陷：
1. 可加式编码在深层网络中容易被残差路径稀释
2. 无法自然表达"第 0 个点和第 127 个点相邻"的闭合轮廓拓扑

### 方案

采用 FLUX.1 风格的逐层旋转位置编码（RoPE），并针对闭合轮廓做循环映射：

```python
class CyclicRoPE1D(nn.Module):
    def __init__(self, head_dim: int, num_points: int = 128):
        # 频率带：几何级数 1/10000^(2i/d)
        freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
```

### 核心机制

**位置角度映射**：将 P 个轮廓点映射到 `[0, 2π)` 的循环角度上

```
θᵢ = 2π × i / P    (i = 0, 1, ..., P-1)
```

这意味着 `i=0` 和 `i=P-1` 的角度差仅为 `2π/P`，天然编码了闭合轮廓的首尾相邻关系。

**旋转操作**：对 Q 和 K 的每对维度应用 2D 旋转

```python
# 将 Q/K 拆分为 (even, odd) 对
x1 = x[..., 0::2]  # 偶数维度
x2 = x[..., 1::2]  # 奇数维度
# 旋转
out1 = x1 * cos(θ) - x2 * sin(θ)
out2 = x1 * sin(θ) + x2 * cos(θ)
```

### 关键性质

| 性质 | 说明 |
|------|------|
| 闭合拓扑 | 点 0 和点 P-1 的角度差极小，自然表达相邻关系 |
| 相对位置 | 注意力分数仅依赖 `(θᵢ - θⱼ) mod 2π`，与起点选择无关 |
| 起点兼容 | 与 `pretrain_evolution.py` 中的轮廓对齐算法完全兼容 |
| 逐层注入 | 每层独立应用（FLUX 风格），而非仅在输入端加一次 |

### 为什么只在自注意力中使用

交叉注意力的 context（全局/局部视觉特征）来自不同的序列空间，具有不同的位置语义，因此不应用 RoPE。仅对轮廓点序列（自注意力）施加循环位置编码。

---

## 设计四：DiTBlockV2 全面升级（M4）

DiTBlockV2 是 V2 的核心计算单元，相比 V1 进行了 7 项升级：

### 4.1 RMSNorm 替代 LayerNorm

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)
```

| 对比项 | LayerNorm | RMSNorm |
|--------|-----------|---------|
| 计算 | `x - mean` → `/ std` | `x / rms` |
| 均值中心化 | 需要 | 不需要 |
| 速度 | 较慢 | 更快 |
| 稳定性 | 好 | 更好（少一步减法） |

RMSNorm 去除了均值中心化操作，仅保留缩放不变性。这是 LLaMA 3、PaLM 2、Mistral、DeepSeek 等 2024 年主流模型的标准选择。

**为什么可以去掉均值中心化**：adaLN 的 shift 参数已经提供了平移调制能力，Norm 本身只需负责缩放稳定。

### 4.2 QK-RMSNorm 防止注意力爆炸

```python
# 在注意力计算前，对每个 head 的 Q 和 K 做 RMSNorm
q = self.rope.apply_rotary(self.qk_norm(q))
k = self.rope.apply_rotary(self.qk_norm(k))
```

深层 Transformer 中，Q 和 K 的范数可能随层数增长而增大，导致注意力 logits `(Q·K^T) / sqrt(d)` 数值过大，softmax 后梯度消失。QK-Norm 将 Q/K 约束到单位球面上，保证注意力分数始终在合理范围内。

**操作顺序**：先 QK-Norm，再 RoPE 旋转。因为 RoPE 是正交变换，不改变向量范数，所以 Norm 在前在后等价，但先 Norm 再旋转更符合直觉。

### 4.3 SwiGLU 前馈网络

```python
class SwiGLU(nn.Module):
    def forward(self, x):
        return self.w2(F.silu(self.v(x)) * self.w1(x))
```

**结构**：
```
x ──→ W1 ──→ gate ──┐
                    ├──→ ⊙ ──→ W2 ──→ output
x ──→ V  ──→ Swish ─┘
```

**2/3 隐藏维规则**：为保持与标准 SiLU-MLP 相同的 FLOPs，隐藏维设为 `dim × 8/3`（而非 `dim × 4`）：

```python
hidden_dim = int(dim * 8 / 3)  # 256 × 8/3 ≈ 683 → 对齐到 704
hidden_dim = ((hidden_dim + 63) // 64) * 64  # GPU 内存对齐
```

**为什么有效**：门控机制让网络可以动态选择哪些特征通过，比静态激活函数（ReLU/GELU/SiLU）具有更强的表达能力。这是 LLaMA 系列模型的核心设计之一。

### 4.4 9 参数 adaLN-Zero 全门控

V1 仅在自注意力和 FFN 上应用 adaLN（6 参数），交叉注意力没有门控。V2 将 adaLN 扩展到全部三个子模块：

```python
# 9 参数：SA(3) + CA(3) + FFN(3)
shift_sa, scale_sa, gate_sa, \
shift_ca, scale_ca, gate_ca, \
shift_ff, scale_ff, gate_ff = mod.chunk(9, dim=1)
```

每个子模块获得三组调制参数：

| 参数 | 作用 | 公式 |
|------|------|------|
| shift | 平移归一化后的特征 | `x * (1 + scale) + shift` |
| scale | 缩放归一化后的特征 | 同上 |
| gate | 控制残差连接的贡献 | `x + gate * sublayer_output` |

**Zero-init 初始化**：

```python
nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
```

训练初期，所有 gate=0、scale=0、shift=0，整个 Block 退化为恒等映射。这保证了深层网络的训练稳定性——模型从"什么都不做"开始，逐步学习有用的变换。

### 4.5 交叉注意力时间门控（V2 新增）

V1 中交叉注意力没有门控：
```python
# V1: 无 gate，交叉注意力始终全量注入
x = x + self.cross_attn(x_norm, image_context, image_context)[0]
```

V2 中交叉注意力获得独立的 adaLN 门控：
```python
# V2: gate_ca 由时间嵌入控制
x_ca = modulate(self.norm2(x), shift_ca, scale_ca)
x = x + gate_ca.unsqueeze(1) * self._cross_attention(x_ca, image_context)
```

**意义**：在扩散早期（高噪声阶段），模型可以学会 `gate_ca ≈ 0`，忽略图像上下文（因为此时预测主要依赖先验）；在扩散后期（低噪声阶段），`gate_ca ≈ 1`，充分利用图像细节精修轮廓。

### 4.8 门控残差连接的完整实现

DiTBlockV2 的核心创新之一是**时间条件的门控残差连接**。每个子模块（Self-Attention、Cross-Attention、FFN）都有独立的可学习 gate 参数，由时间嵌入动态控制残差连接的贡献强度。

#### 传统残差连接 vs 门控残差连接

**传统 Transformer 残差**（如 BERT、GPT）：
```python
x = x + self.sublayer(self.norm(x))  # 固定系数 = 1
```

问题：残差贡献固定为 1，无法根据时间步动态调整。

**DiT V1 残差**：
```python
# adaLN-Zero：仅 SA 和 FFN 有门控
x = x + gate_msa.unsqueeze(1) * self.self_attn(x_norm)
x = x + self.cross_attn(x_norm2, context)  # CA 无门控
x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm3)
```

问题：Cross-Attention 缺乏时间条件控制。

**DiT V2 门控残差**（全部三个子模块）：
```python
# 从时间嵌入预测 9 个调制参数
mod = self.adaLN_modulation(t_emb)  # (N, 9 * dim)
shift_sa, scale_sa, gate_sa, \
shift_ca, scale_ca, gate_ca, \
shift_ff, scale_ff, gate_ff = mod.chunk(9, dim=1)

# Self-Attention 门控残差
x_sa = modulate(self.norm1(x), shift_sa, scale_sa)
x = x + gate_sa.unsqueeze(1) * self._self_attention(x_sa)

# Cross-Attention 门控残差
x_ca = modulate(self.norm2(x), shift_ca, scale_ca)
x = x + gate_ca.unsqueeze(1) * self._cross_attention(x_ca, image_context)

# SwiGLU FFN 门控残差
x_ff = modulate(self.norm3(x), shift_ff, scale_ff)
x = x + gate_ff.unsqueeze(1) * self.mlp(x_ff)
```

#### 数学公式

每个子模块的残差连接遵循：

```
x_out = x_in + gate(t) × SubLayer(modulate(Norm(x_in)))
```

其中：
- `gate(t)`：由时间嵌入 `t_emb` 通过 MLP 生成，形状 `(N, dim)`，经 `unsqueeze(1)` 扩展为 `(N, 1, dim)`
- `modulate(x, shift, scale) = x × (1 + scale) + shift`
- `SubLayer`：注意力或 FFN 操作
- `Norm`：RMSNorm（不带可学习 affine 参数）

#### 维度广播机制

```python
gate_sa.unsqueeze(1)  # (N, dim) → (N, 1, dim)
self._self_attention(x_sa)  # 输出形状 (N, P, dim)
gate_sa.unsqueeze(1) * output  # (N, 1, dim) × (N, P, dim) → (N, P, dim)
x + gated_output  # (N, P, dim) + (N, P, dim) → (N, P, dim)
```

gate 在 token 维度（P）上广播，所有 128 个轮廓点共享同一个 gate 值，但 gate 本身由时间步决定，随每条轮廓（N）独立。

#### 时间步对 gate 的影响

| 扩散阶段 | 时间步 t | gate 典型值 | 语义 |
|----------|---------|------------|------|
| 高噪声 | t ≈ 1000 | gate ≈ 0 | 子模块贡献被压制，特征主要依赖输入 |
| 中噪声 | t ≈ 500 | gate ≈ 0.5 | 子模块部分参与，信息渐进融合 |
| 低噪声 | t ≈ 100 | gate ≈ 1 | 子模块全量注入，精修细节 |

这种动态门控让模型在不同扩散阶段灵活分配计算资源：
- **高噪声阶段**：预测噪声主要依赖全局统计规律，局部上下文（Cross-Attention）贡献应较低
- **低噪声阶段**：需要精细利用图像细节，Cross-Attention 和 FFN 应更活跃

#### 与 Zero-Init 的协同

```python
nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
```

初始化后：
- `gate_sa = 0`, `gate_ca = 0`, `gate_ff = 0`
- 整个 Block 的前向传播：`x_out = x_in + 0 × SubLayer(...) = x_in`

训练初期，所有层都是恒等映射，梯度可以无损通过深层网络。随着训练进行，模型逐步学会在何时（哪个时间步）、何处（哪个子模块）启用残差连接。

#### 实验验证的必要性

门控残差连接的理论优势需要实验验证：
1. **是否学到有意义的时间-门控关系**？可可视化 `gate(t)` 曲线
2. **门控值范围**？理论上可以是任意实数，但训练后通常落在 `[-1, 2]` 范围
3. **CA 的 gate 是否随噪声水平变化**？这是 V2 的核心改进点

### 4.6 手动 QKV 投影

V1 使用 `nn.MultiheadAttention` 封装，V2 改为手动投影：

```python
self.q_proj = nn.Linear(dim, dim, bias=False)
self.k_proj = nn.Linear(dim, dim, bias=False)
self.v_proj = nn.Linear(dim, dim, bias=False)
self.sa_out_proj = nn.Linear(dim, dim, bias=False)
```

**原因**：
- 需要在 Q/K 上分别应用 QK-Norm 和 RoPE，封装接口不支持
- 手动投影提供更细粒度的控制
- 移除 bias 配合 Norm 使用，避免冗余参数

### 4.7 Dropout = 0

```python
dropout: float = 0.0  # 默认关闭
```

DiT 原论文指出，在大规模扩散模型中，Dropout 会损害生成质量。扩散模型本身通过噪声注入和逐步去噪已经具有强正则化效果，额外的 Dropout 反而破坏学习到的去噪轨迹。

---

## 设计五：adaLN 调制输出头（M6）

### 问题

V1 的输出头是简单的 `LayerNorm + Linear`，与时间步无关。但 DiT 论文发现，在最终层应用 adaLN 调制对生成质量至关重要。

### 方案

```python
class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int = 2):
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )
        # 全部零初始化
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, t_emb):
        shift, scale = self.adaLN(t_emb).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)
```

### 为什么有效

扩散模型在不同时间步需要输出不同量级的预测值：
- **高噪声阶段**（t 大）：预测值量级大，需要更大的 scale
- **低噪声阶段**（t 小）：预测值量级小，需要更精细的 scale

adaLN 让输出层的缩放和平移参数随时间步动态调整，确保输出尺度与扩散噪声水平严格匹配。

### 双重零初始化

`adaLN` 和 `linear` 同时零初始化，保证训练初期整个输出头退化为零映射。这与 DiT Block 的 zero-init 策略一致，确保模型从恒等映射开始稳定训练。

---

## 时间嵌入

沿用 V1 已验证的设计：

```python
self.time_emb_net = nn.Sequential(
    SinusoidalTimeEmbedding(dim=64),   # 正弦位置编码
    nn.Linear(64, 256),                # 投影到 state_dim
    nn.SiLU(),                         # 非线性
    nn.Linear(256, 256),               # 最终嵌入
)
```

生成的 `t_emb (N, 256)` 被送入每个 DiTBlockV2 的 adaLN 调制器和 FinalLayer，作为全网络的时间条件信号。

---

## 数据流完整示例

以 `state_dim=256, num_layers=6, num_heads=8, num_points=128` 为例：

```
输入:
  cnn_feature:  (B, 64, 128, 128)   YOLOv8 P2 特征图
  sampled_feat: (N, 64, 128)        轮廓点处采样的局部特征
  x_t:          (N, 128, 2)         含噪位移/轮廓坐标
  t:            (N,)                扩散时间步

Step 1 - 时间嵌入:
  t → Sinusoidal(64) → Linear → SiLU → Linear → t_emb (N, 256)

Step 2 - 多粒度视觉上下文 [M1]:
  cnn_feature → Perceiver → global_ctx (B, 256, 256)
    → [py_ind 扩展] → (N, 256, 256)
  sampled_feat → transpose → MLP → local_ctx (N, 128, 256)

Step 3 - 分离点嵌入 [M2]:
  x_t (N,128,2)     → coord_embed → (N, 128, 64)
  sampled_feat (N,64,128) → feat_embed → (N, 128, 192)
  拼接 → x (N, 128, 256)

Step 4 - DiTBlockV2 × 6 [M3 + M4 + 残差连接]:
  每层内部（以单层为例，输入 x (N,P,256), t_emb (N,256), context (N,L,256)):
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                    DiTBlockV2 单层数据流                         │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  t_emb (N,256)                                                   │
    │      │                                                           │
    │      ▼                                                           │
    │  adaLN_modulation: SiLU → Linear(256→2304)                       │
    │      │                                                           │
    │      ▼                                                           │
    │  chunk(9) → [shift_sa, scale_sa, gate_sa,                        │
    │              shift_ca, scale_ca, gate_ca,                        │
    │              shift_ff, scale_ff, gate_ff]  各 (N,256)            │
    │                                                                  │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │ Self-Attention 分支                                        │  │
    │  │                                                            │  │
    │  │  x (N,P,256)                                               │  │
    │  │      │                                                     │  │
    │  │      ▼                                                     │  │
    │  │  RMSNorm(x) → (N,P,256)                                    │  │
    │  │      │                                                     │  │
    │  │      ▼                                                     │  │
    │  │  modulate: x × (1+scale_sa) + shift_sa → x_sa (N,P,256)    │  │
    │  │      │                                                     │  │
    │  │      ▼                                                     │  │
    │  │  Self-Attention(QK-Norm + CyclicRoPE) → sa_out (N,P,256)  │  │
    │  │      │                                                     │  │
    │  │      ▼                                                     │  │
    │  │  gate_sa.unsqueeze(1) × sa_out → (N,P,256)                 │  │
    │  │      │                                                     │  │
    │  │      ▼                                                     │  │
    │  │  残差加法: x = x + gated_sa_out                            │  │
    │  └───────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │ Cross-Attention 分支                                       │  │
    │  │                                                            │  │
    │  │  x (更新后, N,P,256)                                       │  │
    │  │      │                                                     │  │
    │  │      ▼                                                     │  │
    │  │  RMSNorm(x) → modulate(shift_ca, scale_ca) → x_ca          │  │
    │  │      │                                                     │  │
    │  │      ▼                                                     │  │
    │  │  Cross-Attention(x_ca, context) → ca_out (N,P,256)        │  │
    │  │      │                                                     │  │
    │  │      ▼                                                     │  │
    │  │  gate_ca.unsqueeze(1) × ca_out → gated_ca_out             │  │
    │  │      │                                                     │  │
    │  │      ▼                                                     │  │
    │  │  残差加法: x = x + gated_ca_out                            │  │
    │  └───────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │ SwiGLU FFN 分支                                            │  │
    │  │                                                            │  │
    │  │  x (更新后, N,P,256)                                       │  │
    │  │      │                                                     │  │
    │  │      ▼                                                     │  │
    │  │  RMSNorm(x) → modulate(shift_ff, scale_ff) → x_ff          │  │
    │  │      │                                                     │  │
    │  │      ▼                                                     │  │
    │  │  SwiGLU: (x_ff·V ⊙ Swish(x_ff·W1)) · W2 → ffn_out         │  │
    │  │      │                                                     │  │
    │  │      ▼                                                     │  │
    │  │  gate_ff.unsqueeze(1) × ffn_out → gated_ffn_out            │  │
    │  │      │                                                     │  │
    │  │      ▼                                                     │  │
    │  │  残差加法: x = x + gated_ffn_out                           │  │
    │  └───────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  输出: x (N,P,256) → 传入下一层或 FinalLayer                    │
    └─────────────────────────────────────────────────────────────────┘
    
    注：偶数层 context=global_ctx (N,256,256)，奇数层 context=local_ctx (N,P,256)

Step 4.1 - 6 层残差累积效果:
  经过 6 层 DiTBlockV2 后，输入特征 x 被层层叠加修改：
  
  x_0 (初始嵌入, N,P,256)
    │
    ├─ Layer 0: x_1 = x_0 + gate_sa_0×SA_0 + gate_ca_0×CA_0 + gate_ff_0×FFN_0
    ├─ Layer 1: x_2 = x_1 + gate_sa_1×SA_1 + gate_ca_1×CA_1 + gate_ff_1×FFN_1
    ├─ Layer 2: x_3 = x_2 + gate_sa_2×SA_2 + gate_ca_2×CA_2 + gate_ff_2×FFN_2
    ├─ Layer 3: x_4 = x_3 + gate_sa_3×SA_3 + gate_ca_3×CA_3 + gate_ff_3×FFN_3
    ├─ Layer 4: x_5 = x_4 + gate_sa_4×SA_4 + gate_ca_4×CA_4 + gate_ff_4×FFN_4
    ├─ Layer 5: x_6 = x_5 + gate_sa_5×SA_5 + gate_ca_5×CA_5 + gate_ff_5×FFN_5
    │
    ▼
  x_out (N,P,256) → FinalLayer
  
  展开后：
  x_out = x_0 + Σ_{i=0..5} (gate_sa_i×SA_i + gate_ca_i×CA_i + gate_ff_i×FFN_i)
  
  每个子模块的最终贡献 = 时间条件门控 × 子模块输出
  这确保了信息流动被时间步精确控制

Step 5 - FinalLayer [M6]:
  x (N,128,256) + t_emb (N,256)
    → adaLN 调制 → RMSNorm → Linear → pred (N, 128, 2)

输出:
  eps_pred: (N, 128, 2)  预测的噪声/位移
```

---

## 配置方式

```yaml
# 启用 DiT V2
use_diffusion_evolution: true
use_dit_v2: true

# DiT V2 架构参数
dit_num_layers: 6        # DiTBlockV2 层数
dit_num_heads: 8         # 注意力头数
dit_state_dim: 256       # 隐藏层维度
```

---

## 参考文献

| 技术 | 来源 |
|------|------|
| DiT / adaLN-Zero | Peebles & Xie, "Scalable Diffusion Models with Transformers" (ICCV 2023) |
| FLUX.1 逐层 RoPE | Black Forest Labs, "FLUX.1" (2024) |
| SwiGLU | Shazeer, "GLU Variants Improve Transformer" (2020) |
| RMSNorm | Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019) |
| QK-Norm | "Transformers without Tears" (2024) |
| RoPE | Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" |
| 分离编码 | ContourFormer (CVPR 2025) |
| Perceiver | Jaegle et al., "Perceiver IO" (ICLR 2022) |
