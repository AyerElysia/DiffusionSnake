# DiT Denoiser V3.0 - Perceiver Semantics

## 架构概览

```
================================================================================
                        DiT DENOISER V3.0 ARCHITECTURE
================================================================================

Input:
  cnn_feature:  (B, 64, H, W)    -- YOLO P2 feature map
  sampled_feat: (N, 64, P)       -- Per-point sampled features (local)
  x_t:          (N, P, 2)        -- Noisy displacement
  t:            (N,)             -- Timesteps
  py_ind:       (N,)             -- Contour-to-image index

================================================================================
[STAGE 1: EMBEDDING]
================================================================================

  Time t ──────────────────────────────────────────────────────────────────────►
                    │
                    ▼
          ┌─────────────────────┐
          │ SinusoidalTimeEmbed │ ──► Linear ──► SiLU ──► Linear ──► t_emb (N, 256)
          └─────────────────────┘


  cnn_feature (B, 64, H, W) ─────► ┌──────────────────────┐
                                   │ PerceiverCompressor  │
                                   │  - 256 learnable     │
                                   │    queries           │
                                   │  - Cross-Attention   │
                                   └──────────────────────┘
                                              │
                                              ▼
                                    global_ctx (B, 256, 256)
                                              │
                                    py_ind indexing
                                              │
                                              ▼
                                    global_ctx (N, 256, 256)


  sampled_feat (N, 64, P) ───────► Transpose ──► Linear ──► SiLU ──► Linear
                                                                       │
                                                                       ▼
                                                              local_ctx (N, P, 256)


  x_t (N, P, 2) ─────────────────► ┌──────────────────────────┐
          +                        │ SeparatePointEmbedding   │
  sampled_feat (N, 64, P) ───────► │  - coord_embed: 2 → 64   │
                                  │  - feat_embed: 64 → 192  │
                                  └──────────────────────────┘
                                              │
                                              ▼
                                       x (N, P, 256)

================================================================================
[STAGE 2: DiT BLOCKS V3 × 6 LAYERS]
================================================================================

  For each layer i in range(6):
    context = global_ctx if (i % 2 == 0) else local_ctx  # Alternating

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        DiTBlockV3                                        │
    │                                                                          │
    │  x ──► norm1 ──► adaLN(shift_sa, scale_sa) ──► Self-Attention ──► + ──► │
    │         │                              │              │                  │
    │         │                              ▼              │                  │
    │         │                    QK-Norm + CyclicRoPE    │                  │
    │         │                              │              │                  │
    │         └──────────────────────────────┴──────────────┘                  │
    │                                                                          │
    │  x ──► norm2 ──► adaLN(shift_ca, scale_ca) ──► Cross-Attention ──► + ──►│
    │         │                              │              │                  │
    │         │                              ▼              │                  │
    │         │                         QK-Norm             │                  │
    │         │                              │              │                  │
    │         └──────────────────────────────┴──────────────┘                  │
    │                                                                          │
    │  x ──► norm3 ──► adaLN(shift_ff, scale_ff) ──► SwiGLU FFN ──► + ──►    │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
[STAGE 3: OUTPUT]
================================================================================

  x (N, P, 256) ──► FinalLayer(adaLN + Linear) ──► pred (N, P, 2)

================================================================================
```

## 核心组件

### 1. PerceiverCompressor (全局语义提取)

**作用**: 将 H×W 的特征图压缩为 256 个语义 token

**实现** (`dit_blocks.py:118-173`):
```python
class PerceiverCompressor(nn.Module):
    def __init__(self, in_dim=64, out_dim=256, num_queries=256):
        # 可学习的 queries（无位置编码）
        self.queries = nn.Embedding(num_queries, out_dim)
        
        # Cross-Attention: queries attend to image features
        self.cross_attn = nn.MultiheadAttention(embed_dim=out_dim, num_heads=8)
        
    def forward(self, image_feat):
        # image_feat: (B, 64, H, W) → (B, H*W, 256)
        img = self.input_proj(image_feat.flatten(2).transpose(1, 2))
        
        # Queries cross-attend to image
        queries = self.queries.weight.expand(B, -1, -1)
        compressed, _ = self.cross_attn(query=queries, key=img, value=img)
        
        return compressed  # (B, 256, 256)
```

**特点**:
- 256 个可学习 queries，通过 cross-attention 自动关注不同图像区域
- 无显式位置编码，让网络自行学习空间对应关系
- 计算量: O(256 × H×W)，相比全注意力减少约 64 倍

### 2. DiTBlockV3 (Transformer Block)

**执行顺序**: Self-Attention → Cross-Attention → FFN

**实现** (`dit_blocks_v3.py:22-128`):
```python
class DiTBlockV3(nn.Module):
    def forward(self, x, image_context, t_emb):
        # 1. Self-Attention (点间协调，带 CyclicRoPE)
        x_sa = modulate(self.norm1(x), shift_sa, scale_sa)
        x = x + gate_sa.unsqueeze(1) * self._self_attention(x_sa)
        
        # 2. Cross-Attention (与图像上下文交互)
        x_ca = modulate(self.norm2(x), shift_ca, scale_ca)
        x = x + gate_ca.unsqueeze(1) * self._cross_attention(x_ca, image_context)
        
        # 3. SwiGLU FFN (非线性变换)
        x_ff = modulate(self.norm3(x), shift_ff, scale_ff)
        x = x + gate_ff.unsqueeze(1) * self.mlp(x_ff)
        
        return x
```

**关键特性**:
| 特性 | 实现 | 作用 |
|------|------|------|
| QK-Norm | `RMSNorm(head_dim)` | 稳定注意力计算 |
| CyclicRoPE | 周期位置编码 | 闭合轮廓建模 |
| adaLN-Zero | 9 参数调制 | 时间条件控制 |
| SwiGLU | 门控 FFN | 更强表达能力 |

### 3. Alternating Context Injection

**策略**: 偶数层用全局语义，奇数层用局部特征

```python
for i, dit_layer in enumerate(self.dit_layers):
    context = global_ctx if (i % 2 == 0) else local_ctx
    x = dit_layer(x, context, t_emb)
```

**设计意图**:
- 全局语义 (256 tokens): 提供形状约束和上下文
- 局部特征 (128 tokens): 提供边界细节和精确位置

---

## 配置示例

```yaml
# configs/btcv_diffusion_dit_v3.yaml
use_diffusion_evolution: true
use_dit_v3: true                    # 启用 V3.0
dit_num_layers: 6
dit_num_heads: 8
dit_state_dim: 256
diffusion_timesteps: 1000
use_ddim_inference: true
```

---

## 与其他版本对比

| 特性 | V3.0 | V3.1 | V2 |
|------|------|------|-----|
| 全局语义 | Perceiver (256 queries) | Patchify (16×16) | Perceiver |
| Block 结构 | Self→Cross→FFN | Self→Cross→FFN | Self→Cross→FFN |
| 位置编码 | Queries 无位置 | 2D 可学习 PE | - |
| 显存占用 | 中 | 低 | 中 |
| 空间感知 | 隐式学习 | 显式编码 | 隐式 |

---

## 参数量统计

```
DiTDenoiserV3:
├── time_emb_net:        262,400
├── global_compressor:   1,051,136
├── local_proj:           131,328
├── point_embed:           65,600
├── dit_layers (×6):   12,582,912
└── final_layer:         131,330
────────────────────────────────────
Total:                  ~14.2M
```

---

## 使用示例

```python
from lib.networks.diffusion.dit_denoiser_v3 import DiTDenoiserV3

denoiser = DiTDenoiserV3(
    state_dim=256,
    feature_dim=64,
    num_layers=6,
    num_heads=8,
    num_points=128,
)

# Forward
pred, L = denoiser(
    cnn_feature,    # (B, 64, H, W)
    sampled_feat,   # (N, 64, P)
    x_t,            # (N, P, 2)
    t,              # (N,)
    py_ind=py_ind,  # (N,)
)
```

---

## 参考文献

1. **Perceiver IO**: Jaegle et al., "Perceiver IO: A General Architecture for Structured Inputs & Outputs", ICLR 2022
2. **DiT**: Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023
3. **CyclicRoPE**: Adapted from RoFormer for closed contour modeling
4. **SwiGLU**: Shazeer, "GLU Variants Improve Transformer", 2020