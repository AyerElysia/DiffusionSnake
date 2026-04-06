# DiT Denoiser V3.1 - Patchify Semantics

## 架构概览

```
================================================================================
                        DiT DENOISER V3.1 ARCHITECTURE
================================================================================

Input:
  cnn_feature:  (B, 64, H, W)    -- YOLO P2 feature map (typically 128×128)
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
                                   │  PatchifyEmbedding   │
                                   │  - Conv2d(p=8, s=8)  │
                                   │  - 16×16 patches     │
                                   │  - 2D pos_embed      │
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
[STAGE 2: DiT BLOCKS V3.1 × 6 LAYERS]
================================================================================

  For each layer i in range(6):
    context = global_ctx if (i % 2 == 0) else local_ctx  # Alternating

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        DiTBlockV3_1                                     │
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

### 1. PatchifyEmbedding (全局语义提取)

**作用**: 将 H×W 特征图转换为 patch tokens，类似 ViT

**实现** (`dit_blocks_v3_1.py:30-71`):
```python
class PatchifyEmbedding(nn.Module):
    """
    ViT-style patch embedding for image features.
    
    For 128×128 feature map with patch_size=8:
      - Output: 16×16 = 256 patches
      - Each patch: 8×8 region → 256-dim vector
    """
    
    def __init__(self, in_channels=64, patch_size=8, out_dim=256):
        super().__init__()
        self.patch_size = patch_size
        
        # Non-overlapping convolution = patch extraction
        self.proj = nn.Conv2d(in_channels, out_dim, 
                              kernel_size=patch_size, stride=patch_size)
        
        # 2D absolute positional embedding
        max_grid = 16  # Supports up to 16×16 = 256 patches
        self.pos_embed = nn.Parameter(torch.zeros(1, max_grid * max_grid, out_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - e.g., (B, 64, 128, 128)
        Returns:
            (B, 256, out_dim) - e.g., (B, 256, 256)
        """
        B, C, H, W = x.shape
        
        # Patch extraction via conv
        x = self.proj(x)           # (B, out_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, out_dim)
        
        # Add positional embedding
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]
        
        return x
```

**与 Perceiver 的对比**:

| 特性 | PatchifyEmbedding (V3.1) | PerceiverCompressor (V3.0) |
|------|--------------------------|---------------------------|
| 压缩方式 | 非重叠卷积 | Cross-Attention |
| 位置信息 | 显式 2D 位置编码 | Queries 无位置编码 |
| 空间对应 | 直接保留 | 隐式学习 |
| 可解释性 | 高（类似 ViT） | 低（latent queries） |
| 计算量 | O(H×W/p²) | O(num_queries × H×W) |

**适用场景**:
- ✅ 输入分辨率固定（如 BTCV 的 128×128）
- ✅ 需要显式空间位置信息
- ✅ 希望减少显存占用（无 cross-attention）

### 2. DiTBlockV3_1 (Transformer Block)

**结构与 V3.0 完全相同**: Self-Attention → Cross-Attention → FFN

**实现** (`dit_blocks_v3_1.py:77-206`):
```python
class DiTBlockV3_1(nn.Module):
    """
    Same structure as V3:
    1. Self-Attention (with CyclicRoPE)
    2. Cross-Attention (with image context)
    3. FFN (SwiGLU)
    
    Uses adaLN-Zero with 9 modulation parameters.
    """
    
    def forward(self, x, image_context, t_emb):
        # 1. Self-Attention
        x = x + gate_sa.unsqueeze(1) * self._self_attention(x_sa)
        
        # 2. Cross-Attention  
        x = x + gate_ca.unsqueeze(1) * self._cross_attention(x_ca, image_context)
        
        # 3. FFN
        x = x + gate_ff.unsqueeze(1) * self.mlp(x_ff)
        
        return x
```

**注意**: V3.1 Block 与 V3.0 Block 代码完全一致，仅命名不同。

### 3. Alternating Context Injection

**策略**: 与 V3.0 相同，偶数层用全局语义，奇数层用局部特征

```python
for i, dit_layer in enumerate(self.dit_layers):
    context = global_ctx if (i % 2 == 0) else local_ctx
    x = dit_layer(x, context, t_emb)
```

---

## 配置示例

```yaml
# configs/btcv_diffusion_dit_v3_1.yaml
use_diffusion_evolution: true
use_dit_v3_1: true                   # 启用 V3.1
dit_num_layers: 6
dit_num_heads: 8
dit_state_dim: 256
diffusion_timesteps: 1000
use_ddim_inference: true
```

---

## V3.1 vs V3.0 选择指南

### 选择 V3.1 (Patchify) 的场景

- ✅ 输入分辨率固定
- ✅ 希望显式保留空间位置信息
- ✅ 需要更低的显存占用
- ✅ 希望架构更接近标准 ViT，便于理解和调试

### 选择 V3.0 (Perceiver) 的场景

- ✅ 需要处理可变分辨率输入
- ✅ 希望 queries 自动学习关注区域
- ✅ 有充足的显存预算

### 性能对比建议

建议在相同数据集上训练两个版本，对比：
- IoU / Dice
- Hausdorff 距离
- 训练收敛速度
- 推理延迟

---

## 参数量统计

```
DiTDenoiserV3_1:
├── time_emb_net:        262,400
├── image_embed:         1,049,600    # Patchify (vs Perceiver: 1,051,136)
├── local_proj:           131,328
├── point_embed:           65,600
├── dit_layers (×6):   12,582,912
└── final_layer:         131,330
────────────────────────────────────
Total:                  ~14.2M        # 与 V3.0 相近
```

---

## 使用示例

```python
from lib.networks.diffusion.dit_denoiser_v3_1 import DiTDenoiserV3_1

denoiser = DiTDenoiserV3_1(
    state_dim=256,
    feature_dim=64,
    num_layers=6,
    num_heads=8,
    num_points=128,
    patch_size=8,  # 128×128 → 16×16 patches
)

# Forward
pred, L = denoiser(
    cnn_feature,    # (B, 64, 128, 128)
    sampled_feat,   # (N, 64, P)
    x_t,            # (N, P, 2)
    t,              # (N,)
    py_ind=py_ind,  # (N,)
)
```

---

## Patchify 可视化

```
输入: 128×128×64 特征图

┌─────────────────────────────────────┐
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐  │
│  │ p │ p │ p │ p │ p │ p │ p │ p │  │
│  ├───┼───┼───┼───┼───┼───┼───┼───┤  │    p = 8×8 patch
│  │ p │ p │ p │ p │ p │ p │ p │ p │  │    每个 patch → 256-dim token
│  ├───┼───┼───┼───┼───┼───┼───┼───┤  │
│  │ p │ p │ p │ p │ p │ p │ p │ p │  │    16×16 = 256 tokens
│  ├───┼───┼───┼───┼───┼───┼───┼───┤  │
│  │...│...│...│...│...│...│...│...│  │
│  ├───┼───┼───┼───┼───┼───┼───┼───┤  │
│  │ p │ p │ p │ p │ p │ p │ p │ p │  │
│  └───┴───┴───┴───┴───┴───┴───┴───┘  │
└─────────────────────────────────────┘

输出: 256×256 (256 tokens, 256-dim each)
+ 2D positional embedding
```

---

## 参考文献

1. **ViT**: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021
2. **DiT**: Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023
3. **CyclicRoPE**: Adapted from RoFormer for closed contour modeling
4. **SwiGLU**: Shazeer, "GLU Variants Improve Transformer", 2020