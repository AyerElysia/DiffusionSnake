# Diffusion Snake V3 Series

V3 系列包含两个版本，核心区别在于**全局语义提取方式**。

---

## 版本对比

| 版本 | 全局语义提取 | README |
|------|-------------|--------|
| **V3.0** | Perceiver (256 learnable queries) | [README_V3_DENOISER.md](./README_V3_DENOISER.md) |
| **V3.1** | Patchify (ViT-style, 16×16 patches) | [README_V3_1_DENOISER.md](./README_V3_1_DENOISER.md) |

---

## 共同架构

### Block 结构 (两者相同)

```
Self-Attention → Cross-Attention → SwiGLU FFN
```

### 执行流程

```
1. Time Embedding: Sinusoidal → MLP
2. Global Context: Perceiver 或 Patchify
3. Local Context: sampled_feat → Linear projection
4. Point Embedding: Separate (coord + feature)
5. DiT Blocks × 6: Alternating global/local context
6. Output: FinalLayer (adaLN + Linear)
```

### 关键组件

| 组件 | 实现 |
|------|------|
| Normalization | RMSNorm |
| Attention | QK-Norm + Scaled Dot-Product |
| Position Encoding | CyclicRoPE (Self-Attn only) |
| FFN | SwiGLU |
| Conditioning | adaLN-Zero (9 params) |

---

## 全局语义提取对比

### V3.0: PerceiverCompressor

```python
# 256 个可学习 queries，无位置编码
queries = nn.Embedding(256, 256)

# Cross-Attention: queries attend to image
compressed = cross_attn(query=queries, key=image, value=image)
```

**特点**:
- ✅ Queries 自动学习关注不同区域
- ✅ 无需预设空间结构
- ❌ 无显式位置信息
- ❌ 需要 cross-attention 计算

### V3.1: PatchifyEmbedding

```python
# 非重叠卷积提取 patches
patches = Conv2d(64, 256, kernel_size=8, stride=8)(image)  # 128×128 → 16×16

# 添加 2D 位置编码
tokens = patches + pos_embed
```

**特点**:
- ✅ 显式空间位置编码
- ✅ 类似 ViT，易理解
- ✅ 无 cross-attention，显存更低
- ❌ 需要固定输入分辨率

---

## 配置选择

```yaml
# V3.0 (Perceiver)
use_dit_v3: true

# V3.1 (Patchify) - 推荐
use_dit_v3_1: true
```

---

## 选择建议

| 场景 | 推荐版本 |
|------|---------|
| 固定分辨率输入 | V3.1 |
| 需要显式空间信息 | V3.1 |
| 可变分辨率输入 | V3.0 |
| 显存受限 | V3.1 |
| 希望架构接近 ViT | V3.1 |

---

## 详细文档

- **V3.0 (Perceiver)**: [README_V3_DENOISER.md](./README_V3_DENOISER.md)
- **V3.1 (Patchify)**: [README_V3_1_DENOISER.md](./README_V3_1_DENOISER.md)