# DiT V2 架构升级实施报告

**日期**: 2026-04-02  
**Git Commit**: `7a6e875` (`feat: DiT V2 architecture upgrade`)  
**状态**: ✅ 实施完成，所有测试通过

---

## 一、工作概要

基于前期对 20 篇 2023-2025 年核心论文的系统研究，本次实施将 DiffusionSnake 的 DiT 去噪器从 V1 全面升级为 V2。主要改动包括 6 项架构改进（M1-M6），均有明确的论文支撑，且与现有代码完全兼容。

### 完成清单

| # | 改进项 | 文件 | 状态 |
|---|--------|------|------|
| M1 | 多尺度视觉特征融合 | `dit_denoiser_v2.py` | ✅ |
| M2 | 分离点嵌入 | `dit_blocks_v2.py` | ✅ |
| M3 | Cyclic-RoPE 逐层旋转位置编码 | `dit_blocks_v2.py` | ✅ |
| M4a | RMSNorm 替换 LayerNorm | `dit_blocks_v2.py` | ✅ |
| M4b | SwiGLU 替换 SiLU-MLP | `dit_blocks_v2.py` | ✅ |
| M4c | QK-RMSNorm | `dit_blocks_v2.py` | ✅ |
| M4d | Cross-Attention adaLN 门控 (6→9 param) | `dit_blocks_v2.py` | ✅ |
| M6 | Final adaLN 输出头 | `dit_blocks_v2.py` | ✅ |
| — | GRPO kwargs 转发 bug 修复 | `grpo_evolution.py` | ✅ |
| — | BTCV DiT V2 配置文件 | `btcv_diffusion_dit_v2.yaml` | ✅ |

---

## 二、新增/修改文件清单

### 新增文件

| 文件 | 说明 | 行数 |
|------|------|------|
| `lib/networks/diffusion/dit_blocks_v2.py` | V2 核心构建块：RMSNorm、SwiGLU、CyclicRoPE1D、SeparatePointEmbedding、DiTBlockV2、FinalLayer | ~280 行 |
| `lib/networks/diffusion/dit_denoiser_v2.py` | V2 完整去噪器，集成所有 M1-M6 改进 | ~156 行 |
| `configs/btcv_diffusion_dit_v2.yaml` | BTCV 数据集 DiT V2 训练配置 | ~80 行 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `lib/networks/diffusion/pretrain_evolution.py` | 1. 导入 `DiTDenoiserV2`<br>2. 新增 `use_dit_v2` 参数（三级选择 V2>V1>GCN）<br>3. `predict_eps` 识别 V2 实例 |
| `lib/networks/diffusion/ct_snake.py` | 传递 `use_dit_v2` 配置参数 |
| `lib/networks/diffusion/__init__.py` | 导出 `DiTDenoiserV2` |
| `lib/networks/diffusion/grpo_evolution.py` | **Bug 修复**: `**kwargs` 未转发给 `super().__init__()` |

---

## 三、架构对比：V1 vs V2

### 3.1 结构对比

```
DiTDenoiser V1                          DiTDenoiserV2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SinusoidalTimeEmb → MLP                 SinusoidalTimeEmb → MLP (same)
PerceiverCompressor (global only)       PerceiverCompressor + LocalProj [M1]
cat(x_t, feat) → Linear                SeparatePointEmbedding [M2]
SnakePosEncoding (additive, fixed)      CyclicRoPE1D (rotary, per-layer) [M3]
DiTBlock × 6:                           DiTBlockV2 × 6:
  LayerNorm                               RMSNorm [M4a]
  MHA (no QK-norm)                        SA + QK-RMSNorm + RoPE [M4c]
  ×  (no cross-attn gate)                 CA + adaLN gate [M4d]
  SiLU → Linear FFN                       SwiGLU FFN [M4b]
  6 adaLN params                          9 adaLN params [M4d]
LayerNorm + Linear                      FinalLayer + adaLN [M6]
```

### 3.2 参数量对比

| 模型 | 参数量 | 可训练 | 增长 |
|------|--------|--------|------|
| DiTDenoiser V1 | 9.13M | 9.13M | — |
| **DiTDenoiserV2** | **10.64M** | **10.64M** | **+16.6%** |

### 3.3 V2 各模块参数分布

| 模块 | 参数量 | 说明 |
|------|--------|------|
| `time_emb_net` | 0.082M | 时间嵌入 (与 V1 相同) |
| `global_compressor` | 0.346M | 全局 Perceiver (与 V1 相同) |
| `local_proj` | 0.082M | **新增** — 局部特征投影 |
| `point_embed` | 0.054M | **改进** — 分离的坐标/特征嵌入 |
| `dit_layers` | 9.947M | **升级** — 6 个 DiTBlockV2 |
| `final_layer` | 0.132M | **新增** — adaLN 输出头 |

---

## 四、各改进详解

### M1: 多尺度视觉特征融合

**问题**: V1 只用 Perceiver 压缩全局特征，丢失边界细节  
**方案**: 保留全局 Perceiver + 新增局部逐点特征投影  
**实现**: DiT 偶数层用全局特征 cross-attend，奇数层用局部特征  
**成本**: +0.082M 参数（`local_proj` — 一个两层 MLP）

### M2: 分离点嵌入

**问题**: `cat(x_t[2], feat[64])` 中 2 维坐标被 64 维特征淹没  
**方案**: 坐标用独立 MLP (2→64)，特征用独立 MLP (64→192)，拼接得 256  
**论文**: ContourFormer (CVPR 2025) 分离编码策略

### M3: CyclicRoPE1D

**问题**: 固定 additive PE 与起始点对齐冲突；不逐层传播  
**方案**: `θᵢ = 2π·i/P` 的旋转位置编码，对 Q/K 逐层注入  
**兼容**: RoPE 编码相对位置 `(i-j) mod P`，与起始点对齐完美兼容  
**论文**: FLUX.1 (2024) 验证 per-layer RoPE >> input-only PE

### M4: DiTBlockV2 全面升级

| 子项 | 改进 | 论文 |
|------|------|------|
| M4a RMSNorm | 去除均值中心化，更快更稳 | LLaMA 3, PaLM 2 |
| M4b SwiGLU | 门控 FFN，2/3 rule 保持 FLOPs | Shazeer 2020, LLaMA |
| M4c QK-Norm | 防止 attention logit 爆炸 | 2024 best practice |
| M4d CA Gate | Cross-Attention 也有时间步门控 | SD3/MMDiT |

### M6: Final adaLN

**问题**: V1 输出头无时间步信息  
**方案**: 最终层使用 adaLN 调制 + zero-init  
**论文**: DiT 原论文强调 "final layer norm matters"

---

## 五、Bug 修复

### GRPO kwargs 转发

`grpo_evolution.py` 的 `__init__` 使用 `**kwargs` 参数但未转发给 `super().__init__()`。
这意味着 `use_dit_denoiser`、`dit_num_layers` 等参数在 GRPO 模式下被**静默丢弃**。

```diff
 super().__init__(
     state_dim=state_dim,
     ...
     loss_type=loss_type,
+    **kwargs,
 )
```

---

## 六、测试验证

| 测试 | 结果 |
|------|------|
| 模块导入 (RMSNorm, SwiGLU, CyclicRoPE, DiTBlockV2, FinalLayer) | ✅ 通过 |
| DiTDenoiserV2 构造 | ✅ 10.64M params |
| 前向传播 `(2, 128, 2) → (2, 128, 2)` | ✅ shape 一致 |
| CyclicRoPE1D 旋转 `(2, 8, 128, 32)` | ✅ shape 不变 |
| SwiGLU 前向 `(2, 128, 256)` | ✅ shape 不变 |
| `DiffusionEvolution` + V2 集成 | ✅ predict_eps 正常 |
| V1 配置 (use_dit_v2=false) 兼容性 | ✅ 回退 V1 正常 |

---

## 七、使用方式

### 启用 DiT V2 训练

```bash
# 方式 1: 使用专用配置文件
CFG_FILE=configs/btcv_diffusion_dit_v2.yaml python diffusion_train.py

# 方式 2: 在任意配置中添加
use_dit_v2: true
```

### 配置文件关键字段

```yaml
# 在已有配置文件中添加即可切换到 V2:
use_dit_denoiser: false   # V1 关闭
use_dit_v2: true           # V2 启用
dit_num_layers: 6          # DiT 层数
dit_num_heads: 8           # 注意力头数
dit_state_dim: 256         # 状态维度
```

### 向后兼容

| 配置 | 效果 |
|------|------|
| `use_dit_v2: true` | 使用 DiTDenoiserV2 |
| `use_dit_denoiser: true, use_dit_v2: false` | 使用 DiTDenoiser V1 |
| `use_dit_denoiser: false, use_dit_v2: false` | 使用 GCN SnakeDenoiser |

---

## 八、后续规划

| 阶段 | 任务 | 状态 |
|------|------|------|
| Phase 1 | 核心架构升级 (M1-M4+M6) | ✅ 完成 |
| Phase 2 | BTCV 数据集训练 & 对比实验 | ⏳ 待执行 |
| Phase 3 | v-prediction 训练目标 (M5a) | 📋 待实施 |
| Phase 4 | Rectified Flow (M5b) | 📋 可选 |

### 推荐下一步

1. **在 BTCV 上跑 V2 训练** — 使用 `btcv_diffusion_dit_v2.yaml`
2. **消融实验** — 逐一关闭 M1-M6 验证各项贡献
3. **v-prediction** — 低风险高收益，仅需改 2 处

---

*报告版本: 1.0 | 日期: 2026-04-02*
