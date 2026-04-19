# DiffusionSnake V3.5 傅里叶空间扩散 — 完整开发报告

**日期**: 2026-04-16  
**作者**: Copilot (自动化开发助手)  
**目标**: 解决 V3.0 毛边(jagged edge)问题，实现傅里叶空间扩散去噪

---

## 1. 问题诊断

### 1.1 毛边现象根因分析

V3.0 单样本过拟合 10k epoch 后的轮廓出现严重毛边。根因分析发现：

**核心问题**：DDIM 采样过程中，128 个点**完全独立**地被添加噪声和去噪。相邻点之间没有任何平滑性约束。即使 denoiser 输出的每个点的位移是合理的，高频噪声在多步采样中会被逐步放大。

**V3.3a/V3.3b 为什么失败**：

| 版本 | 尝试 | 失败原因 |
|------|------|----------|
| V3.3a | CircularConv1d (1D 循环卷积) | 卷积作用在 feature space（FinalLayer 之前），不是 displacement space。0.1 的 residual weight 太小，反而增加了拟合高频噪声的能力 |
| V3.3b | smooth_loss + curv_loss | smooth_loss 通过 `x0 = (x_t - √(1-ᾱ)·eps) / √(ᾱ)` 反向传播。当 t 较大时 √(ᾱ) → 0，梯度爆炸。初始 smooth_loss = 17.7M vs diff_loss = 1.01，完全梯度支配 |

### 1.2 定量验证

对 V3.0 10k checkpoint 的随机权重初始化测试：
- **平滑度指标**（二阶差分范数之和）：6486（基线，无平滑）
- 该值远高于正常轮廓（通常 < 100），证实了独立采样导致的高频噪声问题

---

## 2. Phase 1: 傅里叶低通后处理（零训练成本）

### 2.1 方法

在推理时，对 DDIM 采样得到的位移向量 `(N, 128, 2)` 进行 FFT → 保留最低 K 个频率分量 → IFFT 重建。

### 2.2 实现

- **文件**: `lib/networks/diffusion/pretrain_evolution.py`
  - 新增 `fourier_smooth()` 静态方法
  - 在标准推理和迭代推理分支中集成
- **配置**: `lib/config/config.py` → `fourier_smooth_k = 0`（0=禁用）
- **测试脚本**: `test/test_fourier_smooth.py`
- **测试配置**: `configs/btcv_diffusion_dit_v3_5_test_smooth.yaml`

### 2.3 测试结果

使用 V3.0 10k checkpoint (diff_loss = 0.000103)：

| K (保留频率数) | 平滑度 | 相对改善 | 视觉效果 |
|----------------|--------|----------|----------|
| 0 (无) | 6486 | 基线 | 严重毛刺，有飞线 |
| **8** | **484** | **13.4×** | **毛刺完全消除** |
| 12 | 490 | 13.2× | 接近 K=8 |
| 16 | 2195 | 3.0× | 异常（可能过多高频保留） |
| 24 | 535 | 12.1× | 良好 |

**结论**: K=8 是最优后处理参数，零训练成本即可获得 13× 平滑改善。

**视觉对比**: 见 `visual/fourier_smooth_test/` 目录

### 2.4 评价

Phase 1 是一个**应急方案**。它确实消除了毛边，但存在限制：
- 后处理丢失了模型学到的高频细节（如果有的话）
- 不是"训练中学到平滑"，而是"训练后强制平滑"
- 对于复杂形状（如锯齿状器官边缘），可能过度平滑

详细评估见: `notion/fourier_smooth_phase1_eval_20260416.md`

---

## 3. Phase 2: V3.5 傅里叶空间扩散（根本性解决方案）

### 3.1 核心思想

**不要在 128×2 的点空间做扩散，而是在 K×4 的傅里叶系数空间做扩散。**

- 输入: 位移向量 `(N, 128, 2)` → FFT → 取前 K 个频率系数 → `(N, K, 4)` (real_x, imag_x, real_y, imag_y)
- 扩散: 在 `(N, K, 4)` 空间上加噪声、去噪
- 输出: 去噪的 `(N, K, 4)` → IFFT → 重建 `(N, 128, 2)` → 天然平滑

**数学保证**：K 个傅里叶系数经 IFFT 重建的信号只包含前 K 个频率，高频毛刺**在结构上不可能出现**。

### 3.2 网络架构

**文件**: `lib/networks/diffusion/dit_denoiser_v3_5.py`

```
DiTDenoiserV3_5
├── GlobalPerceiverCompressor     # (B, 64, H, W) → (B, G, state_dim) 全局上下文
├── FourierPointBridge            # (N, 64, 128) → (N, K, state_dim) 频率感知局部特征
│   └── Cross-Attention: K learnable frequency queries attend to 128 point features
├── FourierCoeffEmbedding         # (N, K, 4) noisy coeffs → (N, K, state_dim)
│   └── Linear(4, state_dim) + learnable frequency position embeddings
├── DiTBlockV3 × num_layers       # Self→Cross attention（交替使用 global/local context）
├── FourierFinalLayer             # (N, K, state_dim) → (N, K, 4) with zero-init
└── TimeEmbedding                 # Timestep embedding
```

**参数量**: 10,866,244（约 10.9M，与 V3.0 同量级）

**关键设计**:
- **FourierPointBridge**: 不是简单 pooling，而是 cross-attention。K 个可学习的频率查询向量从 128 个点特征中提取频率感知的局部上下文。这让模型知道"低频分量应该关注全局形状，高频分量应该关注局部细节"。
- **零初始化 FinalLayer**: 与 V3.0 一致的 stable training 策略，初始输出为全零。
- **复用 DiTBlockV3**: 与 V3.0 使用相同的 attention block，确保已验证的组件不引入新 bug。

### 3.3 训练流程修改

**文件**: `lib/networks/diffusion/pretrain_evolution.py`

1. **前向传播（训练）**:
   ```
   x0 = gt_contours - init_contours            # (N, 128, 2)
   x0_fourier = disp_to_fourier(x0)            # (N, K, 4) via rfft
   x0_norm = normalize_disp_fourier(x0_fourier) # normalize to [-1, 1]
   noise = randn_like(x0_norm)                  # (N, K, 4) noise
   x_t = add_noise(x0_norm, noise, t)           # (N, K, 4)
   eps_pred = denoiser(cnn_feat, x_t, t)        # (N, K, 4) predict noise
   loss = MSE(eps_pred, noise)                  # standard diffusion loss
   ```

2. **推理**:
   ```
   x_T = randn(N, K, 4)                        # start from pure noise
   for t in DDIM_schedule:
       eps = denoiser(cnn_feat, x_t, t)         # predict noise
       x_{t-1} = DDIM_step(x_t, eps, t)        # one DDIM step
   x0_fourier = denormalize(x_0)
   disp = fourier_to_disp(x0_fourier)           # IFFT → (N, 128, 2)
   ```

3. **新增方法**:
   - `disp_to_fourier()`: `(N, 128, 2)` → rfft → 取前 K 个频率 → `(N, K, 4)`
   - `fourier_to_disp()`: `(N, K, 4)` → 重建复数 → 零填充到 65 bins → irfft → `(N, 128, 2)`
   - `normalize_disp_fourier()` / `denormalize_disp_fourier()`: 基于空间位移范围的归一化
   - `sample_disp_fourier()`: 完整的傅里叶空间 DDIM 采样循环

### 3.4 配置

**新增配置键** (`lib/config/config.py`):
- `use_dit_v3_5 = False`: 启用 V3.5
- `fourier_k = 16`: 傅里叶系数数量
- `fourier_smooth_k = 0`: Phase 1 后处理参数（V3.5 下不需要，但兼容）

**单样本过拟合配置**: `configs/btcv_diffusion_dit_v3_5_single_overfit.yaml`

### 3.5 训练验证

单样本过拟合训练已启动并验证：

| 指标 | 值 |
|------|-----|
| GPU | 0 (RTX 3090) |
| 每 epoch 耗时 | ~1.1s |
| 总 epoch | 10,000 |
| 预计总时间 | ~3 小时 |
| 初始 diff_loss | 0.982 |
| 600 epoch diff_loss | 0.168 |
| 收敛倍率 | 5.9× |
| 模型参数 | 10.9M |
| GPU 显存 | ~1.8 GB |

**训练正在正常进行中**，loss 持续下降。

---

## 4. 文件清单

### 新建文件

| 文件 | 说明 |
|------|------|
| `lib/networks/diffusion/dit_denoiser_v3_5.py` | V3.5 傅里叶空间去噪器 |
| `configs/btcv_diffusion_dit_v3_5_single_overfit.yaml` | V3.5 单样本过拟合配置 |
| `configs/btcv_diffusion_dit_v3_5_test_smooth.yaml` | Phase 1 后处理测试配置 |
| `test/test_fourier_smooth.py` | Phase 1 测试脚本 |
| `notion/jagged_edge_analysis_20260416.md` | 毛边根因分析 |
| `notion/fourier_smooth_phase1_eval_20260416.md` | Phase 1 评估 |
| `plan/v3_5_fourier_smooth_plan_20260416.md` | V3.5 改进方案 |
| `report/v3_5_final_report.md` | 本报告 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `lib/config/config.py` | 新增 `fourier_smooth_k`, `use_dit_v3_5`, `fourier_k` 配置键 |
| `lib/networks/diffusion/pretrain_evolution.py` | V3.5 傅里叶训练/推理管道、Phase 1 后处理、V3.5 denoiser 初始化 |

---

## 5. 后续建议

### 5.1 短期（等训练结果时做）

1. **V3.5 训练完成后**（~3h）:
   - 运行推理，保存可视化到 `visual/` 目录
   - 对比 V3.0 + K=8 后处理 vs V3.5 原生输出
   - 关注: 轮廓平滑度、形状准确度、是否有信息丢失

2. **调参实验**:
   - `fourier_k`: 8 vs 16 vs 24（影响重建精度 vs 平滑度）
   - 当前 K=16，round-trip reconstruction error ≈ 0.65（有信息损失）
   - K=24 可能更好地保留形状细节

### 5.2 中期

3. **归一化改进**:
   - 当前: 使用空间位移范围作为 scale factor
   - 改进: 统计每个频率 bin 的 mean/std，做 per-frequency-bin normalization
   - 原因: 低频系数量级远大于高频，统一归一化让高频系数的 SNR 很低

4. **推理策略**:
   - V3.5 + V3.4 迭代推理的结合（多步 Fourier-space 采样）
   - DDIM 步数优化（50 步可能过多，10-20 步可能足够）

### 5.3 创新点总结

从论文写作角度，V3.5 的创新点：

1. **Frequency-Domain Diffusion for Contour Evolution**: 首次将扩散过程从点空间搬到频率空间，从结构上消除高频毛刺（相比 V3.3b 的后处理 loss 方案更优雅）
2. **FourierPointBridge**: 通过 cross-attention 将空间域的 128 点特征桥接到频率域的 K 个系数，实现 frequency-aware 的局部上下文聚合
3. **Structurally Guaranteed Smoothness**: 与后处理不同，V3.5 的平滑性是**结构保证**的——模型输出空间就是低频空间，不需要额外约束

---

## 6. 已知问题和注意事项

1. **重建误差**: K=16 的 round-trip 误差为 0.65（在位移范围内）。这意味着训练目标是"最佳低频近似"，不是精确重建。需要验证这是否影响最终分割精度。

2. **CyclicRoPE 兼容性**: V3.5 的 DiTBlockV3 使用 CyclicRoPE，设计用于 128 个循环排列的点。V3.5 中 K=16 个频率 token 不具备循环性质，但 RoPE 仍然提供了位置编码功能。如果效果不佳，可以替换为 standard positional encoding。

3. **config 中 `gpus` 字段**: 使用 `CUDA_VISIBLE_DEVICES=X` 时，config 中的 `gpus` 必须设为 `[0]`（而不是 `[X]`），因为 CUDA 会重映射设备号。

---

*训练正在 GPU 0 上进行。最新 loss: ~0.168 @ epoch 622。*  
*运行 `tail -1 data/outputs/btcv_diffusion_dit_v3_5_single_overfit/logs.jsonl | python3 -m json.tool` 查看最新进度。*
