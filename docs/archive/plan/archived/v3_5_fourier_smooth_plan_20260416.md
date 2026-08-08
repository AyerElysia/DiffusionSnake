# V3.5 改进方案：傅里叶空间扩散 + 低通后处理 (2026-04-16)

## 目标
彻底解决V3.0的毛边问题，让预测轮廓平滑且贴合器官边界。

---

## 方案概述

V3.5 采用 **两阶段策略**：
- **Phase 1**：零成本后处理（不需要重新训练，立即可验证）
- **Phase 2**：傅里叶空间扩散（需要重新训练，从根本上解决问题）

---

## Phase 1：傅里叶低通后处理（零训练成本）

### 原理
对 DDIM 采样得到的位移向量做 FFT → 截断高频 → IFFT。
128个点的DFT有128个频率分量，只保留最低K个。

### 已验证的效果（随机权重测试）
| K值 | 保留频率数 | 平滑度提升 | 说明 |
|-----|-----------|-----------|------|
| K=8 | 16/128 | 6000× | 非常平滑，可能丢失细节 |
| K=12| 24/128 | ~1000× | 平滑且保留大部分形状 |
| K=16| 32/128 | 273× | 良好平衡 |
| K=32| 64/128 | 13× | 轻度平滑 |

### 推荐参数
- 先用 K=12 测试现有 V3.0 10k checkpoint
- K=8~16 之间做 grid search

### 实现步骤

1. 在 `pretrain_evolution.py` 的推理分支添加后处理：
```python
def fourier_smooth(disp, k=12):
    """对预测的位移做傅里叶低通滤波。
    disp: (B, N_points, 2) 位移向量
    k: 保留的低频分量数（单侧），总共保留 2k 个频率
    """
    B, N, C = disp.shape
    # 对每个坐标维度做 FFT
    freq = torch.fft.rfft(disp, dim=1)  # (B, N//2+1, 2)
    # 截断高频
    freq[:, k+1:, :] = 0
    # IFFT 恢复
    smooth_disp = torch.fft.irfft(freq, n=N, dim=1)  # (B, N, 2)
    return smooth_disp
```

2. 在 `sample_disp()` 返回前调用：
```python
if self.cfg.fourier_smooth_k > 0:
    x0 = fourier_smooth(x0, k=self.cfg.fourier_smooth_k)
```

3. 注册配置项（`lib/config/config.py`）：
```python
cfg.fourier_smooth_k = 0  # 0=禁用, >0 保留的低频数
```

4. 创建测试配置 `configs/btcv_diffusion_dit_v3_5_test_smooth.yaml`

### 验证计划
- 用V3.0的10k checkpoint + 后处理，跑单样本推理对比
- K=8,10,12,16 各跑一次，对比视觉效果
- **预计耗时：30分钟内完成全部测试**

---

## Phase 2：傅里叶空间扩散（核心创新）

### 动机

Phase 1 的后处理是"事后补救"——信息已经丢失了。
理想方案是让扩散模型 **直接在低频空间** 工作。

### 核心思想：Fourier Diffusion

**不预测 128 个点的 xy 位移，而是预测 K 个傅里叶系数。**

```
传统：扩散模型 → (B, 128, 2) 位移 → 加到初始轮廓
V3.5 ：扩散模型 → (B, K, 2) 傅里叶系数 → IFFT → (B, 128, 2) 位移 → 加到初始轮廓
```

### 数学框架

1. **训练时**：
   - 计算 GT 位移 `x0 = gt_contour - init_contour`，形状 (B, 128, 2)
   - 对每个坐标做 FFT：`freq_x0 = FFT(x0)`，取前 K 个系数 (B, K, 2)
   - 归一化傅里叶系数（需要新的统计量）
   - 扩散过程在 (B, K, 2) 空间进行加噪/去噪
   - Denoiser 输入维度变为 K（而非128），输出也是 K
   - Loss: `MSE(eps_pred, eps)` 在傅里叶空间

2. **推理时**：
   - 从 (B, K, 2) 的高斯噪声开始 DDIM 采样
   - 得到 (B, K, 2) 的傅里叶系数
   - IFFT 重建 128 点位移
   - **天然平滑，因为只有 K 个低频分量**

### K 的选择

K 太小 → 丢失形状细节（肾上腺等小器官可能变形）
K 太大 → 回到高频问题

建议 K=16~24：
- K=16 保留 32 个频率，足够表达大多数器官轮廓
- K=24 保留 48 个频率，可以表达更复杂的形状

### 架构修改

1. **FinalLayer 输出维度**：128×2 → K×2（实际是K个复数的实部和虚部，共K×4）
   ```python
   # 修改 FinalLayer
   self.linear = nn.Linear(hidden_size, K * 4)  # K个复数×2坐标
   ```

2. **Denoiser 输入序列长度**：128 → K
   - Perceiver 的 latent tokens 数量不需要变（全局特征提取）
   - Cross-Attention 的 query 从 128 点变为 K 个频率分量
   - 但 GCN 特征采样仍然在 128 个点位置进行 → 需要一个聚合层

3. **训练数据准备**：
   - 需要预先计算所有样本的傅里叶系数统计量（均值、方差）
   - 或者直接用 [-1,1] 归一化

### 与 V3.4 迭代推理的兼容性

V3.5 的傅里叶空间扩散可以 **和 V3.4 的多步迭代推理组合**：
- 每步预测 K 个傅里叶系数 → IFFT → 位移 → 更新轮廓 → 重新采样特征 → 下一步
- 每步都天然平滑，累积也平滑

---

## Phase 2 的工程复杂度评估

| 组件 | 修改量 | 难度 |
|------|--------|------|
| 训练数据 FFT 预处理 | 20行 | ⭐ |
| FinalLayer 输出维度 | 5行 | ⭐ |
| Denoiser 适配 K 序列 | 50行 | ⭐⭐⭐ |
| 归一化统计量计算 | 30行 | ⭐⭐ |
| 推理 IFFT 重建 | 10行 | ⭐ |
| 配置注册 | 5行 | ⭐ |
| **总计** | ~120行 | **中等** |

### 最大挑战
Denoiser 目前的输入是 (B, 128, feat_dim)——128个点位置采样的CNN特征。
如果改成 K 个傅里叶系数，需要重新设计特征提取：
- **方案A**：仍在128点采样CNN特征 → 对特征做FFT取前K个 → 输入Denoiser
- **方案B**：仍在128点采样CNN特征 → Cross-Attention中作为KV，K个频率token作为Q
- **方案C（推荐）**：Perceiver保持不变，128点特征进入latent → latent提取全局特征 → Q改为K个频率位置embedding → Denoiser输出K个系数

方案C最优雅：Perceiver的bottleneck天然起到降维+全局化的作用，然后用 K 个可学习的 frequency embeddings 作为 query，从 Perceiver latent 中提取对应频率的信息。

---

## TODO 实施顺序

### 立即执行（Phase 1, 30分钟）
1. ✅ 实现 `fourier_smooth()` 后处理函数
2. ✅ 注册 `fourier_smooth_k` 配置
3. ✅ 测试 V3.0 10k checkpoint + K=8/12/16
4. ✅ 视觉对比，确认有效

### 短期执行（Phase 2, 1-2天）
5. 设计 Fourier Denoiser（基于V3.0的DiTDenoiserV3，修改输出层）
6. 修改训练数据流（displacement → FFT → normalize）
7. 计算傅里叶空间的归一化统计量
8. 单样本过拟合验证
9. 全数据集训练

### 中期（Phase 2 + V3.4 组合）
10. V3.5 + V3.4 迭代推理组合测试

---

## 这个方案为什么比 V3.3 好？

| 对比 | V3.3（失败） | V3.5（提议） |
|------|-------------|-------------|
| 思路 | 在现有架构上"打补丁" | 从表示空间根本改变 |
| 平滑约束 | 通过额外 loss 惩罚 | 通过表示结构保证 |
| 训练稳定性 | 两个 loss 冲突 | 只有一个 diff_loss |
| 是否需要重训 | Phase1 否 | Phase2 是 |
| 信息瓶颈 | 无（仍128点） | K系数（天然低维） |

**核心区别：V3.3 试图让模型"学会平滑输出"，V3.5 让模型"不可能输出不平滑的结果"。**

---

## 文献支撑

1. **FourierNet** (CVPR)：用傅里叶描述子表示轮廓，在频率空间预测形状
2. **ContourDiff** (MELBA 2024)：在扩散过程中使用轮廓约束，保持解剖一致性
3. **DiffuseReg** (MICCAI 2024)：在位移场上做扩散，用正则化保证平滑
4. **Spectral Diffusion**：在频率域做扩散过程，低频收敛快且稳定

---

## 风险与备选方案

### 如果 Phase 2 不work？
- K选错 → 尝试不同K值（12/16/20/24）
- Denoiser不收敛 → 用V3.0权重warm-start（Perceiver部分可复用）
- 形状精度下降 → K选大一点，或者混合方案（低频FFT+高频残差）

### 备选方案B：Laplacian扩散
- 在扩散过程中对噪声做Laplacian smooth
- 即 `x_t = sqrt(α)*x0 + sqrt(1-α)*smooth(ε)` 而非原始高斯噪声
- 采样时也用smooth noise → 天然减少高频
- 优点：不改变模型结构，只改noise schedule
- 缺点：可能影响扩散过程的理论保证

### 备选方案C：后处理Pipeline
- 扩散预测粗轮廓 → 传统活动轮廓（Active Contour/Snake）做精细化
- 利用图像梯度信息，在保持平滑的同时对齐边缘
- 优点：经典方法成熟可靠
- 缺点：增加推理复杂度
