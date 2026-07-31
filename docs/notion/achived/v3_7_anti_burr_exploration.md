# V3.7 Anti-Burr Denoiser — 探索记录

## 日期: 2026-04-19

## 问题
DiffusionSnake轮廓预测中存在"毛刺"(burr)问题,尤其在小轮廓上严重。

### 根因分析 (来自 BURR_ANALYSIS_COMPLETE_SUMMARY.md)
1. 固定128点对小轮廓造成过高的点密度
2. 相邻点距离小 → 信噪比低
3. MSE损失对所有频率等权处理,未惩罚高频噪声
4. ODE积分过程中噪声逐步累积

## V3.7 设计方案

### 三层抗毛刺策略

#### 层级1: 网络架构 (dit_denoiser_v3_7.py)
- **Circular 1D Conv Smoothing**: 在DiT transformer块之后、Final Layer之前加入环形1D卷积
  - kernel_size=9, 2层
  - 强制相邻点的隐藏特征保持局部一致性
  - 可学习gate控制平滑强度 (初始sigmoid(2.0) ≈ 88%)
- **Laplacian正则化**: 对预测的速度场施加二阶差分惩罚
  - weight=0.1

#### 层级2: 训练损失 (flow_matching_evolution.py 修改)
- **频谱损失分解**: 将MSE分为低频+高频两部分
  - 低频(前16个Fourier系数): 权重=1.0
  - 高频(剩余系数): 权重=0.1
  - 使模型优先匹配平滑的形状
- **使用L_reg**: 之前denoiser返回的正则化损失被丢弃(`_`), 现在正式使用

#### 层级3: 推理平滑
- **ODE每步Fourier平滑**: 在Euler积分的每一步后应用低通滤波 (k=12)
  - 防止噪声在ODE积分过程中累积
- **增加ODE步数**: 50步 (原10步), 提高积分精度
- **后处理Fourier平滑**: k=12 作为最后安全网

### 配置参数
```yaml
use_dit_v3_7: true
v3_7_smooth_kernel: 9
v3_7_num_smooth_layers: 2
v3_7_laplacian_weight: 0.1
v3_7_spectral_loss_k: 16
v3_7_hf_loss_weight: 0.1
v3_7_ode_smooth_k: 12
flow_ode_steps: 50
```

## 实验记录

### 实验1: V3.7 初版 (频谱损失 + Laplacian 0.1 + ODE平滑)
- **GPU**: 2, **训练**: 10K epochs, lr=1e-4
- **Step ~5800**: loss=0.048
- **推理结果 (标准推理 ODE50)**: Mean IoU = 79.4%
- **推理结果 (noise_0.1 + AVG50 + ODE10)**: **Mean IoU = 87.8%** ← 最佳!
- **关键发现**: 轮廓平滑无毛刺，但频谱损失限制了精度

### 实验2: V3.7.1 (纯MSE + Laplacian 0.01)
- **GPU**: 1, **训练**: 10K epochs, lr=2e-4
- **Step ~1400**: loss=0.016
- **推理结果 (noise_0.1 + AVG50 + ODE20)**: Mean IoU = 78.9%
- **注**: 训练还在早期，应该会继续提升

### 实验3: V3.7.2 (纯MSE + 无正则化)
- **GPU**: 3, **训练**: 10K epochs, lr=3e-4
- **Step ~1400**: loss=0.002 (最低!)
- **推理结果**: Mean IoU = 45.4% (最差!)
- **关键发现**: 低训练loss ≠ 高IoU! 无正则化导致极严重毛刺

### 实验4: V3.7.3 (低噪声流匹配 — 核心创新)
- **GPU**: 0, **训练**: 20K epochs, lr=3e-4
- **创新点**: `flow_noise_scale=0.1` (标准=1.0)
  - 使速度场近似常数 v ≈ x_1，大幅降低学习难度
  - 训练和推理对齐：都用低噪声
- **推理设置**: noise_0.1 + AVG50 + ODE10
- **状态**: 刚开始训练

## 关键发现总结

### 1. 推理策略对IoU影响巨大
| 推理策略 | V3.7 Mean IoU |
|----------|---------------|
| 标准 (noise=1.0, ODE50, per-step smooth) | 78.4% |
| 无迭代优化 (noise=1.0, ODE50) | 3.8% (需要per-step smooth!) |
| 零初始化 ODE5 | 84.3% |
| noise_0.1 + AVG50 + ODE10 | **87.8%** |

### 2. 迭代优化(iterative refinement)有害!
- 有迭代: 73.4% → 无迭代: 81.2% (V3.7, 标准noise)
- 原因: 部分移动后的特征采样位置偏移，模型处于OOD状态

### 3. 低训练loss ≠ 高IoU
- V3.7.2: loss=0.002, IoU=45% (最差)
- V3.7: loss=0.048, IoU=87.8% (最好)
- 原因: 流匹配loss衡量单步速度精度，IoU衡量ODE全程轨迹质量

### 4. 噪声尺度 0.1 是甜蜜点
- noise=0: 78.2% (确定性但偏移)
- noise=0.1: 81.2% (最佳单次)
- noise=0.1+AVG: 87.8% (最佳聚合)
- noise=1.0: 需要per-step fourier smooth才能工作

---
*此文件由Copilot自动生成,记录V3.7的探索过程*
