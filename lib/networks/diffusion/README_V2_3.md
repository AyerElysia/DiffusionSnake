# DiT V2.3: Flow Matching (Rectified Flow) Evolution

## 1. 概述 (Overview)
DiT V2.3 是对去噪概率范式的根本性变革。它摒弃了 DDPM/DDIM 的随机微分方程（SDE），采用了 **Rectified Flow (Flow Matching)**。它不学习如何从高斯噪声中退化，而是学习如何沿着**直线**将噪声推向目标。

## 2. 核心组件 (Key Components)

| 模块 | 实现类/方法 | 功能描述 | 核心收益 |
|:---|:---|:---|:---|
| **直线轨迹器** | `FlowMatchingEvolution` | 采样 X_t = (1-t)X0 + tX1。 | 实现最短路径收敛，方差极低。 |
| **速度场预测器** | `v_target = x1 - x0` | 网络拟合的是常数向量场，即从 X0 到 X1 的位移。 | 极大降低了 Transformer 的学习难度。 |
| **10步 ODE 求解器** | `sample_disp (Euler)` | 使用简单的欧拉法进行积分推理。 | 推理耗时比 V2.2 降低 80% (50步 -> 10步)。 |

## 3. ASCII 网络结构图 (Architecture Flow)

```text
      ┌──────────┐                     ┌──────────┐
      │ 初始噪声 X0 │                     │  目标轮廓 X1 │
      └────┬─────┘                     └────┬─────┘
           │          采样时间 t [0, 1]     │
           │          (Linear Interp)      │
           └───────────────┬───────────────┘
                           │
                 ┌─────────┴─────────┐
                 │  混合状态 X_t (Noisy)│
                 └─────────┬─────────┘
                           │
      ┌─────────────┐      ▼      ┌─────────────────────────┐
      │  CNN 特征图  │ ──► DiT ──► │  预测速度 Vt (Velocity)   │
      └─────────────┘     V2.3    └────────────┬────────────┘
                                               │
                                       ┌───────┴───────┐
                                       │    Loss 计算   │
                                       │ MSE(Vt, X1-X0)│
                                       └───────────────┘

[推理采样阶段 (Inference Stage)]
X_0 ──► Vt_0 ──► X_0.1 ──► Vt_0.1 ──► ... ──► X_1.0 (最终结果)
        (Euler Step dx = v*dt, dt=0.1)
```

## 4. 算法逻辑差异 (Flow Matching vs Diffusion)
- **Diffusion (V2.2及以前)**: 目标是预测噪声 $\epsilon$。采样轨迹通常是弯曲的漫步过程（Langevin Dynamics），需要非常多的步数才能逼近目标。
- **Flow Matching (V2.3)**: 目标是预测单位时间内的速度 $V_t$。采样轨迹是一条**直线**。如果你从 $X_0$ 出发，且速度场预测准确，你可以用极大的步长（甚至 1 步）直接到达 $X_1$。

## 5. 配置参数 (Config)
```yaml
use_dit_v2_3: true
use_flow_matching: true
flow_ode_steps: 10   # 默认 10 步欧拉推理
dit_num_layers: 6
dit_num_heads: 8
```
