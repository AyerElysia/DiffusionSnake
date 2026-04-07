# DiT Denoiser V3.2 - Flow Matching

## 架构概览

```
================================================================================
                    DiT DENOISER V3.2 (FLOW MATCHING) ARCHITECTURE
================================================================================

Input:
  cnn_feature:  (B, 64, H, W)    -- YOLO P2 feature map (typically 128×128)
  sampled_feat: (N, 64, P)       -- Per-point sampled features (local)
  x_t:          (N, P, 2)        -- Current state on the flow path
  t:            (N,)             -- Time in [0, 1] (scaled by 1000 for embedding)
  py_ind:       (N,)             -- Contour-to-image index

================================================================================
[STAGE 1: EMBEDDING]  -- 与 V3.1 相同
================================================================================

  Time t ──────────────────────────────────────────────────────────────────────►
                    │
                    ▼
          ┌─────────────────────┐
          │ SinusoidalTimeEmbed │ ──► Linear ──► SiLU ──► Linear ──► t_emb (N, 256)
          └─────────────────────┘


  cnn_feature (B, 64, H, W) ─────► PatchifyEmbedding ──► global_ctx (N, 256, 256)

  sampled_feat (N, 64, P) ───────► Linear ──► SiLU ──► Linear ──► local_ctx (N, P, 256)

  x_t (N, P, 2) + sampled_feat ──► SeparatePointEmbedding ──► x (N, P, 256)

================================================================================
[STAGE 2: DiT BLOCKS × 6 LAYERS]  -- 与 V3.1 相同
================================================================================

  Same as V3.1: Self-Attention → Cross-Attention → SwiGLU FFN
  Alternating global/local context injection

================================================================================
[STAGE 3: OUTPUT]
================================================================================

  x (N, P, 256) ──► FinalLayer(adaLN + Linear) ──► v_pred (N, P, 2)
                                                    │
                                                    ▼
                                              Velocity Field v

================================================================================
```

## 核心特性：Flow Matching

### 与 DDPM (V3.0/V3.1) 的本质区别

| 特性 | DDPM (V3.0/V3.1) | Flow Matching (V3.2) |
|------|-----------------|---------------------|
| **训练目标** | 预测噪声 ε | 预测速度场 v |
| **轨迹形状** | 复杂曲线 | 直线（Rectified Flow） |
| **采样步数** | 50-1000 步 | **10-20 步** |
| **训练稳定性** | 中等 | **高** |
| **推理速度** | 慢 | **快 5-10x** |

### Flow Matching 原理

```
DDPM:       x_0 (GT) ←───── 曲线轨迹 ─────→ x_T (噪声)
                    ↖       ↖       ↖
                    ε_pred  ε_pred  ε_pred
                    
Flow Matching:  x_0 (GT) ←───── 直线轨迹 ─────→ x_1 (噪声)
                    ↖       ↖       ↖
                    v_pred  v_pred  v_pred
                    (velocity = x_1 - x_0, 常量！)
```

**关键洞察**：
- 在 Rectified Flow 中，速度场 v = x_1 - x_0 是**常量**
- 沿直线轨迹只需预测一个方向
- ODE 积分器只需很少步数就能精确采样

---

## 训练目标

### 数学公式

**DDPM (噪声预测)**:
```
Loss = ||ε - ε_pred(x_t, t)||²
```

**Flow Matching (速度预测)**:
```
v_target = x_1 - x_0           # 从噪声到 GT 的方向
x_t = (1-t) * x_0 + t * x_1    # 直线插值

Loss = ||v_target - v_pred(x_t, t)||²
```

### 训练代码示例

```python
# V3.2 训练循环
def train_step(denoiser, cnn_feat, sampled_feat, x_0, x_1):
    # x_0: 初始轮廓 (octagon init)
    # x_1: 目标轮廓 (GT)
    
    # 随机采样时间 t ∈ [0, 1]
    t = torch.rand(N, device=device)
    
    # 直线插值
    x_t = (1 - t.view(N, 1, 1)) * x_0 + t.view(N, 1, 1) * x_1
    
    # 目标速度（常量）
    v_target = x_1 - x_0
    
    # 预测速度
    v_pred = denoiser(cnn_feat, sampled_feat, x_t, t * 1000)
    
    # 损失
    loss = F.mse_loss(v_pred, v_target)
    
    return loss
```

---

## 推理过程：ODE Solver

### Euler 方法

```python
def inference(denoiser, cnn_feat, sampled_feat, x_1, num_steps=10):
    """
    从噪声 x_1 演化到轮廓 x_0
    
    Args:
        x_1: 初始噪声/octagon init
        num_steps: ODE 积分步数（通常 10-20）
    """
    x = x_1
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = i / num_steps  # t ∈ [0, 1)
        
        # 预测速度
        v = denoiser(cnn_feat, sampled_feat, x, t * 1000)
        
        # Euler 步进：x = x + v * dt
        x = x + v * dt
    
    return x  # 最终轮廓
```

### 可视化

```
t=0.0    t=0.1    t=0.2    t=0.3    ...    t=0.9    t=1.0
  │        │        │        │              │        │
  ▼        ▼        ▼        ▼              ▼        ▼
x_1 ────► x ────► x ────► x ────► ... ────► x ────► x_0
  │        │        │        │              │        │
  └─v_pred─└─v_pred─└─v_pred─└── ... ───────└─v_pred─┘
  
  每一步：x_{t+dt} = x_t + v_pred(x_t, t) * dt
```

---

## 配置示例

```yaml
# configs/btcv_diffusion_dit_v3_2.yaml
use_diffusion_evolution: true
use_flow_matching: true             # 启用 Flow Matching
use_dit_v3_2: true                  # 启用 V3.2
dit_num_layers: 6
dit_num_heads: 8
dit_state_dim: 256
ode_steps: 10                       # ODE 积分步数（推荐 10-20）
```

---

## V3 系列完整对比

| 特性 | V3.0 | V3.1 | V3.2 |
|------|------|------|------|
| **全局语义** | Perceiver | Patchify | Patchify |
| **训练方式** | DDPM | DDPM | **Flow Matching** |
| **预测目标** | 噪声 ε | 噪声 ε | **速度 v** |
| **采样步数** | 50-1000 | 50-1000 | **10-20** |
| **推理速度** | 慢 | 慢 | **快 5-10x** |
| **训练稳定性** | 中 | 中 | **高** |
| **轨迹** | 曲线 | 曲线 | **直线** |

---

## 选择建议

### 选择 V3.2 的场景

- ✅ 需要快速推理（实时应用）
- ✅ 希望训练更稳定
- ✅ 采样步数敏感场景
- ✅ 对轨迹可解释性有要求

### 选择 V3.0/V3.1 的场景

- ✅ 已有成熟的 DDPM 训练流程
- ✅ 需要使用 DDIM 等现有采样器
- ✅ 对推理速度不敏感

---

## 性能预期

| 指标 | V3.0/V3.1 (DDPM) | V3.2 (Flow Matching) |
|------|-----------------|---------------------|
| 训练收敛速度 | 中 | 快 |
| 推理时间 (ms) | ~200 (50步) | **~20 (10步)** |
| 采样质量 | 高 | 高（理论上相当） |
| OOM 风险 | 低 | 低（相同架构） |

---

## 使用示例

```python
from lib.networks.diffusion.dit_denoiser_v3_2 import DiTFlowMatchingV3_2

denoiser = DiTFlowMatchingV3_2(
    state_dim=256,
    feature_dim=64,
    num_layers=6,
    num_heads=8,
    num_points=128,
    patch_size=8,
)

# 训练: 预测速度场
v_pred, L = denoiser(
    cnn_feature,    # (B, 64, H, W)
    sampled_feat,   # (N, 64, P)
    x_t,            # (N, P, 2) - 直线插值点
    t,              # (N,) - 时间 [0, 1]
    py_ind=py_ind,  # (N,)
)

# 推理: ODE 积分
x_final = ode_solve(denoiser, x_1, num_steps=10)
```

---

## 参数量统计

```
DiTFlowMatchingV3_2:
├── time_emb_net:        262,400
├── image_embed:         1,049,600    # Patchify
├── local_proj:           131,328
├── point_embed:           65,600
├── dit_layers (×6):   12,582,912
└── final_layer:         131,330
────────────────────────────────────
Total:                  ~14.2M        # 与 V3.1 完全相同
```

---

## 参考文献

1. **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
2. **Rectified Flow**: Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", ICLR 2023
3. **SD3**: Stability AI, "Stable Diffusion 3" - 使用 Rectified Flow
4. **DiT**: Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023