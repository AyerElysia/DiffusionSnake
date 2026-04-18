# DiffusionSnake: 基于扩散模型的轮廓演化网络

端到端的医学图像分割框架，融合 **YOLO 检测** + **扩散模型轮廓演化**。支持多种 DiT 去噪器版本和初始化策略。

---

## 目录

- [整体架构](#整体架构)
- [详细流程](#详细流程)
  - [1. 数据加载与预处理](#1-数据加载与预处理)
  - [2. YOLO 检测模块](#2-yolo-检测模块)
  - [3. 初始化轮廓生成](#3-初始化轮廓生成)
  - [4. 扩散模型演化](#4-扩散模型演化)
  - [5. 训练与推理](#5-训练与推理)
    - [5.1 训练流程](#51-训练流程)
    - [5.2 推理流程](#52-推理流程)
    - [5.3 边缘感知平滑后处理](#53-边缘感知平滑后处理)
    - [5.4 单样本过拟合训练](#54-单样本过拟合训练)
    - [5.5 位移场归一化](#55-位移场归一化)
    - [5.6 GRPO 强化学习训练](#53-grpo-强化学习训练-实验性)
- [模型版本详解](#模型版本详解)
- [配置文件说明](#配置文件说明)
- [快速开始](#快速开始)
- [关键文件索引](#关键文件索引)

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DiffusionSnake 整体架构                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入图像 (H×W×3)                                                           │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     YOLOv8-P2 检测网络                               │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │   │
│  │  │ Backbone    │───▶│ Neck (PAN)  │───▶│ Detect Head (P2-P5)     │  │   │
│  │  │ (CSPDarknet)│    │             │    │ - P2: stride=4 (细粒度) │  │   │
│  │  └─────────────┘    └─────────────┘    │ - P3-P5: 多尺度         │  │   │
│  │                                         └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ├──▶ 检测结果: [B, N, 6] (x1,y1,x2,y2,score,cls)                      │
│       │                                                                     │
│       └──▶ P2 特征图: [B, 64+nc, H/4, W/4]                                  │
│              │                                                              │
│              ▼                                                              │
│         ┌────────────┐                                                      │
│         │ CNN Proj   │  1×1 Conv: 64+nc → 64                                │
│         └────────────┘                                                      │
│              │                                                              │
│              ▼                                                              │
│         cnn_feature: [B, 64, H/4, W/4]                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     初始化轮廓生成                                    │   │
│  │                                                                      │   │
│  │   检测框 ──▶ get_init() ──▶ 4点矩形/菱形 ──▶ get_octagon() ──▶ 12点 │   │
│  │                                  │                     │             │   │
│  │                                  ▼                     ▼             │   │
│  │                           V1/V2: 矩形            V3: 八边形          │   │
│  │                                  │                     │             │   │
│  │                                  └─────────┬───────────┘             │   │
│  │                                            ▼                         │   │
│  │                                   uniform_upsample()                  │   │
│  │                                            │                         │   │
│  │                                            ▼                         │   │
│  │                              init_poly: [N, 128, 2]                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│              │                                                              │
│              ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     扩散模型演化 (DiT Denoiser)                       │   │
│  │                                                                      │   │
│  │   ┌──────────────────────────────────────────────────────────────┐  │   │
│  │   │  输入:                                                        │  │   │
│  │   │  - init_poly: 初始轮廓 [N, 128, 2]                            │  │   │
│  │   │  - cnn_feature: 视觉特征 [B, 64, H/4, W/4]                    │  │   │
│  │   │  - t: 时间步 [N]                                              │  │   │
│  │   │  - x_t: 带噪声位移场 [N, 128, 2]                              │  │   │
│  │   └──────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                       │   │
│  │                              ▼                                       │   │
│  │   ┌──────────────────────────────────────────────────────────────┐  │   │
│  │   │  DiT Denoiser (V1/V2/V3):                                     │  │   │
│  │   │                                                                │  │   │
│  │   │  1. 时间嵌入: Sinusoidal + MLP → t_emb                        │  │   │
│  │   │  2. 视觉编码:                                                  │  │   │
│  │   │     - Global: Perceiver / SpatialAnchor → [256, dim]          │  │   │
│  │   │     - Local: grid_sample from cnn_feature → [N, 64, 128]      │  │   │
│  │   │  3. 点嵌入: SeparatePointEmbedding (坐标+特征独立)            │  │   │
│  │   │  4. 位置编码: CyclicRoPE (闭环轮廓拓扑)                       │  │   │
│  │   │  5. DiT Blocks ×6: Cross-Attention + Self-Attention + FFN     │  │   │
│  │   │  6. Output Head: 预测噪声 eps [N, 128, 2]                      │  │   │
│  │   └──────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                       │   │
│  │                              ▼                                       │   │
│  │   ┌──────────────────────────────────────────────────────────────┐  │   │
│  │   │  DDPM/DDIM 采样:                                              │  │   │
│  │   │  - 训练: 预测噪声，MSE Loss                                   │  │   │
│  │   │  - 推理: 50步 DDIM 去噪 → disp_pred                           │  │   │
│  │   └──────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                       │   │
│  │                              ▼                                       │   │
│  │                    disp_pred: [N, 128, 2]                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│              │                                                              │
│              ▼                                                              │
│      pred_poly = init_poly + disp_pred                                      │
│              │                                                              │
│              ▼                                                              │
│      最终轮廓: [N, 128, 2]                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 详细流程

### 1. 数据加载与预处理

**入口文件**: `lib/datasets/dataset_catalog.py`

项目使用 **COCO 格式 JSON 标注** + 图像目录的数据结构：

```python
# dataset_catalog.py 中的条目示例
{
    'data_root': 'path/to/images/',          # 图像目录
    'ann_file': 'path/to/annotations.json',   # COCO 格式标注
}
```

实际数据路径在 `dataset_catalog.py` 中配置。当前本机 BTCV 数据位于:
- 训练: `/mnt/sdb1/leijh/DiffusionSnake/Datasets/BTCV/btcv_png_new_snake`
- 测试: `/mnt/sdb1/leijh/DiffusionSnake/Datasets/BTCV/btcv_png_test_new_snake`

> 注意: `dataset_catalog.py` 中仍保留部分旧机器路径 (`/home/medteam/Zhrch/...`)，
> 迁移数据时需同步更新。

**数据处理流程**:

```
原始图像 (H×W×3)
    │
    ├──▶ 数据增强 (随机翻转、缩放、颜色抖动)
    │
    ├──▶ Resize 到 (800, 800)
    │
    └──▶ 归一化 (mean/std)
          │
          ▼
    inp: [3, 800, 800]  (输入网络)


原始掩码
    │
    ├──▶ 二值化 → 多边形轮廓提取
    │
    ├──▶ 提取极点 (Top, Left, Bottom, Right)
    │
    ├──▶ 构造初始化轮廓:
    │    octagon = get_octagon(extreme_points)  # 12点
    │    init_poly = uniformsample(octagon, 128)  # 上采样到128点
    │
    └──▶ 构造 GT 轮廓:
         gt_poly = uniformsample(polygon, 128)
```

**关键字段说明**:

| 字段 | 形状 | 说明 |
|------|------|------|
| `inp` | [B, 3, 800, 800] | 输入图像 |
| `i_it_py` | [N, 128, 2] | 初始化轮廓 (八边形上采样) |
| `i_gt_py` | [N, 128, 2] | GT 目标轮廓 |
| `ct_01` | [B, M] | 有效实例掩码 |
| `orig_img` | [B, H, W, 3] | 原始图像 (可视化用) |

---

### 2. YOLO 检测模块

**入口文件**: `lib/networks/snake/ct_snake.py`

```python
class Network(nn.Module):
    def __init__(self, ...):
        # YOLOv8-P2: 包含 P2 层，stride=4
        yolo_yaml = 'lib/networks/YOLOV8/cfg/models/v8/yolov8-p2.yaml'
        self.yolo = DetectionModel(cfg=yolo_yaml, ch=3, nc=num_classes)

        # P2 特征投影: 64+nc → 64
        self.cnn_proj = nn.Conv2d(64 + nc, 64, kernel_size=1)
```

**前向传播流程**:

```
输入: x [B, 3, H, W]
      │
      ▼
┌─────────────────────────────────────┐
│           YOLOv8-P2                 │
│  - Backbone: CSPDarknet             │
│  - Neck: PAN (双向特征金字塔)        │
│  - Head: Detect (P2, P3, P4, P5)    │
└─────────────────────────────────────┘
      │
      ├──▶ yolo_y: [B, 4+nc, HW]  (检测输出)
      │
      └──▶ yolo_feats: List[[B, C_i, H_i, W_i]]  (多尺度特征)
            │
            └──▶ p2 = yolo_feats[0]  [B, 64+nc, H/4, W/4]
                  │
                  ▼
            cnn_feature = cnn_proj(p2)  [B, 64, H/4, W/4]
```

**NMS 后处理**:

```python
# 使用 YOLO 内置 NMS
from lib.networks.YOLOV8.utils.ops import non_max_suppression

detection = non_max_suppression(
    pred,
    conf_thres=0.01,      # 置信度阈值
    iou_thres=0.45,       # NMS IoU 阈值
    max_det=100,          # 最大检测数
)
# detection: [B, N, 6] → (x1, y1, x2, y2, score, cls)
```

---

### 3. 初始化轮廓生成

**入口文件**: `lib/utils/snake/snake_decode.py`

#### 3.1 从检测框到初始形状

```python
def get_init(box):
    """
    box: [..., 4] → (x1, y1, x2, y2)

    根据 snake_config.init 选择初始化方式:
    - 'quadrangle': 菱形 (4点)
    - 'octagon': 八边形 (12点) [V3默认]
    - 'box': 矩形 (4点)
    """
    if snake_config.init == 'quadrangle':
        return get_quadrangle(box)  # 菱形
    elif snake_config.init == 'octagon':
        ex = get_quadrangle(box)
        return get_octagon(ex)       # 八边形 [V3]
    else:
        return get_box(box)          # 矩形
```

#### 3.2 八边形构造 (V3)

```python
def get_octagon(ex):
    """
    ex: [..., 4, 2]  极点顺序: Top, Left, Bottom, Right

    输出: [..., 12, 2]  canonical octagon (DeepSnake 风格)

    算法:
    1. 计算宽高: w = R.x - L.x, h = B.y - T.y
    2. 对每个极点，向两侧延伸 w/8 或 h/8
    3. 使用 min/max 裁剪，防止越过相邻极点边界
    """
    w = ex[..., 3, 0] - ex[..., 1, 0]  # Right.x - Left.x
    h = ex[..., 2, 1] - ex[..., 0, 1]  # Bottom.y - Top.y
    x = 8.0  # 延伸系数

    octagon = [
        # Top 极点附近 (2点)
        ex[..., 0, 0], ex[..., 0, 1],
        max(ex[..., 0, 0] - w/x, l), ex[..., 0, 1],

        # Left 极点附近 (3点)
        ex[..., 1, 0], max(ex[..., 1, 1] - h/x, t),
        ex[..., 1, 0], ex[..., 1, 1],
        ex[..., 1, 0], min(ex[..., 1, 1] + h/x, b),

        # Bottom 极点附近 (3点)
        max(ex[..., 2, 0] - w/x, l), ex[..., 2, 1],
        ex[..., 2, 0], ex[..., 2, 1],
        min(ex[..., 2, 0] + w/x, r), ex[..., 2, 1],

        # Right 极点附近 (3点)
        ex[..., 3, 0], min(ex[..., 3, 1] + h/x, b),
        ex[..., 3, 0], ex[..., 3, 1],
        ex[..., 3, 0], max(ex[..., 3, 1] - h/x, t),

        # 回到 Top (1点)
        min(ex[..., 0, 0] + w/x, r), ex[..., 0, 1],
    ]
    return torch.stack(octagon, dim=-1).view(*ex.shape[:-2], 12, 2)
```

#### 3.3 上采样到 128 点

```python
def uniform_upsample(poly, p_num):
    """
    poly: [B, N, V, 2]  V 可以是 4 (矩形) 或 12 (八边形)
    p_num: 128

    输出: [B, N, 128, 2]

    算法: 按边长比例均匀采样
    """
    # 计算每条边的长度
    edge_len = (next_poly - poly).pow(2).sum(3).sqrt()

    # 按比例分配点数
    edge_num = round(edge_len * p_num / total_edge_len)

    # 在每条边上均匀采样
    ...
```

---

### 4. 扩散模型演化

**入口文件**: `lib/networks/diffusion/pretrain_evolution.py`

#### 4.1 训练阶段

```python
def forward(self, output, cnn_feature, batch):
    # 1. 准备训练数据
    init = snake_gcn_utils.prepare_training(output, batch)
    i_it_py = init['i_it_py']  # 初始轮廓 [N, 128, 2]
    i_gt_py = init['i_gt_py']  # GT 轮廓 [N, 128, 2]

    # 2. 方向对齐 (顺时针/逆时针)
    area_init = signed_area(i_it_py)
    area_gt = signed_area(i_gt_py)
    if orient_mismatch:
        i_gt_py = torch.flip(i_gt_py, dims=[1])

    # 3. 起点对齐 (最近点 roll)
    d2 = (i_it_py[:, :1] - i_gt_py).pow(2).sum(-1)
    nearest = argmin(d2, dim=1)
    i_gt_py = torch.roll(i_gt_py, shifts=-nearest, dims=1)

    # 4. 计算目标位移场
    x0 = i_gt_py - i_it_py  # [N, 128, 2]
    x0 = normalize_disp(x0)  # 归一化

    # 5. 加噪
    t = torch.randint(0, T, (N,))  # 随机时间步
    noise = torch.randn_like(x0)
    x_t = add_noise(x0, noise, t)  # q(x_t | x_0)

    # 6. 预测噪声
    eps_pred, L = predict_eps(cnn_feature, i_it_py, c_it_py, py_ind, x_t, t)

    # 7. 计算损失
    loss = F.mse_loss(eps_pred, noise)

    return loss
```

#### 4.2 推理阶段

```python
def sample_disp(self, cnn_feature, i_it_py, c_it_py, py_ind, steps=50):
    """
    DDIM 采样

    输入:
    - cnn_feature: [B, 64, H/4, W/4]
    - i_it_py: [N, 128, 2] 初始轮廓

    输出:
    - disp: [N, 128, 2] 预测位移场
    """
    # 1. 从纯噪声开始
    x = torch.randn(N, 128, 2)

    # 2. 设置 DDIM 调度器
    self.scheduler.set_timesteps(steps)

    # 3. 逐步去噪
    for t in self.scheduler.timesteps:  # [1000, 980, ..., 0]
        # 预测噪声
        eps_pred, _ = self.predict_eps(cnn_feature, i_it_py, c_it_py, py_ind, x, t)

        # DDIM step
        x = self.scheduler.step(model_output=eps_pred, timestep=t, sample=x).prev_sample

    # 4. 反归一化
    disp = self.denormalize_disp(x)

    return disp

# 最终轮廓
pred_poly = i_it_py + disp
```

---

### 5. 训练与推理

#### 5.1 训练流程

**入口文件**: `diffusion_train.py`

```
┌─────────────────────────────────────────────────────────────┐
│                      训练流程                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  for epoch in range(num_epochs):                            │
│      for batch in data_loader:                              │
│          │                                                  │
│          ├──▶ 1. 数据移至 GPU                                │
│          │                                                  │
│          ├──▶ 2. 前向传播                                    │
│          │       output, loss, loss_stats, _ = model(batch) │
│          │                                                  │
│          ├──▶ 3. 反向传播                                    │
│          │       optimizer.zero_grad()                      │
│          │       loss.backward()                            │
│          │       clip_grad_value_(parameters, 40)           │
│          │       optimizer.step()                           │
│          │                                                  │
│          ├──▶ 4. 学习率调度                                  │
│          │       scheduler.step()  # 余弦退火               │
│          │                                                  │
│          └──▶ 5. 日志记录                                    │
│                  json_logger.log(entry)                     │
│                  wandb.log(entry)                           │
│                                                             │
│      if (epoch + 1) % save_ep == 0:                         │
│          save_checkpoint(epoch)                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**损失函数**:

```python
# 总损失 = YOLO 检测损失 + 扩散去噪损失
loss = det_loss * det_weight + diff_loss * diff_weight

# 扩散损失
diff_loss = MSE(eps_pred, eps_gt)  # 噪声预测误差

# 可选: 平滑损失 / 曲率损失 (默认权重为 0，需在配置中显式开启)
# smooth_loss: Laplacian 平滑，惩罚相邻顶点间的剧烈变化
# curv_loss: 二阶导数损失，惩罚轮廓曲率过大
# 开启方式: 在配置文件的 loss_scales 中设置 smooth > 0 或 curv > 0
```

#### 5.3 GRPO 强化学习训练 (实验性)

**入口文件**: `grpo_train.py`

使用 Group Relative Policy Optimization 对扩散模型进行强化学习微调，以边界 Dice 作为奖励信号：

```
┌─────────────────────────────────────────────────────────────┐
│                    GRPO 训练流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  for epoch in range(num_epochs):                            │
│      for batch in data_loader:                              │
│          │                                                  │
│          ├──▶ 1. 从当前策略采样多条轨迹                        │
│          │     (多次 diffusion rollout, 记录 logprob)         │
│          │                                                  │
│          ├──▶ 2. 计算奖励: mBoundDice (多容忍度边界 Dice)     │
│          │                                                  │
│          ├──▶ 3. GAE 优势估计                                │
│          │                                                  │
│          ├──▶ 4. PPO-clip 策略梯度更新 (带 KL 正则)          │
│          │                                                  │
│          └──▶ 5. 记录奖励/散度/kl                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**关键文件**:
- `lib/networks/diffusion/grpo_evolution.py` — GRPO 演化 wrapper，含 `sample_with_logprob` 和 GAE 优势计算
- `lib/networks/diffusion/ddpm_with_logprob.py` — DDPM 采样返回 log-probability
- `lib/train/rewards/region_reward.py` — mBoundF 奖励函数（多容忍度边界 Dice）
- `lib/train/trainers/diffusion_grpo_trainer.py` — GRPO 训练 wrapper

**注意**: GRPO 训练仍在实验中，`grpo_train.py` 标注为"需调整"状态。

#### 5.2 推理流程

**入口文件**: `infer_v3_refinement.py`（兼容入口，实际实现位于 `scripts/infer_v3_final.py`）

```python
def run_inference(model, batch):
    # 1. YOLO 检测 + 特征提取
    yolo_out = model.yolo(batch['inp'])
    p2 = yolo_out[1][0]
    cnn_feature = model.cnn_proj(p2)

    # 2. 准备初始轮廓 (V3: 八边形)
    i_it_py, ind, valid_mask = prepare_v3_init(batch)

    # 3. 扩散采样
    c_it_py = img_poly_to_can_poly(i_it_py)
    disp = model.gcn.sample_disp(cnn_feature, i_it_py, c_it_py, ind, steps=50)

    # 4. 得到最终轮廓
    pred_poly = i_it_py + disp

    # 5. 可视化
    # GT: 蓝色, 初始: 黄色, 预测: 红色
    draw_results(orig_img, pred_poly, i_it_py, gt_poly)
```

#### 5.3 自适应曲率平滑后处理

**入口文件**: `edge_smoothing.py`, `scripts/infer_v3_with_smoothing.py`

扩散模型输出轮廓后，可通过 **曲率自适应平滑** 消除锯齿状边缘，同时保留尖角特征。

**核心机制**: 根据每个顶点的局部曲率自动调节平滑强度。

```python
class EdgeAwareSmoothing:
    def compute_curvature(self, contour):
        # 环形二阶差分近似曲率 (使用 torch.roll)
        d1 = contour - torch.roll(contour, 1, dims=1)
        d2 = d1 - torch.roll(d1, 1, dims=1)
        return torch.norm(d2, dim=-1, keepdim=True)

    def smooth(self, contour):
        curvature = self.compute_curvature(contour)
        # 关键公式: 高曲率 → 低平滑权重; 平坦 → 高平滑权重
        smooth_weight = torch.exp(-curvature / self.curvature_threshold)
        # 局部平滑: (prev + 2*contour + next) / 4
        smoothed = (torch.roll(contour, 1, dims=1)
                    + 2 * contour
                    + torch.roll(contour, -1, dims=1)) / 4
        return smooth_weight * smoothed + (1 - smooth_weight) * contour
```

> 以上为简化示意，完整实现见 `edge_smoothing.py`。

**行为示意**:

```
轮廓顶点曲率分布:
   平坦区域  ───────  尖角 ^  ───────  平坦区域
   smooth_weight≈0.9    smooth_weight≈0.01   smooth_weight≈0.9

效果:
   平坦处强力消除锯齿    尖角几乎不碰          平坦处强力消除锯齿
```

**使用**:

```bash
# 带自适应平滑的 V3 推理
python scripts/infer_v3_with_smoothing.py --ckpt <path>

# 单独测试平滑效果（可调 curvature_threshold，默认 5.0）
python verify_edge_smoothing.py
```

**参数说明**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `curvature_threshold` | 5.0 | 越小越激进平滑（高曲区也受影响） |
| `iterations` | 2 | 平滑迭代次数 |

#### 5.4 单样本过拟合训练

用于快速验证架构改动是否有效。选取单一样本进行过拟合训练，观察模型是否能拟合该样本的轮廓：

```bash
# 以 V3.5 单样本过拟合为例
export CFG_FILE=configs/btcv_diffusion_dit_v3_5_single_overfit.yaml
python diffusion_train.py

# 批量推理对比所有版本
python infer_all_single_overfit.py
# 输出: 各版本可视化对比图
```

支持的版本：V2, V2.1, V2.2, V2.3, V3, V3.1, V3.2, V3.3a, V3.3b, V3.4, V3.5，均有对应的 `_single_overfit` 配置文件。

#### 5.5 位移场归一化

**入口文件**: `compute_disp_stats.py`

计算训练集中位移场的统计信息（均值、标准差），用于归一化扩散模型的位移目标：

```bash
python compute_disp_stats.py
# 输出: data/stats/btcv_disp_stats.json
```

配置中通过 `diffusion_disp_norm: true` 启用归一化，推理时自动反归一化。

---

## 模型版本详解

### 版本对比表

| 版本 | 初始化 | 去噪器 | 视觉编码 | 位置编码 | 注意力模式 | 特殊功能 |
|------|--------|--------|----------|----------|-----------|---------|
| **V1** | 矩形 | `DiTDenoiser` | Perceiver | SnakePosEnc | Cross | 基础 Cross-Attention |
| **V2** | 矩形 | `DiTDenoiserV2` | Perceiver + Local | CyclicRoPE | Cross (奇偶交替) | CyclicRoPE, 奇偶交替注意力 |
| **V2.1** | 矩形 | `DiTDenoiserV2` | SpatialAnchor + Local | CyclicRoPE | Cross (奇偶交替) | 换 SpatialAnchor 编码器 |
| **V2.2** | 矩形 | `DiTDenoiserV2_2` | Patchify | - | Joint (MM-DiT) | Patchify 全局上下文 |
| **V2.3** | 矩形 | FlowMatching | MM-DiT Joint | - | **Flow Matching** | ODE 连续向量场演化 |
| **V3** | **八边形** | `DiTDenoiserV3` | SpatialAnchor + Local | CyclicRoPE | Cross → Self | 八边形初始化 + Self-then-Cross |
| **V3.1** | **八边形** | `DiTDenoiserV3_1` | Perceiver + Patchify | CyclicRoPE | Cross → Self | 双全局上下文对齐 V2 语义 |
| **V3.2** | **八边形** | `DiTFlowMatchingV3_2` | Self+Cross+Patchify | CyclicRoPE | **Flow Matching** | ODE 演化 + 八边形 |
| **V3.3** | **八边形** | `DiTDenoiserV3_3` | Perceiver + Local | CyclicRoPE | Cross → Self | Circular Conv1d 平滑约束 |
| **V3.5** | **八边形** | `DiTDenoiserV3_5` | Perceiver + FourierBridge | CyclicRoPE | Fourier-Space | **傅里叶空间扩散 (K=16)** |

### V3 关键改进

```python
# V3 配置
use_dit_v3: True          # 启用 V3

# V3 初始化: 八边形
if cfg.use_dit_v3:
    snake_config.init = 'octagon'

# V3 注意力流: Cross → Self
class DiTBlockV3:
    def forward(self, x, context, t_emb):
        # 1. 先做 Cross-Attention (聚合图像上下文)
        x = x + self.cross_attn(x, context, t_emb)

        # 2. 再做 Self-Attention (细化点间关系)
        x = x + self.self_attn(x, t_emb)

        # 3. FFN
        x = x + self.ffn(x, t_emb)

        return x
```

### V3.2: Flow Matching 演化

**入口文件**: `lib/networks/diffusion/flow_matching_evolution.py`

用 ODE 连续向量场替代离散 DDPM 采样，通过 Rectified Flow 训练直接预测速度场而非噪声：

```python
# Flow Matching 训练: 预测速度场 v = dx/dt
class FlowMatchingEvolution:
    def forward(self, output, cnn_feature, batch):
        # 1. 计算 x0 (目标位移), x1 (纯噪声)
        # 2. 线性插值: x_t = t * x1 + (1-t) * x0
        # 3. 速度场: v = x1 - x0 (恒定)
        # 4. 预测: v_pred = denoiser(x_t, t, context)
        # 5. 损失: MSE(v_pred, v)
        loss = F.mse_loss(v_pred, x1 - x0)
        return loss
```

推理时通过 ODE 积分从噪声逐步演化到位移场，支持 V2.3/V3.2 去噪器。

### V3.3: Circular Conv1d 平滑约束

**入口文件**: `lib/networks/diffusion/dit_denoiser_v3_3.py`

在最终输出层前加入 Circular Conv1d，利用局部邻域一致性约束防止异常顶点：

```python
# V3.3a/b 变体:
# V3.3a: CircularConv1d 替换最后 FFN 的输出投影
# V3.3b: CircularConv1d 作为额外分支与 FFN 输出相加
class DiTDenoiserV3_3:
    def forward(self, x, context, t_emb):
        # ... standard DiT blocks ...
        x = self.circular_conv1d(x)  # kernel 沿轮廓维度滑动，首尾相连
        return self.output_head(x)
```

### V3.5: 傅里叶空间扩散 (最新)

**入口文件**: `lib/networks/diffusion/dit_denoiser_v3_5.py`, `lib/networks/diffusion/pretrain_evolution.py`

不再在 128 个顶点坐标上直接扩散，而是将轮廓变换到傅里叶空间（K=16 个复系数），扩散过程在频域进行：

```python
# Evolution wrapper 中的变换 (pretrain_evolution.py)
def disp_to_fourier(self, disp):
    # 空间坐标 [N, 128, 2] → 傅里叶系数 [N, K, 4]
    coeffs = torch.fft.rfft(disp, n=K, dim=1)
    coeffs = torch.cat([coeffs.real, coeffs.imag], dim=-1)
    return coeffs

def fourier_to_disp(self, coeffs):
    # 傅里叶系数 → IFFT → 空间坐标
    real = coeffs[..., :2]
    imag = coeffs[..., 2:]
    complex_coeffs = real + 1j * imag
    return torch.fft.irfft(complex_coeffs, n=128, dim=1)

# DiTDenoiserV3_5 在傅里叶空间操作 (接收已变换的系数)
class DiTDenoiserV3_5:
    def forward(self, x_t, context, t_emb):
        # x_t 已经是傅里叶系数 [N, K, 4]
        # 纯注意力网络，无任何 FFT 操作
        return self.fourier_denoiser(x_t, context, t_emb)
```

优势：天然保证输出轮廓平滑（高频分量被截断），无需后处理。

---

## 配置文件说明

**V3 配置**: `configs/btcv_diffusion_dit_v3.yaml`

```yaml
# ===== 模型配置 =====
model: 'sbd'
network: 'ro_34'
task: 'snake'

# ===== DiT 版本选择 =====
use_dit_v3: true          # V3 启用八边形初始化

# ===== 扩散参数 =====
diffusion_timesteps: 1000
use_ddim_inference: true
diffusion_loss_weight: 1.0
diffusion_disp_stats: "data/stats/btcv_disp_stats.json"

# ===== 训练参数 =====
train:
  lr: 5e-5
  batch_size: 64
  epoch: 1000
  warmup_steps: 1000
  save_ep: 100

# ===== 检测参数 =====
det_conf_thresh: 0.01
det_iou_thresh: 0.45
det_max_det: 100

# ===== 损失权重 =====
loss_scales:
  det: 0      # 冻结 YOLO 时设为 0
  py: 1.2
```

---

## 快速开始

### 环境配置

```bash
conda create -n snake1 python=3.10
conda activate snake1
pip install torch torchvision diffusers opencv-python numpy pyyaml tqdm
pip install ultralytics  # YOLOv8
```

### 训练

```bash
# V3 训练 (推荐)
export CFG_FILE=configs/btcv_diffusion_dit_v3.yaml
python diffusion_train.py

# 多卡训练
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 diffusion_train.py
```

### 推理

```bash
# V3 推理
export CFG_FILE=configs/btcv_diffusion_dit_v3.yaml
python infer_v3_refinement.py --ckpt data/outputs/btcv_diffusion_dit_v3/checkpoints/latest.pt
# 输出目录: visual/v3_clean_eval/
```

### 验证八边形初始化

```bash
python verify_octagon_v3.py
# 输出: visual/octagon_comparison.png
#       visual/octagon_multi_boxes.png
```

---

## 关键文件索引

### 核心模块

| 文件 | 说明 |
|------|------|
| `lib/networks/snake/ct_snake.py` | 主网络定义 (YOLO + Evolution) |
| `lib/networks/diffusion/pretrain_evolution.py` | 扩散训练主模块，去噪器选择与调度 |
| `lib/networks/diffusion/flow_matching_evolution.py` | Flow Matching (ODE) 演化 wrapper |
| `lib/networks/diffusion/grpo_evolution.py` | GRPO 强化学习演化 wrapper |
| `lib/networks/diffusion/dit_denoiser_v3.py` | V3 DiT 去噪器 |
| `lib/networks/diffusion/dit_denoiser_v3_2.py` | V3.2 Flow Matching 去噪器 |
| `lib/networks/diffusion/dit_denoiser_v3_3.py` | V3.3 Circular Conv1d 去噪器 |
| `lib/networks/diffusion/dit_denoiser_v3_5.py` | V3.5 傅里叶空间去噪器 |
| `lib/utils/snake/snake_decode.py` | 初始化轮廓生成 |
| `lib/utils/snake/snake_gcn_utils.py` | 训练/测试数据准备 |
| `lib/datasets/voc/snake.py` | 数据集加载 |

### 训练相关

| 文件 | 说明 |
|------|------|
| `diffusion_train.py` | 主训练入口 (36KB) |
| `grpo_train.py` | GRPO 强化学习训练 |
| `lib/train/trainers/diffusion_trainer.py` | 标准训练 wrapper |
| `lib/train/trainers/diffusion_grpo_trainer.py` | GRPO 训练 wrapper |
| `lib/train/rewards/region_reward.py` | mBoundF 奖励函数 |

### 推理与分析脚本

| 文件 | 说明 |
|------|------|
| `infer_v3_refinement.py` | V3 推理兼容入口 |
| `scripts/infer_v3_final.py` | 当前 V3 推理实现 |
| `scripts/infer_v3_with_smoothing.py` | V3 + 边缘平滑推理 |
| `scripts/infer_v3_2_refinement.py` | V3.2 Flow Matching 推理 |
| `scripts/infer_all_versions.py` | 多版本对比推理 |
| `scripts/infer_single_sample.py` | 单样本推理 |
| `scripts/infer_without_yolo.py` | 使用 GT 检测的推理 |
| `infer_all_single_overfit.py` | 单样本过拟合批量推理 |

### 工具与分析

| 文件 | 说明 |
|------|------|
| `verify_octagon_v3.py` | 八边形初始化验证 |
| `edge_smoothing.py` | 边缘感知平滑模块 |
| `compute_disp_stats.py` | 位移场统计计算 |
| `compute_octagon_stats.py` | 八边形初始化统计 |
| `analyze_init_quality.py` | 初始化轮廓质量分析 |
| `test_fourier_smooth.py` | V3.5 傅里叶平滑测试 |
| `test_v3_5_inference.py` | V3.5 推理测试 |
| `sync_logs_to_wandb.py` | JSON 日志同步至 WandB |
| `tools/crf/crf_postprocessor.py` | CRF 后处理模块 |
| `lib/networks/vision_mamba2/` | Vision Mamba2 集成 |

---

## 可视化

训练/推理过程自动生成可视化结果 (目录名含时间戳):

```
visual/
├── v3_clean_eval/                # 当前 V3 推理结果
│   └── CLEAN_v3_*.png
│       ├── GT: 蓝色
│       ├── Init: 黄色
│       └── Pred: 红色
├── edge_aware_postprocess_*/     # 自适应曲率平滑结果
├── fourier_smooth_test/          # V3.5 傅里叶平滑测试
├── fourier_postprocess_*/        # V3.5 傅里叶后处理
├── single_sample_all_models/     # 单样本多版本对比
├── fourier_normal_single_sample_*/  # V3.5 单样本结果
├── fourier_stronger_v3_*/        # V3.5 强化实验
├── single_sample_normal_latest_*/   # 最新单样本结果
└── training_sample_*_zoomed.png  # 训练样本局部放大
```

---

## 更新日志

- **2026-04-17**: V3.5 傅里叶空间扩散 pipeline 完成，推理测试 & 配置
- **2026-04-16**: V3.3 Circular Conv1d 平滑约束 (V3.3a/b 变体)；边缘感知平滑后处理；单样本过拟合训练流程 (V2~V3.5)
- **2026-04-15**: 单样本过拟合配置批量创建；批量推理对比脚本
- **2026-04-13**: 训练可视化改进，初始化质量分析
- **2026-04-04**: 修复 V3 八边形初始化，实现 canonical 12 点版本
- **2026-04-03**: 完成 V3 DiT 去噪器实现
- **2026-04-02**: V2 系列稳定版本
- **2026-03-11**: V1 DiT 基础版本

---

## 参考文献

- DeepSnake: [Peng et al., CVPR 2020]
- DiT: [Peebles & Xie, ICCV 2023]
- MM-DiT / SD3: [Esser et al., 2024]
- YOLOv8: [Ultralytics, 2023]
- DDPM: [Ho et al., NeurIPS 2020]
- DDIM: [Song et al., ICLR 2021]
- Flow Matching / Rectified Flow: [Lipman et al., ICLR 2023], [Liu et al., 2022]
- GRPO: [Shao et al., 2024] (DeepSeekMath)
- Vision Mamba: [Hatamizadeh et al., 2024]

---

## License

MIT License
