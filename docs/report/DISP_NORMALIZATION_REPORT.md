---
description: Displacement (disp) normalization report (dataset-wide min/max)
---

# 位移向量（disp）归一化实现报告（DiffusionSnake-12-30）

## 1. 目标与背景

在扩散分支中，模型学习预测轮廓点的位移向量：

- 训练目标位移（GT disp）：`x0 = i_gt_py - i_init_train_py`，形状 `[N, P, 2]`
- 推理输出位移（pred disp）：采样得到的 `disp`，用于生成最终轮廓：`py = i_it_py + disp`

在不同数据集/不同类别/不同器官大小下，像素尺度位移的范围差异很大，会导致：

- 扩散噪声建模尺度不稳定（位移范围过大时相对噪声比例变化）
- 训练收敛速度变慢，或 loss 数值不易对齐

因此引入 **按数据集统计得到的全局 min/max** 作为尺度，对位移向量做归一化，并在推理时反归一化回像素尺度。

## 2. 归一化规则

采用 **按通道分别统计** 的 min/max：

- `dx_min, dx_max`
- `dy_min, dy_max`

并将像素尺度的位移 `disp` 映射到 `[-1, 1]`：

- `disp_norm = (disp - disp_min) * 2/(disp_max - disp_min) - 1`

其中：

- `disp_min = [dx_min, dy_min]`（broadcast 到 `[1,1,2]`）
- `disp_max = [dx_max, dy_max]`

反归一化（推理输出回像素尺度）：

- `disp = (disp_norm + 1) * 0.5 * (disp_max - disp_min) + disp_min`

> 若 stats 文件不存在/加载失败，归一化逻辑会自动回退为“无操作”（不改变 disp）。

## 3. 统计 min/max 的位移定义（与训练对齐）

为了避免“统计口径”和“训练口径”不一致，统计时使用与训练相同的构造方式：

1) 从数据中取 GT 多边形 `i_gt_py` 与 GT 4点 `i_gt_4py`
2) 用 GT 4点外接矩形生成训练用 init 轮廓：

- `gt_rect4 = snake_decode.get_box(gt_boxes)`
- `i_init_train_py = uniform_upsample(gt_rect4, poly_num)`

3) 对齐方向与起点（与训练代码一致）：

- 方向：通过 signed area 判断是否需要 flip
- 起点：将 GT 多边形 roll，使其起点与 init 轮廓的第一个点最近

4) 计算位移：

- `disp = i_gt_py - i_init_train_py`

并在全训练集上统计 `disp[...,0]` 与 `disp[...,1]` 的全局 min/max。

## 4. 代码改动点（实现位置）

### 4.1 配置项默认值

文件：`lib/config/config.py`

新增默认配置（不改动原有训练行为，默认关闭）：

- `cfg.diffusion_disp_norm = False`
- `cfg.diffusion_disp_stats = ''`

### 4.2 扩散模块中加载 stats + 归一化/反归一化

文件：`lib/networks/diffusion/pretrain_evolution.py`

实现内容：

- 初始化读取：
  - `diffusion_disp_norm`：是否启用归一化
  - `diffusion_disp_stats`：stats JSON 路径
- 增加函数：
  - `normalize_disp(disp)`
  - `denormalize_disp(disp_norm)`

**训练时**：

- 在 `x0 = i_gt_py - i_init_train_py` 后立刻调用：
  - `x0 = self.normalize_disp(x0)`

**推理时**：

- 推理分支用 `sample_disp()` 产生 `disp`
- `sample_disp()` 内部在返回前执行：
  - `return self.denormalize_disp(x)`

因此推理端拿到的 `disp` 已经是**像素尺度位移**，直接 `py = i_it_py + disp`。

### 4.3 GRPO 采样输出反归一化

文件：`lib/networks/diffusion/grpo_evolution.py`

为了保证 reward 与几何计算在像素尺度上进行，采样末端位移修改为：

- `disp = self.denormalize_disp(x)`
- `py = i_it_py + disp`

## 5. 离线统计脚本

文件：`compute_disp_stats.py`

功能：

- 使用 `--cfg_file` 加载指定数据集配置
- 遍历训练集 DataLoader
- 计算并保存 `{dx_min, dx_max, dy_min, dy_max}` 到 `cfg.diffusion_disp_stats`

输出格式（JSON）：

```json
{
  "dx_min": -89.17,
  "dx_max": 55.79,
  "dy_min": -77.85,
  "dy_max": 49.46
}
```

## 6. 如何使用（推荐流程）

### 6.1 在 yaml 中配置

示例（BTCV）：`configs/btcv_diffusion_snake.yaml`

- `diffusion_disp_norm: true`
- `diffusion_disp_stats: "data/stats/btcv_disp_stats.json"`

示例（RAOS）：`configs/raos_diffusion_snake.yaml`

- `diffusion_disp_norm: true`
- `diffusion_disp_stats: "data/stats/raos_disp_stats.json"`

示例（processed1232）：`configs/processed1232_diffusion_snake.yaml`

- `diffusion_disp_norm: true`
- `diffusion_disp_stats: "data/stats/processed1232_disp_stats.json"`

### 6.2 先统计，再训练/推理

1) 统计（只需做一次，或数据集更新后重做）：

```bash
python compute_disp_stats.py --cfg_file configs/btcv_diffusion_snake.yaml
```

2) 训练（示例）：

```bash
python diffusion_train.py --cfg_file configs/btcv_diffusion_snake.yaml
```

3) 推理（示例）：

```bash
python infer_without_yolo.py --cfg_file configs/btcv_diffusion_snake.yaml --ckpt /path/to/latest.pt
```

## 7. 常见问题（FAQ）

### Q1: 忘记反归一化会怎样？

推理时如果 `disp` 仍在 `[-1,1]` 空间，直接加到像素坐标会导致位移严重偏小（几乎不动），几何结果错误。

本实现中已在：

- `sample_disp()` 返回前
- `GRPOEvolution.sample_with_logprob()` 返回前

做了 `denormalize_disp()`，避免遗漏。

### Q2: stats 文件缺失怎么办？

如果启用了 `diffusion_disp_norm=true` 但 stats 文件路径不存在或无法解析，本实现会回退为“不做归一化”。

推荐做法：

- 训练/推理前先跑一次 `compute_disp_stats.py`
- 确认 `data/stats/*.json` 存在

