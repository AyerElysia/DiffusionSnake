# DiffusionSnake-12-30 三阶段学习率与检测头训练策略报告

## 1. 目标
本次修改的目标是：

- 将训练脚本中的学习率调度从 **余弦 (cosine_with_restarts)** 改为 **阶梯式/阶段式 (staircase / stage)**。
- 支持 **三阶段训练**：每个阶段进入时使用一个新的初始学习率。
- **检测头 (YOLO) 只在第一阶段训练**；第二、三阶段冻结检测头，不再参与反向传播。
- 保持向后兼容：不新增配置时，尽量不影响现有训练流程。

涉及文件：

- `diffusion_train.py`


## 2. 关键结论（先说明最重要的事实）

- **AdamW 本身不会“自动调整学习率”**。
  - AdamW 只会根据一阶/二阶动量等内部状态改变“参数更新量”，但 `lr` 数值本身不会自动变化。
  - **学习率变化只能来自 scheduler** 或手动修改 optimizer 的 `param_groups[].lr`。

因此，如果你希望：

- 同一阶段内 lr 不变
- 只有进入下一阶段时才重新设置 lr

那么需要做的是：

- 阶段内 **不调用 scheduler.step()**（或者不创建 scheduler）
- 阶段切换时 **手动重置 lr**


## 3. 实现内容

### 3.1 增加 `lr_schedule` 模式
在 `diffusion_train.py` 的单阶段 epoch-based 训练循环中，新增读取：

- `cfg.train.lr_schedule`（字符串）

支持两种模式：

- `lr_schedule: step`
  - 使用 `MultiStepLR`（可选 warmup）
  - 每个 iteration 调用 `scheduler.step()`

- `lr_schedule: stage`
  - **不创建 scheduler**，也不会 step scheduler
  - 学习率在每个 stage 内保持恒定
  - **只在 stage 边界重置为新 lr**

默认值为 `step`（如果配置不写该字段）。


### 3.2 三阶段训练参数
新增（可选）配置字段：

- 阶段长度（单位：epoch）
  - `cfg.train.stage1_epoch`（默认：等于 `cfg.train.epoch`）
  - `cfg.train.stage2_epoch`（默认：0）
  - `cfg.train.stage3_epoch`（默认：0）

- 三阶段初始学习率
  - `cfg.train.lr_stage1`（默认：`cfg.train.lr` 或 optimizer 初始 lr）
  - `cfg.train.lr_stage2`（默认：同上）
  - `cfg.train.lr_stage3`（默认：同上）

训练时按 epoch 判定当前 stage：

- stage 1：`epoch < stage1_epoch`
- stage 2：`stage1_epoch <= epoch < stage1_epoch + stage2_epoch`
- stage 3：剩余 epoch

进入新 stage 时：

- 将 optimizer 的每个 param_group 的 `lr` 设为该 stage 的 lr


### 3.3 检测头只在第一阶段训练
新增辅助函数 `_set_yolo_trainable(wrapper, trainable)`，在 stage 切换时调用：

- stage 1：`trainable=True`
  - YOLO 参数 `requires_grad=True`
  - YOLO 进入 `train()`

- stage 2/3：`trainable=False`
  - YOLO 参数 `requires_grad=False`
  - YOLO 进入 `eval()`

这保证：

- 第二、三阶段 **检测头参数不会更新**
- 训练更聚焦在扩散/蛇分支


## 4. 使用方法（配置示例）

### 4.1 阶段内 lr 恒定（符合“阶段内不变，阶段切换重置”）
在配置文件 `configs/*.yaml` 的 `train:` 下加入：

```yaml
train:
  lr_schedule: stage
  stage1_epoch: 50
  stage2_epoch: 30
  stage3_epoch: 20

  lr_stage1: 5e-5
  lr_stage2: 1e-6
  lr_stage3: 1e-8
```

说明：

- stage1 会训练 YOLO；stage2/3 冻结 YOLO。
- stage 内 lr 固定不变。


### 4.2 阶梯式衰减（阶段内也衰减）
若你想阶段内也做阶梯式衰减：

```yaml
train:
  lr_schedule: step
  milestones: [60000, 90000]
  gamma: 0.5
  warmup_steps: 1000
```

说明：

- `milestones` 是以 **step** 为单位（这里的实现按训练 step 来 step scheduler）。
- 如果不提供 `milestones`，代码会使用 `0.6 * total_steps` 和 `0.85 * total_steps` 作为默认衰减点。


## 5. 如何验证是否按预期生效

1) 观察 `logs.jsonl` 中的 `lr` 字段：

- `lr_schedule: stage` 时
  - 同一 stage 内 `lr` 应该保持恒定
  - 进入新 stage 时 `lr` 应出现跳变

2) 验证 YOLO 是否冻结：

- stage2/3 时，`det_loss_scaled` 应接近 0 或不再影响总 loss（即使 det_loss 仍被计算/记录，参数不应更新）。
- 更严格的方式是观察 YOLO 参数 `requires_grad` 状态（必要时可临时打印）。


## 6. 向后兼容性说明

- 不添加任何新配置时：
  - `stage1_epoch = cfg.train.epoch`
  - `stage2_epoch = 0`，`stage3_epoch = 0`
  - `lr_schedule = 'step'`
  - 等价于：单阶段训练 + 阶梯式 scheduler（替代原 cosine）。


## 7. 注意事项

- 你的配置里如果希望启用“阶段内恒定 lr”，必须显式设置：
  - `train.lr_schedule: stage`

否则默认仍会构建 scheduler 并 step，导致 lr 在阶段内变化。

- 如果你想让 det loss 完全不参与（不算、也不反传），除了冻结 YOLO 参数外，也可以将 `loss_scales.det` 配置为 0。但这属于损失层面的开关，与“冻结参数”是两个概念。
