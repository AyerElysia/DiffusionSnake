# DiffusionSnake 项目行动计划

> **制定日期**: 2026-04-14  
> **目标**: 修复性能问题 → 获得评估指标 → SAM 实验 → 确定论文方向

---

## Phase 0: 紧急 Bug 修复（立刻做，1小时）

### 0.1 删除 empty_cache (训练速度 +10-15%)
```
文件: lib/train/trainers/trainer.py:127
操作: 删除 torch.cuda.empty_cache() 这一行
影响: 训练速度立即提升
```

### 0.2 修复 num_workers 被强制为 0 (数据加载 ×3-4)
```
文件: diffusion_train.py:125
操作: 删除 cfg.train.num_workers = 0 这一行，让 yaml 配置的 4 生效
注意: 确保数据集没有多进程安全问题
```

### 0.3 修复 .item() 循环 (每 step 减少 GPU 停顿)
```
文件: lib/networks/diffusion/pretrain_evolution.py:396-405
操作: 
  nearest_cpu = nearest.cpu().tolist()
  for i in range(i_gt_py.size(0)):
      s = nearest_cpu[i]
      ...
```

### 0.4 删除无用的 clone
```
文件: lib/networks/diffusion/pretrain_evolution.py:376
操作: 删除 i_gt_py_orig = i_gt_py.clone()
```

### 0.5 修复数值安全问题
```
文件: lib/utils/snake/snake_decode.py:120,126
操作: torch.sqrt(torch.clamp(b1.pow(2) - 4*a1*c1, min=0))

文件: lib/utils/net_utils.py:34-35
操作: torch.log(pred.clamp(min=1e-6))
```

---

## Phase 1: 评估（修完 Bug 后立刻做，2-3小时）

### 1.1 写快速评估脚本
```python
# tools/quick_eval.py
# 加载现有 checkpoint → 跑验证集 → 计算 IoU/Dice/HD95
# 对 V2 epoch_200, epoch_100 和 V3 epoch_200, epoch_100 分别评估
# 关键: 用 DDIM 20步 而不是 50步来加速，先看趋势
```

### 1.2 评估矩阵
```
需要评估的组合:
| Checkpoint        | DDIM Steps | 评估 |
|-------------------|-----------|------|
| V2 epoch_200      | 50        | 基线 |
| V2 epoch_200      | 20        | 快速 |
| V2 latest (ep47)  | 20        | 当前 |
| V3 epoch_200      | 50        | 基线 |
| V3 epoch_200      | 20        | 快速 |
```

### 1.3 预期产出
- 知道当前模型的**实际性能**
- 知道 DDIM 步数减少对性能的影响
- 知道 V2 vs V3 的**真实差距**

---

## Phase 2: SAM 实验（Phase 1 完成后，1-2天）

### 2.1 准备工作
- 下载 MobileSAM 权重 (mobile_sam.pt)
- 确认项目中已有 SAM 代码 (`lib/networks/YOLOV8/models/sam/`)
- 写 `lib/utils/snake/sam_init.py`

### 2.2 推理时 SAM 初始化
```
修改 lib/utils/snake/snake_gcn_utils.py:prepare_testing_init()
添加 cfg.use_sam_init 开关
SAM 接收 YOLO 检测框作为 box prompt → 输出 mask → 转轮廓 → upsample
```

### 2.3 评估矩阵
```
| 初始化方式 | DDIM Steps | 评估 |
|-----------|-----------|------|
| 矩形 (V2) | 50        | 基线 |
| 矩形 (V2) | 20        | 速度 |
| 八边形 (V3)| 50        | 对比 |
| MobileSAM  | 50        | SAM基线 |
| MobileSAM  | 20        | SAM快速 |
| MobileSAM  | 10        | SAM极速 |
```

### 2.4 如果 SAM 有效
→ 写 Phase 2 的 SAM offline teacher (预计算训练集的 SAM 轮廓)
→ 重训练 denoiser，让它学习小位移的 refinement
→ 更新 disp_stats

---

## Phase 3: 确定论文方向（Phase 2 出结果后，1天）

### 3.1 可能的论文方向

**方向 A: "SAM-guided Diffusion Contour Evolution"**
- 核心创新: SAM 提供强初始化 + Diffusion 做精细边界修正
- 对比: 纯 SAM vs 纯 Diffusion vs SAM + Diffusion
- 优势: 结合了两个领域的最新方法
- 适用于: 医学影像分割会议 (MICCAI, MedIA)

**方向 B: "Efficient Contour Diffusion with Foundation Model Priors"**  
- 核心创新: 利用基础模型 (SAM) 大幅减少扩散步数
- 卖点: 50步→10步，5倍加速，精度不降
- 适用于: 效率导向的会议 (ECCV, CVPR workshop)

**方向 C: "Diffusion DiT for Medical Image Contour Segmentation"**
- 核心创新: 将 DiT 架构引入轮廓演化
- 消融: CyclicRoPE, SwiGLU, Multi-scale context 的贡献
- 适用于: 方法论会议 (CVPR, ICCV)

### 3.2 决策依据
- 如果 SAM + Diffusion > 纯 SAM > 纯 Diffusion → 方向 A
- 如果 SAM 初始化 + 少步去噪 ≈ 50步去噪 → 方向 B
- 如果 V2 效果就很好 → 方向 C (不需要 SAM)

---

## Phase 4: 训练与实验（持续进行）

### 4.1 训练计划

**V2 训练 (已在运行，不要停)**
- 继续在 GPU 5-7 跑
- 目标: 至少 200 epoch
- 每 50 epoch 做一次评估

**V3 训练 (修完 Bug 后重启)**
- 在 GPU 0 上跑 (单卡, batch_size 4-6)
- 或者等 ubuntu 释放 GPU 后用更多卡
- 目标: 至少和 V2 同等 epoch 数

**SAM 实验 (不需要训练)**
- 推理时替换初始化器即可
- 在 GPU 0 上跑

### 4.2 GPU 使用规划

| GPU | 任务 | 优先级 |
|-----|------|--------|
| 0 | 评估 + SAM 实验 | 高 |
| 1-4 | ubuntu 占用中 | N/A |
| 5-7 | V2 继续训练 | 中 |

### 4.3 等训练的时候做什么

1. **写评估代码** (Phase 1)
2. **写 SAM 初始化代码** (Phase 2)
3. **画表格、准备可视化代码**
4. **文献调研**: SAM + contour, Diffusion segmentation 最新论文
5. **写论文的 Introduction 和 Related Work** (不需要实验结果也能写)
6. **整理现有可视化结果** (visual/ 目录下已有很多)

---

## 需要重点关注的风险

| 风险 | 概率 | 应对 |
|------|------|------|
| V2/V3 跑了 200 epoch 效果仍然差 | 中 | 检查 loss 是否还在下降，如果plateau考虑调lr |
| SAM 在 BTCV CT 数据上效果差 | 低 | SAM 在 CT 上通常 OK，可换 SAM-Med2D |
| GPU 资源不够 | 高 | 和 ubuntu 用户协调，或错峰使用 |
| 论文 deadline 前出不了结果 | 中 | 优先跑最有价值的实验（SAM对比） |
| 代码 Bug 导致之前的训练结果不可靠 | 低 | 大部分 Bug 不影响核心训练逻辑，empty_cache只影响速度 |

---

## 检查清单 (Checklist)

### 今天必须完成 ✅
- [ ] 修复 5 个 P0 Bug
- [ ] 重启 V2 训练 (应用 Bug 修复)
- [ ] 写 quick_eval.py 评估脚本
- [ ] 跑一次现有 checkpoint 评估

### 明天必须完成
- [ ] SAM 初始化代码
- [ ] SAM vs octagon vs box 对比评估
- [ ] 决定 DDIM 步数

### 本周完成
- [ ] 确定论文方向
- [ ] 开始写论文框架
- [ ] 完整的 ablation study

---

## 附录: 快速命令参考

```bash
# 修完 Bug 后重启 V2 训练
tmux new-session -d -s v2_fixed 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate snake1 && cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30 && DDP_FIND_UNUSED_PARAMETERS=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 torchrun --nnodes=1 --nproc_per_node=3 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29541 diffusion_train.py --cfg_file configs/btcv_diffusion_dit_v2.yaml gpus "[5,6,7]" train.batch_size 20 train.num_workers 4 train.data_path /mnt/sdb1/leijh/DiffusionSnake/Datasets/BTCV/btcv_png_new_snake diffusion_disp_stats data/stats/btcv_disp_stats.json 2>&1 | tee -a data/outputs/btcv_diffusion_dit_v2/train_fixed.log'

# 在 GPU 0 上评估
CUDA_VISIBLE_DEVICES=0 python tools/quick_eval.py --cfg_file configs/btcv_diffusion_dit_v2.yaml --checkpoint data/outputs/btcv_diffusion_dit_v2/checkpoints/epoch_200.pt --ddim_steps 20

# 在 GPU 0 上跑 V3 单卡训练
CUDA_VISIBLE_DEVICES=0 python diffusion_train.py --cfg_file configs/btcv_diffusion_dit_v3.yaml gpus "[0]" train.batch_size 6 train.num_workers 4 train.data_path /mnt/sdb1/leijh/DiffusionSnake/Datasets/BTCV/btcv_png_new_snake
```
