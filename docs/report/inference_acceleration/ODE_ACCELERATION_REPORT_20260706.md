# DiffusionSnake 推理加速报告

**日期**：2026-07-06  
**作者**：自动整理  
**针对版本**：DiffusionSnake-12-30（V4.1/V4.6c，Rectified Flow）

---

## 1. 背景

DiffusionSnake 使用 Rectified Flow（流匹配）驱动轮廓演化。推理阶段代价分析：

| 阶段 | 消耗 |
|------|------|
| 外层迭代（`iterative_num_steps`） | 默认 10次 |
| 每次内层 ODE 积分（`iterative_ode_steps`） | 默认 20步 |
| 每步网络前向（DiT V4.1） | 1次（Euler）或 2次（Heun） |
| **总 NFE** | **200次**（Euler × 10 × 20） |

主要瓶颈是 ODE 积分的**每步都要跑完整 cross-attention**，其中 K/V 投影占 cross-attention 约 2/3 计算量。

本次工作从两个正交方向切入：
1. **更高阶 ODE 求解器（AB2）**：相同步数下更准确，或步数减半保持精度
2. **KV 缓存**：ODE 步间 context 固定，K/V 只算一次然后复用

---

## 2. 方法

### 2.1 Adams-Bashforth 2 阶求解器（AB2）

**原理**  
Euler 法用当前时刻速度积分，误差 O(dt²)：
```
x_{n+1} = x_n + dt · v_n
```
AB2 用当前和上一步两点外推，误差 O(dt³)，每步仍只需 1 次 NFE：
```
x_{n+1} = x_n + dt · (3/2 · v_n  −  1/2 · v_{n−1})
```
第一步没有历史，用 Euler 引导；从第二步开始切换 AB2。

**代码位置**  
`lib/networks/diffusion/flow_matching_evolution.py`，`_sample_disp_from_sampled_feat()` ODE 循环。

**启用方式**  
在 YAML config 里加一行：
```yaml
v3_7_ode_solver: 'ab2'
```
可选值：`euler`（默认）、`heun`（2阶但 2×NFE）、`ab2`（2阶且 1×NFE）。

---

### 2.2 Cross-Attention KV 缓存

**原理**  
ODE 积分期间，image context（`global_ctx` 和 `local_ctx`）在步间完全不变（当 `flow_resample_feat_at_xt=False` 时）。K/V 投影可以在循环前算一次，每步只算 Q 和 attention。

**实现架构（修复后最终版）**  

缓存状态存在 `DiTBlockV3` 内部，正常 `forward` 路径自动使用，不需要单独的旁路函数：

```
FlowMatchingEvolution._set_denoiser_kv_cache()
  └─ 遍历 denoiser.dit_layers
     └─ DiTBlockV3.set_kv_cache(context)   # 存 self._cached_k / _cached_v
        
ODE 循环（调用正常 denoiser.forward）
  └─ DiTBlockV3._cross_attention()
     └─ if self._cached_k: 复用缓存，跳过 K/V 投影

ODE 循环结束后
  └─ FlowMatchingEvolution._clear_denoiser_kv_cache()
     └─ DiTBlockV3.clear_kv_cache()
```

**关键设计决策**  
最初实现（v1，已废弃）在 `DiTFlowMatchingV3_4` 父类加了 `forward_with_kv_cache`，绕过了 `V4.1` 的 MoE 头和 per-point delta，导致精度大幅下降（mIoU 降低 0.031）。修复后将缓存状态内置到 block，正常 forward 路径不变，精度损失仅 0.0008。

**代码位置**  
- `lib/networks/diffusion/dit_blocks_v3.py`：`DiTBlockV3.set_kv_cache / clear_kv_cache / _cross_attention`  
- `lib/networks/diffusion/flow_matching_evolution.py`：`_set_denoiser_kv_cache / _clear_denoiser_kv_cache`

**启用/禁用**  
默认自动启用。强制禁用：
```bash
FLOW_DISABLE_KV_CACHE=1 python scripts/eval_v37_full_iou.py
```
自动跳过条件：`flow_resample_feat_at_xt=True`，或使用 `locate_context`（context 会变化）。

---

## 3. 实验结果

### 3.1 权重修正说明

前一版实验使用的 `1232_final_v4_6c_2d_fm_s_cond_gpu2/checkpoints/latest.pt` 仅完成约35.5%训练（epoch=3550 / step=234300 / total_steps=660000），属于欠训练权重，不能作为最终加速结论依据。

当前正式复核固定使用成熟RL权重：

```text
data/outputs/1232_final_v5_geom8_delta_nsd_bs6_gpu7/checkpoints/best_iou.pt
```

该权重属于同一1232_final数据集和V4.6c/DiT/MoE主干，checkpoint step=3350，已有177/177样本完整评测。所有新消融固定该checkpoint、`eval_seed=20260504`、5个outer refinement steps和相同fractions，禁止跨checkpoint比较。

配置中的 `flow_ode_steps: 10` 不是迭代推理的实际内层步数；启用 `use_iterative_refinement` 后实际读取 `iterative_ode_steps`。本文统一使用 `outer5_innerN` 命名，总NFE=5×N。

### 3.2 成熟RL权重：AB2与步数消融

| 配置 | 求解器 | 实际步数 | 总NFE | mIoU | Dice | mBoundF | NSD |
|---|---|---:|---:|---:|---:|---:|---:|
| Euler outer5_inner20（已有基线） | Euler | 20 | 100 | 0.854005 | 0.920005 | 0.778915 | 0.868963 |
| AB2 outer5_inner20 | AB2 | 20 | 100 | 0.853102 | 0.919452 | 0.777529 | 0.866206 |
| AB2 outer5_inner5 | AB2 | 5 | 25 | 0.857016 | 0.921867 | 0.782278 | 0.875039 |
| AB2 outer5_inner3 | AB2 | 3 | 15 | 0.857344 | 0.922101 | 0.782988 | 0.876632 |

四组均为177/177成功。相对同一成熟权重的Euler inner20，AB2 inner20为−0.000903，inner5为+0.003010，inner3为+0.003339。当前证据显示该成熟RL权重减少内层步数到5/3没有精度下降，略有提升；旧欠训练s-conditioned权重上的“少步必然掉点”不能泛化。

结果目录：`report/inference_acceleration/delta_nsd_*`。

### 3.3 KV缓存精度消融

| 配置 | KV缓存 | mIoU | Dice | mBoundF | NSD |
|---|---|---:|---:|---:|---:|
| AB2 outer5_inner20 | 开 | 0.853102 | 0.919452 | 0.777529 | 0.866206 |
| AB2 outer5_inner20 | 关 | 0.853102 | 0.919452 | 0.777529 | 0.866206 |

两组使用完全相同checkpoint、随机种子和配置，仅通过 `FLOW_DISABLE_KV_CACHE=1` 切换，177个样本指标逐项相同，缓存没有可测精度下降。

---

## 4. 实测Wall-clock耗时（RTX 4090D单卡）

使用 `scripts/bench_ode_speed.py`，batch=1，前5个样本warmup，计时20个样本；以下数据固定成熟RL checkpoint、AB2、`outer5_inner20`。

| 配置 | ODE-only均值 | Full-stage均值 |
|---|---:|---:|
| KV开 | 2512.3 ms | 2532.9 ms |
| KV关 | 2635.8 ms | 2656.8 ms |

关闭KV后纯ODE耗时增加约4.9%，Full-stage增加约4.9%；因此成熟RL权重上KV cache真实收益约4.7~4.9%，应报告为“约5%”，而不是旧欠训练权重测得的6.5%或早期按FLOPs估算的15~25%。两组精度完全一致，可默认开启。

### 4.1 当前结论边界

可以确认：KV cache修复后无可测精度损失；成熟RL权重上约5%实测加速；AB2与Euler接近；成熟RL权重上inner5/3暂未显示精度下降。

不能确认：这些结果是否适用于尚未完成训练的MoonViT feat256 checkpoint；inner1是否稳定；非GT-init或完整detector forward下收益是否相同。inner5/3的wall-clock仍需单独补测，不能沿用旧权重时间曲线。

## 5. 后续工作

### 5.1 Reflow 轨迹直化（中期，需 finetune）

**动机**：AB2 步数消融显示 velocity field 弯曲度高，1/3步效果差。Reflow 通过将随机 (x0, x1) 对替换为 ODE 解出的真实轨迹端点重新训练，使轨迹变直，理论上 1步可用。

**方法**（InstaFlow，Liu et al. 2023）：
1. 用当前模型对训练集跑 ODE 推理，收集 `(x0_actual, x1_predicted)` 对
2. 用这些直线对重新训练 1~2 epoch（finetune，不是从头训练）
3. 验证 1步 AB2 的精度

**预期效果**：1~3步达到现在 10步的精度，总 NFE 降低 7~10×

### 5.2 Consistency Distillation（长期）

训练学生网络直接从任意 t 映射到 x1，实现 1步推理，代价是需要专门的蒸馏训练流程。

### 5.3 渐进式 KV 缓存失效

当前缓存整个 ODE 过程不更新。可以每 N 步刷新一次（N=3~5），在精度和速度间取得更好平衡，尤其对 `resample_feat_at_xt=True` 的场景。

---

## 6. 文件改动清单

| 文件 | 改动 |
|------|------|
| `lib/networks/diffusion/dit_blocks_v3.py` | 新增 `set_kv_cache / clear_kv_cache`；`_cross_attention` 支持内部缓存；`forward` 删除 `cached_kv` 参数 |
| `lib/networks/diffusion/dit_denoiser_v3_4.py` | 新增 `precompute_kv_cache / forward_with_kv_cache`（v1，已废弃，被 v2 取代） |
| `lib/networks/diffusion/flow_matching_evolution.py` | AB2 求解器分支；`_set_denoiser_kv_cache / _clear_denoiser_kv_cache`；`FLOW_DISABLE_KV_CACHE` 环境变量开关；`predict_velocity` 签名更新 |
| `scripts/bench_ode_speed.py`（新增） | 独立 wall-clock 计时脚本，隔离 `sample_disp_iterative` 本身耗时，用于第 4 节实测数据 |

---

## 7. 如何复现

```bash
cd /home/medteam/Zhrch/DiffusionSnake-12-30
CKPT=data/outputs/1232_final_v4_6c_2d_fm_s_cond_gpu2/checkpoints/latest.pt

# Euler baseline（10步）
CFG_FILE=configs/eval_ep3200_euler_s10o20.yaml EVAL_GPU=0 CKPT=$CKPT \
  SAVE_DIR=visual/euler_s10 SAVE_VISUALS=0 \
  conda run -n snake1 python scripts/eval_v37_full_iou.py

# AB2（10步，推荐）
CFG_FILE=configs/eval_ep3200_ab2_s10o20.yaml EVAL_GPU=0 CKPT=$CKPT \
  SAVE_DIR=visual/ab2_s10 SAVE_VISUALS=0 \
  conda run -n snake1 python scripts/eval_v37_full_iou.py

# 禁用 KV 缓存（消融对照）
CFG_FILE=configs/eval_ep3200_ab2_s10o20.yaml EVAL_GPU=0 CKPT=$CKPT \
  FLOW_DISABLE_KV_CACHE=1 SAVE_DIR=visual/ab2_s10_nokv SAVE_VISUALS=0 \
  conda run -n snake1 python scripts/eval_v37_full_iou.py
```

### 7.1 Wall-clock 耗时复现（第 4 节数据）

```bash
cd /home/medteam/Zhrch/DiffusionSnake-12-30
CKPT=data/outputs/1232_final_v4_6c_2d_fm_s_cond_gpu2/checkpoints/latest.pt

# Euler s10 baseline
CFG_FILE=configs/eval_ep3200_euler_s10o20.yaml EVAL_GPU=6 CKPT=$CKPT \
  BENCH_N=40 BENCH_WARMUP=5 conda run -n snake1 python scripts/bench_ode_speed.py

# AB2 s10（KV 开，默认）
CFG_FILE=configs/eval_ep3200_ab2_s10o20.yaml EVAL_GPU=6 CKPT=$CKPT \
  BENCH_N=40 BENCH_WARMUP=5 conda run -n snake1 python scripts/bench_ode_speed.py

# AB2 s10，关闭 KV 缓存（消融对照，得到 4.2 节的 6.5% 数字）
CFG_FILE=configs/eval_ep3200_ab2_s10o20.yaml EVAL_GPU=6 CKPT=$CKPT FLOW_DISABLE_KV_CACHE=1 \
  BENCH_N=40 BENCH_WARMUP=5 conda run -n snake1 python scripts/bench_ode_speed.py
```

脚本会打印每个样本的 `ode=`（纯 ODE 演化耗时）和 `full=`（含特征提取的完整阶段耗时），末尾汇总 mean/median/std。`BENCH_N`/`BENCH_WARMUP` 控制计时/热身样本数，`EVAL_SEED` 控制采样种子（默认与 eval 脚本一致）。
