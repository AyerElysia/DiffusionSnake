# H1 严格 GT-box / Memory-off 可视化审计

日期：2026-08-03  
状态：完成；未启动新训练

## 1. 审计对象与隔离条件

- 当前推荐模型：H1 dense-residual 蒸馏头；主干来自 v0.5 step2300。
- checkpoint：`data/outputs/volmem/output_head_h0_h1_h2_20260803/distilled/h1_distilled_full.pt`
- 验证病例：`sub-verse010`、`sub-verse011`、`sub-verse013`，共 333 张切片。
- 固定 seed：`20260731`；Batch 8。
- box：`gt`。配置同时启用 `skip_heatmap_detector_when_gt`，因此本次推理不调用检测器。
- Memory：`parallel-off`，evidence source 与 selection policy 均为 `none`，测得 mean read delta 为 `0.0`。
- 输出图目录：`visual/memflowdit/h1_distilled_gt_memory_off_strict_batch8_20260803/`。

上述条件把检测器与 Memory 同时隔离；可视化中的误差属于 GT 初始化后的 2D 特征、扩散/流演化和输出头组合，不能归因于检测器。

## 2. 严格复现结果

| 指标 | 数值 |
|---|---:|
| Volume mean Dice | 0.796703 |
| Foreground-slice mean Dice | 0.770191 |
| Class mean Dice | 0.751118 |
| All-slice mean Dice | 0.879230 |
| 前景切片 | 175/333 |
| 有预测的前景切片 | 172/175 |
| 推理速度 | 2.395 slice/s |
| 峰值显存 | 2.540 GB |

`all-slice mean Dice` 被 158 张空切片抬高，不作为质量主结论。应优先看 Volume、foreground-slice 和 class mean Dice。

病例结果：

| 病例 | Volume Dice |
|---|---:|
| sub-verse010 | 0.796724 |
| sub-verse011 | 0.819899 |
| sub-verse013 | 0.773486 |

该结果与原 H1 严格评估的 `0.796703` 精确一致，说明保存可视化没有改变推理口径。

## 3. 可视化说明与观察

图例：黄色为 GT 与预测重叠，绿色为 GT-only 漏分，红色为 prediction-only 多分。全量 175 张前景图位于 `viz/`，六张代表图及拼图位于 `selected/`。

代表切片：

- `sub-verse010 slice0064`：典型，Dice 0.798；
- `sub-verse010 slice0117`：困难，Dice 0.713；
- `sub-verse011 slice0012`：典型，Dice 0.825；
- `sub-verse011 slice0007`：困难，Dice 0.745；
- `sub-verse013 slice0064`：典型，Dice 0.776；
- `sub-verse013 slice0044`：困难，Dice 0.632。

主要现象：

1. 大部分目标已经被正确初始化并覆盖，失败不是整块漏检；误差主要表现为轮廓沿解剖边界的系统性收缩、膨胀和局部偏移。
2. 小目标、薄结构和断裂/部分可见结构误差显著，单点双线性采样对 MoonViT 粗语义网格的局部边界解析可能不足。
3. `sub-verse013` 对视野、尺度和成像差异更敏感，是当前最差病例；不能只用均值掩盖。
4. H1 相对 H0 的严格增益只有约 `+0.000129` Volume Dice，H2 也未拉开差距。继续复杂化输出头 MoE 不太可能解决图中主误差。

## 4. 下一步建议：先归因，再允许短训练

### 4.1 零训练演化归因

在同 checkpoint、seed、病例、GT box 下保存初始轮廓与每个 refinement stage 的 Dice，严格回答误差是在初始化、早期演化还是末段过冲中出现；同时比较当前求解流程与统一 8-NFE AB2。若求解器/调度即可恢复质量，不修改网络。

### 4.2 最小采样候选

只有在归因指向局部特征不足时，才测试一个简洁候选：用共享的 3x3 MoonViT token 邻域软聚合替代当前单点采样。它不增加法向细节分支、不增加专家、不新增辅助 loss，仅保留一个共享投影。先做短训练证伪；相对 H1 的三病例 Volume Dice 提升低于 0.002，或最差病例没有改善，则删除。

### 4.3 整卷并行 3D 推理第一阶段

在独立工作区复用 H1/Dense 主干，只做零训练与短训练证伪：

1. 两轮并行/Jacobi Memory；
2. 单轮双向 Volume Feature Context；
3. `parallel-off` 无 3D 对照。

统一使用 8-NFE AB2、同 checkpoint/seed/病例，同时测试 GT 与 predicted box、Batch 1/4/8/16。报告真实 DiT 调用、有效整卷 pass、端到端延迟、峰值显存、Volume/FG Dice；coarse+refine 明确记为两遍。

质量门：相对 AR-8 平均 Dice 下降不超过 0.003、最差病例下降不超过 0.01；速度门：至少额外 2x。feature-context 若不能比 parallel-off 提升约 0.001 Dice，则删除。

## 5. 当前决策

- 当前质量锚点暂时保留 H1 dense residual，不继续追加输出头 MoE 复杂度。
- 尚无证据支持直接进行大规模训练。
- 下一次计算预算优先用于零训练归因和整卷并行严格筛选；只允许通过门槛的单一最小改动进入短训练。
