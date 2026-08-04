# MemFlowDiT 历史最佳结果审计与结论纠正

日期：2026-08-03  
状态：已审计 175 个可解析 `summary.json`

## 1. 纠正结论

`0.773345` 不是 MemFlowDiT 的历史最好结果，也不能作为当前绝对主线性能。它只是“剥离旧输出专家后，从共享基线重新训练 1000 step”的输出头结构对照结果。

在完全相同的三个验证体积、GT box、Memory-off/parallel-off、seed `20260731`、333 张切片协议下，当前历史最高结果是：

`verse_memflowdit_v0_5_minimal_gpu6/checkpoints/step_002300.pt`

其 common-noise Batch8 Volume Dice 为 **0.796574**。用户记得的约 `0.79` 结果是正确的。

## 2. 可比结果排序

| 模型/checkpoint | Step | Volume Dice | FG slice Dice | Batch8 slice/s | 说明 |
|---|---:|---:|---:|---:|---|
| **v0.5 minimal** | 2300 | **0.796574** | **0.770027** | 1.789 | 当前三体积绝对最好 |
| v0.6 Odd-3 E4K1 | 6100 | 0.794934 | 0.770508 | 1.637 | DiT-MoE 长训练 |
| v0.5 minimal | 8000 | 0.793664 | 0.768609 | — | 长训后反而低于 step2300 |
| 旧 Dense-6 对照 | 1000 | 0.790760 | 0.764289 | 1.733 | 保留预训练旧输出头 |
| 旧 Odd-3 对照 | 1000 | 0.790041 | 0.762932 | 约 1.61 | 保留预训练旧输出头 |
| 新 L0 fair-head | 1000 | 0.774311 | 0.724233 | 1.719 | 旧专家被剥离后重训 |
| 新 D1 fair-head | 1000 | 0.773345 | 0.726761 | 2.175 | 新 dense residual 头重训 |

另有 v0.5 step8000 在单个 `sub-verse010` 上达到 `0.797593`，但只有一个体积，不能替代三体积排名。

## 3. 为什么新结果掉到 0.773

历史 v0.5、旧 Dense/Odd/All/共享消融和新 D1/L0 使用的原始来源其实相同：

`data/outputs/sagittal_2d_v4_6c_moonvit_evolution_gt/checkpoints/epoch_130.pt`

`latest.pt` 正是该文件的软链接，SHA256 均为：

`83d87390c98ca2bd38f3679af623b562b69595b80ddae030d5a6e5bb9e8cd56e`

差异不在数据或源 checkpoint，而在输出头初始化：

1. 历史 v0.5 和旧 DiT 消融直接继承 epoch130 中已经训练成熟的旧输出 MoE；
2. 新 fair-head 实验为了不让 L0 继承成熟专家而占便宜，生成 `shared_base_only.pt`，移除了输出头 15 个 specialist/router tensor；
3. D1 residual 分支和 L0 specialist 分支都需要从近似共享线性函数重新学习；
4. 只训练 1000 step 后，两组都没有恢复旧成熟输出头的绝对能力；
5. 因此该实验能回答“在相同退化起点下 D1 与 L0 谁更高效”，不能回答“D1 是否超过历史主线”。

## 4. 对已有结论的影响

### 4.1 输出头

D1 在 fair-head 组中质量近似打平 L0，并明显更快、更小，因此仍是值得研究的效率候选；但在完成函数保持迁移或 teacher distillation 前，不能替换 v0.5 的成熟输出头作为绝对主线。

### 4.2 DiT MoE

保留成熟旧输出头的旧严格实验已经给出更可靠结论：Dense-6 比 Odd-3 高 `+0.000712 Dice`，前景高 `+0.002712`，Batch1/8 分别快约 12.7%/10.9%，总参数少 12%。All-6 和共享专家也均未超过 Odd-3。因此目前没有证据支持在 DiT FFN 中保留 MoE。

### 4.3 Memory

新 D1 checkpoint 的 Memory 因果审计仍能证明“正确历史明显优于打乱历史”，但其绝对 Dice 不能代表历史主线。历史 v0.5 step2300 同样是 off `0.796574` 高于 frozen predicted `0.796389` 和 autoregressive `0.795397`，所以“Memory 尚无净收益、无限 bank 不应采用”的方向不因本次纠正而改变。

## 5. 当前正确主线与下一步

在新的高质量迁移实验完成前：

- 绝对质量 anchor：v0.5 step2300，GT/3-volume/off Dice `0.796574`；
- DiT：Dense-6；
- 输出头：保留成熟旧输出头作为质量主线，D1 仅为效率候选；
- Memory：默认 off，bounded Memory 只保留为研究能力。

如果继续推进 D1，必须从 v0.5 step2300 的成熟输出头做函数保持迁移或蒸馏，再与原 v0.5 在同协议、同训练预算下比较。不能再用剥离后仅训练 1000-step 的 `0.773` 组替代历史主线。

## 6. 证据位置

- v0.5 step2300：`data/outputs/volmem/diagnostics/parallel_memory_common_noise_v05_step2300/parallel_off/summary.json`
- v0.6 step6100：`data/outputs/volmem/rl3d/base_bakeoff_v1/v06_step6100/summary.json`
- 旧 Dense/Odd：`data/outputs/volmem/diagnostics/moe_layer_ablation_odd3_vs_dense6_20260802/comparison.json`
- 旧 All-6：`data/outputs/volmem/diagnostics/moe_layer_ablation_odd3_vs_all6_20260802/comparison.json`
- 旧共享专家：`data/outputs/volmem/diagnostics/moe_shared_ablation_20260802/comparison.json`
- 新 fair-head：`data/outputs/volmem/diagnostics/output_head_moe_2026_confirm_1000_20260803/comparison.json`

## 7. 后续闭环：成熟头蒸馏

上述函数保持迁移已经于 2026-08-03 完成。固定 v0.5 step2300 的全部非输出头权重，
从成熟 E8 Top-2 教师蒸馏得到的 H1 Dense Residual 在同一严格协议取得 Dice
`0.796703`，与历史 anchor `0.796574` 等价，同时总参数减少 1.71%，当前环境
Batch-8 端到端时间减少 21.7%。输出稀疏 E4 与抗退化 E2 均未超过 H1。

因此输出头建议从“暂时保留成熟旧头”更新为：**H1 是当前主线候选，H0 保留为教师和
回退 checkpoint**。详细证据见
`docs/report/OUTPUT_HEAD_DISTILLATION_H0_H1_H2_20260803.md`。
