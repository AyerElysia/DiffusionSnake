# MemFlowDiT：六层 Dense 与三层 DiT-MoE 直接消融（2026-08-02）

## 1. 证据审计

此前没有完成“六层 Dense FFN vs 奇数三层 E4 Top-1”的同初始化、同 seed、1000-step 严格对照。

已有证据只有：

- 50-step、单验证体积：Dense `0.788544`，Odd-3 E4K1 `0.790186`，差值 `+0.001642`；
- step-6100 成本审计中 Odd-3 比 output-MoE dense-DiT 高约 `+0.0008 Dice`，但两者来自不同训练分支，不能做因果归因。

因此不能在正式叙事中宣称“三层 MoE 已被证明优于六层 Dense”。本实验补齐该证据。

## 2. 严格协议

| 项目 | Dense-6 | Odd-3 MoE |
|---|---|---|
| 6 个 DiT FFN | 全部 dense | 奇数三层 E4 Top-1，偶数三层 dense |
| 输出头 | 8 专家 Top-2 hard-φ | 相同 |
| 初始 checkpoint | 同一个 dense 2D checkpoint | 相同 |
| 训练 | 1000 step、chunks=4 | 相同 |
| 训练 seed | 20260802 | 20260802 |
| 评估 | 三体积、GT box、Memory-off | 相同 |
| 速度 | Batch 1/8、两轮交叉 GPU | 相同 |

Odd-3 使用已完成的 checkpoint：`data/outputs/volmem/verse_memflowdit_moe_shared_ablation_control_gpu0/checkpoints/step_001000.pt`。

## 3. 参数审计

| 指标 | Dense-6 | Odd-3 | Odd-3 相对变化 |
|---|---:|---:|---:|
| 总参数 | 40.560054M | 45.429174M | +12.00% |
| 可训练参数 | 17.251930M | 22.121050M | +28.22% |
| 每条路由条件参数 | 40.560054M | 40.563126M | +0.008% |

Odd-3 的激活参数理论上与 Dense-6 几乎相同，但存储参数增加 12%，且专家分发可能造成真实延迟。只有质量提升明确且与速度代价匹配，才保留三层 MoE。

## 4. 运行位置

- Dense 配置：`configs/volmem/verse_memflowdit_moe_layer_ablation_dense6_gpu1.yaml`
- Dense 输出：`data/outputs/volmem/verse_memflowdit_moe_layer_ablation_dense6_gpu1`
- 汇总：`data/outputs/volmem/diagnostics/moe_layer_ablation_odd3_vs_dense6_20260802`
- 自动评估：`tools/volmem/watch_and_eval_moe_dense_ablation.sh`

状态：Dense-6 1000-step 训练已在 GPU1 启动；评估会等待 All-6 实验释放 GPU4–7 后自动执行。最终结论待回填。
