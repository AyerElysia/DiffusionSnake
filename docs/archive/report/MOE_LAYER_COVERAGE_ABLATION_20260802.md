# MemFlowDiT DiT-MoE 层覆盖率消融（2026-08-02）

## 1. 审计结论

新版 `PrototypePhiMoE` 此前只运行过奇数三层配置，没有运行过同结构、同训练协议的六层全 MoE。报告中“保留稳定 dense 路径、避免路由叠加、参考视觉 DiT-MoE 交错经验”属于设计先验，不是本任务实验结论。

旧 V4.10 存在六层 legacy FFN-MoE 配置，但它使用旧线性/循环路由、Top-2、共享 dense 路径和不同数据任务，不能证明当前 prototype E4 Top-1 应该或不应该覆盖六层。

## 2. 直接对照

| 项目 | Odd-3 Control | All-6 |
|---|---|---|
| DiT-MoE 层 | 1、3、5 | 0–5 全部 |
| 专家 | 每层 E4 | 每层 E4 |
| 激活 | Top-1 | Top-1 |
| 路由 | contour-level prototype + φ | 相同 |
| 共享专家 | 无 | 无 |
| 输出头 | 8 专家 Top-2 hard-φ | 相同 |
| 初始权重 | 同一 dense checkpoint | 同一 dense checkpoint |
| 训练 | 1000 step、chunks=4、seed 20260802 | 相同 |

Odd-3 复用已完成的严格 control：`data/outputs/volmem/verse_memflowdit_moe_shared_ablation_control_gpu0/checkpoints/step_001000.pt`。All-6 使用独立 GPU0 和独立输出目录训练。

## 3. 训练前参数审计

| 指标 | Odd-3 | All-6 | All-6 相对变化 |
|---|---:|---:|---:|
| 总参数 | 45.429174M | 50.298294M | +10.72% |
| 可训练参数 | 22.121050M | 26.990170M | +22.01% |
| 每条路由条件参数 | 40.563126M | 40.566198M | +0.008% |
| FP32 参数存储 | 173.299 MiB | 191.873 MiB | +10.72% |

All-6 相对 output-MoE dense-DiT 基线 `40.560M` 的总参数增加约 24.0%。理论激活参数基本不变，因为每层仍只执行一个与 dense FFN 等宽的专家；但真实实现会新增三层专家分组、索引和 kernel launch，必须用 Batch 1/8 实测判断。

### 3.1 step-200 早期趋势（非最终结论）

双方同为 step 200 时进行了三体积、GT box、Memory-off、Batch 8 快速评估：

| 指标 | Odd-3 | All-6 | All-6 - Odd-3 |
|---|---:|---:|---:|
| volume Dice | **0.787691** | 0.784626 | **-0.003064** |
| foreground slice Dice | **0.757703** | 0.757174 | -0.000529 |
| class mean Dice | 0.736099 | **0.736669** | +0.000570 |
| Batch-8 吞吐 | **1.663782** | 1.519215 | **-8.69%** |
| 峰值推理显存 | **2.5734 GB** | 2.6006 GB | +1.06% |

三个体积的 volume Dice 均下降：`-0.003990`、`-0.002614`、`-0.002589`。因此早期信号明显不支持 All-6；但仍继续完成预注册的 1000-step 训练，避免把收敛速度差异误判为最终结构差异。

## 4. 评估协议与决策

- 三个固定验证体积；
- GT box、Memory-off；
- 固定评估 seed `20260731`；
- Batch 1 和 Batch 8；
- 两轮交叉物理 GPU 测速；
- 同时比较 Dice、前景 Dice、类别均值、路由 hard CV、总参数、激活参数、训练显存和吞吐。

只有 All-6 带来明确且稳定的质量提升，且效率代价与提升匹配，才考虑替换 Odd-3。噪声级改善不足以接受额外 10.72% 总参数和三层路由调度。

## 5. 运行状态

- 配置：`configs/volmem/verse_memflowdit_moe_layer_ablation_all6_gpu0.yaml`
- 输出：`data/outputs/volmem/verse_memflowdit_moe_layer_ablation_all6_gpu0`
- 汇总：`data/outputs/volmem/diagnostics/moe_layer_ablation_odd3_vs_all6_20260802`
- 自动评估：`tools/volmem/watch_and_eval_moe_layer_ablation.sh`

状态：All-6 1000-step 训练已启动；完成后自动运行质量评估和两轮交叉 GPU 吞吐测试，最终结果待回填。
