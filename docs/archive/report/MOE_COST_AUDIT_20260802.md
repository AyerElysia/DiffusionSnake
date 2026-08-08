# MemFlowDiT 纯 MoE 参数量与推理成本审计

日期：2026-08-02
状态：参数审计、三体积 batch1/batch8 实测和代码路径审计均已完成

## 一、结论

当前候选的参数增长可以接受，但推理实现不能整体判定为可接受。

- 参数：相对“输出头 MoE + dense DiT”基线，总参数从 `40.560M` 增至 `45.429M`，增长 `12.00%`，低于本次设定的 15% 容量门槛；训练峰值显存只增长 `0.61%`。
- 顺序推理：batch1 吞吐从 `0.3957` 降至 `0.3649 slice/s`，新增三层 E4K1 的额外减速为 `7.79%`，通过 10% 门槛。
- 并行推理：batch8 吞吐从 `1.9410` 降至 `1.6367 slice/s`，新增三层 E4K1 的额外减速为 `15.68%`，未通过 10% 门槛。
- 旧输出头 MoE 还有一个独立效率问题：代码在 Top-2 选择前已经计算全部 8 个专家，因此参数上是 MoE，计算上不是稀疏 MoE。它在 batch8 单独造成约 19.1% 的吞吐损失。

因此，当前决策是：保留 E4K1 的参数容量设计，不扩为 E8 或六层全 MoE；在完成稀疏/分组专家调度优化前，不把当前实现标记为“并行部署可接受”。

## 二、审计对象与公平条件

“纯 MoE”在当前代码中的准确含义是：

1. 输出头仍为 8 专家 Top-2 MoE；
2. 6 个 DiT block 中仅第 1、3、5 层的 dense SwiGLU FFN 被 E4 Top-1 替换；
3. 这 3 个稀疏 block 内不保留 shared dense FFN 或 inactive dense 备份；
4. 其余 3 个 DiT block 仍为 dense FFN。

它不是“6 层 DiT 全部替换成 MoE”。本次比较三种结构：

| 名称 | 输出头 | DiT FFN |
|---|---|---|
| fully dense | standard | 6 层 dense |
| output-MoE control | 8 专家 Top-2 | 6 层 dense |
| pure E4K1 odd | 8 专家 Top-2 | 奇数 3 层 E4 Top-1，偶数 3 层 dense |

推理固定使用：

- v0.5/v0.6 各自 step6100 checkpoint；
- 同 3 个 validation volumes，共 333 个有效评估切片；
- GT-box，排除检测器影响；
- Memory-off，排除 Memory 路径影响；
- seed `20260731`；
- batch1 与 batch8 分别成对并行运行。

主结果目录：`data/outputs/volmem/diagnostics/moe_cost_audit_20260802/`
机器可读总表：`data/outputs/volmem/diagnostics/moe_cost_audit_20260802/comparison.json`

## 三、精确参数量

统计对象是完整 `MemFlowDiTSnake`，包含 slice network 与 Memory wrapper，而不是只统计 DiT 子模块。

| 结构 | 总参数 | 可训练参数 | FP32 权重存储 | 相对前一项 |
|---|---:|---:|---:|---:|
| fully dense | 39.600542M | 16.292418M | 151.064 MiB | — |
| output-MoE control | 40.560054M | 17.251930M | 154.724 MiB | 总参数 +2.42% |
| pure E4K1 odd | 45.429174M | 22.121050M | 173.299 MiB | 总参数 +12.00% |

pure E4K1 相对 fully dense 的总增长为 `14.72%`；相对 output-MoE control 的可训练参数增长为 `28.22%`。后一个比例较大，是因为 backbone 中大量冻结参数稀释了总参数比例。

三个 `PrototypePhiMoE` 每层包含：4 个专家总计 `2.162688M` 参数，单专家 `0.540672M`，prototype/router 仅 `0.001024M`，Top-1 单条路由实际涉及 `0.541696M` 参数。

全模型总容量为 `45.429M`，但单条路由条件使用量约为 `40.563M`，几乎等于 output-MoE dense 对照的 `40.560M`。所以增加的是专家容量，不是每个 token 同时执行的理论 FFN 宽度。

## 四、checkpoint 与训练显存

| 结构 | 单 checkpoint | 差值 |
|---|---:|---:|
| output-MoE dense DiT | 300.746 MB | — |
| pure E4K1 odd | 359.228 MB | +58.482 MB / +19.45% |

若滚动保留 20 个 checkpoint，额外磁盘约为 1.17 GB（十进制）。该增长可接受，但应继续滚动保留，不能无限保存每 100-step 权重。

对 step4200–6100 的 1901 个训练 step 做日志统计：

| 结构 | median step time | trimmed mean | 峰值显存 |
|---|---:|---:|---:|
| output-MoE dense DiT | 10.125 s | 10.712 s | 30.536 GB |
| pure E4K1 odd | 9.400 s | 9.531 s | 30.722 GB |

两次训练位于不同物理 GPU，且可能受到不同并发负载影响，因此不能宣称 pure MoE 训练更快；可以确认的是没有观察到训练减速，峰值显存只增加 `0.185 GB / 0.61%`。

## 五、真实三体积推理速度

### 5.1 batch1

| 结构 | slice/s | 相对前一项 | 峰值显存 | Volume Dice |
|---|---:|---:|---:|---:|
| fully dense | 0.429833 | — | 0.509 GB | 0.172230 |
| output-MoE control | 0.395707 | -7.94% | 0.514 GB | 0.793707 |
| pure E4K1 odd | 0.364900 | -7.79% | 0.541 GB | 0.794462 |

pure E4K1 相对 fully dense 的累计减速为 `15.11%`，但相对实际可用的 output-MoE 主线，新增部分为 `7.79%`，通过 10% 门槛。

### 5.2 batch8

| 结构 | slice/s | 相对前一项 | 峰值显存 | Volume Dice |
|---|---:|---:|---:|---:|
| fully dense | 2.398583 | — | 2.539 GB | 0.171802 |
| output-MoE control | 1.940962 | -19.08% | 2.546 GB | 0.794160 |
| pure E4K1 odd | 1.636650 | -15.68% | 2.573 GB | 0.794934 |

pure E4K1 相对 fully dense 的累计减速为 `31.77%`。新增 DiT-MoE 自身的 15.68% 已超过门槛，因此当前实现不适合作为并行推理的最终版本。

fully dense 的标准输出头没有作为独立模型重新训练到 step6100，它只加载了 checkpoint 中兼容的共享基线头；其 Dice 不能作为公平的性能消融，但可代表相同主干下去掉输出专家计算后的延迟下界。output-MoE 与 pure E4K1 来自不同训练分支，两者约 `+0.0008` 的 Dice 差异也不能作为严格因果收益来抵扣速度成本。

## 六、速度根因

### 6.1 输出头 Top-2 是伪稀疏计算

`MoEFinalHead.forward` 当前先通过两次 dense `einsum` 生成全部 8 个专家的输出，再根据 Top-2 router 做 gather。未选专家虽然最终没有梯度贡献，但其前向矩阵乘法已经执行。batch 越大，全部专家计算越明显，因此输出头减速从 batch1 的 7.94% 放大到 batch8 的 19.08%。

### 6.2 新 DiT E4K1 是真实稀疏，但调度未融合

`PrototypePhiMoE` 只对被选专家执行 FFN，理论 token FLOPs 与一个 dense FFN 同量级。但实现按 4 个专家循环，使用 `nonzero`、子集索引和 `index_add_` 分派 token。每个 ODE step 的 3 个稀疏层会产生多次小矩阵乘法和 kernel launch；batch8 时这种分组调度开销扩大，导致 15.68% 的额外减速。

## 七、准入决策与后续动作

本轮使用两个门槛：相对 output-MoE dense DiT 总参数增长不超过 15%；相同 batch 下新增 E4K1 推理减速不超过 10%。结果为：参数 `+12.00%`，通过；batch1 `-7.79%`，通过；batch8 `-15.68%`，失败。

因此不扩专家数、不改为六层全 MoE。下一步优化顺序应为：

1. 将输出头改为先路由、再只计算选中专家，保留现有 8 专家 Top-2 权重和输出语义；
2. 为 E4K1 使用按专家排序/分组的 fused dispatch，减少 `nonzero/index_add` 和小 GEMM 启动；
3. 优化后重新跑同一三体积 batch1/batch8 配对；
4. 同时通过数值等价测试、Dice 不回退和两种 batch 的 10% 速度门槛，才标记为部署可接受。

在这些条件满足前，可以继续把 E4K1 作为训练研究候选，但不能把当前实现直接当作最终并行推理主线。
