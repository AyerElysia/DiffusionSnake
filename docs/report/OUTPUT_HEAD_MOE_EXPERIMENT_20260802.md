# MemFlowDiT 2026 输出头 MoE 严格筛选实验

日期：2026-08-02，更新于 2026-08-03  
状态：五组 300-step 筛选及完整评估已完成；D1 与 L0 的 1000-step 复核运行中

## 1. 研究问题

本实验不预设输出头 MoE 必然有效，而是回答三个可证伪问题：

1. 输出头增加非线性容量，是否优于原始线性输出头？
2. 在总 specialist 参数几乎相等时，条件专家是否优于单个 dense MLP？
3. 如果条件专家有效，第二个激活专家带来的质量是否值得其额外计算？

DiT 固定为 odd-3 E4 Top-1 prototype MoE；数据、Memory 训练协议、MoonViT、随机种子、chunks 和起始 checkpoint 均相同，唯一变量是最终位移输出头。

## 2. 公平起点

V4.6c 原 checkpoint 已含训练过的旧输出专家。如果直接使用，L0 会继承旧专家，而 D1/M1 的新分支只能随机初始化，比较不公平。

因此生成了共享基线 checkpoint：

`data/outputs/volmem/output_head_moe_2026/checkpoints/shared_base_only.pt`

处理仅移除 `gcn.denoiser.final_layer` 下的 15 个 specialist/router tensor，保留：

- RMSNorm；
- adaLN；
- shared linear displacement predictor；
- 其余 386 个 checkpoint tensor。

来源 SHA256：`83d87390c98ca2bd38f3679af623b562b69595b80ddae030d5a6e5bb9e8cd56e`  
公平 checkpoint SHA256：`b5252e8436100ac1fabb0288bec2a763bba05b2cc5aadf4fa3c7bc7fc0369734`

完整移除清单：

`data/outputs/volmem/output_head_moe_2026/checkpoints/shared_base_only.manifest.json`

## 3. 五组结构

| ID | 输出头 | 专家/MLP | 路由 | 目的 |
|---|---|---:|---|---|
| D0 | 标准 RMSNorm + adaLN + linear | 无 | 无 | 最简 dense 基线 |
| D1 | shared linear + dense residual MLP | hidden=1024 | 无 | 与 M1 的四专家总参数匹配 |
| L0 | 旧 shared MLP + E8 Top-2 | hidden=256 | 逐点 noisy/cyclic | 旧设计控制组 |
| M1-K2 | shared linear + E4 residual MLP | 4×hidden256，激活2 | 轮廓 prototype | 推荐现代候选 |
| M1-K1 | shared linear + E4 residual MLP | 4×hidden256，激活1 | 轮廓 prototype | 第二专家必要性 |

D1 residual MLP 参数为 265,218；M1 四个 residual expert 总参数为 265,224，差 6 个参数。因而 D1 与 M1 的质量差异不能简单归因于 specialist 参数总量。

## 4. 新 M1 实现

### 4.1 计算路径

1. checkpoint-compatible shared linear 预测共同位移；
2. 对 adaLN 后的轮廓 token 做 mean pooling + LayerNorm；
3. 与 4 个真实数据初始化的 learnable prototype 做余弦路由；
4. 先形成被选轮廓索引；
5. 只执行收到轮廓的专家；
6. routed residual 与 shared linear 相加。

没有 point embedding、cyclic router、router noise、shared heavy MLP、类别路由、法向/曲率采样或 gate-residual。

### 4.2 真稀疏验证

已通过以下测试：

- Top-1 只有 1 个 expert `forward` 被调用；
- Top-2 只有 2 个 expert `forward` 被调用；
- 未选的其余专家调用次数严格为 0；
- residual 为 0 时输出与 shared linear 逐元素严格一致；
- router prototype、selected expert 和输入均有非零梯度；
- D1 与 E4 expert pool 参数匹配；
- 旧 `MoEFinalHead` 和 DiT `PrototypePhiMoE` 回归测试继续通过。

完整 M1 构建后额外执行了 13 个 smoke step：无 NaN/Inf，峰值训练显存 11.912 GB，最后一步约 4.77 s；输出头 4 个专家均有激活，证明配置、checkpoint bridge、训练 loss 和诊断链已经打通。

## 5. 训练设置

- seed：`20260802`；
- max steps：300；
- save every：100；
- chunks per step：4；
- gradient accumulation：2；
- prediction evidence 在本阶段尚未启动；
- 现有 All-6/Dense-6 实验不停止。

首批：

- D0：GPU4，PID 4113468；
- D1：GPU5，PID 4113471；
- M1-K2：GPU6，PID 4113477；
- M1-K1：GPU7，PID 4113481；
- watcher：PID 4113485；
- L0：等待 D0 完成后自动使用 GPU4。

启动器：`tools/volmem/run_output_head_moe_2026_screen.sh`  
自动流程：`tools/volmem/watch_output_head_moe_2026_screen.sh`

## 6. 自动评估

全部 step300 落盘后，固定同一物理 GPU4 顺序运行：

1. GT box + memory-off + batch1，测顺序推理；
2. GT box + memory-off + batch8，测并行推理与主要因果质量；
3. GT box + autoregressive + batch1，确认候选不破坏真实记忆路径。

每组固定 3 个 validation volumes、seed `20260731`。同卡串行评估避免 GPU 个体差异和并发 I/O 影响小幅 latency 结论。

结果根目录：

`data/outputs/volmem/diagnostics/output_head_moe_2026_screen_20260802/`

自动汇总：

- `comparison.json`；
- `summary.tsv`；
- `summary.txt`；
- 每组每模式独立 `summary.json` 与 `slices.json`。

## 7. 判定规则

300-step 是筛选，不直接作为最终论文结论。进入 1000-step 的候选必须：

1. 相对 D1 的 volume Dice 有稳定正趋势；目标达到 +0.002，或后续配对 CI 排除 0；
2. foreground slice Dice、NSD/HD95 和可视化不恶化；
3. batch8 相对 D1 减速不超过 10%，目标不超过 5%；
4. 无死亡输出专家；
5. autoregressive 不出现额外退化。

若 M1 不能明显超过 D1，则删除输出头 MoE。若 M1-K2 不明显优于 M1-K1，则选择更简单的 Top-1。zero/linear 轻量专家只有在基础 M1 通过后才允许进入下一轮。

## 8. Step300 完整结果（2026-08-02 23:38）

五组训练以及 GT-box 的 memory-off、autoregressive 评估均已完成。两个评估模式的排序一致，D1 dense residual 均为第一。

| 结构 | 总参数 | 输出头参数 | Memory-off Volume Dice | Autoregressive Volume Dice | 前景切片 Dice | batch8 slice/s | 输出头死亡专家 |
|---|---:|---:|---:|---:|---:|---:|---:|
| D0 linear | 44.487M | 0.132M | 0.566857 | 0.590864 | 0.449857 | 1.9779 | — |
| **D1 dense residual** | **44.753M** | **0.398M** | **0.637586** | **0.644637** | **0.519479** | **2.1301** | — |
| L0 legacy E8K2 | 45.447M | 1.092M | 0.627521 | 0.637290 | 0.509550 | 1.7162 | 2 |
| M1 modern E4K2 | 44.754M | 0.399M | 0.603666 | 0.623005 | 0.485211 | 1.9568 | 1 |
| M1 modern E4K1 | 44.754M | 0.399M | 0.614113 | 0.634924 | 0.494376 | 1.9501 | 1 |

相对 D1：

- L0 Volume Dice `-0.010065`；
- M1-K1 `-0.023473`；
- M1-K2 `-0.033920`；
- autoregressive 下，L0 相对 D1 仍为 `-0.007347`，没有发生排序反转；
- M1-K1 比 M1-K2 高 `+0.010447`，第二个激活专家在本阶段明显有害；
- D1 与 M1 总参数几乎严格相同，因此差距不能由总 specialist 参数容量解释；
- D1 最后 50 step 平均 diff loss 为 0.012941，优于 L0 的 0.013403、M1-K1 的 0.013442 和 M1-K2 的 0.014011。

路由结果也没有通过门槛：

- L0 hard CV 1.205，2 个专家低于 1%；
- M1-K2 hard load `[9.91%, 39.24%, 0.85%, 50.00%]`，1 个死亡专家；
- M1-K1 hard load `[3.75%, 46.91%, 0.01%, 49.33%]`，1 个死亡专家。

五组的 autoregressive Dice 都高于 memory-off：D0 `+0.024007`、D1 `+0.007051`、L0 `+0.009769`、M1-K2 `+0.019339`、M1-K1 `+0.020811`。这证明本轮协议中 Memory 路径确实产生了正收益，并非完全失效；但不能外推为所有体数据和长历史设定都已解决。

当前证据支持“输出头需要非线性容量”，但反对“这份容量应该由 MoE 提供”。D1 同时优于原始 linear、旧 MoE 和现代 MoE。M1 未达到进入 1000-step 的质量、路由健康和效率门槛，因此停止扩展 zero/linear 专家，不再继续堆叠路由设计。

单次速度表还存在执行顺序与 GPU warm-up 影响，质量排序不受该问题影响。D1 与最佳 MoE 控制组 L0 进入 1000-step 复核，同时执行同卡、双轮、反向顺序测速。

## 9. 1000-step 复核（2026-08-03 启动）

复核只保留当前最优 D1 和 MoE 中最优 L0，仍从同一个 `shared_base_only.pt`、相同 seed 和训练协议重新训练，而不是从 step300 续训，避免训练轨迹与筛选阶段相互污染。

- D1：GPU5，输出 `data/outputs/volmem/verse_memflowdit_output_head_confirm_d1_1000_gpu5/`；
- L0：GPU4，输出 `data/outputs/volmem/verse_memflowdit_output_head_confirm_l0_1000_gpu4/`；
- 自动评估：GPU6；
- 质量：GT-box memory-off batch8 与 autoregressive batch1；
- 速度：GT-box memory-off 的 batch1、batch8 各执行两轮，并在第二轮颠倒 D1/L0 顺序；
- 汇总：`data/outputs/volmem/diagnostics/output_head_moe_2026_confirm_1000_20260803/`。

启动器：`tools/volmem/run_output_head_confirm_d1_vs_l0_1000.sh`  
自动评估：`tools/volmem/watch_output_head_confirm_d1_vs_l0_1000.sh`  
汇总器：`tools/volmem/summarize_output_head_confirm_1000.py`

## 10. 1000-step 最终结论（2026-08-03）

| 指标 | D1 dense residual | L0 legacy E8K2 | D1 - L0 |
|---|---:|---:|---:|
| Memory-off Volume Dice | 0.773345 | **0.774311** | -0.000966 |
| Autoregressive Volume Dice | 0.772983 | **0.773881** | -0.000898 |
| Foreground slice Dice | **0.726761** | 0.724233 | +0.002528 |
| Batch1 slice/s | **0.385203** | 0.343674 | +12.08% |
| Batch8 slice/s | **2.175116** | 1.718635 | +26.56% |

L0 的体积均值优势不足 `0.001`，且三个体积方向不一致；D1 的前景切片 Dice 更高。D1 还减少约 0.694M 总参数、输出头参数减少 63.6%，并在反序双轮测速中获得 12.08%/26.56% 的 Batch1/Batch8 吞吐优势。L0 输出路由在 step1000 仍有 1 个低于 1% 的死亡专家。

本组内部决策：D1 是 fair-head 组的效率候选，L0 不继续优化。但历史审计发现，本组从剥离 15 个成熟输出 specialist/router tensor 的 `shared_base_only.pt` 重新学习，绝对 Dice 比保留成熟输出头的 v0.5 step2300 `0.796574` 低约 `0.0232`。因此本节**不能据此宣布 D1 替换绝对主线输出头**；在函数保持迁移或蒸馏完成前，质量主线仍保留 v0.5 成熟输出头。详见 `docs/report/HISTORICAL_BEST_AUDIT_20260803.md`。

结果：`data/outputs/volmem/diagnostics/output_head_moe_2026_confirm_1000_20260803/comparison.json`

## 11. 成熟头函数蒸馏复核（2026-08-03）

上述“需要函数保持迁移或蒸馏”的缺口已经补齐。以 v0.5 step2300 成熟 E8 Top-2
输出头为教师、固定全部非输出头权重，在 training split 的真实推理轨迹上蒸馏后，D1
在严格三验证体积协议达到 Volume Dice `0.796703`，与教师 `0.796574` 等价；总参数从
40.560M 降至 39.866M，当前环境 Batch-8 时间从 179.070s 降至 140.281s。
Batch-1 下 D1 与教师速度持平（0.41784 vs 0.42021 slices/s），Dice 为 0.796296 vs
0.796106。

新增共享头 + 稀疏专家也做了验证：E4 出现 2 个死专家；改为有界 cosine routing、
动态负载偏置的 E2 后，负载为 62.8% / 37.2%，死亡专家归零，但 Dice `0.796658`，
仍低于 D1，且速度更慢。因此最终证据支持 D1，不再保留输出头 MoE。

完整协议、参数、路由与结果见：
`docs/report/OUTPUT_HEAD_DISTILLATION_H0_H1_H2_20260803.md`。
