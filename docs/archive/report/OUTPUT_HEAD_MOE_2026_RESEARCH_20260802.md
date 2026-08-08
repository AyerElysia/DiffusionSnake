# MemFlowDiT 输出头 MoE：2026 新设计调研与验证方案

日期：2026-08-02  
状态：调研完成，尚未修改主线、尚未启动本课题训练

## 1. 先给结论

当前不能证明“输出头必须使用 MoE”。旧输出头虽然已经用 hard-φ 修复了 hard routing 退化，但该修复只证明路由更健康，没有证明分割质量更好；而且当前 Top-2 在实现上是伪稀疏：8 个专家全部计算后才 gather，batch8 相对标准头慢 19.08%。

因此下一步不应直接把旧 8 专家头换成另一个复杂 MoE。最负责任的顺序是：

1. 先做同训练协议的 dense 输出头与参数匹配 dense-MLP 输出头，证明条件专家本身是否必要；
2. 若 MoE 确实优于两种 dense 对照，再验证一个最小的现代稀疏残差头；
3. 只有基础稀疏头证明有效，才增加 MMOE 启发的 zero/linear 轻量路线；
4. 任一组件若没有独立质量或效率收益就删除。

建议的第一候选不是照搬某篇论文，而是把 2026 年证据中与本任务真正兼容的部分组合为：

> **共享线性基线位移 + 轮廓级 E4 Top-2 残差专家 + ProMoE prototype router + 真稀疏 dispatch。**

其中“共享”只保留便宜且可精确继承 checkpoint 的线性基线，不保留旧的重 shared MLP；专家负责预测小残差。路由以整条目标轮廓为单位，避免 128 个相邻点频繁切专家破坏位移场一致性。

## 2. 当前实现审计

当前 `MoEFinalHead` 的有效结构为：

- 8 个 MLP 专家，hidden=256；
- 每个轮廓点独立 Top-2；
- point embedding + time embedding + circular Conv1d + linear router；
- shared base 为 `linear(x + shared_mlp(x))`；
- 最终输出为 shared base 加 routed expert delta；
- hard-φ 使用 hard load EMA 形成拥塞价格，再由当前 soft probability 承载梯度。

### 2.1 已经证实的问题

1. **伪稀疏。** 前向先用两次 dense `einsum` 计算所有 8 个专家，再做 Top-2 gather。未选专家仍消耗前向计算。
2. **路由粒度过细。** 每个点单独路由，9,139,200 次决策中专家 2/5 曾合计占 88.94%。这种细粒度选择没有形成可解释的任务分工。
3. **均衡不等于专门化。** hard-φ 将 100-step hard CV 从 0.9139 降至 0.5698，但三体积 Dice 只变化 +0.000012，前景切片 Dice -0.001338。
4. **没有类别专家证据。** 类别—专家 NMI 为 0.000087，时间—专家 NMI 0.014106，尺度—专家 NMI 0.007215；当前集中不能解释为“各椎体类别各司其职”。
5. **尚无公平的 dense 质量对照。** 成本审计中的 standard head 只加载兼容共享权重，没有独立训练到同一 step，因此其低 Dice 只能用于延迟下界，不能证明输出 MoE 有质量必要性。

### 2.2 成本

- standard head 全模型：39.600542M 参数；
- 旧 output-MoE 全模型：40.560054M，增加 0.959512M / 2.42%；
- batch1：0.429833 -> 0.395707 slice/s，慢 7.94%；
- batch8：2.398583 -> 1.940962 slice/s，慢 19.08%。

参数量不大，真正不能接受的是它没有质量因果证据，却在并行推理中产生显著延迟。

## 3. 2026 年论文中真正相关的证据

### 3.1 MMOE：最新扩散 Transformer 的效率方向

MMOE（2026-07-27）在 SiT 中系统比较 routed、shared、lightweight、gate-residual 和 attention-residual。对本输出头最有价值的不是照搬整个 block，而是三点：

- 只 dispatch 被选中的 token，不能先计算全部专家；
- 专家池不必全是重 MLP，可以有 zero/copy/constant 等轻量路线；
- 路由稳定性、激活计算和质量需要一起评估。

不能直接迁移的部分：attention residual 和跨层 gate residual 都依赖多个 Transformer block；我们的输出头只有一个末端层。copy expert 也不能直接用于 `256 -> 2`，因为输入输出维度不同。constant 2D 位移缺少几何合理性。对本任务真正自然的轻量路线只有 zero correction，以及可选的廉价 linear correction。

来源：[MMOE: Modernizing Diffusion Transformers with Efficient Expert Design](https://arxiv.org/abs/2607.24665)

### 3.2 2026 密集分割 MoE：最贴近输出端的证据

《Design and Behavior of Sparse Mixture-of-Experts Layers in CNN-based Semantic Segmentation》直接研究 dense prediction：

- 4 或 8 专家通常最好，继续增大专家数没有稳定收益；
- Top-2 通常比 Top-1 更准，但继续增加激活专家会恶化；
- 单个靠近 decoder 输出端的 MoE 常优于堆多个 MoE，多层替换甚至显著退化；
- 过复杂 gate 会妨碍学习；
- 其主要 shared-expert 对照不如独立专家；
- 适度 patch routing 优于整图路由，但最佳粒度与架构有关；
- 理论 FLOPs 小不等于实际免费，实测推理开销仍可到 14.45%。

这支持我们只研究一个末端 MoE、限制在 E4/E8、显式测真实速度；也反对继续叠 shared expert、复杂 gate 和更多输出 MoE 层。轮廓任务中的“局部块”不能机械等同于固定点段，因为闭合轮廓起点可能没有稳定语义；现有项目实验又显示逐点路由会退化、整轮廓 prototype routing 能避免死亡，所以第一候选使用轮廓级路由更稳妥。

来源：[Design and Behavior of Sparse Mixture-of-Experts Layers in CNN-based Semantic Segmentation](https://arxiv.org/abs/2604.13761)

### 3.3 ProMoE：视觉路由需要显式语义结构

ProMoE（ICLR 2026）指出视觉 token 冗余会阻碍专家专门化，使用 learnable prototype 与 routing contrastive guidance。我们在 DiT 内部已经验证“整轮廓 descriptor + 真实数据 prototype 初始化”能消除随机 prototype 的专家死亡，因此该路由比重新引入随机线性 point router 更有本项目证据。

来源：[Routing Matters in MoE / ProMoE](https://arxiv.org/abs/2510.24711)

### 3.4 φ-Balancing：只作为防死亡约束

φ-Balancing 直接优化 population-level expert balance，并用 EMA 在线校正。它适合处理小 batch 下 noisy mini-batch balancing，但均衡只能避免专家死亡，不能证明专家与数据对齐。当前 hard-φ 应保留为轻量防退化约束，但权重必须小，不能把“完全均匀”设成目标。

来源：[$\phi$-Balancing for Mixture-of-Experts Training](https://arxiv.org/abs/2605.15403)

### 3.5 暂不作为首轮的方案

- DirMoE 将“选哪些专家”与“专家权重”拆成 Bernoulli + Dirichlet，并以可微随机估计训练；思路新，但 ELBO、Gumbel-Sigmoid 和调度会引入多项新变量，不符合当前简洁性原则。
- Routing-Free MoE 删除 router、softmax、Top-K 和传统平衡；主要证据来自通用/语言 MoE，尚缺扩散轮廓输出验证，首轮风险过高。

来源：[DirMoE](https://arxiv.org/abs/2602.09001)、[Routing-Free MoE](https://arxiv.org/abs/2604.00801)

## 4. 推荐候选：最小现代稀疏残差头

### 4.1 计算图

给定末层轮廓 token `x` 和扩散时间 `t`：

1. 沿用 RMSNorm + adaLN；
2. 共享线性基线 `v_base = Linear(x)`，从旧 checkpoint 精确加载；
3. 对轮廓 token 做 mean pooling，与 time descriptor 融合并归一化；
4. 与 4 个 learnable prototypes 做余弦相似度；
5. Top-2 选择两个残差 MLP 专家；
6. 先分组索引，再只执行被选专家；
7. 输出 `v = v_base + alpha * sum(gate_e * delta_e(x))`。

初始化时 `delta_e` 的最后一层近零，使新模型起点严格接近 dense 基线。`alpha` 从小值学习，防止早期专家扰乱已训练位移场。

### 4.2 为什么是 E4 Top-2

- 2026 dense segmentation 中 4/8 专家通常足够，更多专家没有稳定收益；
- 当前类别/时间/尺度统计都没有支持 8 个输出专家的明确多模态结构；
- Top-2 是最新 dense prediction 中更常胜出的设置，也允许两个残差场平滑组合；
- E4 Top-2 比旧 E8 全计算显著节省容量和激活计算。

但 E4 Top-2 是待验证假设，不是已成立结论。首轮同时保留 E4 Top-1 的低成本速度/质量对照；如果 Top-2 提升小于噪声就选择 Top-1。

### 4.3 是否使用 shared expert

使用“共享线性基线”，不使用“共享重 MLP 专家”。原因是：

- 线性基线能无损继承 checkpoint，并承担所有样本共有的位移规律；
- 重 shared MLP 会与 routed MLP 重复计算；
- 2026 dense segmentation 的 shared-expert 对照没有显示稳定优势；
- 当前旧头已经有 shared MLP，却没有给出质量收益证据。

### 4.4 MMOE 轻量专家何时加入

仅在基础 E4 稀疏头证明优于 dense 后，增加第二候选：`4 heavy + 1 zero correction + 1 linear correction`。

- zero 表示当前样本只需要 shared base，不做专家修正；
- linear 是 `256 -> 2` 的廉价输入相关修正，替代不适用于本任务的 copy/constant；
- 如果轻量路线使用率低于 5%，或没有降低延迟/改善质量，就删除，不进入主线。

## 5. 实验矩阵与判定门槛

所有实验从同一 checkpoint、相同 seed、相同数据顺序开始；DiT-MoE、Memory、detector 和采样流程固定，唯一变量是输出头。

### 5.1 第一阶段：证明 MoE 是否必要

| ID | 输出头 | 目的 |
|---|---|---|
| D0 | 原始标准线性 head | 最简 dense 下界 |
| D1 | shared linear + 单个 dense residual MLP | 排除“只是参数变多”的解释 |
| L0 | 旧 E8 Top-2 point-wise hard-φ | 旧设计控制组 |
| M1 | shared linear + E4 Top-2 contour prototype，真稀疏 | 推荐现代候选 |
| M1-K1 | M1 改为 Top-1 | 判断第二个激活专家是否值得 |

先并行训练 300 step 做淘汰；只有优于 D1 且趋势稳定者进入 1000-step 确认。两天预算内不做 10 万 step。

### 5.2 第二阶段：只验证有依据的增量

仅当 M1 通过第一阶段时，增加 M2：`4 heavy + zero + linear`。不同时加入 DirMoE、gate residual、point segment、class-conditioned router 等变量。

### 5.3 评估指标

质量：

- GT-box、memory-off 的 volume Dice / IoU / foreground slice Dice；
- GT-box、autoregressive 的同组指标，确认输出头没有掩盖 Memory 问题；
- NSD / HD95 与逐椎体结果；
- 固定 seed 配对差异与 bootstrap CI；
- 全量可视化只写入 `visual/`。

路由：

- hard Top-1/Top-2 load、CV、死亡专家；
- expert pairwise output disagreement；
- 每个样本强制经过各专家的 velocity loss；
- selected expert loss、oracle best expert loss、router regret；
- expert advantage 与 diffusion time / scale /类别的关系。

真正的专门化判据不是负载均匀，而是“router 选中的专家在该样本上确实比其他专家误差低”。

效率：

- 完整模型参数、activated parameters；
- batch1 / batch8 三体积实测吞吐、显存；
- profiler 确认未选专家没有执行。

### 5.4 主线准入

输出 MoE 只有同时满足下列条件才保留：

1. 相对 D1 的 Dice 提升达到至少 +0.002，或配对置信区间明确排除 0；
2. NSD/HD95 与可视化不恶化；
3. 相对 D1 的 batch8 延迟不超过 10%，目标不超过 5%；
4. 参数增长不超过 2.5%；
5. 无死亡专家，且 selected-vs-oracle 指标证明存在有效路由对齐。

若 M1 不能明显超过 D1，则删除输出头 MoE，只保留已经有证据的 DiT 内部 MoE。这会是更简洁、也更容易向审稿人解释的结果。

## 6. 待讨论的决策

建议批准第一阶段五组短实验，其中 D0/D1 是必须补齐的因果对照。当前不建议直接把 MMOE 全套、DirMoE 或 Routing-Free MoE写入主线。

如果方向确认，实施顺序应为：先实现 D1 和 M1 的统一接口与真稀疏 dispatch，完成数值/梯度/未选专家不执行测试，再利用空闲 GPU 并行跑 300-step screening。
