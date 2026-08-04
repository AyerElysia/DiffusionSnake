# MemFlowDiT 近期工作综合报告

## 演化失败的数据根因、MoE 关键性消融与 2026 路由优化

日期：2026-07-31  
项目：`DiffusionSnake-12-30`  
状态：数据主线已修正，v0.5 正式训练持续运行；DiT-MoE、输出头去退化及二者联合候选已完成阶段验证

---

## 一、结论先行

近期工作中最重要的发现，不是“模型训练得还不够久”，而是旧 MemFlowDiT 训练目标本身存在严重的数据工程损失。

step 3250 的全量可视化表现出轮廓残缺、系统性欠分割和局部形态不可信。随后在同一 checkpoint 上使用 GT box 并关闭时序记忆，结果与自回归评估几乎相同，因此检测器和记忆传播都不是当时质量差的主因。进一步审计发现：

1. 旧数据管线对每个类别只保留最大连通域，完整数据前景像素只保留 **81.88%**；
2. 旧的输出空间多边形面积阈值会把小椎体实例继续过滤掉，在审计样本中曾造成 **209 个前景验证切片中的 26 个切片丢失全部实例**；
3. 旧轮廓点在 MoonViT 特征图上使用零填充采样，靠近边界时可能把有效视觉特征替换成零；
4. 旧记忆证据只有单通道，训练又主要依赖 teacher-forced GT evidence，与实际自回归推理的数据分布不一致。

其中第一项是最关键的数据根因。旧 checkpoint 对两个诊断体积的预测前景分别只有 GT 的约 **84.5%** 和 **81.7%**，与“训练目标只保留 81.88% 前景”高度吻合。换言之，模型很可能不是完全没有学会演化，而是在学习一个已经被数据管线裁残的目标。

修复后的主线保留每类最多 4 个显著连通域、面积下限 2、单切片全局最多 32 个实例，训练集和验证集的前景保留率分别提高到 **99.52%** 和 **99.73%**；同时降低小实例过滤阈值，改用 border 特征采样，并使用 26 通道类别感知记忆证据。考虑到设计简洁性，最终 v0.5 又删掉了法向细节采样、曲率重加权和未经充分证明的 layer 18+26 双层特征，只保留 MoonViT layer 18 与必要的数据、采样和记忆修复。

第二个关键发现来自输出头消融：在同一个已加载 checkpoint 上关闭 routed experts、只保留 shared baseline 后，单体积 GT-box Dice 从 **0.770740** 降到 **0.641614**，绝对下降 **0.129126**。这说明旧输出头里的 routed expert contribution 不是装饰性模块，而是当前演化映射的重要组成部分。这次消融直接促使我们把 MoE 从“历史遗留组件”提升为需要独立研究的核心课题，并进一步探索在 DiT 内部以 MoE 替换 FFN。

围绕 MoE，近期又完成了两项改进：

- DiT 内部采用简洁的 4 专家 Top-1、奇数层 FFN-MoE；以整条轮廓为路由单位，使用余弦 prototype、真实数据原型初始化与 population-level φ-balancing。1000-step、3 体积 GT/memory-off Dice 达到 **0.793873**，全程无死专家。
- 旧输出头存在 soft entropy 很高、hard Top-1 却集中到专家 2 和 5 的“伪健康”现象。类别不均衡确实存在，但条件统计不支持“专家按椎体类别各司其职”解释当前集中。真正的直接缺陷是旧 hard-load 项由离散 Top-K 计数得到，不能给 router 传梯度。最小 hard-φ 修复把训练期 hard CV 从 **0.9139** 降到 **0.5698**，三体积固定 seed 的体积 Dice 基本持平，说明路由退化已显著缓解且没有破坏主任务。

当前判断是：v0.5 继续作为正在运行的数据修正版正式训练；下一次独立长程候选采用 v0.6，将 E4K1 DiT-MoE 与输出头 hard-φ 合并。不能在运行中的 v0.5 checkpoint 中途热切换结构。

---

## 二、问题是怎样暴露出来的

### 2.1 可视化首先否定了“只看均值指标”

触发本轮深入诊断的是：

`visual/memflowdit/step_003200_gt_autoregressive/viz/viz_sub-verse010_slice0076.png`

这并不是孤立坏例。全量可视化中反复出现以下现象：

- 轮廓只覆盖主体的一部分，附属或分离区域缺失；
- 预测面积系统性小于完整 GT；
- 部分类别在连续切片中重复欠分割；
- 自回归传播没有把错误显著修好；
- 数值均值尚可，但形态质量不能接受。

这说明必须把问题拆成检测、初始化、演化网络、记忆传播、训练目标和评估协议六条链分别检查，不能仅凭“训练只有 3200 step、以往需要 10 万 step”就把问题归因于训练量。

### 2.2 先排除检测器

step 3250 使用 GT box 评估，检测器不会决定实例框和类别，因此该结果已经排除了检测漏检、错框和分类错误对轮廓演化的影响。

两个完整体积上的结果为：

| 设置 | 体积 Dice | 体积 IoU | 前景切片 Dice |
|---|---:|---:|---:|
| GT box，autoregressive memory | 0.722084 | 0.565416 | 0.713752 |
| GT box，memory-off | 0.722742 | 0.566219 | 0.715033 |
| 自回归相对 memory-off | -0.000658 | -0.000803 | -0.001281 |

关闭记忆后只改善了约 0.00066 Dice。由此可以确认：

- 检测器不是这批差可视化的原因；
- 自回归记忆当时没有带来收益，但也不足以解释主要失败；
- 主要问题已经存在于“GT 框初始化 → 单帧特征 → Flow/DiT 演化”主路径或其监督数据中。

### 2.3 欠分割比例指向训练目标

同一 GT-box、memory-off 诊断中：

| 体积 | GT 前景体素 | 预测前景体素 | 预测/GT |
|---|---:|---:|---:|
| `sub-verse010` | 259,791 | 219,538 | 84.5% |
| `sub-verse011` | 186,432 | 152,204 | 81.7% |

这不是随机噪声式失败，而是明显的系统性前景缺失。随后进行全数据目标构造审计，发现旧的 largest-only 规则只保留 81.88% 前景。预测/GT 比例与监督保留率处在同一范围，构成了“现象—数据代码—全量统计”三方面互相支持的证据链。

需要保持严谨：这是一条很强的根因证据，但不是严格的单变量因果实验。最终性能是否完全恢复，仍必须由修复后长程 checkpoint 的配对评估确认。

---

## 三、数据工程根因与修复

### 3.1 根因一：每类只保留最大连通域

旧 `_mask_to_instances` 对每个椎体类别调用 `findContours` 后，只选择面积最大的轮廓。这个假设对“一个类别永远对应一个连通多边形”的普通实例任务可能成立，但不适合当前矢状位椎体掩码：

- 同一椎体在切片上可能出现多个彼此分离的有效区域；
- 单个 polygon 无法表达不连通区域；
- largest-only 会把其余真实区域直接从训练目标中删除；
- 评估仍使用完整标签，因此训练目标和评估目标不一致。

全数据审计结果：

| 目标生成规则 | 前景像素保留率 |
|---|---:|
| 旧 largest-only | 81.88% |
| 新 significant components，训练集 | 99.52% |
| 新 significant components，验证集 | 99.73% |

新的规则为：

- 每类最多保留 4 个显著连通域；
- 原图连通域面积至少为 2；
- 每个连通域作为同一类别的独立实例轮廓；
- 每张切片全局最多 32 个实例；
- 超出上限时按面积优先保留。

全局 32 实例上限不是额外模型机制，而是必要的数据安全边界，用于避免噪声掩码重新触发历史上单切片 51 个轮廓造成的显存问题。

### 3.2 根因二：小实例被二次过滤

父类 `get_valid_polys` 在 1/4 输出空间使用固定 `Polygon.area > 5`。矢状位边缘切片上的小椎体在原图中是真实前景，但缩放到输出空间后可能小于该阈值。

审计中发现，209 个前景验证切片里有 26 个会因此丢失全部实例。修复后将阈值做成可配置项，当前主线使用：

`min_poly_area_output: 0.5`

这一步避免了“掩码里有前景，但送进演化监督后变成空样本”的隐性数据损失。

### 3.3 根因三：MoonViT 边界采样不适配旧零填充

旧 v0.3 使用：

`gcn_sample_mode: half_pixel`  
`gcn_sample_padding_mode: zeros`

MoonViT 是 patch token 特征，经重排、投影和上采样后再由轮廓点采样。轮廓点靠近特征图边缘或数值上轻微越界时，zero padding 会突然返回零向量。对旧 CNN 特征而言这已经不理想，对归一化后的 MoonViT 表征更容易形成分布断点。

修复后，特征重采样和 GCN 点采样统一使用 `border`：

- `locate_feat_resample_padding_mode: border`
- `gcn_sample_padding_mode: border`

这项修复的定位是“消除不合理的边界特征断点”，不能单独宣称带来多少 Dice；它与完整目标修复一起构成新的数据—特征契约。

### 3.4 记忆证据从二值占用改为类别感知

旧 v0.3 记忆只使用 1 个 mask channel。对 25 类椎体而言，单通道只能表示“这里有前景”，不能可靠保留椎体身份。训练时又主要写入 GT evidence，推理时写入 previous prediction，存在 teacher forcing 偏差。

v0.4/v0.5 改为：

- 26 通道类别感知 mask evidence；
- 训练早期使用稳定 GT evidence；
- 从 step 500 开始逐渐加入预测 evidence；
- 4500 step 线性升至最高 50%；
- 推理仍使用前序预测；
- 记忆容量为 4，保存的是压缩后的最近状态，不是所有历史原图或所有历史 token。

因此，预测当前帧时并不是无损利用“历史全部信息”，而是利用当前帧 MoonViT 特征与最近 4 个压缩记忆槽；更早信息只有在已经被前序状态间接传递时才可能留下影响。

---

## 四、从 v0.4 回到简洁的 v0.5

第一版修复 v0.4 同时加入了完整连通域、border sampling、26 通道记忆、预测证据调度、MoonViT layer 18+26、法向细节上下文和曲率重加权。它能验证修复链可运行，但混入了过多机制，难以归因。

根据“影响不明确的模块不要留在主线”的原则，v0.5 做了以下收缩：

- 保留完整显著连通域目标；
- 保留小实例阈值修复；
- 保留 half-pixel + border 采样；
- 保留 26 通道类别感知记忆和预测证据调度；
- MoonViT 只保留 layer 18；
- 删除 layer 26 拼接；
- 删除法向细节采样；
- 删除曲率重加权；
- 不增加新的轮廓细节分支。

从同一基础 checkpoint 进行 2-step 适配烟测后，单体积 GT-box、自回归结果为：

| 配置 | 体积 Dice | 体积 IoU | 前景切片 Dice |
|---|---:|---:|---:|
| v0.4：18+26 + detail/curvature | 0.708858 | 0.549017 | 0.598701 |
| v0.5：layer 18 minimal | 0.769071 | 0.624789 | 0.770316 |
| 差值 | +0.060213 | +0.075772 | +0.171614 |

这不是严格单变量消融，因为特征层和细节机制同时变化；它能支持“简化没有破坏主路径，并显著恢复已加载基线能力”，不能单独证明全部增益来自 layer 18。

v0.5 已作为正式训练重新启动。启动时曾设置 100,000 step，但该上限按最新训练预算要求已经取消：当前运行采用“双硬限制”，在 **step 6800** 或 **2026-08-02 04:00（Asia/Shanghai）** 二者先到时停止，总训练墙钟时间不超过两天。以后启动脚本的默认上限也已改为 6800。2026-07-31 约 09:52 首次核对时：

- 进程：PID `1572713`
- 物理 GPU：6
- 已运行到 step 990
- 最新常规 checkpoint：step 900
- 最近 100 step 平均总 loss：0.004705
- 最近 100 step 平均 diffusion loss：0.005020
- 最近 100 step 平均 memory read delta：0.000182
- 训练中无 NaN、无崩溃

上述两天限制由独立 watchdog 约束当前既有进程，因此不需要重启训练，也不会破坏现有优化器连续状态。达到 step 上限时会等待对应的整百 checkpoint 落盘再停止；若先达到墙钟期限，则直接停止并保留此前最近的整百 checkpoint。

本轮随后在 step 1600 完成了新的固定 seed、GT-box、memory-off 与 autoregressive 配对评估。它比早期烟测可靠，但仍只有 3 个体积，不能据此宣布最终视觉质量已经解决。

### 4.1 step 1600 最新落盘评估

2026-07-31 对正式 v0.5 最新不可变 checkpoint `step_001600.pt` 做了固定 `seed=20260731`、GT box、3 个完整体积、333 张切片的配对评估。两组唯一变量为 memory mode。

| 设置 | 体积 Dice | 体积 IoU | 前景切片 Dice | class mean Dice |
|---|---:|---:|---:|---:|
| memory-off | **0.792168** | **0.656264** | 0.765609 | 0.747620 |
| autoregressive | 0.792125 | 0.656225 | **0.765794** | **0.747883** |
| autoregressive - off | -0.000043 | -0.000039 | +0.000185 | +0.000263 |

自回归平均 memory read delta 为 `0.0000965`，说明记忆分支确实执行了读取；但体积主指标变化只有 `-0.000043`，当前仍不能认为记忆产生了可重复净收益。

memory-off 的逐体积 Dice 为：

- `sub-verse010`：0.791492；
- `sub-verse011`：0.815618；
- `sub-verse013`：0.769394。

同为 GT box、memory-off 的前两个体积上，旧 v0.3 step 3250 Dice 为 0.722742；当前 v0.5 step 1600 对应两体积均值为 0.803555，绝对提高 0.080813。这一结果强烈支持完整目标和简洁主线修复的有效性，但仍不是只改变单一数据变量的严格消融。

新的误差形态也发生了变化。memory-off 三个体积的预测/GT 前景比为 1.087、1.035、1.078；旧 v0.3 的对应诊断主要是 0.82–0.85 的系统性欠分割。说明 largest-only 数据损失已经不再主导结果，但当前出现了约 3.5%–8.7% 的轻度前景过量，需要在后续全量可视化中重点检查边界外扩，而不是继续沿用“缺失连通域”这一旧解释。

结果位置：

- `data/outputs/volmem/verse_memflowdit_v0_5_minimal_gpu6/eval_step_001600_gt_off_3vol_seed20260731/summary.json`
- `data/outputs/volmem/verse_memflowdit_v0_5_minimal_gpu6/eval_step_001600_gt_autoregressive_3vol_seed20260731/summary.json`

---

## 五、关键消融：MoE 不是可有可无

### 5.1 消融设计

旧 V4.6c 输出头由 shared baseline 与 routed experts 共同产生位移。为了判断 MoE 是否只是复杂但作用很小的历史设计，在同一个已加载 checkpoint、同一个体积、GT box 和相同 Euler-20 推理下，将：

`v4_6_moe_routed_expert_scale: 1.0`

改为：

`v4_6_moe_routed_expert_scale: 0.0`

这样保留 shared baseline，只移除 routed expert contribution，不改变前面的 DiT、MoonViT、检测框和输入数据。

### 5.2 结果

| 输出头设置 | 体积 Dice | 体积 IoU | 前景切片 Dice |
|---|---:|---:|---:|
| 完整输出 MoE | 0.770740 | 0.626995 | 0.769855 |
| 仅 shared baseline | 0.641614 | 0.472336 | 0.496088 |
| 关闭 routed experts 的损失 | -0.129126 | -0.154659 | -0.273767 |

这是近期最重要的结构消融之一。它说明：

- 当前 checkpoint 的关键演化能力大量存储在 routed experts 中；
- shared baseline 不能独立承担完整输出映射；
- MoE 对当前网络不是可删除的轻微修饰；
- 后续不能只把 MoE 看成“最后一层实现细节”，必须监控它是否真正专家化、是否退化、是否可稳定训练。

另做过 `standard` 输出头替换，Dice 为 0.228537。但该头无法从旧 MoE checkpoint 完整继承对应权重，因此不是公平的架构消融，只能说明“不能直接换成未适配的 dense head”，不把它作为 MoE 优越性的正式证据。

### 5.3 对研究路线的影响

这次消融改变了后续优先级：

1. 先研究现有输出 MoE 的真实路由健康，而不是默认其工作正常；
2. 参考 2026 年 MoE 研究，重新设计更适合高度相关轮廓 token 的路由；
3. 探索把 DiT block 内部 FFN 替换成 MoE；
4. 同时坚持结构简洁，不引入法向细节采样、复杂阶段调度或无法归因的附加分支。

---

## 六、DiT 内部 FFN-MoE：从失败路由到可用候选

### 6.1 旧思路为什么不直接照搬

项目旧输出 MoE 带有 DeepSeek-MoE 风格的多专家、Top-K、shared expert、点位 embedding、cyclic router 和 router noise。对语言 token 适用的路由假设并不一定适合轮廓：

- 同一轮廓的 128 个点高度相关；
- 相邻点不是 128 个独立语义样本；
- batch 较小且序列相关，单 batch 均衡统计噪声大；
- MoonViT 提供的是高维视觉语义，旧式点序号和法向细节路由可能反而主导捷径；
- 专家数过多会使每个专家获得的数据不足。

结合 2026 年 φ-balancing、prototype routing 和视觉 DiT-MoE 的研究，本轮选择重做一个最小方案，而不是继续叠加旧机制。

### 6.2 最终最小设计

当前 DiT-MoE 候选为：

- 6 层 DiT 只替换第 1、3、5 层 FFN；
- 每层 4 个 SwiGLU 专家；
- Top-1，每条轮廓激活 1 个专家，激活比例 25%；
- 以整条轮廓的 pooled descriptor 路由，不逐点独立路由；
- LayerNorm 后与可学习 prototype 做余弦相似度；
- 首批真实数据使用 farthest-point + spherical k-means 初始化 prototype；
- 使用 population-level φ-balancing；
- 使用轻量 routing contrastive guidance；
- 不保留 shared FFN；
- 不使用点序号 embedding、cyclic router、router noise；
- 不使用法向/曲率细节采样；
- 旧 dense FFN 只在加载 checkpoint 时一次性复制到所有专家，模型内不保留 inactive dense FFN。

该设计只有一个核心问题：同一 DiT 层的轮廓级语义是否需要不同 FFN 专家处理。

### 6.3 失败试验提供的结论

第一版“单点 token + 随机 prototype”很快退化。第 10 step：

| 设置 | soft entropy | hard CV | 最多死亡专家数 |
|---|---:|---:|---:|
| E4 K1 | 0.9823 | 1.4071 | 3 |
| E8 K1 | 0.9839 | 1.3727 | 4 |
| E8 K2 | 0.9839 | 1.0647 | 3 |

soft entropy 看似很好，但 hard 选择已经死亡。改成整轮廓路由后，如果仍使用随机 prototype，第 3 层约 99.98% 路由仍落到同一专家。最终只有“整轮廓路由 + 真实数据 prototype 初始化”一起使用，退化才得到修复。

E4K1 在 step 50：

- 平均 soft entropy：0.9991
- 平均 hard CV：0.3504
- 三个 MoE 层死亡专家：0

这说明本任务里路由粒度和初始化都不是次要实现细节。

### 6.4 专家数量和激活比

同训练预算、同一基础 checkpoint、同 seed、50 step、单体积 GT/memory-off：

| 设置 | 体积 Dice | IoU | 前景切片 Dice | 平均 hard CV |
|---|---:|---:|---:|---:|
| 无 DiT-MoE | 0.788544 | 0.650906 | 0.776869 | — |
| E4 K1 | 0.790186 | 0.653147 | **0.780562** | 0.5252 |
| E8 K1 | **0.790228** | **0.653203** | 0.777384 | 0.5000 |
| E8 K2 | 0.788583 | 0.650960 | 0.774369 | **0.3985** |

E8K2 最均衡，却没有更好的分割；E8K1 相对 E4K1 只高 0.000042。于是选择更简单的 E4K1：

- 参数和显存更小；
- 每个专家获得更多数据；
- 当前数据量不足以证明 8 专家有必要；
- Top-2 多激活一倍专家，也没有转化为任务收益。

### 6.5 1000-step 长程结果

E4K1 独立训练到 1000 step 后：

- 三层 hard CV：0.4089、0.2204、0.7471；
- 三层死亡专家数均为 0；
- 3 体积 GT/memory-off：Dice 0.793873，IoU 0.658593，前景切片 Dice 0.765275；
- 3 体积 GT/autoregressive：Dice 0.790851，IoU 0.654528，前景切片 Dice 0.762436；
- autoregressive 相对 memory-off 低 0.003023；
- 同一 `sub-verse010` 上，1000-step memory-off Dice 比 50-step 高 0.003412。

结论是：E4K1 至少在 1000 step 内没有路由死亡，并保持了小幅正向训练趋势；但记忆路径尚未形成净收益，不能把 MoE 与记忆问题继续捆绑增加复杂度。

---

## 七、新问题：输出头 MoE 的 hard 激活不均衡

### 7.1 现象

对正式 v0.5 输出头记录了真实 hard Top-1/Top-K。step 100、一个完整体积、GT box、memory-off 共得到 9,139,200 次路由决策：

| 专家 | Top-1 占比 |
|---|---:|
| 0 | 0.72% |
| 1 | 0.74% |
| 2 | 62.89% |
| 3 | 4.03% |
| 4 | 1.61% |
| 5 | 26.05% |
| 6 | 1.88% |
| 7 | 2.09% |

专家 2 和 5 合计承担约 88.94% 的 Top-1 决策。与此同时 soft normalized entropy 仍有 0.97934。旧监控因此给出“概率分布很健康”的错觉，而实际离散路由已经高度集中。

以后 MoE 健康监控不能只看 soft entropy，必须同时记录：

- soft load；
- hard Top-K load；
- Top-1 load；
- hard-load CV；
- 低于 1% 的专家数量；
- 分层调用次数和 token/contour 数。

### 7.2 类别不均衡假设是否成立

“不同输出专家分别擅长不同椎体，而类别数据不均衡导致专家负载不均衡”是合理假设，因此没有直接把负载集中等同于退化。

全训练集统计首先确认数据确实不均衡：

- 80 个病例；
- 14,384 张训练切片；
- 单类前景像素最大/最小约 25.8 倍；
- 单类出现切片数最大/最小约 24.9 倍；
- 病例覆盖数从 3 到 64。

随后在 step 400 做条件路由诊断，记录 10,200 个前向事件和 71,400 条轮廓事件，类别匹配错误为 0。结果为：

- 全局 Top-1：`[0.0090, 0.0096, 0.6240, 0.0447, 0.0219, 0.2475, 0.0206, 0.0227]`；
- 已观察的 5 个类别都约 62% 使用专家 2、约 25% 使用专家 5，没有类别间的专家换挡；
- 类别均衡反事实负载与原负载 L1 距离只有 0.000116；
- 类别—专家归一化互信息 0.000087；
- 时间—专家互信息 0.014106；
- 尺度—专家互信息 0.007215；
- 点位—专家互信息 0.000577。

因此：

- “训练数据类别不均衡”这个前提是真的；
- “当前两个主导专家是按类别各司其职”不符合现有证据；
- 时间和轮廓尺度有少量关系，但远不足以解释 87% 以上的集中；
- 当前不应贸然加入 class-conditioned router，否则会增加复杂度并进一步切碎低频类别数据。

### 7.3 直接根因：hard-load 项没有路由梯度

旧平衡损失同时计算 soft importance 与 hard Top-K load，但 hard load 来自离散 `topk/scatter/count`。该 hard 分支只能产生一个数值，不能对 router logits 反向传播。

因此训练会出现：

- soft probability 被约束得较平；
- Top-1 的微小排序偏差长期积累；
- 少数专家持续赢得 hard 选择；
- 日志中的 hard-load 惩罚看似存在，实际上不能直接纠正 hard 拥塞。

这比“类别不均衡”更直接地解释了 soft entropy 高、hard routing 集中的矛盾。

---

## 八、输出头去退化：最小 hard-φ 修复

### 8.1 已完成的修复

不改变现有 8 个输出专家、Top-2、shared baseline 和 checkpoint 参数，只替换平衡机制：

1. 用真实 hard Top-K 选择更新专家负载 EMA；
2. 根据 EMA 计算不反传的拥塞价格；
3. 让当前 soft probability 承载梯度；
4. 过载专家收到降低 logits 的方向，闲置专家收到提高 logits 的方向；
5. 默认仍保留 `legacy` 模式保证旧实验兼容，候选配置显式启用 `hard_phi`。

没有增加类别路由、法向细节采样、新输出专家结构或多阶段调度。单元测试已经验证梯度方向正确。

### 8.2 100-step 隔离对照

两组从同一 checkpoint、同一 seed 和同一数据顺序训练 100 step，DiT 内部 MoE 关闭，唯一变量是输出头平衡方式。

| 设置 | hard CV | 专家 2 | 专家 5 | 其余 6 专家合计 |
|---|---:|---:|---:|---:|
| legacy | 0.9139 | 59.52% | 28.07% | 12.41% |
| hard-φ | **0.5698** | 40.49% | 21.74% | **37.77%** |

hard-φ 明显降低了专家垄断，且没有把路由强行压成完全均匀。

固定 `seed=20260731`、GT box、memory-off、3 体积：

| 设置 | 体积 Dice | IoU | 前景切片 Dice | 全切片 Dice |
|---|---:|---:|---:|---:|
| legacy | 0.788267 | 0.650934 | **0.762288** | **0.875076** |
| hard-φ | **0.788280** | **0.650954** | 0.760950 | 0.874373 |
| 差值 | +0.000012 | +0.000020 | -0.001338 | -0.000703 |

正确结论不是“hard-φ 已提升分割”，而是：

- 路由健康显著改善；
- 体积主指标基本严格持平；
- 切片均值有轻微下降；
- 修复解决的是可训练性和长期退化风险，性能收益需要更长训练验证。

类别 8–24 的单类 Dice 变化范围为 -0.00169 到 +0.00337，且与类别频率没有单调关系，再次不支持按类别专门化解释旧塌缩。

### 8.3 与 DiT-MoE 联合

v0.6 联合候选同时使用：

- DiT 奇数层 E4K1 prototype MoE；
- 旧输出头 hard-φ；
- v0.5 minimal 数据和 MoonViT 主路径。

50-step 训练中四个 MoE 模块均无死亡专家。固定 seed、单体积 GT/memory-off：

| 设置 | 体积 Dice | IoU | 前景切片 Dice | class mean Dice |
|---|---:|---:|---:|---:|
| E4K1 + legacy 输出头 | 0.788461 | 0.650793 | 0.777263 | 0.758233 |
| E4K1 + hard-φ 输出头 | **0.788794** | **0.651247** | **0.777433** | **0.758972** |

输出头校正没有扰乱 DiT 三层 prototype 路由，所有主指标均有极小正变化。这足以证明两项机制可以整合进同一个候选配置，但不足以宣布联合模型最终优于 v0.5。

---

## 九、主线决策与下一步

### 9.1 当前主线

在两天硬预算内继续运行：

`configs/volmem/verse_memflowdit_v0_5_minimal_gpu6.yaml`

原因：

- 它已经包含最重要的数据目标、边界采样和记忆证据修复；
- 正式训练已经启动，不应中途改变参数结构和优化器状态；
- 当前首要任务仍是验证“正确数据目标 + 简洁 MoonViT 主路径”经过充分训练后的真实上限。
- 当前运行不再追求 100,000 step；有效上限是 step 6800 或 2026-08-02 04:00 二者先到。

### 9.2 下一次独立候选

候选配置：

`configs/volmem/verse_memflowdit_v0_6_moe2026_combined_gpu7.yaml`

它整合：

- v0.5 minimal；
- 奇数层 E4K1 DiT FFN-MoE；
- 输出头 hard-φ 去退化。

### 9.3 必须完成的验证

1. 在 v0.5 足够训练后进行固定 seed 的 GT-box 评估；
2. 同 checkpoint 比较 memory-off 与 autoregressive，单独判断记忆净收益；
3. 保存全量可视化到 `visual/memflowdit/...`，继续把形态质量作为硬门槛；
4. v0.6 独立完成至少 1000-step 联合训练；
5. 对 v0.5/v0.6 使用同体积、同 seed、同推理步数的配对比较；
6. 长程监控输出头和 DiT 各层 hard Top-1/Top-K、CV 与死专家数；
7. 只有出现稳定多体积、多 seed 增益后，才讨论扩大专家数或增加类别条件。

### 9.4 当前明确不做

- 不恢复法向细节采样；
- 不恢复曲率重加权；
- 不因为类别不均衡就直接增加 class router；
- 不增加复杂专家调度；
- 不追求“绝对均匀”的专家负载；
- 不停止或热切换正在运行的 v0.5；
- 不把短程烟测写成最终性能结论。

---

## 十、关键代码、配置与结果

### 数据与正式训练

- `lib/datasets/sagittal_2d_fixed/snake.py`
- `lib/networks/snake/ct_snake.py`
- `tools/volmem/train_memflowdit.py`
- `configs/volmem/verse_memflowdit_v0_5_minimal_gpu6.yaml`
- `data/outputs/volmem/verse_memflowdit_v0_5_minimal_gpu6/`

### 演化失败诊断

- `data/outputs/volmem/verse_memflowdit_v0_3_2day_gpu6/eval_step_3250_viz/summary.json`
- `data/outputs/volmem/diagnostics/step_003250_gt_off_2vol/summary.json`
- `visual/memflowdit/step_003200_gt_autoregressive/`

### MoE 重要性消融

- `configs/volmem/diagnostics/v05_head_ablate_moe.yaml`
- `configs/volmem/diagnostics/v05_head_ablate_base.yaml`
- `data/outputs/volmem/diagnostics/v05_head_moe_euler20_loaded_1vol_20260731/summary.json`
- `data/outputs/volmem/diagnostics/v05_head_base_euler20_loaded_1vol_20260731/summary.json`

### 新 DiT-MoE

- `lib/networks/diffusion/prototype_phi_moe.py`
- `lib/networks/diffusion/dit_blocks_v3.py`
- `lib/networks/diffusion/dit_denoiser_v3.py`
- `configs/volmem/verse_memflowdit_moe2026_dataproto_phi_e4k1_odd_gpu0.yaml`
- `data/outputs/volmem/verse_memflowdit_moe2026_dataproto_e4k1_odd_long1000_gpu1/`

### 输出头诊断与修复

- `lib/networks/diffusion/dit_denoiser_v4.py`
- `lib/networks/diffusion/dit_denoiser_v4_1.py`
- `lib/networks/diffusion/flow_matching_evolution.py`
- `data/outputs/volmem/diagnostics/moe_output_head_conditional_step400_gt_off_1vol_v3/summary.json`
- `configs/volmem/verse_memflowdit_outputmoe_step100_control_gpu7.yaml`
- `configs/volmem/verse_memflowdit_outputmoe_step100_hardphi_gpu7.yaml`
- `tests/test_moe_final_head_conditional_diagnostics.py`

### 联合主线候选

- `configs/volmem/verse_memflowdit_v0_6_moe2026_combined_gpu7.yaml`
- `data/outputs/volmem/verse_memflowdit_v0_6_moe2026_combined_smoke50_gpu7/`

### MoE 专项报告

- `docs/report/MOE_2026_RESEARCH_20260731.md`

---

## 十一、最终认识

这轮工作的核心不是给网络继续叠模块，而是重新建立可信的因果顺序：

1. 先通过可视化发现均值指标掩盖了形态失败；
2. 用 GT box 和 memory-off 排除检测器与记忆；
3. 从系统性欠分割追到 largest-only 和小实例过滤的数据目标损失；
4. 修正数据契约，再删掉法向细节等无充分证据的冗余设计；
5. 用隔离消融确认 routed MoE 对当前输出映射至关重要；
6. 研究适合轮廓和 MoonViT 表征的新 DiT-MoE；
7. 用 hard 条件统计发现旧输出头的“soft 健康、hard 退化”；
8. 验证类别不均衡不是当前退化的主要解释；
9. 用最小 hard-φ 修复路由梯度，而不是增加更复杂的类别专家结构；
10. 将两项 MoE 改进整合为 v0.6 候选，同时保护正在运行的 v0.5 正式训练。

因此，当前最重要的技术结论可以概括为：

> 之前差的演化效果既有训练量不足的问题，但更先验、更严重的是训练目标被数据管线系统性裁残；MoE 则已被消融证明是当前输出映射的重要组成部分，但旧路由监控和均衡损失不足以防止 hard 激活退化。新的主线必须同时保证数据目标完整、结构简洁，并用真实 hard 路由统计约束 MoE 的长期可训练性。

---

## 十二、3D Memory 与并行推理补充（2026-07-31 晚）

完整设计、实现、运行记录和逐项指标见：

- `docs/report/MEMFLOWDIT_PARALLEL_3D_MEMORY_EXPERIMENT_20260731.md`

严格同噪声、GT-box、固定 3 个 validation volumes 的配对结果显示：旧 v0.5 step2300 的 parallel-off Volume Dice 为 `0.796574`，feature-only、GT-oracle、predicted bidirectional Memory 分别为 `0.796354`、`0.796343`、`0.796363`，三种 Memory 均无收益，且 read delta 都约为 `1.089e-4`。旧自回归相对顺序 off 下降 `0.000276` Dice，吞吐下降 17.7%。因此当前问题不是 bank 太小，也不能主要归因于预测误差累积；更直接的原因是 26 通道稀疏 mask 在 1152 通道 MoonViT 特征中被数值淹没，随后 Memory residual 又被近零输出投影二次压低。

已实现参数无关的 causal/bidirectional/shuffled 选择、冻结两遍预测、feature-only 一遍双向推理和按 volume 同噪声评估。并行关闭 Memory 的吞吐达到 `1.789 slice/s`，约为顺序 off 的 4.5 倍，证明“整卷批量并行”本身值得保留；但 predicted two-pass 只有 `0.859 slice/s` 且无精度收益，因此不进入主线。

新的 v0.7 仅把 feature 与 mask 分开投影、对 mask 单独归一化并固定比例相加，没有增加 router、长期 bank 或法向细节分支。首次 run 在 step9 发现长训练遗留的 4500-step 预测证据爬升不适合本轮 2000-step/两天上限，已完整归档并将正式日程修正为 step250 开始、750 steps 爬升、step1000 后保持 50% predicted evidence。v0.5/v0.6 原训练未受影响。后续只有在 v0.7 的 causal GT-oracle 明确转正后，才允许继续扫描 bank 大小与更复杂的选择策略。

2026-08-01 补充：v0.7 step500 的 3-volume、GT-box、同 seed 因果门槛已经完成。off Volume Dice 为 `0.790857`，causal GT-oracle 为 `0.790106`，差值 `-0.000750`；三个 volume 的差值均非正，吞吐下降 7.9%。因此 balanced entrance 虽把活跃区间的 Memory read delta 提高到约 `2.7e-4` 至 `3.1e-4`，更新方向仍然有害，step500 不能称为有效 Memory。训练暂时继续到预测证据达到 50% 的 step1000 再做最终 gate；若仍不转正，停止 v0.7 并淘汰该方案。完整故障与实验记录仍以专项报告第 8.6 节为准。

2026-08-01 Bank 复核补充：用户关于 K4 历史范围过窄的怀疑部分成立。三个验证体积的矢状位间距分别约为 1/2/3 mm，最近 K4 实际只覆盖 3/6/9 mm；更关键的是当前实现只传入 slice index，完全丢失了毫米间距，使相同 index 距离在不同体积中被错误地视为同一种三维几何关系。论文审计也不支持把全部原始历史无限平铺：SAM 2 使用受限空间 Memory 与间隔旧帧，RMem 直接报告扩张 bank 会被冗余和混淆特征拖累，XMem 则把长期历史压缩保存。现已增加“固定 K4、保留最近层并间隔采样旧层”的诊断模式，并用同 checkpoint 同噪声对照 recent K4、strided K4、K8/K16 与单体积 all-history；正式候选优先考虑物理毫米坐标下的有界多尺度历史，而不是无上限原始 bank。完整数据审计、token 预算和预设判据见 Memory 专项报告第 10 节。

2026-08-01 进一步结论：同一 v0.7 step500、sub-verse010、GT-box、同噪声配对已经排除“只因 bank 太小”。off 为 `0.791592`；GT recent K4/K8/K16 分别为 `0.791451/0.789947/0.788471`，差值随容量扩大到 `-0.003120`；固定 token 但扩大跨度的 strided K4 为 `0.789143`；all-history 为 `0.790383`，前景 slice Dice 下降 `0.009363`、吞吐下降 28.6%，且误差随历史长度增大。feature-only K4 比 off 低 `0.001551`，加入 GT mask 后恢复 `0.001411`，说明 mask evidence 已有益，真正有害的是历史 MoonViT feature/距离内容路径。

据此已实现并启动简洁 v0.8：MoonViT 只作 key，class-aware mask 只作 value，距离只进 key，位置改为 NIfTI 毫米坐标，bank 有界为 K7；不增加 selector、双 bank、router 或法向分支。全量物理清单覆盖 160 volumes/26,589 slices。14 项单测与最终 smoke 通过；正式训练从 v0.5 step2300 初始化并将 Memory read 清零，以严格 2D identity 起步。GPU5 train PID `2745777`、watchdog PID `2745778`，最晚 `2026-08-03T07:00:00+08:00` 或 step2000 停止；启动后 step1–8 数值稳定，read delta 由 0 平滑打开到约 `5e-5`–`6e-5`。v0.7 GPU1 训练未停止，检查时到 step800。完整表格、失败 smoke 原因、论文依据和运行路径见 Memory 专项报告第 10.4–10.7 节。

v0.8 step100 的 1-volume、GT-box、同噪声 off/oracle K7 门槛已由不可变守护脚本启动等待，PID `2748662`，输出目录为 `data/outputs/volmem/diagnostics/memflowdit_v08_step000100_gt_gate/`；脚本启动后不再覆盖原文件。

---

## 十三、v0.8 最终结论与 v0.9 compact 3D 尝试（2026-08-02）

v0.7/v0.8 均已自然训练到 step2000。v0.8 的最终判断不再依赖早期 step100：对 step1900 和 step2000 分别做同 checkpoint、同噪声、GT-box 的 off/oracle K7 门控。Volume Dice 差值仅为 `+0.000402/+0.000111`；step2000 的前景 slice Dice 反而 `-0.002234`、class Dice `-0.001027`，吞吐下降 19.4%。它没有达到预设的 `+0.001`、前景同向、减速不超过 10% 门槛，因此 v0.8 不再续训，也不能称为有效 3D Memory。

这次失败进一步确认：真实毫米位置和 evidence-only value 修复了数据语义，却不足以让“多张局部历史图”自动升级为有效三维状态。继续增大 raw bank 已被 K8/K16/all-history 反证，下一步应压缩旧历史，而不是保存更多原始 token。

v0.9 因而只增加一个参数无关的 compact 摘要：保留最近 K4 的 256 个局部 token，把全部更老且非空的 mask evidence 在线平均成 4×4 的 16 个全局 token，总预算固定为 272。全局 token 不写入伪造 slice distance；空切片由 valid mask 屏蔽；没有 selector、双 bank attention、额外 router、法向细节或几何分支。零训练探针中 K4+G16 的 Volume Δ 为 `+0.000180`，略优于 K7+G16 的 `+0.000160`，且 token 更少，因此正式配置选择 K4+G16。旧权重下前景指标仍为负，这一探针只用于选容量，不作为有效性结论。

17 项 Memory 单测和 2-step smoke 已通过。正式训练从 v0.8 step2000 的 `461/461` 完整权重继续适配，GPU6 PID `3403812`，watchdog PID `3403813`，最晚 step2000 或 `2026-08-03T23:50:00+08:00` 停止。首步 loss `0.003758`、read delta `0.000191`、显存 `12.98GB`，8 个并行 bank 均实际读到 `4 local + 1 global`。

step100 已挂三路守护评估（PID `3404945`）：

1. Memory off；
2. oracle local K4；
3. oracle local K4 + global16。

输出目录为 `data/outputs/volmem/diagnostics/memflowdit_v09_step000100_compact_gate/`。只有第三路相对 off 达到 `+0.001`、前景同向，并且相对第二路存在可复现额外增益，才认为 compact global token 值得继续；否则停止继续堆叠长期模块。

---

## 十四、v0.9 step100 门控：compact global 尚未生效（2026-08-02）

三路同 checkpoint、GT-box、sub-verse010、同 seed 的评估已经完成：off / local K4 / local K4+global16 的 Volume Dice 分别为 `0.794863 / 0.793886 / 0.793874`。local K4 相对 off 为 `-0.000977`；加入 global16 后相对 off 为 `-0.000989`，相对 local K4 仅 `-0.0000119`。compact 的前景 slice Dice 相对 off 为 `-0.001912`，吞吐下降 18.2%。三项预设门槛全部未通过。

这意味着目前不能保证 Memory 有效：局部历史仍在轻微伤害结果，16 个全局 token 也没有提供额外净收益。global state、非空 valid mask 和独立 token 路径均已实际运行，因此不能把零增益归因于“代码没走到”；更可能是简单均值摘要没有保留可判别结构，或 controller 尚未学会条件性使用历史。

考虑到新增路径只适配了 100 steps，v0.9 暂时不停，但设置了明确的最终止损点：只训练到 step500 再做固定 3-volume 的 off / K4 / K4+G16 复核。watcher PID 为 `3422258`，结果目录为 `data/outputs/volmem/diagnostics/memflowdit_v09_step000500_compact_gate_3vol/`。若仍未达到 `Volume +0.001、前景同向、减速不超过 10%`，立即停止 v0.9，并淘汰当前 compact 在线均值方案，不再通过增加 bank 或 token 数量掩盖失败。

---

## 十五、v0.9 最终止损与 MoE 成本审计（2026-08-02）

v0.9 step500 的 3-volume 门控已经完成：off / local K4 / K4+G16 的 Volume Dice 分别为 `0.789150 / 0.789112 / 0.789166`。compact 相对 off 仅 `+0.000016`，相对 local 仅 `+0.000054`；虽然前景 slice Dice 为 `+0.001310`，吞吐仍下降 18.09%，没有达到 Volume `+0.001` 与减速不超过 10% 的门槛。自动止损守护在结果落盘后核对 PID 与配置并发送 SIGTERM，训练于 step608 停止。当前 compact 在线均值方案正式淘汰，Memory 不能声称有效。

另完成独立 MoE 成本审计。当前联合候选是输出头 8 专家 Top-2，加奇数 3 层 E4 Top-1，并非 6 层全 MoE。相对 output-MoE dense DiT，总参数增长 12.00%，训练峰值显存仅增长 0.61%，参数成本可接受；但 batch1 新增减速为 7.79%，batch8 新增减速为 15.68%。旧输出头还会先计算全部 8 个专家再 Top-2 gather，是并行扩展的主要冗余。完整表格、代码根因和准入决策见 `docs/report/MOE_COST_AUDIT_20260802.md`。
