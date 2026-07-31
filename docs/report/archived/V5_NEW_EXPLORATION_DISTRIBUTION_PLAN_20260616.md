# V5 新探索分布计划：从手写噪声走向数据驱动 proposal

日期：2026-06-16  
项目：DiffusionSnake / V5 geom post-training  
目标：寻找一个比当前 Fourier / geom action 更有效、更符合轮廓误差结构的 RL 探索分布。

## 1. 背景和核心问题

我们当前的后训练问题已经不是“RL optimizer 怎么调得更好”，而是：

**模型在接近收敛的轮廓附近，应该沿什么样的候选变形方向探索，才可能稳定采到比 deterministic baseline 更好的轮廓。**

现有 V5 geom/Fourier 探索有一个明确优点：它能采到正收益候选。近期 5-step geom 对照中：

- step 1-50：`quality_best_mean = +0.00886`
- step 51-92：`quality_best_mean = +0.00834`
- 局部最大候选收益可到 `+0.032 ~ +0.036`
- step 50 fixed eval：`IoU = 0.856250`, `mBF = 0.790849`

但它也有明显局限：

- 它仍然是人为设定的低频几何扰动；
- 模式表达能力有限；
- 不一定匹配真实 GT residual 的统计结构；
- 对“哪些 contour 该动、往哪里动、动多少”没有直接建模。

因此我们希望寻找一个新的探索分布，不一定用 Fourier，而是把轮廓残差建模成一种更合适的结构化分布。

## 2. 已尝试的特殊分布：Structured Mixture Laplace

我们已经在 `grpo_train_v5_geom_action.py` 中实现并测试了一个初版特殊分布：

**per-contour structured mixture of Laplace normal residuals**

主要设计：

- 每条 contour 采样一个 mode，而不是每个点独立采样；
- `M=4` 个 mixture mode；
- 每个 mode 输出沿法线方向的 per-point residual；
- 分布参数包括：
  - mode logits；
  - per-point loc；
  - per-point logscale；
  - contour-level gate；
- PPO 更新时重新计算同一个 mode 的 logprob；
- checkpoint 中保存 `mixture_explorer_state_dict`；
- 额外记录：
  - `mixture_entropy`
  - `mixture_prob_max_mean`
  - `mixture_scale_px`
  - `mixture_gate_mean`
  - `mixture_loc_abs_px`
  - `mixture_freeze_flow`

使用配置：

- config：`configs/1232_final_v5_structured_mixture_5step_from3500_gpu0.yaml`
- output：`data/outputs/1232_final_v5_structured_mixture_5step_unfreeze_gpu0`
- 起点 checkpoint：epoch 3500
- action policy：`structured_mixture`
- steps：`[4]`
- modes：`4`
- `mixture_loc_scale_px = 0.35`
- `mixture_init_scale_px = 0.12`
- `mixture_lr = 5e-5`
- `freeze_flow = false`

## 3. 当前实验结果：初版失败

截至 step 140：

- step 1 fixed eval：`IoU = 0.855673`, `mBF = 0.790613`
- step 50 fixed eval：`IoU = 0.855614`, `mBF = 0.790558`
- step 100 fixed eval：`IoU = 0.855354`, `mBF = 0.790073`

探索候选质量：

- step 1-50：`quality_best_mean = -0.02059`
- step 51-100：`quality_best_mean = -0.01848`
- step 101-140：`quality_best_mean = -0.01710`
- 到 step 140 为止，最大 `quality_best_mean` 也只有 `-0.00305`

对比普通 V5 geom：

- geom step 1-50：`quality_best_mean = +0.00886`
- geom step 51-92：`quality_best_mean = +0.00834`
- geom step 50 fixed eval：`IoU = 0.856250`, `mBF = 0.790849`

结论：

**当前 structured mixture Laplace 版本可训练、稳定、能反传，但探索候选系统性差于 deterministic baseline，也明显差于普通 V5 geom。**

它的问题不是工程不可行，而是分布参数化不对：它采样的是“形式上结构化”的 normal residual，但不一定是训练集 GT 证明过有用的残差方向。

## 4. 对失败原因的初步判断

### 4.1 从零学习 proposal 太弱

RL reward 是稀疏的，当前 mixture 分布又从接近零均值的小尺度 Laplace 开始学。它需要同时学：

- 哪些 contour 需要动；
- 应该选哪个 mode；
- 每个点沿法线该移动多少；
- scale 应该放大还是收缩；
- 如何避免 burr/detail 惩罚。

这些目标全靠 PPO reward 推，信号太弱。

### 4.2 分布“结构化”，但没有“数据驱动”

Laplace normal residual 看起来比独立高斯更合理，但它并没有利用 GT residual 的真实统计。

真实有用的残差可能不是简单的局部法线扰动，而可能包含：

- 局部成段收缩/外扩；
- 尾部/尖端模式；
- 初始化定位误差；
- contour-level 平移/缩放；
- 低频形状变化和高频边界修正的组合；
- 多器官/多形态下的 mode 分工。

当前分布没有显式编码这些模式。

### 4.3 当前尺度可能仍偏大

`mixture_init_scale_px = 0.12` 表面看不大，但在边界已接近收敛时，随机法线扰动容易制造 burr/detail 惩罚。

观察中：

- reward 总体为负；
- `quality_best_mean` 长期为负；
- detail/burr penalty 明显吃掉候选收益。

所以小尺度版仍值得作为 sanity check，但它不是根本解。

### 4.4 与 geom 对照不完全公平

当前 mixture 配置使用了更强的 detail/regression 相关项，而普通 geom 对照的 detail/regression 配置更轻。因此初版结论应表述为：

**当前 mixture 配置明显不如 geom，但还需要做 reward 对齐的小尺度 sanity check，确认不是配置差异造成的。**

不过，即使考虑配置差异，140 step 无一次正收益候选仍然是强负信号。

## 5. 我们真正想要的新探索分布

理想 proposal 不应该只是“一个漂亮的概率分布”，而应该满足：

1. **正收益密度高**  
   采样出来的候选中，至少一部分应经常优于 deterministic baseline。

2. **有轮廓级结构**  
   不能每个点独立乱动，应该有 segment / contour / organ-level 相关性。

3. **能表达多模式**  
   同一个图像/轮廓附近可能存在多个合理修正方向，单均值不够。

4. **能学会少动或不动**  
   对已经好的 init/final contour，proposal 应该自动收缩。

5. **与 GT residual 统计一致**  
   最重要：探索方向应来自训练集上真实 residual，而不是纯手写噪声。

6. **PPO logprob 可计算**  
   如果继续用 policy gradient，分布必须能稳定计算 sample logprob。

7. **可与 supervised/WTA 结合**  
   最好能先用 GT 监督预训练 proposal，再用 RL 小幅 fine-tune。

## 6. 建议的新路线：GT Residual Prototype Mixture

我建议下一代探索分布从 GT residual 出发，而不是从零学习 Laplace noise。

### 6.1 离线统计真实 residual

对训练集运行当前 V5 deterministic 推理，得到：

```text
pred_contour = model(image, init)
gt_contour   = ground truth contour
residual     = gt_contour - pred_contour
```

然后把 residual 投影到局部坐标：

```text
normal_residual  = dot(residual, normal)
tangent_residual = dot(residual, tangent)
```

优先只建模 normal residual，因为 tangent 方向容易造成点序错位和边界毛刺。

需要统计：

- residual 幅度分布；
- residual 的频谱/平滑度，但不一定用于 Fourier；
- 成段相关长度；
- 不同器官/不同 contour 质量下的 residual 模式；
- init 好/坏时 residual 是否不同；
- tail / endpoint / high-curvature 区域的残差是否有特殊模式。

### 6.2 学 residual prototypes

从真实 residual 中聚类出 K 个 prototype，例如：

```text
prototype_k: [N_points] normal residual pattern
```

候选方法：

- k-means / spherical k-means on normalized residual curves；
- PCA basis + mixture；
- VQ codebook；
- small autoencoder latent codebook；
- 按器官类别或 contour difficulty 分组聚类；
- 按局部段而不是整条 contour 聚类。

### 6.3 RL 采样时使用 prototype mixture

新的 proposal 可以是：

```text
mode k ~ Categorical(pi(image, contour))
alpha  ~ Normal(mu_alpha, sigma_alpha)
eps    ~ small smooth noise

delta_normal = gate * (alpha * prototype_k + eps)
new_contour  = contour + delta_normal * normal
```

其中：

- `pi` 由网络预测，表示当前 contour 适合哪个 prototype；
- `alpha` 控制幅度和方向；
- `gate` 控制要不要动；
- `eps` 只做小扰动，不负责主探索方向。

这个设计的核心优点：

**主方向来自 GT residual 统计，因此候选更可能落在真实修正方向附近。**

### 6.4 可监督预训练

对每个训练样本，可以用 GT residual 找最近 prototype：

```text
k* = argmin_k || residual - alpha_k * prototype_k ||
```

然后监督训练：

- mode classification：预测 `k*`
- alpha regression：预测最佳缩放系数
- gate regression：根据 residual 大小预测是否该动
- optional WTA loss：多个 hypothesis 中最近 GT 的那个接收梯度

这样 proposal 在进入 RL 前就已经知道真实误差模式，不需要 PPO 从零发现。

## 7. 候选分布设计

### 7.1 Prototype Mixture Distribution

形式：

```text
p(delta | image, contour)
  = sum_k pi_k * p(alpha | k) * p(eps)

delta = gate * (alpha * prototype_k + eps)
```

优点：

- 数据驱动；
- logprob 相对容易算；
- mode 有可解释性；
- 不依赖 Fourier；
- 可以监督预训练。

风险：

- 整条 contour prototype 可能太粗；
- 器官间差异大时需要 conditional prototype；
- residual 对点对应关系敏感。

### 7.2 Segment Prototype Mixture

不是整条 contour 一个 prototype，而是把 contour 分成若干 segment：

```text
delta_segment_j = alpha_j * prototype_{k_j}
```

优点：

- 更适合局部错误；
- 可以处理尾部/局部边界问题；
- 比整条 contour 更灵活。

风险：

- segment 边界可能不连续；
- logprob 和实现更复杂；
- 需要平滑约束。

### 7.3 Low-rank Residual Distribution

不用 Fourier，而是从真实 residual 学一个低秩 basis：

```text
delta = B z
z ~ mixture / Gaussian / Laplace
```

其中 `B` 来自 PCA、autoencoder decoder 或 learned dictionary。

优点：

- 比 Fourier 更贴近数据；
- 比 prototype 更连续；
- 可控维度。

风险：

- 单 Gaussian latent 可能又回到均值问题；
- 需要 mixture latent 才能表达多模式。

### 7.4 Energy-shaped Proposal

先采样多个候选，再用一个轻量 quality model 或 GT 监督的 surrogate score 重加权：

```text
q(delta) proportional to base_distribution(delta) * exp(score(delta) / tau)
```

优点：

- 可以把 learned ranker/quality head 融入 proposal；
- 有机会提高正收益候选密度。

风险：

- ranker 之前在相似 seed 候选上失败过；
- 如果候选本身缺少多样性，reweight 没用；
- 实现和稳定性更复杂。

### 7.5 Normalizing Flow Residual Distribution

用 conditional normalizing flow 建模：

```text
residual ~ p_theta(residual | image, contour)
```

优点：

- 表达力强；
- logprob 精确；
- 可以直接最大似然训练 GT residual。

风险：

- 工程复杂；
- 数据量可能不足；
- 容易过拟合；
- 采样是否产生正收益候选需要验证。

## 8. 推荐执行计划

### Phase 0：停掉当前失败配置

当前 `structured_mixture` 已经足够形成负结论：

- step 140 仍无正收益候选；
- eval 没涨；
- 比 geom 对照明显差。

建议停止当前 run，保留 step50/step100/step140 日志和 checkpoint。

### Phase 1：公平小尺度 sanity check

目的：排除“只是尺度/配置不公平”的可能。

修改：

- `mixture_init_scale_px: 0.12 -> 0.06` 或 `0.08`
- reward 权重对齐普通 V5 geom
- 只跑 100 step

判据：

- 若 `quality_best_mean` 仍长期 < 0，停止 Laplace mixture 方向；
- 若能转正，再考虑继续调尺度和 mode。

### Phase 2：GT residual 统计

写离线脚本：

```text
scripts/analyze_v5_gt_residual_distribution.py
```

输出：

- `data/analysis/v5_residual_stats/*.npz`
- residual 幅度直方图；
- normal/tangent residual 比例；
- curvature/tail 区域 residual；
- residual 与 init IoU/final IoU 的关系；
- 聚类可视化。

核心问题：

**真实 GT residual 到底像什么？**

在看清这个之前，不应继续拍脑袋设计探索噪声。

### Phase 3：Prototype Mixture proposal

基于 Phase 2 的 residual prototypes，实现：

- prototype codebook；
- contour-level mode predictor；
- alpha/gate head；
- small smooth noise；
- logprob；
- PPO 接入。

先监督预训练 proposal：

- mode classification；
- alpha regression；
- gate regression；
- WTA residual loss。

再用 RL 做小步 fine-tune。

### Phase 4：与 geom/Fourier 做固定协议对照

所有 proposal 用同一协议比较：

- 同一 epoch3500 起点；
- 同一 fixed eval set；
- 同一 reward 权重；
- 同一 steps；
- 同一 noise/eval 配置；
- 记录 `quality_best_mean`、`reward_mean`、`eval_iou`、`mBF`、burr/detail。

硬性早停规则：

- 100 step 内 `quality_best_mean` 均值不能转正，杀；
- fixed eval 比 baseline 掉超过 `0.0005 IoU`，杀；
- burr/detail penalty 明显高于 geom，杀或降尺度。

## 9. 想请其他 AI 重点评审的问题

我们希望外部 AI 不要泛泛讨论 RL，而是重点回答以下问题。

### Q1：GT residual prototype mixture 是否是合理 proposal？

核心假设：

**用训练集真实 residual 聚类出的 prototype，比手写 Fourier/geom 或从零学习 Laplace 更适合作为轮廓探索方向。**

请评估：

- 这个假设是否成立；
- 哪些情况下会失败；
- 是否会过拟合训练 residual；
- 是否适合医学轮廓/多器官场景。

### Q2：prototype 应该是整条 contour、局部 segment，还是 latent basis？

三种候选：

1. whole-contour prototype；
2. segment-level prototype；
3. PCA/autoencoder low-rank latent basis。

请评估哪种更适合：

- 512 点轮廓；
- 尾部/局部边界错误；
- init/locate 错误；
- 多器官形态差异。

### Q3：如何保证 proposal 可用于 PPO？

我们需要 sample logprob。请评估以下形式是否合适：

```text
k ~ Categorical(pi)
alpha ~ Normal(mu, sigma)
eps ~ Normal(0, sigma_eps^2 S)
delta = gate * (alpha * prototype_k + eps)
```

问题：

- logprob 应该只算 `k, alpha, eps`，还是需要考虑 `delta` 的变换 Jacobian？
- 如果 `eps` 是低维平滑噪声，如何定义稳定 logprob？
- gate 是 deterministic 时是否影响 PPO？

### Q4：是否应该先监督训练 proposal，再 RL？

我们倾向于：

```text
GT residual -> assign nearest prototype -> train mode/alpha/gate -> RL fine-tune
```

请评估：

- 是否比直接 PPO 更合理；
- supervised objective 应该怎么设计；
- WTA loss 是否必要；
- 如何避免 proposal 只复制 GT residual 而不适应模型动态误差。

### Q5：怎样处理“少动/不动”？

当前后训练的一个核心问题是：好 init 经常被改坏。

新 proposal 需要能表达：

```text
delta = 0
```

或非常小的修正。

请评估：

- 是否应该把 zero-action 作为一个显式 mode；
- gate 应该监督训练还是 RL 学；
- gate target 应该来自 residual norm、IoU improvement，还是 init-vs-GT 距离。

### Q6：是否还有比 prototype mixture 更好的特殊分布？

约束：

- 不想继续依赖 Fourier；
- 需要 contour-level structured exploration；
- 最好能利用 GT；
- 最好能算 logprob；
- 实现复杂度不能过高。

请提出替代方案，并说明它们相对 prototype mixture 的优缺点。

## 10. 当前建议结论

我的当前判断：

1. **探索仍然可能革新，但不能再靠手写漂亮分布。**

2. **当前 structured Laplace mixture 是负结果。**  
   它证明了“特殊分布可接入、可训练”，但也证明了“从零学 proposal + 随机法线 residual”不够。

3. **下一步应该把 GT residual 统计写进 proposal。**  
   也就是从“探索噪声”转向“数据驱动候选修正方向”。

4. **最有价值的下一版是 Residual Prototype Mixture。**  
   它兼顾结构化、多模式、可监督预训练和 PPO logprob。

5. **在做大改前，先跑一个小尺度公平对照。**  
   如果 `scale=0.06/0.08 + reward对齐` 仍不能让 `quality_best_mean` 转正，就应彻底停止 Laplace mixture，直接进入 GT residual prototype。

