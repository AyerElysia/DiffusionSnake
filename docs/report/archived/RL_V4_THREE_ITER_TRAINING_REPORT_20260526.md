# RL V4 三步 outer trajectory 训练报告

日期：2026-05-26

## 结论

V4 的核心不是把 FM / ODE 的每个去噪子步当作策略轨迹，而是把**三次外层 refinement**当作策略轨迹：

1. 第 1 次 outer step
2. 第 2 次 outer step
3. 第 3 次 outer step

每一步都对应一个连续动作：**当前轮廓状态下的位移 displacement**。  
随机 latent / flow noise 只负责探索，不直接作为策略更新对象。

当前实现文件：

- `grpo_train_v4_three_iter.py`
- `lib/networks/diffusion/flow_matching_evolution.py`
- `configs/btcv_select_v4_6c_rl_v4_three_iter_long_gpu5.yaml`

---

## 1. V4 到底在训练什么

V4 训练的是一个**三步连续动作策略**。  
策略的状态不是 ODE 中间步，而是每个 outer refinement 时刻的 contour state：

- `i_it_py`：当前轮廓
- `c_it_py`：canonical 轮廓
- `cnn_feature`：YOLO / CNN 特征
- `py_ind`：轮廓所属实例索引
- `fraction`：当前 outer step 的阶段比例

每个 outer step 产生一个动作：

$$
a_t \in \mathbb{R}^{N \\times 2}
$$

表示所有轮廓点的二维位移。

---

## 2. 一次训练迭代的完整流程

### 2.1 构造上下文

先用 `manual_context` 走和推理一致的路径：

- YOLO 提特征
- `cnn_proj` / `cnn_proj_p3`
- 取初始轮廓 `i_it_py`
- 取 GT 轮廓 `i_gt_py`
- 构造 `py_ind`

这一步的目标是：**训练时的状态定义必须和推理时一致**。

### 2.2 先算 deterministic baseline

对同一个样本，先跑一条**确定性的三步 refinement**：

- latent 置零
- outer fraction 依次取 `[0.3333, 0.5, 1.0]`
- 得到 baseline final contour

baseline 只用于算 reward 的参照，不直接更新权重。

### 2.3 采样 K 条 rollout

对每个样本采样 `K` 条轨迹。  
每条轨迹都走三次 outer step：

1. 当前轮廓 `state`
2. 变到 canonical space
3. 采一个随机 latent
4. 跑一次 FM rollout
5. 得到当前 outer action `action`
6. 更新轮廓

这里的随机 latent 是探索来源。

### 2.4 计算 reward

每条 rollout 的最终轮廓和 GT 比较，使用现有 region reward：

- region
- dice
- iou
- distance / boundary

最终 reward 不是单步 reward，而是**整条三步轨迹的终局质量**。

### 2.5 用 PPO 更新三步动作

每条 rollout 记录：

- `state`
- `c_state`
- `action`
- `old_logprob`
- `fraction`

然后在 PPO inner epochs 里重算当前策略下的 `new_logprob`，再做 ratio 更新。

---

## 3. 动作概率函数怎么产生

这是 V4 最关键的地方。

### 3.1 先说结论

V4 的动作概率不是离散策略，也不是 FM 子步概率。  
它是一个**高斯连续动作分布**：

$$
\\pi_\\theta(a \\mid s) = \\mathcal{N}(a; \\mu_\\theta(s), \\sigma^2 I)
$$

其中：

- `s` = 当前 outer state
- `a` = 当前 outer displacement
- `\\mu_\\theta(s)` = 当前 state 下的**确定性三步外层位移**
- `\\sigma` = 固定动作标准差

### 3.2 mean 怎么来

mean 不是网络单独输出一个 policy head。  
而是：

1. 把 latent 设成 0
2. 用当前 state 跑一次 FM / flow rollout
3. 得到 deterministic displacement
4. 再乘当前 outer step 的 fraction

代码对应：

- `_flow_disp_from_latent(...)`
- `_outer_action_mean(...)`

因此：

$$
\\mu_\\theta(s) = f_{flow}(s, z=0) \\cdot fraction
$$

也就是说，**动作均值就是“零噪声版本的外层 refinement 位移”**。

### 3.3 action 怎么来

采样时不是用 mean 直接走，而是：

1. 采一个随机 latent `z ~ N(0, I)`
2. 跑 flow 得到 `raw_disp`
3. 再乘 `fraction`
4. 得到 sampled action

所以探索来自 latent noise，但最终被 RL 约束的对象仍然是 outer displacement。

### 3.4 std 怎么定

配置里是：

- `rl_v4_action_std_px = 1.5`

然后除以 `snake_config.down_ratio` 转成模型坐标系：

$$
\\sigma = \\frac{action\\_std\\_px}{down\\_ratio}
$$

### 3.5 logprob 怎么算

`_action_logprob(action, mean, std)` 的实现是逐点高斯对数概率：

$$
\\log p(a \\mid s) = \\mathbb{E}_{points}\\left[
-\\frac{(a - \\mu)^2}{2\\sigma^2} - \\log \\sigma - \\frac{1}{2}\\log(2\\pi)
\\right]
$$

实际代码里是对轮廓点和坐标维度求平均。  
也就是把每个 contour point 的 2D 位移都看成独立高斯变量。

这意味着：**动作概率函数本质上是对连续位移场的 Gaussian policy**。

---

## 4. 旧 logprob 和新 logprob 怎么用

### 4.1 采样时存 old_logprob

采样 rollout 时，先用当前策略算：

- `mean`
- `action`
- `old_logprob = log π_old(action | state)`

这个 `old_logprob` 存下来，后面 PPO 用。

### 4.2 更新时重算 current logprob

在 PPO inner epoch 中，对同一个 state / action：

- 重算 `mean_cur`
- 算 `lp_cur = log π_new(action | state)`
- 得到 ratio：

$$
r = \\exp(lp_{cur} - lp_{old})
$$

然后做 clipped PPO loss：

$$
L = -\\min(rA, clip(r, 1-\\epsilon, 1+\\epsilon)A)
$$

再加一个对 frozen reference flow 的 KL 约束。

---

## 5. baseline 和 gate

V4 不直接拿 sampled rollout 和 GT 比就更新，而是先和 deterministic baseline 比。

定义：

$$
quality = score(sample) - score(baseline)
$$

如果 best rollout 没有超过 `gate_margin`，该 batch 的更新会被门控压掉。  
这能减少明显负样本把策略拉偏。

---

## 6. 为什么 V4 和以前不一样

### 以前更像什么

以前很多 RL / GRPO 方案，实际优化对象会混进：

- ODE 子步
- 蒸馏目标
- latent ranker
- best-of-k 选择

这会造成 reward 来源和 policy logprob 不一致。

### V4 现在的约束

V4 明确把学习目标收束到：

- **outer 3-step displacement**
- **Gaussian action logprob**
- **PPO ratio 更新**

所以它是一个真正意义上的**三步连续策略**，而不是 FM 子步轨迹搜索器。

---

## 7. 当前实现要点

- 训练入口：`grpo_train_v4_three_iter.py`
- 基线 checkpoint：`data/outputs/btcv_diffusion_dit_v4_6c_mlp_shared_moe_newdist_long_gpu5/checkpoints/latest.pt`
- 训练数据：`/home/medteam/Zhrch/Datasets/BTCV/btcv_select/train`
- 验证数据：`/home/medteam/Zhrch/Datasets/BTCV/btcv_select/test`
- outer steps：`3`
- fractions：`[0.3333, 0.5, 1.0]`
- action 分布：Gaussian continuous policy

---

## 8. 一句话总结

V4 的训练对象是：  
**“三次 outer refinement 产生的连续轮廓位移”**。

V4 的动作概率函数是：  
**“以 deterministic outer refinement 为均值、以固定高斯标准差为噪声的连续 Gaussian policy”**。

