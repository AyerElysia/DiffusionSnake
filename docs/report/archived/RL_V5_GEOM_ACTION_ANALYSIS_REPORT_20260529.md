# RL V5 几何动作强化学习分析报告

日期：2026-05-29

## 1. 结论先行

当前仓库里的 **V5 强化学习**，不是早期 `V5.0 / V5.1 / V5.2` 那条 **SAM 初始化链路**，而是另一条更晚的 **RL V5 几何动作策略**：

- 训练脚本：`grpo_train_v5_geom_action.py`
- 主配置：`configs/btcv_select_v4_6c_rl_v5_geom_action_gpu3.yaml`
- 保守版配置：`configs/btcv_select_v4_6c_rl_v5b_geom_action_small_gpu7.yaml`

它的核心不是把随机性放在 inner ODE / SDE 每一个小 step 上，而是把 **3 次 outer refinement** 当作策略轨迹；每一步动作都被限制成：

> **沿当前轮廓法线方向、由低频傅里叶基控制的几何位移**

这是 V5 最关键的设计点。它的目标很明确：

1. 保留 V4.6c / FM 主干已经学到的“平均修轮廓能力”；
2. 只让 RL 在一个更低维、更可控的几何子空间里做微调；
3. 用更精确的 action log-prob，避免之前“高维动作 + 近似 logprob”带来的不稳定。

一句话概括：

> **V5 = 在 V4.6c 的 deterministic 三段细化均值上，叠加一个低频法向几何策略，并用 PPO/GRPO 风格更新这个几何策略。**

---

## 2. 这条 V5 对应仓库里的哪几部分

### 2.1 训练入口

- `grpo_train_v5_geom_action.py:1-24`
- `scripts/run_v5_geom_action_gpu3.sh`
- `scripts/run_v5b_geom_action_small_gpu7.sh`

这两个 shell 脚本都先：

1. `cd /home/medteam/Zhrch/DiffusionSnake-12-30`
2. `conda activate snake1`
3. `export CFG_FILE=...`
4. `python grpo_train_v5_geom_action.py --cfg_file "$CFG_FILE"`

这和你当前主线里“**先设置 CFG_FILE，再 import 项目模块**”的约束是对齐的。

### 2.2 基座模型

V5 不是从头训新模型，而是先加载一个已经训好的 V4.6c / FM 基座：

- `resume_path: data/outputs/btcv_diffusion_dit_v4_6c_mlp_shared_moe_newdist_long_gpu5/checkpoints/latest.pt`

见：

- `configs/btcv_select_v4_6c_rl_v5_geom_action_gpu3.yaml:10-12`
- `configs/btcv_select_v4_6c_rl_v5b_geom_action_small_gpu7.yaml:10-12`
- `grpo_train_v5_geom_action.py:397-409`

所以这条 V5 的真实含义是：

> **在 V4.6c 的 flow-matching contour refiner 上做 RL 后训练，而不是重写整个 contour model。**

### 2.3 数据范围

两份配置都只跑 BTCV select 子集：

- 训练集：`/home/medteam/Zhrch/Datasets/BTCV/btcv_select/train`
- 测试集：`/home/medteam/Zhrch/Datasets/BTCV/btcv_select/test`

见配置：

- `configs/btcv_select_v4_6c_rl_v5_geom_action_gpu3.yaml:16-20, 37-40`
- `configs/btcv_select_v4_6c_rl_v5b_geom_action_small_gpu7.yaml:16-20, 37-40`

所以它不是 BTCV 全量 full-train 的 RL 报告，而是 **select split 上的 RL 后训练实验**。

---

## 3. V5 整体训练流程是怎么跑的

下面按脚本真实执行顺序拆。

### 3.1 先强制打开 diffusion / flow-matching / GRPO 相关开关

见：

- `grpo_train_v5_geom_action.py:323-338`

脚本开头直接设：

- `cfg.use_diffusion_evolution = True`
- `cfg.use_diffusion_trainer = True`
- `cfg.use_flow_matching = True`
- `cfg.use_grpo = True`

然后通过 `cv(...)` 从配置里读超参。注意这里虽然叫 V5，**配置命名空间仍然沿用了 `rl_v4_*`**，例如：

- `rl_v4_train_steps`
- `rl_v4_k`
- `rl_v4_geom_lowfreq_modes`
- `rl_v4_reward_*`

这说明：

> **V5 是方法升级，但配置命名还没彻底重构。**

### 3.2 加载基座 checkpoint，只训练 flow/gcn 侧

见：

- `grpo_train_v5_geom_action.py:397-427`

关键动作：

1. `make_network(cfg)` / `make_trainer(cfg, network)`
2. 从 `resume_path` 读取预训练权重
3. 检查 load ratio，低于 `min_load_ratio` 就直接报错
4. 如果 `freeze_yolo=true`，冻结：
   - `yolo`
   - `cnn_proj`
   - `cnn_proj_p3`
   - `swin_snake_feature`
5. 保持 `inner.gcn` 可训练
6. 冻结 BN running stats
7. 构造一个 `ref_flow = freeze_ref_flow(inner)` 作为 KL 参照模型

这意味着：

> **V5 并不想重学 detector，也不想改 feature encoder；它只想在 contour evolution / flow policy 这一层做受限微调。**

### 3.3 先固定一小批 eval batch

见：

- `grpo_train_v5_geom_action.py:429-440`

脚本会从 val loader 里先抓固定数量的 batch（默认 `rl_v4_eval_batches: 8`），后续训练过程中反复在这批固定样本上评估 deterministic policy。

这点很重要：

> 日志里的 `eval_iou / eval_dice / eval_mboundf` 不是全量正式评估，而是 **固定 8 个 val batch 的 smoke eval**。

---

## 4. V5 的 state、action、policy mean 是怎么定义的

这是整条方法最核心的部分。

### 4.1 state：当前 contour 状态 + CNN contour feature

见：

- `grpo_train_v5_geom_action.py:486-524`
- `lib/networks/diffusion/flow_matching_evolution.py:771-795`

`_manual_context(batch)` 会：

1. 跑 detector / feature extraction，拿到 `cnn_feature`
2. 取出当前初始化轮廓 `i_it_py`
3. 取出 GT 轮廓 `i_gt_py`
4. 对 GT 做方向和起点对齐 `_align_gt(...)`
5. 构造 `py_ind`

最终得到：

- `cnn_feature`
- `i_it_py`
- `c_it_py`
- `i_gt_py`
- `py_ind`
- `image_hw`

也就是说 V5 的 policy state 本质上是：

> **当前轮廓位置 + 该轮廓对应的 CNN / detail feature 上下文**

### 4.2 deterministic mean：三段 outer refinement 的 FM 均值动作

见：

- `grpo_train_v5_geom_action.py:283-316`
- `grpo_train_v5_geom_action.py:542-553`
- `lib/networks/diffusion/flow_matching_evolution.py:991-1065`

这里有两层：

#### 第一层：单步 outer action 的均值

```python
latent = torch.zeros_like(i_state)
disp = flow rollout from latent=0
action_mean = disp * frac
```

也就是：

1. 从零 latent 出发；
2. 用当前 FM backbone 在 `ode_steps=20` 的 inner rollout 下，算出这一步该往哪里修；
3. 再乘这一段的 outer fraction（默认 `[0.3333, 0.5, 1.0]`）。

#### 第二层：三段 deterministic rollout

```python
for frac in fractions:
    action = outer_action_mean(...)
    current = current + action
```

所以 deterministic baseline 不是一步，而是：

> **3 段连续 refinement 后的最终轮廓**

这个 deterministic rollout 一方面被拿来算 baseline，另一方面也被拿来做训练中的 eval。

---

## 5. V5 的随机动作到底是什么

这是 V5 和之前 RL 版本差异最大的地方。

### 5.1 动作不再是整张高维位移场，而是“低频法向几何扰动”

见：

- `grpo_train_v5_geom_action.py:245-280`
- `grpo_train_v5_geom_action.py:556-586`

V5 定义了几件事：

1. `_contour_normals(poly)`：求每个 contour 点的单位法线；
2. `_lowfreq_basis(n_points, n_modes)`：构造低频傅里叶基；
3. `_geom_delta_from_z(poly, z, sigma)`：把低维噪声 `z` 投影成沿法线方向的几何位移；
4. `_project_geom_z(...)`：反向把动作残差投回低频基系数；
5. `_geom_action_logprob(...)`：在这个低维 `z` 空间里算高斯 log-prob。

实际采样过程是：

```python
mean = outer_action_mean(...)
z ~ N(0, I), shape=(B, geom_modes)
action = mean + normal_direction_lowfreq_delta(z, sigma)
```

默认超参：

- `geom_lowfreq_modes = 8`
- `outer_steps = 3`
- aggressive V5：`geom_sigma_px = [1.2, 0.8, 0.5]`
- conservative V5b：`geom_sigma_px = [0.6, 0.4, 0.25]`

见：

- `configs/btcv_select_v4_6c_rl_v5_geom_action_gpu3.yaml:131-158`
- `configs/btcv_select_v4_6c_rl_v5b_geom_action_small_gpu7.yaml:131-158`

### 5.2 为什么说它的 log-prob 更“精确”

脚本文件头第一行就写了：

> `exact low-frequency action probabilities`

原因在于：

- V5 采样时真正采的是低维 `z`
- action 是 `z -> lowfreq basis -> 法向位移` 的确定性映射
- 更新时又把 `action - mean` 投影回同一个低频子空间，恢复出 `z`
- 然后直接对 `z` 用标准高斯算 log-prob

见：

- `grpo_train_v5_geom_action.py:240-280`

所以：

> **只要动作确实在这个“低频 + 法向”子空间里，V5 的 log-prob 是自洽的。**

这和 V4b 很不一样。V4b 虽然也是 3 段 outer policy，但它更像是：

- 从高维 latent 噪声生成一个 action
- 然后在 action 空间里用一个各向同性高斯 surrogate 去近似 log-prob

见：

- `grpo_train_v4b_three_iter_geom.py:240-245`
- `grpo_train_v4b_three_iter_geom.py:510-538`

所以可以把 V5 理解成：

> **把 V4b 的“几何动作”进一步收缩成一个更低维、更可逆、更容易算准 log-prob 的子空间策略。**

---

## 6. reward 是怎么定义的

### 6.1 基础质量分数

见：

- `grpo_train_v5_geom_action.py:526-540`
- `lib/train/rewards/region_reward.py:51-122`

最终质量分数 `score` 由四部分加权组成：

1. `mBoundF / boundary score`
2. `Dice`
3. `IoU`
4. `boundary distance score`

当前权重：

- `region = 0.30`
- `dice = 0.10`
- `iou = 0.25`
- `dist = 0.35`

其中 `boundary distance score` 不是简单均值距离，而是：

- 双向 contour distance
- `mean distance` 与 `quantile distance` 混合
- 再归一成 `[0, 1]` 风格分数

### 6.2 burr / spike 惩罚

见：

- `grpo_train_v5_geom_action.py:184-198`
- `grpo_train_v5_geom_action.py:744-753`

V5 还额外对“轮廓毛刺/尖刺”做惩罚：

- 用 contour Laplacian 衡量局部尖锐程度
- 和 GT 的曲率水平比较
- 看 95 分位的 excess spike
- 再得到 `burr_penalty`

当前默认：

- `reward_burr_weight = 0.06`
- `reward_burr_margin_px = 0.50`
- `reward_burr_max_px = 1.5`
- `reward_burr_quantile = 95`

最终 sampled rollout 的 reward 用的是：

```python
final_score_reward = score - 0.06 * burr_penalty
```

### 6.3 baseline 是什么

见：

- `grpo_train_v5_geom_action.py:730-764`

baseline 不是初始轮廓分数，而是：

> **当前 deterministic 三段 refinement 的最终分数**

即：

```python
baseline_score = score(deterministic_three_step)
quality = final_score_reward(sampled) - baseline_score
```

这意味着 RL 只有在下面两件事都做到时才算正向：

1. sampled rollout 比 deterministic FM baseline 更好；
2. 而且改进幅度还得覆盖 burr 惩罚。

这个设计非常保守。

### 6.4 一个很重要的细节：baseline 没扣 burr，sampled rollout 扣了 burr

代码里：

- `baseline_score`：只算 `_quality_score(...)`
- `final_score_reward`：算 `_quality_score(...) - burr_weight * burr_penalty`

也就是说：

> **sampled rollout 被额外要求“比 baseline 更好，同时还要更平滑”**

这会明显压制高频冒进更新，整体倾向于“宁可少改，也别抖”。

---

## 7. advantage、gate、PPO 更新怎么做

### 7.1 K 路 rollout

见：

- `grpo_train_v5_geom_action.py:733-761`

每个 batch 会采：

- `k = 8` 条 rollout

每条 rollout 都走完整 3 段 outer step，最后只对 **最终结果** 算 reward。

### 7.2 advantage

见：

- `grpo_train_v5_geom_action.py:761-765`

做法：

```python
quality = sampled_reward - baseline_score
adv = quality / std(quality across k)
adv = clip(adv, -2, 2)
```

这里是按 contour 维度做标准化，不是全局 batch 统一一个 advantage。

### 7.3 gate

见：

- `grpo_train_v5_geom_action.py:762-764`

```python
gate = 1 if max_k quality > gate_margin else 0
adv = adv * gate
```

当前 `gate_margin = 0`，意思是：

> 只有当某个 contour 至少存在一条 rollout 真正超过 deterministic baseline 时，这个 contour 才参与策略梯度。

这进一步强化了“只学习真实有收益的探索样本”。

### 7.4 PPO 更新

见：

- `grpo_train_v5_geom_action.py:773-823`

对每条 rollout 的每个 outer step：

1. 重新算当前策略下的 `mean_cur`
2. 用 `_geom_action_logprob(...)` 得到 `lp_cur`
3. 算 `ratio = exp(lp_cur - old_log)`
4. 做 PPO clipping
5. 加 KL 正则

当前关键超参：

- `ppo_inner_epochs = 2`
- `ppo_clip = 0.05`
- `ppo_kl_target = 0.002`
- `kl_beta = 0.01`
- `grad_clip_norm = 0.3`

### 7.5 KL 不是对完整 action 场算，而是对“均值差在低频子空间里的投影”算

见：

- `grpo_train_v5_geom_action.py:794-801`

做法是：

1. 用冻结的 `ref_flow` 求参考均值 `mean_ref`
2. 看当前均值 `mean_cur - mean_ref`
3. 把这个均值差投影到同样的 low-frequency 几何子空间
4. 对投影后的系数做二次项约束

所以这个 KL 的真实含义是：

> **限制当前 policy 不要在“RL 允许改动的低频法向子空间”里偏离 warm-start 太远。**

这比直接在全动作空间做 KL 更贴合 V5 的动作定义。

---

## 8. V5 和前几代 RL 的本质区别

## 8.1 对比 V4e / flow-GRPO

V4e 那条线的标题就是：

> `Flow-GRPO SDE logprob over stochastic ODE-to-SDE inner steps`

见：

- `grpo_train_v4e_sde_three_iter.py:1`

它的随机性主要发生在 **inner ODE/SDE 小步** 上。

而 V5 不是。V5 把随机性抬到了 **outer 3-step 几何动作** 上。

所以：

- V4e：更像“在采样过程内部做 RL”
- V5：更像“在外层 refine decision 上做 RL”

## 8.2 对比 V4b

V4b 已经是 3 段 outer policy，但动作更宽，log-prob 更近似。

V5 相比 V4b 的新增约束是：

1. **动作只允许沿法线方向**
2. **动作只保留低频模式**
3. **log-prob 在该子空间里更精确**
4. **burr 惩罚改成 GT-relative quantile 版本，更针对毛刺**

因此 V5 的总体风格是：

> **更小搜索空间、更强几何先验、更保守的 RL。**

---

## 9. 现有日志和产物说明了什么

我直接看了仓库里已经跑出的 V5 产物：

- `data/outputs/btcv_select_v4_6c_rl_v5_geom_action_gpu3/posttrain_rl_v5_geom_action/`
- `data/outputs/btcv_select_v4_6c_rl_v5b_geom_action_small_gpu5/posttrain_rl_v5_geom_action/`
- `visual/rl_v5_btcv_select_v4_6c_rl_v5_geom_action_gpu3/`
- `visual/rl_v5_btcv_select_v4_6c_rl_v5b_geom_action_small_gpu7/`

以及两份 `v5_hparams.json` 与 `logs.jsonl`。

### 9.1 已跑配置摘要

| 版本 | 动作噪声 `geom_sigma_px` | 当前日志步数 | best eval IoU | best step | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| V5 | `[1.2, 0.8, 0.5]` | 2077 | 0.925384 | 1750 | 更激进 |
| V5b | `[0.6, 0.4, 0.25]` | 1337 | 0.924771 | 1300 | 更保守 |

这些 best eval 都来自固定 eval 子集，不是全量正式 benchmark。

### 9.2 从日志看，RL 提升是“有，但不大”

从 `logs.jsonl` 里直接汇总：

- V5：
  - 初始 `eval_iou = 0.924280`
  - 最好 `eval_iou = 0.925384`
  - 增幅约 `+0.00110`
- V5b：
  - 初始 `eval_iou = 0.924119`
  - 最好 `eval_iou = 0.924771`
  - 增幅约 `+0.00065`

这说明：

> **V5 更像是对 deterministic policy 的小幅打磨，而不是大幅改写 base model 行为。**

### 9.3 reward 大部分时间仍然是负的

根据现有日志：

- V5 的 `reward_mean > 0` 占比约 `0.1%`
- V5b 的 `reward_mean > 0` 占比约 `0.75%`

而且两条线后期最新记录仍然是负 reward：

- V5 latest step 2077：`reward_mean ≈ -0.00990`
- V5b latest step 1337：`reward_mean ≈ -0.00436`

这和前面分析是一致的，因为它的优化目标本来就很保守：

1. 必须超过 deterministic baseline；
2. sampled rollout 还要承担 burr 惩罚；
3. 动作空间又被限制成低频法向子空间。

所以 V5 并不是一个“到处探索、经常找到正样本”的 RL，而是一个：

> **偶尔找到更好几何修正，然后慢慢把 mean 往那个方向推一点点** 的 RL。

### 9.4 V5b 的命名有残留不一致

当前仓库里：

- 配置文件名：`btcv_select_v4_6c_rl_v5b_geom_action_small_gpu7.yaml`
- 但文件内容里：
  - `model_dir` 写的是 `...small_gpu5`
  - `gpus: [5]`
- shell 脚本默认 `CUDA_VISIBLE_DEVICES` 也是 `5`

见：

- `configs/btcv_select_v4_6c_rl_v5b_geom_action_small_gpu7.yaml:7-9`
- `scripts/run_v5b_geom_action_small_gpu7.sh:8-10`

这说明当前 repo 里 **文件名和真实运行 GPU / 输出目录并不完全一致**，读日志时要以配置内容和输出目录为准，不要只看文件名。

---

## 10. 我对这条 V5 方法的判断

### 10.1 它为什么合理

这条线的优点很清楚：

1. **充分复用 V4.6c 基座**
   - 不破坏已有 deterministic FM 能力；
2. **动作空间被压到低维几何子空间**
   - 更容易稳定；
3. **只沿法线改，不沿切线乱漂**
   - 更符合 contour refinement 直觉；
4. **只保留低频模式**
   - 天然抑制毛刺；
5. **reward 里显式加入 burr 惩罚**
   - 进一步防高频尖刺；
6. **KL 是对子空间均值偏移做约束**
   - 和动作定义匹配。

整体上，这是一种很典型的：

> **“别让 RL 接管全部动作，而是只让它在安全子空间里做最后一点几何校正”**

的设计。

### 10.2 它为什么提升不大

从代码逻辑和现有日志看，提升不大的原因也很合理：

1. **搜索空间太小**
   - 只有法向、只有低频、只有 8 个 mode；
2. **baseline 太强**
   - baseline 已经是 warm-start 后的 deterministic 三段 refinement；
3. **reward 太保守**
   - sampled rollout 扣 burr，baseline 不扣；
4. **只看最终 reward**
   - 中间 3 步没有单独 shaped reward；
5. **eval 看的也是 deterministic policy**
   - 训练虽然在探索 stochastic rollout，但最后测的是 mean policy，提升天然有限。

所以我对 V5 的定位判断是：

> **它更像“高质量后处理型 RL 微调”，不是“强探索型 RL 重建策略”。**

---

## 11. 如果后面还要继续推这条线，最值得盯的点

基于现有实现，我觉得最重要的不是先大改模型，而是盯这几个地方：

1. **奖励对称性**
   - 现在 sampled rollout 扣 burr、baseline 不扣，过于保守；
2. **动作子空间大小**
   - 8 个 low-frequency mode 可能有点紧；
3. **三段 sigma 调度**
   - aggressive V5 和 conservative V5b 已经说明这块很敏感；
4. **评估口径**
   - 当前训练内 eval 只是固定 8 个 batch，不能替代 full eval；
5. **deterministic mean 是否真的被推优**
   - 这条线最终要看的不是 sampled reward，而是 mean policy 的正式全量指标。

---

## 12. 相关产物位置

### 12.1 代码

- `grpo_train_v5_geom_action.py`
- `lib/train/rewards/region_reward.py`
- `lib/networks/diffusion/flow_matching_evolution.py`

### 12.2 配置与运行脚本

- `configs/btcv_select_v4_6c_rl_v5_geom_action_gpu3.yaml`
- `configs/btcv_select_v4_6c_rl_v5b_geom_action_small_gpu7.yaml`
- `scripts/run_v5_geom_action_gpu3.sh`
- `scripts/run_v5b_geom_action_small_gpu7.sh`

### 12.3 日志与 checkpoint

- `data/outputs/btcv_select_v4_6c_rl_v5_geom_action_gpu3/posttrain_rl_v5_geom_action/`
- `data/outputs/btcv_select_v4_6c_rl_v5_geom_action_gpu3/checkpoints/`
- `data/outputs/btcv_select_v4_6c_rl_v5b_geom_action_small_gpu5/posttrain_rl_v5_geom_action/`
- `data/outputs/btcv_select_v4_6c_rl_v5b_geom_action_small_gpu5/checkpoints/`

### 12.4 可视化

- `visual/rl_v5_btcv_select_v4_6c_rl_v5_geom_action_gpu3/`
- `visual/rl_v5_btcv_select_v4_6c_rl_v5b_geom_action_small_gpu7/`

---

## 13. 最后的归纳

如果只用一句话总结当前仓库里的 V5 强化学习：

> **它是在 V4.6c 的 deterministic 三段 FM refine 之上，加了一个“低频 + 法向 + 有精确 logprob 的几何动作策略”，再用带 KL 和 burr 惩罚的 PPO 做保守型后训练。**

这条线的优点是解释性强、稳定、容易控；缺点是搜索空间小、收益偏有限。  
从现有日志看，它确实能把 fixed eval 的 deterministic IoU 往上推一点，但目前更像“精修增益”，还不是“代际跃迁”。
