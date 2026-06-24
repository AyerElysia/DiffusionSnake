# 几何桥流匹配范式：设计、M1 证据与"特征重采样"搁置记录

日期：2026-06-20
作者：Claude（编排）+ Codex/ai02（执行）
状态：**重采样(resample-at-x_t)暂时搁置；先按 fixed-feature 跑通整条范式。范式已验证成立（单桥推理单轮廓 0.98 IoU）。**
相关记忆：`geom_bridge_m1_exposure_bias`、`v5_detail_reward_unsaturated`

> **🎯 关键更新（2026-06-20 晚）：M1 一切"变差"的真凶是 iterative 推理 wrapper，不是 exposure-bias、不是范式缺陷。**
> 单样本 5 路诊断证明：**单桥推理（`use_iterative_refinement: false`，10 ODE 步）把轮廓从 0.859 爬到 0.980 IoU**；iterative wrapper 反而毁掉（0.814，overshoot 2.3×，因为每个 outer step 桥预测完整剩余位移却只应用 frac、还从未训练过的半移动轮廓重跑全桥）。**修法：bridge 下推理用单桥，已写进两个 config（iterative off + flow_ode_steps 10）。** 早期所有 M1 负结果都是 M1 eval 走 iterative 造成的假象。下面第 2 节的 resample exposure-bias 结论是在 iterative 污染下得出的，需在单桥下重测才算数——但既然用户已决定搁置 resample，先不重测。

---

## 1. 为什么要这个新范式（动机）

V5 RL 后训练撞在 0.860 天花板，统一定论是：天花板 = **低频 geom 动作空间表达上限**。根因——预训练 FM 在 latent/位移空间很强，但 RL 后训练只能在它上面挂一层 8 模态低频 Fourier 法向动作；**模型空间 ≠ 动作空间**，reward 推不动高频细节。

用户提出从根上换动作空间：把 FM 从"**纯噪声 → 位移**"改成"**初始轮廓 → GT 轮廓**"的 data-to-data 直线桥（rectified-flow / bridge）。这样模型的原生输出就是"每点几何位移（128×2 全分辨率）"——正是 RL 需要的几何动作空间。预训练直线路径简单没关系，多步曲线由 RL 后训练补上。

范式定义：
- x0 = 初始轮廓（零位移），x1 = GT 位移，x_t = (1-t)·x0 + t·x1 = init→GT 直线插值。
- 速度目标 = x1 - x0 = 完整 init→GT 位移（直线，常速度场）。
- 推理：标准 FM Euler 积分，但起点从纯噪声换成初始轮廓。

---

## 2. 我们【想做但暂时搁置】的那一刀：特征在 x_t 上重采样

**动机（用户的核心卖点）：** 训练时把每点局部图像特征采在**中间插值轮廓 x_t 的实际位置**上（而不是固定 init 轮廓），让训练特征分布匹配推理时"轮廓正在往前爬、在当前位置采特征"。这给模型**实时特征反馈**——轮廓爬到哪，就看哪里的图像，像经典 active contour / snake 的精神。

**为什么搁置（M1 8 样本过拟合证据，2026-06-20）：**

| 变体 | run 内 baseline→桥后 IoU | gap | 训练 loss 降 |
|---|---|---|---|
| A 原始 resample | 0.798→0.691 | **−0.107** | 143× |
| A + x0_jitter 0.05 | 0.764→0.733 | −0.034 | 9× |
| A + x0_jitter 0.15 | 0.812→0.765 | −0.047 | 6.6× |
| A + scheduled-sampling(inner=2) | 0.787→0.746 | −0.041 | 111× |
| **B 固定 init 特征** | 0.844→0.845 | **+0.001** | 182× |

（注：各 run 起点 IoU 不可横比——dataloader 未 seed，每次 8 样本不同；只看 run 内"起点→桥后"。）

**结论：**
- 所有 resample(A) 变体即便训练 loss 降到 1e-5，桥推理后 IoU **都变差**（gap 全负）。
- 标准暴露偏置解法（x0 路径加噪 / scheduled-sampling 自 rollout）把伤害从 −0.107 收窄到 ~−0.04，但**从未转正**，jitter 还拖慢收敛。
- 只有 B（固定特征 = 常速度场）rollout 干净（持平不变差），但 B 也几乎不动（flat）。

**根因假说（两层）：**
1. **暴露偏置 / 复合误差**：训练时 x_t=t·x1_真，特征永远沿【真实路径】采；推理时 x_t 是【累积预测】位移，轮廓一漂特征就采错位置，误差逐步放大。
2. **更深一层（未定论）**：8 样本下模型≈无泛化查找表，velocity 在训练 (x_t,t) 对上记死，inference rollout 访问到的状态略偏流形即崩。**这可能意味着 8 样本 M1 对生成式 rollout 判据过严，不足以单独证伪重采样——但也可能是范式结构缺陷。** 需要有泛化的 M2 才能区分。

**因此决策（用户 2026-06-20）：重采样太麻烦，先搁置，按 fixed-feature 把其它跑通。**

---

## 3. 重采样以后怎么续做（钩子已留）

代码里所有重采样逻辑都在 flag 后面，默认关闭、关闭时逐位等价旧路径。续做时直接开 flag + 调参，不用重写：

- `flow_geom_bridge`：范式总开关（fixed-feature 路线**要开**）。
- `flow_resample_feat_at_xt`：在 x_t 上重采特征——**当前搁置，设 false**。
- `flow_geom_infer_resample_per_ode_step`：推理每 ODE 步重采——配合上一项，**搁置时 false**。
- `flow_geom_x0_jitter`：训练路径加噪（暴露偏置解法①）。
- `flow_geom_sched_sampling` / `flow_geom_sched_inner_steps` / `flow_geom_sched_prob`：scheduled-sampling 自 rollout（暴露偏置解法②，已实现）。

**未来若重启重采样，建议先验证的方向（按优先级）：**
1. 先在 fixed-feature 范式跑通、拿到正常 IoU 之后，再以它为起点**微调**开 resample（而非从零），让模型先有泛化能力。
2. 试"只在 outer-step 之间重采、ODE 内不重采"——介于 A/B 之间，复合误差更小。
3. 试特征软对齐 / detach 范围调整。
4. 区分"判据过严"vs"结构病"：在 M2（多样本、几十 epoch）下比 A(各修法) vs B 谁能正向超 init——这是关键判别实验。

代码位置：`lib/networks/diffusion/flow_matching_evolution.py`（训练 forward 约 line 2238-2300 的 bridge 块、推理 helper `_sample_disp_geom_bridge` 约 line 1679）。M1 脚本 `test/m1_geom_bridge_sanity.py`。

---

## 4. 当前路线：fixed-feature 几何桥（先跑通这个）

- config：`configs/1232_final_diffusion_dit_v4_6c_geom_bridge_scratch_noresample_gpu2.yaml`
  （`flow_geom_bridge: true`, `flow_resample_feat_at_xt: false`, `flow_geom_infer_resample_per_ode_step: true`, `v3_7_use_contour_norm: true`, rich-state/small-disp off, from scratch）。
- 这就是范式的"安全版"：init→GT 直线桥 + 固定 init 特征 = 常速度场，rollout 干净。仍然给 RL 全分辨率几何动作空间，只是特征不在移动轮廓上采。
- 里程碑：M1（多样本/更多步确认 B 能正向移动超 init）→ M2 短训 vs octagon 基线 → M3 全量 vs V4.6c。

---

## 5. 已验证的正向事实（副产物）
1. **桥恒等式精确成立**：contour-norm 下 `denormalize_pred_disp(0)=0`，x0=0 ⇔ init 轮廓；disp round-trip 误差 4.8e-7。
2. **feat_poly 始终 in-grid**（M1 断言全过，坐标落在 ~[20,127] / 136 网格内）。
3. **训练侧完全 work**：所有变体 loss 都正常下降（B 降 182×）。问题纯在 rollout 推理侧。
4. **eval 必须绕开未训练 detector**：M1 早期 0.5 IoU 是因为 eval 走 `prepare_testing` 用随机 detection head 的 init；修成从 GT-octagon init 直接调 evolution.sample_disp_iterative 后才拿到可信数。
