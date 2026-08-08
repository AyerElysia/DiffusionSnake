# BTCV V4 / V5 模型分析报告

## 1. 结论先行

V4 和 V5 不是同一类改法：

- **V4 系列**主要改的是 **FM / diffusion 轮廓细化器内部**：更细的局部特征、更强的 per-point 残差、更保守的更新、更强调边界点。
- **V5 系列**主要改的是 **细化之前的初始化链路**：用 SAM / EfficientSAM / SAMSnake 把初始轮廓做得更像目标，再交给 V4.1 FM 细化。

换句话说：

- **V4 = 让“修轮廓的模型”更强**
- **V5 = 让“起始轮廓”更好**

当前仓库里，**V4 已经有完整验证指标**；**V5 主要是架构和流程升级，尚未看到同等级的全量指标定稿**，所以 V5 的“效果”更适合做设计层面的判断，而不是硬报数值。

---

## 2. 共同底座

V4 / V5 都建立在同一条 BTCV diffusion snake 主线上：

- `flow_matching_evolution.py`
- `ct_snake.py`
- `dit_denoiser_v3_4.py`

共同特点：

1. 仍然是 **Flow Matching / Rectified Flow** 主干。
2. 仍然保留 **V3.4 兼容 trunk**，方便 warm start。
3. 都使用 **128 点轮廓**。
4. 都依赖检测框 / 初始化轮廓，再进入 diffusion 迭代。

所以，V4 和 V5 的区别不是“换了一个完全新模型”，而是：

- V4 在 **预测器内部** 做增强
- V5 在 **输入初始化层** 做增强

---

## 3. V4 系列：内部增强型

### 3.1 V4.0：多尺度 detail + per-point delta

核心配置：

- `use_dit_v4: true`
- `v4_use_p3_features: true`
- `v4_use_detail_context: true`
- `v4_detail_context_mode: normal_tangent`
- `v4_use_per_point_delta: true`
- `v4_per_point_delta_scale: 0.25`

代码上对应 `DiTFlowMatchingV4`：

- 在 V3.4 trunk 上加 **detail_local_proj / detail_point_proj**
- 增加一个 **zero-init per-point delta head**
- detail context 不是单点采样，而是 normal / tangent 的局部差分信息

设计意图：

- 让模型更敏感于边缘细节
- 允许每个点做自己的小修正
- 通过 zero-init 保持 warm start 稳定

效果上，V4.0 相比 V3.4 基线已经是明显前进，但还不是最优。

---

### 3.2 V4.1：保守版 delta + curvature reweight

核心配置：

- `use_dit_v4_1: true`
- `v4_1_use_p3_features: true`
- `v4_1_use_detail_context: true`
- `v4_1_detail_context_mode: normal`
- `v4_1_use_per_point_delta: true`
- `v4_1_per_point_delta_scale: 0.10`
- `v4_1_use_curvature_reweight: true`
- `v4_1_small_disp_prob: 0.10`

这一版的关键变化不是“更多模块”，而是“更克制”：

- delta head 从 `0.25` 降到 `0.10`
- detail context 从 `normal_tangent` 收缩到 `normal`
- loss 里加入 **curvature reweight**
- 训练时增加 **small-disp** 样本

这说明 V4.1 在设计上更像“稳定化版本”：

- 不追求花哨
- 优先减少过修正
- 更照顾高曲率点和边界细节

这也是 V4 系列里最强的一版。

---

### 3.3 V4.2：curvature conditioning + gated delta

核心配置：

- `use_dit_v4_2: true`
- `v4_2_use_detail_context: true`
- `v4_2_detail_context_mode: normal_band`
- `v4_2_use_per_point_delta: true`
- `v4_2_use_delta_gate: true`
- `v4_2_use_curvature_conditioning: true`
- `v4_2_use_curvature_reweight: true`

代码上，V4.2 在 V4.1 之上又加了两层控制：

1. **Curvature conditioning**
   - 用轮廓曲率去调制 local_ctx 和 point embedding
2. **Gated per-point delta**
   - delta head 外面再套一个 sigmoid gate
   - 默认 gate bias = `-2.0`，即初始非常保守

这版的理论目标是：

- 高曲率区域更重视
- delta 只在需要时放大
- 低曲率区域少动，避免抖动

但它的代价也很明显：

- 约束更强
- 纠偏能力更容易被压住
- 训练更容易变得“太稳，稳到不够改”

从结果看，V4.2 反而掉得比较明显，说明这套“再加一层控制”的思路没有打赢 V4.1 的保守简化版。

---

### 3.4 V4.3：boundary-aware 版本

V4.3 不是一个全新 denoiser 类，而是：

- **V4.2 架构**
- + **Soft Chamfer Loss**
- + **更强 small-disp**
- + **V4.1 curvature reweight**

它的目标很直接：

- 不只看点对点 displacement
- 还直接优化轮廓边界贴合

Soft Chamfer loss 的意义是：

- 让预测轮廓和 GT 边界之间的几何距离更小
- 比纯 MSE 更贴近 `mBoundF` 这种边界指标

所以 V4.3 的定位很清楚：

- 不是单纯更大模型
- 而是把训练目标从“点位对齐”进一步推进到“边界对齐”

---

## 4. V4 的效果对比

来自 `V4_COMPARE_20260507` 的全量结果：

| 版本 | IoU | mBoundF | Dice | IoU Std |
| --- | ---: | ---: | ---: | ---: |
| V4.0 | 0.898003 | 0.786290 | 0.944859 | 0.022506 |
| V4.1 | 0.903324 | 0.794250 | 0.947976 | 0.021994 |
| V4.2 | 0.885273 | 0.762090 | 0.937337 | 0.025257 |
| V4.3 | 0.903157 | 0.793894 | 0.947951 | 0.020343 |

### 结论

1. **V4.1 最好**
   - IoU / mBoundF / Dice 都是最高
   - 而且方差最稳

2. **V4.3 非常接近 V4.1**
   - 边界相关指标很强
   - 但没有超过 V4.1

3. **V4.0 比 V4.1 稍弱**
   - 说明“多尺度 detail + delta”有效，但还不够稳

4. **V4.2 明显落后**
   - 这条线的额外控制太多，收益没有兑现

### 设计层面的解释

V4.1 赢，不是因为它最复杂，而是因为它最平衡：

- 有 detail context
- 有 per-point delta
- 但 delta 足够小
- 再用 curvature reweight 把梯度集中到真正难点

V4.2 则像是“把安全措施叠太多”：

- curvature conditioning
- gate
- stronger local context
- curvature reweight

最终可能出现 **过保守** 或 **校正被抑制** 的问题。

---

## 5. V5 系列：初始化增强型

V5 的思路跟 V4 完全不同。

V5 不再主要改 FM denoiser，而是改 **轮廓初始化阶段**：

- `contour_init_method: sam` / `efficient_sam`
- `sam_prompt_source: yolo_box` / `gt_box`
- `sam_use_in_train: true`
- `v5_2_use_samsnake_refine: true`

也就是说，V5 先把初始轮廓做得更接近目标，再让 V4.1 FM 负责最后的精修。

---

### 5.1 V5.0：YOLO box -> SAM mask -> contour -> V4.1 FM

核心配置：

- `contour_init_method: 'sam'`
- `sam_prompt_source: 'yolo_box'`
- `sam_train_prompt_source: 'yolo_box'`
- `resume_path` 指向 V4.1 checkpoint

流程是：

1. YOLO 给框
2. SAM 根据框出 mask
3. mask 转成 128 点 contour
4. V4.1 FM 再 refine

这版的本质是：

- 把 octagon 初始化替换成 SAM 初始化
- 但仍然保留 detector 质量依赖

它的优点：

- 初始化轮廓形状更贴近真实器官
- FM 需要修正的位移更小

它的风险：

- detector 框不好，SAM 也会跟着偏
- 训练链更长，失败点更多

---

### 5.2 V5.1：GT box -> SAM mask -> contour -> V4.1 FM

核心配置：

- `contour_init_method: 'sam'`
- `sam_prompt_source: 'gt_box'`
- `sam_train_prompt_source: 'gt_box'`
- `use_gt_det: true`

这版和 V5.0 的区别只有一个：

- prompt 从 **YOLO box** 换成 **GT box**

所以 V5.1 不是为了部署，而是为了回答一个问题：

> 如果 prompt 是正确的，SAM 初始化到底能把轮廓拉到什么程度？

因此 V5.1 更像 **oracle upper bound**：

- 它告诉你“初始化层最多能帮到多少”
- 如果 V5.1 明显强于 V5.0，说明瓶颈在 detector
- 如果两者差不多，说明 SAM 初始化本身已经不是主要瓶颈

---

### 5.3 V5.2：EfficientSAM + SAMSnake coarse refine + V4.1 FM

核心配置：

- `detector_backend: 'samsnake_fm'`
- `contour_init_method: 'efficient_sam'`
- `sam_backend: 'efficient_sam'`
- `v5_2_use_samsnake_refine: true`
- `samsnake_refine_stride: 4.0`
- `samsnake_refine_max_disp_frac: 0.20`
- `fm_max_disp_frac: 0.25`

流程进一步变成：

1. GT box 先送 EfficientSAM
2. EfficientSAM 输出 mask
3. mask 转成 contour
4. 进入 `SAMSnakeRefine` 做一次 coarse correction
5. 再进入 V4.1 FM

这版的核心变化有两个：

#### A. 初始化从 SAM 换成 EfficientSAM

EfficientSAM 的定位是更轻、更快的 mask proposer。

#### B. 加了 SAMSnake coarse refine

`SAMSnakeRefine` 本身是一个小型学习模块：

- 先从 feature + center + contour 抽局部特征
- 再预测 per-point offsets
- 最后用 `max_disp_frac` 限制粗修幅度

它的作用是：

- 先把 SAM contour 粗调一遍
- 再把 FM 的工作变成“精修”

所以 V5.2 是 V5 里最“系统工程化”的版本：

- 初始化
- 粗修
- 细修

三级串联。

---

## 6. V4 vs V5：设计上的本质区别

| 维度 | V4 | V5 |
| --- | --- | --- |
| 主要改动位置 | FM denoiser 内部 | 初始化链路 / coarse refine |
| 主要目标 | 提升轮廓细化能力 | 提升起始轮廓质量 |
| 复杂度来源 | 局部特征、delta、curvature、loss | SAM / EfficientSAM / SAMSnake 多级前端 |
| 对 detector 依赖 | 仍然依赖，但不是重点 | V5.0 强依赖，V5.1 用 GT box 降低该依赖 |
| 风险 | 过约束、过保守、更新不足 | 前端失败传播、链路更长、算力更高 |
| 更像什么 | “把刀磨快” | “先把菜切好再下刀” |

### 简单说

- **V4** 是模型内生优化：让网络更会修边界。
- **V5** 是输入前置优化：让网络面对的初始轮廓更靠谱。

这两种路线并不冲突，但关注点不同：

- V4 更偏 **算法优化**
- V5 更偏 **系统集成**

---

## 7. 效果理解：V4 和 V5 各自能带来什么

### V4 的收益

V4 的收益主要体现在：

- 边界更细
- 高曲率区域更稳
- 轮廓小位移更精准

从指标上看，V4.1 已经把 V4 系列拉到最好，说明当前主线里：

- **“适度增强” > “更激进增强”**

也就是，边界细化这件事，靠稳定的局部残差比靠更多约束更有效。

### V5 的收益

V5 的收益主要体现在：

- 减少初始轮廓与真实边界的初始距离
- 降低 FM 需要补偿的幅度
- 在定位正确的前提下，提升最终收敛上限

但 V5 的实际收益会更依赖：

- 检测框质量
- SAM mask 质量
- prompt 类型（YOLO box vs GT box）
- coarse refine 是否过拟合

因此 V5 更像是在解决：

> “起点太差，后面怎么修都费劲”

而不是：

> “FM 模型本身还不够强”

---

## 8. 总体判断

1. **V4 是当前已经证明有效的主线**
   - 其中 V4.1 是最优点
   - V4.3 则是更贴近边界目标的加强版

2. **V4.2 的思路更复杂，但效果更差**
   - 说明这条线不是“越加越强”
   - 而是“控制力度要刚刚好”

3. **V5 是路线切换，不是 V4 的简单延续**
   - 它把主要矛盾从“怎么 refine”转成“怎么 init”

4. **V5.1 是最适合做上界分析的版本**
   - 因为它把 prompt 误差从方框检测里剥离掉了

5. **V5.2 是最完整的系统版**
   - EfficientSAM + SAMSnake coarse refine + V4.1 FM
   - 但也最依赖工程稳定性

---

## 9. 资料来源

- `configs/btcv_diffusion_dit_v4_fm_multiscale_detail_gpu67.yaml`
- `configs/btcv_diffusion_dit_v4_1_fm_detail_curv_gpu4.yaml`
- `configs/btcv_diffusion_dit_v4_2_fm_yolom_gpu5.yaml`
- `configs/btcv_diffusion_dit_v4_3_fm_boundary_gpu67.yaml`
- `configs/btcv_diffusion_dit_v5_0_fm_sam_init.yaml`
- `configs/btcv_diffusion_dit_v5_1_fm_sam_gtbox.yaml`
- `configs/btcv_diffusion_dit_v5_2_samsnake_fm_gtbox.yaml`
- `lib/networks/diffusion/dit_denoiser_v4.py`
- `lib/networks/diffusion/dit_denoiser_v4_1.py`
- `lib/networks/diffusion/dit_denoiser_v4_2.py`
- `lib/networks/diffusion/dit_denoiser_v3_4.py`
- `lib/networks/diffusion/flow_matching_evolution.py`
- `lib/networks/snake/ct_snake.py`
- `lib/networks/snake/samsnake_refine.py`
- `lib/utils/snake/sam_init.py`
- `docs/archive/notion/archived/V4_COMPARE_20260507.md`
- `archive/report/archived/BTCV_V3_4_FM_RL_POSTTRAINING_REPORT_20260507.md`

