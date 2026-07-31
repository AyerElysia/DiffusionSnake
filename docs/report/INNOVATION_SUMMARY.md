# 创新点总结（论文形式）

> 本文档汇总 DiffusionSnake 框架的三个核心创新点，按论文 Contributions 的形式组织。
> 状态标注：✅ 已成立并有实验支撑；🔶 已实现但存在明确瓶颈；🕐 暂定 / 待具体化。
>
> 面向任务：医学图像器官轮廓分割（当前主战场 BTCV 腹部 CT），后续迁移到 3D 体数据。

---

## 摘要式定位（Abstract-style Positioning）

我们提出一个端到端的医学图像轮廓分割框架，将 **检测（YOLO）→ 初始化轮廓 → 轮廓演化** 串联为一条流水线。
其核心在于**用 Flow Matching 重构轮廓演化过程**，并在此之上引入**强化学习后训练**来优化不可微的几何质量，
最终以一套**伪 3D 处理机制**为跨切片 / 3D 体数据的迁移铺路。三者共同构成本工作的方法学贡献。

---

## Contribution 1 ✅ — Flow Matching 驱动的轮廓演化，与 DeepSnake 迭代机制融合

**一句话**：我们用 Flow Matching (FM) 取代 DeepSnake 中基于 GCN 的一步位移回归，
把轮廓演化建模为一个"初始轮廓 → 真值轮廓"的连续速度场积分过程，
并保留 Snake 的迭代式"边爬边采特征"几何先验。

### 方法要点

1. **建模对象是轮廓点的位移场，而非分割掩膜。**
   - 位移流（默认 / V4.6c）：FM 建模归一化位移 `x1 = (GT − init) / contour_scale`，
     线性插值 `x_t = (1−t)·x0 + t·x1`，速度目标 `v = x1 − x0`，损失 `MSE(v_pred, v_target)`。
   - 几何位置桥（Geom Bridge）：直接建模 `init → GT` 的 data-to-data 直线桥
     （`x0 = 归一化 init 位置 / 零位移`，`x1 = 归一化 GT 位置`），即 rectified-flow 桥，
     起点从纯高斯噪声替换为**有意义的初始轮廓**。

2. **两层 ODE 的融合结构（本贡献的核心）。**
   - **内层**：FM 的 ODE Euler 积分（连续速度场估计），推理仅需约 10 步，
     远少于 DDPM 类方法的上千步去噪。
   - **外层**：保留 DeepSnake 的迭代 deform 思想，按 `fractions = [1/3, 1/2, 1]`
     多轮推进轮廓，**每轮在当前轮廓的实际位置重新采样图像特征**
     （`get_gcn_feature` + 法向 / 切向 detail context），实现 Snake 式"轮廓爬到哪就看哪里"。

3. **几何先验的保留**：轮廓点顺序、闭环拓扑（CyclicRoPE）、法向 / 切向局部上下文采样，
   都被移植进 FM 去噪器，使概率生成模型与几何演化先验统一。

### 与已有工作的区别

- 相比 DDPM 类扩散分割（在 mask 上加噪去噪）：本工作在**轮廓点坐标 / 位移空间**做 flow matching，
  推理步数少、与 Snake 迭代机制天然兼容。
- 相比原始 DeepSnake（一步 GCN 回归）：FM 提供更强的建模表达力与多步可控演化。

### 实验证据（当前）

- Geom Bridge 单桥单轮廓推理达 **0.98 IoU**（M1 sanity，8 样本过拟合）。
- 全量泛化里程碑（M2 多样本 / M3 vs V4.6c 基线）标注为待完成。

---

## Contribution 2 🔶 — 面向几何质量的 GRPO 强化学习后训练

**一句话**：在监督预训练的 FM 模型之上，用 GRPO（Flow-GRPO 风格）做后训练，
以不可微的几何质量（IoU / 曲率 / 毛刺）为奖励，优化监督损失无法触达的目标。

### 方法要点

1. **策略（Policy）不改 FM 权重**，而是在 FM 的确定性输出上叠加一层**低频法向探索动作**：
   随机 latent 经低频 Fourier 基（8 模态）投影、乘轮廓法向，得到平滑几何扰动（`geom_action`）。
   动作空间 = "确定性 FM 位移均值 + 采样的低频法向扰动"。

2. **奖励（Reward）是几何质量综合分**：region score（IoU / Dice / boundary F）
   + 曲率细节匹配 + 毛刺（burr）惩罚。组内优势 `advantage = (quality − baseline) / std`，
   配 gate（组内至少一条超 baseline 才回传梯度）。

3. **动机**：监督 FM 只能学 init→GT 的直线均值场，多步曲线路径与高频细节
   无法靠单步 MSE 得到；GRPO 用组间 advantage 绕过奖励不可微的限制。

### 已诊断的瓶颈（诚实标注，作为后续改进方向）

- **奖励计算层**：K=8 探索平均劣于确定性输出（terminal quality mean ≈ −0.02），
  约 31% 样本整组被 gate 杀掉——低频法向扰动对已很强的轮廓平均是伤害。
- **信用分配层**：terminal-only advantage 把一个标量复制给全部演化步，
  但伤害集中在后段，step-1 与 terminal 仅弱相关（Spearman ≈ 0.34），
  导致约 25% 的早期动作被推向错误方向（经典 credit-assignment lagging）。
- **改进方向**：per-step shaped reward（把 quality 张量化为 `[K, n_steps]`），
  以及用几何桥使模型原生输出对齐 RL 的几何动作空间。

---

## Contribution 3 🕐 — 面向 3D 迁移的伪 3D 处理机制（暂定 / 待具体化）

**一句话**：我们设计一套伪 3D（pseudo-3D）处理机制，使基于 2D 切片训练的
FM + Snake 轮廓演化框架能够利用切片间的空间连续性，为迁移到 3D 体数据分割奠定基础。

> 说明：本贡献点方向已确定（服务于后续 3D 迁移），**具体技术方案待进一步实验确定**。
> 以下为候选设计思路，供后续收敛。

### 候选设计方向

1. **跨切片轮廓传播（inter-slice contour propagation）**：
   利用医学体数据相邻切片器官轮廓的高度连续性，将上一切片演化好的轮廓
   作为下一切片的初始轮廓（替代 / 增强 YOLO 检测框初始化），
   使 FM 桥的起点在切片方向上"接力"传递。

2. **2.5D 特征聚合**：在采样图像特征时，除当前切片外，融合相邻 ±k 切片的上下文，
   为 2D 演化提供体方向的证据（与现有法向 / 切向 detail context 采样机制自然衔接）。

3. **切片方向的一致性约束**：在演化 / 奖励中加入相邻切片轮廓的几何一致性项，
   抑制层间抖动，为最终 3D 表面重建提供平滑先验。

### 与前两个创新点的衔接

- Contribution 1 的"边爬边采特征"机制可直接扩展到 2.5D 特征采样；
- Contribution 2 的几何奖励可加入层间一致性项；
- 因此伪 3D 是前两点在体数据上的自然延伸，而非独立割裂的模块。

---

## 三者关系图（Narrative）

```
        [Contribution 1: FM × DeepSnake]
                 连续轮廓演化建模
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
[Contribution 2: GRPO 后训练]   [Contribution 3: 伪 3D 机制]
  优化不可微几何质量              向 3D 体数据迁移
  (IoU/曲率/毛刺)                (跨切片传播 / 2.5D 特征)
```

- **1 是地基**：定义了"用什么方式演化轮廓"。
- **2 是精修**：在 1 之上，补监督学不到的几何质量。
- **3 是外延**：把 1 + 2 从 2D 切片推广到 3D 体数据。

---

## 关键文件索引

| 模块 | 文件:行号 |
|---|---|
| FM + Snake 核心 | `lib/networks/diffusion/flow_matching_evolution.py`（训练 `forward` :2211-2748；bridge 推理 `_sample_disp_geom_bridge` :1762-1887；iterative :1963-2076；归一化 :657-697） |
| 演化模块接线 | `lib/networks/snake/ct_snake.py:810-834` |
| GRPO 强化学习 | `grpo_train_v5_geom_action.py`（几何动作 :648-695；reward :1359-1419；FM 均值动作 :720-752；advantage/gate :2137-2140） |
| Geom Bridge 范式设计 | `report/GEOM_BRIDGE_PARADIGM_DESIGN_AND_RESAMPLE_DEFERRAL_20260620.md` |
| RL 信用分配诊断 | `report/CREDIT_DIAGNOSIS_FINDINGS_20260625.md` |
| 2D FM 曲线推理失败分析 | `report/2D_FM_CURVE_FAILURE_THEORETICAL_ANALYSIS_20260626.html` |

---

*生成时间：2026-07-05 · 状态：Contribution 1/2 已有实现，Contribution 3 方向待具体化*
