# 评估方式核对 — 与 VerSe 论文及相关工作一致性（2026-08-09）

> 由 FlowSnake 的 WorkBuddy 端总负责人核对。目的：确认当前主线评估方法，并判断其与 VerSe 论文官方协议、以及相关工作的领域惯例是否一致。
> 依据：仓库评估代码只读侦察（eval-code-scout，file:line 见下）+ VerSe 论文协议（verse-paper-scout）+ 领域惯例（related-work-scout）。
> 分工：实施类归其他 AI；本文为核对/确认，提交 master。

## 1. 当前主线评估实现（代码事实）
- **头牌 Dice（0.7940）是 volume-level 前景池化二值 Dice，不是逐椎体平均**：
  `eval_memflowdit_v03.py:585-605` 把每卷所有切片前景体素汇总后算 `2·Σ∩/(Σpred+Σgt)`，再对卷取均值。这是 binary foreground Dice，非 per-label 平均。
- **NSD@2 单位是 voxel 不是 mm**：`tools/volmem/rl3d/refine_metrics3d.py:103-131` 的 `surface_nsd` 容差 `taus=(0.5,1.0,2.0,3.0)` 是**体素**；同文件 `:43-49` 注明 manifest **无物理 spacing**，故"不可与发表的 mm-NSD 比较"。
- **3D 指标栈（rl3d/）**算 `nsd@1/2/3`（voxel）+ `asd` + `hd95`（pooled），**无 full HD、无 mm 单位**。
- **实例匹配 = 标签直接对应，无 overlap 指派**：`lib/evaluators/sagittal_2d_fixed/snake.py:606` `if label == class_id`；预测轮廓的 class_id 来自检测器/头（`snake.py:580`）。假设检测器标签正确。
- **"full-38" = val 切分 38 卷 / 6160 slices**（`dataset_catalog.py:90` `VolMemVal`），非 `--volume-ids` 白名单；脚本用 `--volume-start/--volume-end/--max-volumes`（`eval_memflowdit_v03.py:48-51`）。
- **代码缺口**：README 引用的 `compute_stage_a_metrics.py` 仓库内**不存在**；eval 脚本里 grep 不到 `dev5` / `verse010/011/013` 字符串——即 AGENTS §5 的评估红线是**文档约定，未在代码中强制**。

## 2. 与 VerSe 论文一致性
VerSe 官方（Sekuboyina et al., *Medical Image Analysis* 2021, arXiv 2001.09193；评测码 `github.com/anjany/verse`）：
- **官方指标**：Dice(%) + Hausdorff Distance（**full HD，mm**，局部最大距离）。**不含 NSD、不含 HD95**。
- **匹配**：分割按整型标签直接对应（1–24，T13=28）；识别(labelling)是独立任务（质心最近邻 + <20mm）。
- **粒度**：先算**逐椎体(per-label) Dice/HD**，再取数据集均值/中位数（榜单 91.7% = per-label 平均）。注意官方示例 `evaluate.ipynb` 直接对整张多标签 mask 算 pooled Dice，那是**陷阱**，不等于官方口径。
- **NSD 溯源**：NSD（"surface Dice similarity coefficient"）来自 **Nikolov et al., *Nature Medicine* 2021（DeepMind 头颈）**，τ 常 1–2mm，**非 VerSe 指标**。
- **识别/分割交互**：漏检椎体不计入 MLD/HD；标签错位在 per-label 下 Dice≈0（判失败），但在 pooled Dice 下**不惩罚**。

**一致性判定（偏离项）：**
| # | 项 | 当前实现 | VerSe 官方 | 是否一致 |
|---|----|---------|-----------|---------|
| a | Dice 口径 | 前景池化 volume Dice | 逐椎体平均 | ❌ 偏离（不可对标 91.7%） |
| b | NSD@2 | voxel 单位，且非 VerSe 指标 | 不报 NSD | ❌ 偏离（单位+出处） |
| c | Hausdorff | hd95(voxel) | full HD(mm) | ❌ 偏离（变体+单位） |
| d | 识别/分割分离 | 无；错标在 pooled 下不罚 | 仅正确识别椎体算分 | ❌ 偏离 |
| e | 物理单位 | 无 spacing → voxel | 1mm 各向同性 → mm | ❌ 偏离 |

一致的只有：报 Dice、标签直接对应匹配。

## 3. 与相关工作一致性（领域惯例）
- **Dice**：VerSe 榜单与 top 方法（Payer/SpineNet 2020 等）均用 **per-vertebra 平均**；volume/前景池化非主流。→ 当前 pooled Dice ❌ 偏离。
- **NSD**：VerSe 挑战未采用；但 **Metrics Reloaded（2024）** 推荐 NSD 为默认（τ=1–3mm），nnU-Net 亦支持。当前报 NSD 其实**比 VerSe 更现代**，但单位应为 mm（当前 voxel）❌ 偏离。
- **Hausdorff**：VerSe 用 full HD(mm)；近期工作转 **HD95(mm)** 更稳健。当前 hd95 方向对，但单位 voxel ❌ 偏离。
- **识别/分割分离**：VerSe 用 ID rate + MLD(20mm) 门控，仅正确识别椎体算 Dice。当前无此分离 ❌ 偏离。

## 4. 结论
当前主线评估与 VerSe 论文**不完全一致**，存在 5 处可修正偏离（a–e）。**0.7940 / 0.8094 是按非标准定义算出的真实输出，但引用时不可直接对标 VerSe 榜单/文献 mm-NSD**，须加注口径。

## 5. 使其 VerSe 一致 / 诚实标注 的建议
1. 增加 **per-vertebra 平均 Dice** 口径，与 volume-pooled 并列报告。
2. 引入 **spacing**，重采样 1mm 各向同性后报 **mm**；NSD 注明引自 Nikolov 2021、τ=2mm；HD 报 **full HD(mm)** 兼 HD95(mm)。
3. **分离识别与分割**：加 ID rate + MLD(20mm) 门控，仅对正确识别椎体算 Dice/HD，漏检跳过。
4. 在代码中落实 **dev5 / locked 白名单**（与 AGENTS §5 对齐），或反过来更新 AGENTS 说明代码现状。
5. 修复文档/代码不一致：README 引用的 `compute_stage_a_metrics.py` 不存在。

## 6. 待你拍板
1. 是否授权我把"评估口径修正"作为待办写入 backlog / 同步 AGENTS？
2. 0.7940 / 0.8094 对外引用时，采用「加注非标准口径」还是「先修正评估再报」？
3. 是否要我进一步核查 `eval_memflowdit_v03.py` 是否真的用了 val 全量（full-38）而非 dev5，确认 AGENTS §5 红线的实际落点？

## 7. 备注
- 所有代码结论均来自只读侦察，标注 file:line；文献结论标注论文与年份。
- 更多长期事实见项目记忆 `MEMORY.md`。
