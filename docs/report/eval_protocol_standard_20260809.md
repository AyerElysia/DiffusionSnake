# 评估协议标准（必须与 VerSe 论文及相关工作对齐）

- 日期：2026-08-09
- 负责人：WorkBuddy 端总负责人（评估口径 owner）
- 状态：已定为项目**硬要求**；当前主线实现偏离，待实现 AI 按本规范改造

---

## 0. 决策（来自 Ayer）

评估指标口径**必须**与 **VerSe 论文官方协议**（Sekuboyina et al., "VerSe: A Vertebrae
Labelling and Segmentation Benchmark", *Medical Image Analysis* 2021；官方评测脚本
`anjany/verse`）**以及相关工作惯例**（Metrics Reloaded, Wiesenfarth et al. 2024；nnU-Net；
SpineNet / VerTeBra 等）**保持一致**。

当前主线评估在 5 个维度偏离标准协议，因此现有 `0.7940` / `0.8094` 等数字**不能直接对标
VerSe 榜单**，必须按本规范重跑后，才以新数字作为对标 VerSe 的官方数字。

---

## 1. 标准协议（实现目标）

### 1.1 物理单位（must）
- 从 manifest 读取每例 `spacing`；预测与 GT **重采样到 1mm 各向同性**后，再算所有表面距离类指标。
- 所有 surface / Hausdorff 指标单位 = **mm**，**禁止 voxel**。
- 当前缺陷：manifest 无 spacing → 默认 voxel（见 `refine_metrics3d.py:103-131`）。

### 1.2 Dice（核心，must）
- **逐椎体（per-label / per-instance）Dice**：每个椎体实例单独算 Dice，再对椎体取平均。
- 聚合方式对齐 VerSe：同时报告 **aggregate**（所有 GT 椎体）与 **single**（pred 与 GT 均出现的椎体）两种均值。
- 报告 mean ± std（跨 volume）。
- **禁止** "前景池化 volume-level 二值 Dice"（当前 `eval_memflowdit_v03.py:585-605` 的实现）。
  该口径会掩盖错标与漏检，不可对标 VerSe 榜单（逐椎体均值 ~91.7%）。

### 1.3 NSD（Normalized Surface Dice，adopt）
- 报告 **NSD@1mm 与 NSD@2mm**，单位 **mm**。
- 渊源：Nikolov et al. 2021（非 VerSe 原生指标）；现经 Metrics Reloaded 成为领域标准，作为现代补充采用。
- 当前缺陷：NSD@2 以 voxel 计且 manifest 无 spacing（同 1.1）。

### 1.4 Hausdorff（must）
- 报告 **HD95（mm）**，可选 full HD（mm）。
- 当前缺陷：仅 `hd95(voxel)`（`refine_metrics3d.py`），无 mm 口径、无 full HD。

### 1.5 识别 / 分割分离（must，VerSe 核心）
- 单独报告 **vertebra identification rate（ID rate）**。
- 用 **minimum label distance（MLD）20mm 门控**（VerSe 标准）：预测实例仅当其最近 GT 椎体的
  MLD ≤ 20mm 才算"正确识别"。
- **仅正确识别的椎体计入分割指标**；错标 / 误标不冒充分割成功。
- 另报检测式 precision / recall（假阳 / 假阴椎体计数）。
- 当前缺陷：无识别步骤，错标在 pooled Dice 下不罚。

### 1.6 实例匹配
- VerSe 用标签直接对应 + overlap 校验；相关工作用 per-label 直接对应（标签已由识别步骤给出）。
- 匹配后未匹配的 pred / GT 计入 FP / FN。

### 1.7 可复现性来源（复用 AGENTS §4 归因铁律）
每个数字必须注明：box 模式（GT / predicted）、Memory 模式、seed、评估集
（dev5 / full-38；locked 永不计）、步数、评估协议版本（本规范）。

---

## 2. 当前主线实现的偏离（已知，待修）

| # | 维度 | 当前实现 | 标准要求 | 偏离 |
|---|------|----------|----------|------|
| a | Dice 口径 | 前景池化 volume-level 二值 | 逐椎体平均（aggregate+single） | ❌ |
| b | NSD@2 | voxel，且非 VerSe 原生 | mm（@1/@2），作为现代补充 | ❌ |
| c | Hausdorff | hd95(voxel) | HD95(full) in mm | ❌ |
| d | 识别/分割分离 | 无 | ID rate + MLD 20mm 门控 | ❌ |
| e | 单位 | 无 spacing→voxel | 1mm 各向同性→mm | ❌ |

一致部分：报 Dice、标签直接对应匹配。

---

## 3. 过程性漏洞（顺带）

- README / 工具表引用 `compute_stage_a_metrics.py`，但仓库中（疑似）不存在 → 需补实现或移除引用。
- AGENTS §5 的 dev5 / locked-volume 红线是**文档约定，代码未强制**（eval 脚本用 `--max-volumes`，
  grep 不到 dev5/locked 字符串）→ 有踩红线风险；需把 `--volume-ids` 白名单下沉为所有 eval 脚本默认。

---

## 4. 对现有数字的处理

- `0.7940`（full-38 mean-volume Dice）、`0.8094`（NSD@2）是真实跑出的输出，但定义非标准。
- **引用规则**：在按本规范重跑前，任何对外引用必须加注"非标准口径（volume-level pooled Dice / voxel NSD）"，
  **不得**直接对标 VerSe 榜单。
- 重跑后，以本规范的逐椎体 + mm 口径数字作为对标 VerSe 的官方数字。

---

## 5. 给实现 AI 的落地点

- 主改：`tools/volmem/eval_memflowdit_v03.py` 的 Dice（`585-605`）、
  `refine_metrics3d.py` 的 NSD/HD（`103-131`）。
- 新增：spacing 读取 + 1mm 重采样；per-vertebra 聚合（aggregate+single）；
  ID rate + MLD 20mm 门控；HD95(mm) + 可选 full HD(mm)。
- 所有 eval 脚本默认 `--volume-ids` 白名单；补 `compute_stage_a_metrics.py` 或删除引用。
- 实现后先写 `docs/report/<主题_YYYYMMDD>.md`，再提交（AGENTS §7/§8）。

---

## 6. 参考文献

- Sekuboyina et al., **VerSe: A Vertebrae Labelling and Segmentation Benchmark**, Medical Image Analysis 2021. 官方评测：`anjany/verse`。
- Nikolov et al., "Deep learning method for fully automated segmentation of the head and neck region…", 2021（NSD 起源）。
- Wiesenfarth et al., **Metrics Reloaded**, 2024（NSD / HD95 领域标准口径）。
- Isensee et al., **nnU-Net** 及 SpineNet / VerTeBra 等脊柱分割相关工作（per-vertebra Dice 均值惯例）。
