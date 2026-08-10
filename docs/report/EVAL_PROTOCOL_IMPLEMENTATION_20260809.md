# 评估协议实施状态与行动计划（2026-08-09）

## 执行总结

**任务**：确定 VerSe 评估协议标准，对比当前实现，明确改造路径。

**结论**：
1. **规范已存在**：`docs/report/eval_protocol_standard_20260809.md` 已完整定义 VerSe 标准协议（per-vertebra Dice、mm 单位、ID rate 门控、NSD@1mm/2mm、HD95），且 README 已引用，标注为"硬要求"。
2. **官方工具已 vendor**：`lib/evaluators/verse2021_3d/`（来自 `anjany/verse @02b292b`，MIT）提供 `compute_dice()` 和 `get_hits()` 等官方 primitives。
3. **实现全部缺失**：当前 `tools/volmem/eval_memflowdit_v03.py` 的 Dice 是 **volume-level 前景池化**（lines 585-605），不是 per-vertebra；无 spacing 读取、无 1mm 重采样、无 ID rate、无 mm 单位的 HD95/NSD。
4. **5 个维度偏离标准**（见下表），当前数字（0.7940 Dice / 0.8094 NSD@2）**不可对标 VerSe 榜单**。

| 维度 | 当前 | 标准 | 状态 |
|---|---|---|---|
| Dice 定义 | volume-level 前景池化 | per-vertebra mean (aggregate+single) | ❌ 未实现 |
| 物理单位 | voxel（manifest 无 spacing） | 1mm 重采样 → mm | ❌ 未实现 |
| HD95 | `hd95(voxel)` | HD95(mm)，可选 full HD | ❌ 未实现 |
| NSD | voxel，非 VerSe 原生 | NSD@1mm / @2mm (mm) | ❌ 未实现 |
| 识别/分割分离 | 无 | ID rate + MLD 20mm 门控 | ❌ 未实现 |

---

## 一、VerSe 标准协议（来源：规范文档 + agent 调研）

### 官方文献
- **Sekuboyina et al., "VerSe: A Vertebrae Labelling and Segmentation Benchmark", *Medical Image Analysis* 2021**
- 官方评测脚本：`anjany/verse`（MIT，已 vendor 到本仓库 `lib/evaluators/verse2021_3d/`）
- VerSe 2019/2020 challenge 网站（协议相同）

### 核心指标

#### 1. Dice Similarity Coefficient（必须）
- **定义**：`2|A∩B| / (|A| + |B|)`
- **计算层级**：**per-vertebra**（每个椎体实例单独算，再取均值），**不是** volume-level 前景池化
- **聚合方式**（VerSe 标准）：
  - **aggregate**：所有 GT 椎体的均值（包括漏检的）
  - **single**：仅 pred 与 GT 均出现的椎体的均值
- **报告格式**：mean ± std（跨 volume），百分比，保留 2 位小数
- **官方实现**：`lib/evaluators/verse2021_3d/eval_utilities.py::compute_dice(im1, im2)`

#### 2. Identification Rate（必须，VerSe 核心）
- **20mm MLD 门控**：预测椎体仅当其质心与最近 GT 椎体的距离 ≤ 20mm 才算"正确识别"
- **仅正确识别的椎体计入分割指标**，错标 / 误标不算
- **官方实现**：`lib/evaluators/verse2021_3d/eval_utilities.py::get_hits(cent_list_gt, cent_list_pred, max_vert_idx)`
- **报告**：ID rate (%), precision (hits/pred_count), recall (hits/gt_count)

#### 3. Hausdorff Distance（必须）
- **单位**：**mm**（物理距离）
- **报告**：HD95（95th percentile），可选 full HD
- VerSe 论文提到 "Hausdorff Surface Distance" 但未明确 HD95，现代实践（Metrics Reloaded）标准化为 HD95

#### 4. 物理单位要求（必须）
- 从原始 NIfTI 文件读取 `spacing`（或从 manifest）
- 将预测与 GT **重采样到 1mm 各向同性** spacing
- 所有表面距离（HD、NSD）用 **mm**，禁止 voxel
- **官方工具**：`lib/evaluators/verse2021_3d/data_utilities.py::resample_nib(img, voxel_spacing=(1,1,1))`

#### 5. NSD（现代补充，采纳）
- **来源**：Nikolov et al. 2021（非 VerSe 原生），经 Metrics Reloaded 2024 成为领域标准
- **报告**：NSD@1mm 和 NSD@2mm，单位 mm
- **定义**：表面点在容差范围内视为匹配，报告匹配比例

### 相关工作惯例（agent 调研）
- **nnU-Net, SpineNet, VerTeBra** 等脊柱分割工作：均用 per-vertebra Dice mean
- **VerSe leaderboard** top 方法：~90-92% per-vertebra Dice（aggregate）
- **Metrics Reloaded** (Wiesenfarth et al. 2024)：标准化 NSD@1mm/2mm 和 HD95 作为现代 Dice 补充

---

## 二、当前实现的具体问题

### 问题 1：Dice 是 volume-level 前景池化
**位置**：`tools/volmem/eval_memflowdit_v03.py:585-605`

```python
# 当前实现（错误）
for volume_id, stats in sorted(volume_stats.items()):
    union = stats["gt"] + stats["pred"] - stats["intersection"]
    denominator = stats["gt"] + stats["pred"]
    per_volume[volume_id] = {
        "dice": (
            float(2 * stats["intersection"]) / float(denominator)
            if denominator else 1.0
        ),
        # ...
    }
```

这是把整个 volume 的 gt/pred 前景像素累加后算一个 Dice，**不是** per-vertebra。

**后果**：
- 漏检椎体被掩盖（其他椎体的 TP 稀释了它）
- 错标椎体不罚（只要预测了前景就算贡献）
- 数字虚高且无法对标 VerSe 榜单（榜单是 per-vertebra ~91%，我们的 0.7940 口径完全不同）

**修复方向**：
1. 遍历每个 volume 的 GT 椎体标签（1-28）
2. 对每个标签 `L`，提取 `gt_mask_L` 和 `pred_mask_L`（需实例匹配）
3. 调用 `lib/evaluators/verse2021_3d/eval_utilities.py::compute_dice(gt_mask_L, pred_mask_L)`
4. 先过 ID rate 门控（MLD 20mm），仅正确识别的椎体计入
5. 计算 aggregate（所有 GT 椎体）和 single（仅匹配椎体）两种均值

### 问题 2：无 spacing，默认 voxel
**位置**：manifest 不含 spacing，`refine_metrics3d.py:103-131` 的 NSD/HD 直接按 voxel 算

**后果**：
- NSD@2 / HD95 的单位是 voxel，不是 mm
- 不同 spacing 的病例无法公平比较
- VerSe 标准要求 mm

**修复方向**：
1. 从原始 NIfTI（`/home/medteam/Zhrch/detect_3D_lgz2/datasets/VerSe20/dataset-verse20training/rawdata/` 或类似路径）读取 spacing
2. 将其写入 manifest，或在评估时现场读
3. 调用 `lib/evaluators/verse2021_3d/data_utilities.py::resample_nib(img, voxel_spacing=(1,1,1))` 重采样到 1mm
4. 所有距离计算在重采样后的坐标系中进行

### 问题 3：无 ID rate 门控
**位置**：当前无此步骤

**后果**：
- 错标椎体（预测标签与 GT 不符）被当作成功分割
- 无法分离"识别错误"与"分割质量"

**修复方向**：
1. 提取每个预测椎体的质心坐标（或用预测 mask 的质心）
2. 提取每个 GT 椎体的质心坐标
3. 调用 `lib/evaluators/verse2021_3d/eval_utilities.py::get_hits(cent_list_gt, cent_list_pred, max_vert_idx)`
4. 返回 hit list（哪些椎体被正确识别，MLD < 20mm）
5. 仅对 hit list 中的椎体计算分割指标

### 问题 4：HD95 / NSD 无 mm 实现
**位置**：`refine_metrics3d.py` 有 `hd95(voxel)`，但无 mm 版本

**修复方向**：
- HD95(mm)：在 1mm 重采样后的坐标系中计算 Hausdorff，用 scipy 或 SimpleITK
- NSD@1mm / @2mm：表面点提取（marching cubes 或边界检测）+ KDTree 最近邻搜索，容差 1mm / 2mm

### 问题 5：3D volume 重建
**项目特有**：我们是 2D sagittal 逐 slice 预测，需重建成 3D volume 才能与 GT NIfTI 对齐

**当前不明确**：`eval_memflowdit_v03.py` 似乎在做 2D slice 累积，但不确定是否有 3D 重建步骤

**修复方向**：
- 确认当前 evaluator 是否已做 3D 重建（检查 `evaluator.summarize()` 的上游代码）
- 若无，需从 2D slice predictions 堆叠成 3D volume（按 slice 索引对齐）
- 然后才能与 GT 3D NIfTI 做 per-vertebra 比对

---

## 三、行动计划（优先级递减）

### P0 — 立即阻断（blocking 深度/宽度结论）
**现状**：深度实验（P0/P1 step 2000 checkpoints 已在）只有 loss 数字，无 Dice/NSD/HD95。文档已明确"loss 代理与分割指标未建立相关性，负结果不能推广"。

**阻断原因**：在实现标准协议前，**任何基于当前 0.7940 Dice 的深度/宽度判断都不可信**。

**任务**：
1. **实现 per-vertebra Dice + ID rate 门控**（最小可用协议）
2. **提取 spacing + 1mm 重采样**
3. **在 dev5（sub-verse022,024,071,150,264）上跑 P0 L=6 step 2000 vs P1 L=8 step 2000**
4. **对比 per-vertebra Dice（aggregate + single）**，判断 L=8 是否真的无收益

**交付物**：
- 新评估脚本：`tools/volmem/eval_memflowdit_verse_standard.py`（调用 vendored primitives）
- 结果文档：`docs/report/DEPTH_SWEEP_VERSE_EVAL_20260809.md`（包含 per-vertebra Dice 表格）
- 若 L=8 在标准协议下**仍为负**，则深度结论坐实；若**翻正**，则 loss 代理误判

### P1 — 完整协议实施
**在 P0 后**：
5. 实现 HD95(mm) 和 NSD@1mm / @2mm
6. 报告完整表格：Dice (agg/single) / HD95 / NSD@1 / NSD@2 / ID rate
7. 在 full-38（排除 locked）上跑主线 H1 checkpoint，得到可对标 VerSe 榜单的官方数字

### P2 — 文档与 README 同步
8. 将 P1 的结果写入 README，替换当前的"非标准口径"警告
9. 标注旧数字（0.7940 / 0.8094）为"已废弃，非标准口径"
10. 在 README 评估章节加"与 VerSe leaderboard 对比"小节（我们的数字 vs top 方法 ~91%）

### P3 — 工具化
11. 把新评估脚本设为 `tools/volmem/eval_memflowdit.py` 默认（或改名），旧脚本 deprecate
12. 所有 AI 评估前必读 `docs/report/eval_protocol_standard_20260809.md`（AGENTS 规则已有，确保执行）

---

## 四、给接手 AI 的实施清单

**优先做 P0（深度实验阻塞中）**：

1. **读 spacing**：
   ```python
   import nibabel as nib
   nii = nib.load('/path/to/sub-verse022.nii.gz')
   spacing = nii.header.get_zooms()[:3]  # (x, y, z) in mm
   ```

2. **重采样到 1mm**：
   ```python
   from lib.evaluators.verse2021_3d.data_utilities import resample_nib
   resampled_nii = resample_nib(nii, voxel_spacing=(1, 1, 1))
   ```

3. **per-vertebra Dice**：
   ```python
   from lib.evaluators.verse2021_3d.eval_utilities import compute_dice
   for label in range(1, 29):  # VerSe 椎体标签 1-28
       gt_mask_L = (gt_volume == label)
       pred_mask_L = (pred_volume == label)  # 需实例匹配
       dice_L = compute_dice(gt_mask_L, pred_mask_L)
   ```

4. **ID rate 门控**：
   ```python
   from lib.evaluators.verse2021_3d.eval_utilities import get_hits
   hits, hit_list = get_hits(cent_list_gt, cent_list_pred, max_vert_idx=28)
   # 仅对 hit_list 中的椎体计算分割指标
   ```

5. **aggregate vs single**：
   - aggregate：对所有 GT 椎体（1-28 中存在的）算 Dice，取 mean
   - single：对 GT 与 pred **均出现**的椎体算 Dice，取 mean

**实现后先写 `docs/report/<主题_YYYYMMDD>.md`，再同步 README。**

---

## 五、对现有数字的处理

| 数字 | 口径 | 状态 | 引用规则 |
|---|---|---|---|
| 0.7940 (full-38 Dice) | volume-level 前景池化 | ❌ 非标准 | 必须加注"非标准口径，不可对标 VerSe 榜单" |
| 0.8094 (NSD@2) | voxel 单位 | ❌ 非标准 | 同上 |
| dev5 / 3-case 历史数字 | 同上 | ❌ 非标准 | 同上 |

**在 P1 完成前，任何对外展示（论文 / slides / 报告）必须明确标注口径非标准。**

**P1 完成后，以新数字作为官方对标数字，旧数字可保留作历史记录但不再引用。**

---

## 六、估算

- **P0 实现工作量**：~1-2 天（假设熟悉代码）
  - per-vertebra Dice + ID rate：~4-6 小时
  - spacing 读取 + 1mm 重采样：~2-3 小时
  - 3D volume 重建（若需）：~3-4 小时
  - 在 dev5 上跑 P0/P1：~2 小时（2 checkpoints × 5 volumes）
- **P1 HD95/NSD 实现**：~1 天
- **P2 文档同步**：~2 小时

**总计 ~3-4 天可完成全部协议实施。**

---

## 七、参考

- VerSe 规范文档：`docs/report/eval_protocol_standard_20260809.md`
- Vendored 官方工具：`lib/evaluators/verse2021_3d/`（已在库，MIT license）
- 当前评估脚本：`tools/volmem/eval_memflowdit_v03.py`（Dice at lines 585-605）
- Agent 调研报告：本轮 agent `adaa118999761d396` 输出
- Sekuboyina et al., VerSe benchmark paper, *Medical Image Analysis* 2021
- Metrics Reloaded (Wiesenfarth et al. 2024)
