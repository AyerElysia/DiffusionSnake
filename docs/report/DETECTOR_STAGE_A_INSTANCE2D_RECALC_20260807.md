# Detector Stage A：Mamba Snake 式逐实例 2D 指标探索性重算（2026-08-07）

## 结论与状态

- 已基于已签名 full38 D/A/B 预测轮廓和冻结 GT，在 CPU 上完成只读重算；没有模型推理、GPU、训练、阈值搜索、重新匹配或缓存修改。
- 结果状态为 `exploratory_cpu_recalculation_complete_not_formal_project_metric`。当前项目正式指标仍是已签名的逐卷 3D mean-volume Dice；本报告不能替换正式指标。
- 按 Mamba Snake 论文公式，对每个二维实例分别计算 Dice/IoU 后实例等权平均。D-native 在全部 6160 张矢状位切片上的 mDice 为 **0.750394**（N=31,772）；每卷取一张几何中矢状位切片时为 **0.763290**（N=744）。
- 既有正式 D-native 3D mean-volume Dice 为 **0.794041**。相对差值分别为 -0.043646 和 -0.030751，但只作描述性对照：两者的单位、权重、切片范围和失败敏感性不同，不能解释成模型性能变化。

## 论文公式与本项目适配

Mamba Snake（Unified Medical Image Segmentation with State Space Modeling Snake，arXiv:2507.12760）定义：

- `mDice = (1/N) * sum_i 2|A_i ∩ B_i| / (|A_i| + |B_i|)`
- `mIoU  = (1/N) * sum_i |A_i ∩ B_i| / |A_i ∪ B_i|`
- 论文将 `A_i`、`B_i` 分别描述为第 i 个器官的 GT 和预测，并说明 VerSe 使用每位患者的 mid-sagittal plane。

论文没有明确给出中平面的提取/偶数切片 tie rule、漏检器官是否作为空预测保留在 N 中、同类断开组件是否分别作为 i。因此，本报告把下列选择预先固定为项目适配，并明确标为探索性：

1. 一个 `i` 是冻结 exact-significant Flow 合同中的一个 retained 二维连通组件/`instance_id`，不是整类合并掩膜。
2. GT 是原始 uint16 标签 PNG 中对应 `label_id` 的 8 连通组件；利用冻结 `RETR_EXTERNAL` raw-contour-area rank 绑定到 `instance_id`。
3. 预测轮廓使用 `np.rint(polygon_xy)` 后 `cv2.fillPoly` 栅格化，与冻结评估器一致。
4. 不重新匹配；只使用签名 `instance_id`。
5. 几何中矢状位：每卷在完整连续切片索引范围内选择离 `(min_idx + max_idx) / 2` 最近的一张，精确平局取较小索引；该选择不使用 GT。

## 两种实例集合口径

### 1. matched/retained 共同集合

- N=13,233：A/B 已签名、已匹配且实际进入 Flow 的相同实例；D 限制到同一组 ID。
- 不包含漏检 GT；也不包含 unmatched FP。
- 适合看已覆盖实例上的轮廓/框几何表现，不是端到端检测指标。

### 2. full-GT significant，漏检作为空预测

- N=31,772：冻结 full38 exact-significant 合同中的全部 retained GT 实例。
- A/B 缺失预测的 18,539 个实例按空预测处理，Dice=IoU=0；存在但栅格为空同样计 0。
- 该口径纳入漏检影响，但仍不含 unmatched FP，因此是 FN-aware、非完整 FP-aware 的端到端近似。
- “full GT”仅指冻结的 31,772 个 significant/retained 目标，不代表被合同过滤掉的所有原始微小组件。

## 重算结果

### 全部 6160 张矢状位切片

| 实例集合 | 条件 | N | 存在预测 | 缺失预测 | zero Dice | mDice | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| matched | D：完整 GT | 13,233 | 13,233 | 0 | 0 | 0.777342 | 0.644361 |
| matched | A：检测覆盖 + GT 几何/类 | 13,233 | 13,233 | 0 | 0 | 0.777152 | 0.644128 |
| matched | B：同覆盖 + 检测几何 + GT 类 | 13,233 | 13,233 | 0 | 14 | 0.574293 | 0.430363 |
| full-GT | D：完整 GT | 31,772 | 31,772 | 0 | 3 | **0.750394** | **0.612991** |
| full-GT | A：漏检置零 | 31,772 | 13,233 | 18,539 | 18,539 | 0.323683 | 0.268279 |
| full-GT | B：漏检置零 | 31,772 | 13,233 | 18,539 | 18,553 | 0.239192 | 0.179246 |

### 每卷一张几何中矢状位切片（38 volumes）

| 实例集合 | 条件 | N | 存在预测 | 缺失预测 | zero Dice | mDice | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| matched | D：完整 GT | 342 | 342 | 0 | 0 | 0.810630 | 0.687741 |
| matched | A：检测覆盖 + GT 几何/类 | 342 | 342 | 0 | 0 | 0.811151 | 0.688273 |
| matched | B：同覆盖 + 检测几何 + GT 类 | 342 | 342 | 0 | 0 | 0.601919 | 0.459452 |
| full-GT | D：完整 GT | 744 | 744 | 0 | 0 | **0.763290** | **0.628861** |
| full-GT | A：漏检置零 | 744 | 342 | 402 | 402 | 0.372868 | 0.316383 |
| full-GT | B：漏检置零 | 744 | 342 | 402 | 402 | 0.276689 | 0.211200 |

## 与正式 3D 指标的边界

| 指标 | 数值 | 聚合单位 |
|---|---:|---|
| 探索性 D-native，full-slices 2D mDice | 0.750394 | 31,772 个二维实例等权 |
| 探索性 D-native，geometric-mid 2D mDice | 0.763290 | 744 个二维实例等权 |
| 已有正式 D-native 3D mean-volume Dice | 0.794041 | 每卷体素聚合 Dice，再对 38 卷等权 |

逐实例 2D 口径让很小、边缘和 rank1+ 组件与大组件权重相同；3D mean-volume 则先在卷内按体素聚合，再对病例等权。因此数值不同是预期的统计现象，不能据此作模型退化、改进或跨论文 SOTA 结论。

## 合同验证与限制

- 输入计数与身份合同全部通过：38 volumes、6160 slices、D=31,772、A=B=13,233；D ID 与 full GT 完全一致，A/B ID 完全一致且为 GT 子集。
- 每个 GT 实例均复核冻结 raw contour area 和 component rank；D/A/B 三份轮廓按同一切片身份逐行读取。
- D full-GT 有 3 个 zero-Dice，B matched 有 14 个 zero-Dice；它们是“存在轮廓但与精确组件无像素重叠”，不是缺失预测。
- C 条件继续 blocked：没有登记过的非 oracle C1-L6 class provider。A/B 均使用 oracle GT class，不能称部署端到端结果。
- A/B 的 unmatched FP 未进入签名 Flow 轮廓文件，无法在本次实例等权重算中惩罚；不得将 full-GT 数值称作完整检测器 Precision/FP 指标。
- Mamba Snake 论文仅给公式与 mid-sagittal 描述，没有公开足够细的 VerSe 提取/缺失处理合同；本报告的 mid-slice 和组件实例化是显式、可复现的项目适配。

## 权威产物

- 机器结果：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_instance2d_recalc_v1_20260807/stage_a_instance2d_metrics_v1.json`  
  SHA256 `6ad55da13cc0eb7e1853526051f0e07a30355f487972f2cf9d62882603375d83`
- 逐实例表：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_instance2d_recalc_v1_20260807/stage_a_instance2d_per_instance_v1.csv.gz`  
  SHA256 `2a5eb757e5cd15cf43bd6c0ff4f322c0681a483a23b1a1c92a0d816b4e9a37d4`
- 运行 manifest：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_instance2d_recalc_v1_20260807/STAGE_A_INSTANCE2D_RECALC_MANIFEST_V1.json`  
  SHA256 `69876901e0ad8aee621795bab6815b4c43f3b2eac9a4ca74fd6aef07937e45b7`
- 重算实现：`/home/medteam/Zhrch/DiffusionSnake-12-30-detector-stage-a-20260804/tools/volmem/compute_stage_a_instance2d_metrics.py`  
  SHA256 `21c16f4250453d8d34ba3636a6ff5a8a9c1a13c465e08f2741c38a82c134122f`
- 输入哈希、每卷中矢状位选择和逐卷摘要均已写入机器 JSON；未覆盖任何既有签名产物。
