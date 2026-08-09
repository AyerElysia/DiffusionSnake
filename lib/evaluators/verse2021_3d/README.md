# lib/evaluators/verse2021_3d — Vendored 官方 VerSe 评估原语

> **来源（provenance，勿改）**：[github.com/anjany/verse](https://github.com/anjany/verse) @ commit `02b292b86021a8873043124982f5f58da9ba1cb8`（2021-07-23）。
> **许可**：MIT License（Copyright (c) 2020 Anjany Kumar Sekuboyina）。
> **官方说明**：该仓库 README 标注 "Evaluation (as employed in the 2020 challenge): Evaluation utilities"——即 VerSe 论文（Sekuboyina et al., Medical Image Analysis 2021）所用评估器的**官方原语**。

## 本目录提供什么（官方原版，逐字节保留，禁止修改）

| 文件 | 关键 API | 说明 |
|---|---|---|
| `eval_utilities.py` | `compute_dice(im1, im2)` | 成对 Dice 原子函数：`2·intersection / (im1.sum()+im2.sum())`。公式与本项目一致，但**本项目主评估错误地把它用在整个 volume 池化 mask 上**，官方协议要求**逐椎体调用后平均**。 |
| `eval_utilities.py` | `get_hits(cent_list_gt, cent_list_pred, max_vert_idx)` | **识别率（ID rate）+ MLD 20mm 门控**：正确标签且距离 < 20mm 才算命中。这是论文识别协议的官方实现。 |
| `data_utilities.py` | `resample_nib(img, voxel_spacing=(1,1,1), order=0)` | 重采样到 **1mm 各向同性**（论文/官方评估在 1mm 下做）。 |
| `data_utilities.py` | `rescale_centroids(ctd, img, (1,1,1))` / `load_centroids(path)` | 质心随重采样缩放 / 读取 centroid JSON。 |

## 官方仓库**没有**提供什么（必须由本项目补齐）

官方 `utils/` 只给原子原语 + 一个示例 notebook（`evaluate.ipynb`，且示例里 `compute_dice` 是对**整 mask 池化**算的，未逐椎体；**没有 Hausdorff 函数**）。因此完整评估管线需本项目按论文公式自建适配器：

1. **逐椎体编排**：对每个椎体 label 调 `compute_dice(pred_vert, gt_vert)`，再对存在的椎体取平均（论文公式 `Dice=(1/N)Σ_i 2|P_i∩T_i|/(|P_i|+|T_i|)`）。
2. **间距→mm**：当前 `sagittal_2d_fixed` manifest **无物理 spacing**，需从源 NIfTI 找回 spacing 后 `resample_nib` 到 1mm，否则距离类指标不可用。
3. **Hausdorff（full HD, mm）**：论文报 full HD（非 HD95）于 1mm 表面；官方 utils 无此函数，需按论文定义实现。
4. **NSD@1/2mm**：非 VerSe 原生指标（源自 Nikolov 2021 / Metrics Reloaded），作为现代补充，按 Metrics Reloaded 口径报。
5. **预测→3D 体素适配器**：本项目预测是矢状位 2D 轮廓，需堆叠成每例 3D label volume（label 直接对应，但需加识别门控），再喂官方原语。

## 使用纪律

- 这些文件是**外部权威代码**，只进不出：需要修复/升级时**重新 vendored 并登记新 commit**，不要在本地改。
- 主评估指标必须经由本目录官方原语计算，禁止再用 `eval_memflowdit_v03.py` / `refine_metrics3d.py` 里的自定义池化 voxel 指标作为头牌数字。
- 对应 AGENTS §10 backlog #5（评估协议对齐，硬要求）。
