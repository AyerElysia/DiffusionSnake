# VerSe 官方评估器 vendored 落地报告（2026-08-09）

## 背景
Ayer 拍板：评估口径必须与 VerSe 论文（Sekuboyina et al. 2021）+ 相关工作**完全一致**，且须用原论文代码。经核查：

- 官方评估代码**存在且公开**：`github.com/anjany/verse`（MIT），README 明写 "Evaluation utilities (as employed in the 2020 challenge)"。
- 远程机 `med-tun` **无 github 出网**，故由 WorkBuddy 侧克隆（commit `02b292b`，2021-07-23）再传远程。
- 本仓库内早有 `data/outputs/.../verse2021_3d` 脚手架（引用 `from verse2021_3d import ...` + contract SHA256），但**核心模块不在仓库/任何 conda env**，属未完成脚手架，主评估未接线。

## 官方代码实际提供边界（关键认知）
官方 `utils/` 只含**原子原语**，非开箱即用评估器：
- `eval_utilities.py`：`compute_dice`（成对 Dice，公式与本项目一致）、`get_hits`（**ID rate + 20mm 门控**）。
- `data_utilities.py`：`resample_nib`(→1mm)、`rescale_centroids`、`load_centroids`。
- 示例 `evaluate.ipynb` 里 `compute_dice` 是对**整 mask 池化**算的，**无逐椎体平均、无 Hausdorff 函数**。
⇒ 论文"逐椎体 Dice 平均 + full HD(mm) + ID rate/MLD20mm"的完整编排官方仓库没有现成脚本，需本项目按论文公式自建适配器，但**原子 Dice 与 ID-rate/20mm 用官方原版**以保证一致。

## 本次落地动作（commit 见尾）
1. Vendor 官方 `eval_utilities.py` + `data_utilities.py` + `LICENSE`（MIT）到 `lib/evaluators/verse2021_3d/`，逐字节保留、禁止本地改。
2. 写 `lib/evaluators/verse2021_3d/README.md`：provenance + 提供边界 + 适配器待补清单 + 使用纪律。
3. 本文件登记：官方提供 vs 待建适配器的完整边界。

## 待建适配器（backlog #5 的剩余工作，建议派后台 agent / 其他 AI）
- [ ] 预测→3D label volume 适配器（矢状位 2D 轮廓堆叠，加识别门控）。
- [ ] 从源 NIfTI 找回物理 spacing → `resample_nib` 到 1mm（消除 voxel 单位缺陷）。
- [ ] 逐椎体 `compute_dice` 平均（替换主评估池化 Dice）。
- [ ] full Hausdorff(mm) 实现（论文口径，非 HD95）。
- [ ] `get_hits` 接入 ID rate / MLD20mm 报告。
- [ ] 可选 NSD@1/2mm（Metrics Reloaded 口径）。
- [ ] 主评估 `eval_memflowdit_v03.py` 接线官方原语；eval 默认 `--volume-ids` 白名单（dev5）。
- [ ] 回收 `data/outputs/.../verse2021_3d` 脚手架，补全后提升为 master 源码（或弃用，改走本 vendored 目录）。

## 一致性判定（更新）
- 至此，**原子指标数学已锁定为论文官方原版**；剩余不一致风险仅在"适配器编排 + spacing 找回"，均有明确待办。
- 旧的 `0.7940`（前景池化 volume Dice）/ `0.8094`（voxel NSD@2）在按本管线重跑前**仍不得**直接对标 VerSe 榜单，须加注「非标准口径」。
