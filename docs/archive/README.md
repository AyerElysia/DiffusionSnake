# docs/archive — 历史留档索引

> 整理日期：2026-08-08。本目录存放**已完成使命、不再指导当前工作**的文档与实验产物。
> 目录结构镜像整理前的原始路径（`docs/xxx/` → `docs/archive/xxx/`），因此归档件之间的相对引用全部保持有效。
>
> **当前主线文档在 `docs/report/`，不在这里。** 阅读入口见仓库根 `README.md`。

## 为什么归档而不是删除

`docs/` 共 47,395 个文件 / 2.5 GB，其中约 47,000 个是 `.gitignore` 忽略的原始评测产物——**git 无法找回**。
因此整理采取「宁归档勿删除」：只删除了 4 项经对抗复核确认无唯一内容的项（浏览器 profile 缓存 ×2、
HTML 渲染自检截图、可由 `archive/make_pptx.py` 重生成的 pptx），合计 636 文件 / 25 MB。
其余全部原样保留。

## 分区

| 目录 | 规模 | 内容 |
|------|------|------|
| `report/` | 47,244 文件 / 2.4 GB | 历史实验报告与评测产物（下表细分） |
| `pics/` | 30 文件 / 29 MB | V3–V4 网络结构图（`.svg`/`.png`/AI 渲染图） |
| `notion/archived/` | 31 文件 | Notion 导出的早期分析笔记（V3.3 失败分析、毛刺/轮廓尺寸分析等） |
| `plan/archived/` | 23 文件 | 早期迭代计划与执行方案（含 `back/` 更早一层） |
| `backup/` | 11 文件 | DiT V2 架构设计、毛刺分析总结等备份稿 |
| `net/` | 5 文件 | V3.7/V6B 网络结构图（graphviz 源 + 已渲染 PNG；**本机未装 graphviz，PNG 不可重生成**） |
| 根部单文件 | 5 项 | `V3_CODE_AUDIT_20260404.md`、`HANDOFF_V3_ALIGNMENT.md`、两份组会/训练 HTML 报告、`make_pptx.py` |

### `report/` 细分

**BTCV 时代（2026-03 ~ 2026-06）** — DDPM/DDIM 原型，战场为腹部 CT，与当前矢状位椎骨主线不可直接比较：
`BTCV_Detector_Research_Report.md`、`BTCV_检测器与极值点任务总结.md`、`DiT_V2_Implementation_Report.md`、
`V3_AC_Update_Report.md`、`V3_BugFix_Report.md`、`POSTTRAIN_DIAGNOSIS_20260501.md`、`archived/`

**已淘汰路线的完整证据链** — 结论已进入 README「已淘汰路线」小节，原始记录留此备查：
- MoE：`MOE_2026_RESEARCH_20260731.md`、`MOE_COST_AUDIT_20260802.md`、三份消融（`DENSE_VS_ODD3`/`LAYER_COVERAGE`/`SHARED_EXPERT`）、`OUTPUT_HEAD_MOE_*`
- 3D Memory：`MEMFLOWDIT_PARALLEL_3D_MEMORY_EXPERIMENT_20260731.md`
- per-point FM 尺度策略：`perpoint_fm_scale_design.md`、两份 HTML 报告、`perpoint_*` 四个产物目录
- 2D FM 曲线推理路径：两份 `2D_FM_*_20260626.html`

**历史评测产物目录**（体积主体，均为 gitignore 的原始 dump）：
`nsd_full_eval_20260708/`、`nsd_clear_full_20260708/`、`perpoint_final_eval/`、
`geom8_delta_nsd_full177/`、`sagittal_pseudo3d_assets/`、`moonvit_pseudo3d_formal_training_assets/`、
`locate_e_series_figures/`、`curve/`、`pic/`、`RL/`

## 已知取舍

- `report/curve/` 的 3 张 PNG 与 `report/archived/` 中同名文件字节相同（重复），但 JSON 是 gitignore 的孤本，故整目录保留。
- 归档件中指向 `docs/report/` 的路径引用已在 2026-08-08 统一校正：目标仍是活文档的保持原样，目标已归档的改写为 `docs/archive/report/...`。
