# DiffusionSnake: Flow Matching 轮廓演化框架

端到端医学图像实例轮廓分割框架：**检测 → 框初始化 → Flow Matching 轮廓演化**。
用 Flow Matching (FM) 重构 DeepSnake 式轮廓演化：内层为 FM 的 ODE 连续速度场积分，外层保留 Snake "边爬边采特征" 的迭代精修，两层 ODE 融合是本工作的核心方法贡献。

当前主战场已从早期 BTCV 腹部 CT（V1–V3.5 扩散原型）迁移到 **VerSe 矢状位椎体分割**（25 类，C1–C7 / T1–T12 / L1–L6）。

---

## 论文贡献层级

| 层级 | 内容 | 状态 |
|------|------|------|
| 第一贡献 | **Flow Matching 轮廓演化**：FM 速度场建模 `init → GT` 位移，推理约 8 NFE；外层按 fractions 多轮推进并在当前轮廓位置重新采样图像特征 | ✅ 已成立 |
| 第二贡献 | **Contour RL**：GRPO 后训练，以不可微几何质量（IoU / 曲率 / 毛刺 / NSD）为奖励 | 🔶 有实现，瓶颈已诊断 |
| 第三层级 | 伪 3D / 顺序体数据能力扩展（跨切片传播、整卷并行推理） | 🕐 方向已定 |
| 系统支撑 | 检测器（LocateAnything 接入）与推理加速 | 性能支撑，非核心创新 |

贡献详述见 `docs/report/INNOVATION_SUMMARY.md`（写于 2026-07-05，**战场与超参描述已落后于当前主线，阅读时以本 README 为准**）。

---

## 当前主线状态（2026-08-05）

### 冻结主线

- **结构**：Dense-6 DiT + H1 Dense Residual 输出头（由 E8 Top-2 输出 MoE 蒸馏而来，函数相对误差 0.48%）。
- **Checkpoint**：`data/outputs/volmem/output_head_h0_h1_h2_20260803/distilled/h1_distilled_full.pt`（SHA256 `5e28f12d…`）。
- **推理调度**：AB2，2 outer × 4 inner = **8 NFE**，outer fractions `[0.6667, 1.0]`，每个 outer 在更新后的轮廓位置重采特征。
- **特征**：冻结 MoonViT layer-18（center-only），离线缓存读取。
- **接口契约**：Flow interface manifest v1.1（`label_id 1..25 → flow_class_id 0..24`）。
- 冻结细节见 `docs/report/FLOW_MAIN_HANDOVER_STATUS_20260804.md` 与 `docs/report/FLOW_GT_ORACLE_AND_INTERFACE_STATUS_20260804.md`。

### 关键数字（均 GT box、Memory-off、seed 20260731）

| 实验 | 协议 | Volume Dice |
|------|------|---:|
| GT-oracle 三病例隔离上界 | 3 例 / 333 slices / 8-NFE AB2 | 0.7940（0.7960 / 0.8168 / 0.7691） |
| full-38 D 条件（完整 GT） | 38 例 / 6160 slices / 31,772 实例 | 0.7940（mean-volume）；NSD@2 0.8094 |
| v0.5 step2300 / H1 历史锚点 | 3 例 / Batch-8 | 0.7967 |

### Detector Stage A 端到端损失归因（full-38，2026-08-05）

**结论：当前端到端损失主要来自检测器覆盖不足，其次是 matched box 定位几何，不能归因于 Flow。**

| 因子 | mean-volume Dice 损失 | NSD@2 损失 |
|------|---:|---:|
| D→A coverage（检测覆盖，recall 0.4165） | 0.1293 | 0.1767 |
| A→B geometry（框定位，oracle class） | 0.0894 | 0.0636 |
| D→B 合计 | 0.2190 | 0.2406 |

38/38 病例方向一致为下降，10,000 次 paired bootstrap 95% CI 均不跨 0。条件 C（predicted class）因无已登记分类器 blocked。详见 `docs/report/DETECTOR_STAGE_A_AB_ATTRIBUTION_STATUS_20260805.md`。

### 已淘汰路线（有严格证据，不再回退）

- **输出 MoE**：H1 蒸馏严格支配（质量保持、头参数 -63.6%、Batch-8 吞吐 +27.7%）。
- **3D Memory v0.7 / v0.8 / v0.9**：严格门控（Volume +0.001、前景同向、减速 ≤10%）下均无净收益；整卷并行吞吐 4.5× 保留为加速支撑。
- **数据管线 largest-only**：旧管线只保留 81.88% 前景，是早期演化失败的根因；修复后（每类 top-4 significant components、面积 ≥2、cap 32）前景保留 99.5%+。

---

## 方法流水线

```
矢状位切片
   │
   ├─▶ MoonViT (冻结, layer-18) ──▶ 视觉特征（离线缓存）
   │
   ├─▶ 检测：LocateAnything 离线预测 ──▶ external_detection [B,N,6]
   │        (x1,y1,x2,y2,score,class_id)；隔离实验用 GT box
   │
   ├─▶ 框初始化：box → 4 点矩形 → 1/4 分辨率 Flow 网格 → 128 点均匀上采样轮廓
   │
   └─▶ Flow Matching 演化（两层 ODE）
        ├─ 内层：FM 速度场 v(x_t, t)，AB2 积分 4 NFE（≈推理 8 步内）
        ├─ 外层：fractions [0.6667, 1.0] 多轮推进，每轮在当前轮廓位置重采特征
        └─ 输出头 H1：linear(x + residual_mlp(x)) → 位移 → 最终轮廓 [N, 128, 2]
```

训练目标：归一化位移 `x1 = (GT − init) / contour_scale`，线性插值 `x_t = (1−t)·x0 + t·x1`，速度目标 `v = x1 − x0`，损失 `MSE(v_pred, v)`。
几何先验：闭环拓扑位置编码（CyclicRoPE）、轮廓点序、法向/切向局部上下文采样。

### 数据契约（sagittal_2d_fixed）

- 每类最多 4 个显著连通域，raw contour area ≥2，单切片全局 cap 32；
- 输出网格多边形面积 > `min_poly_area_output: 0.5`，退化框剔除；
- `label_id 1..25`（解剖 mask 标签）→ `instance_id`；`flow_class_id = label_id − 1`；
- 特征采样统一 `border` padding（修复旧 zero-padding 边界断点）。

---

## 五任务分工（2026-08-04 起）

| 任务 | 职责 |
|------|------|
| 轮廓演化 / Flow 主线 | FM 方法、训练目标、采样路径、8-NFE 冻结调度、inner/outer 归因 |
| 推理加速 / 整卷并行 | 真实 pass / DiT calls / 吞吐 / 显存、Physical Volume Memory 代码 |
| 检测器 / 初始化与覆盖 | LocateAnything、检测缓存、coverage/geometry/class 隔离归因 |
| 强化学习 / Contour RL | 在冻结 Dense-6 + H1 主线上单独验证 GRPO 收益 |
| 论文统筹 | 证据审计、数字机器可读落盘、中英文写作 |

归因铁律：GT box 隔离轮廓演化能力，predicted box 评估部署链路，两者不混用；oracle class 不得冒充 predicted class；任何部署质量下降先做 detector/evolution isolation。

---

## 关键文件索引

### 核心模块

| 文件 | 说明 |
|------|------|
| `lib/networks/diffusion/flow_matching_evolution.py` | FM 演化核心：训练目标、geom bridge 推理、外层迭代精修、位移归一化 |
| `lib/networks/diffusion/dit_denoiser_v4.py` | Dense DiT + H1 Dense Residual 输出头（含 SharedDenseSparseResidualHead） |
| `lib/networks/diffusion/prototype_phi_moe.py` | DiT FFN-MoE（E4K1 prototype 路由 + φ-balancing，研究候选） |
| `lib/networks/snake/ct_snake.py` | 主网络接线（检测 → 初始化 → 演化） |
| `lib/datasets/sagittal_2d_fixed/snake.py` | 矢状位数据契约（significant components、border 采样） |

### 训练 / 评估 / 工具

| 文件 | 说明 |
|------|------|
| `tools/volmem/train_memflowdit.py` | MemFlowDiT 训练入口 |
| `tools/volmem/eval_memflowdit_parallel.py` | 整卷并行评估、蒸馏轨迹缓存 |
| `tools/volmem/distill_output_head.py` | H1/H2 输出头蒸馏、权重移植、参数统计 |
| `tools/volmem/compute_stage_a_metrics.py` | Stage A 冻结指标（NSD@2 / HD95） |
| `configs/volmem/verse_memflowdit_v0_5_minimal_gpu6.yaml` | v0.5 minimal 主线配置 |
| `scripts/extract_sagittal_moonvit_features.py` | MoonViT 特征离线提取 |

### 工作留档（docs/report/）

| 报告 | 内容 |
|------|------|
| `INNOVATION_SUMMARY.md` | 三大创新点总结（论文 Contributions 形式） |
| `FLOW_MAIN_HANDOVER_STATUS_20260804.md` | Flow 主线职责、Dense-6 + H1 冻结、归因规则 |
| `FLOW_GT_ORACLE_AND_INTERFACE_STATUS_20260804.md` | GT-oracle 隔离结果与接口冻结契约 |
| `DETECTOR_STAGE_A_STATUS_20260804.md` | 检测器 Stage A：契约冻结与四条件定义 |
| `DETECTOR_STAGE_A_AB_ATTRIBUTION_STATUS_20260805.md` | full-38 A→B 质量归因（coverage/geometry 损失分解） |
| `DETECTOR_STAGE_A_D_ZERO_CONTROL_STATUS_20260805.md` | D 条件 zero-control 验证 |
| `OUTPUT_HEAD_DISTILLATION_H0_H1_H2_20260803.md` | 输出头蒸馏实验（H1 胜出） |
| `MEMFLOWDIT_RECENT_WORK_REPORT_20260731.md` | 数据根因修复、MoE 消融、Memory v0.7–v0.9 全程记录 |
| `FLOW_MEMORY_3D_READONLY_REVIEW_20260804.md` | Memory / 3D 只读复核与淘汰结论 |
| `LOCATEANYTHING_DIFFUSIONSNAKE_INTEGRATION_REPORT_2026-07-31.md` | LocateAnything 检测接入契约 |
| `MEMFLOWDIT_NEXT_STAGE_EXECUTION_20260803.md` | DiT FFN 对照、Memory 因果审计执行链 |
| `DETECTOR_STAGE_A_INSTANCE2D_RECALC_20260807.md` | 逐实例 2D 指标探索性重算（只读；**不替代**正式 3D mean-volume Dice） |
| `DETECTOR_RECTANGLE_INIT_ABLATION_20260807.md` | bbox→初始轮廓几何消融（5 病例开发集，非 full-38 正式结果） |
| `FLOW_PURE2D_DIT4_TOTAL10K_BASELINE_AND_SLIM_B_GATE_20260807.md` | Pure-2D DiT-4 10k baseline 未达 H1 参考，slim-B 判为 NO-GO |
| `HISTORICAL_BEST_AUDIT_20260803.md` | 175 份 summary.json 审计，纠正“0.773345 是历史最佳”的误述 |
| `DETECTOR_EVOLUTION_ISOLATION_20260803.md` | 检测器与演化的隔离规则（含 2026-08-04 活契约条款） |
| `DISP_NORMALIZATION_REPORT.md` | disp 归一化统计规范——`flow_matching_evolution.py` 中 `_load_disp_stats/normalize_disp` 的唯一书面依据 |
| `GEOM_BRIDGE_PARADIGM_DESIGN_AND_RESAMPLE_DEFERRAL_20260620.md` | Geom Bridge 几何位置桥范式设计与重采样延后决策 |
| `LOCATE_FEATURE_REPLACEMENT_ARCHITECTURE_EXPLAINED_20260615.md` | `ct_snake.py` 中 `LocateFeatReplacer` 的架构说明（代码仍在用） |
| `locate_integration_analysis_20260611.md` | Locate 接入分析 + 2026-06-12 结果更新与推荐配置 |
| `RL_WORK_SUMMARY_20260708.md` | RL 阶段总结（§5 仍有未完成 TODO） |
| `RL/POLICY_GRADIENT_GRPO_EXPLANATION.md`、`RL/V5_GEOM_POLICY_GRADIENT_DETAILED.md` | 现行 RL 入口 `grpo_train_v5_geom_action.py` / `grpo_train_v7_seedflow_grpo.py` 的原理说明 |
| `CREDIT_DIAGNOSIS_FINDINGS_20260625.md` | RL 反馈信号离线诊断（reward 计算 vs 轨迹问责两条假说） |
| `inference_acceleration/ODE_ACCELERATION_REPORT_20260706.md` | 推理加速：AB2 积分器与 KV cache 基准 |
| `2D_FM_CURVE_FAILURE_THEORETICAL_ANALYSIS_20260626.html` | 2D FM 曲线失效的理论分析 |

### 设计文档（docs/design/）

| 文档 | 内容 |
|------|------|
| `VOLMEM_NAMING_AND_CODE_BOUNDARY.md` | VolMem 体数据主线的命名约定与代码边界（`volmem` 前缀、工作名 VolMemSnake） |

### 历史留档（docs/archive/）

已完成使命的文档与实验产物统一收于 `docs/archive/`，目录结构镜像整理前的原始路径。
索引见 `docs/archive/README.md`。**该目录不指导当前工作**，仅供追溯 BTCV 时代原型、
已淘汰路线（MoE / 3D Memory / per-point FM 尺度策略）的完整证据链，以及历史评测产物。

---

## 遗留版本说明（2026-03 ~ 2026-04，BTCV 时代）

早期基于 DDPM/DDIM 的扩散轮廓演化原型（V1–V3.5）在 BTCV 腹部 CT 上开发，包括：
V1 基础 Cross-Attention、V2 CyclicRoPE 奇偶交替、V2.2 MM-DiT Patchify、V2.3/V3.2 Flow Matching、V3 八边形初始化、V3.3 Circular Conv1d、V3.5 傅里叶空间扩散。
其后 V4 系列引入 MoE 输出头与 geom bridge，GRPO v5 引入几何动作 RL。

这些版本已被当前 FM 主线取代，仅作历史参考：

- V2 系列封存于 `archive/v2_legacy_2026-04-19/`；
- 早期文档（八边形初始化、DDIM 采样、边缘平滑、单样本过拟合流程等）见 git 历史中的旧版 README；
- 早期验证脚本（`verify_octagon_v3.py`、`edge_smoothing.py`、`compute_disp_stats.py` 等）已于 2026-08-08 清理，需要时从 git 历史找回；
- BTCV 时代 RL 训练脚本（grpo_train v1/v2/v4 系列）同样已清理，当前 RL 入口为 `grpo_train_v5_geom_action.py` 与 `grpo_train_v7_seedflow_grpo.py`。

---

## 快速开始

```bash
conda create -n snake1 python=3.10
conda activate snake1
pip install torch torchvision diffusers opencv-python numpy pyyaml tqdm

# 矢状位主线训练（v0.5 minimal 配置）
export CFG_FILE=configs/volmem/verse_memflowdit_v0_5_minimal_gpu6.yaml
python tools/volmem/train_memflowdit.py

# MoonViT 特征离线提取（训练/评估前必需）
python scripts/extract_sagittal_moonvit_features.py
```

---

## 更新日志

- **2026-08-08**: docs 整理——历史留档迁入 `docs/archive/`（镜像原路径），`docs/report/` 只保留活文档；删除 636 个无唯一内容的文件（浏览器 profile 缓存、渲染自检截图、可重生成的 pptx），其余一律归档不删
- **2026-08-07**: Pure-2D DiT-4 10k baseline 未保持 H1 质量，slim-B 判 NO-GO；bbox→初始轮廓 Rectangle 消融（开发集）；逐实例 2D 指标探索性重算（不替代正式指标）
- **2026-08-05**: Detector Stage A full-38 A→B 归因完成（coverage Dice -0.1293、geometry -0.0894，38/38 一致，bootstrap CI 不跨 0）；D zero-control 通过；README 按当前主线重写
- **2026-08-04**: Flow 主线接管与五任务分工；Flow interface manifest v1.1 与 H1 checkpoint 冻结；GT-oracle 三病例隔离上界 0.7940；Memory/3D 只读复核结论
- **2026-08-03**: 输出头 H0/H1/H2 蒸馏：H1 质量保持且吞吐 +27.7%，成为主线输出头；DiT FFN 四组结构对照启动
- **2026-08-02**: MoE 成本审计；3D Memory v0.8/v0.9 严格门控失败淘汰
- **2026-07-31**: 数据工程根因修复（前景保留 81.88% → 99.5%+）；MoE 重要性消融（-0.129 Dice）；DiT FFN-MoE E4K1；输出头 hard-φ 去退化；LocateAnything 检测接入
- **2026-07-29**: VolMem v0.2/v0.3 记忆条件 Flow DiT 原型
- **2026-07-21**: MoonViT 冻结特征伪 3D 矢状位正式训练
- **2026-07-10**: RL 修复 sampled_feat 转置，移除法向策略，新增 NSD 奖励
- **2026-07-07**: per-point FM 尺度策略（tanh 有界）
- **2026-06-24**: Geom Bridge 几何位置桥范式（单桥 0.98 IoU sanity）
- **2026-06-13**: GRPO v5 几何动作与信用分配诊断；Locate 集成 E 系列评估
- **2026-04 及更早**: BTCV 时代 V1–V3.5 扩散原型迭代（傅里叶空间扩散、Circular Conv1d、八边形初始化等）

---

## 参考文献

- DeepSnake: [Peng et al., CVPR 2020]
- Flow Matching / Rectified Flow: [Lipman et al., ICLR 2023], [Liu et al., 2022]
- DiT: [Peebles & Xie, ICCV 2023]
- DDPM: [Ho et al., NeurIPS 2020] / DDIM: [Song et al., ICLR 2021]
- GRPO: [Shao et al., 2024] (DeepSeekMath)
- MoonViT / LocateAnything / SAM 2 / XMem / RMem（Memory 对照依据）

---

## License

MIT License
