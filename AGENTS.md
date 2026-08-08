# AGENTS — DiffusionSnake AI 协作规范

本文件是所有 AI agent 进入本仓库时**必须第一个读**的文件。
读完再看 README，再看 `docs/report/`，不要跳过。

---

## 1. 项目一句话定位

端到端矢状位医学图像实例轮廓分割：**检测 → 框初始化 → Flow Matching 轮廓演化**。
LocateAnything 检测器给出检测框，Flow Matching 网络从框初始化轮廓演化到精确轮廓。
数据集：VerSe 脊椎，25 类，矢状切片。

---

## 2. 冻结主线（直接引用，不要重新推导）

| 项目 | 值 |
|------|-----|
| 结构 | Dense-6 DiT + H1 Dense-Residual 输出头 |
| Checkpoint | `data/outputs/volmem/output_head_h0_h1_h2_20260803/distilled/h1_distilled_full.pt` |
| SHA256 | `5e28f12df357ec4d18fc9f0baf67b5a57655932a585b4ae1a0254d8449ecfc72` |
| 推理调度 | AB2，2 outer × 4 inner = 8 NFE，outer fractions `[0.6667, 1.0]` |
| 全集指标 | full-38 mean-volume Dice **0.7940**，NSD@2 **0.8094**（GT box、Memory-off、seed 20260731） |

> ⚠️ 上表数字为**非标准口径**（Dice = volume-level 前景池化；NSD@2 = voxel 单位），不可直接对标 VerSe 榜单。对标 VerSe 须按 §5 评估红线、以 `docs/report/eval_protocol_standard_20260809.md` 标准协议重跑。

**冻结主线 checkpoint 永远不动。** 任何实验都开新 worktree + 新配置文件，禁止覆盖
`h1_distilled_full.pt` 或修改 `verse_memflowdit_output_head_h1_distilled_dense_gpu0.yaml`。

---

## 3. 已判定的关键决策

### 3.1 训推初始化 — 主线走 Route B（2026-08-09 判定，不可回退）

训练和推理**都**从检测框四边中点构造 12 点八边形：

```python
get_octagon(get_quadrangle(box))   # lib/utils/snake/snake_voc_utils.py
```

配置开关：`evolve_init: bbox_octagon`（训练侧）+ `init: octagon`（推理侧）。
**训练初始化不再使用 GT 极值点。** LocateAnything 不输出极值点，用了就训推不一致。

证据（dev5，1248 slices，GT box，Memory-off，step 600，seed 20260731，三臂唯一差异是 init 开关）：

| 臂 | 前景切片 mDice | 逐卷胜负 |
|----|---:|---|
| baseline（训推不一致） | 0.760831 | — |
| route_A（统一矩形） | 0.788292 | 5W/0L |
| **route_B（统一 bbox 八边形）** | **0.790968** | **5W/0L** |

选 B 不选 A：A 的索引对齐优势全部来自重采样链，修掉重采样链后归零；B 的形状质量优势保留。
完整分析见 `docs/report/INIT_TRAIN_INFER_UNIFICATION_20260808.md`。

### 3.2 frac 语义（禁止反着读）

```python
i_init_train += full_disp * frac      # frac = 已走完的 GT 位移比例 = 外层进度
x1_raw = full_disp * (1 - frac)      # 剩余残差
```

- `frac=0`：原始初始轮廓 —— 推理第 1 步永远从这里出发，是最重要的训练状态
- `frac=1`：在 GT 上，残差为零

### 3.3 外层状态采样连续化 — 设计已定（2026-08-09）

现状缺陷（2M 采样实测）：

- `frac≈0`（±0.05）只占 **1.25%**，但 100% 推理轨迹从这里出发
- `v4_9_discrete_fractions: [0.3333, 0.5, 1.0]` 被当绝对进度消费，`1.0` 被 clamp 到 0.999，
  13.3% 样本浪费在近退化状态；根因是 `iterative_fractions`（残差比例）和
  `v4_9_discrete_fractions`（绝对进度）**单位混用**
- 28.3% 样本落在 ≥0.95，中位数 0.823

新设计：中心 `[0, 0.3333, 0.5, 0.80, 0.97]`，权重 ∝ 该步绝对进度份额，
均匀混合 λ=0.30，折叠高斯 σ=0.05，15% 均匀底噪。
效果：frac≈0 从 1.25% → **17.78%**，中位数 0.823 → **0.449**。

**实现状态（2026-08-09 已完成）**：gated 分支已插入，commit `01f5304`。
门控 `v4_10_use_continuous_sampling: true`，`v4_9_*` 路径一行不动。
四臂配置：`configs/volmem/init_unify_route_B_v410.yaml`（Route B + v4_10）。

### 3.4 重采样链不一致 — 已修（2026-08-09，commit `67158bb`）

训练：control → 128 点（一步）
推理：control → 40 点 → ÷4 → 128 点（两步，40 点中间态截角）

合同测试剩余残差全部来自此处，独立于 A/B 选择。

**修复**：`prepare_testing`（`lib/utils/snake/snake_gcn_utils.py`）中
`init=='octagon'` 分支改为 `_box_to_octagon_init(valid_boxes, poly_num)` —— 在完整图像
坐标下直接构造 128 点八边形，与训练链 `build_box_octagon_from_poly` 完全等价。
GCN 特征提取（`i_it_4py` at /4 coords）保持不变。
合同测试（4 实例合成批次）：max |train−infer| = **0.000000**。commit: `67158bb`。

---

## 4. 归因铁律（违反则数字无效）

1. **GT box** 隔离 Flow 轮廓演化能力；**predicted box** 评估部署链路；两者绝对不混用
2. oracle class 不得冒充 predicted class
3. 任何部署质量下降，先做 detector/evolution 隔离归因，不得直接比较混合指标
4. 质量数字**必须注明**：box 模式、Memory 模式、seed、评估集、步数

---

## 5. 评估红线

### Locked volumes（只读，实验结果不得引用）

**sub-verse010 / sub-verse011 / sub-verse013** — 最终 hold-out，任何开发实验不得包含。

### dev5 正确指定方式

dev5 = sub-verse022/024/071/150/264（val split 第 5/7/14/23/31 个 volume，共 1248 slices）

```bash
--volume-ids sub-verse022,sub-verse024,sub-verse071,sub-verse150,sub-verse264
```

**`--max-volumes 5` 不是 dev5。** 它按 case_id 字典序取前 5 个（010/011/013/016/018），
其中 3 个是 locked。这是已知陷阱，`eval_memflowdit_v03.py` 已加 `--volume-ids` 白名单，
其他评估脚本请同步检查。

### 评估协议必须与 VerSe 论文及相关工作对齐（红线）

评估口径**必须**与 VerSe 官方协议（Sekuboyina et al. 2021；官方评测 `anjany/verse`）+
相关工作惯例（Metrics Reloaded 2024、nnU-Net、SpineNet/VerTeBra）**一致**。
完整规范见 `docs/report/eval_protocol_standard_20260809.md`（README 有摘要章节）。要点：
- 单位：manifest spacing → 1mm 各向同性 → **mm**，禁止 voxel；
- Dice：**逐椎体平均**（aggregate + single），禁止 volume-level 前景池化；
- NSD：NSD@1mm / NSD@2mm（mm）；
- Hausdorff：HD95(mm)（可选 full HD mm）；
- **识别/分割分离**：ID rate + MLD 20mm 门控，仅正确识别椎体计入分割；

当前主线在 5 个维度偏离（Dice 池化 / NSD voxel / HD95 voxel / 无识别分离 / 无 spacing-mm），
**改评估代码前必须先对齐本规范**，不得用非标数字对标 VerSe 榜单。

---

## 6. SSH 与运行环境

```
稳定连接: med-tun（直连）
不稳定:   med（ProxyCommand，会随机 Connection closed，尽量别用）
工作目录: /home/medteam/Zhrch/DiffusionSnake-12-30
Python:   /home/medteam/miniconda3/envs/qy_esnake/bin/python
运行方式: PYTHONPATH=. python <script>
```

- 每次 Bash 调用是独立 SSH 会话，**没有持久 cwd**，每次都要 `cd` 到工作目录
- med-tun 也打印 post-quantum 警告，是噪音，grep 过滤掉即可
- numpy 在**本地不可用**，只在服务器上有，本地调试脚本要 scp 传上去运行
- SSH 命令默认超时约 2 分钟，长任务用 `nohup` 后台挂起，通过 log 文件确认状态

---

## 7. Git 规范

**实验**：`git worktree add` 开新 worktree，不在 master 上改训练代码。
惯例路径：`/home/medteam/Zhrch/DiffusionSnake-12-30-<实验名>-<YYYYMMDD>`

**文档/工具/评估脚本**：提交到 master。

禁止操作：`git push --force`、`git reset --hard`（master 上）、`git clean -f`（未确认前）。

提交信息格式：`type(scope): 一句话` — type 用 `feat/fix/docs/tools/eval/config`。

新实验结果落盘顺序：先写 `docs/report/<主题_YYYYMMDD>.md`，数字写进去，再提交。

---

## 8. 文档同步要求（每次决策必须执行）

**README.md 是其他 AI 看到的主线真相，必须始终与实际情况一致。**

| 触发事件 | 必须更新 |
|----------|---------|
| 主线结构 / checkpoint 变化 | README `§冻结主线` + `§关键数字` + 更新日志 |
| 实验结论判定（GO / NO-GO） | README 对应小节改状态 + `§已淘汰路线` 或主线变更 |
| 新发现不一致 / 陷阱 | `docs/report/<主题>.md` + README 对应章节 |
| 任何新数字 | `docs/report/` 落盘，README 引用文件路径 |

更新日志条目格式（README 末尾 `## 更新日志`，最新在最上）：

```
- **YYYY-MM-DD**: 一句话说做了什么，关键数字是多少，结论是什么
```

---

## 9. 已知陷阱速查

| 陷阱 | 正确做法 |
|------|---------|
| `--max-volumes N` 取字典序前 N 个，含 locked | 用 `--volume-ids` 白名单精确指定 |
| `iterative_fractions` 是残差比例，`v4_9_discrete_fractions` 是绝对进度 | 读 `_progress_targets_to_residual_fractions()` 的注释再用；不要互换 |
| `frac` 读反（误以为是残差） | `frac=0` 是 raw init，`frac=1` 是 on-GT |
| 并行 SSH 两条命令同时超时 | 串行执行，或 nohup 后台 + 检查 log |
| 本地运行脚本报 `No module named numpy` | scp 传到服务器再运行 |
| 修改了 H1 主线 config | 新建 config，不碰 `verse_memflowdit_output_head_h1_distilled_dense_gpu0.yaml` |
| heredoc 在此环境下不可靠 | 本地写文件 → scp 传上去 → 远程执行 |

---

## 10. 当前待完成工作

按优先级排序，接手前先确认哪些已完成：

1. **[完成 01f5304] 连续采样 gated 分支**：`flow_matching_evolution.py` 加 `v4_10_*` 分支，
   `v4_9_*` 路径一行不动；配置 `configs/volmem/init_unify_route_B_v410.yaml`
2. **[待启动] 第 4 训练臂**：Route B + v4_10，验证连续化设计的实际收益；需确认 GPU
3. **[完成 67158bb] 重采样链**：`prepare_testing` 中 octagon 分支改为直接 128 点，
   合同测试 max |delta|=0.000000
4. **[未整理] `volmem_acceleration/`（9 文件）和 `test/codex_brief_*.md`（5 文件）**
5. **[待实现] 评估协议对齐 VerSe/相关工作**：按 `docs/report/eval_protocol_standard_20260809.md`
   改造 `tools/volmem/eval_memflowdit_v03.py` / `refine_metrics3d.py`：spacing→1mm 各向同性、
   逐椎体 Dice（aggregate+single）、NSD@1/2mm、HD95(mm)、ID rate + MLD 20mm 门控；
   所有 eval 脚本默认 `--volume-ids` 白名单；补/删 `compute_stage_a_metrics.py` 引用

---

## 11. 关键文件索引

| 用途 | 路径 |
|------|------|
| 核心演化网络（最重要的文件） | `lib/networks/diffusion/flow_matching_evolution.py` |
| Init 分支实现 | `lib/utils/snake/snake_voc_utils.py` → `get_evolution_init()` |
| H1 主线配置（只读参考） | `configs/volmem/verse_memflowdit_output_head_h1_distilled_dense_gpu0.yaml` |
| 实验配置 Route A/B | `configs/volmem/init_unify_{baseline,route_A,route_B}.yaml` |
| 采样分析工具 | `tools/volmem/analyze_outer_state_sampling.py` |
| 采样设计 + 可视化生成 | `tools/volmem/design_continuous_sampling.py` |
| 评估脚本 | `tools/volmem/eval_memflowdit_v03.py` |
| Init 统一完整报告 | `docs/report/INIT_TRAIN_INFER_UNIFICATION_20260808.md` |
| 三臂对照数字 | `data/outputs/init_unify/eval_dev5_gtbox_step600/COMPARISON.md` |
| 采样概率可视化 | `data/outputs/init_unify/quantification/outer_state_sampling_design.html` |
