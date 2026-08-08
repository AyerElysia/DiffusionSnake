# WorkBuddy 端总负责人 — 工作整理与现状评估（2026-08-09）

> 本文由 FlowSnake 的 WorkBuddy 端总负责人整理，**非实验结果**，仅做工作状态汇总与现状评估。
> 依据：`AGENTS.md`（仓库 AI 协作规范，最高优先级）、`README.md` 冻结主线章节、git 历史，以及本端对远程仓库的只读侦察。
> 分工边界：**实施类任务由其他 AI 负责**；本文档归属"整理 / 协调 / 声明对齐核查"，按 AGENTS §7 提交到 master。

## 1. 项目定位（已核实）
端到端矢状位医学图像实例轮廓分割：检测 → 框初始化 → Flow Matching 轮廓演化。数据集 VerSe 脊椎 25 类，矢状切片。

## 2. 冻结主线（引用 AGENTS §2 / README §冻结主线，2026-08-09 核实）
- 结构：Dense-6 DiT + H1 Dense-Residual 输出头（由 E8 Top-2 MoE 蒸馏，函数相对误差 0.48%）
- Checkpoint：`data/outputs/volmem/output_head_h0_h1_h2_20260803/distilled/h1_distilled_full.pt`
- 推理调度：AB2，2 outer × 4 inner = **8 NFE**，outer fractions `[0.6667, 1.0]`
- 特征：**冻结 MoonViT layer-18（center-only）**，离线缓存读取
- 全集指标：full-38 mean-volume Dice **0.7940**，NSD@2 **0.8094**（GT box / Memory-off / seed 20260731）

⚠️ **冻结 checkpoint 路径经核实在磁盘不存在**（数据目录 gitignored，可能在外置存储或从未落本机）。这是当前最需澄清的风险点——无法据此复现/验证 0.7940。

## 3. 现状评估（"看看情况"）

### 3.1 运行环境
- 规范 Python 环境经核实为 **`qy_esnake`**（`/home/medteam/miniconda3/envs/qy_esnake/bin/python`，3.8.20 / torch 2.4.1+cu124）。早期侦察误报为 `snake1`（也存在但非规范）。运行式 `PYTHONPATH=. python <script>`。
- 连接：稳定用 `med-tun`；`med`（ProxyCommand 代理）会随机 `Connection closed`，禁用。

### 3.2 MoonViT 特征契约：意图 vs 实装存在落差（待你拍板）
- 冻结主线（README 第 30 行）明确使用 **layer-18 center-only**，未提及 layer-26 / half_pixel。
- 但代码中存在 layer-26 引用：`volmem/data/contracts.py` 硬要求 `moonvit_layer_18`+`moonvit_layer_26`；`train_prototype.py:65` 要求 `[18,26]`；`train_memflowdit.py:91` 仅接受 `[18]`；契约测试 `test_sagittal_moonvit_contract.py` 只验 layer_18。
- 这与此前所述设计意图"固定 MoonViT layer 18+26 half_pixel"**不一致**。建议确认：冻结主线到底采用 layer-18 还是 layer-18+26？若以冻结主线（layer-18 center-only）为准，则 contracts.py / train_prototype 的"两层强制"属于非主线残留，应统一或移除以消除歧义。

### 3.3 AGENTS §10 待办与 git 实际状态不一致（文档同步缺口）
- AGENTS §10 将"连续采样 gated 分支 v4_10"标为 **[待实现]**，但 git 最新提交 `01f5304 feat: v4_10 continuous outer-loop sampling + 4th-arm config` 显示该项**疑似已实现**（含第 4 臂 config）。
- 即：① 可能已完成（AGENTS 滞后）；② 第 4 臂 config 在但训练**未启动**（需 GPU）；③④（重采样链 / 清理孤儿目录）仍待处理。
- 建议：由负责该工作的 AI 同步更新 AGENTS §10 状态，符合 AGENTS §8 文档同步要求。

### 3.4 架构与仓库健康（早期侦察结论，仍成立）
- 架构债：`lib/networks/diffusion/` 11 个 DiT denoiser 变体 + 4 个 dit_blocks 多死分支；两个并行 Snake 类 canonical 不清；`volmem_acceleration/`（9 文件）孤儿，无人 import。
- 测试：`tests/`（~20 pytest 风格）+ `test/`（临时）存在但无 pytest.ini/conftest/CI，未验证。
- Hygiene：`visual/`（44M）未进 .gitignore，有污染提交风险；`data/wandb/logs` 已正确忽略。

## 4. 工作整理（backlog）
| # | 任务 | 状态 | 负责方 | 备注 |
|---|------|------|--------|------|
| 1 | 连续采样 gated 分支 v4_10 + 第4臂 config | 疑似已实现（commit 01f5304） | 其他 AI | AGENTS §10 仍标[待实现]，需同步状态 |
| 2 | 第 4 训练臂（Route B + 新采样） | config 在 / 训练未启动 | 其他 AI | 需确认 GPU |
| 3 | 重采样链统一 control→128 一步 | 待修复 | 其他 AI | 独立 PR |
| 4 | 清理 `volmem_acceleration/`(9) + `test/codex_brief_*.md`(5) | 待整理 | 其他 AI | — |
| 5 | 定位/恢复冻结 ckpt `h1_distilled_full.pt` | **未解决（风险）** | WorkBuddy lead 协查 | 路径不存在，见 §2 |
| 6 | MoonViT 特征契约 意图 vs 实装对齐 | **待你拍板** | Ayer + WorkBuddy lead | 见 §3.2 |
| 7 | 接 pytest + `visual/` 加 .gitignore | 待处理 | 其他 AI / lead | 仓库健康 |

## 5. 待你拍板的问题
1. 冻结主线 MoonViT 究竟采用 layer-18 还是 layer-18+26？据此决定是否清理 contracts.py / train_prototype 的两层强制。
2. 冻结 ckpt 缺失如何处理——是否在本机/外置存储定位，或由其他 AI 重新产出？
3. 是否授权我在你确认后同步更新 AGENTS §10（消除文档滞后）？
4. WorkBuddy 端是否还需对其他声明（Dice 0.7940、蒸馏 -63.6%、吞吐 4.5×）做可复现性核查？这些数字目前仅存 report，无可直接复现代码入口。

## 6. 备注
- 本文件非实验结果，按 AGENTS §7 文档类提交 master。
- 所有结论均标注"已核实/待确认"，未做无依据断言。
- 更多长期事实见项目记忆 `MEMORY.md`（WorkBuddy 端）。
