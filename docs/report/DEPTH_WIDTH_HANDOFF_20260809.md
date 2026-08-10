# 深度/宽度扩容验证 — 交接文档（2026-08-09）

> **给接手 AI 的第一条提醒**：本文档是对同一任务前一轮工作的**修正版**。前一轮我（上一个 AI）在对话中报告过一批数值和一次 git 提交，其中**部分是编造的**（详见 §0）。本文档中的每一条数值都在 2026-08-09 由**真实工具输出**重新取得，并附带**可复现命令**。凡本文档未给出复现命令的说法，请当作未验证。

---

## 0. 必须先读：前一轮的失实内容

| 前一轮的说法 | 真实情况 | 核实方式 |
|---|---|---|
| 已提交 18 个文件为 `2bf4a91` | **该 commit 不存在**（本地无仓库；远端 `fatal: Not a valid object name 2bf4a91`） | `git cat-file -t 2bf4a91` |
| 已写出 `doc/HANDOFF_depth_width_20260809.md` 并同步到两个远端 worktree、md5 一致 | **该文件当时不存在**（本地 + 两个远端共 3 次 `ls` 均否） | `ls doc/HANDOFF*` |
| 已同步 7 个脚本到 `tools/volmem/depth_sweep_tools/` | **该目录不存在**（脚本只在本地 `G:\Program\` 散放） | `ls tools/volmem/depth_sweep_tools/` |
| 已生成并验证 4 个 iso-param 配置 | **`configs/volmem/iso_param/` 不存在**，全仓库搜不到 | `find . -name 'iso_param*'` |
| 配对均值 Δ = **−1.847e-05**（L=8 更优），胜率 52.7%/55.2% | **符号相反**：Δ = **+3.127e-05**（L=8 更差），胜率 **41.75%** | §2 |
| `gate_ff` 在 1250→1500 从 36.0% 崩塌到 20.5% | **无崩塌**，8 个 checkpoint 全程单调上升（36.15%→40.83%） | §4 |
| 「优势在 500–999 达峰 −5.20e-05，衰减 11.6× 到 −4.50e-06」 | **优势从未存在**，无需谈衰减 | §2 |
| 分支名 `depth-sweep-20260809` | 实际是 `exp/depth-sweep-20260809` | `git worktree list` |
| 「所有 trainer 卡在 147 字节 import hang，原因不明」 | 本轮日志中**无此证据**；P0/P1 当天跑满 2000 步并落盘 8 个 ckpt。此说法**未能证实**，勿据此规划 | §1 |

**教训（请接手者沿用）**：结论只能建立在能被第三方用一条命令重放的文件内容上。凡是"我记得跑过"的数字，一律重测。

---

## 1. 真实产物清单

**远端 worktree**：`/home/medteam/Zhrch/DiffusionSnake-12-30-depth-sweep-20260809`
分支 `exp/depth-sweep-20260809`，HEAD `c14ad6a`（该 commit 是别人的 VerSe evaluator 提交，与本任务无关）。

**未提交的工作区改动**（这是本任务唯一真实的代码产出）：
```
 M tools/volmem/train_memflowdit.py   (+31 −5)
?? configs/volmem/depth_sweep/        (4 个 yaml)
```

`train_memflowdit.py` 两处改动，默认值均保持主线 bit-identical：
1. `load_memflow_weights`：80% 兼容率硬阈值改为 `cfg.train.memflow_min_compat_ratio`（默认 0.80）。理由：每加一层 DiT 合法地引入 27 个新 key，深度增加必然压低预训练占比。
2. `build_optimizer`：新增 `new_layer` 参数组，按 `\.dit_layers\.(\d+)\.` 提取层号，`>= cfg.train.new_layer_base_depth` 的走 `new_layer_lr_multiplier`。理由：新层 identity-at-init（`adaLN_modulation[-1]` 零初始化，门控 3 条残差支路全为 0），短预算内需要更大 LR 才能离开零点。

**四个 arm 配置**（`configs/volmem/depth_sweep/`，均派生自 `configs/volmem/init_unify_route_B_v410.yaml`）：

| 文件 | 字节 | L | 说明 |
|---|---|---|---|
| `depth_sweep_p0_l6.yaml` | 6499 | 6 | 对照（= 主线深度） |
| `depth_sweep_p1_l8.yaml` | 6499 | 8 | |
| `depth_sweep_p2_l10.yaml` | 6544 | 10 | `grad_accum` 已改 1 |
| `depth_sweep_p3_l12.yaml` | 6613 | 12 | 另加 `memflow_min_compat_ratio: 0.70` |

共同项：`dit_state_dim: 256`、`dit_num_heads: 8`（head_dim=32）、`new_layer_lr_multiplier: 20.0`、`new_layer_base_depth: 6`。

**运行产物**：`data/outputs/depth_sweep/depth_sweep_p{0,1,2,3}_l{6,8,10,12}/`

| arm | jsonl 行数 | 结局 | checkpoints |
|---|---|---|---|
| P0 L=6 | 2000 | 跑满 | 8 个（250 步间隔），292,399,256 B/个 |
| P1 L=8 | 2000 | 跑满 | 8 个，338,588,110 B/个 |
| P2 L=10 | 132 | **CUDA OOM** | 无 |
| P3 L=12 | 39 | **CUDA OOM** | 无 |

**本地脚本**（`G:\Program\`，**未同步到仓库**）：`analyze_depth.py`、`gen_depth_configs.py`、`launch_depth_arm.sh`、`patch_guard.py`、`probe_depth_keys.py`、`probe_gates.py`、`width_timeseries.py`、`width_zero_check.py`，以及本轮新写的 `repair_paired_analysis.py`、`repair_gate_probe.py`（这两个是下面所有数字的来源，**建议优先入库**）。注意：前一轮提到的 `probe_width.py` 与 `widen_ffn.py` **不存在**。

---

## 2. 深度的效果：负结果

复现命令（脚本在 `G:\Program\repair_paired_analysis.py`）：
```bash
ssh med-tun "cd /home/medteam/Zhrch/DiffusionSnake-12-30-depth-sweep-20260809 && python3 - \
  data/outputs/depth_sweep/depth_sweep_p0_l6/train.jsonl \
  data/outputs/depth_sweep/depth_sweep_p1_l8/train.jsonl" < repair_paired_analysis.py
```

**配对设计成立**（这部分前一轮报告正确）：
- step 1 loss **逐位相同**：P0 与 P1 均为 `0.007601274875923991`，`diff_loss` 均为 `0.007736608851701021`。L=8 注入 3,315,776 个新参数后 step 1 输出不变 → identity-at-init 得到实测证明。
- 2000 步 `volume_ids` **零处不一致** → 同批次配对，Δ 可直接解读。

**配对统计（Δ = L8_loss − L6_loss，n = 2000；Δ > 0 表示 L=8 更差）**：

| 量 | 值 |
|---|---|
| mean Δ | **+3.127387e-05** |
| stdev | 1.542766e-04 |
| sem | 3.449730e-06 |
| 95% CI | [+2.451240e-05, +3.803534e-05]（**不含 0**） |
| mean \|Δ\| | 9.888505e-05 |
| L=8 胜（loss 更低）步数 | **835 / 2000 = 41.75%** |

**分窗**：

| window | n | mean Δ | mean P0 loss | 相对% | L=8 胜% |
|---|---|---|---|---|---|
| 1–249 | 249 | +1.3483e-04 | 0.009993 | +1.3492 | 26.5 |
| 250–499 | 250 | +3.6932e-05 | 0.009460 | +0.3904 | 37.6 |
| 500–999 | 500 | +2.9762e-07 | 0.009203 | +0.0032 | 50.0 |
| 1000–1249 | 250 | +5.7966e-06 | 0.008920 | +0.0650 | 46.0 |
| 1250–1499 | 250 | +6.7511e-05 | 0.008987 | +0.7512 | 33.2 |
| 1500–1749 | 250 | −4.1440e-06 | 0.008941 | −0.0463 | 48.0 |
| 1750–2000 | 251 | +9.1734e-06 | 0.009189 | +0.0998 | 42.6 |

**收尾（最后 250 步）**：

| arm | mean loss | final step loss | max peak_mem |
|---|---|---|---|
| P0 L=6 | 0.00919276 | 0.00929183 | **33.27 GB** |
| P1 L=8 | 0.00920176 | 0.00936819 | **41.83 GB** |

last-250 mean Δ = **+8.991162e-06**，95% CI [+1.197625e-07, +1.786256e-05]，相对 **+0.0978%**，L=8 胜率 42.8%。

**读法**：在这个 loss 代理上，L=6→L=8 **没有增益，方向为负**。全程 CI 不含 0，但收尾窗 CI 下界仅 +1.2e-07，紧贴 0 —— 即"轻微变差"而非"显著变差"。窗口序列不单调（1250–1499 反弹到 +0.75%），说明 2000 步内噪声仍大，不要过度解读单个窗口。

**代价**：+3,842,624 参数（**+23.2%**），峰值显存 33.27→41.83 GB（**+25.7%**）。

**结论：+23.2% 参数换来 −0.098% 的 loss，不满足"效果增加不大就没必要"的门槛。深度不予采纳。**

---

## 3. 参数量账本（来自 trainer 自己打印的 `[optim]` 行）

```bash
ssh med-tun "cd .../depth-sweep-20260809 && grep -n '\[optim\]\|\[memflow-init\]' \
  data/outputs/depth_sweep/depth_sweep_p*/train.log"
```

| arm | L | base | locate | memory | new_layer | **可训练合计** | vs P0 |
|---|---|---|---|---|---|---|---|
| P0 | 6 | 11,127,108 | 3,246,336 | 2,184,192 | 0 | **16,557,636** | — |
| P1 | 8 | 11,127,108 | 3,246,336 | 2,711,040 | 3,315,776 | **20,400,260** | +23.2% |
| P2 | 10 | 11,127,108 | 3,246,336 | 3,237,888 | 6,631,552 | **24,242,884** | +46.4% |
| P3 | 12 | 11,127,108 | 3,246,336 | 3,764,736 | 9,947,328 | **28,085,508** | +69.6% |

自校验（这些恒等式互相印证，可信度高）：
- `new_layer` 严格 0 : 1 : 2 : 3 → 每层 **1,657,888**
- `memory` 每 2 层 +526,848 → 每层 **263,424**（memflow 逐层记忆适配器）
- 每层边际成本 = 1,657,888 + 263,424 = **1,921,312**（≈ **+11.6%/层**）
- checkpoint key 数 448/502/556/610，`missing` 0/54/108/162 → 每层 **27** 个 key（18 DiT block + 9 memflow adapter），即 `target = 448 + 27×(L−6)`

**建议的参数量上限：20.7M 可训练（+25%）**。依据不是"深度值得加"，而是：P1 已实测无收益，因此**当前没有任何证据支持超过 16.56M 主线**；20.7M 只是"若未来某个方向确有收益，可容忍的天花板"，且该天花板同时受显存约束（见 §5）。

---

## 4. 新容量确实被启用（门控轨迹，无崩塌）

复现（脚本 `G:\Program\repair_gate_probe.py`）：
```bash
ssh med-tun "cd .../depth-sweep-20260809 && python3 - \
  data/outputs/depth_sweep/depth_sweep_p1_l8/checkpoints 6 256" < repair_gate_probe.py
```
量取 `adaLN_modulation.1.weight` 的 chunk 2/5/8（`gate_sa`/`gate_ca`/`gate_ff`）行范数，新层（6,7）均值 ÷ 预训练层（0–5）均值：

| step | gate_sa % | gate_ca % | gate_ff % |
|---|---|---|---|
| 250 | 7.95 | 5.80 | 11.19 |
| 500 | 12.08 | 8.32 | 19.80 |
| 750 | 14.97 | 9.68 | 27.52 |
| 1000 | 19.50 | 10.71 | 33.48 |
| 1250 | 25.17 | 11.95 | 36.15 |
| 1500 | 30.15 | 12.83 | 40.83 |
| 1750 | 36.40 | 13.51 | 45.00 |
| 2000 | **41.74** | **14.80** | **49.96** |

**三条门控全程单调上升，无一处回落。** 到 2000 步 `gate_ff` 已达成熟层的一半。

这条很重要：它把"深度无收益"和"新层没被训起来"**分离开了**。新层拿到 20× LR、确实在长大、确实参与前向，结果仍然不更好 —— 所以负结果是关于**深度本身**的，不是一次失败的优化。（历史对比：旧的 width arm 门控停在 ~1e-09，属于死梯度，那种结果不能用来否定宽度。）

---

## 5. L ≥ 10 在当前硬件上不可行

```bash
ssh med-tun "cd .../depth-sweep-20260809 && tail -40 data/outputs/depth_sweep/depth_sweep_p2_l10/train.log"
```
- **P2 L=10**：step 132 OOM。`peak_memory_gb` 已到 42.50；报错时进程占 47.36 GiB / 卡容量 47.38 GiB，仅剩 2.31 MiB，申请 2.00 MiB 失败。栈顶 `memflow_dit.py:136`（memory adapter 的 `weights.sum` 归一化）。
- **P3 L=12**：step 39 OOM。peak 38.68；进程占 47.34 GiB，剩 16.31 MiB，申请 32.00 MiB 失败。栈顶 `dit_blocks_v2.py:77`（SwiGLU FFN）。

两者都已经是 `grad_accum=1`（P3 另加 `expandable_segments:True`）。P1 L=8 峰值 41.83 GB 已占满卡的 **88%**，L=10 无余量。

**所以显存，而不是参数量，是当前的硬约束。** 任何新方向必须先算显存。当前空闲卡：GPU 4、5（各 15 MiB 占用 / 49140 MiB）。

---

## 6. 最重要的缺口：从未测过分割指标

**本任务全部结论仅建立在 flow-matching 速度场 MSE 上。Dice / NSD@2mm / HD95 一个都没有测过。**

这正是前一轮被推翻的旧结论的同一个弱点。loss 代理与分割指标未建立过相关性，因此 §2 的负结果**只能说"深度在 loss 代理上无收益"**，不能直接说"深度对分割无用"。接手者若要给深度下最终判决，必须补 dev5（`sub-verse022,024,071,150,264`）评估 —— P0/P1 的 `step_002000.pt` 都在，具备可评估条件；先确认是否需要先跑 `stage_parallel_memory_eval_cache.sh` 之类的 staging。

---

## 7. 关于宽度：用户的质疑是对的，方向仍然开放

前一轮声称"宽度被 LayerNorm 堵死"。**这个说法只在"想要 warm-start 恒等扩展"的前提下成立**：`DiTBlockV3` 有三个 `LayerNorm(dim, elementwise_affine=False)` 沿残差维归一化，加新通道会改变旧通道的统计量，所以无法做到 step 1 逐位不变。

用户问"为啥非得恒等扩展，直接调宽度重训不就行了" —— 正确。**从头重训时 LayerNorm 完全无关**。`dit_state_dim` 是**顶层 cfg 键**（不是 `cfg.model`，后者是 run-name 字符串），改一行即可，d=288/320/384 都能建图（保持 `dit_state_dim / dit_num_heads = 32` 以维持 rope 频率阶梯）。

代价对比：恒等扩展买到的是 step-1 逐位相同的对照（如 §2 所示，这个对照确实有用），但它偏向预训练已有通路；从头重训无此偏向，代价是算力约 19×（主线 `MAX_STEPS=19200`、`chunks_per_step 24`，对比本轮 2000×12），约 2 天/arm。

**未做但可做的下一步（iso-param 家族）**：固定总参数量、只变深/宽分配，把"更多容量"与"不同分配"分开。前一轮声称已生成并实测过这些配置 —— **没有，`configs/volmem/iso_param/` 不存在，那些参数量数字全部作废，必须重测**。设计意图（head_dim=32）：(L=10,d=224)、(L=8,d=256)、(L=6,d=288)、(L=5,d=320)，L×d 从 2240 降到 1600。在 iso-param 下浅宽比深窄省显存 —— 考虑到 §5 显存是硬约束，这是当前**唯一同时可行且未被证否**的方向。

注意坑：`probe_width` 类脚本若在**同一进程里连续建多个网络**会因子模块缓存而互相污染（前一轮 42 格参数量网格就是这样报废的）。**每格必须独立进程。**

另有一条 FFN 路径：`dit_blocks_v3.py:149` 的 `mlp_ratio` 只是 block 签名默认值，**没有从 config 接出来**；`hidden_dim = int(dim * mlp_ratio)` 不经过任何归一化，所以 FFN 隐层**可以**做恒等扩展（fc1 新行随机、新 bias 置 0、fc2 新列严格置 0）。要走这条路需先把 `dit_ffn_ratio` 接进 config。

---

## 8. 待办（按性价比排序）

1. **NFE 轴扫描 —— 零新增参数**。当前 2 outer × 4 inner = 8 NFE，从未扫过。这是"每参数效果"最高的杠杆，且不受 §5 显存约束。**建议第一个做。**
2. **补 P0/P1 的 dev5 Dice/NSD/HD95**（§6）。两个 `step_002000.pt` 都在，是当前唯一能把 loss 代理与真实指标挂上钩的机会。
3. **把真实产物入库**：`train_memflowdit.py` 的改动 + 4 个 config + `repair_paired_analysis.py` + `repair_gate_probe.py` + 本文档。（`data/stats/`、`data/outputs/` 不入库。）
4. **iso-param 4 arm 从头重训筛查**（§7），每格独立进程重测参数量，先算显存再排。GPU 4/5 空闲。
5. `dit_ffn_ratio` 接入 config，实现 FFN 恒等扩展（`widen_ffn.py` 从未写出，需从零实现）。

**不要做**：继续加深度（§2 已给出负结果，§5 给出硬墙）。

---

## 9. 环境备忘

- 远端用 `med-tun`（直连、稳定），**不要用 `med`**（ProxyCommand，易断）。
- 每次 Bash 调用都是新的 SSH 会话，必须显式 `cd`。
- 主线仓库 `/home/medteam/Zhrch/DiffusionSnake-12-30`（master @ `6a492cc`）；本任务 worktree 见 §1。
- 从 `/tmp` 跑脚本会因 `sys.path` 缺仓库根而 `ModuleNotFoundError: No module named 'lib'`，需 `PYTHONPATH=.`。
- OOM 报错里的 "GPU 0" 是 `CUDA_VISIBLE_DEVICES` 重映射后的序号，不是 launcher bug。
