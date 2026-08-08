# 训推初始化统一实验（2026-08-08，进行中）

## 一句话

训练用 GT 轮廓极值点构造八边形，推理只有检测框、构造不出同一个八边形。本实验把两侧统一，
两条候选路线（矩形 / bbox 八边形）已证明能做到**逐点精确相同**，训练臂正在跑，尚无质量结论。

状态：**进行中，勿引用为定论**。合同测试与静态量化已完成且可复现；训练/评估未完成。

---

## 1. 问题

同一个 `get_octagon()` 函数，两侧喂的输入分布不同：

| | 输入 | 代码位置 |
|---|---|---|
| 训练 | GT 轮廓真实极值点 | `lib/datasets/voc/snake.py:466` → `snake_voc_utils.get_evolution_init` |
| 推理 | 检测框四边中点 | `snake_decode.get_init` → `get_octagon(get_quadrangle(box))` |

GT 极值点（轮廓最上/左/下/右的实际点）**只在训练期存在**。LocateAnything 只输出
`[x1,y1,x2,y2,score,class]`，无法还原极值点，所以推理侧只能退化成"框四边中点八边形"。

这是训推不一致（train/inference skew）。Flow 学的是逐点位移场 `x1 = (GT − init)/scale`，
init 分布一变，速度场就落在训练时没见过的区域。

---

## 2. 已证明：两条路线都真正统一了

合同测试 `tools/volmem/validate_init_unify_contract.py`（只读，无模型无 CUDA），
对 6 个不同长宽比/尺度的框，逐点比较两侧构造出的初始轮廓：

| arm | 控制点数 训练/推理 | ctrl_maxabs | p128_maxabs | 纯重采样残差 |
|---|---|---|---|---|
| baseline（现主线） | 12 / 12 | **102.40** | 113.07 | 42.67 |
| route_B（bbox 八边形） | 12 / 12 | **0.000000** | 42.67 | 42.67 |
| route_A（8 点矩形） | 8 / 8 | **0.000000** | 16.35 | 16.35 |

`ctrl_maxabs = 0.0` 是精确 0，不是数值近似。baseline 的 102.40 px 就是当前不一致的量级。

config 接线同时验证：三个模式正确解析，非法值抛 `ValueError`。

产物：`data/outputs/init_unify/contract/init_unify_contract.json`

---

## 3. 意外发现：重采样链本身也不一致

残差**全部**来自重采样，不是形状差异（`p128_maxabs` 与 `resample_only` 完全相等）：

- 训练：control → 128，一步（numpy `uniformsample`，按全周弧长均匀分配）
- 推理：control → 40 → ÷4 → 128，两步（torch `uniform_upsample`，按边等分）

40 点中转把角点切掉，且两种分配方式给出不同的点序参数化。**同一条轮廓，训练时的第 i 个点
和推理时的第 i 个点不是同一个位置**。对逐点速度场这是真实误差，不是无害的表示差异。

Route A 的残差比 Route B 小 2.6 倍（16.35 vs 42.67 px）。

这条链是**独立于 A/B 选择的第三个修复点**，本实验未修，留作后续。

---

## 4. 静态量化（dev5：sub-verse022/024/071/150/264，1248 slices，4401 实例）

初始轮廓 vs GT mask，脚本
`tools/volmem/quantify_init_mismatch.py`（在 init-unify worktree 内）：

| 形状 | init Dice mean | init IoU mean | 到训练八边形 chamfer |
|---|---|---|---|
| 训练用 GT 极值点八边形 | 0.8373 | 0.7273 | — |
| 推理用伪八边形（现状） | 0.7866 | 0.6616 | 1.8806 |
| 8 点矩形 | 0.7650 | 0.6316 | 1.8908 |

逐点（index 对齐）距离：训练 vs 推理 5.01 px（bbox 对角线的 6.06%）；训练 vs 矩形 5.75 px（6.28%）。

**关键一条**：矩形到训练八边形的 chamfer 距离（1.8908）与现状伪八边形（1.8806）
只差 0.54%。对已训好的模型，换矩形几乎不增加"陌生度"——这解释了为什么
`DETECTOR_RECTANGLE_INIT_ABLATION_20260807.md` 里只换推理侧的消融没有掉点
（GT box +0.0012，检测框 +0.0029）。

### 数字来源与一次更正（2026-08-08）

本节数字来自可复核产物
`DiffusionSnake-12-30-init-unify-20260808/data/outputs/init_unify/quantification/init_shape_mismatch_metrics.json`
（4401 实例 / 1248 slices / 5 病例）。

本报告首次提交时引用的是一轮**产物已丢失**的运行（记为 4155 实例，
Dice 0.8350 / 0.7828 / 0.7608，chamfer 1.9900 / 1.9882）。该轮的 json 与脚本在磁盘上
均已不存在，无法复核，故按重跑结果整体替换。两轮实例数不同（4155 vs 4401），
重跑脚本为重新转写，连通域过滤可能有细微差异。

**一处方向变化**：旧轮里矩形到训练八边形的距离略**小于**伪八边形（1.9882 < 1.9900），
重跑后略**大于**（1.8908 > 1.8806）。两轮幅度都在 0.6% 以内、可忽略，
但"矩形更接近训练分布"这一措辞不再成立，正确表述是**两者到训练八边形的距离基本相同**。
结论方向不变：换矩形不显著增加陌生度。


产物 SHA256：`228c3af90b054ce89af313519cec97149010683b2deab3fede3b0a8bce8706e6`

---

## 5. 两条路线的取舍

| | Route A（矩形） | Route B（bbox 八边形） |
|---|---|---|
| 初始 Dice | 0.7650（低 0.0216） | **0.7866** |
| 索引对齐残差 | **16.35 px** | 42.67 px（差 2.6×） |
| 换过去的代价 | 几乎免费（chamfer 与现状持平） | 免费（就是现状的推理形状） |
| 控制点数 | 8 | 12 |

Route B 起点更好，Route A 索引对齐更好。静态数据判不出哪个赢，必须训练。

---

## 6. 实现

新增一个训练侧开关，与已有的推理侧开关配对：

| | 训练侧 `evolve_init_shape` | 推理侧 `box_init_shape` |
|---|---|---|
| baseline | `octagon` → `get_octagon(GT极值点)` | `octagon` → `get_octagon(quadrangle(bbox))` |
| route_A | `bbox_rectangle` → `get_box(bbox)` 8 点 | `rectangle` → `get_rectangle(bbox)` 8 点 |
| route_B | `bbox_octagon` → `get_octagon(quadrangle(bbox))` | `octagon` → 同左 ✓ |

改动文件：

- `lib/utils/snake/snake_voc_utils.py` — `get_evolution_init` 按 `evolve_init` 分派
- `lib/utils/snake/snake_config.py` — 新增 `get_evolve_init_mode()`，读 `cfg.evolve_init_shape`
- `lib/config/config.py` — 新增 `cfg.evolve_init_shape` 键
- `lib/utils/snake/snake_gcn_utils.py` — `_get_evolve_init_from_extreme` 失败关闭加固
- `lib/utils/snake/snake_decode.py` — 移植 `get_rectangle`（来自 detector-stage-a `899c184`）

**失败关闭加固的必要性**：`_get_evolve_init_from_extreme` 原本对非 octagon 模式一律
返回 `snake_decode.get_box()`（4 个角点）。若 bbox 模式走到这条路径，矩形臂会静默变成
4 角点，训推合同破裂且无任何报错。现在直接抛 `RuntimeError`。

覆盖率确认：`prev_contour_init_prob` 默认 0.0 且主线配置未设置，`use_prev_contour` 恒为
False，**100% 训练样本走 `prepare_evolution`**，开关无死代码、无部分覆盖。

---

## 7. 正在跑的训练臂

worktree `/home/medteam/Zhrch/DiffusionSnake-12-30-init-unify-20260808`，分支
`exp/init-unify-20260808`。

三臂除 init 两个键外逐行相同（diff 验证），同一 seed 20260808，同一 600 步预算，
同起点：冻结 H1 `data/model/volmem_frozen/h1_distilled_full.pt`
（SHA256 `5e28f12df357ec4d18fc9f0baf67b5a57655932a585b4ae1a0254d8449ecfc72`，已核对），
经 `--init-memflow-ckpt` 加载，`compatible=448/448 missing=0`。

配置：`configs/volmem/init_unify_{baseline,route_A,route_B}.yaml`

| arm | GPU | 状态 |
|---|---|---|
| route_A | 6 | 训练中 |
| route_B | 7 | 训练中 |
| baseline | 待卡 | **未启动**（0/1/4/5 被邻居项目占用，每臂需 23 GB，不挤占他人任务） |

约 15–19 s/step，600 步约 2.5–3 h。

评估计划：dev5，逐实例 2D mDice/mIoU 为主口径（沿用
`DETECTOR_STAGE_A_INSTANCE2D_RECALC_20260807.md` 的口径声明），per-volume 仅作一致性交叉检查。
推理侧 init 必须与各臂训练侧匹配。

---

## 8. 尚未回答

1. **哪条路线质量更好** —— 训练未完成，这是本实验的唯一目的。
2. **baseline 控制臂未跑** —— 没有它无法区分"init 合同变了"与"多训了 600 步"。**必须补**。
3. **重采样链未统一** —— 第 3 节的 16–43 px 残差在两条路线里都还在。
4. **batch 与主线不同** —— 日志显示 `chunks=12`，即 `--chunks_per_step 6` 未生效（配置值优先）。
   三臂一致所以对比有效，但绝对值不能直接当主线数字用。
5. **只在 dev5 上评** —— 不是 full-38 正式协议，结论是方向性的。

---

## 9. 复现

```bash
cd /home/medteam/Zhrch/DiffusionSnake-12-30-init-unify-20260808

# 合同测试（只读，CPU，约 3 min，机器空闲时更快）
/usr/bin/python3.8 tools/volmem/validate_init_unify_contract.py

# 单臂训练
CUDA_VISIBLE_DEVICES=6 /usr/bin/python3.8 -u tools/volmem/train_memflowdit.py \
  --cfg_file configs/volmem/init_unify_route_A.yaml \
  --device cuda:0 --max_steps 600 --save_every 200 --log_every 10 \
  --seed 20260808 \
  --init-memflow-ckpt data/model/volmem_frozen/h1_distilled_full.pt
```

踩过的坑（避免重犯）：

- `resume_path` 走 `load_initial_weights(base_network)`，只吃 2D V4.6c state dict。
  完整 MemFlowDiT 检查点必须走 `--init-memflow-ckpt`，否则 `0/448` 匹配失败。
- `run_guard` 要求 `basename(model_dir) == model`，不一致直接 `ValueError`。
- 输出目录已有运行记录会拒绝启动，崩溃后需移开残留目录。
- 两臂共卡会 OOM（每臂 23 GB / 47 GB 卡），一卡一臂。

---

## 10. 相关文档

- `DETECTOR_RECTANGLE_INIT_ABLATION_20260807.md` — 前置工作，只换推理侧的消融。
  其结论"若要成为长期训练合同，需要另做训练侧 rectangle 对齐实验"正是本实验。
- `DETECTOR_STAGE_A_INSTANCE2D_RECALC_20260807.md` — 逐实例 2D 口径声明。
- `DETECTOR_EVOLUTION_ISOLATION_20260803.md` — 归因铁律。本实验全程 GT box。


---

## 6. 训练结果：三臂 600 步微调，dev5 GT-box（2026-08-08）

三臂均从冻结主线 `h1_distilled_full.pt`（SHA256 `5e28f12d…ecfc72`）起微调 600 步，
配置只差两个 init 键（已 `diff` 验证）。评测：dev5 显式白名单、`--box-mode gt`、
`--memory-mode off`、seed 20260731、5 outer × 20 inner。

### 有效性闸门

| arm | vols | slices | fg slices | fg w/ pred | GT boxes | dev5 | locked |
|---|---:|---:|---:|---:|---:|---|---|
| baseline | 5 | 1248 | 429 | 422 | 4340 | PASS | none |
| route_A | 5 | 1248 | 429 | 422 | 4340 | PASS | none |
| route_B | 5 | 1248 | 429 | 422 | 4340 | PASS | none |

四项计数三臂完全一致；三臂 checkpoint step 均为 600。

### 主口径：2D 逐前景切片 / 逐类

| 指标 | baseline | route_A | route_B | A−base | B−base |
|---|---:|---:|---:|---:|---:|
| fg-slice mDice | 0.760831 | 0.788292 | 0.790968 | **+0.027461** | **+0.030137** |
| fg-slice mIoU | 0.624421 | 0.661599 | 0.666274 | +0.037178 | +0.041853 |
| class mDice | 0.737925 | 0.764256 | 0.766683 | +0.026331 | +0.028758 |
| class mIoU | 0.602892 | 0.636955 | 0.641849 | +0.034063 | +0.038956 |

### 次要结果检查：逐卷物理指标

| 指标 | baseline | route_A | route_B |
|---|---:|---:|---:|
| volume mDice | 0.791974 | 0.830261 | 0.834922 |
| volume mIoU | 0.655977 | 0.710507 | 0.717580 |

### 逐病例一致性

| volume | baseline | route_A | route_B |
|---|---:|---:|---:|
| sub-verse022 | 0.794875 | 0.827703 | 0.840433 |
| sub-verse024 | 0.808805 | 0.851214 | 0.859278 |
| sub-verse071 | 0.811765 | 0.861138 | 0.867405 |
| sub-verse150 | 0.783093 | 0.817752 | 0.813229 |
| sub-verse264 | 0.761334 | 0.793497 | 0.794266 |

**route_A 5胜0负、route_B 5胜0负**（对 baseline）。两条统一路线对不统一的 baseline
是一致压制，不是病例互有胜负。

### 结论

1. **统一训推 init 有实质收益：+0.027~0.030 fg-slice mDice，逐病例 5/5 一致。**
   注意方向：baseline 是唯一留在其预训练原生 regime 里的臂，两条路线各只用 600 步
   适应新 init 分布，却仍然反超。因此该增益对路线是**保守**估计。

2. **这与 `DETECTOR_RECTANGLE_INIT_ABLATION_20260807` 的"打平"结论不矛盾。**
   那次只换推理侧、不重训，它的两臂**都是训推不一致**的（octagon 臂训练用 GT 极值点
   八边形、推理用 bbox 中点伪八边形；rectangle 臂训练同样是 GT 极值点八边形）。
   两个不一致互比 → 打平（+0.0012，3胜2负）。本次是"不一致 vs 已统一" → +0.027，5胜0负。
   **增益来自统一本身，不来自形状选择。**

3. **Route A vs Route B 不具决定性。** B 领先 +0.002676 fg-slice mDice；逐病例 4胜1负
   （150 由 A 胜）；逐类 17 类中 B 胜 10、A 胜 7，且 B 在 n 较大的 20–24 类上胜、
   A 在 n 较小的中胸段 9/10/11/13/14/18/19 上胜。5 例规模下这个差距不足以定论。

### 与外层状态采样的联动

`tools/volmem/analyze_outer_state_sampling.py`（200 万样本镜像
`flow_matching_evolution.py:2558-2640`）测得：推理每条轨迹必经的 `frac=0` 状态，
训练质量仅 **0.499%**（±0.02）；`frac ≤ 0.05` 仅 1.2447%。而 `v4_9_discrete_fractions`
里的 `1.0` 被 `.clamp_(0.0, 0.999)` 削成退化空转态，占全部样本 13.3%；连同 near_zero，
**28.35% 的样本进度 ≥ 0.95**，中位数 frac = 0.823，分布严重后倾。

单位混淆的铁证：代码自身的兜底式 `[1.0/(iter_steps-i) for i in range(iter_steps)]`
在 `iter_steps=3` 时正好是 `[0.3333, 0.5, 1.0]`，与 `v4_9_discrete_fractions` 一字不差——
一份**残差比例**表被塞进了按**绝对进度**消费的旋钮。三步迭代真正经过的进度应是
`{0, 0.3333, 0.6667}`。旁证：`1232_final_v5_geom_action_*` 系列里
`v4_9_discrete_fractions == v4_9_infer_target_fractions`，那才是正确纪律。

这解释了为何 init 形状选择（A vs B）影响小、而统一性影响大：init 形状只在 `frac≈0`
附近主导网络输入，而那里几乎没有训练质量。

### 尚未修复

**resample 链仍不一致**：训练 control→128 一步，推理 control→40→÷4→128 两步，
40 点中间态截角。契约测试中 `p128_maxabs == resample_only_maxabs` 精确相等——
两条路线的**全部**残差都来自这里（A 16.35 px，B 42.67 px），与 A/B 选择无关。
若修掉此项，两路线残差均归零，A 的索引对齐优势随之消失。

### 事故与坑（其他 AI 请注意）

- **`--max-volumes 5` 不是 dev5。** val 按 `case_id` 排序后前五个是
  010/011/013/016/018，其中 **010/011/013 是 locked**。dev5 实际在索引 5/7/14/23/31
  （022/024/071/150/264，切片 204+73+350+204+417 = 1248）。已给
  `tools/volmem/eval_memflowdit_v03.py` 增加 fail-closed `--volume-ids` 白名单。
- **locked 集不再是零访问。** 本次冒烟测试用 `--max-volumes 1` 读到了
  `sub-verse010` 一次（只读评测，其数值未进入任何结论）。
  `DETECTOR_RECTANGLE_INIT_ABLATION_20260807` 记载的"locked 010/011/013：访问数 0"
  自本次起不再成立。

### 机器结果

- `data/outputs/init_unify/eval_dev5_gtbox_step600/{baseline,route_A,route_B}/summary.json`
- 汇总脚本：`tools/volmem/compare_init_unify_eval.py`
- 采样分布：`data/outputs/init_unify/quantification/outer_state_sampling.json`
- 三臂 checkpoint：`data/outputs/init_unify/init_unify_{arm}/checkpoints/step_000600.pt`
- 全部位于 worktree `DiffusionSnake-12-30-init-unify-20260808`
