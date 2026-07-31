# LocateAnything → DiffusionSnake 检测接入报告

日期：2026-07-31

## 1. 目标与边界

本次工作将 LocateAnything 的离线检测结果接入：

`/home/medteam/Zhrch/DiffusionSnake-12-30`

接入仅发生在 detector 输出与 contour initializer 之间：

```text
LocateAnything raw response
  -> canonical cache
  -> external_detection [B,N,6]
  -> 既有 bbox-to-contour initializer
  -> 既有 Snake / Flow Matching / MemFlowDiT 演化
```

没有修改 Flow Matching、Diffusion、Snake GCN、MemFlowDiT 速度场或 Memory 算法。

## 2. 已实现合同

`external_detection` 固定为：

```text
[x1, y1, x2, y2, score, class_id]
```

- shape：`[B,N,6]`；
- 坐标：当前网络输入像素空间；
- score：有效检测必须严格 `>1e-4`；
- class_id：0-based，范围 `[0,24]`，对应 C1–C7、T1–T12、L1–L6；
- padding：必须为全零行；
- 缺 cache 默认报错，不做 basename 模糊匹配；
- 训练态禁止使用 cached detection，直到实现同类+IoU+一对一匹配。

## 3. 修改位置

- `volmem/detection/locany_cache.py`
  - canonical cache loader；
  - class_id 严格校验；
  - score/尺寸门禁、class-aware NMS；
  - 翻转与 `trans_input` 仿射映射；
  - padded `[B,N,6]` 构建。
- `volmem/adapters/v4_6c.py`
  - 可选 cache provider；
  - 评估态 batch 注入；
  - 训练态 fail-closed。
- `lib/networks/snake/ct_snake.py`
  - 三种 detector backend 在演化前统一调用 `apply_external_detection()`；
  - 校验有限值、正面积、score、padding、class_id；
  - 不改变 detector feature 与 contour evolution。
- `lib/utils/snake/snake_gcn_utils.py`
  - 只修复检测输入边界：在 `snake_decode.get_init()` 前过滤 dense padding；
  - 全空 batch 返回有形状的空 tensor；
  - 不修改后续演化。
- `tools/volmem/eval_prototype.py`
- `tools/volmem/eval_memflowdit_v03.py`
- `tools/volmem/eval_memflowdit_v03_viz.py`
  - 新增 `--box-source detector|locany_cached`；
  - 新增 `--locany-cache-path`。
- `tools/volmem/normalize_locany_cache.py`
  - 将 class-aware raw JSONL 规范化为 `volmem_locany_cache_v1`。

## 4. 关键发现

### 4.1 当前 1500-step 新基线不能直接接入

`outputs/eval_locany_v3a_tier_a_base_1500_quick200/predictions.jsonl` 的 742 个框全部只有通用标签 `vertebra`，没有绝对 `class_id`。它是更好的椎体几何检测器，但目前不是可供 25 类 Snake 直接消费的绝对编号检测器。

因此 adapter 明确拒绝：

- 裸 `<box>`；
- 通用 `<ref>vertebra</ref>`；
- 缺失 class_id 的结构化实例；
- 越界或非整数 class_id。

禁止把这些框伪装成 class 0。

### 4.2 已有 class-aware checkpoint 可以生成 C1–L6 标签，但质量仍不足

历史 checkpoint-20000 的完整测试：

- class-aware IoU@0.25 F1：0.3906；
- precision：0.4142；
- recall：0.3696；
- FP/image：6.87。

它能形成合法 canonical cache，但不能被当作成熟生产检测器。当前正确路线是保留几何更强的 1500-step 主线，并单独解决绝对编号，而不是退回旧 class-aware 模型作为最终方案。

### 4.3 真实 canonical 样例触发了正确质量门禁

真实样例含两个框，其中一个约 1.16×0.76 px，被 `min_box_side=1.0` 拒绝；另一个 C1 框成功映射为 `[B,1,6]`。这证明 adapter 没有盲目把历史微小伪框送入 Snake。

## 5. 验证结果

- Python 编译检查：通过；
- `git diff --check`：通过；
- cache/坐标/class合同：40/40 tests 通过；
- adapter/padding/空 batch合同：10/10 tests 通过；
- 真实 canonical affine smoke：通过；
- Flow Matching、Snake evolution、MemFlowDiT 实现文件：本次未修改。

安全备份分支：

```text
backup/pre-locany-integration-20260731-1931
```

## 6. 启用方式

评估时使用：

```bash
python tools/volmem/eval_memflowdit_v03.py \
  --cfg_file <config.yaml> \
  --ckpt <checkpoint.pt> \
  --box-mode predicted \
  --box-source locany_cached \
  --locany-cache-path <volmem_locany_cache_v1.json> \
  --result-dir <output_dir>
```

默认 `box_source=detector`，因此不启用时保持原行为。

## 7. 当前结论

代码层接入已经完成，且没有触碰演化算法。生产层仍差一项：需要一个同时满足“1500-step 几何质量”和“C1–L6 绝对编号”的 canonical cache。当前不能用通用 `vertebra` 输出伪造类别，也不应把旧 class-aware checkpoint 当最终模型。