# DiffusionSnake V3 代码审计与修复报告

## 概述

本报告记录了对 DiffusionSnake V3（八边形初始化版本）的全面代码审计结果。V3 基于 V2.0 升级，核心变化是将初始轮廓从矩形/菱形改为八边形（DeepSnake 风格），并引入了反转注意力流（Cross→Self）的 DiT Block。

审计发现了 **5 个关键 Bug**，均已修复并通过训练/推理测试验证。

---

## Bug 1：`get_octagon` 缺失边界夹紧（Critical）

**文件**: `lib/utils/snake/snake_decode.py` → `get_octagon()`

**问题**: 张量版本的 `get_octagon` 与参考 numpy 实现（`snake_voc_utils.py`）不一致。具体来说，12 个八边形顶点中的第 1 点和第 2 点缺少边界夹紧操作：

```python
# 修复前（错误）
ex[..., 0, 0] - w / x, ex[..., 0, 1],              # 点1：可能超出左边界
ex[..., 1, 0], ex[..., 1, 1] - h / x,              # 点2：可能超出上边界

# 修复后（正确，与 DeepSnake 参考一致）
torch.maximum(ex[..., 0, 0] - w / x, l), ex[..., 0, 1],  # 夹紧到左边界
ex[..., 1, 0], torch.maximum(ex[..., 1, 1] - h / x, t),  # 夹紧到上边界
```

**影响**: 缺少夹紧会导致八边形顶点超出检测框范围，产生不合法的初始轮廓。在小目标或极端宽高比的情况下尤为明显。训练时的位移向量（GT - init）会包含异常大的值，影响扩散模型的收敛。

**参考**: `snake_voc_utils.py` 第 375-384 行的 numpy 参考实现使用了 `max()` 和 `min()` 进行边界夹紧。

---

## Bug 2：V3 DiTBlock 注意力顺序错误（Critical）

**文件**: `lib/networks/diffusion/dit_blocks_v3.py` → `DiTBlockV3.forward()`

**问题**: V3 的设计文档（README_V3.md、文件头注释、类注释）明确定义了反转注意力流为 V3 的核心升级：

> "Key Upgrades from V2: Reversed Attention Flow (Cross-Attention -> Self-Attention)"

但实际实现仍然是 V2 的顺序（Self-Attention → Cross-Attention），且代码中的注释已经承认了这一点：

```python
# 修复前
# 1. 沿用 V2 设计：Self-Attention -> Cross-Attention  ← 与设计意图矛盾！
x_sa = modulate(self.norm2(x), shift_sa, scale_sa)
x = x + gate_sa.unsqueeze(1) * self._self_attention(x_sa)
x_ca = modulate(self.norm1(x), shift_ca, scale_ca)
x = x + gate_ca.unsqueeze(1) * self._cross_attention(x_ca, image_context)

# 修复后（Cross → Self，匹配设计文档）
x_ca = modulate(self.norm1(x), shift_ca, scale_ca)
x = x + gate_ca.unsqueeze(1) * self._cross_attention(x_ca, image_context)
x_sa = modulate(self.norm2(x), shift_sa, scale_sa)
x = x + gate_sa.unsqueeze(1) * self._self_attention(x_sa)
```

**设计理由**: Cross→Self 的顺序先通过交叉注意力从图像特征中定位边界信息，然后通过自注意力在 128 个轮廓点之间协调几何关系。这比 V2 的顺序更合理，因为自注意力的协调应该基于已经融入了图像信息的点特征。

---

## Bug 3：`use_dit_v2_1` 参数未传递（Critical）

**文件**: `lib/networks/diffusion/ct_snake.py` → `Network.__init__()`

**问题**: V3 配置文件指定了 `use_dit_v2_1: true`（启用 SpatialAnchorCompressor），注释也标注了"必开，防止 48GB OOM"。但 `ct_snake.py` 在调用 `make_evolution()` 时没有传递此参数：

```python
# 修复前：缺少 use_dit_v2_1, use_dit_v2_2, use_dit_v2_3
self.gcn = make_evolution(
    use_dit_v2=getattr(cfg, 'use_dit_v2', False),
    # ... 缺少 use_dit_v2_1 等参数
)

# 修复后：完整传递所有版本控制参数
self.gcn = make_evolution(
    use_dit_v2=getattr(cfg, 'use_dit_v2', False),
    use_dit_v2_1=getattr(cfg, 'use_dit_v2_1', False),
    use_dit_v2_2=getattr(cfg, 'use_dit_v2_2', False),
    use_dit_v2_3=getattr(cfg, 'use_dit_v2_3', False),
    # ...
)
```

**影响**: 由于 `DiffusionEvolution.__init__` 中 `use_dit_v2_1` 默认为 `False`，V3 的去噪器始终使用 PerceiverCompressor（256 个可学习查询）而非 SpatialAnchorCompressor（16×16 CNN 锚点池化）。这不仅浪费显存（可能导致 OOM），而且空间感知能力退化。

---

## Bug 4：位移归一化统计值与初始化形状不匹配（Critical）

**文件**: `compute_disp_stats.py`

**问题**: 位移归一化是将 `disp = GT - init` 映射到 [-1, 1] 范围。但统计计算脚本 **始终使用矩形初始化** 来计算位移范围：

```python
# 修复前（错误：始终用矩形）
gt_rect4 = snake_decode.get_box(gt_boxes)[0]
i_init_train_py = uniform_upsample(gt_rect4.unsqueeze(0), poly_num)[0]

# 修复后（使用数据集实际产生的初始轮廓）
i_init_train_py = init['i_it_py']  # 直接使用数据集的 octagon init
```

**统计对比**:

| 指标 | 旧值（矩形 init） | 新值（八边形 init） | 变化 |
|------|-------------------|-------------------|------|
| dx_min | -89.17 | -78.95 | +11.4% |
| dx_max | 55.80 | 54.70 | -2.0% |
| dy_min | -77.85 | -56.90 | +26.9% |
| dy_max | 49.46 | 33.42 | -32.4% |

八边形 init 更接近 GT，因此位移范围更小。使用错误的矩形统计值会导致归一化后的位移分布偏移，扩散模型在实际值域只使用了 [-1, 1] 的部分范围，降低了学习效率。

---

## Bug 5：代码质量问题（Minor）

**文件**: `lib/networks/diffusion/pretrain_evolution.py`

**问题**: `predict_eps()` 方法内部每次调用都重复导入 5 个模块：

```python
# 修复前：每次调用 predict_eps 都执行
from .dit_denoiser import DiTDenoiser
from .dit_denoiser_v2 import DiTDenoiserV2
# ... 重复导入

# 修复后：移至模块顶层，一次导入
```

其中 `DiTDenoiser` 和 `DiTDenoiserV2` 已在文件顶层导入，函数内的导入完全冗余。虽然 Python 的 import 机制会缓存已导入模块，但这仍是不必要的开销和代码冗余。

---

## 验证测试

### 训练测试
```
[DiffusionEvolution] Using DiT Denoiser V3 (Reversed Attention Flow) (layers=6, heads=8, dim=256)
Total params: 13,670,918 | Trainable: 13,670,902
Step 1/5: loss=1.004855
Step 2/5: loss=0.994450
Step 3/5: loss=0.992251
Step 4/5: loss=0.992949
Step 5/5: loss=1.000743
[OK] Training loop completed. Avg loss: 0.997050
```

### 推理测试
```
[*] Starting Batch Inference for 5 samples
[✔] Inference Complete! All results in: visual/v3_latest_eval
```

训练和推理均正常完成，无报错。

---

## 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `lib/utils/snake/snake_decode.py` | 修复 `get_octagon` 的 2 处缺失 clamp |
| `lib/networks/diffusion/dit_blocks_v3.py` | 修正注意力顺序为 Cross→Self |
| `lib/networks/diffusion/ct_snake.py` | 补充传递 `use_dit_v2_1/v2_2/v2_3` 参数 |
| `lib/networks/diffusion/pretrain_evolution.py` | 移除冗余函数内导入，提升至模块级 |
| `compute_disp_stats.py` | 使用数据集实际 init 轮廓计算统计 |
| `data/stats/btcv_disp_stats.json` | 更新为八边形 init 的正确统计值 |

---

## 建议

1. **重新训练 V3**: 由于上述 5 个 bug 同时存在，之前的训练结果不可靠。建议从头训练。
2. **监控位移分布**: 训练初期检查 `diff_loss` 是否从 ~1.0 开始稳步下降（纯噪声预测的 MSE 约为 1.0）。
3. **注意力可视化**: 建议添加注意力权重可视化，验证 Cross→Self 顺序确实改善了边界感知。
