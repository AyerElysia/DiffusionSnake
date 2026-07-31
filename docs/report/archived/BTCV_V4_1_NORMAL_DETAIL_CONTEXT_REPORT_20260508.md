# BTCV V4.1 Normal Detail Context 说明报告

本文解释 `v4_1_network_diagram.png` 里 **normal detail context** 是怎么实现的，以及它在 V4.1 FM 网络里的作用。

## 1. 结论先行

V4.1 不是重新设计一套新主干，而是：

1. 复用 V3.4-FM 主体。
2. 在轮廓点附近增加 **normal detail context**。
3. 再加一个更保守的 per-point delta head。

其中 **normal detail context** 的本质是：

> 沿轮廓法线方向，在点的两侧做多尺度采样，用“内外侧特征差”来描述边界附近的局部几何和语义变化。

这比只看轮廓点本身更容易判断“边界往哪边修、修多少”。

---

## 2. 配置入口

V4.1 在配置里直接打开了这个分支：

- `v4_1_use_detail_context: true`
- `v4_1_detail_context_mode: 'normal'`
- `v4_1_use_p3_features: true`
- `v4_1_use_per_point_delta: true`
- `v4_1_per_point_delta_scale: 0.10`
- `v4_1_per_point_delta_reg_weight: 0.0002`

对应文件：

- `configs/btcv_diffusion_dit_v4_1_fm_detail_curv_gpu4.yaml`

---

## 3. normal detail context 具体怎么做

核心实现不在 denoiser 里，而是在 `FlowMatchingEvolution.sample_detail_features()`。

### 3.1 先算切线，再旋转成法线

对轮廓点 `img_poly`，先取前后邻点：

```python
prev_pt = torch.roll(img_poly, 1, dims=1)
next_pt = torch.roll(img_poly, -1, dims=1)
tangent = next_pt - prev_pt
tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-6)
normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
```

也就是：

1. 用相邻点估计切线。
2. 把切线旋转 90° 得到单位法线。

### 3.2 根据轮廓尺度自适应采样半径

采样半径不是固定像素，而是跟轮廓大小走：

```python
contour_scale = max(span_x, span_y)
radius_1 = clamp(contour_scale / 64, 0.75, 2.0)
radius_2 = clamp(contour_scale / 32, 1.5, 4.0)
```

这意味着：

- 小目标不会采样太远。
- 大目标会自动拉开 normal 上的采样距离。

### 3.3 在法线两侧取特征

用 `snake_gcn_utils.get_gcn_feature()` 在 CNN feature map 上做双线性采样：

- `poly + normal * r1`
- `poly - normal * r1`
- `poly + normal * r2`
- `poly - normal * r2`

这些点分别代表轮廓两侧、近距离和远距离的局部上下文。

### 3.4 组织成 detail feature

`normal` 模式下，代码最终拼接 3 组特征：

```python
detail_terms = [
    plus_1 - minus_1,
    plus_2 - minus_2,
    0.5 * (plus_1 + minus_1) - sampled_feat,
]
```

含义是：

1. **`plus_1 - minus_1`**：近距离法线两侧的对比。
2. **`plus_2 - minus_2`**：远一点的法线两侧对比。
3. **`midpoint - sampled_feat`**：两侧中点与轮廓点本身的差异。

最终 `normal` 模式输出的是 **3 倍通道** 的 detail feature，所以配置里 `detail_feature_dim = feature_dim * 3`。

---

## 4. 这些 detail feature 怎么进模型

V4.1 的 denoiser 继承自 V3.4 主干，detail feature 进来后做两次注入：

1. `detail_local_proj(detail_ctx)` 加到 `local_ctx`
2. `detail_point_proj(detail_ctx)` 加到点特征 `x`

对应文件：

- `lib/networks/diffusion/dit_denoiser_v3_4.py`
- `lib/networks/diffusion/dit_denoiser_v4_1.py`

这两个投影层都采用 **zero-init** 末层初始化，目的是：

- 热启动时不破坏 V3.4 已有能力
- 让 detail 分支先从“几乎不影响输出”开始学习

之后 6 层 DiT 采用 **global / local 交替上下文**：

- 偶数层走 global context
- 奇数层走 local context

---

## 5. 图里的 normal detail context 怎么理解

图上那个 “normal detail context” 可以理解为：

> 在每个轮廓点附近，沿法线两侧取一圈局部特征，再用这些特征差分去表达边界是否清晰、边界往哪边偏、偏多少。

它的作用不是单纯增加更多特征，而是把“边界方向信息”显式化：

- **法线方向**：告诉模型该往哪边修。
- **两侧特征差**：告诉模型边界是否真的在那里。
- **多尺度半径**：让模型同时看近边界和稍远一点的上下文。

所以这套设计更像是“边界几何提示器”，不是普通的 patch feature 拼接。

---

## 6. 为什么 V4.1 只保留 normal，而不是 normal_tangent

V4 系列里更早的版本有 `normal_tangent` / `normal_band` 之类的扩展，但 V4.1 收缩成了 `normal`。

原因很直接：

1. **更保守**：减少冗余上下文，避免过修正。
2. **更稳定**：法线两侧的局部差分已经足够表达边界。
3. **更适合 warm start**：V4.1 是在 V3.4 之上做增量增强，不追求花哨。

也就是说，V4.1 的重点不是“采样更多”，而是“采样更准、更稳”。

---

## 7. 输出端的配合

V4.1 最终输出不是只靠 shared head，而是：

1. shared velocity head
2. conservative per-point delta head
3. curvature reweight
4. small-displacement training

这和 normal detail context 是配套的：

- detail context 负责把边界局部信息喂进去。
- per-point delta 负责做小幅、点级别修正。
- curvature reweight 让高曲率边界更受重视。

---

## 8. 一句话总结

V4.1 的 **normal detail context** 就是：

> 先由轮廓邻点算法线，再沿法线正负方向做多尺度采样，把轮廓两侧的特征差拼成 detail feature，最后注入到 V3.4-FM 主干里，专门增强边界附近的小修正能力。

---

## 9. 参考文件

- `v4_1_network_diagram.png`
- `v4_1_architecture_diagram.png`
- `configs/btcv_diffusion_dit_v4_1_fm_detail_curv_gpu4.yaml`
- `lib/networks/diffusion/flow_matching_evolution.py`
- `lib/networks/diffusion/dit_denoiser_v3_4.py`
- `lib/networks/diffusion/dit_denoiser_v4_1.py`
- `lib/utils/snake/snake_gcn_utils.py`

