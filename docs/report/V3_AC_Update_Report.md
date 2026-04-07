# V3 A+C 更新报告

## 结论

本次完成了两项指定修改：

- A：将 `V3 / V3.1 / V3.2` 的训练超参数对齐到 `V2` 基线
- C：按当前真实训练初始化重新计算八边形位移统计，并切换到独立统计文件

同时，为避免 `V2` 继续误用八边形统计，本次也将 `V2` 系列拆分到独立的 box 统计文件。

---

## 实际修改

### 1. V3 系列超参数对齐

修改文件：

- `configs/btcv_diffusion_dit_v3.yaml`
- `configs/btcv_diffusion_dit_v3_1.yaml`
- `configs/btcv_diffusion_dit_v3_2.yaml`

修改内容：

- `train.lr: 1e-5 -> 5e-5`
- `train.batch_size: 32 -> 64`

### 2. 位移统计拆分

新增统计文件：

- `data/stats/btcv_disp_stats_box.json`
- `data/stats/btcv_disp_stats_octagon.json`

配置切换：

- `V3 / V3.1 / V3.2` 使用 `data/stats/btcv_disp_stats_octagon.json`
- `V2 / V2.1 / V2.2 / V2.3 / hybrid` 使用 `data/stats/btcv_disp_stats_box.json`

原因：

- 现在 `V2` 训练初始化是 box
- 现在 `V3` 训练初始化是 octagon
- 两者继续共用同一份统计，会把归一化范围弄混

---

## 统计结果

### Box 统计

文件：`data/stats/btcv_disp_stats_box.json`

```json
{
  "dx_min": -79.70527648925781,
  "dx_max": 56.91911315917969,
  "dy_min": -86.64105224609375,
  "dy_max": 45.46656799316406
}
```

### Octagon 统计

文件：`data/stats/btcv_disp_stats_octagon.json`

```json
{
  "dx_min": -57.31488037109375,
  "dx_max": 74.89698791503906,
  "dy_min": -46.23408126831055,
  "dy_max": 31.885009765625
}
```

可以看到，两种初始化的位移分布明显不同，继续共用旧统计是不合适的。

---

## 验证结果

### 1. 配置检查

已确认：

- `btcv_diffusion_dit_v2.yaml` -> `btcv_disp_stats_box.json`
- `btcv_diffusion_dit_v3.yaml` -> `btcv_disp_stats_octagon.json`
- `btcv_diffusion_dit_v3_1.yaml` -> `btcv_disp_stats_octagon.json`
- `btcv_diffusion_dit_v3_2.yaml` -> `btcv_disp_stats_octagon.json`

### 2. GPU 最小训练检查

在 GPU 上对 `V2 / V3 / V3.1 / V3.2` 各做了 1 次最小训练前向，全部通过：

| 配置 | 结果 | loss | 统计文件 |
|------|------|------|---------|
| `btcv_diffusion_dit_v2.yaml` | OK | `0.951693` | `data/stats/btcv_disp_stats_box.json` |
| `btcv_diffusion_dit_v3.yaml` | OK | `0.989540` | `data/stats/btcv_disp_stats_octagon.json` |
| `btcv_diffusion_dit_v3_1.yaml` | OK | `1.007608` | `data/stats/btcv_disp_stats_octagon.json` |
| `btcv_diffusion_dit_v3_2.yaml` | OK | `1.003498` | `data/stats/btcv_disp_stats_octagon.json` |

说明：

- 新统计文件可正常加载
- 新配置不会导致训练链路报错
- `V2` 没被这次修改破坏

---

## 本次涉及文件

### 配置

- `configs/btcv_diffusion_dit_v2.yaml`
- `configs/btcv_diffusion_dit_v2_1.yaml`
- `configs/btcv_diffusion_dit_v2_2.yaml`
- `configs/btcv_diffusion_dit_v2_2_hybrid.yaml`
- `configs/btcv_diffusion_dit_v2_3.yaml`
- `configs/btcv_diffusion_dit_v2_3_hybrid.yaml`
- `configs/btcv_diffusion_dit_v3.yaml`
- `configs/btcv_diffusion_dit_v3_1.yaml`
- `configs/btcv_diffusion_dit_v3_2.yaml`

### 数据统计

- `data/stats/btcv_disp_stats_box.json`
- `data/stats/btcv_disp_stats_octagon.json`

---

## 建议

下一步建议直接开始新的 `V3` 对照训练：

- 使用当前更新后的 `V3` 配置
- 从头训练，不接旧 checkpoint
- 与当前 `V2` 基线做公平比较

如果这一步之后 `V3` 仍明显落后，再进入下一轮：

- 最优循环对齐
- 起点无关训练

