# 方向 B：检测分支从 YOLO P2 解耦方案

日期：2026-05-06

---

## 背景与问题

当前架构中，YOLO backbone 的 P2 特征（stride=4）同时服务于两个目标：

1. **检测输出**（v8DetectionLoss 的梯度回传到 backbone）
2. **Snake GCN 输入**（ex/py loss 的梯度也回传到 backbone）

两个方向互相干扰。主训练设 `det: 0` 后 YOLO 完全不更新，检测退化为随机初始化的 COCO pretrained head。此外还有两个已知 bug：
- `ct_snake.py:397`：`rotated=True`（OBB 旋转框标志，BTCV 是轴对齐 bbox，应为 `False`）
- `heads: {'ct_hm': 9}`：数据只有 mask_1~mask_8 共 8 类，nc=9 多一个空类

---

## 核心思路

```
Input → YOLO backbone+neck → P2 feature (stride=4)
                                   /              \
                          (normal grad)        (detached)
                                ↓                   ↓
                     cnn_proj(p2)         LightDetHead(p2.detach())
                                ↓                   ↓
                     cnn_feature → GCN       ct_hm + wh → boxes
                     (Snake/Diffusion)       (FocalLoss + IndL1)
```

- **YOLO backbone**：梯度只来自 ex / py / diffusion loss，P2 朝 Snake 边界方向优化
- **LightDetHead**：独立轻量 heatmap head，只接受 detached P2，只被 det loss 更新
- 两个目标完全解耦，可以同时训练，互不干扰

---

## P2 通道数说明

`yolo_feats[0]` 是 Detect 头处理后的 P2 输出，通道数固定为 `reg_max * 4 + nc = 64 + nc`，
与 YOLO scale（n/s/m）无关。现有 `cnn_proj` 的 `in_ch = 64 + nc` 就是这个。
`LightDetHead` 的 `in_channels` 直接复用这个值即可。

---

## 具体改动清单

### 改动 1：ct_snake.py — 新增 LightDetHead 类

位置：`HeatmapResNetDetector` 之后、`Network` 类之前插入。

```python
class LightDetHead(nn.Module):
    """轻量 heatmap 检测头，接受 detached P2 特征（64+nc 通道），不回传梯度到 backbone。"""
    def __init__(self, in_channels, num_classes, head_conv=128):
        super().__init__()
        self.ct_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes, 1, bias=True),
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, 1, bias=True),
        )
        nn.init.constant_(self.ct_head[-1].bias, -2.19)  # focal loss 初始化

    def forward(self, p2_detached):
        ct_hm = net_utils.sigmoid(self.ct_head(p2_detached))
        wh    = F.relu(self.wh_head(p2_detached))
        return ct_hm, wh
```

### 改动 2：ct_snake.py — Network.__init__ 中注册 LightDetHead

在 `self.cnn_proj = nn.Conv2d(in_ch, 64, ...)` 之后加：

```python
self.use_light_det = bool(getattr(cfg, 'use_detached_det_head', False))
if self.use_light_det:
    nc_det    = int(getattr(cfg, 'det_head_classes', 8))
    head_conv = int(getattr(cfg, 'det_head_conv', 128))
    self.light_det_head = LightDetHead(
        in_channels=in_ch,   # 与 cnn_proj 一致，= 64 + nc
        num_classes=nc_det,
        head_conv=head_conv,
    )
```

### 改动 3：ct_snake.py — Network.forward 中替换检测逻辑

在 `cnn_feature = self.cnn_proj(p2)` 之后，原有 NMS/decode 逻辑之前，插入分支：

```python
if self.use_light_det:
    ct_hm, wh = self.light_det_head(p2.detach())
    ct, raw_det = self.decode_detection_from_heatmap(ct_hm, wh)
    detection = self.filter_detection_candidates(raw_det)
    output = {
        'detection': detection,
        'ct':        ct,
        'ct_hm':     ct_hm,   # 供 FocalLoss 使用
        'wh':        wh,      # 供 IndL1Loss 使用
        'feat_hw':   (h, w),
    }
    if getattr(cfg, 'use_gt_det', False):
        self.use_gt_detection(output, batch)
    if not self.freeze_snake:
        output = self.gcn(output, cnn_feature, batch)
    # 注意：use_light_det 时不暴露 yolo_preds，不走 v8DetectionLoss
    return output
# else: 继续走原有 YOLO decode + NMS 路径
```

同时修复 NMS bug（在 else 分支里）：
```python
rotated=False,   # 原来是 True，BTCV 是轴对齐 bbox
```

### 改动 4：diffusion_trainer.py — det loss 切换为 heatmap 损失

在 `DiffusionPretrainNetworkWrapper.__init__` 里，检测到 `use_detached_det_head=True` 时：

```python
if getattr(cfg, 'use_detached_det_head', False):
    self.det_crit  = None   # 不再用 v8DetectionLoss
    self.ct_crit   = net_utils.FocalLoss()
    self.wh_crit   = net_utils.IndL1Loss1d('smooth_l1')
    self.heatmap_wh_weight = float(getattr(cfg, 'det_head_wh_weight', 0.1))
```

在 `forward` 里替换 det loss 计算段：

```python
if getattr(cfg, 'use_detached_det_head', False) and 'ct_hm' in output:
    ct_loss  = self.ct_crit(output['ct_hm'], batch['ct_hm'].to(device))
    wh_loss  = self.wh_crit(output['wh'], batch['wh'].to(device),
                             batch['ct_ind'].to(device), batch['ct_01'].to(device))
    det_loss = ct_loss + self.heatmap_wh_weight * wh_loss
    det_weight = float(self.loss_scales.get('det', 0.3))
    loss = loss + det_weight * det_loss
    scalar_stats.update({'det_ct': ct_loss, 'det_wh': wh_loss, 'det_loss': det_loss})
```

### 改动 5：Config 新增开关

在主 config（`btcv_diffusion_dit_v3_4_fm_full_noleak.yaml`）中修改：

```yaml
# 启用解耦检测 head
use_detached_det_head: true
det_head_classes: 8          # BTCV 实际类数（mask_1~mask_8）
det_head_conv: 128
det_head_wh_weight: 0.1

# nc 修正（原来是 9，实际数据只有 8 类）
heads: {'ct_hm': 8, 'wh': 2}

# 恢复 det loss（现在不影响 backbone，可以打开）
loss_scales: {det: 0.3, ex: 1.0, py: 1.2}
```

---

## 训练策略

- **不需要两阶段**：LightDetHead 是新增层，随机初始化，直接从已有 V3.4-FM checkpoint 继续训练
- `det: 0.3` 足够，heatmap head 小，收敛比 YOLO 快得多
- YOLO backbone 梯度只来自 ex/py/diffusion，P2 特征专注于 Snake 边界
- 如果 LightDetHead 收敛后 recall 还不够，可调大 `det` 权重至 0.5

---

## 优势

| 点 | 说明 |
|----|------|
| 解耦彻底 | 两个优化目标梯度完全隔离 |
| 无需额外阶段 | 直接在主训练中一起跑 |
| 参数量极小 | LightDetHead ≈ 1M 参数，训练快且稳定 |
| 兼容现有代码 | `use_detached_det_head: false` 时退回原路径 |
| 复用已有基础设施 | FocalLoss / IndL1Loss / decode_detection_from_heatmap 都已实现 |

---

## 潜在风险与注意事项

1. **heatmap 检测上限低于 YOLO anchor-free**
   - 对 recall 通常足够，但极细小器官（胆囊等，class 2 只在 34% 切片出现）可能 miss
   - 补救：对 ct_hm 用更低 conf_thresh（当前已是 0.01）

2. **P2 完全无 det 梯度，backbone 可能缺少定位语义**
   - 补救（软解耦模式）：去掉 `p2.detach()`，同时保留 `v8DetectionLoss` 但权重极小（`det: 0.05`）
   - 这是从"完全解耦"到"加权多目标"的中间态，根据实验结果判断是否需要

3. **batch['ct_hm'] 尺寸对齐**
   - 需确认 `batch['ct_hm'].shape == (B, 8, H/4, W/4)`，与 LightDetHead 输出一致
   - 如果原来 ct_hm 是按 nc=9 生成的，需要同步修改数据 pipeline 的类数

---

## 可选扩展：软解耦模式

如果完全 detach 后 backbone 定位能力不足，可以用两路并行：

```python
# LightDetHead 不 detach（让极小梯度传回 backbone）
ct_hm, wh = self.light_det_head(p2)   # 不 detach

# v8DetectionLoss 也保留但权重极小
loss_scales: {det: 0.05, ex: 1.0, py: 1.2}  # v8DetectionLoss 极小权重
det_head_weight: 0.25                         # heatmap head 独立权重
```

---

## 实施顺序建议

1. 先修两个 bug（`rotated=False`，nc=8），验证现有 pipeline 不受影响
2. 实现 `LightDetHead` 和 `use_detached_det_head` 开关
3. 从现有 checkpoint 继续，观察 `det_ct` / `det_wh` loss 下降情况
4. 约 50~100 epoch 后，用 `use_gt_det: false` 评估检测 recall
5. 根据结果决定是否切换到软解耦模式
