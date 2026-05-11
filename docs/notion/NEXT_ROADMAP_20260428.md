# DiffusionSnake 下一步路线图

> 撰写时间：2026-04-28  
> 基于会话：V3.7 泛化诊断 + V4 FM 有效性分析  
> 前置文档：`V3.7_COPILOT_DIAGNOSIS_20260423.md`

---

## 一、背景：为什么要重新审视方向？

用户反馈："V4 FM 效果真的很不错"。

**V4 FM 是什么？**  
指 `scripts/train_v37_gen_v4.py`，即单样本 Flow Matching 训练脚本的第 4 次迭代。其核心贡献不是架构变化，而是一个很小的工程决策：

> **RAW MODE — 直接在原始位移空间操作，去掉 normalize/denormalize。**

消除归一化带来的误差放大，是让单样本 IoU 从停滞跳升到 97%+ 的关键转折点。

单样本 FM 脚本演化链：

```
gen.py      → 基线
gen_v2.py   → mixed x0
gen_v3.py   → trajectory loss + EMA
gen_v4.py   → RAW MODE（无归一化）← 关键拐点
gen_v5.py   → PerPointFinalLayer
gen_v5h.py  → multi-t 方差削减
gen_v6.py   → contour-balanced loss
gen_v6b.py  → scale conditioning + IoU 触发 LR decay → 97.38%
```

**泛化训练中的对应机制**：泛化配置里的 `diffusion_disp_norm: true` + `diffusion_disp_stats` 是数据集级别的标准化，这和 V4 RAW 的精神一致——用全局统计量而不是 per-sample 动态归一化。这对泛化更稳定。

---

## 二、当前流水线的检测器现状

### 2.1 架构链路

```
512×512 RGB
    ↓
YOLOv8-P2（lib/networks/YOLOV8/cfg/models/v8/yolov8-p2.yaml）
    ↓ 检测框 + P2 特征
P2 feature: stride=4, H/4×W/4 = 128×128 空间
Detect head 输出在 P2 处：reg_max×4 + nc = 64 + 9 = 73 channels
    ↓ cnn_proj（Conv2d 73→64, 1×1）
64-ch feature map（128×128）
    ↓ 逐点采样
DiT V3.7 Flow Matching 轮廓演化
```

### 2.2 YOLOv8-P2 的关键发现

**硬编码 + 无预训练**：

```python
# ct_snake.py line 22
yolo_yaml = 'lib/networks/YOLOV8/cfg/models/v8/yolov8-p2.yaml'
# ...所有 V3.7 configs 均未设置 load_yolo_pretrained，默认 false
```

每次训练均从随机初始化开始训练 YOLO。这是一个重要的性能瓶颈。

**检测失败（failed_samples）的含义**：  
`eval_v37_full_iou.py` 中 `failed_indices` 是 eval 时抛出异常的样本集合，不一定全是"无检测结果"，也可能包含：

- 确实无任何检测框（YOLO 漏检）
- 检测到的框全部被 NMS 滤掉
- 特殊输入导致数值异常（NaN 等）

以 `v6o_ep40_det` 结果（evaluated=125, failed=25）为例，在 `det_conf_thresh: 0.01` 已经极低的情况下仍有 25 例失败，说明存在 YOLO 漏检问题。

### 2.3 BTCV 检测的特殊挑战

| 挑战 | 描述 |
|------|------|
| 类别不平衡 | 9 个器官类尺寸差异极大（食管 vs 肝脏） |
| 小器官 | 食管、胆囊等在 512×512 下极小 |
| 低对比度 | CT 切片中相邻组织灰度相近 |
| 无预训练 | 从随机初始化在小数据集上训练，泛化弱 |

---

## 三、检测器升级方案

### 方案一：启用 YOLO 预训练权重（最低成本，立竿见影）

**现状**：所有 V3.7 配置的 `load_yolo_pretrained: false`，YOLO 从随机初始化训练。

**方案**：
1. 下载 `yolov8n-p2.pt` 或 `yolov8s-p2.pt` 预训练权重（COCO 预训练）
2. 在主配置添加：
   ```yaml
   yolo_pretrained: "data/pretrained/yolov8n-p2.pt"
   load_yolo_pretrained: true
   ```
3. 重新训练或从当前 checkpoint 继续训练

**预期收益**：显著减少漏检（failed_samples），提升小器官检测精度。  
**风险**：COCO 预训练的 nc=80，需要适配到 nc=9（仅更换 Detect head）。  
**难度**：★☆☆（`ct_snake.py` 已有预训练加载逻辑，只需配置正确权重路径）

---

### 方案二：升级到 YOLO11-P2（推荐）

**YOLO11 vs YOLOv8 关键改进**：

| 指标 | YOLOv8n-P2 | YOLO11n | 说明 |
|------|-----------|---------|------|
| Backbone block | C2f | C3k2 | C3k2 = 选择性 CSP，更强特征提取 |
| Attention | 无 | C2PSA@P5 | 增强全局感受野 |
| 参数量 | ~3.0M | ~2.6M | 更轻量 |
| 同等参数 mAP | - | +2-3 mAP | COCO 基准 |
| P2 支持 | ✅（已有） | 需新建 yaml | 参照 yolov8-p2.yaml 改 |

**所需改动**：

1. 在 `lib/networks/YOLOV8/cfg/models/v8/` 或 `yoloe/ultralytics/cfg/models/11/` 新建 `yolo11-p2.yaml`：
   ```yaml
   # 在 YOLO11 head 基础上，在 C3k2 P2 路径后插入 P2 检测分支
   # 参考 yolov8-p2.yaml 的 head 逻辑，C2f → C3k2
   ```
2. `ct_snake.py` 第 22 行改为 `yolo_yaml = 'path/to/yolo11-p2.yaml'`
3. 更新 `cnn_proj` 的 in_ch（若 reg_max 保持 16，64+nc 不变）

**预期收益**：提升小器官检测 mAP，间接提升 Snake 演化成功率。  
**难度**：★★☆（yaml 编写 + 少量 ct_snake.py 改动）

---

### 方案三：RT-DETR 替换（Transformer 检测器）

代码库中已有：`lib/networks/YOLOV8/cfg/models/rt-detr/rtdetr-l.yaml`（4 个变体）

**RT-DETR 优势**：
- AIFI（Attention-based Intra-scale Feature Interaction）注意力机制
- 更强的跨尺度特征融合
- 端到端无 NMS（降低复杂场景漏检）
- 对小目标和密集目标更友好

**关键挑战**：
- 标准 RT-DETR 从 P3 (stride=8) 开始，没有 P2 输出
- 需要在 RT-DETR backbone 上添加 P2 分支（stride=4 特征）
- HGStem backbone 与 `cnn_proj` 的通道对接需要适配

**可行路径**：
1. 使用 RT-DETR backbone 的 P2 层（HGStem 第一阶段输出）作为特征图
2. 独立的 Snake 特征 branch：RT-DETR backbone → P2 feature → cnn_proj
3. RT-DETR 检测头保持正常做检测，P2 特征单独引出做 Snake

**难度**：★★★（需要较大的架构改动，双分支设计）

---

### 方案四：YOLOE 开放集检测（最具探索价值）

代码库中已有 `yoloe/` 目录（YOLO11 + 开放集检测，ICCV 2025 论文）

**YOLOE 的额外能力**：
- 文本提示模式：可传入器官名称（"liver", "spleen"...）作为检测条件
- 视觉提示模式：用 reference image/contour 作为 visual prompt
- Prompt-free 模式：等效标准 YOLOv11

**对本项目的意义**：
- CT 扫描中器官检测是高度特定领域任务，使用器官名称文本提示可提升小器官召回
- 训练时可联合文本特征，提升跨患者泛化
- Re-parameterization 后，推理速度与 YOLOv11 相当

**挑战**：
- 集成复杂度最高（需要适配 text encoder + prompt 逻辑）
- YOLOE 使用独立 ultralytics fork，与当前 YOLOV8 代码库有分歧

**难度**：★★★★（集成工作量大，适合作为远期研究方向）

---

## 四、FM（Flow Matching）改进方向

### 4.1 将 V4 RAW Mode 精神推广到泛化训练

**现状**：泛化训练中使用 `diffusion_disp_norm: true` + stats 文件，是全局统计量归一化。

**优化点**：
- 当前 stats 文件 `btcv_disp_stats_octagon.json` 是基于八边形初始化的统计量
- 训练中实际初始化来自 YOLO 检测框，分布可能有偏移
- 建议：重新计算基于实际 YOLO 检测初始化的 displacement stats，更精确地校准 FM 的输入分布

### 4.2 V3.10 配置 Bug（已记录）

参见 `V3.7_COPILOT_DIAGNOSIS_20260423.md`。  
V3.10 配置中 `use_dit_v3_1: true` 未被架构选择链识别，静默降级到 V3.2。  
**修复**：改为 `use_dit_v3_7: true`。

### 4.3 Curvature Reweight Loss（免费收益）

所有 V3.7 配置均有：
```yaml
v3_7_use_curvature_reweight: false  # 硬关闭
```

此 loss 会在高曲率轮廓段（拐弯处）增加梯度权重，正是 v6b 泛化失败的核心位置。  
**建议**：开启 `v3_7_use_curvature_reweight: true` 并测试，这是对"复杂拐弯泛化差"问题的直接响应。

### 4.4 Detail Context Coverage（局部感受野）

`v3_7_detail_context_coverage` 当前最优是 `0.04`（`v6o` 实验）。  
但对于复杂轮廓，更小的 coverage 可能导致上下文窗口内缺少足够信息。  
**建议**：测试 `coverage: 0.06`（稍大局部窗口）+ `curvature_reweight: true` 组合。

---

## 五、其他高优探索方向

### 5.1 自适应点数（V3.10 方向）

V3.10 的核心思想（自适应点数）仍然有效，即根据轮廓复杂度动态分配控制点，让复杂拐弯区域获得更多点数。

但 V3.10 目前有以下问题：
- 架构 bug（use_dit_v3_1 未识别）
- 配置中自适应点数逻辑未完整实现（还在探索阶段）

**建议**：在 V3.7 架构基础上实现自适应点数，不要单独维护 V3.10 分支。

### 5.2 多尺度特征融合

当前只使用 P2（stride=4）特征。对于大器官（肝脏），P3（stride=8）的语义信息可能更有帮助；对于小器官（食管），P2 甚至 P1 可能更重要。

可以探索：
- Multi-scale feature injection：在 DiT V3.7 的特征采样阶段融合 P2+P3
- 代码位置：`flow_matching_evolution.py` 的 `sample_detail_features`

### 5.3 Iterative Refinement Steps

当前 V7_v6bq（最优）使用 `use_iterative_refinement: true`，IoU=0.8932。  
但 ODE steps 为 `flow_ode_steps: 12`。探索 steps=20 或 steps=25 是否进一步提升精度（主要看推理速度可接受程度）。

---

## 六、优先级行动建议

### 立即可做（低代价，高收益）

| 优先级 | 任务 | 说明 | 预期收益 |
|--------|------|------|----------|
| P0 | 开启 `curvature_reweight: true` | 直接针对复杂拐弯问题 | 中等，对漏 turns 区域有直接作用 |
| P0 | 测试 V8 config 最优组合 | query mode + optimal_cyclic_align + iterative + noise_scale=1.0 | 当前未完整组合测试 |
| P1 | 修复 V3.10 架构 bug | use_dit_v3_7 代替 use_dit_v3_1 | 使 V3.10 实际运行期望架构 |
| P1 | 启用 YOLO 预训练初始化 | 下载 COCO P2 权重 + 配置修改 | 减少 failed_samples，提升小器官检测 |

### 中期探索（1-2 周）

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P2 | 创建 YOLO11-P2 配置并测试 | 参照 yolov8-p2.yaml + yolo11.yaml 混合 |
| P2 | 重新计算 YOLO-init displacement stats | 替代八边形 stats，更精确的 FM 归一化 |
| P2 | Multi-scale P2+P3 feature fusion | 改造 sample_detail_features |

### 长期研究（远期方向）

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P3 | RT-DETR P2 adaptation | 添加 P2 分支，双分支设计 |
| P3 | YOLOE text-prompt 集成 | 器官名称作为检测条件 |
| P3 | V3.10 自适应点数完整实现 | 在 V3.7 基础上动态点数分配 |

---

## 七、架构约束清单（换检测器必须检查）

以下是替换 YOLO 检测器时必须保持的架构约束：

```
1. P2 feature 输出必须在 stride=4（H/4 × W/4 = 128×128 for 512×512 input）
2. cnn_proj in_ch = reg_max×4 + nc = 64 + nc（9）= 73，换 detector 需对齐通道数
3. 检测输出格式：(yolo_y, yolo_feats) 其中 yolo_feats[0] = P2 feature
4. NMS pipeline：ct_snake.py Line 140-200，依赖 YOLO detection head 格式
5. det_conf_thresh / det_iou_thresh：保持配置项不变，只是 backbone/head 替换
```

---

## 八、关键文件索引

| 文件 | 用途 |
|------|------|
| `lib/networks/snake/ct_snake.py` | 检测器实例化（Line 22）、P2 特征提取（Line 150-155） |
| `lib/networks/YOLOV8/cfg/models/v8/yolov8-p2.yaml` | 当前检测器 yaml |
| `lib/networks/YOLOV8/cfg/models/rt-detr/rtdetr-l.yaml` | RT-DETR L 配置（可探索） |
| `yoloe/ultralytics/cfg/models/11/yolo11.yaml` | YOLO11 基础配置（无 P2） |
| `yoloe/ultralytics/cfg/models/v8/yolov8-p2.yaml` | YOLOE fork 中的 YOLOv8-P2 |
| `scripts/train_v37_gen_v4.py` | V4 FM 单样本训练（RAW MODE 参考实现） |
| `lib/networks/diffusion/flow_matching_evolution.py` | FM 核心，归一化逻辑在 forward() |
| `configs/btcv_diffusion_dit_v7_v6bq_full_noleak.yaml` | 当前最优泛化配置（0.8932 IoU） |
| `data/stats/btcv_disp_stats_octagon.json` | FM 归一化 stats（基于八边形初始化） |

---

*本文档由 GitHub Copilot 生成，供后续任务接手参考，无代码改动。*
