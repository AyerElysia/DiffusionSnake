# DiffusionSnake V3 推理对齐接手指南 (Handoff)

## 1. 核心目标
将 **DiffusionSnake V3** 的推理逻辑与训练状态进行 100% 几何对齐，解决 V3 版本初期出现的预测不稳、轮廓偏离和收敛缓慢问题。

---

## 2. V3 核心变更对比 (Alignment Matrix)

| 特性 | V1 / V2 | V3 (当前版本) | 备注 |
| :--- | :--- | :--- | :--- |
| **初始化轮廓** | 矩形/菱形 (Rect/Quad) | **八边形 (Octagon)** | DeepSnake 风格，更贴合医学图像目标 |
| **几何点数** | 128 点 | **128 点** | 初始化从 12 点插值到 128 点 |
| **DiT 架构** | Self → Cross (V2) | **Cross → Self (V3)** | 先聚合图像信息，再协调点间几何关系 |
| **视觉压缩** | Perceiver (256 queries) | **SpatialAnchor (16x16)** | 显存消耗降低 ~40%，空间感增强 |
| **位置编码** | SnakePosEnc / RoPE | **CyclicRoPE** | 循环旋转位置编码，处理闭环拓扑 |

---

## 3. 已完成的关键对齐 (Critical Fixes)

截止到 **2026-04-04**，以下关键 Bug 已修复：

1.  **极点提取对齐 (`infer_v3_refinement.py` / `scripts/infer_v3_final.py`)**:
    - **逻辑**: 推理时不再直接用 YOLO Box 四角，而是从原始多边形中提取 True Min/Max (T, L, B, R)。
    - **偏移**: 引入 `+0.5` 像素偏移，确保在输入特征图（Stride=4）中心对齐。
2.  **八边形构造修复 (`snake_decode.py`)**:
    - **Bug**: 原实现缺少边界 Clamp（第 1, 2 点）。
    - **修复**: 在 `get_octagon` 中强制执行 `torch.clamp` 或 `torch.max/min`，防止顶点飞出。
3.  **位移统计更新 (`btcv_disp_stats.json`)**:
    - **Bug**: 之前一直沿用矩形初始化的均值/方差。
    - **修复**: 重新计算了八边形初始化的 `disp = GT - init` 统计量。
    - **现状**: 八边形位移范围比矩形更集中，学习效率更高。
4.  **DiT 权重加载 Bug**:
    - **修复**: 解决了 `ct_snake.py` 中 `use_dit_v2_1` 等版本参数未透传给去噪器底层的缺陷。

---

## 4. 快速接手实验 (Verification)

接手后请立即运行以下脚本验证对齐状态：

### Step 1: 几何对齐验证
```bash
python verify_octagon_v3.py
```
- **检查点**: 查看 `visual/octagon_comparison.png`。黄色八边形必须完美包裹蓝色 GT 极点，且红色采样点分布均匀。

### Step 2: 推理流程验证
```bash
python infer_v3_refinement.py --ckpt data/outputs/btcv_diffusion_dit_v3/checkpoints/latest.pt
```
- **检查点**: 查看 `visual/v3_clean_eval/`。如果预测轮廓呈现合理的凹凸细节而非大范围偏离，说明对齐成功。

---

## 5. 待办事项 (Next Steps)

1.  **重新启动 V3 训练**: 由于之前的训练脚本存在参数透传 Bug，历史权重不可信。必须基于修复后的代码重新 Train。
2.  **监控位移分布**: 在 `diffusion_train.py` 中观察 `diff_loss`。初始 Loss 应该在 `~1.0` 左右（MSE 标准差归一化后），若 Loss 极大则说明归一化统计值仍有误。
3.  **端到端连接**: 验证 YOLO 检测出来的 Box 喂给 `get_octagon` 后是否稳定。

---

## 6. 关键文件索引
- **去噪器定义**: `lib/networks/diffusion/dit_denoiser_v3.py` (Cross→Self 注意力流)
- **初始化逻辑**: `lib/utils/snake/snake_decode.py` -> `get_octagon`
- **训练流程**: `diffusion_train.py` (单阶段联合训练)
- **统计报告**: `docs/report.md` (包含详细的 Bug 审计记录)

---
**Maintainer**: Antigravity AI
**Date**: 2026-04-04
