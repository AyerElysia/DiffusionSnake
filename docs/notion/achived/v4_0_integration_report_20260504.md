# V4.0 Integration Report (2026-05-04)

## 1. Goal
在现有 V3.4-FM 训练体系上，落地一个可继承旧权重、面向细节边界增强的整合方案（V4.0），并完成可运行验证与整套测试集 IoU 评估。

## 2. Main Design Decisions
V4.0 没有推翻主干，而是采用“可热启动增量改造”：

1. 保留 V3.4-FM 主体结构，尽可能继承原有参数。
2. 新增多尺度信息入口：在 Snake 分支启用 P3 融合（零初始化残差注入）。
3. 新增细节上下文：复用现有法线/切线 detail feature 采样链路并注入 denoiser。
4. 新增 per-point delta head：在共享输出头之上叠加点位专属残差分支（零初始化）。

核心原则：
- 旧能力尽量不破坏。
- 新能力从零增量学习。
- 首轮优先验证“能稳定继承 + 能跑通 + 是否提升 IoU”。

## 3. Code and Config Changes
### 3.1 New file
- lib/networks/diffusion/dit_denoiser_v4.py

主要内容：
- DiTFlowMatchingV4 类（继承 DiTDenoiserV3）
- PerPointDeltaHead（零初始化、可选 L2 正则）
- detail_local_proj/detail_point_proj（零初始化）

### 3.2 Modified files
- lib/networks/diffusion/flow_matching_evolution.py
  - 新增 use_dit_v4 分支
  - 新增 v4 相关配置读取与 denoiser 构建
  - 统一 detail context 开关与 mode 选择
- lib/networks/snake/ct_snake.py
  - 将 v4_use_p3_features 纳入 P3 特征融合开关
- lib/config/config.py
  - 新增 V4 开关和默认值：
    - use_dit_v4
    - v4_use_p3_features
    - v4_use_detail_context
    - v4_detail_context_mode
    - v4_use_per_point_delta
    - v4_per_point_delta_scale
    - v4_per_point_delta_reg_weight
- lib/utils/snake/snake_config.py
  - 将 use_dit_v4 纳入 v3 family 判断

### 3.3 New training config
- configs/btcv_diffusion_dit_v4_fm_multiscale_detail_gpu67.yaml

关键配置：
- use_dit_v4: true
- v4_use_p3_features: true
- v4_use_detail_context: true
- v4_detail_context_mode: normal_tangent
- v4_use_per_point_delta: true
- v4_per_point_delta_scale: 0.25
- v4_per_point_delta_reg_weight: 0.0005
- gpus: [6, 7]
- resume_weights_only: true

## 4. Runtime Issues and Fixes
### 4.1 Initialization bug in V4 denoiser
问题：DiTFlowMatchingV4 初始化时访问 self.num_points 报错。

修复：
- 在 DiTFlowMatchingV4.__init__ 中显式接收 num_points
- 传递给父类并保存 self.num_points

结果：
- 语法检查通过
- 网络可实例化并进入训练流程

### 4.2 Torchrun port conflict
问题：首次使用 29624 启动报 Address already in use。

处理：
- 切换为 29634

结果：
- V4 训练成功启动

## 5. Weight Inheritance Verification
使用训练脚本自身的“部分加载 + 形状重叠复制”逻辑验证（不是直接 strict load）。

日志关键结果：
- matched_keys=578
- exact_match_keys=260
- partial_copy_keys=318
- matched_params=13,587,920
- exact_match_params=10,815,328
- partial_copy_params=2,772,592
- missing_after_load=14
- unexpected_ckpt_keys=0

missing 主要集中在新增模块：
- detail_local_proj
- detail_point_proj
- per_point_delta_head

结论：
- 达到“旧权重尽量继承，新模块增量学习”的目标。

## 6. Inference and Full-Test IoU Evaluation
### 6.1 Single sample inference (latest V4 snapshot)
- 脚本：scripts/infer_single_sample.py
- 配置：configs/btcv_diffusion_dit_v4_fm_multiscale_detail_gpu67.yaml
- 权重快照：/tmp/v4_fm_latest_snapshot_20260504.pt
- 后处理：REMOVE_EXTREME_POINTS=0
- 样本：index 25
- 输出图：
  - visual/v4_fm_latest_now/v4_fm_latest_no_post_20260504_idx25_epoch1160.png

### 6.2 Full test IoU (150 samples)
- 脚本：scripts/eval_v37_full_iou.py
- 配置：configs/btcv_diffusion_dit_v4_fm_multiscale_detail_gpu67.yaml
- 权重：/tmp/v4_fm_latest_snapshot_20260504.pt
- SAVE_VISUALS=0
- 输出：
  - visual/v4_fm_eval_latest_20260504/v3_7_full_test_iou_20260504_182952.json
  - visual/v4_fm_eval_latest_20260504/summary_rows_20260504_182952.json

指标：
- mean_iou_sample_avg: 0.889896
- mean_iou_contour_avg: 0.886269
- median_iou_sample_avg: 0.890278
- std_iou_sample_avg: 0.027400
- failed_samples: 0

## 7. Comparison with Previous V3.4-FM Final
已知 V3.4-FM 最终测试集结果：
- mean_iou_sample_avg: 0.880472
- mean_iou_contour_avg: 0.876655

V4.0 最新评估相对提升：
- sample avg: +0.009424
- contour avg: +0.009614

结论：
- 提升幅度不是随机波动级别，整合方向有效。

## 8. Current Artifacts Summary
- V4 训练输出目录：
  - data/outputs/btcv_diffusion_dit_v4_fm_multiscale_detail_gpu67_reusemax
- V4 单图可视化：
  - visual/v4_fm_latest_now/v4_fm_latest_no_post_20260504_idx25_epoch1160.png
- V4 整套测试集评估：
  - visual/v4_fm_eval_latest_20260504/v3_7_full_test_iou_20260504_182952.json
  - visual/v4_fm_eval_latest_20260504/summary_rows_20260504_182952.json

## 9. Recommended Next Steps
1. 做 V3.4 vs V4.0 的样本级误差对比，重点筛选高曲率失败样本。
2. 在 V4.0 上推进 V4.1，优先聚焦高曲率点位（而不是盲目加大 backbone）。
3. 在确认收益后再决定是否开启 SAVE_VISUALS=1 跑全量可视化归档。

---
Report author: GitHub Copilot (GPT-5.3-Codex)
Date: 2026-05-04
