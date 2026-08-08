# V3.4-FM 细节增强后训练报告（2026-05-04）

## 1. 目标

这次工作的目标是：在不重新随机初始化主网络的前提下，基于已有 V3.4-FM 权重继续训练，并重点改善轮廓细节位置的表现。

核心要求：

1. 尽量继承已有 V3.4-FM 权重。
2. 不推翻原网络，只做可热启动的增量增强。
3. 提高细节区域、边界局部变化、曲率较大位置的轮廓质量。
4. 用完整 BTCV 测试集 IoU 验证，而不是只看单张图。

## 2. 当前实验

实验名称：

- `btcv_diffusion_dit_v3_4_fm_full_noleak_yolos_detail_gpu4_reusemax`

训练配置：

- 配置文件：`configs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolos_detail_gpu4_reusemax.yaml`
- 输出目录：`data/outputs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolos_detail_gpu4_reusemax`
- 训练卡：GPU 4
- batch size：32
- num_workers：12
- save_ep：50
- 继承权重：`data/outputs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolos_gpu67_reusemax/checkpoints/latest.pt`
- 继承方式：`resume_weights_only: true`

当前状态：

- 训练仍在运行。
- 训练进程：PID `2842286`
- 当前训练日志最新记录：epoch `3412`，step `78479 / 230000`
- 当前最新 checkpoint：epoch `3400`，step `78200 / 230000`
- 最新 checkpoint 文件：`data/outputs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolos_detail_gpu4_reusemax/checkpoints/latest.pt`
- 最新 checkpoint 保存时间：2026-05-04 18:26

## 3. 已完成改动

### 3.1 权重继承

新增了 `resume_path` 支持，使训练可以从指定实验目录的 checkpoint 继承，而不是只能从当前输出目录恢复。

相关文件：

- `diffusion_train.py`
- `lib/config/config.py`

目的：

- 直接从已有 V3.4-FM 最新权重继续训练。
- 能加载多少旧权重就加载多少，新增结构单独学习。
- 避免重新随机初始化主干，节省训练时间。

### 3.2 细尺度特征增强

加入了 P3 层特征融合。

相关文件：

- `lib/networks/snake/ct_snake.py`

目的：

- 原来的特征尺度对细小边界变化不够敏感。
- P3 特征空间分辨率更细，更适合补充边界局部信息。
- 融合方式采用残差注入，尽量减少对旧模型已有能力的破坏。

对应配置：

- `v3_4_use_p3_features: true`

### 3.3 轮廓法线方向细节上下文

在 V3.4 denoiser 中加入 detail context，让网络在每个轮廓点附近额外看到局部细节信息。

相关文件：

- `lib/networks/diffusion/dit_denoiser_v3_4.py`
- `lib/networks/diffusion/flow_matching_evolution.py`

当前使用模式：

- `v3_4_use_detail_context: true`
- `v3_4_detail_context_mode: normal`

目的：

- 轮廓误差通常不是整体偏移，而是局部边界不够贴合。
- 沿轮廓法线方向采样局部上下文，更直接对应“边界该往里还是往外修”的问题。
- 该方案比盲目加大 backbone 更直接针对细节误差。

### 3.4 推理与评估修正

为了保证训练结构和推理结构一致，修正了推理和全测试集评估脚本。

相关文件：

- `scripts/infer_single_sample.py`
- `scripts/eval_v37_full_iou.py`

修正内容：

- 推理时正确启用 P3 融合。
- 推理时正确启用 V3.4 detail context。
- 修复 detail mode 优先级问题，避免 V4 默认设置误覆盖 V3.4 设置。

说明：

- 曾有一次无效评估，150 个样本全部失败，原因是 detail mode 被错误设成了不匹配的模式。
- 问题已修复，并重新跑完整测试集。
- 无效结果目录：`visual/yolos_detail_gpu4_latest_full_iou_ode50_rerun`
- 该目录结果不应作为实验结论。

### 3.5 输入管线优化

根据训练日志分析，之前慢点主要集中在每个 epoch 的第 1 个 step，原因更像是 DataLoader worker 反复启动和预取不足。

已加入：

- `persistent_workers`
- `prefetch_factor`

相关文件：

- `lib/datasets/make_dataset.py`

当前配置：

- `dataloader_persistent_workers: true`
- `dataloader_prefetch_factor: 4`

目的：

- 减少短 epoch 场景下的边界等待。
- 减少 GPU 等数据的时间。
- 配合 batch size 32 提高 4 号卡利用率。

## 4. 完整测试集 IoU 结果

测试集：

- `BtcvVal`
- 样本数：150

### 4.1 较早 checkpoint，ODE=50

输出文件：

- `visual/yolos_detail_gpu4_ep1600_full_iou_ode50/v3_7_full_test_iou_20260503_213828.json`

结果：

| 指标 | 数值 |
|---|---:|
| mean_iou_sample_avg | 0.894335 |
| mean_iou_contour_avg | 0.890938 |
| median_iou_sample_avg | 0.892708 |
| std_iou_sample_avg | 0.024254 |
| failed_samples | 0 |

### 4.2 较早 checkpoint，ODE=100

输出文件：

- `visual/yolos_detail_gpu4_latest_full_iou_ode100/v3_7_full_test_iou_20260503_214959.json`

结果：

| 指标 | 数值 |
|---|---:|
| mean_iou_sample_avg | 0.894285 |
| mean_iou_contour_avg | 0.891040 |
| median_iou_sample_avg | 0.894116 |
| failed_samples | 0 |

结论：

- ODE 从 50 增加到 100，整体没有明显收益。

### 4.3 epoch 3050 checkpoint，ODE=50

输出文件：

- `visual/yolos_detail_gpu4_latest_full_iou_ode50_fixed/v3_7_full_test_iou_20260504_143518.json`

结果：

| 指标 | 数值 |
|---|---:|
| mean_iou_sample_avg | 0.899966 |
| mean_iou_contour_avg | 0.896827 |
| median_iou_sample_avg | 0.899514 |
| std_iou_sample_avg | 0.022533 |
| failed_samples | 0 |

相对较早 checkpoint 的 ODE=50：

| 指标 | 变化 |
|---|---:|
| mean_iou_sample_avg | +0.005631 |
| mean_iou_contour_avg | +0.005889 |
| median_iou_sample_avg | +0.006807 |
| std_iou_sample_avg | 0.024254 -> 0.022533 |

结论：

- 继续训练后，指标有稳定提升。
- 波动也变小，说明模型整体更稳。

### 4.4 epoch 3050 checkpoint，ODE=150

输出文件：

- `visual/yolos_detail_gpu4_latest_ep3050_full_iou_ode150_20260504_1439/v3_7_full_test_iou_20260504_144526.json`

结果：

| 指标 | 数值 |
|---|---:|
| mean_iou_sample_avg | 0.900164 |
| mean_iou_contour_avg | 0.897071 |
| median_iou_sample_avg | 0.900127 |
| std_iou_sample_avg | 0.022302 |
| failed_samples | 0 |

相对同一 checkpoint 的 ODE=50：

| 指标 | 变化 |
|---|---:|
| mean_iou_sample_avg | +0.000199 |
| mean_iou_contour_avg | +0.000244 |
| median_iou_sample_avg | +0.000613 |

结论：

- ODE=150 比 ODE=50 略高，但收益很小。
- 不建议默认使用 ODE=150。
- 更大的 ODE 步数可以作为最终展示或少量重点样本复查手段，不适合作为主要提升来源。

## 5. 当前判断

### 5.1 细节增强方向是有效的

当前 V3.4-FM detail 实验已经达到：

- mean_iou_sample_avg：约 `0.900`
- mean_iou_contour_avg：约 `0.897`
- failed_samples：`0`

这说明 P3 细尺度特征和法线方向 detail context 的组合是有效的。

### 5.2 主要收益来自训练和结构增强，不来自 ODE 步数

从 ODE=50、100、150 的结果看：

- ODE 加大不会带来明显跃升。
- ODE=150 只比 ODE=50 高约 `0.0002`。
- 后续不应把主要精力放在继续加 ODE 上。

### 5.3 当前最新 checkpoint 还没有补跑完整 IoU

目前已经评估到 epoch 3050 checkpoint。

训练现在已经保存到 epoch 3400 checkpoint，因此还需要补跑：

- epoch 3400
- ODE=50
- 完整测试集 150 样本

如果 epoch 3400 继续提升，说明这个方向仍然没有到平台期。

## 6. 和整合方案的关系

当前 V3.4-FM detail 实验适合作为整合方案里的“稳健基线”：

优点：

- 继承旧权重充分。
- 改动相对克制。
- 已经完整跑通训练和全测试集评估。
- 指标已经达到约 0.900。
- 没有失败样本。

不足：

- 仍然需要更细的样本级分析，尤其是高曲率、薄结构、边界粘连区域。
- 目前只证明整体 IoU 提升，还没有把每类器官和每类失败形态拆开看。
- 最新 epoch 3400 checkpoint 需要补评估。

建议定位：

- V3.4-FM detail：作为可靠、可继承、风险较低的方案。
- V4.0：作为更激进的结构整合方案。
- 两者应使用同一套测试集、同一套 ODE 设置、同一套可视化样本进行对比。

## 7. 下一步建议

### 7.1 先补 epoch 3400 完整测试集 IoU

推荐设置：

- ODE=50
- SAVE_VISUALS=1
- 测试集 150 样本

原因：

- ODE=50 已经足够代表模型能力。
- SAVE_VISUALS=1 可以直接看细节改善是否真实发生。
- epoch 3400 是当前最新 checkpoint，比已评估的 epoch 3050 更新。

### 7.2 做样本级对比

需要对比：

- 原 V3.4-FM
- V3.4-FM detail
- V4.0

重点看：

- IoU 提升最大的样本。
- IoU 下降的样本。
- 高曲率边界。
- 器官贴边区域。
- 小器官和薄结构。

目的：

- 判断提升来自真实轮廓贴合，还是来自局部偶然波动。
- 找出 detail 方案仍然解决不了的失败类型。

### 7.3 不建议继续单纯加大 ODE

原因：

- ODE=150 相对 ODE=50 的提升只有约 `0.0002`。
- 计算成本明显增加。
- 对细节位置的核心问题帮助有限。

更值得做的是：

- 更精细的局部特征。
- 更明确的曲率区域建模。
- 更好的样本级误差分析。

## 8. 结论

V3.4-FM detail 后训练已经跑通，并且完整测试集 IoU 已经达到约 `0.900`。  

这条路线的价值在于：不重开训练，不大幅破坏旧模型，通过细尺度特征和轮廓局部上下文，稳定改善边界细节。

当前最需要补的是 epoch 3400 checkpoint 的完整测试集评估。如果这个 checkpoint 继续提升，V3.4-FM detail 可以作为整合方案中的稳健主线之一。

---

报告日期：2026-05-04  
报告文件：`archive/notion/archived/v3_4_fm_detail_posttrain_report_20260504.md`
