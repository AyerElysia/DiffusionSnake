# V3.4-FM + YOLO Scale 扩展阶段报告（2026-05-04）

## 1. 这次做了什么

这次工作的目标，是把原来的 `V3.4-FM` 主线从固定的 `YOLOv8-n` 扩展到可切换 `YOLOv8-s / YOLOv8-m`，同时尽量复用已经训好的旧模型参数，避免每次换主干都从零开始。

实际完成的内容有四块：

1. 让网络支持从配置里直接指定 `YOLO` 尺度  
   现在可以在配置中用 `yolo_model_scale: 's'` 或 `yolo_model_scale: 'm'` 明确切换主干，不再默认落回 `n`。

2. 改了旧 checkpoint 的加载方式  
   原来只有“形状完全一样才加载”。  
   现在改成两级复用：
   - 形状完全一致：直接加载
   - 形状变大但仍同名同结构：把能对上的前半部分拷进去，剩下部分保留新初始化  
   这使得 `n -> s`、`n -> m` 这种扩容不再等于完全重训。

3. 补了两套正式训练配置  
   - `YOLOv8-s`：`configs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolos_gpu67.yaml`
   - `YOLOv8-m`：`configs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolom_gpu35.yaml`

4. 补了 `DataLoader` 的并行读取配置  
   支持 `persistent_workers` 和 `prefetch_factor`，用于更稳定地喂数据。

---

## 2. 代码改动位置

本次核心改动在下面几个文件：

- `diffusion_train.py`
- `lib/networks/snake/ct_snake.py`
- `lib/config/config.py`
- `lib/datasets/make_dataset.py`
- `configs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolos_gpu67.yaml`
- `configs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolom_gpu35.yaml`

本次提交的 commit：

- `66c6634`  `Add YOLO scale resume configs for V3.4 FM`

说明：

- 这次已经提交到本地仓库
- 还没有做远端 push

---

## 3. 参数复用效果

### 3.1 `n -> s`

最开始只做“同形状直接加载”时，能接上的只有一部分。  
后面改成“同形状直接加载 + 扩维层局部拷贝”之后，`s` 这条可以把能接的基本都接上。

最终启动日志里显示：

- 匹配张量数：`578`
- 完全一致加载：`260`
- 扩维后局部拷贝：`318`
- 总复用参数量：约 `1359 万`

这意味着：

- `YOLOv8-s` 不是从零开始
- 原来 `n` 里能保留的卷积、BN、头部结构，大部分都被继续利用了

### 3.2 `n -> m`

`m` 这条在切到自己的输出目录并正式续训后，最新 checkpoint 已经和当前结构完全匹配。  
也就是说，现在它后续续训已经不再是“部分加载”，而是正常从自己的保存点完整恢复。

---

## 4. 训练配置原则

这次 `s` 和 `m` 都保持了以下原则不变：

- 仍然是 `V3.4-FM`
- 仍然是多步 refinement
- 训练噪声 `1.0`
- 推理噪声 `1.0`
- 仍然只用训练集训练，不合并验证集

也就是：

- 主干变了
- 参数初始化方式变了
- 主训练范式没变

---

## 5. 参数量

### `YOLOv8-s` 版本

- 总参数：`21,288,278`
- 可训练参数：`21,288,262`
- 其中 `YOLO` 主干：`10,639,316`
- 轮廓预测部分：`10,644,290`

### `YOLOv8-m` 版本

- 总参数：`35,703,910`
- 可训练参数：`35,703,894`
- 其中 `YOLO` 主干：`25,054,948`
- 轮廓预测部分：`10,644,290`

结论：

- `s` 和 `m` 的差别，几乎全部来自前面的 `YOLO` 主干
- 后面的轮廓扩散部分基本一样

---

## 6. 训练状态

### `s` 线

输出目录：

- `data/outputs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolos_gpu67_reusemax`

当前状态：

- 已停止
- 最后停在：
  - `epoch 1845`
  - `step 27690`
- 最后 checkpoint 时间：
  - `2026-05-03 22:21:30`

### `m` 线

输出目录：

- `data/outputs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolom_gpu35_reusemax`

当前状态：

- 仍在继续训练
- 截至本报告生成时，日志最新位置：
  - `epoch 3297`
  - `step 49457`
- 最近一次记录的单步时间：
  - `3399.53 ms`
- 最新 checkpoint 时间：
  - `2026-05-04 17:58:03`

结论：

- 当前实际主线只剩 `m`
- `s` 已经停掉，保留作对照

---

## 7. 评估结果

### 7.1 全测试集结果

为了避免只看单张图误判，已经把 `s` 和 `m` 都完整跑过一遍测试集。

#### `s`

结果文件：

- `visual/full_eval_s_20260503/v3_7_full_test_iou_20260503_213246.json`

结果：

- `mean_iou_sample_avg = 0.889102`
- `mean_iou_contour_avg = 0.885497`
- `failed_samples = 0`

#### `m`

结果文件：

- `visual/full_eval_m_20260503/v3_7_full_test_iou_20260503_213817.json`

结果：

- `mean_iou_sample_avg = 0.891106`
- `mean_iou_contour_avg = 0.887428`
- `failed_samples = 0`

### 7.2 全测试集结论

`m` 比 `s` 略高，但差距非常小：

- `m - s ≈ 0.0020`

这说明：

- 把 `YOLO` 主干从 `s` 换到 `m`，是有效的
- 但提升幅度不大
- 当前瓶颈不只是主干容量，后面的轮廓修正机制本身仍然限制上限

---

## 8. 单样本快速观察

为了快速看图，也做了几次当前 checkpoint 的单样本推理。

最新一次 `m` 的单样本结果：

- 结果文件：`visual/current_quicklook_m_20260504/v3_7_metrics_20260504_142057.json`
- 该样本平均 IoU：`0.907750`

这说明：

- 当前模型在较普通样本上已经比较稳定
- 但整套测试集仍停在 `0.891` 左右，说明真正难点还是那些更复杂的样本

---

## 9. 现在已经确认的判断

### 9.1 已经成立的结论

1. `YOLOv8-s / m` 方案可以正常接入现有 `V3.4-FM`
2. 旧模型参数复用是成立的，不需要每次从零训
3. `m` 的确优于 `s`
4. 但 `m` 的提升不大，不足以单独解决当前上限问题

### 9.2 当前更像什么问题

从现有结果看，更像是：

- 主干容量不够不是唯一问题
- 轮廓 refinement 的拟合方式、对齐方式、后期“小修正”能力，仍然是限制因素

也就是说，问题更偏“修正机制仍然有天花板”，不只是“前面 backbone 太弱”。

---

## 10. 关于“零位移训练样本”的判断

这个方向已经讨论过，当前结论是：

- 可以试
- 但不建议一上来大量加入“严格零位移”

更合适的是：

- 小比例加入“近零位移 / 小位移”样本
- 先从 `5%~10%` 开始

原因：

- 多步 refinement 的后几步，本来就应该学会做小修正
- 但如果严格零位移太多，模型会更保守，出现“该修的时候修不动”

所以如果下一步要做整合方案，这个点更适合被当作一个“小修正稳定性增强项”，不是主改动。

---

## 11. 我建议的整合方向

如果现在要整合成下一版，不建议再把精力主要放在 `s` 和 `m` 谁更大。

更合理的方向是：

### 推荐主线

- 以 `V3.4-FM + YOLOv8-m` 为主线
- 保留现在这套“旧参数最大化复用”初始化方案

### 下一步优先改的，不是 backbone，而是下面几个点

1. 训练中补“小位移 / 近零位移”样本  
   目标：让后几步更会收边

2. 明确区分“主评估”和“可视化后处理”  
   避免看图时觉得变好了，但主指标没变化

3. 如果继续做结构尝试，优先考虑 refinement 机制本身  
   因为 `m` 已经证明“只加 backbone”不是决定性突破

---

## 12. 当前可直接复用的资产

配置：

- `configs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolos_gpu67.yaml`
- `configs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolom_gpu35.yaml`

输出目录：

- `data/outputs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolos_gpu67_reusemax`
- `data/outputs/btcv_diffusion_dit_v3_4_fm_full_noleak_yolom_gpu35_reusemax`

评估结果：

- `visual/full_eval_s_20260503/`
- `visual/full_eval_m_20260503/`
- `visual/current_quicklook_m_20260504/`

---

## 13. 一句话总结

这次工作已经把 `V3.4-FM` 成功扩展到 `YOLOv8-s / m`，并且把旧模型权重复用这件事做通了。  
结果上，`m` 比 `s` 更好，但提升很小，当前真正限制上限的更可能是后面的轮廓修正机制，而不是单纯的 backbone 容量。
