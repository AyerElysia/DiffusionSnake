# BTCV 检测器 / 极值点阶段任务总结

## 1. 任务背景

本阶段工作的目标，是把 BTCV 上游的 detector 路线梳理清楚，并判断后续 Snake / octagon / diffusion 初始化究竟应该继续依赖 bbox，还是转向 extreme-point（4 点）主线。

## 2. 初始排查结论

最重要的初始发现是：**当前真正接在 Snake 里的活跃 detector 路径，其实是 YOLOv8n + P2 + bbox detect-only**，而**不是**一个已经成为 BTCV 主线的 extreme-point detector。

也就是说，仓库之前能跑通的主路径，本质上仍然是 bbox 检测，然后再服务于后续轮廓初始化；BTCV 4 点 extreme-point 并没有在主线上完全取代 bbox。

## 3. 本轮补齐的关键接线

### 3.1 BTCV 4 点 pose / extreme-point 训练入口

BTCV 的 4 点 pose（可视为 extreme-point）训练链路已经明确接通，入口脚本为：

- `scripts/train_btcv_yolo_pose.py`

这意味着 BTCV 并不是不能训练 extreme-point detector，而是之前主线没有把它当成默认 detector 路径。

### 3.2 自定义 4 点 pose 验证 bug 修复

在：

- `lib/networks/YOLOV8/models/yolo/pose/val.py`

中修复了自定义 4 点 pose 的验证问题。修复后，验证逻辑不再错误依赖 COCO 17 点的固定 keypoint 设定，**非 COCO keypoint 数量（例如 BTCV 4 点）现在可以被正确验证**。

这一步非常关键，否则即使训练出了 4 点 pose 模型，评估结果也可能失真或直接错误。

## 4. 关键实验结果

### 4.1 现有 bbox baseline

- bbox baseline（existing）
  - `precision_like = 0.826`
  - `recall@0.5 = 0.800`
  - `recall@0.75 = 0.739`
  - `matched_mean_iou = 0.848`

这说明现有 bbox detector 不是完全不可用，但它衡量的是 bbox 检测质量，并不直接代表 extreme-point 初始化质量。

### 4.2 pose / extreme-point 方向实验

- `yolov8n-pose`，训练 3 epochs
  - `pose mAP50 = 0.610`
  - `pose mAP50-95 = 0.428`
  - `pose recall = 0.540`

- `yolov8s-pose`，训练 3 epochs
  - `pose mAP50 = 0.738`
  - `pose mAP50-95 = 0.574`
  - `pose recall = 0.647`

- `yolov8s-pose`，训练 10 epochs
  - `pose mAP50 = 0.851`
  - `pose mAP50-95 = 0.718`
  - `pose recall = 0.793`

## 5. 对实验结果的判断

本轮最明确的结论是：**“基础模型太弱”这个怀疑是对的。**

更具体地说：

1. 之前挂在主线里的 **YOLOv8n bbox detect-only**，并不是一个适合作为 BTCV extreme-point 能力判断依据的 baseline。
2. 它既弱在 backbone / capacity，也弱在任务形式本身——bbox 目标并不等价于极值点初始化目标。
3. 只看短程实验，`yolov8s-pose` 已经明显优于 `yolov8n-pose`，而且 10 epochs 的结果进一步证明：**换到更强的 pose 基线后，BTCV 4 点 extreme-point 路线是能快速起量的。**

因此，**YOLOv8n bbox 是错误 baseline，`yolov8s-pose` 才是更合理的短期推进方向。**

## 6. 对后续集成的结论

对 octagon / diffusion 来说，后续初始化策略应当逐步调整为：

1. **优先使用 extreme-point / 4 点 pose 初始化**
2. 继续保留 bbox 初始化，但只作为 fallback

原因很直接：

- extreme-point 与 octagon / contour 初始化的几何语义更接近
- 它比纯 bbox 更贴近后续 diffusion / Snake 真正需要的结构先验
- bbox 仍有工程价值，但更适合承担兜底角色，而不是长期主线

## 7. 总结

本次 detector / extreme-point 任务完成后，可以把结论概括为三句话：

1. 当前历史主线其实是 **YOLOv8n + P2 + bbox detect-only**，不是 BTCV extreme-point 主线。
2. BTCV 4 点 pose 训练与验证链已经打通，且 `yolov8s-pose` 明显比 `yolov8n bbox` / `yolov8n-pose` 更有前景。
3. 后续 Snake / octagon / diffusion 集成，应以 **extreme-point 初始化优先、bbox fallback 保留** 为方向。

## 8. 非 YOLO 备选：RT-DETR 最小试跑

按“尽量不动原主线”的原则，新增了：

- `scripts/train_btcv_rtdetr.py`

这条线直接调用仓库内 vendored `yoloe/ultralytics` 的 RT-DETR，并额外处理了两类实际兼容问题：

1. `snake1` 的 Python 3.7 与 vendored RT-DETR 代码存在少量 Python 3.8+ 语法 / mixin 兼容点，需要最小修补后才能真正启动。
2. BTCV detect 导出数据本身不能被 RT-DETR/Ultralytics 直接识别：
   - 图片名是 `*_image.png`
   - detect 标签名却是纯数字 `*.txt`
   - 因此脚本会自动生成 runtime dataset，把标签重映射成与图片同 stem 的 `*_image.txt`

最小 smoke run：

```bash
conda activate snake1
python scripts/train_btcv_rtdetr.py \
  --epochs 1 \
  --fraction 0.02 \
  --imgsz 320 \
  --batch 4 \
  --workers 0 \
  --name btcv_rtdetr_trial_smoke \
  --exist-ok
```

实际结果（真实跑通，不是只过参数解析）：

- 输出目录：`data/outputs/btcv_rtdetr_trial_smoke/`
- checkpoint：
  - `weights/best.pt`
  - `weights/last.pt`
- summary：
  - `data/outputs/btcv_rtdetr_trial_smoke/trial_summary.json`

本次 1 epoch / 2% fraction 的极小试跑指标：

- `precision(B) = 0.000183`
- `recall(B) = 0.0399`
- `mAP50(B) = 0.000180`
- `mAP50-95(B) = 0.0000333`

结论：

- **RT-DETR 作为 in-repo 非 YOLO detector 候选，已经可以在 snake1 上被真实拉起并完成一个最小 BTCV 检测训练 / 验证回合。**
- 但在当前“随机初始化 + 极小 fraction + 1 epoch”的 smoke 设定下，效果几乎不可用，离当前 `yolov8s-pose` 主线有明显差距。
- 所以它现在更适合作为“已验证可接入的非 YOLO 备选”，而不是立刻替代现有 BTCV extreme-point 主线。

## 9. 非 YOLO 下一步：ResNet18 heatmap detector（CenterNet 风格）最小主线

在 RT-DETR 证明“能接入但完全不够强”之后，继续按“**非 YOLO，但尽量贴近 Snake 现有结构**”的思路，补了一条新的 heatmap detector baseline。

### 9.1 本轮新增内容

新增 / 修改如下：

- 新配置：
  - `configs/btcv_heatmap_resnet18_detect_only.yaml`
- 新训练脚本：
  - `scripts/train_btcv_heatmap_detector.py`
- 核心接线：
  - `lib/networks/snake/ct_snake.py`
  - `lib/train/trainers/snake.py`
  - `lib/config/config.py`

这条线的设计目标不是立刻替换当前 diffusion 主线，而是先做一个**真正可训练、可评估、可对照 RT-DETR / YOLO bbox 的非 YOLO detector baseline**。

具体实现方式是：

1. 用 `torchvision` 的 `ResNet18` 做 backbone。
2. 在 stride=4 特征上做一个轻量 FPN 风格 top-down 融合。
3. 预测：
   - `ct_hm`
   - `wh`
4. 推理时把 heatmap decode 成现有 Snake 兼容的：
   - `output['detection'] = [x1, y1, x2, y2, score, cls]`
5. 训练时复用仓库里已有的：
   - `FocalLoss`
   - `IndL1Loss1d`

也就是说，这条非 YOLO 线不是另起一套工程，而是**把仓库里原本还在的数据、loss、decode、Snake 接口重新拉起来**。

### 9.2 关键 bug：heatmap 类别索引和 YOLO/eval 口径不一致

这一轮真正最重要的新发现，不是模型本身，而是一个**类标口径错位 bug**：

1. Snake 数据流里，`ct_hm` 的类别索引是 **1-based** 风格。
2. 也就是说，`ct_hm` 的第 0 通道实际上是空占位，真正有效的是后面的通道。
3. 但现有 detector eval / YOLO 标签口径是 **0-based**。

如果直接把 heatmap decode 的类别拿去评估，就会导致：

- 真实类别整体错一位
- 某些本来应命中的框被算成错误类别
- 还会出现看起来像“幽灵类”的大量假阳性

本轮已在 `ct_snake.py` 中修正这个 offset：

- heatmap decode 时跳过空占位通道
- 输出 detection 时统一回到 0-based 类别口径

这一步修掉之后，heatmap baseline 的指标发生了**跳跃式改善**，说明之前那部分“差得离谱”的结果并不全是模型弱，而有相当一部分是**评估/推理口径错位**。

### 9.3 实际 smoke 训练与结果

#### (1) 1 epoch，scratch

最小 detector-only smoke（`freeze_snake=true`）已经真实跑通：

- checkpoint：
  - `data/model/btcv_heatmap_resnet18_detect_only_smoke/0.pth`

初步说明：

- 非 YOLO heatmap detector 在当前 repo / `snake1` / BTCV 数据流里是**真的能训练起来**的
- trainer、loss、decode、eval 都已经闭环

#### (2) 3 epoch，ImageNet 预训练 ResNet18

进一步做了更公平的 smoke：

```bash
conda activate snake1
python scripts/train_btcv_heatmap_detector.py \
  --epochs 3 \
  --batch 4 \
  --workers 0 \
  --save-ep 1 \
  --heatmap-pretrained \
  --model-dir data/model/btcv_heatmap_resnet18_pretrained_smoke
```

训练 loss 走势：

- `det_loss` 大致从 `11.24` 降到 `1.73`

这说明：

- 这条非 YOLO detector 不是“只能过前向”的伪接线
- 它在 BTCV 上是**正常学习**的

### 9.4 3 epoch 预训练 heatmap 的 val 结果（最终采用阈值 0.15）

最终保留的默认阈值为：

- `det_conf_thresh = 0.15`

对应结果：

- checkpoint：
  - `data/model/btcv_heatmap_resnet18_pretrained_smoke/2.pth`
- eval summary：
  - `data/model/btcv_heatmap_resnet18_pretrained_smoke/val_eval_summary_final.json`
- 单样本可视化：
  - `data/model/btcv_heatmap_resnet18_pretrained_smoke/val_sample0_pred_vs_gt_final.png`

核心指标：

- `precision_like = 0.559`
- `recall@0.5 = 0.730`
- `recall@0.75 = 0.301`
- `mean_best_iou = 0.573`
- `matched_mean_iou = 0.658`

### 9.5 阈值扫描结论

对同一个 `2.pth` 做了 score threshold 扫描，发现：

- `0.05`：
  - `precision_like = 0.102`
  - `recall@0.5 = 0.773`
- `0.10`：
  - `precision_like = 0.296`
  - `recall@0.5 = 0.764`
- `0.15`：
  - `precision_like = 0.559`
  - `recall@0.5 = 0.730`
- `0.20`：
  - `precision_like = 0.758`
  - `recall@0.5 = 0.669`
- `0.25`：
  - `precision_like = 0.870`
  - `recall@0.5 = 0.606`

因此当前判断是：

- 如果目标偏向 proposal recall，可以把阈值压低到 `0.10~0.15`
- 如果目标偏向更干净的候选框，可以提到 `0.20`
- 作为当前 smoke 版默认值，`0.15` 是更平衡的折中

### 9.6 对 heatmap 非 YOLO 路线的阶段性判断

这条线现在可以得出一个**比 RT-DETR 明确得多**的结论：

1. **它不是“理论可行”，而是已经在当前 repo 里真实训练、真实评估、真实出图了。**
2. 修正类别 offset 后，它的 `recall@0.5 = 0.730`，已经明显强于之前那个“几乎不可用”的 RT-DETR smoke。
3. 它已经逼近当前 YOLO bbox baseline 的 recall 水平（bbox baseline `recall@0.5 = 0.800`）。
4. 但从整体质量看，它目前仍然**没有超过**当前现有 bbox baseline：
   - bbox baseline：`precision_like = 0.826`，`matched_mean_iou = 0.848`
   - 当前 heatmap smoke：`precision_like = 0.559`，`matched_mean_iou = 0.658`

因此，当前最合理的判断是：

- **heatmap / CenterNet 风格非 YOLO 路线已经从“研究想法”进入“值得继续投入的真实候选”**
- 但在仅 3 epoch 的 smoke 规模下，它还不能直接宣称替代当前最成熟的 bbox baseline
- 它现在的定位应该是：
  - **比 RT-DETR 更值得继续推进**
  - **已经证明非 YOLO 并非走不通**
  - **下一步值得做更长训练和 octagon / diffusion-facing 验证**

## 10. 当前总判断（更新版）

到目前为止，检测部分最可信的排序已经比较清楚：

1. **短期最稳主线：`yolov8s-pose` extreme-point**
2. **最值得继续投入的非 YOLO 下一步：heatmap / CenterNet 风格 detector**
3. **当前不建议继续砸的非 YOLO 方向：RT-DETR bbox**

换句话说：

- “不用 YOLO”这件事，我已经不是只停留在讨论层面，而是实际做了两条路：
  - RT-DETR
  - heatmap detector
- 结果是：
  - **RT-DETR 目前不行**
  - **heatmap 非 YOLO 有真实前景**
- 但如果问“现在马上哪条最能服务你后面的 diffusion 初始化”，答案仍然是：
  - **`yolov8s-pose` extreme-point 先继续当主线**
  - **heatmap detector 作为下一条最值得推进的替代主线**

## 11. 新进展：heatmap detector 长训 + 与 diffusion 的真实联合适配

这一步的目标，不再只是证明“非 YOLO detector 能训练”，而是回答两个更实际的问题：

1. **heatmap detector 本体能不能继续逼近甚至超过旧 bbox baseline？**
2. **把它接到旧 diffusion 主线后，是否真的能改善下游 contour / octagon 初始化？**

### 11.1 heatmap detector 从 3 epoch 继续训练到 10 epoch

在：

- `data/model/btcv_heatmap_resnet18_pretrained_smoke/`

上继续从 `2.pth` 续训到 `9.pth`，并用同一套 detector eval 脚本在 val 全集上重新统计。

最终 checkpoint：

- `data/model/btcv_heatmap_resnet18_pretrained_smoke/9.pth`

最终 val 结果（`score_thresh = 0.15`）：

- `precision_like = 0.648`
- `recall@0.5 = 0.810`
- `recall@0.75 = 0.508`
- `mean_best_iou = 0.653`
- `matched_mean_iou = 0.725`

这组结果比 3 epoch 明显更强：

- 3 epoch：`precision_like = 0.559`，`recall@0.5 = 0.730`，`matched_mean_iou = 0.658`
- 10 epoch：`precision_like = 0.648`，`recall@0.5 = 0.810`，`matched_mean_iou = 0.725`

阶段性判断：

1. **heatmap 非 YOLO detector 本体已经不再只是“有前景”，而是确实做到了一个比较强的 BTCV 检测 baseline。**
2. 它在 `recall@0.5` 上已经达到并略超旧 bbox baseline（旧 bbox baseline `0.800`）。
3. 但它在框质量和纯 proposal cleanliness 上，仍然没有完全超过旧 bbox baseline：
   - 旧 bbox baseline：`precision_like = 0.826`，`matched_mean_iou = 0.848`
   - 当前 10 epoch heatmap：`precision_like = 0.648`，`matched_mean_iou = 0.725`

所以如果只谈 detector，本轮结论可以改写为：

- **“检测太差”这个问题，在非 YOLO heatmap detector 这条线上，已经被明显缓解了。**
- 但如果要求“全面超过旧 bbox baseline”，目前还差最后一截。

### 11.2 直接把新 heatmap detector 换进旧 diffusion checkpoint：几乎失败

为了验证 detector 提升是否能直接传导到下游，新增了：

- `configs/btcv_diffusion_dit_v3_heatmap_resnet18.yaml`
- `test/test_heatmap_diffusion_eval.py`
- `scripts/merge_heatmap_diffusion_checkpoints.py`

其中：

- `test_heatmap_diffusion_eval.py` 负责把 detector checkpoint 与 diffusion checkpoint 合并后做下游 smoke
- `merge_heatmap_diffusion_checkpoints.py` 负责构造可直接给 `diffusion_train.py` 使用的 merged init checkpoint

先做了一个最直接的实验：

- detector：`heatmap 6/9 epoch checkpoint`
- diffusion：旧 `btcv_diffusion_dit_v3` 的 `latest.pt`
- 不做适配，直接替换 detector

10 样本下游 smoke 结果：

- `precision_like = 0.068`
- `mean_best_iou = 0.0145`
- `matched_mean_iou = 0.147`
- `recall@0.5 = 0.0`
- `recall@0.75 = 0.0`

这说明：

- **detector 本体变强，并不等于旧 diffusion / GCN 头可以零成本适配新的 detector feature distribution。**
- 旧 diffusion 主线对原 detector 特征分布依赖很强，不能指望“只替换 detector 权重”就自动工作。

### 11.3 修通 heatmap + diffusion 联合适配训练

为了解决上面的断层问题，本轮还继续把联合适配链路真正接通了：

1. `lib/train/trainers/diffusion_trainer.py`
   - 现在不再写死 YOLO loss；
   - 对 heatmap backend 也能正常构造 detector 分支损失；
2. `scripts/merge_heatmap_diffusion_checkpoints.py`
   - 修复了 checkpoint key 前缀问题；
   - 现在能把：
     - `net.heatmap_detector.*`
     - 旧 diffusion 的 `net.gcn.* / denoiser.*`
     真实合并进一个初始化 checkpoint；
3. 联合适配时最终确认：
   - `matched_keys = 278`
   - `missing_after_load = 0`

也就是说，这条 **heatmap detector + old diffusion checkpoint → 继续联合微调** 的训练链，现在已经真实闭环，不再只是理论方案。

### 11.4 先只优化 diffusion_loss：效果有限

#### (1) 1 epoch 联合适配

10 样本下游 smoke：

- `precision_like = 0.519`
- `mean_best_iou = 0.101`
- `matched_mean_iou = 0.121`
- `recall@0.5 = 0.0`

相比“直接替换几乎完全失败”的 `mean_best_iou = 0.0145`，已经有明显改善。

#### (2) 继续到 5 epoch

在：

- `data/outputs/btcv_diffusion_dit_v3_heatmap_joint_ft5/`

继续训练到 5 epoch 后，再做相同的 10 样本下游 smoke：

- `precision_like = 0.508`
- `mean_best_iou = 0.1088`
- `matched_mean_iou = 0.1316`
- `recall@0.5 = 0.0`
- `recall@0.75 = 0.0`

结论：

1. **联合适配是必要的。** 不适配几乎完全不能用。
2. 但如果把 `det_loss` 权重关成 0，只靠 diffusion_loss 去适配，提升幅度仍然偏慢。
3. 这说明下一步不能只想“多训几轮”，而要重新看 joint objective 本身。

### 11.5 改成 det+diff 联合目标后，提升明显

上面那组实验里，我把：

- `loss_scales.det = 0`

也就是 detector 分支损失完全关掉了。接着我又单独做了一组新的联合适配：

- 从同一个 merged init checkpoint 起跑
- 保留 detector loss
- 使用：
  - `loss_scales.det = 1`
  - `diffusion_loss_weight = 1`

也就是改成真正的 **det + diff 联合目标**。

#### (1) det+diff，3 epoch

10 样本下游 smoke：

- `precision_like = 0.543`
- `recall@0.5 = 0.148`
- `mean_best_iou = 0.247`
- `matched_mean_iou = 0.286`

30 样本下游 smoke：

- `precision_like = 0.537`
- `recall@0.5 = 0.093`
- `mean_best_iou = 0.212`
- `matched_mean_iou = 0.246`

和前面“只优化 diffusion_loss”的 5 epoch 版本相比：

- diffusion-only，30 样本：
  - `mean_best_iou = 0.107`
  - `recall@0.5 = 0.000`
- det+diff，3 epoch，30 样本：
  - `mean_best_iou = 0.212`
  - `recall@0.5 = 0.093`

这个提升已经不是噪声级别，而是方向性差异。

#### (2) det+diff，继续到 5 epoch

在：

- `data/outputs/btcv_diffusion_dit_v3_heatmap_joint_det1_ft5/`

继续从 3 epoch 版本续到 5 epoch 后，再做相同评估：

10 样本下游 smoke：

- `precision_like = 0.531`
- `recall@0.5 = 0.123`
- `mean_best_iou = 0.248`
- `matched_mean_iou = 0.292`

30 样本下游 smoke：

- `precision_like = 0.541`
- `recall@0.5 = 0.107`
- `mean_best_iou = 0.219`
- `matched_mean_iou = 0.253`

这说明：

1. **真正有效的不是“只延长适配训练”，而是“保留 detector loss 的 joint objective”。**
2. det+diff 版本已经把下游 `mean_best_iou` 从最开始的 `0.0145` 拉到了 `0.219`。
3. 它终于开始出现非零的 `recall@0.5`，说明下游 contour 结果已经不是“整体错位”的状态，而是开始进入能命中一部分样本的区间。
4. 但即使如此，它仍然还没有到“可以直接替代旧主线”的水平。

#### (3) det+diff，继续到 10 epoch

在：

- `data/outputs/btcv_diffusion_dit_v3_heatmap_joint_det1_ft10/`

继续从 5 epoch 版本续到 10 epoch 后，再做同样的下游评估。

30 样本下游 smoke：

- `precision_like = 0.542`
- `recall@0.5 = 0.093`
- `mean_best_iou = 0.225`
- `matched_mean_iou = 0.255`

val 全集（150 样本）结果：

- `precision_like = 0.555`
- `recall@0.5 = 0.117`
- `mean_best_iou = 0.245`
- `matched_mean_iou = 0.268`

这组结果说明：

1. 继续把 det+diff 路线拉长，**仍然还有增益**；
2. 但从 5 epoch 到 10 epoch 的增益已经开始放缓：
   - 30 样本 `mean_best_iou`：`0.219 -> 0.225`
   - 30 样本 `recall@0.5`：`0.107 -> 0.093`
3. 因而当前更像是进入了“**需要进一步优化 joint training 设计**”的阶段，而不是只靠无脑拉长 epoch 就能继续大幅上涨。

### 11.6 到这一步的最终判断

如果把“检测问题”拆开来看，现在最可靠的结论是：

1. **Detector 本体问题：已经基本被定位并显著改善。**
   - 旧主线里真正的问题之一，确实是 baseline 太弱（`yolov8n bbox`）。
   - `yolov8s-pose` 和现在的 `heatmap_resnet18` 都已经证明，BTCV 上游检测并不是天然做不好。

2. **非 YOLO detector 问题：已经得到一个真实可用的候选。**
   - RT-DETR 目前不值得继续优先投入；
   - **heatmap / CenterNet 风格是当前最靠谱的非 YOLO 方向。**

3. **Detector → diffusion 一体化问题：还没有彻底解决。**
   - 现在已经知道症结不是 detector 本身，而是 **新 detector 特征分布与旧 diffusion/GCN 头不匹配**；
   - 更关键的新结论是：**joint objective 比单纯延长 adaptation epoch 更重要**；
   - 目前 det+diff 联合目标已经把 val 全集 `mean_best_iou` 拉到 `0.245`、`recall@0.5` 拉到 `0.117`，但距离真正可替换主线仍然不足。

因此，到当前这一轮结束时，最稳妥也最诚实的排序是：

1. **马上要服务你后面实验的短期主线：`yolov8s-pose` extreme-point**
2. **最值得继续深挖、而且已经被证明不是伪路线的非 YOLO 方向：heatmap / CenterNet 风格 detector**
3. **下一阶段真正要啃的难点：heatmap detector 的 det+diff 更长程联合适配 / 联合训练，而不是再继续纠结 RT-DETR**
