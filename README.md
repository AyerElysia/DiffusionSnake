# DiffusionSnake：VerSe MoonViT + Flow 双阶段主线

本仓库只保留一条可复现的正式路线：

> **离线 MoonViT layer-18 特征 → 监督训练特征替换器与 Flow → 冻结特征替换器、只对 Flow 做五阶段傅里叶 full-extrap GRPO → 固定 2×4 AB2（8 NFE）推理。**

这是一份可直接交接的运行手册。权重、数据集和约 84 GB 的 MoonViT 缓存不进入 Git；代码通过路径参数使用它们。

## 1. 科学口径

- 当前发布主线是**逐张矢状位切片的纯 2D 模型**。正式配置不使用 Memory、相邻帧特征、3D 卷积、3D Transformer、SAM 或内部检测器。
- MoonViT 是唯一推荐的视觉特征来源。编码器不在训练图中，只读取提前生成并带元数据的 `layer_18` 缓存。
- 训练与评估使用 GT bounding box 和 GT anatomical class，属于 `oracle_prompt` 口径。检测器框或预测类别必须另表报告。
- Stage 1 同时训练 Flow 和 MoonViT 特征替换器；Stage 2 只训练继承的 Flow。
- Stage 2 的五阶段轨迹只是一种训练课程。所有可报告评估和部署始终使用两外阶段、每阶段 4 次 AB2 函数求值，共 8 NFE。
- `prev_contour_init_prob` 在两个正式配置中固定为 `0.0`。仓库中的上一切片初始化钩子不属于本版本的正式结果，不能静默打开后与主线混报。
- Train72、Val37、Dev8 互相分工；`sub-verse010/011/013` 永远禁止访问。

## 2. 从输入到输出

```text
中心矢状位切片
  └─ 离线 MoonViT：448×448，patch 14，layer 18
       └─ float16 特征 [1152, Htoken, Wtoken]
            └─ 3.246M 特征替换器（Conv + PixelShuffle×2）
                 └─ 仿射映射到 128×128 轮廓特征网格

GT box + GT anatomical class
  └─ Route-B：矩形 → 12 点八边形 → 128 点初始轮廓
       └─ 11.127M Flow
            ├─ 外阶段 1：fraction=0.6667，s=0.0，AB2×4
            └─ 外阶段 2：fraction=1.0，s=0.6667，AB2×4
                 └─ 最终 128 点轮廓（总计 8 NFE）
```

第二外阶段会在第一阶段更新后的轮廓位置重新采样特征。AB2 的 4 次速度预测是数值求解，不是 4 个 RL 动作。

| 组成 | 参数量 | Stage 1 | Stage 2 |
|---|---:|---|---|
| Flow | 11,127,108 | 训练 | 训练 |
| MoonViT 特征替换器 | 3,246,336 | 训练 | 冻结 |
| MoonViT 编码器 | 不进入训练图 | 冻结缓存 | 冻结缓存 |
| 总模型 | 14,373,444 | 14,373,444 可训练 | 11,127,108 可训练 |
| Memory / 内部检测器 | 0 | 关闭 | 关闭 |

## 3. 仓库结构

```text
configs/
  stage1.yaml                         # 唯一监督配置
  stage2_rl.yaml                      # 唯一五阶段傅里叶 RL 配置
  manifests/volmem_fourier_validation37.csv
lib/
  datasets/                           # VerSe 切片、目标构造、划分门
  networks/mainline.py                # MoonViT cache + Flow 唯一网络
  networks/diffusion/                 # Flow 与 AB2 实现
  rl/fourier.py                       # 傅里叶动作、log-prob、KL
  evaluation/                         # TEAMS 2D 与 VerSe 3D 指标
  runtime/gpu.py                      # 两次 15 秒严格空闲 GPU 门
tools/
  verify_installation.py              # CPU 数据/权重/数学核验
  launch_supervised.py                # Stage 1 预检与正式启动
  train_rl.py                         # Stage 2 核心训练器
  launch_rl.py                        # Stage 2 预检与正式启动
  infer.py                            # 固定 2×4 Dev8 推理/可视化
train.py                              # Stage 1 底层训练循环
```

启动正式任务时只调用 `launch_supervised.py` 或 `launch_rl.py`，不要绕过审计入口直接调用底层训练器。

## 4. 环境

已验证环境为 Python 3.11、PyTorch 2.4.1 + CUDA 12.1。先安装与本机 CUDA 匹配的 PyTorch，再安装其余依赖：

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

启动器还要求系统提供 `nvidia-smi`。训练和推理都会对指定物理 GPU 连续采样两次，间隔 15 秒；两次都必须满足：显存占用不超过 20 MiB、利用率为 0、compute app 列表为空。

建议先定义本机路径，后续命令直接复用：

```bash
PYTHON=/path/to/python
DATA_ROOT=/path/to/sagittal_2d_fixed
SLICE_MANIFEST="$DATA_ROOT/manifests/slice_manifest.csv"
MOONVIT_CACHE=/path/to/sagittal_moonvit_cache
STAGE1_SOURCE=/path/to/pure2d_step40000.pt
STAGE2_SOURCE=/path/to/moonvit_stage1_step19000.pt
CASE_METADATA=/path/to/case_metadata.csv
```

模型和缓存不要复制进 Git。`artifacts/checkpoints/...` 只是配置中的说明性默认位置；正式启动器始终使用命令行传入的绝对路径。

## 5. 数据与缓存

### 5.1 固定划分

| 用途 | 口径 | 数量 | 是否更新参数 |
|---|---|---:|---|
| 监督/RL 训练 | Train72 | 72 cases / 13,261 rows | 是 |
| RL 选择 | Val37 | 37 cases / 5,914 rows | 否；固定抽取 80-row panel |
| 最终开发评估 | Dev8 | 8 cases / 1,123 rows | 否 |
| 锁定集 | 010/011/013 | 3 cases | 禁止读取 |

`slice_manifest.csv` 至少包含：

```text
split,case_id,slice_idx,image_path,mask_path
```

`image_path` 和 `mask_path` 可以是绝对路径，也可以相对 `DATA_ROOT`。同一 case 的 `slice_idx` 必须唯一、连续；启动时会严格核对 case 数、row 数和锁定病例。

完整 3D 评估还需要 `CASE_METADATA` CSV，至少包含：

```text
case_id,image_nii_path,mask_nii_path,canonical_shape
```

其中 `canonical_shape` 使用例如 `48x487x633` 的格式。对应 GT 旁还必须存在签名 VerSe centroid JSON。

### 5.2 MoonViT 缓存

缓存目录固定为：

```text
$MOONVIT_CACHE/{training,validation,test}/<case_id>/x<slice_idx:04d>.npz
```

正式规格：

| 字段 | 固定值 |
|---|---|
| MoonViT 输入 | 448×448 |
| patch size | 14 |
| 使用层 | `layer_18` |
| 通道数 | 1152 |
| dtype | float16 |
| 融合模式 | `center_only` |
| 常见 token 网格 | 32×32；非方形图像保留有效宽高映射 |

每个 `.npz` 还必须提供 `grid_hw/orig_hw/resized_hw/padded_hw/pad/scale/patch_size/input_size/normalization/checkpoint/case_id/slice_idx`。数据加载器会核对路径、shape、dtype、坐标空间和元数据，缺失时直接失败，不使用匹配子集降级。

现有缓存的来源标识为：

```text
Eagle/Embodied/work_dirs/1232_final_locany_full_more10000/checkpoint-3000
```

如果原始 MoonViT 权重不可用，可以继续使用已签名缓存做训练和评估；不能用另一份权重重建缓存后仍声称是同一实验血缘。

### 5.3 先做 CPU 核验

```bash
"$PYTHON" tools/verify_installation.py \
  --config configs/stage2_rl.yaml \
  --data-root "$DATA_ROOT" \
  --slice-manifest "$SLICE_MANIFEST" \
  --moonvit-cache "$MOONVIT_CACHE" \
  --checkpoint "$STAGE2_SOURCE" \
  --checkpoint-sha256 a337ba1566fe423c10a82dc4c08f8d6936ce8fc49ff1d61c8f735435854a337f \
  --expected-step 19000
```

`status=PASS` 才能继续。它会读取真实 foreground 样本、核对 Train72 与锁定病例、严格加载全部 156 个 checkpoint 张量、检查参数量，并验证五阶段进度与傅里叶投影。

## 6. Stage 1：监督训练

唯一配置为 `configs/stage1.yaml`。它从兼容的 absolute step40000 权重做 weights-only 迁移，建立新的 AdamW；不会继承旧优化器状态。

固定 source：

```text
step = 40000
SHA256 = 641445aaed9a7ea3acfc8d50833d0ede9cc454bfe2cf34bea2ff0464d33e929b
```

固定训练合同：

| 项目 | 值 |
|---|---:|
| optimizer | AdamW |
| learning rate | 1e-5 |
| warmup | 1,000 updates |
| batch size | 48 |
| local updates | 60,000 |
| checkpoint | 每 1,000；保留 12 个 |
| 里程碑 | 5k / 10k / 20k / 40k / 60k |
| 可训练参数 | Flow + 特征替换器，共 14,373,444 |
| 初始化 | GT Route-B box-octagon + jitter |

### 6.1 两步预检

输出目录必须不存在：

```bash
STAGE1_PREFLIGHT=data/outputs/stage1_preflight_YYYYMMDD

"$PYTHON" tools/launch_supervised.py \
  --mode preflight \
  --preflight-steps 2 \
  --gpu 6 \
  --source-checkpoint "$STAGE1_SOURCE" \
  --data-root "$DATA_ROOT" \
  --slice-manifest "$SLICE_MANIFEST" \
  --moonvit-cache "$MOONVIT_CACHE" \
  --output-root "$STAGE1_PREFLIGHT"
```

预检通过时：

- `PURE2D_TRAINING_LAUNCH.json` 为 `COMPLETED`；
- source step/SHA、参数量、Train72/Dev8 身份通过；
- step1/2 loss 与更新有限；
- Flow 和特征替换器都至少有一个张量更新；
- 不出现 missing/unexpected/shape mismatch。

### 6.2 正式训练

正式运行必须引用同一输入身份的已完成预检，并写入另一个新目录：

```bash
STAGE1_RUN=data/outputs/stage1_60k_YYYYMMDD

"$PYTHON" tools/launch_supervised.py \
  --mode train \
  --gpu 6 \
  --source-checkpoint "$STAGE1_SOURCE" \
  --data-root "$DATA_ROOT" \
  --slice-manifest "$SLICE_MANIFEST" \
  --moonvit-cache "$MOONVIT_CACHE" \
  --preflight-output "$STAGE1_PREFLIGHT" \
  --output-root "$STAGE1_RUN"
```

发布的 Stage 2 source 是一次正式 Stage 1 运行留下的最近完整锚点：

```text
step = 19000
SHA256 = a337ba1566fe423c10a82dc4c08f8d6936ce8fc49ff1d61c8f735435854a337f
```

它是可严格加载的监督锚点，不代表 Stage 1 已完成 60k。不要覆盖原运行，也不要把不完整文件当作恢复源。

## 7. 固定 2×4 部署推理

部署轨迹是不可变的：

| 外阶段 | residual fraction | 输入进度 `s` | AB2 NFE |
|---:|---:|---:|---:|
| 1 | 0.6667 | 0.0 | 4 |
| 2 | 1.0 | 0.6667 | 4 |

累计进度按 `s_next = s + (1-s) × fraction` 更新。两个阶段使用各自的随机初始 latent；RL 对齐测试与生产函数必须使用相同 latent 定义。第二阶段在更新后的轮廓位置重新取特征。

以下都属于错误实验：把 5 阶段训练轨迹当作部署、把 4 次 AB2 调用当作 4 个动作、漏传 `s`、第二阶段复用旧位置特征，或在推理时打开傅里叶探索。

## 8. Stage 2：五阶段傅里叶 full-extrap GRPO

唯一配置为 `configs/stage2_rl.yaml`，唯一 source 是上述 step19000。MoonViT 特征替换器逐 tensor 冻结，只更新 11,127,108 个 Flow 参数。

### 8.1 五阶段训练网格

| 动作 | residual fraction | 实际输入进度 `s` | Fourier sigma（原图 px） | 每阶段 NFE |
|---:|---:|---:|---:|---:|
| a0 | 0.2 | 0.0 | 0.8 | 4 |
| a1 | 0.25 | 0.2 | 0.7 | 4 |
| a2 | 0.3333 | 0.4 | 0.6 | 4 |
| a3 | 0.5 | 0.59998 | 0.5 | 4 |
| a4 | 1.0 | 0.79999 | 0.4 | 4 |

每个阶段只有一个 RL 动作：8 个低频正交傅里叶系数生成标量场，再沿轮廓法向形成几何扰动。一次训练 rollout 有 5 个动作和 20 NFE；`outer_log_count_mean` 必须为 5，不是 20。

### 8.2 full-extrap 奖励归因

对于阶段起点 `x_start`、阶段终点 `x_end` 和 residual fraction `f`，该阶段的完整外推轮廓为：

```text
x_full = x_start + (x_end - x_start) / f
```

每个动作都与同阶段、同 latent 的确定性 Flow 外推端点比较，因此 credit map 为 `[0,1,2,3,4]`。这不是把最终奖励复制五次；每个阶段拥有自己的反事实完整端点评分。

主评分权重固定为：

```text
0.10 × boundary + 0.40 × Dice + 0.40 × IoU + 0.10 × distance
```

另外有权重 0.06、margin 0.5 px、上限 1.5 px、q95 的 bounded burr penalty。奖励是相对同一监督模型确定性 rollout 的改进量。

### 8.3 GRPO/PPO 固定值

| 项目 | 值 |
|---|---:|
| K rollouts | 8 |
| PPO epochs | 2 |
| clip | 0.05 |
| approx-KL stop | 0.002 |
| explicit KL beta | 0.01 |
| advantage std floor / clip | 0.1 / 2.0 |
| grad clip | 0.25 |
| learning rate | 4e-8 |
| 有效更新目标 | 10,000 |
| 诊断/保存间隔 | 50 updates |
| 固定选择面板 | Val37 中 80 rows |

空轮廓 batch 会跳过且不计有效 RL step。approx-KL 超门只结束当前 PPO epoch，不结束正式训练。训练前 step0 面板必须至少达到 IoU 0.45、Dice 0.60，否则说明 source、缓存或路径错了，应立即停止。

### 8.4 两步预检

签名 Val37 清单是 `configs/manifests/volmem_fourier_validation37.csv`，共 5,914 行，路径相对 `DATA_ROOT`，SHA256 为：

```text
24a4f19651edb5d187029f0255e2b59f9dce40f320ee29c14b709e8e92e6e6ad
```

```bash
STAGE2_PREFLIGHT=data/outputs/stage2_rl_preflight_YYYYMMDD

"$PYTHON" tools/launch_rl.py \
  --mode preflight \
  --gpu 6 \
  --python "$PYTHON" \
  --source-checkpoint "$STAGE2_SOURCE" \
  --data-root "$DATA_ROOT" \
  --slice-manifest "$SLICE_MANIFEST" \
  --moonvit-cache "$MOONVIT_CACHE" \
  --output "$STAGE2_PREFLIGHT" \
  --preflight-output "$STAGE2_PREFLIGHT"
```

预检必须同时满足：五阶段训练 helper 与生产采样函数 `max_abs≤1e-5`；固定 2×4 部署 helper 与生产函数 `max_abs≤1e-5`；step1/2 数值有限；每步 5 个动作日志；Flow 更新；3,246,336 个冻结参数零变化；checkpoint step2 完整。

### 8.5 正式训练与恢复

```bash
STAGE2_RUN=data/outputs/stage2_rl_10k_YYYYMMDD

"$PYTHON" tools/launch_rl.py \
  --mode train \
  --gpu 6 \
  --python "$PYTHON" \
  --source-checkpoint "$STAGE2_SOURCE" \
  --data-root "$DATA_ROOT" \
  --slice-manifest "$SLICE_MANIFEST" \
  --moonvit-cache "$MOONVIT_CACHE" \
  --preflight-output "$STAGE2_PREFLIGHT" \
  --output "$STAGE2_RUN"
```

中断后只能从完整 RL checkpoint 恢复到**新输出目录**：

```bash
STAGE2_RESUME=data/outputs/stage2_rl_resume_YYYYMMDD

"$PYTHON" tools/launch_rl.py \
  --mode train \
  --gpu 6 \
  --python "$PYTHON" \
  --source-checkpoint "$STAGE2_SOURCE" \
  --resume-checkpoint /path/to/previous/checkpoints/latest.pt \
  --data-root "$DATA_ROOT" \
  --slice-manifest "$SLICE_MANIFEST" \
  --moonvit-cache "$MOONVIT_CACHE" \
  --preflight-output "$STAGE2_PREFLIGHT" \
  --output "$STAGE2_RESUME"
```

启动器拒绝复用任何已有目录，并把 config、trainer、Python、source 和选择清单的哈希写入 `manifest.json`。

## 9. 推理、评估与可视化

先用少量前景切片验证前向链路。`--smoke-slices N` 会按 Dev8 manifest 顺序选取前 N 张含 GT 前景的切片；结果标记为 `SMOKE_PASS`、`reportable=false`，不会生成或冒充完整 3D 指标：

```bash
SMOKE_OUT=data/outputs/inference_smoke_YYYYMMDD

"$PYTHON" tools/infer.py \
  --gpu 6 \
  --checkpoint "$STAGE2_SOURCE" \
  --result-dir "$SMOKE_OUT" \
  --data-root "$DATA_ROOT" \
  --slice-manifest "$SLICE_MANIFEST" \
  --case-metadata "$CASE_METADATA" \
  --locate-feat-cache-root "$MOONVIT_CACHE" \
  --batch-size 1 \
  --num-workers 0 \
  --smoke-slices 2
```

完整 Dev8 使用同一命令并把 `--smoke-slices` 设为 `0`：

```bash
FULL_OUT=data/outputs/inference_dev8_YYYYMMDD

"$PYTHON" tools/infer.py \
  --gpu 6 \
  --checkpoint /path/to/final_or_milestone.pt \
  --result-dir "$FULL_OUT" \
  --data-root "$DATA_ROOT" \
  --slice-manifest "$SLICE_MANIFEST" \
  --case-metadata "$CASE_METADATA" \
  --locate-feat-cache-root "$MOONVIT_CACHE" \
  --batch-size 16 \
  --num-workers 4 \
  --smoke-slices 0
```

完整运行必须处理 1,123 张 Dev8 切片，严格加载 14,373,444 参数，并输出：

- foreground IoU、Dice、mBoundF、HD95；
- PQ、SQ、RQ、TP/FP/FN；
- VerSe 3D Dice、ID rate、maxHD、dmean；
- Route-B 初始化、GT、预测和 FP/FN 误差可视化；
- 输入/代码/checkpoint SHA、GPU 门、运行时间和完整身份 JSON。

RL 训练的五阶段轨迹从不用于这里；`tools/infer.py` 会强制关闭 RL 探索、上一切片初始化和数据增强，并固定 2×4 AB2。

## 10. 输出验收与故障处理

每个 checkpoint 至少核对：

1. 文件大小稳定且可计算 SHA256；
2. 156 个张量 key/shape 与 source 严格一致；
3. 全部浮点张量有限；
4. Stage 1 中 Flow 与替换器都更新；Stage 2 中只有 Flow 更新；
5. 日志的 reward/loss/grad/ratio/approx-KL 有限；
6. Stage 2 `outer_log_count_mean=5`；
7. 五阶段训练对齐和 2×4 部署对齐持续为 PASS；
8. `manifest` 的状态、输入哈希、物理 GPU 和输出目录正确。

遇到 OOM、非有限值、严格加载失败、冻结张量变化、进程消失或数据门失败时：保留现有日志和 manifest，不覆盖、不自动重试、不删除 checkpoint。修复后使用新输出目录重新预检。

短时 loss 波动不是早停依据。Dev8 不参与调参；最终论文表必须等 Stage 2 完成后，再用完整 Dev8 和相同 `oracle_prompt` 口径一次性冻结。`fully_automatic` 与 `oracle_prompt` 永远不能混榜。

## 11. 新接手者清单

- [ ] 克隆仓库并安装 `requirements.txt`。
- [ ] 准备 Train72/Val37/Dev8 数据、slice manifest、case metadata 和 MoonViT cache。
- [ ] 核对 step40000 与 step19000 两个 source SHA。
- [ ] 运行 `tools/verify_installation.py` 并得到 `PASS`。
- [ ] 选择真正空闲 GPU；不终止、不抢占其他进程。
- [ ] Stage 1：新目录做 2-step preflight，再启动唯一正式运行。
- [ ] Stage 2：固定 step19000 source，新目录做 2-step preflight，再做 10k 有效更新。
- [ ] 恢复训练时使用完整 checkpoint，并写入新目录。
- [ ] 先跑非报告 smoke inference，再跑完整 Dev8。
- [ ] 检查 manifest、SHA、冻结参数和可视化后再汇报结果。

本仓库不包含历史对比模型、探索配置、旧 RL 分支或运行中状态记录；Git 中的两个配置和五个公开入口就是全部正式流程。
