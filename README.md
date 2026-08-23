# DiffusionSnake：VerSe 纯 2D 双阶段轮廓主线

本仓库当前唯一推荐的正式路线是：

> **冻结的 MoonViT layer-18 离线特征 → 监督训练 MoonViT 特征融合器与 Flow → 冻结 MoonViT/融合器，仅对继承的 Flow 做傅里叶 full-extrap GRPO → 按同一双阶段、8 NFE 方式推理。**

本文是可直接交接的运行手册。阅读者不需要预先了解本项目；按“数据检查 → MoonViT 缓存 → 监督阶段 → 强化学习阶段 → Dev8 最终评估”的顺序执行即可。

## 1. 先记住五条边界

1. **主线是逐张切片的纯 2D 模型。** 当前正式训练不读取相邻切片，不使用 Memory，不使用专门的 3D 卷积或 3D Transformer。
2. **MoonViT 是唯一推荐的视觉特征来源。** SAM2、ResNet34、UNETR、Swin 等文件只服务于对比实验，不是正式预训练入口。
3. **部署与所有正式评估固定使用同一套两阶段推理。** 两个外阶段，每阶段 4 次 AB2 内部函数求值，总计 8 NFE。RL 可以采用部署对齐的 2 动作轨迹，也可以采用下文签名锁定的 5 动作训练轨迹；后者只用于优化，绝不能改变报告和部署口径。
4. **强化学习只更新 Flow。** MoonViT 离线编码器和已学得的高分辨率特征融合器全部冻结。
5. **研究主线使用 GT box 与 GT anatomical class。** 它是 oracle-prompt 口径；外部检测器结果必须单列，不能与 oracle-prompt 结果混在同一排行榜。

## 2. 官方训练链

```text
VerSe Train72 原始矢状位切片
        │
        ├─ MoonViT 448×448、patch=14、layer-18 离线编码
        │        └─ data/sagittal_moonvit_cache
        │
        ├─ Stage 1：监督训练
        │        ├─ 训练：MoonViT 特征融合器 + Flow
        │        ├─ 冻结：离线 MoonViT 编码器
        │        └─ 当前官方锚点：step_19000.pt
        │
        ├─ Stage 2：傅里叶 full-extrap GRPO
        │        ├─ 训练：继承的 Flow
        │        ├─ 冻结：MoonViT + 特征融合器
        │        ├─ 训练轨迹：2 动作部署对齐基线 / 5 动作受控候选
        │        ├─ 统一评估：固定 2×4 AB2 / 8 NFE
        │        └─ 目标：10000 个有效 RL 更新
        │
        └─ 最终推理与评估
                 ├─ 同一 Route-B 初始化
                 ├─ 同一 2×4 AB2 / 8 NFE
                 └─ 完整非锁定 Dev8=1123
```

“Stage 1/Stage 2”指训练流程的两个阶段；每次模型推理内部也有两个“外阶段”。为避免混淆，本文将后者始终称为**外阶段 1/2**。

## 3. 固定环境与数据口径

### 3.1 仓库与 Python

```bash
cd /home/medteam/Zhrch/DiffusionSnake-12-30

# 强化学习正式使用的解释器
/home/medteam/miniconda3/envs/sam1_lgz/bin/python --version
```

不要在另一个仓库副本中启动训练。不要覆盖已存在的输出目录；恢复或重跑必须建立新目录，并在 manifest 中记录来源。

### 3.2 数据划分

| 用途 | 固定口径 | 数量 | 作用 |
|---|---:|---:|---|
| 监督/RL 训练 | Train72 | 72 cases / 13,261 rows | 参数更新 |
| RL 选择集 | Val37 | 37 cases / 5,914 rows | 固定 80-row panel 做训练中诊断与选择 |
| 最终开发评估 | Dev8 | 1,123 rows | 训练完成后一次性统一评估 |
| 禁止访问 | locked010/011/013 | 3 cases | 任何训练、选择和调参都不得读取 |

接手者必须先核对配置中的 case 数和 row 数。任何数量不一致、case 泄漏或 locked case 出现，都应停止运行，而不是自动修补。

### 3.3 输入与提示

- 图像：VerSe 矢状位单帧。
- 初始化：Route-B GT bounding-box jitter，box 转 12 点八边形，再均匀重采样为 128 个轮廓点。
- 轮廓坐标：在 `128×128` 轮廓/Flow 网格中演化；MoonViT 高分辨率特征经可学习融合后注入该网格。
- 类别：GT anatomical class。
- 关闭项：Memory、internal detector、邻帧输入、专门 3D 模块。

## 4. Stage 0：唯一的 MoonViT 特征预处理

### 4.1 正式特征规格

| 项目 | 固定值 |
|---|---|
| 输入分辨率 | `448×448` |
| patch size | `14×14` |
| token 网格 | 典型为 `32×32`，非正方形有效区域按原始宽高映射 |
| 特征层 | `layer_18` |
| hidden dimension | `1152` |
| 融合模式 | `center_only` |
| 缓存目录 | `data/sagittal_moonvit_cache` |

缓存约 84 GB，覆盖训练、验证和测试切片。正式训练只读取 `layer_18`，即使个别缓存文件还保存其他层，也不得把它们接入正式主线。

### 4.2 缓存完整性检查

开始训练前至少检查：

```bash
test -d data/sagittal_moonvit_cache
find data/sagittal_moonvit_cache -type f -name '*.npz' | wc -l
```

然后用配置/启动器的审计门核对：样本顺序、原图路径、特征 key、shape、dtype、有限值和数据划分。不要用缺失切片的匹配子集代替完整集合。

### 4.3 重要的可复现性说明

当前缓存元数据记录的原始 MoonViT/LocateAnything checkpoint 路径为：

```text
Eagle/Embodied/work_dirs/1232_final_locany_full_more10000/checkpoint-3000
```

当前仓库中该原始 checkpoint 目录并不存在。因此：

- **继续训练/评估可以直接使用现有签名缓存。**
- **若要从原图重新生成全部缓存，必须先恢复完全相同的 checkpoint。**
- 不得用 smoke 权重或其他 MoonViT 权重重新编码后，仍声称与现有主线同一血缘。

## 5. Stage 1：MoonViT + Flow 监督训练

### 5.1 唯一推荐配置

```text
configs/volmem/depth_sweep/
pure2d_mainline_l6_f256_routeb_v410_moonvit_cached_flowtune60k_from40000.yaml
```

配置 SHA256：

```text
f8095f2754e2c3f0b94c4d7fdedc6b880bca5de40231f2603561ed557db3caaf
```

唯一正式启动器：

```text
tools/volmem/depth_sweep_tools/run_pure2d_moonvit_cached_flowtune60k_training.py
```

### 5.2 起点与参数身份

监督阶段从以下纯 2D robust-box 模型严格加载：

```text
data/outputs/depth_sweep/
pure2d_mainline_l6_f256_routeb_v410_robustbox_batch48_resume31000_v1/
checkpoints/step_40000.pt
```

SHA256：

```text
641445aaed9a7ea3acfc8d50833d0ede9cc454bfe2cf34bea2ff0464d33e929b
```

| 部分 | 参数量 | Stage 1 状态 |
|---|---:|---|
| 总模型 | 14,373,444 | — |
| 继承 Flow | 11,127,108 | 训练 |
| MoonViT 特征融合/替换器 | 3,246,336 | 训练 |
| 离线 MoonViT 编码器 | 不进入训练图 | 冻结、只读缓存 |

这里没有 Memory 参数，也没有内部检测器参数。

### 5.3 监督训练超参数

| 项目 | 固定值 |
|---|---|
| batch size | 48 |
| optimizer | 配置中的 AdamW |
| 初始 LR | `1e-5` |
| warmup | 1,000 updates |
| 本阶段目标 | 60,000 local updates |
| checkpoint | 每 1,000 updates |
| Flow noise scale | 1.0 |
| 外阶段 | `[0.6667, 1.0]` |
| 每个外阶段 | AB2、4 NFE |
| 总推理计算 | 8 NFE |

### 5.4 如何启动一个全新的 Stage 1

1. 核对配置和 source checkpoint 的 SHA256。
2. 确认目标输出目录不存在。
3. 先运行启动器自带的失败关闭预检。
4. 连续两次、间隔至少 15 秒确认同一张 GPU 无进程、显存不高于 20 MiB、利用率为 0。
5. 只启动一个正式根进程，并检查 manifest、首个有限 loss/grad 和首个完整 checkpoint。

入口：

```bash
/usr/bin/python \
  tools/volmem/depth_sweep_tools/run_pure2d_moonvit_cached_flowtune60k_training.py \
  --help
```

先阅读 `--help`，使用启动器提供的正式模式；不要绕过启动器直接调用底层 trainer，因为启动器负责哈希、数据、参数量、严格加载和输出唯一性检查。

### 5.5 当前保留的官方监督锚点

```text
data/outputs/depth_sweep/
pure2d_mainline_l6_f256_routeb_v410_moonvit_cached_flowtune60k_from40000_v1/
checkpoints/step_19000.pt
```

SHA256：

```text
a337ba1566fe423c10a82dc4c08f8d6936ce8fc49ff1d61c8f735435854a337f
```

该运行在 step 19,217 之后遇到 CUDA OOM，原正式进程已退出；`step_19000.pt` 是 OOM 前完整、有限且可严格加载的最近锚点，也是当前 Stage 2 的唯一 source。不要把失败的 run 状态描述成“监督训练已完成 60k”；可以准确描述为“获得了可恢复的 step19000 正式锚点”。

若未来补完 Stage 1，必须从该完整锚点建立**新输出目录**恢复，并重新审计 optimizer/scheduler/RNG；不得覆盖原目录。

## 6. 模型内部的两阶段推理：最重要的一节

监督训练、正式评估和部署都必须执行下面完全相同的 2×4 过程。部署对齐的 2 动作 RL 也直接使用该过程。固定 5 动作 RL 是仅存在于 Stage 2 优化期的时间网格课程；它训练结束后的所有指标仍必须回到本节的 2×4 路径。

### 6.1 外阶段 1：粗到中等尺度

1. 从 Route-B box-octagon 得到 128 个初始轮廓点 `x0`。
2. 在 `x0` 位置采样 MoonViT 融合特征和类别条件。
3. 采样本阶段的随机初始 latent，沿 fraction `0.6667` 的时间段积分。
4. 向 Flow 显式传入累计阶段进度 `s=0.0`。
5. 使用 AB2 求解器；这一外阶段调用 Flow 4 次，即 4 NFE。
6. 得到中间轮廓 `x1`。

### 6.2 外阶段 2：在更新后的位置精修

1. **必须在 `x1` 的新位置重新采样视觉特征。**
2. 以 `x1` 为本阶段起点，采样本阶段的随机初始 latent，沿 fraction `1.0` 的时间段继续积分。
3. 向 Flow 显式传入累计阶段进度 `s=0.6667`。
4. 再执行 4 次 AB2 Flow 调用，即 4 NFE。
5. 输出最终轮廓 `x2`。

### 6.3 为什么是“2×4=8 NFE”

```text
外阶段 1：4 次 Flow 函数求值
外阶段 2：4 次 Flow 函数求值
总计：      8 NFE
```

外阶段不是简单重复。第二阶段看到的是第一阶段更新后的轮廓和重新采样的特征，因此负责局部边界精修。

阶段进度按 `s_next = s + (1-s) × fraction` 更新，因此正式两阶段的输入条件固定为：

```text
stage 0: fraction=0.6667, s=0.0
stage 1: fraction=1.0000, s=0.6667
```

`flow_2d_s_conditioning=true` 时，缺失 `s` 不等价于传入 0。RL 手写 rollout 必须把同一 `s` 传给该阶段内全部 4 次 AB2 速度预测。

训练与推理不允许出现以下错位：

- 使用 5 动作训练，却用 5 动作推理结果冒充固定 2×4 部署结果；
- 没有在 manifest、日志和结果表中明确区分“5 动作训练轨迹”与“2×4 部署轨迹”；
- 训练把 AB2 的每个内部函数求值当成独立 RL 动作；
- 第二阶段仍使用初始轮廓位置的旧特征；
- 正式推理传 `s=[0.0,0.6667]`，RL rollout 却遗漏阶段进度条件；
- RL 使用一套随机 latent，部署又换成另一套初始化定义。

## 7. Stage 2：傅里叶 full-extrap GRPO

### 7.1 部署对齐 2 动作基线文件

配置：

```text
configs/rl/pure2d_moonvit_flow_grpo_2x4_fourier_full_extrap_v2.yaml
SHA256 345876b0a53b790a692c0d4ff35f96f83a982cfe3b3f1afd5591d87cd9839d13
```

trainer：

```text
tools/rl/grpo_train_pure2d_moonvit_2x4_fourier_full_extrap_v2.py
SHA256 79c0689431b9b7dac128d78c441711e5614db56a72e50d53131242161e52290f
```

launcher：

```text
tools/rl/run_pure2d_moonvit_flow_grpo_ab2_2x4_fourier_v2.py
SHA256 16a296b13489f141b0dc4e21f69ef875d2a0a7033175549b82a8b1d226d29914
```

RL source 固定为上一节的 `step_19000.pt`，SHA256 必须等于 `a337ba15...a337f`。

### 7.2 哪些参数会更新

| 部分 | 参数量 | Stage 2 状态 |
|---|---:|---|
| 继承 Flow | 11,127,108 | **训练** |
| MoonViT 特征融合/替换器 | 3,246,336 | 冻结 |
| MoonViT 离线编码器 | 不进入训练图 | 冻结 |
| 总模型 | 14,373,444 | — |

预检和每个里程碑都要检查：Flow 至少一个张量发生变化；所有非 Flow 张量变化数必须为 0；所有张量必须有限。

### 7.3 傅里叶动作是什么

每个外阶段只采样**一个低频傅里叶轮廓扰动动作**。扰动直接作用在轮廓几何上，不是额外的独立策略网络，也不是在 Flow 均值周围随意加 Gaussian 动作。

正式固定值：

| 项目 | 外阶段 1 | 外阶段 2 |
|---|---:|---:|
| 低频 modes | 8 | 8 |
| 系数 sigma（px） | 0.5 | 0.4 |
| 预期逐点 RMS（px） | 0.125 | 0.100 |

一条 rollout 有两个外阶段，因此只有两个 RL 动作：`a0` 和 `a1`。每个外阶段内部的 4 次 AB2 调用属于数值求解，不是 4 个额外动作。

### 7.4 full-extrap 如何归因

两阶段动作 credit map 固定为：

```text
[0, 1]
```

含义是：最终完整轮廓的奖励同时回传到两个真实外阶段动作。4 次 AB2 内部函数求值只属于数值积分，不单独获得动作编号。

### 7.5 奖励函数

最终奖励使用固定的重叠/边界复合项：

```text
reward = 0.1 × boundary
       + 0.4 × Dice
       + 0.4 × IoU
       + 0.1 × distance
       + 0.06 × bounded_burr_aux
```

- 主奖励权重严格为 `[0.1, 0.4, 0.4, 0.1]`。
- burr 辅助项上限 1.5，margin 0.5，使用 q95。
- 空轮廓批次只记录 skip，不计入“有效 RL step”；连续空批次超过失败关闭门时应停止并排查数据。

### 7.6 GRPO/PPO 固定参数

| 项目 | 固定值 |
|---|---:|
| group size K | 8 |
| PPO epochs | 2 |
| clip | 0.05 |
| approx-KL stop | 0.002 |
| explicit KL beta | 0.01 |
| advantage floor | 0.1 |
| advantage clip | 2.0 |
| grad clip | 0.25 |
| learning rate | `4e-8` |
| 有效更新目标 | 10,000 |
| checkpoint/评估间隔 | 250 |
| 里程碑 | 250/500/1000/2000/5000/10000 |

approx-KL 超门只终止当前 PPO epoch，不应停止整个 10k 正式训练。

### 7.7 训练选择与最终评估必须分离

- 训练期间只使用 Val37 中固定、签名锁定的 80-row panel 做诊断。
- Dev8=1123 不参与超参数选择，不根据 Dev8 反复改配置。
- locked010/011/013 永远禁止访问。
- 10k 完成后，才在完整 Dev8 上冻结最终结果。

### 7.8 固定 5 动作训练变体（当前受控候选）

固定 5 动作分支和 2 动作基线使用完全相同的 MoonViT `step_19000.pt`、Train72、Val37 固定 panel、Flow-only 更新、奖励函数和 PPO 参数。唯一实质变化是：**训练 rollout 使用更密的 5 动作时间网格，但所有报告和部署仍固定使用 2×4 AB2。**

正式文件及 SHA256：

| 角色 | 文件 | SHA256 |
|---|---|---|
| config | `configs/rl/pure2d_moonvit_flow_grpo_5step_fourier_train_2x4_deploy_v1.yaml` | `e358ca41f51834269b1b37faf5d9a6a5f066a8b7402156b8c9f3955990ff8b63` |
| trainer | `tools/rl/grpo_train_pure2d_moonvit_5step_fourier_train_2x4_deploy_v1.py` | `8772fa80aa12b0e6e8757e762b558c63c310c36ac287e444abd232467b46122f` |
| launcher | `tools/rl/run_pure2d_moonvit_flow_grpo_5step_fourier_train_2x4_deploy_v1.py` | `3bcebe6e5fad0fe0d0a502096e2ec05a36240e13c9190a1ba7668eb338808e9c` |

训练时间网格：

| RL 动作 | residual fraction | 累计阶段进度 `s` | 名义绝对进度增量 | Fourier sigma（px） | 预期逐点 RMS（px） |
|---:|---:|---:|---:|---:|---:|
| `a0` | 0.2000 | 0.0000 | 0.20 | 0.8 | 0.200 |
| `a1` | 0.2500 | 0.2000 | 0.20 | 0.7 | 0.175 |
| `a2` | 0.3333 | 0.4000 | 0.20 | 0.6 | 0.150 |
| `a3` | 0.5000 | 0.59998 | 0.20 | 0.5 | 0.125 |
| `a4` | 1.0000 | 0.79999 | 0.20 | 0.4 | 0.100 |

fraction 是“当前剩余时间”的比例，因此 `[0.2,0.25,0.3333,0.5,1.0]` 对应约 20%/40%/60%/80%/100% 的累计进度。运行时必须使用日志记录的实际 `s=[0.0,0.2,0.4,0.59998,0.79999]`，不能把近似值默默改成另一套网格。

每个外阶段只拥有一个 8-mode 低频傅里叶动作；阶段内部 4 次 AB2 Flow 调用仍只是数值求解。因此：

- 每条训练 rollout：5 个 RL 动作、`5×4=20 NFE`；
- full-extrap credit map：`[0,1,2,3,4]`；
- 每条部署/报告 rollout：2 个外阶段、`2×4=8 NFE`；
- `outer_log_count_mean` 在该训练分支必须等于 5，不能等于 20；
- MoonViT 缓存编码器和 3,246,336 参数融合器冻结，只更新继承 Flow 的 11,127,108 参数。

5 动作训练消耗高于 2 动作训练。比较两条分支时既要报告有效 RL update，也要报告累计训练 NFE；不能只按 update 数宣称哪条更高效。

## 8. 当前 RL 运行状态与接手命令

### 8.1 2 动作部署对齐基线：有效预检

```text
data/outputs/rl/
pure2d_moonvit_flowonly_grpo_ab2_2x4_fourier_full_extrap_from19000_preflight_v3
```

预检证据：

- manifest：`COMPLETED`
- source：156 tensors 严格加载
- 完整两阶段 AB2 deployment alignment：`max_abs=0`
- 阶段条件：`s=[0.0, 0.6667]`
- step0 固定面板基线：IoU `0.669590`、Dice `0.793519`、mBoundF `0.694760`、有效轮廓 `132`
- step 1/2：reward、loss、grad、ratio 全部有限
- changed Flow tensors：138
- changed frozen tensors：0
- non-finite tensors：0
- preflight checkpoint SHA256：`9454f1b5e4954711a171883a0f531289e2d2ea98e1f3040b24c73f941df83aea`

旧 `preflight_v2` / 正式 `v2` 漏传两阶段 `s`，其所谓对齐只覆盖单阶段 4-NFE；固定面板 Dice 因此错误地降到 `0.3472`。它们只保留为失败证据，严禁续训、选模或写入论文结果。

### 8.2 2 动作部署对齐基线：正式输出

```text
data/outputs/rl/
pure2d_moonvit_flowonly_grpo_ab2_2x4_fourier_full_extrap_from19000_v3
```

截至 2026-08-20，正式任务已在物理 GPU1 启动，manifest 为 `STARTED`。完整两阶段 AB2 对齐 `max_abs=0`，`s=[0.0,0.6667]`；step0 基线与预检一致，前两个有效 step 的 reward、reward_std、policy loss、grad、ratio、approx-KL 均为有限值。

**如果该目录已存在，绝对不要启动第二份。** 接手者只应读取 manifest、进程树、日志和 checkpoint。

常用只读检查：

```bash
cd /home/medteam/Zhrch/DiffusionSnake-12-30

cat data/outputs/rl/pure2d_moonvit_flowonly_grpo_ab2_2x4_fourier_full_extrap_from19000_v3/manifest.json

tail -n 40 \
  data/outputs/rl/pure2d_moonvit_flowonly_grpo_ab2_2x4_fourier_full_extrap_from19000_v3/posttrain_rl_fourier_outer_action/logs.jsonl

find data/outputs/rl/pure2d_moonvit_flowonly_grpo_ab2_2x4_fourier_full_extrap_from19000_v3/checkpoints \
  -maxdepth 1 -type f -name '*.pt' -printf '%f %s bytes\n' | sort -V
```

trainer 日志为正式输出同级的 `_v3.launch.log`，nohup 启动器日志为 `_v3.launcher.log`。检查时重点搜索：`Traceback`、`CUDA out of memory`、`non-finite`、`strict mismatch`。

### 8.3 2 动作分支：只有在没有任何正式目录时才允许启动

```bash
# 一次性预检
/home/medteam/miniconda3/envs/sam1_lgz/bin/python \
  tools/rl/run_pure2d_moonvit_flow_grpo_ab2_2x4_fourier_v2.py \
  --mode preflight --gpu N

# 预检 COMPLETED 后，重新通过两次 15 秒空闲 GPU 门，再启动正式训练
nohup /home/medteam/miniconda3/envs/sam1_lgz/bin/python \
  tools/rl/run_pure2d_moonvit_flow_grpo_ab2_2x4_fourier_v2.py \
  --mode train --gpu N \
  > data/outputs/rl/pure2d_moonvit_flowonly_grpo_ab2_2x4_fourier_full_extrap_from19000_v3.launcher.log 2>&1 &
```

`N` 必须是物理 GPU 编号。启动器会再次执行文件 SHA、source、数据、参数、预检血缘和输出唯一性门。

### 8.4 5 动作受控候选：预检、正式输出与接手

已通过的一次性预检：

```text
data/outputs/rl/
pure2d_moonvit_flowonly_grpo_5step_fourier_train_2x4_deploy_from19000_preflight_v1
```

预检 manifest 为 `COMPLETED`，并确认：5 阶段训练 rollout 对齐 PASS、生产 2×4 rollout 对齐 PASS、changed Flow tensors=138、changed frozen tensors=0、non-finite tensors=0。预检 `latest.pt` SHA256 为：

```text
7d8ca7a084158c1822aa43e4404cd678fc009e77b957959dad68ced30e1f8f83
```

唯一正式输出：

```text
data/outputs/rl/
pure2d_moonvit_flowonly_grpo_5step_fourier_train_2x4_deploy_from19000_v1
```

该目录已经存在，**绝对不要启动第二份**。截至 2026-08-20，完整 checkpoint 已到 step350；固定 2×4、132 个有效轮廓的训练中诊断为 IoU `0.712568`、Dice `0.825254`、mBoundF `0.722386`。这只是 Val37 固定 panel 的中间诊断，不是完整 Dev8，也不是最终论文结果。

接手时只读检查：

```bash
cat data/outputs/rl/pure2d_moonvit_flowonly_grpo_5step_fourier_train_2x4_deploy_from19000_v1/manifest.json

tail -n 40 \
  data/outputs/rl/pure2d_moonvit_flowonly_grpo_5step_fourier_train_2x4_deploy_from19000_v1/posttrain_rl_fourier_outer_action/logs.jsonl

find data/outputs/rl/pure2d_moonvit_flowonly_grpo_5step_fourier_train_2x4_deploy_from19000_v1/checkpoints \
  -maxdepth 1 -type f -name '*.pt' -printf '%f %s bytes\n' | sort -V
```

只有在上述预检和正式目录都不存在的新复现实验中，才允许依次运行：

```bash
# 一次性预检
/home/medteam/miniconda3/envs/sam1_lgz/bin/python \
  tools/rl/run_pure2d_moonvit_flow_grpo_5step_fourier_train_2x4_deploy_v1.py \
  --mode preflight --gpu N

# 预检 COMPLETED、并重新通过两次 15 秒空闲门后启动唯一正式训练
nohup /home/medteam/miniconda3/envs/sam1_lgz/bin/python \
  tools/rl/run_pure2d_moonvit_flow_grpo_5step_fourier_train_2x4_deploy_v1.py \
  --mode train --gpu N \
  > data/outputs/rl/pure2d_moonvit_flowonly_grpo_5step_fourier_train_2x4_deploy_from19000_v1.nohup.log 2>&1 &
```

launcher 会再次校验 trainer/config/source/Val37 manifest SHA、模型参数量、Flow-only 冻结门、预检血缘、GPU 空闲门和输出唯一性。不要绕过 launcher 直接调用 trainer。

## 9. checkpoint 验收

每个正式 checkpoint，尤其是 step250，必须完成以下检查：

1. 文件写入完整，大小稳定，计算 SHA256。
2. 所有 tensor 都能读取且有限。
3. source/model tensor key 与 shape 严格匹配。
4. 只有 Flow tensor 相对 source 发生变化。
5. MoonViT 融合器保持逐 tensor 不变。
6. 日志中的 `outer_log_count_mean` 必须与所选训练分支一致：2 动作基线为 2，5 动作候选为 5；AB2 内部调用不能计为动作。
7. reward、reward_std、policy_loss、grad_norm、ratio、approx_kl 有限。
8. 对齐门继续为 PASS：2 动作分支检查训练/部署 2×4；5 动作分支同时检查 5 阶段训练 rollout 和生产 2×4 rollout。

loss 短时波动不是早停理由。只有 OOM、非有限值、进程消失、严格加载失败、冻结参数变化或数据门失败才应判为异常。

## 10. 训练完成后的统一评估

### 10.1 评估原则

- 完整非锁定 Dev8：1,123 rows。
- 使用与训练一致的 MoonViT layer-18 缓存、Route-B、2×4 AB2、8 NFE。
- 关闭傅里叶探索，执行确定性的部署路径。
- 先报告 oracle GT-box/GT-class；detector-box 另表报告。
- 2D/TEAMS 与 3D VerSe 指标同时给出，不挑单个最好数字。

### 10.2 指标

至少报告：

- foreground IoU、Dice；
- mBoundF、HD95；
- PQ、SQ、RQ、TP/FP/FN；
- VerSe 3D Dice、ID、maxHD、dmean；
- 固定病例可视化：原图、GT、预测、叠加轮廓、FP/FN 误差图。

统一的 mask-folder 评估入口：

```text
tools/comparison_benchmark/evaluate_mask_folder.py
```

当前纯 2D 推理和 Dev8 可视化入口：

```text
tools/volmem/depth_sweep_tools/run_pure2d_detector_free_inference.py
```

运行前先查看各入口的 `--help`，固定 config/checkpoint/output，并确认输出目录不存在。fully-automatic 和 oracle-prompt 结果永不混榜。

## 11. 哪些文件是正式主线，哪些不是

### 11.1 正式主线与受控 Stage 2 候选

| 阶段/角色 | 文件 |
|---|---|
| Stage 0 cache | `data/sagittal_moonvit_cache` |
| Stage 1 config | `configs/volmem/depth_sweep/pure2d_mainline_l6_f256_routeb_v410_moonvit_cached_flowtune60k_from40000.yaml` |
| Stage 1 launcher | `tools/volmem/depth_sweep_tools/run_pure2d_moonvit_cached_flowtune60k_training.py` |
| Stage 1 anchor | `...moonvit_cached_flowtune60k_from40000_v1/checkpoints/step_19000.pt` |
| Stage 2：2 动作部署对齐基线 config | `configs/rl/pure2d_moonvit_flow_grpo_2x4_fourier_full_extrap_v2.yaml` |
| Stage 2：2 动作部署对齐基线 trainer | `tools/rl/grpo_train_pure2d_moonvit_2x4_fourier_full_extrap_v2.py` |
| Stage 2：2 动作部署对齐基线 launcher | `tools/rl/run_pure2d_moonvit_flow_grpo_ab2_2x4_fourier_v2.py` |
| Stage 2：5 动作受控候选 config | `configs/rl/pure2d_moonvit_flow_grpo_5step_fourier_train_2x4_deploy_v1.yaml` |
| Stage 2：5 动作受控候选 trainer | `tools/rl/grpo_train_pure2d_moonvit_5step_fourier_train_2x4_deploy_v1.py` |
| Stage 2：5 动作受控候选 launcher | `tools/rl/run_pure2d_moonvit_flow_grpo_5step_fourier_train_2x4_deploy_v1.py` |

### 11.2 对比模型

SAM2.1、ResNet34-FPN、nnU-Net、Swin-Unet、TransUNet、UNETR、TEAMS、DeepSnake、GAMED 等是比较实验资产。它们可以保留和评估，但不能被写成“推荐预训练”，也不能替换上述 MoonViT 主线 source。

## 12. 新接手者的一页检查清单

### 开始前

- [ ] 位于 `/home/medteam/Zhrch/DiffusionSnake-12-30`。
- [ ] locked010/011/013 没有进入任何 split。
- [ ] Train72=13,261 rows，Val37=5,914 rows，Dev8=1,123 rows。
- [ ] MoonViT cache 存在，使用 `layer_18`、1152 dim、448 input、patch14。
- [ ] 官方 config/trainer/launcher/source SHA 全部匹配本文。
- [ ] 目标输出不存在；如果已存在，只监控，不重复启动。

### Stage 1

- [ ] source 是 robust-box step40000，SHA 匹配。
- [ ] 训练 Flow 11,127,108 + 融合器 3,246,336。
- [ ] 模型总参数 14,373,444。
- [ ] batch48、LR `1e-5`、warmup1000、2×4 AB2。
- [ ] checkpoint 完整、有限、可恢复。

### Stage 2

- [ ] source 是 MoonViT step19000，SHA 匹配。
- [ ] 只有 Flow 可训练；冻结张量变化数为 0。
- [ ] 完整两阶段部署对齐 `max_abs≤1e-5`，阶段条件严格为 `s=[0.0,0.6667]`。
- [ ] RL 更新前 step0 固定面板至少满足 IoU `0.45`、Dice `0.60`；低于门槛必须停止排查。
- [ ] 已明确选择并记录训练分支：2 动作部署对齐基线，或签名锁定的 5 动作受控候选。
- [ ] 2 动作分支：fractions `[0.6667,1.0]`、sigma `[0.5,0.4]` px、credit `[0,1]`、每 rollout 8 NFE。
- [ ] 5 动作分支：fractions `[0.2,0.25,0.3333,0.5,1.0]`、sigma `[0.8,0.7,0.6,0.5,0.4]` px、credit `[0,1,2,3,4]`、每 rollout 20 NFE。
- [ ] 无论训练分支为何，所有训练中评估、Dev8 和最终部署都固定使用 2×4 AB2 / 8 NFE。
- [ ] reward 权重固定为 `[0.1,0.4,0.4,0.1]`，burr 辅助权重为 0.06。
- [ ] K8、PPO2、LR `4e-8`、目标10k有效更新。
- [ ] 训练中只看 Val37 固定 panel；不反复查看 Dev8。

### 完成后

- [ ] step10000 checkpoint 通过 SHA/有限性/冻结门。
- [ ] 关闭傅里叶探索，以原部署两阶段路径推理。
- [ ] 完整 Dev8 一次性统一评估。
- [ ] oracle-prompt 与 detector-box 分榜。
- [ ] 保存指标 JSON、运行 manifest、checkpoint SHA 和固定可视化。

---

如果实际文件、SHA、数据数量或参数量与本文任一硬门不一致，先停止并记录证据。不要通过跳过 tensor、缩小评估子集、覆盖旧目录或启动第二份任务来“继续跑”。
