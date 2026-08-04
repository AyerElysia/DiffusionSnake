# DiffusionSnake 轮廓演化 / Flow 主线接管记录（2026-08-04）

## 职责更正

本任务 ID 为 `019fb3e3-3c35-72d2-ab0d-418a302def49`。

本任务的正式职责是 **轮廓演化 / Flow 主线**，不是推理加速 / 整卷并行。此前以
“推理加速 / 整卷并行”身份发出的职责确认作废，以本记录为准。

真正的推理加速 / 整卷并行任务 ID 为
`019fc203-8a9e-76f1-a244-bab23d8f9bd5`。本记录不修改、不删除或重写该任务的实验结果、
报告与代码。

## 项目协作名单

1. 轮廓演化 / Flow 主线：`019fb3e3-3c35-72d2-ab0d-418a302def49`
2. 推理加速 / 整卷并行：`019fc203-8a9e-76f1-a244-bab23d8f9bd5`
3. 检测器 / 初始化与覆盖：`019fb3d5-abc9-7662-8731-8b8cb0c44755`
4. 强化学习 / Contour RL：`019fc1f4-6777-7752-9b19-244f5651882a`
5. 项目统筹、证据审计及中英文论文写作：`019fc08c-78a7-78f2-a56a-aba921e423ef`

项目负责人是全链路负责人和最终决策者。

凡是会影响论文主张、贡献层级、实验数字、评估口径、失败结论或文字表述的事项，直接
向论文统筹任务 `019fc08c-78a7-78f2-a56a-aba921e423ef` 同步。

## 统一论文层级

- **第一贡献：Flow Matching 轮廓演化。**
- **第二贡献：Contour RL。**
- 3D 是顺序体数据能力扩展，不与 native voxel 3D 竞争。
- 检测器与推理加速属于系统性能支撑，不作为核心创新。

## 本任务职责

### 1. Flow Matching 轮廓演化第一贡献

- 维护主线 Flow Matching 轮廓演化方法、训练目标、采样路径和推理调度。
- 明确扩散/流演化相对传统轮廓回归的作用，避免把检测器、Memory 或 RL 收益错误计入
  Flow 主线。

### 2. Matched direct-offset / DeepSnake 公平对照

- 对 direct-offset、DeepSnake 与 Flow 使用匹配的数据、初始化、检测框、训练预算、
  checkpoint 选择和评估协议。
- 记录 Volume Dice、foreground Dice、轮廓质量、失败病例和推理成本，保证第一贡献有
  可复查的同条件证据。

### 3. Inner NFE 与 outer stage 归因

- 分开审计单个 Flow stage 内的数值积分步数（inner NFE）和多阶段迭代精修
  （outer stage）。
- 报告真实 denoiser 调用次数、有效 stage 数、端到端延迟和质量变化，不能把增加 outer
  stage 伪装成单次采样 NFE 的收益。
- 对 AB2、Euler、Heun 或其他调度的比较保持相同总调用预算，保证质量—速度归因成立。

### 4. Dense-6 + H1 主线冻结

- 当前主线结构固定为 Dense-6 DiT + H1 Dense Residual 输出头。
- 新的检测器、RL、Memory 或加速模块应在该冻结主线上单独验证；除非项目负责人批准，
  不同时改 Flow 主干和外围模块。
- H1 是输出头蒸馏结果，质量差异约为 1e-4 量级；论文中应表述为质量保持和结构简化，
  不能声称统计显著提升。

### 5. 轮廓质量与失败分析

- 不只报告平均 Dice，还要检查断裂、粘连、漏分实例、边界偏移、拓扑异常、困难椎体和
  最差病例。
- GT box 用于隔离轮廓演化能力；predicted box 用于部署链路评估。两者不得混用归因。
- 可视化必须覆盖代表性成功、一般病例和系统性失败，不以少量最好样本替代全量证据。

## 当前跨任务同步重点

### 与检测器任务同步

- 固定 GT-box 与 predicted-box 的定义、检测缓存、阈值和覆盖版本。
- 检测器提供漏检、错类、框偏移和覆盖率审计；Flow 主线据此区分初始化错误与轮廓演化
  错误。
- 任何部署质量下降先做 detector/evolution isolation，不能直接归因于 Flow。

### 与 Contour RL 任务同步

- RL 作为第二贡献，应建立在冻结的 Dense-6 + H1 Flow 主线上。
- 同步 Flow checkpoint、状态表示、噪声/时间条件、推理调度和奖励评估接口。
- RL 收益必须相对同一 Flow 基线单独归因；不同时改变 Flow 训练、检测器或 3D Memory。

### 与推理加速 / 整卷并行任务同步

- 共同冻结 H1 checkpoint、8-NFE AB2 调度、输入输出接口和病例/seed。
- 加速任务负责真实 pass、DiT calls、吞吐、显存和顺序体扩展；Flow 主线负责采样质量与
  inner/outer 调用预算定义。
- 旧 feature-context、Jacobi 和新 Physical Volume Memory 的结论归加速任务维护；未经
  其机器结果和报告确认，不进入 Flow 第一贡献的论文主张。

### 与论文统筹同步

- 第一贡献的每个数字必须同时给出机器可读结果、落盘报告、配置和 checkpoint。
- 重点同步 matched direct-offset / DeepSnake、inner NFE / outer stage、Dense-6 + H1
  冻结范围，以及代表性失败可视化。
- 明确证据边界：小规模验证、蒸馏结果、GT-box 隔离实验和部署 predicted-box 结果分别
  表述，不做跨协议比较。

## 本次更正的操作边界

- 没有修改共同论文文件。
- 没有启动训练、评估或 GPU smoke。
- 没有删除、覆盖或重写加速任务及其他任务的实验结果。
- 仅新增本 Flow 演化模块自己的接管记录，并向相关协作任务发送职责更正。
