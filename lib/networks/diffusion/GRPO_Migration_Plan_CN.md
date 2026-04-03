# EnergySnake 扩散-Snake 引入 GRPO（参考 Flow-GRPO）技术方案

## 1. 背景与目标
- **现状**：当前系统以 `YOLOv8` 检测 + `Snake` 轮廓演化的扩散去噪为核心（参考 `lib/networks/diffusion/ct_snake.py`、`lib/networks/diffusion/evolution.py`）。推理使用 `DDPMScheduler` 逐步去噪，训练使用 MSE 监督噪声预测。
- **目标**：参考 Flow-GRPO 的做法，引入基于轨迹 log-prob 与分组相对优势（Group Relative Policy Optimization, GRPO）的强化学习训练，利用两类奖励：
  - 初始化奖励：检测框与 GT 框的 Dice（鼓励生成高质量初始轮廓）。
  - 区域重叠奖励：分割 mIoU 的逐步增量（鼓励轮廓演化带来区域匹配提升）。
- **约束**：
  - 尽量复用已有函数与模块；
  - 与 Flow-GRPO 的接口与训练逻辑保持一致或等价；
  - 本文仅为技术方案与落地设计，不修改现有代码。


## 2. 总体思路（与 Flow-GRPO 对齐）
- 将扩散过程视为“策略”在离散时间步上生成的轨迹（`x_t → x_{t-1}`），`SnakeDenoiser` 相当于策略网络；
- 在一次采样中，收集时间窗口内的每步 log-prob，并将最终（或窗口内增量）奖励赋予整条轨迹；
- 对同一图像/目标，重复采样 K 次（组内样本共享相同条件，只随机噪声），计算组内相对优势 A_i；
- 以 Flow-GRPO 相同形式，以 `-A_i * sum(log_prob_t)` 为策略损失，更新去噪器与其上游特征通道投影；
- 初始化 Dice 奖励与检测分支更相关；区域重叠增量奖励与扩散演化更相关（可为主要驱动）。


## 3. 复用点与模块对齐
- **特征与几何**：
  - 点特征抽取 `snake_gcn_utils.get_gcn_feature()`（`lib/utils/snake/snake_gcn_utils.py`）。
  - 初始矩形构造、上采样与对齐（`uniform_upsample`、`img_poly_to_can_poly`、`snake_decode.get_box()`）。
- **扩散与去噪**：
  - 去噪器 `SnakeDenoiser` + `DiffusionEvolution.predict_eps()` + `DDPMScheduler`（`lib/networks/diffusion/evolution.py`）。
- **检测**：
  - `ct_snake.Network` 内的 YOLOv8 集成与 NMS（`lib/networks/diffusion/ct_snake.py`）。
- **可视化与日志**：
  - 单样本脚本的 `JsonLogger` 与可视化工具（`diffusion_one_sample.py`）。


## 4. 带对数概率的 DDPM 步进（对齐 Flow-GRPO 的 SDE log-prob）
- Flow-GRPO 使用 `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py` 的 `sde_step_with_logprob()`，在每步返回：
  - `prev_sample`、`log_prob`、`prev_sample_mean`、`std_dev_t`。
- 我们使用 `diffusers.schedulers.DDPMScheduler`（`prediction_type='epsilon'`）。需设计等价的 `ddpm_step_with_logprob()`：
  - 复刻 `scheduler.step()` 的均值-方差计算（posterior q(x_{t-1}|x_t, x_0)），得到本步条件高斯 `N(mean_t, var_t)`；
  - 若提供 `prev_sample` 则以其评估 log-prob，否则按 `var_t` 采样得到 `prev_sample` 并计算 log-prob；
  - 依 Flow-GRPO，将 log-prob 在除 batch 维外取均值，得到每样本的标量。
- 数值细节：
  - 统一在 `float32` 内部计算；
  - 避免使用 `torch.pi`（兼容性），常数项用 `math.pi`（参见项目约定）。
- 窗口策略（可选，复用 Flow-GRPO 经验）：
  - 仅统计一个时间窗口内的 log-prob，排除最后一两步以避免数值过尖；
  - 配置 `grpo.window_size` 与 `grpo.window_range`。


## 5. 采样与分组（K 次重复）
- 对 batch 中的每张图像/每个检测到的目标，形成一个“组”；
- 组内进行 K 次独立采样（随机初始噪声与/或调度细节），得到 K 条轨迹；
- 每条轨迹：保存窗口内每步的 `log_prob_t`、关键时间点的轮廓（用于 mIoU 增量）、以及最终轮廓；
- 与 Flow-GRPO 一致：组内共享条件（图像、特征、初始轮廓），仅随机噪声导致采样差异。


## 6. 奖励函数设计与实现
- 统一在仿射输入坐标系或特征图坐标系上计算，保持与训练数据构造一致。

- **初始化奖励 r_init**（检测框 vs GT 框的 Dice）：
  - 定义：`Dice_init = 2|B_init ∩ B_gt| / (|B_init| + |B_gt|)`，奖励 `r_init = w0 * Dice_init`；
  - 复用：`snake_gcn_utils.get_box_match_ind()` 可用于匹配预测与 GT；`data_utils.box_iou()` 提供 IoU，可推导 Dice：`Dice = 2*IoU/(1+IoU)`；
  - 用途：更直接地激励检测器，但此奖励对扩散策略（去噪器）是常数，组内相对优势会抵消（若仅更新扩散策略）。
    - 建议：
      - 记录并用于监控；
      - 未来可在“检测分支的 RL/SL 联合训练”中用于更新 YOLO（需单独定义 YOLO 的 log-prob 或继续用 SL）。

- **区域重叠奖励 r_region**（mIoU 增量）：
  - 定义：`mIoU = (1/N) Σ_k |A_k ∩ B_k| / |A_k ∪ B_k|`；在时间步 `t` 的增量：`Δ mIoU_t = mIoU^(t) - mIoU^(t-1)`；
  - 轨迹奖励：`r_region = w1 * Σ_{t∈window} Δ mIoU_t = w1*(mIoU^end - mIoU^start)`；
  - 实现要点：
    - 由轨迹保存的多步轮廓与 GT 多边形计算 mIoU；
    - 多边形→二值掩码，可用 OpenCV `fillPoly` 或 torch/自写 rasterizer；
    - 复用：`diffusion_one_sample.py` 的坐标变换与可视化逻辑，可借其仿射映射工具（`data_utils.get_affine_transform`）。

- **总奖励**：`r = r_init + r_region`（短期可令 `w0=0` 或较小，仅记录初始化奖励，不影响扩散策略）。


## 7. GRPO 损失与训练流程
- 组内样本 `i=1..K`：
  - 计算每条轨迹的标量奖励 `r_i`；
  - 计算组平均 `\bar{r}`，优势 `A_i = r_i - \bar{r}`（可选标准化/温度缩放）。
- 轨迹 log-prob：
  - 将窗口内每步的 `log_prob_t` 求和：`L_i = Σ_t log_prob_{i,t}`；
- 策略损失（对齐 Flow-GRPO）：
  - `L_policy = - (1/K) Σ_i stop_grad(A_i) * L_i`；
  - 可加熵正则（鼓励多样性）。
- 训练组织：
  - 与现监督 MSE 可采用“多任务加权”或“阶段切换”：
    - 先以监督 MSE 稳定训练（现流程），再切至 GRPO 微调；
    - 或 `L_total = λ_rl * L_policy + λ_sup * L_mse`（建议先从纯 RL 开始）。


## 8. 接口与代码改造点（拟新增/扩展，不立即实现）
- 新增：`lib/networks/diffusion/ddpm_with_logprob.py`
  - `def ddpm_step_with_logprob(scheduler, eps_pred, t, x_t, prev_sample=None) -> (prev_sample, log_prob, mean, std)`
  - 复刻 `DDPMScheduler.step()` 的一步推导，补充 log-prob 计算。

- 扩展：`lib/networks/diffusion/evolution.py`
  - 在 `DiffusionEvolution` 内新增：
    - `sample_with_logprob(cnn_feature, i_it_py, c_it_py, py_ind, steps, window)`：返回 `[latents_t]、[log_prob_t]、[timesteps]` 与最终轮廓；
    - 采样中调用 `predict_eps()` 与 `ddpm_step_with_logprob()`；
  - 训练模式新增 RL 分支（不与现有监督冲突）：
    - 根据 cfg 切换：`use_grpo = True` 时走 RL；
    - 接入组内 K 重复采样与奖励计算，返回 `{'grpo_loss': ..., 'metrics': ...}`。

- 新增：`lib/train/rl/grpo_trainer.py`
  - 组织 K 重复分组采样、优势计算与反传；
  - 复用 `make_trainer/optimizer/recorder` 现有工厂；
  - 与 `diffusion_one_sample.py` 的 `JsonLogger` 对齐输出。

- 奖励工具函数（建议增补到 `lib/utils/snake/snake_gcn_utils.py` 或新建 `lib/utils/metrics/seg_metrics.py`）：
  - `compute_box_dice(pred_boxes, gt_boxes)`（可由 IoU 推导）；
  - `poly_to_mask(polys, H, W)`、`mask_iou(pred_mask, gt_mask)`、`mean_iou(pred_masks, gt_masks)`；
  - `incremental_miou(traj_polys, gt_polys, window)`。

- 配置新增（`cfg`）：
  - `use_grpo: bool`
  - `grpo.k: int`（每目标重复采样数）
  - `grpo.window_size: int`, `grpo.window_range: tuple`
  - `grpo.num_inference_steps: int`（RL 采样步数）
  - `grpo.loss_weights: {rl: float, sup: float}`
  - `reward.w0, reward.w1`


## 9. 训练数据与分组细节
- 训练数据准备复用 `snake_gcn_utils.prepare_training()`：
  - 已提供 `i_it_py / i_gt_py / py_ind` 等关键张量；
- 分组粒度：
  - 建议以“每个 GT 目标”作为一组；多检测框可通过匹配函数对齐到对应 GT；
  - 每组重复 K 次采样（不同随机噪声种子）。


## 10. 数值与工程注意事项
- **数值稳定**：
  - `log_prob` 计算请在 `float32`；
  - 避免使用 `torch.pi`，使用 `math.pi`；
  - 窗口排除最后一步，或限制方差下限，避免 log-prob 爆炸。
- **显存与效率**：
  - 组内 K 次采样可顺序执行，跨组可并行（按 batch 维）；
  - 采样步数先从 10~20 步起，窗口 4~8 步；
  - 可启用 AMP，谨慎在 log-prob 内部混合精度。


## 11. 实验与超参建议（初始）
- `k=4`、`num_inference_steps=20`、`window_size=6`、`window_range=(2,12)`；
- `w0=0.0~0.1`、`w1=1.0`（先聚焦演化质量）；
- 学习率与优化器：沿用现有 `AdamW`；
- 先单独 RL 微调扩散分支（冻结 YOLO），观察 mIoU 提升与稳定性；
- 记录：每组奖励分布、零标准差比例（参考 Flow-GRPO 的 `calculate_zero_std_ratio` 思路）。


## 12. 渐进式落地里程碑（不改现代码的实现计划）
- **M1：日志概率**
  - 新增 `ddpm_step_with_logprob()`，在单样本脚本中验证能返回稳定的 `[log_prob_t]` 与 `[latents_t]`。
- **M2：窗口与K重复采样**
  - 在 `DiffusionEvolution` 增加 `sample_with_logprob()`，支持窗口采样；
  - 单图单目标 K 轨迹采样与 JSON 记录。
- **M3：奖励实现**
  - `box_dice` 与 `poly_to_mask + mIoU` 工具实现并在单样本上验证；
  - 验证 `Σ ΔmIoU = mIoU_end - mIoU_start` 一致性。
- **M4：GRPO 训练环**
  - 新增 `grpo_trainer`，实现组内优势与策略损失；
  - 在小数据集上跑通若干步，监控奖励与 mIoU 改善。
- **M5：与现训练集成**
  - 加入配置开关，支持 RL-only / RL+Sup 两模式切换；
  - 加日志与可视化对齐 `diffusion_one_sample.py`。


## 13. 与 Flow-GRPO 的关键一致性清单
- **每步 log-prob**：通过自定义 `ddpm_step_with_logprob()` 实现；
- **时间窗口**：提供 window 策略，减少末步数值不稳的影响；
- **K 重复 + 组内优势**：分组采样、相对优势（`A_i = r_i - \bar{r}`）；
- **策略更新**：`-A_i * Σ log_prob_t`；
- **统计与可视化**：对齐其日志结构，便于对照实验。


## 14. 可能的扩展
- 将初始化奖励真正用于 YOLO 分支的更新（需要定义 YOLO 的“动作-概率”或继续使用 SL 混合）；
- 从 DDPM 切换到 SDE/Flow-Matching 调度，完全复用 Flow-GRPO 的 `sde_step_with_logprob`；
- 引入边界平滑、熵奖励等正则项；
- 对应 CT 3D/序列扩展的轨迹奖励与采样策略。


## 15. 参考文件（现有工程内）
- `lib/networks/diffusion/ct_snake.py`：检测-扩散主流程与特征拼接。
- `lib/networks/diffusion/evolution.py`：`DiffusionEvolution` 与 `SnakeDenoiser`。
- `lib/utils/snake/snake_gcn_utils.py`：点特征、上采样、匹配与几何工具。
- `diffusion_one_sample.py`：单样本可视化与 JSON 记录参考。
- Flow-GRPO：
  - `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`
  - `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob*.py`
  - `scripts/train_sd3.py`

---

本方案不修改任何现有代码，仅给出模块化改造与训练流程的详细设计，后续可按“里程碑”循序实现与集成。
