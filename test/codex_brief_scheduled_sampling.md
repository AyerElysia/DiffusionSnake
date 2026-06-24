你是执行层（GPT）。在 `lib/networks/diffusion/flow_matching_evolution.py` 给"几何桥"(geometric bridge)流匹配训练加一个 scheduled-sampling / 自rollout 训练选项，修复 resample-at-x_t 的暴露偏置。严格只改这一个文件，flag 门控，默认关闭时逐位等价现状。工作目录 `/home/medteam/Zhrch/DiffusionSnake-12-30`。

## 背景（已诊断，不要质疑）
几何桥范式：x0=init轮廓(零位移)，x1=GT位移(normalized)，x_t=(1-t)*x0+t*x1，特征在插值轮廓 x_t 上重采(flow_resample_feat_at_xt)。问题：训练时 x_t=t*x1_真，特征永远沿【真实路径】采；推理时 x_t 是【累积预测】位移，轮廓漂移→特征采错→误差累积，IoU 反而变差(0.80→0.69)。修法 = scheduled sampling：训练时让 x_t 落在【模型自己 rollout 出来的路径】上，再对【剩余真实位移】做速度匹配，让模型学会从 off-path 状态纠偏。

## 现状关键位置（训练 forward，`if self.training:` 分支）
- 约 line 2230-2236：x0 构造（bridge 时 x0=zeros，可选 +x0_jitter）。
- 约 line 2238-2240：`t = self.sample_train_t(N,...)`；`x_t = (1.0-t)*x0 + t*x1`。
- 约 line 2242-2248：`feat_poly` 计算（bridge+resample 时 `feat_poly=(i_init_train_py+denorm(x_t)).detach()`），随后三处采样 sampled_feat_curr/detail_feat_curr/locate。
- 约 line 2290-2300：`v_pred,L_reg = self.predict_velocity(..., x_t, t.view(-1), ...)`；`v_target = x1 - x0`（约 line 2305）；loss = MSE(v_pred, v_target)（约 line 2334）。
- 已有 helper：`self.denormalize_pred_disp(x_t, contour_scale)`（contour-norm 下线性过原点）；`self.predict_velocity(cnn_feature, i_init_train_py, c_init_train_py, sampled_feat, detail_feat, py_ind, x_t, t_tensor, contour_scale=contour_scale_flat, x_self_cond=None, locate_context=...)`；`snake_gcn_utils.get_gcn_feature(cnn_feature, poly, py_ind, h, w)`；`self.sample_detail_features(cnn_feature, poly, py_ind, h, w, sampled_feat=..., contour_scale=...)`。

## 要加的功能：bridge scheduled-sampling
新 flag（__init__ 里 getattr 默认）：
- `flow_geom_sched_sampling` → `self._geom_sched_sampling`（bool, 默认 False）
- `flow_geom_sched_inner_steps` → `self._geom_sched_inner_steps`（int, 默认 2）
- `flow_geom_sched_prob` → `self._geom_sched_prob`（float, 默认 1.0；这批样本里以此概率走 sched-sampling，否则走原直线-真路径，便于混合）

仅当 `self._geom_bridge and self._geom_sched_sampling and self._resample_feat_at_xt` 时启用（否则完全走现有逻辑，不变）。

### 训练逻辑（替换 bridge 下 x_t 构造 + 特征采样 + 预测那一段，约 line 2238-2300）
启用时，对启用 sched 的样本子集（用 `self._geom_sched_prob` 抽 mask，简单起见可整 batch 启用即 prob=1）：
1. 随机选一个落点步 `k`：在 0..(K) 之间均匀采（K=`self._geom_sched_inner_steps`），表示"模型自己 rollout 走 k 步后到达的状态"。dt_inner = 1.0/(K+1) 或用一个固定 inner 步长（例如把 t∈[0,1] 切成 K+1 段）。设 `t_land = k * dt_inner`。
2. 用 **no_grad** 从 x=0 自 rollout 到第 k 步：
   ```
   with torch.no_grad():
       x_roll = torch.zeros_like(x1)
       for j in range(k):
           t_j = j * dt_inner
           cur_j = (i_init_train_py + self.denormalize_pred_disp(x_roll, contour_scale)).detach()
           sf = snake_gcn_utils.get_gcn_feature(cnn_feature, cur_j, py_ind, h, w)
           df = self.sample_detail_features(cnn_feature, cur_j, py_ind, h, w, sampled_feat=sf, contour_scale=contour_scale)
           t_tensor_j = torch.full((N,), t_j, device=device, dtype=x1.dtype)
           v_j, _ = self.predict_velocity(cnn_feature, i_init_train_py, c_init_train_py, sf, df, py_ind, x_roll, t_tensor_j, contour_scale=contour_scale_flat, x_self_cond=None, locate_context=locate_context_curr)
           x_roll = x_roll + v_j * dt_inner
   ```
   （注意 k 可能逐样本不同。最简单实现：整 batch 用同一个随机 k，每个 minibatch step 重抽 k，省去逐样本 gather。这样实现简单且足够。）
3. 现在 `x_t = x_roll.detach()`（模型自己到达的 off-path 状态），`t = t_land`（标量→broadcast 成 (N,1,1)）。**带梯度的那一步预测**：在 `x_t` 处重采特征（feat_poly = i_init_train_py + denorm(x_t)，detach），预测 `v_pred`。
4. **target**：在 scheduled-sampling 下，目标速度应指向真实终点 x1。直线桥的瞬时速度恒为 (x1 - x0)=x1（x0=0），与当前在哪无关。所以 `v_target = x1`（即 x1 - x0），**保持和现状一致**。loss = MSE(v_pred, v_target)。这样模型学到的是"无论我漂到哪，都把剩余速度指回 x1 方向"——正是纠偏。
   （重要:不要把 target 改成 (x1 - x_t)/(1-t) 之类——直线桥速度场是常量 x1，这才是 rectified flow 的正确目标。）
5. 其余 loss 分支（curvature reweight/spectral/endpoint/chamfer/gate/L_reg）保持，对 v_pred/v_target 照常计算。

不启用 sched 时（默认或 prob 抽到否），完全走现有 line 2238-2300 逻辑，不得有任何数值变化。

## 约束
- 仅改 `lib/networks/diffusion/flow_matching_evolution.py`。
- 关闭 flag 时逐位等价现状（回归保护硬要求）。
- no_grad 包住自 rollout 的 K 步，只有最后一步预测带梯度（省显存、稳定）。
- 整 batch 同一个随机 k 即可，不必逐样本。

## 验收标准
1. `python -m py_compile lib/networks/diffusion/flow_matching_evolution.py` 通过。
2. 新增三个 flag，默认 False。
3. flag 关闭时训练路径不变。

## 产出要求
完成后只输出：变更点一句话说明（不贴完整代码）+ 是否 py_compile 通过。
