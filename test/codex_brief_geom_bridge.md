你是执行层（GPT），在一个 PyTorch 医学轮廓分割项目里实现一个新的流匹配（Flow Matching）预训练范式。严格只改我指定的文件，全部用 config flag 门控，绝不破坏现有路径。工作目录 `/home/medteam/Zhrch/DiffusionSnake-12-30`。

## 目标
在 `lib/networks/diffusion/flow_matching_evolution.py`（类 `FlowMatchingEvolution`）里新增一个"几何桥"(geometric bridge)范式：把 FM 从"纯噪声→位移"改成"初始轮廓→GT 轮廓"的直线桥，并让特征在中间插值轮廓 x_t 上重采样。一切由新 flag 门控，flag 全部默认 False，关闭时数值与现状逐位一致。

## 背景事实（已核实，不要改这些理解）
- 训练 forward 在 `forward()` 的 `if self.training:` 分支（约 line 1835-2268）。关键点：
  - line ~1945: `x1_raw = i_gt_py - i_init_train_py`（clean target 是 GT 位移）
  - line ~1948-2105: `if self.use_iterative_refinement:` 块做 rich-state 状态增强（把 init 推进 frac）
  - line ~2107: `if self._small_disp_prob > 0 and not used_mixed_iter_interp:` 小位移增强
  - line ~2124-2132: `contour_scale=...; x1=self.normalize_target_disp(x1_raw, contour_scale); N=x1.size(0); t=self.sample_train_t(...); x0=self.sample_train_x0(x1); x_t=(1.0-t)*x0+t*x1`
  - line ~2135-2144: `sampled_feat_curr = snake_gcn_utils.get_gcn_feature(cnn_feature, i_init_train_py, py_ind, h, w)`，`detail_feat_curr = self.sample_detail_features(cnn_feature, i_init_train_py, py_ind, h, w, sampled_feat=sampled_feat_curr, contour_scale=contour_scale)`
  - line ~2147-2152: `build_locate_token_context(batch, i_init_train_py, py_ind, contour_scale=contour_scale)`
  - line ~2195: `v_target = x1 - x0`
- `get_gcn_feature(cnn_feature, img_poly, ind, h, w)` 在传入 img_poly 的【绝对像素坐标】上采样。
- `self.denormalize_pred_disp(x_t, contour_scale)` 把归一化位移反归一到【像素位移】；当 `v3_7_use_contour_norm=True` 时是线性过原点（即 0→0），这是新范式正确性的关键。
- 推理路径：`_sample_disp_from_sampled_feat`（约 line 1578-1671，内部 `x_t=torch.randn_like(i_it_py)*noise_scale` 起步，Euler 积分，特征用外部传入的固定 sampled_feat/detail_feat）；`sample_disp`（约 1673-1727）；`sample_disp_iterative`（约 1729-1811，outer loop 每步在 current_contour 重采特征后调 `_sample_disp_from_sampled_feat`）。

## 要做的改动

### A. 在 `__init__` 里加 flag（放在约 line 558 `self._max_disp_frac` 附近，用 getattr，默认 False/0.0）
```
self._geom_bridge = bool(getattr(global_cfg, 'flow_geom_bridge', False))
self._resample_feat_at_xt = bool(getattr(global_cfg, 'flow_resample_feat_at_xt', False))
self._geom_infer_resample_per_ode = bool(getattr(global_cfg, 'flow_geom_infer_resample_per_ode_step', False))
self._geom_x0_jitter = float(getattr(global_cfg, 'flow_geom_x0_jitter', 0.0))
```

### B. 训练 forward：x0 = init 轮廓（改 x0/x_t 构造，约 line 2129-2132）
把 `x0 = self.sample_train_x0(x1)` 改成门控：
- 非 bridge：保持原样 `x0 = self.sample_train_x0(x1)`
- bridge：`x0 = torch.zeros_like(x1)`；若 `self._geom_x0_jitter > 0` 再 `x0 = x0 + torch.randn_like(x1) * self._geom_x0_jitter`
`x_t = (1.0 - t) * x0 + t * x1` 不变。`v_target = x1 - x0`（line 2195）不要动。

### C. 训练 forward：特征在 x_t 上重采（在 line ~2135 三处采样【之前】插入 feat_poly 计算）
```
if self._geom_bridge and self._resample_feat_at_xt:
    xt_disp_raw = self.denormalize_pred_disp(x_t, contour_scale)
    feat_poly = (i_init_train_py + xt_disp_raw).detach()
else:
    feat_poly = i_init_train_py
```
然后把这三处的 `i_init_train_py` 改为 `feat_poly`：
- `get_gcn_feature(cnn_feature, feat_poly, py_ind, h, w)`（line 2135）
- `self.sample_detail_features(cnn_feature, feat_poly, py_ind, h, w, sampled_feat=sampled_feat_curr, contour_scale=contour_scale)`（line 2136-2144）
- `self.build_locate_token_context(batch, feat_poly, py_ind, contour_scale=contour_scale)`（line 2147-2152）
注意：`predict_velocity(...)` 调用里的 `polys=i_init_train_py` 与 `c_init_train_py` 【保持不变】（polys 驱动 adjacency，不是特征采样）。

### D. 训练 forward：bridge 下禁用 init 推进（rich-state guard）
- line ~1948：`if self.use_iterative_refinement:` 改为 `if self.use_iterative_refinement and not self._geom_bridge:`
- line ~2107：`if self._small_disp_prob > 0 and not used_mixed_iter_interp:` 改为 `if self._small_disp_prob > 0 and not used_mixed_iter_interp and not self._geom_bridge:`

### E. 新增推理 helper `_sample_disp_geom_bridge`（放在 `_sample_disp_from_sampled_feat` 附近）
签名 `(self, cnn_feature, i_it_py, c_it_py, py_ind, steps=None, noise_scale=None, locate_context=None)`，返回像素位移张量 disp（形状同 i_it_py）。逻辑：
```
if steps is None: steps = self.ode_steps
device = i_it_py.device; N = i_it_py.size(0); h, w = cnn_feature.size(2), cnn_feature.size(3)
contour_scale = self.compute_contour_scale(i_it_py); contour_scale_flat = contour_scale.view(-1)
x_t = torch.zeros_like(i_it_py)
if self._geom_x0_jitter > 0: x_t = x_t + torch.randn_like(i_it_py) * self._geom_x0_jitter
dt = 1.0 / steps
for i in range(steps):
    t_val = i * dt
    cur = i_it_py + self.denormalize_pred_disp(x_t, contour_scale)
    sampled_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, cur, py_ind, h, w)
    detail_feat = self.sample_detail_features(cnn_feature, cur, py_ind, h, w, sampled_feat=sampled_feat, contour_scale=contour_scale)
    t_tensor = torch.full((N,), t_val, device=device, dtype=torch.float32)
    v_pred, _ = self.predict_velocity(cnn_feature, i_it_py, c_it_py, sampled_feat, detail_feat, py_ind, x_t, t_tensor, contour_scale=contour_scale_flat, x_self_cond=None, locate_context=locate_context)
    x_t = x_t + v_pred * dt
disp = self.denormalize_pred_disp(x_t, contour_scale)
return self.clamp_pred_disp(disp, i_it_py)
```
（self-conditioning / disp-gate / latent-policy / avg-samples 在本 helper 里一律跳过，加一行注释说明 v4_6c 不用这些。）

### F. 接入推理 helper（门控 `self._geom_bridge and self._geom_infer_resample_per_ode`）
- 在 `sample_disp`（约 line 1673）：在做特征采样之前，若 bridge+per_ode 开，构造 locate_context（沿用现有逻辑：若 `self._locate_token_enabled and batch is not None` 才构造，否则 None），然后 `return self._sample_disp_geom_bridge(cnn_feature, i_it_py, c_it_py, py_ind, steps=steps, locate_context=locate_context)`。
- 在 `sample_disp_iterative` 的 outer loop 内（约 line 1780-1802），当 bridge+per_ode 开时，把对 `_sample_disp_from_sampled_feat` 的调用替换为对 `_sample_disp_geom_bridge(cnn_feature, current_contour, c_it_py, py_ind, steps=ode_steps, noise_scale=step_ns, locate_context=locate_context)` 的调用（current_contour 作为该 outer step 的起点 i_it_py）。avg_n>1 的分支同理用 helper。

## 约束
- 所有改动【仅限】`lib/networks/diffusion/flow_matching_evolution.py` 这一个文件。
- 不改 reward/PPO/denoiser 网络/snake_gcn_utils。
- 关闭所有新 flag 时，代码路径与行为必须与改动前【完全一致】（这是回归保护的硬要求）。
- helper 的可选参数照现有风格。

## 验收标准
1. `python -m py_compile lib/networks/diffusion/flow_matching_evolution.py` 通过。
2. 新 flag 默认 False；bridge 关闭时三处采样仍等价指向 `i_init_train_py`（通过 feat_poly 的 else 分支）。
3. 新 helper `_sample_disp_geom_bridge` 存在且语法正确。

## 产出要求
完成后只输出：变更文件列表 + 每个改动点一句话说明（不要贴完整代码）。
