你是执行层（GPT）。写一个【单样本逐步诊断】脚本，定位为什么几何桥 fixed-feature 训练 loss 极低(1e-5)但桥推理后 IoU 反而比 init 还差。工作目录 `/home/medteam/Zhrch/DiffusionSnake-12-30`，Python `/home/medteam/miniconda3/envs/snake1/bin/python`。

## 背景
几何桥范式(`flow_geom_bridge=true`)：x0=init轮廓(零位移)，x1=GT位移(contour-norm)，x_t=(1-t)x0+t*x1，v_target=x1。fixed-feature 版(`flow_resample_feat_at_xt=false`)是常速度场。M1 过拟合(32样本1000步)训练 loss 降到 1e-5，但 eval 桥后 IoU(0.739) < init baseline(0.772)。这自相矛盾——velocity 匹配到 1e-5 却 rollout 变差。要找出训练目标和 rollout/eval 之间的系统性不一致。

## 已知 eval 路径
M1 eval：用 `snake_gcn_utils.prepare_training({}, batch)` 取 i_it_py(octagon init)/i_gt_py(GT)；hook 抓训练前向时喂给 evolution 的 cnn_feature；然后调 `evolution.sample_disp_iterative(cnn_feature, i_it_py, c_it_py, py_ind, num_iter_steps, fractions, ode_steps, batch)` 得 disp，pred=i_it_py+disp，与 i_gt_py 算 polygon IoU(×down_ratio)。config 有 `use_iterative_refinement:true, iterative_fractions:[0.5,0.85,1.0], iterative_ode_steps:4, flow_ode_steps:4`。

## 要写的诊断脚本 `test/bridge_single_diag.py`
复用 M1 脚本里已有的 batch 构建/cnn_feature hook/IoU 逻辑（可以直接 import 或复制其中的函数）。参数 `--cfg`、`--gpu`、`--steps`(默认600)、`--n_samples`(默认16)。流程：
1. 同 M1：构 fixed batch，过拟合训练网络 --steps 步（fixed-feature bridge config）。
2. 训练后，对【第 0 个轮廓】做 5 种推理并各自算 IoU 和"位移量级"，全部打印对比：
   a. **init baseline**：disp=0，pred=i_it_py，IoU。
   b. **单桥 full（无 iterative）**：直接调 `evolution.sample_disp(cnn_feature, i_it_py, c_it_py, py_ind, steps=10, batch=batch)` 得 disp（fixed-feature 下走 `_sample_disp_from_sampled_feat`，不是 geom helper——因为 `flow_geom_infer_resample_per_ode_step` 控制是否走 helper；这里 fixed-feature 仍想要"从 x_t=0 起步"，所以**请确认** sample_disp 在 bridge 下起点是 0 还是 randn：看 `_sample_disp_from_sampled_feat` 里 `x_t=randn*noise_scale`——bridge 时 noise_scale=0 所以 x_t=0，OK)。打印 IoU + disp.abs().mean()。
   c. **iterative 桥**（M1 eval 用的路径）：`sample_disp_iterative(...)`，打印 IoU + disp 量级。
   d. **直接用训练 target 反推的"理想 pred"**：从 `prepare_training` 拿对齐后的 i_gt_py，pred_ideal = i_gt_py（即完美桥应到达的位置），IoU（应该很高，验证 IoU 计算和对齐本身没问题——这是关键对照，排除 IoU/对齐 bug）。
   e. **训练分支自测**：把网络设 train()，手动构造 x_t=（t=1 处即 x1，或 t=0.99），跑一次 forward 看 `pred_contours`/`pred_disp` 与 i_gt_py 的 IoU——验证"模型在训练用的 (x_t,t,固定特征) 输入下，预测的终点 x1_pred 是否真的逼近 GT"。这一步**最关键**：如果训练自测 IoU 高但 rollout(b/c) IoU 低，说明问题在 ODE 积分/起点/坐标；如果训练自测 IoU 也低，说明 v_target/normalize 有问题或 loss 低但方向错。
3. 额外打印：对第 0 轮廓，i_it_py 的坐标范围、i_gt_py 范围、单桥 disp 范围、contour_scale，以及 `evolution.denormalize_pred_disp(torch.zeros..., contour_scale)` 是否=0（确认 contour-norm 过原点）。
4. 关键诊断输出：把 (a) init IoU、(b) 单桥 IoU、(c) iterative IoU、(d) 理想 pred IoU、(e) 训练自测 IoU 并排打印。**这能定位问题在哪一环**。

## 约束
- 只新增 `test/bridge_single_diag.py`，不改其它文件。
- 复用 M1 脚本的工具函数（polygon_iou、batch trim、extract_compacted_init/gt、cnn_feature hook）。
- 几分钟内跑完。

## 产出要求
只输出：新增文件名 + 运行命令 + 是否 py_compile 通过（无 GPU 不强求实跑）。不贴完整代码。
