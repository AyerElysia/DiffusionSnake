你是执行层（GPT）。修复 `/home/medteam/Zhrch/DiffusionSnake-12-30/test/m1_geom_bridge_sanity.py` 的一个评测-harness 缺陷。

## 已诊断的问题（不要质疑这个诊断，直接修）
脚本训练侧正常（diff_loss 下降 ~100x），但 eval IoU 只有 ~0.5。根因：脚本 eval 用 `net.eval(); net(batch["inp"], batch)` 走推理分支，推理分支里 `prepare_testing(output)` 用的是 **detection head 输出** 来构建初始轮廓。但本 M1 是 `resume:false` 从零、且只过拟合训练了 FM denoiser，**detection head 是随机初始化的垃圾** → eval 初始轮廓是错的 → IoU 永远 ~0.5。这不是范式的问题，是评测路径用错了 init。

## 要做的修复
让 eval 从【与训练相同的 GT-octagon 初始轮廓】出发跑桥，绕开未训练的 detection。具体：

1. 训练时网络内部用 `snake_gcn_utils.prepare_training(output, batch)` 得到 `i_it_py`(octagon init)、`i_gt_py`(GT)。在 eval 阶段，复用**同一个 batch**，调用 `prepare_training({}, batch)`（脚本里已有 `extract_compacted_gt` 这么取 GT）拿到 `i_it_py`(octagon init)、`c_it_py`、`py_ind`、`i_gt_py`。注意：训练分支里 `train_dict = snake_gcn_utils.prepare_training(output, batch)` 的 `i_it_py` 就是 octagon init；用 `prepare_training({}, batch)` 同样能拿到（output 传空 dict 即可，因为它不依赖 detection，只依赖 batch 里的 GT/extreme）。请核对 prepare_training 返回的 key，取 `i_it_py`/`c_it_py`/`py_ind`(或 'ind')/`i_gt_py`。
2. 直接调用 evolution 模块的桥推理，而不是走 `net(...)` 的推理分支。evolution 模块的拿法：网络对象里有 FlowMatchingEvolution 实例（在 ct_snake/snake 网络里，名字可能是 `net.evolution` 或类似；用 `for m in net.modules(): if isinstance(m, FlowMatchingEvolution)` 找到它，`from lib.networks.diffusion.flow_matching_evolution import FlowMatchingEvolution`）。还需要 cnn_feature：训练前向时网络把 backbone 特征算出来喂给 evolution。最稳妥做法：复用网络已有的前向到 cnn_feature 的路径——
   - 方案A（推荐，最省事）：临时设 `cfg.use_pred_extreme_init_for_inference=False` 不行（仍走 detection）。改为：在 eval 时把 batch 里的 GT-octagon init 通过 `output` 注入。看 `flow_matching_evolution.py` 推理分支开头（约 line 2384-2405）：若 `use_pred_extreme_init_for_inference` 且 `output['ex']` 存在，则用 `output['ex']` 建 octagon init。所以可以：`net.eval()`，先跑 backbone+detection 得到 output，然后**覆盖** `output['ex'] = <GT extreme points>` 并设 `cfg.use_pred_extreme_init_for_inference=True`，再调 evolution。但 GT extreme 不一定在 batch 里现成。
   - 方案B（更直接，优先用这个）：手动复刻网络从 inp 到 cnn_feature 的前向（参考 snake 网络 forward：backbone→fpn/p3 fusion→cnn_feature），拿到 cnn_feature 后**直接调** `evolution.sample_disp_iterative(cnn_feature, i_it_py, c_it_py, py_ind, num_iter_steps=..., fractions=..., ode_steps=..., batch=batch)` 或在 bridge+per_ode 开时它内部会走 `_sample_disp_geom_bridge`。得到 disp 后 `py = i_it_py + disp`。
   - 如果方案B 里"从 inp 到 cnn_feature"的复刻太繁琐，用最简单的 hook：在训练前向时用 forward hook 抓住传给 evolution.forward 的 cnn_feature（evolution.forward 签名是 `forward(self, output, cnn_feature, batch)`），缓存最后一次的 cnn_feature；eval 时直接用缓存的 cnn_feature（因为 batch 固定不变，cnn_feature 一致）+ GT-octagon init 调 `evolution.sample_disp_iterative(...)`。**这个 hook 方案最稳，优先用。**
3. 用 `i_it_py + disp` 作为 pred 轮廓，与 `i_gt_py` 算 polygon IoU（脚本已有 polygon_iou）。注意坐标系：训练/推理 i_it_py 都在下采样后的输出坐标系（×down_ratio 还原到图像系——脚本现在 eval 就是 ×down_ratio，保持一致）。
4. 同时打印一个 baseline：octagon-init vs GT 的 IoU（即 disp=0 时），这样能看出桥到底移动了多少、是否真的逼近 GT。
5. 判据不变：loss 下降>5x 且 桥后 mean IoU>0.85 → M1 PASS exit 0。另外打印 baseline IoU 供对比。

## 约束
- 只改 `test/m1_geom_bridge_sanity.py`。
- 不改 flow_matching_evolution.py 或其它项目文件。
- 保留已有的 in-grid 断言、loss 打印。

## 产出要求
完成后只输出：改了哪几处 + 一行运行命令。不要贴完整代码。说明你本地是否能跑（无 GPU 跑不了就只做 py_compile）。
