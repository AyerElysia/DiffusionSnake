你是执行层（GPT）。在 `/home/medteam/Zhrch/DiffusionSnake-12-30` 写一个 M1 sanity/过拟合验证脚本，证明新"几何桥"流匹配范式能学习。

## 上下文
- 新范式已实现在 `lib/networks/diffusion/flow_matching_evolution.py`，由 config flag 门控。
- 目标 config：`configs/1232_final_diffusion_dit_v4_6c_geom_bridge_scratch_gpu0.yaml`（flow_geom_bridge=true, flow_resample_feat_at_xt=true, v3_7_use_contour_norm=true）。
- 训练入口参考 `diffusion_train.py`：用 `from lib.networks import make_network`、`from lib.datasets import make_data_loader`、`from lib.train.trainers import make_trainer` 构建。设置 CFG_FILE 后再 import 项目模块（关键规则）。Python 用 `/home/medteam/miniconda3/envs/snake1/bin/python`。
- 训练前置：`cfg.use_diffusion_evolution=True; cfg.use_diffusion_trainer=True`（diffusion_train.py 里就这么设的）。
- 网络 forward 返回 dict，训练时含 `diff_loss`（标量）、`pred_contours`(N,128,2)、`py`(N,128,2)。GT 轮廓在 batch 里（参考 trainer 如何取 i_gt_py；可从 prepare_training 的产物或 batch['i_gt_py']）。

## 要写的脚本
`test/m1_geom_bridge_sanity.py`，参数 `--cfg`（默认上面的 config）、`--gpu`（默认 0）、`--steps`（默认 400）、`--n_samples`（默认 8）。逻辑：
1. 设置 `os.environ['CFG_FILE']=args.cfg`，import cfg；设 `cfg.use_diffusion_evolution=True; cfg.use_diffusion_trainer=True`；把 batch_size 调成 min(n_samples, 现值)。
2. `make_network(cfg)` → `.cuda().train()`。`make_data_loader(cfg, is_train=True)`，**只取第一个 batch 并固定复用**（过拟合用），裁到 n_samples 个样本/实例。
3. 优化器 AdamW lr=1e-4（过拟合用稍大 lr），对同一 batch 反复前向反向 `--steps` 步。
4. **关键断言（in-grid）**：在第 1 步，hook 或在 forward 里拿不到的话，就在脚本里独立复算一次 feat_poly：从该 batch 的 i_init 与一个 mid-bridge x_t 反推（或直接 monkeypatch flow_matching_evolution 里 get_gcn_feature 记录传入坐标的 min/max），断言所有传入 get_gcn_feature 的坐标在 [-8, h+8]（h≈136）。最简单可靠做法：在脚本顶部 monkeypatch `snake_gcn_utils.get_gcn_feature` 包一层，记录每次调用的 img_poly.min()/max() 到全局 list，跑完打印整体 min/max 并断言。
5. 每 50 步打印 `diff_loss`。结束后：用网络 eval 模式对这 n_samples 跑一次推理（`net.eval()` 前向，取 `py`），与 GT 算 mask IoU（用 cv2.fillPoly 栅格化到合适尺寸，或复用项目里已有的 IoU/mask 工具；若有 `lib/.../iou` 工具优先复用，否则自写一个简单 polygon->mask IoU）。
6. 打印汇总：初始 loss、末步 loss、loss 下降倍数、过拟合样本平均 IoU、feat_poly 坐标 min/max。
7. 判据并以 exit code 体现：loss 下降 >5x 且 平均 IoU >0.85 → 打印 "M1 PASS" exit 0；否则 "M1 FAIL" exit 1。

## 约束
- 只新增 `test/m1_geom_bridge_sanity.py`，不改其它文件。
- 不要开长训练；脚本几百步、单 batch、几分钟内跑完。
- GPU 用 args.gpu，设 `os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)`（在 import torch 前）。
- 复用项目已有 IoU/mask 工具优先；找不到再自写最简实现。

## 产出要求
完成后只输出：新增文件名 + 脚本如何运行的一行命令 + 你本地是否实际跑通（如果环境无 GPU 跑不了就说明阻塞点）。不要贴完整代码。
