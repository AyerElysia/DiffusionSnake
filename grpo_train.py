"""
GRPO强化学习训练脚本
这个暂时不能用，需要一些调整
"""

# 导入系统库
import os               # 文件系统操作
import cv2              # OpenCV图像处理
import torch            # PyTorch深度学习框架
import numpy as np      # 数值计算
import datetime         # 时间处理
from pathlib import Path# 路径操作
import json             # JSON数据处理

# 导入项目内部模块
from lib.config import cfg, args                     # 配置管理
from lib.networks import make_network                # 网络构建
from lib.train.trainers import make_trainer          # 训练器构建
from lib.train.optimizer import make_optimizer        # 优化器构建
from lib.train.recorder import make_recorder          # 记录器构建
from lib.datasets import make_data_loader             # 数据加载器构建
from lib.utils.snake import snake_config              # Snake配置
from lib.utils import data_utils                      # 数据工具


def draw_results(orig_img_bgr, det_b, pred_poly, gt_poly, save_path,
                 init_poly=None, gt4_poly=None):
    """
    绘制训练结果的可视化图像
    """
    img = orig_img_bgr.copy()

    # 将张量数据转换为CPU并分离梯度
    if det_b is not None and isinstance(det_b, torch.Tensor):
        det_b = det_b.detach().float().cpu()
    if pred_poly is not None and isinstance(pred_poly, torch.Tensor):
        pred_poly = pred_poly.detach().float().cpu().numpy()
    if gt_poly is not None and isinstance(gt_poly, torch.Tensor):
        gt_poly = gt_poly.detach().float().cpu().numpy()
    if init_poly is not None and isinstance(init_poly, torch.Tensor):
        init_poly = init_poly.detach().float().cpu().numpy()
    if gt4_poly is not None and isinstance(gt4_poly, torch.Tensor):
        gt4_poly = gt4_poly.detach().float().cpu().numpy()

    # 绘制检测结果（绿色矩形框）
    if det_b is not None and det_b.size(0) > 0:
        for i in range(det_b.shape[0]):
            x1, y1, x2, y2, score, cls_id = det_b[i, :6]
            x1 = float(x1.item()) if hasattr(x1, 'item') else float(x1)
            y1 = float(y1.item()) if hasattr(y1, 'item') else float(y1)
            x2 = float(x2.item()) if hasattr(x2, 'item') else float(x2)
            y2 = float(y2.item()) if hasattr(y2, 'item') else float(y2)
            score = float(score.item()) if hasattr(score, 'item') else float(score)
            cls_id = int(cls_id.item()) if hasattr(cls_id, 'item') else int(cls_id)
            if score <= 0:
                continue
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 1)
            cv2.putText(img, f"{cls_id}:{score:.2f}", (p1[0], max(0, p1[1]-2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    # 绘制预测轮廓（红色）
    if pred_poly is not None and len(pred_poly) > 0:
        for k in range(pred_poly.shape[0]):
            poly = pred_poly[k]
            poly = np.concatenate([poly, poly[:1]], axis=0)
            cv2.polylines(img, [poly.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

    # 绘制GT轮廓（蓝色）
    if gt_poly is not None and len(gt_poly) > 0:
        for k in range(gt_poly.shape[0]):
            poly = gt_poly[k]
            poly = np.concatenate([poly, poly[:1]], axis=0)
            cv2.polylines(img, [poly.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

    # 绘制初始轮廓（黄色）
    if init_poly is not None and len(init_poly) > 0:
        for k in range(init_poly.shape[0]):
            poly = init_poly[k]
            poly = np.concatenate([poly, poly[:1]], axis=0)
            cv2.polylines(img, [poly.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=1)

    # 绘制GT 4点极值轮廓（紫色线条和点）
    if gt4_poly is not None and len(gt4_poly) > 0:
        for k in range(gt4_poly.shape[0]):
            pts = gt4_poly[k].astype(np.int32)
            loop = np.concatenate([pts, pts[:1]], axis=0)
            cv2.polylines(img, [loop], isClosed=True, color=(255, 0, 255), thickness=1)
            for p in pts:
                cv2.circle(img, (int(p[0]), int(p[1])), 2, (255, 0, 255), -1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def infer_and_save(trainer, batch, tag: str):
    """
    执行模型推理并保存可视化结果
    """
    net = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    net.eval()
    with torch.no_grad():
        output, _, _, _ = net(batch)

    inp_bgr = batch['orig_img'][0]
    if isinstance(inp_bgr, torch.Tensor):
        inp_np = inp_bgr.detach().float().cpu().numpy()
        if inp_np.max() <= 255.0 and inp_np.ndim == 3 and inp_np.shape[-1] == 3:
            if inp_np.dtype != np.uint8:
                inp_np = np.clip(inp_np, 0, 255).astype(np.uint8)
        else:
            inp_np = inp_np - inp_np.min()
            if inp_np.max() > 0:
                inp_np = inp_np / inp_np.max()
            inp_np = (inp_np * 255.0).astype(np.uint8)
        inp_img = inp_np
    else:
        inp_img = np.array(inp_bgr)

    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    center = _to_numpy(batch['meta']['center'][0])
    scale = _to_numpy(batch['meta']['scale'][0])
    input_w, input_h = int(snake_config.voc_input_w), int(snake_config.voc_input_h)
    trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h]).astype(np.float32)

    def apply_affine_pts(pts, M):
        return data_utils.affine_transform(pts.reshape(-1, 2), M).reshape(pts.shape)

    det = output.get('detection', None)
    det_aff = None
    if det is not None and isinstance(det, torch.Tensor) and det.size(0) > 0:
        det_b = det[0].detach().float().cpu()
        det_raw = det_b.numpy().copy()
        for i in range(det_raw.shape[0]):
            x1, y1, x2, y2 = det_raw[i, :4]
            p = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            p_aff = apply_affine_pts(p, trans_input)
            det_raw[i, 0:2] = p_aff[0]
            det_raw[i, 2:4] = p_aff[1]
        det_aff = torch.from_numpy(det_raw)

    pred_py = output.get('py', None)
    pred_aff = None
    if pred_py is not None:
        last = pred_py[-1] if isinstance(pred_py, list) else pred_py
        if isinstance(last, torch.Tensor) and last.numel() > 0:
            pred_aff = last.detach().float().cpu().numpy() * float(snake_config.down_ratio)

    init_py = output.get('it_py', None)
    init_aff = None
    if isinstance(init_py, torch.Tensor) and init_py.numel() > 0:
        init_aff = init_py.detach().float().cpu().numpy() * float(snake_config.down_ratio)

    gt_aff = None
    if 'i_gt_py' in batch and 'meta' in batch and 'ct_num' in batch['meta']:
        ct_meta = batch['meta']['ct_num']
        ct_num = int(ct_meta[0].item()) if isinstance(ct_meta, torch.Tensor) else int(ct_meta)
        gt = batch['i_gt_py'][0][:ct_num]
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().float().cpu().numpy()
        gt_aff = gt * float(snake_config.down_ratio)

    gt4_aff = None
    if 'i_gt_4py' in batch and 'meta' in batch and 'ct_num' in batch['meta']:
        ct_meta = batch['meta']['ct_num']
        ct_num = int(ct_meta[0].item()) if isinstance(ct_meta, torch.Tensor) else int(ct_meta)
        gt4 = batch['i_gt_4py'][0][:ct_num]
        if isinstance(gt4, torch.Tensor):
            gt4 = gt4.detach().float().cpu().numpy()
        gt4_aff = gt4 * float(snake_config.down_ratio)

    save_dir = os.path.join(os.path.dirname(__file__), 'visual', 'grpo_train')
    os.makedirs(save_dir, exist_ok=True)
    save_path_aff = os.path.join(save_dir, f'vis_affine_{tag}.png')
    draw_results(inp_img, det_aff, pred_aff, gt_aff, save_path_aff, init_poly=init_aff, gt4_poly=gt4_aff)


def main():
    """
    GRPO强化学习训练主函数
    """
    # === 基础配置 ===
    is_main = True

    # === GRPO相关配置参数设置 ===
    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True
    cfg.use_grpo = True
    cfg.grpo_first_contour_only = False
    
    cfg.freeze_yolo = True
    
    # 环境变量
    cfg.grpo_steps = int(os.environ.get('GRPO_STEPS', '50'))
    cfg.grpo_k = int(os.environ.get('GRPO_K', '8'))
    cfg.grpo_window_size = int(os.environ.get('GRPO_WINDOW', str(cfg.grpo_window_size)))
    cfg.grpo_window_range = cfg.grpo_window_range
    cfg.reward_w_region = float(os.environ.get('GRPO_REWARD_W', '1.0')) 

    # === 构建训练组件 ===
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, trainer.network)
    recorder = make_recorder(cfg)

    # === 数据加载器 ===
    data_loader = make_data_loader(cfg, is_train=True, is_distributed=False)
    try:
        vis_loader = make_data_loader(cfg, is_train=False, is_distributed=False)
        vis_iter = iter(vis_loader)
        vis_batch = next(vis_iter)
        for k in list(vis_batch.keys()):
            if k == 'meta':
                continue
            v = vis_batch[k]
            if isinstance(v, torch.Tensor):
                vis_batch[k] = v.cuda(non_blocking=True)
    except Exception:
        vis_batch = None

    # === 加载预训练权重 ===
    ckpt_path = '/home/medteam/Zhrch/EnergeSnake1GRPO/data/one_sample/model_final.pth'
    try:
        w = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
        
        # 1. 获取模型当前定义的总层数
        model_state_dict_keys = list(w.state_dict().keys())
        total_layers = len(model_state_dict_keys)

        sd = torch.load(ckpt_path, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
            
        # 2. 加载权重 (修复: 使用strict=False以支持不完全加载和层数统计)
        missing, unexpected = w.load_state_dict(sd, strict=False) 
        
        # 3. 计算统计信息
        missing_count = len(missing)
        unexpected_count = len(unexpected)
        loaded_count = total_layers - missing_count
        load_ratio = (loaded_count / total_layers * 100) if total_layers > 0 else 0

        # 4. 打印详细报告
        print('='*50)
        print(f'[GRPO] 预训练权重加载报告')
        print(f'[GRPO] 检查点路径: {ckpt_path}')
        print(f'[GRPO] 模型定义总参数层数: {total_layers}')
        print(f'[GRPO] 成功匹配并加载层数: {loaded_count}')
        print(f'[GRPO] 缺失层数 (未加载):   {missing_count}')
        print(f'[GRPO] 多余层数 (未使用):   {unexpected_count}')
        print(f'[GRPO] >>> 权重加载成功率: {load_ratio:.2f}% <<<')
        
        if missing_count > 0:
            print(f'[GRPO] 缺失键示例 (前3个): {missing[:3]} ...')
        print('='*50)

    except Exception as e:
        if is_main:
            print(f'[GRPO] 加载预训练权重失败: {e}')
            import traceback
            traceback.print_exc()

    # === GRPO训练循环 ===
    steps = int(os.environ.get('GRPO_TRAIN_STEPS', '1000'))
    it = iter(data_loader)
    
    log_dir = Path(__file__).resolve().parent / 'data' / 'outputs' / 'grpo'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'logs.jsonl'
    
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(data_loader)
            batch = next(it)
    
        # === 数据预处理 ===
        for k in list(batch.keys()):
            if k == 'meta':
                continue
            v = batch[k]
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda(non_blocking=True)
        
        # === 模型训练步骤 ===
        trainer.network.train()

        # [修复 1]: 即使在训练模式下，强制冻结所有BN层。
        # 否则小批次GRPO采样会破坏预训练的统计分布 (running_mean/var)，导致性能恶化。
        for m in trainer.network.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d, torch.nn.SyncBatchNorm)):
                m.eval()

        print(batch.keys())
        print(batch['inp'].shape)  
        output, loss, loss_stats, _ = trainer.network(batch)
        loss = loss.mean()
        
        # === 反向传播和优化 ===
        optimizer.zero_grad()
        loss.backward()
        
        clip_target = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
        torch.nn.utils.clip_grad_value_(clip_target.parameters(), 40)
        
        # [修复 2]: 如果奖励权重为0 (GRPO_REWARD_W=0)，理论上 Loss 梯度接近 0。
        # 此时必须禁用 Weight Decay，否则权重会持续衰减（被正则化项“吃掉”），导致模型退化。
        if cfg.reward_w_region == 0:
            for group in optimizer.param_groups:
                group['weight_decay'] = 0.0

        optimizer.step()
        
        # === 可视化保存 ===
        if is_main and step % 100 == 0:
            infer_and_save(trainer, vis_batch if vis_batch is not None else batch, tag=f'step{step}')
            # 恢复训练模式（注意：循环开始处的 BN freeze 会在下一次循环再次生效，这里只需恢复 .train）
            trainer.network.train()
        
        # === 日志记录 ===
        try:
            ls = {k: float(getattr(v, 'item', lambda: v)()) for k, v in loss_stats.items()}
            
            if is_main and step % 20 == 0:
                print(f"[GRPO] 步骤 {step} "
                      f"总损失={float(loss.item()):.4f} "
                      f"GRPO损失={ls.get('grpo_loss', 0.0):.4f} "
                      f"奖励均值={ls.get('grpo_reward_mean', 0.0):.4f} "
                      f"奖励标准差={ls.get('grpo_reward_std', 0.0):.4f}")
            
            log_item = {
                'timestamp': datetime.datetime.now().isoformat(),
                'step': int(step),
                'loss': float(loss.item()),
                'grpo_loss': float(ls.get('grpo_loss', 0.0)),
                'reward_mean': float(ls.get('grpo_reward_mean', 0.0)),
                'reward_std': float(ls.get('grpo_reward_std', 0.0)),
                'grpo_steps': int(cfg.grpo_steps),
                'grpo_k': int(cfg.grpo_k),
                'grpo_window_size': int(cfg.grpo_window_size),
                'grpo_window_range': tuple(getattr(cfg, 'grpo_window_range', (2, 12))),
                'beta': float(getattr(getattr(cfg, 'train', object()), 'beta', 0.0)),
                'clip_range': float(getattr(getattr(cfg, 'train', object()), 'clip_range', 0.0)),
                'adv_clip_max': float(getattr(getattr(cfg, 'train', object()), 'adv_clip_max', 0.0)),
            }
            
            if is_main:
                with open(str(log_path), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_item, ensure_ascii=False) + '\n')
                
        except Exception:
            if is_main:
                log_item = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'step': int(step),
                    'loss': float(loss.item())
                }
                with open(str(log_path), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_item, ensure_ascii=False) + '\n')

    # === 最终可视化 ===
    if is_main:
        infer_and_save(trainer, vis_batch if 'vis_batch' in locals() and vis_batch is not None else batch, tag='final')

    # === 保存GRPO微调后的模型 ===
    try:
        out_dir = Path(__file__).resolve().parent / 'data' / 'outputs' / 'grpo_t'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = 'model_grpo.pth'
        save_path = out_dir / base_name
        if save_path.exists():
            save_path = out_dir / f'model_grpo_{ts}.pth'
        
        w = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
        if is_main:
            torch.save(w.state_dict(), str(save_path))
            print(f'[GRPO] 模型检查点已保存: {save_path}')
        
    except Exception as e:
        if is_main:
            print(f'[GRPO] 保存检查点失败: {e}')

if __name__ == '__main__':
    main()