"""
EnergySnake GRPO强化学习训练脚本

该脚本实现了基于Group Relative Policy Optimization (GRPO)的强化学习训练流程，
用于对预训练的扩散分割模型进行精细调优。通过奖励信号指导模型优化，
提升脊柱结构分割的精度和质量。

主要功能：
- GRPO强化学习训练循环
- 奖励计算和优势值估计
- 模型可视化结果生成
- 训练过程日志记录
- 模型检查点保存

使用方法：
GRPO_STEPS=20 GRPO_K=4 GRPO_WINDOW=6 python grpo_train.py
"""

# 导入系统库
import os               # 文件系统操作
import cv2              # OpenCV图像处理
import torch            # PyTorch深度学习框架
import numpy as np      # 数值计算
import datetime         # 时间处理
from pathlib import Path # 路径操作
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
    
    该函数将检测结果、预测轮廓、GT轮廓等多种信息绘制在一张图像上，
    用于直观地评估模型训练效果和分割质量。
    
    Args:
        orig_img_bgr: 原始BGR格式的背景图像
        det_b: 检测框结果 [N, 6] - [x1,y1,x2,y2,score,cls_id]
        pred_poly: 预测的轮廓点 [N, P, 2]
        gt_poly: 真值轮廓点 [N, P, 2]
        save_path: 保存路径
        init_poly: 初始轮廓点 [N, P, 2]（可选）
        gt4_poly: GT 4点极值轮廓 [N, 4, 2]（可选）
    
    可视化说明：
    - 绿色矩形：YOLO检测结果，显示类别ID和置信度
    - 红色轮廓：模型预测的分割结果
    - 蓝色轮廓：Ground Truth真值轮廓
    - 黄色轮廓：初始轮廓（扩散演化的起点）
    - 紫色轮廓+点：GT 4点极值框（用于训练数据构造）
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
            # 确保坐标为Python浮点数类型
            x1 = float(x1.item()) if hasattr(x1, 'item') else float(x1)
            y1 = float(y1.item()) if hasattr(y1, 'item') else float(y1)
            x2 = float(x2.item()) if hasattr(x2, 'item') else float(x2)
            y2 = float(y2.item()) if hasattr(y2, 'item') else float(y2)
            score = float(score.item()) if hasattr(score, 'item') else float(score)
            cls_id = int(cls_id.item()) if hasattr(cls_id, 'item') else int(cls_id)
            # 跳过低置信度检测
            if score <= 0:
                continue
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            # 绘制矩形框
            cv2.rectangle(img, p1, p2, (0, 255, 0), 1)
            # 添加类别和置信度标签
            cv2.putText(img, f"{cls_id}:{score:.2f}", (p1[0], max(0, p1[1]-2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    # 绘制预测轮廓（红色）
    if pred_poly is not None and len(pred_poly) > 0:
        for k in range(pred_poly.shape[0]):
            poly = pred_poly[k]
            # 将轮廓闭合（添加第一个点到末尾）
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
            # 绘制连线
            loop = np.concatenate([pts, pts[:1]], axis=0)
            cv2.polylines(img, [loop], isClosed=True, color=(255, 0, 255), thickness=1)
            # 绘制关键点
            for p in pts:
                cv2.circle(img, (int(p[0]), int(p[1])), 2, (255, 0, 255), -1)

    # 创建保存目录并保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def infer_and_save(trainer, batch, tag: str):
    """
    执行模型推理并保存可视化结果
    
    该函数将网络输出转换为可视化格式，包括坐标变换、数据类型转换等，
    然后调用draw_results函数生成训练过程的可视化图像。
    
    Args:
        trainer: 训练器对象，包含训练好的网络模型
        batch: 输入数据批次，包含图像、元数据等信息
        tag: 图像保存的标签，用于区分不同步骤的结果
    
    流程说明：
    1. 执行模型前向推理
    2. 准备背景图像（转换为BGR uint8格式）
    3. 计算仿射变换矩阵（从特征图坐标到图像坐标）
    4. 转换各种检测结果和轮廓坐标
    5. 生成可视化图像并保存
    """
    # 设置为评估模式，禁用梯度计算
    # 若网络被数据并行封装，需要取出实际模型
    net = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
    net.eval()
    with torch.no_grad():
        output, _, _, _ = net(batch)

    # 准备仿射输入空间的背景图像
    inp_bgr = batch['orig_img'][0]
    if isinstance(inp_bgr, torch.Tensor):
        inp_np = inp_bgr.detach().float().cpu().numpy()
        # 处理不同的图像格式和数据范围
        if inp_np.max() <= 255.0 and inp_np.ndim == 3 and inp_np.shape[-1] == 3:
            if inp_np.dtype != np.uint8:
                inp_np = np.clip(inp_np, 0, 255).astype(np.uint8)
        else:
            # 归一化到[0, 255]范围
            inp_np = inp_np - inp_np.min()
            if inp_np.max() > 0:
                inp_np = inp_np / inp_np.max()
            inp_np = (inp_np * 255.0).astype(np.uint8)
        inp_img = inp_np
    else:
        inp_img = np.array(inp_bgr)

    def _to_numpy(x):
        """将张量转换为numpy数组的辅助函数"""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    # 计算从图像坐标到仿射输入坐标的变换矩阵
    center = _to_numpy(batch['meta']['center'][0])
    scale = _to_numpy(batch['meta']['scale'][0])
    input_w, input_h = int(snake_config.voc_input_w), int(snake_config.voc_input_h)
    trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h]).astype(np.float32)

    def apply_affine_pts(pts, M):
        """对点集应用仿射变换"""
        return data_utils.affine_transform(pts.reshape(-1, 2), M).reshape(pts.shape)

    # 处理检测结果（YOLO检测框）
    det = output.get('detection', None)
    det_aff = None
    if det is not None and isinstance(det, torch.Tensor) and det.size(0) > 0:
        det_b = det[0].detach().float().cpu()
        det_raw = det_b.numpy().copy()
        # 将检测框坐标从特征图空间转换到仿射输入空间
        for i in range(det_raw.shape[0]):
            x1, y1, x2, y2 = det_raw[i, :4]
            p = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            p_aff = apply_affine_pts(p, trans_input)
            det_raw[i, 0:2] = p_aff[0]
            det_raw[i, 2:4] = p_aff[1]
        det_aff = torch.from_numpy(det_raw)

    # 处理预测轮廓
    pred_py = output.get('py', None)
    pred_aff = None
    if pred_py is not None:
        last = pred_py[-1] if isinstance(pred_py, list) else pred_py
        if isinstance(last, torch.Tensor) and last.numel() > 0:
            # 从特征图坐标转换到图像坐标（乘以下采样比例）
            pred_aff = last.detach().float().cpu().numpy() * float(snake_config.down_ratio)

    # 处理初始轮廓
    init_py = output.get('it_py', None)
    init_aff = None
    if isinstance(init_py, torch.Tensor) and init_py.numel() > 0:
        init_aff = init_py.detach().float().cpu().numpy() * float(snake_config.down_ratio)

    # 处理GT轮廓
    gt_aff = None
    if 'i_gt_py' in batch and 'meta' in batch and 'ct_num' in batch['meta']:
        ct_meta = batch['meta']['ct_num']
        ct_num = int(ct_meta[0].item()) if isinstance(ct_meta, torch.Tensor) else int(ct_meta)
        gt = batch['i_gt_py'][0][:ct_num]  # 只取有效数量的轮廓
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().float().cpu().numpy()
        gt_aff = gt * float(snake_config.down_ratio)

    # 处理GT 4点极值轮廓
    gt4_aff = None
    if 'i_gt_4py' in batch and 'meta' in batch and 'ct_num' in batch['meta']:
        ct_meta = batch['meta']['ct_num']
        ct_num = int(ct_meta[0].item()) if isinstance(ct_meta, torch.Tensor) else int(ct_meta)
        gt4 = batch['i_gt_4py'][0][:ct_num]
        if isinstance(gt4, torch.Tensor):
            gt4 = gt4.detach().float().cpu().numpy()
        gt4_aff = gt4 * float(snake_config.down_ratio)

    # 创建保存目录并生成可视化图像（与 diffusion_one_sample.py 保持一致）
    save_dir = os.path.join(os.path.dirname(__file__), 'visual', 'grpo_train')
    os.makedirs(save_dir, exist_ok=True)
    save_path_aff = os.path.join(save_dir, f'vis_affine_{tag}.png')
    draw_results(inp_img, det_aff, pred_aff, gt_aff, save_path_aff, init_poly=init_aff, gt4_poly=gt4_aff)


def main():
    """
    GRPO强化学习训练主函数
    
    该函数实现了完整的GRPO训练流程，包括：
    1. 配置参数设置和环境变量读取
    2. 模型、训练器、优化器构建
    3. 预训练权重加载
    4. GRPO训练循环
    5. 可视化和日志记录
    6. 模型检查点保存
    
    环境变量配置：
    - GRPO_STEPS: GRPO训练步数（默认20）
    - GRPO_K: 采样数量（默认4）
    - GRPO_WINDOW: 时间窗口大小（默认6）
    - GRPO_REWARD_W: 奖励权重（默认1.0）
    - GRPO_TRAIN_STEPS: 总训练步数（默认100）
    """
    # === 基础配置 ===
    is_main = True

    # === GRPO相关配置参数设置 ===
    # 启用扩散演化模块
    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True
    cfg.use_grpo = True
    # 仅使用每张图像的第一个轮廓参与GRPO训练（降低显存、满足“只训练第一个轮廓”的需求）
    cfg.grpo_first_contour_only = False
    
    # 冻结YOLO检测网络，只训练扩散演化部分
    cfg.freeze_yolo = True
    
    # 从环境变量读取GRPO超参数，提供默认值
    cfg.grpo_steps = int(os.environ.get('GRPO_STEPS', '50'))           # GRPO采样步数
    cfg.grpo_k = int(os.environ.get('GRPO_K', '4'))                     # 每个prompt的采样数量
    cfg.grpo_window_size = int(os.environ.get('GRPO_WINDOW', str(cfg.grpo_window_size)))
    cfg.grpo_window_range = cfg.grpo_window_range                                     # 时间窗口范围
    cfg.reward_w_region = float(os.environ.get('GRPO_REWARD_W', '0')) # 区域奖励权重

    # === 构建训练组件 ===
    # 构建网络模型
    network = make_network(cfg)
    # 构建训练器
    trainer = make_trainer(cfg, network)
    # 构建优化器
    optimizer = make_optimizer(cfg, trainer.network)
    # 构建记录器
    recorder = make_recorder(cfg)

    # === 数据加载器 ===
    # 创建训练数据加载器
    data_loader = make_data_loader(cfg, is_train=True, is_distributed=False)
    # 为可视化创建一个固定的测试数据批次，避免训练增强导致的差异
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
    # 预训练模型路径
    ckpt_path = '/home/medteam/Zhrch/EnergeSnake1GRPO/data/one_sample/model_final.pth'
    try:
        # 获取网络权重对象（兼容数据并行封装）
        w = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
        
        # 1. 获取模型当前定义的总层数（所有参数键）
        model_state_dict_keys = list(w.state_dict().keys())
        total_layers = len(model_state_dict_keys)

        # 加载检查点文件
        sd = torch.load(ckpt_path, map_location='cpu')
        # 处理不同的检查点格式
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
            
        # 2. 加载权重（使用strict=False避免严格匹配）
        missing, unexpected = w.load_state_dict(sd, strict=True)
        
        # 3. 计算统计信息
        missing_count = len(missing)
        unexpected_count = len(unexpected)
        # 成功加载的层数 = 模型总层数 - 缺失的层数
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
    # 从环境变量读取训练步数
    steps = int(os.environ.get('GRPO_TRAIN_STEPS', '100'))
    it = iter(data_loader)
    
    # 准备JSONL日志文件路径
    log_dir = Path(__file__).resolve().parent / 'data' / 'outputs' / 'grpo'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'logs.jsonl'
    
    # 主训练循环
    for step in range(1, steps + 1):
        try:
            # 获取下一个批次数据
            batch = next(it)
        except StopIteration:
            # 数据加载器耗尽时重新创建迭代器
            it = iter(data_loader)
            batch = next(it)
        
        # === 数据预处理 ===
        # 将批次中的张量移动到GPU
        for k in list(batch.keys()):
            if k == 'meta':
                continue  # 元数据不需要移动到GPU
            v = batch[k]
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda(non_blocking=True)
        
        # === 模型训练步骤 ===
        # 设置为训练模式
        trainer.network.train()
        # 前向传播
        print(batch.keys())
        print(batch['inp'].shape)  
        output, loss, loss_stats, _ = trainer.network(batch)
        # 多卡训练包裹时 loss 可能是张量列表，需要显式取均值
        loss = loss.mean()
        
        # === 反向传播和优化 ===
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 反向传播计算梯度
        # 梯度裁剪，防止梯度爆炸（若使用DP封装需针对实际模型）
        clip_target = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
        torch.nn.utils.clip_grad_value_(clip_target.parameters(), 40)
        optimizer.step()       # 更新模型参数
        
        # === 可视化保存 ===
        # 每200步保存一次可视化结果
        if is_main and step % 200 == 0:
            infer_and_save(trainer, vis_batch if vis_batch is not None else batch, tag=f'step{step}')
            trainer.network.train()  # 恢复训练模式
        
        # === 日志记录 ===
        try:
            # 提取损失统计信息
            ls = {k: float(getattr(v, 'item', lambda: v)()) for k, v in loss_stats.items()}
            
            # 每20步打印一次训练进度
            if is_main and step % 20 == 0:
                print(f"[GRPO] 步骤 {step} "
                      f"总损失={float(loss.item()):.4f} "
                      f"GRPO损失={ls.get('grpo_loss', 0.0):.4f} "
                      f"奖励均值={ls.get('grpo_reward_mean', 0.0):.4f} "
                      f"奖励标准差={ls.get('grpo_reward_std', 0.0):.4f}")
            
            # 构建日志条目
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
            
            # 写入JSONL日志文件
            if is_main:
                with open(str(log_path), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_item, ensure_ascii=False) + '\n')
                
        except Exception:
            # 损失统计格式异常时的备用日志记录
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
        # 创建GRPO输出目录
        out_dir = Path(__file__).resolve().parent / 'data' / 'outputs' / 'grpo'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成带时间戳的文件名
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = 'model_grpo.pth'
        save_path = out_dir / base_name
        # 若已存在，则使用时间戳后缀
        if save_path.exists():
            save_path = out_dir / f'model_grpo_{ts}.pth'
        
        # 获取要保存的网络权重
        w = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
        # 保存检查点
        if is_main:
            torch.save(w.state_dict(), str(save_path))
            print(f'[GRPO] 模型检查点已保存: {save_path}')
        
    except Exception as e:
        if is_main:
            print(f'[GRPO] 保存检查点失败: {e}')

if __name__ == '__main__':
    """
    程序入口点
    
    当直接运行此脚本时，启动GRPO强化学习训练流程。
    
    使用示例：
    # 基础训练
    python grpo_train.py
    
    # 自定义参数训练
    GRPO_STEPS=30 GRPO_K=8 GRPO_WINDOW=10 GRPO_TRAIN_STEPS=200 python grpo_train.py
    
    # 查看训练日志
    tail -f data/outputs/grpo/logs.jsonl
    
    # 查看可视化结果
    ls visual/grpo_train/
    """
    main()
