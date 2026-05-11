"""
GRPO / reward post-training script for diffusion snake models.
"""

# 导入系统库
import os               # 文件系统操作
import cv2              # OpenCV图像处理
import gc               # 垃圾回收
import torch            # PyTorch深度学习框架
import numpy as np      # 数值计算
import datetime         # 时间处理
from pathlib import Path# 路径操作
import json             # JSON数据处理
import random           # 随机种子控制
from typing import Optional
from torch.cuda.amp import GradScaler, autocast

# 导入项目内部模块
from lib.config import cfg, args                     # 配置管理
from lib.networks import make_network                # 网络构建
from lib.train.trainers import make_trainer          # 训练器构建
from lib.train.optimizer import make_optimizer        # 优化器构建
from lib.train.recorder import make_recorder          # 记录器构建
from lib.datasets import make_data_loader             # 数据加载器构建
from lib.utils.snake import snake_config              # Snake配置
from lib.utils import data_utils                      # 数据工具
from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict


_THIS_DIR = Path(__file__).resolve().parent


def _cfg_file_used() -> str:
    return str(getattr(args, 'cfg_file', '') or os.environ.get('CFG_FILE', '')).strip()


def _cfg_stem() -> str:
    cfg_file = _cfg_file_used()
    if cfg_file:
        return Path(cfg_file).stem
    model_dir = str(getattr(cfg, 'model_dir', '') or '').strip()
    if model_dir:
        return Path(model_dir).name
    return 'btcv_diffusion_dit_v3_1_fm_posttrain'


def _candidate_stems():
    stems = []
    for raw in (_cfg_stem(), Path(str(getattr(cfg, 'model_dir', '') or '')).name):
        raw = str(raw or '').strip()
        if not raw:
            continue
        stems.append(raw)
        for suffix in ('_fm_posttrain', '_posttrain', '_fm_grpo', '_grpo_fm', '_grpo'):
            if raw.endswith(suffix):
                stems.append(raw[:-len(suffix)])
    seen = set()
    ordered = []
    for stem in stems:
        if stem and stem not in seen:
            seen.add(stem)
            ordered.append(stem)
    return ordered


def _resolve_checkpoint_path() -> Optional[Path]:
    explicit = os.environ.get('CKPT_PATH', '').strip()
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    arg_ckpt = str(getattr(args, 'ckpt', '') or '').strip()
    if arg_ckpt:
        candidates.append(Path(arg_ckpt))

    resume_path = str(getattr(cfg, 'resume_path', '') or '').strip()
    if resume_path:
        candidates.append(Path(resume_path))
        candidates.append(_THIS_DIR / resume_path)

    model_dir = str(getattr(cfg, 'model_dir', '') or '').strip()
    if model_dir:
        model_path = Path(model_dir)
        candidates.append(model_path)
        candidates.append(model_path / 'checkpoints' / 'latest.pt')
        candidates.append(_THIS_DIR / model_path)
        candidates.append(_THIS_DIR / model_path / 'checkpoints' / 'latest.pt')

    for stem in _candidate_stems():
        candidates.append(_THIS_DIR / 'data' / 'outputs' / stem / 'checkpoints' / 'latest.pt')
        candidates.append(_THIS_DIR / 'data' / 'outputs' / stem / 'latest.pt')

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate.suffix not in ('.pt', '.pth'):
            continue
        candidate = candidate.resolve() if candidate.exists() else candidate
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def _project_path(path_like) -> Path:
    path = Path(str(path_like)).expanduser()
    if path.is_absolute():
        return path
    return _THIS_DIR / path


def _output_dir() -> Path:
    model_dir = str(getattr(cfg, 'model_dir', '') or '').strip()
    if model_dir:
        return _project_path(model_dir)
    return _THIS_DIR / 'data' / 'outputs' / _cfg_stem()


def _resolve_reference_path() -> str:
    raw = str(
        os.environ.get('GRPO_REF_PATH', '')
        or getattr(cfg, 'grpo_reference_path', '')
        or getattr(cfg, 'grpo_ref_path', '')
        or ''
    ).strip()
    if not raw:
        return ''
    path = _project_path(raw)
    if not path.exists():
        raise FileNotFoundError(f'GRPO reference checkpoint not found: {path}')
    return str(path.resolve())


def _extract_state_dict(ckpt_obj):
    if not isinstance(ckpt_obj, dict):
        return ckpt_obj
    for key in ('state_dict', 'model', 'net', 'network'):
        if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
            return ckpt_obj[key]
    return ckpt_obj


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, '').strip().lower()
    if raw == '':
        return bool(default)
    return raw in ('1', 'true', 'yes', 'y', 'on')


def _parse_int_pair_env(name: str, default):
    raw = os.environ.get(name, '').strip()
    if not raw:
        return tuple(default)
    parts = raw.replace(',', ' ').split()
    if len(parts) != 2:
        raise ValueError(f'{name} must contain exactly two integers, got: {raw}')
    return int(parts[0]), int(parts[1])


def _set_train_seed(seed: int):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    print(f'[GRPO] train seed: {seed}')


def _adapt_state_dict_for_model(model, state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict

    state_dict = remap_legacy_state_dict(state_dict)
    model_keys = list(model.state_dict().keys())
    if not model_keys or not state_dict:
        return state_dict

    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    needs_net_prefix = any(k.startswith('net.') for k in model_keys)
    has_net_prefix = any(k.startswith('net.') for k in state_dict.keys())
    if needs_net_prefix and (not has_net_prefix):
        state_dict = {f'net.{k}': v for k, v in state_dict.items()}
    return state_dict


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
    if bool(getattr(cfg, 'use_dit_v3_1', False)) and (not bool(getattr(cfg, 'use_flow_matching', False))):
        print('[GRPO] V3.1 post-training defaults to Flow Matching; enabling cfg.use_flow_matching=True')
        cfg.use_flow_matching = True

    cfg.grpo_first_contour_only = bool(getattr(cfg, 'grpo_first_contour_only', False))
    cfg.freeze_yolo = bool(getattr(cfg, 'freeze_yolo', True))

    # 环境变量
    cfg.grpo_steps = int(os.environ.get('GRPO_STEPS', str(getattr(cfg, 'grpo_steps', 20 if cfg.use_flow_matching else 50))))
    cfg.grpo_k = int(os.environ.get('GRPO_K', str(getattr(cfg, 'grpo_k', 2 if cfg.use_flow_matching else 8))))
    cfg.grpo_window_size = int(os.environ.get('GRPO_WINDOW', str(getattr(cfg, 'grpo_window_size', 0 if cfg.use_flow_matching else 6))))
    cfg.grpo_window_range = _parse_int_pair_env(
        'GRPO_WINDOW_RANGE',
        getattr(cfg, 'grpo_window_range', (0, max(cfg.grpo_steps - 1, 0)) if cfg.use_flow_matching else (2, 12)),
    )
    cfg.reward_w_region = float(os.environ.get('GRPO_REWARD_W', str(getattr(cfg, 'reward_w_region', 1.0))))
    cfg.reward_w_dice = float(os.environ.get('GRPO_REWARD_W_DICE', str(getattr(cfg, 'reward_w_dice', 0.0))))
    cfg.reward_w_iou = float(os.environ.get('GRPO_REWARD_W_IOU', str(getattr(cfg, 'reward_w_iou', 0.0))))
    cfg.grpo_reward_delta = _parse_bool_env('GRPO_REWARD_DELTA', bool(getattr(cfg, 'grpo_reward_delta', False)))
    cfg.grpo_pure_rl_loss = _parse_bool_env('GRPO_PURE_RL_LOSS', bool(getattr(cfg, 'grpo_pure_rl_loss', False)))
    cfg.grpo_adv_center = os.environ.get('GRPO_ADV_CENTER', str(getattr(cfg, 'grpo_adv_center', 'group'))).strip().lower()
    cfg.grpo_reference_path = _resolve_reference_path()
    cfg.grpo_reward_abs_weight = float(os.environ.get('GRPO_REWARD_ABS_W', str(getattr(cfg, 'grpo_reward_abs_weight', 0.0 if cfg.grpo_reward_delta else 1.0))))
    cfg.grpo_reward_delta_weight = float(os.environ.get('GRPO_REWARD_DELTA_W', str(getattr(cfg, 'grpo_reward_delta_weight', 1.0 if cfg.grpo_reward_delta else 0.0))))
    cfg.grpo_smoothness_weight = float(os.environ.get('GRPO_SMOOTH_W', str(getattr(cfg, 'grpo_smoothness_weight', 0.0))))
    train_seed = int(os.environ.get('GRPO_SEED', str(getattr(cfg, 'grpo_seed', 20260504))))
    _set_train_seed(train_seed)
    if cfg.grpo_reference_path:
        print(f'[GRPO] fixed reference checkpoint: {cfg.grpo_reference_path}')

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

    # === 加载预训练权重 / 续训状态 ===
    ckpt_path = _resolve_checkpoint_path()
    resume_state = _parse_bool_env('GRPO_RESUME_STATE', False)
    resume_step = 0
    loaded_ckpt = None
    try:
        w = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
        if ckpt_path is None:
            raise FileNotFoundError(
                'No checkpoint found for post-training. '
                'Set CKPT_PATH or keep a latest.pt under data/outputs/<cfg-stem>/checkpoints/.'
            )

        model_state_dict_keys = list(w.state_dict().keys())
        total_layers = len(model_state_dict_keys)

        loaded_ckpt = torch.load(str(ckpt_path), map_location='cpu')
        sd = _extract_state_dict(loaded_ckpt)
        sd = _adapt_state_dict_for_model(w, sd)

        missing, unexpected = w.load_state_dict(sd, strict=False)

        missing_count = len(missing)
        unexpected_count = len(unexpected)
        loaded_count = total_layers - missing_count
        load_ratio = (loaded_count / total_layers * 100) if total_layers > 0 else 0

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
        min_load_ratio = float(os.environ.get('GRPO_MIN_LOAD_RATIO', '95.0'))
        if load_ratio < min_load_ratio:
            raise RuntimeError(
                f'Checkpoint load ratio too low for post-training: '
                f'{load_ratio:.2f}% < {min_load_ratio:.2f}%'
            )

        if resume_state:
            if not isinstance(loaded_ckpt, dict):
                raise RuntimeError('GRPO_RESUME_STATE=1 requires a dict checkpoint with optimizer and step state.')
            posttrain_meta = loaded_ckpt.get('posttrain', {})
            if not bool(getattr(posttrain_meta, 'get', lambda *_: False)('use_grpo', False)):
                raise RuntimeError(
                    'GRPO_RESUME_STATE=1 requires a GRPO posttrain checkpoint; '
                    f'got checkpoint without posttrain.use_grpo=true: {ckpt_path}'
                )
            optimizer_state = loaded_ckpt.get('optimizer', None)
            if not isinstance(optimizer_state, dict):
                raise RuntimeError(
                    'GRPO_RESUME_STATE=1 requires optimizer state in the checkpoint; '
                    f'missing optimizer in {ckpt_path}'
                )
            saved_ref = str(posttrain_meta.get('grpo_reference_path', '') or '').strip()
            if cfg.grpo_reference_path:
                if not saved_ref:
                    raise RuntimeError(
                        'GRPO_RESUME_STATE=1 with fixed reference requires checkpoint metadata '
                        f'posttrain.grpo_reference_path; missing in {ckpt_path}'
                    )
                saved_ref_resolved = str(_project_path(saved_ref).resolve())
                current_ref_resolved = str(_project_path(cfg.grpo_reference_path).resolve())
                if saved_ref_resolved != current_ref_resolved:
                    raise RuntimeError(
                        'GRPO fixed reference mismatch on resume: '
                        f'checkpoint has {saved_ref_resolved}, current config has {current_ref_resolved}'
                    )
            resume_step = int(loaded_ckpt.get('step', 0) or 0)
            optimizer.load_state_dict(optimizer_state)
            print(f'[GRPO] 续训模式已开启: 从 step {resume_step} 恢复优化器与模型状态')

    except Exception as e:
        if is_main:
            print(f'[GRPO] 加载预训练权重失败: {e}')
            import traceback
            traceback.print_exc()
        raise

    # === GRPO训练循环 ===
    steps = int(os.environ.get('GRPO_TRAIN_STEPS', str(getattr(cfg, 'grpo_train_steps', 1000))))
    it = iter(data_loader)
    use_amp = bool(int(os.environ.get('GRPO_AMP', '1')))
    scaler = GradScaler(enabled=use_amp)
    grad_accum_steps = max(int(getattr(getattr(cfg, 'train', object()), 'gradient_accumulation_steps', 1)), 1)
    
    out_dir = _output_dir()
    ckpt_dir = out_dir / 'checkpoints'
    log_dir = out_dir / 'posttrain_grpo'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'logs.jsonl'

    if is_main:
        print(f'[GRPO] AMP enabled: {use_amp}')
        print(f'[GRPO] Gradient accumulation steps: {grad_accum_steps}')

    def _save_checkpoint(step: int, final: bool = False):
        w = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
        state_dict = {
            k: v for k, v in w.state_dict().items()
            if not (k.startswith('ref_flow.') or k.startswith('ref_gcn.'))
        }
        ckpt = {
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),
            'step': int(step),
            'timestamp': datetime.datetime.now().isoformat(),
            'cfg_file': _cfg_file_used(),
            'base_ckpt': str(ckpt_path),
            'posttrain': {
                'use_grpo': bool(getattr(cfg, 'use_grpo', False)),
                'use_flow_matching': bool(getattr(cfg, 'use_flow_matching', False)),
                'grpo_k': int(getattr(cfg, 'grpo_k', 0)),
                'grpo_loss_weight': float(getattr(cfg, 'grpo_loss_weight', 0.0)),
                'grpo_action_std': float(getattr(cfg, 'grpo_action_std', 0.0)),
                'diffusion_loss_weight': float(getattr(cfg, 'diffusion_loss_weight', 0.0)),
                'grpo_pure_rl_loss': bool(getattr(cfg, 'grpo_pure_rl_loss', False)),
                'grpo_seed': int(train_seed),
                'grpo_reward_image_scale': bool(getattr(cfg, 'grpo_reward_image_scale', False)),
                'grpo_reward_delta': bool(getattr(cfg, 'grpo_reward_delta', False)),
                'grpo_reward_abs_weight': float(getattr(cfg, 'grpo_reward_abs_weight', 0.0)),
                'grpo_reward_delta_weight': float(getattr(cfg, 'grpo_reward_delta_weight', 0.0)),
                'grpo_smoothness_weight': float(getattr(cfg, 'grpo_smoothness_weight', 0.0)),
                'grpo_adv_center': str(getattr(cfg, 'grpo_adv_center', 'group')),
                'grpo_reference_path': str(getattr(cfg, 'grpo_reference_path', '') or ''),
                'reward_w_region': float(getattr(cfg, 'reward_w_region', 0.0)),
                'reward_w_dice': float(getattr(cfg, 'reward_w_dice', 0.0)),
                'reward_w_iou': float(getattr(cfg, 'reward_w_iou', 0.0)),
            },
        }
        latest_path = ckpt_dir / 'latest.pt'
        torch.save(ckpt, str(latest_path))
        if final:
            torch.save(ckpt, str(ckpt_dir / f'final_step_{step}.pt'))
        return latest_path
    
    optimizer.zero_grad()

    start_step = resume_step + 1 if resume_state else 1
    if start_step > steps:
        if is_main:
            print(f'[GRPO] 当前 checkpoint 已到 step {resume_step}，目标总步数={steps}，无需继续训练')
        return

    for step in range(start_step, steps + 1):
        trainer.network.train()

        # [修复 1]: 即使在训练模式下，强制冻结所有BN层。
        # 否则小批次GRPO采样会破坏预训练的统计分布 (running_mean/var)，导致性能恶化。
        for m in trainer.network.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d, torch.nn.SyncBatchNorm)):
                m.eval()

        loss_sum = 0.0
        stats_sum = {}
        batch = None
        output = None

        for accum_idx in range(grad_accum_steps):
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

            with autocast(enabled=use_amp):
                output, loss, loss_stats, _ = trainer.network(batch)
                loss = loss.mean()

            loss_sum += float(loss.detach().item())
            for stat_key, stat_value in loss_stats.items():
                stats_sum[stat_key] = stats_sum.get(stat_key, 0.0) + float(getattr(stat_value, 'item', lambda: stat_value)())

            loss_to_backward = loss / grad_accum_steps
            if use_amp:
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

        loss_value = loss_sum / grad_accum_steps
        loss_stats = {k: v / grad_accum_steps for k, v in stats_sum.items()}

        # === 反向传播和优化 ===
        if use_amp:
            scaler.unscale_(optimizer)

        clip_target = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
        torch.nn.utils.clip_grad_value_(clip_target.parameters(), 40)

        # [修复 2]: 如果奖励权重为0 (GRPO_REWARD_W=0)，理论上 Loss 梯度接近 0。
        # 此时必须禁用 Weight Decay，否则权重会持续衰减（被正则化项“吃掉”），导致模型退化。
        if cfg.reward_w_region == 0:
            for group in optimizer.param_groups:
                group['weight_decay'] = 0.0

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

        save_every = int(os.environ.get('GRPO_SAVE_EVERY', str(getattr(cfg.train, 'save_every', 100))))
        if is_main and save_every > 0 and step % save_every == 0:
            save_path = _save_checkpoint(step)
            print(f'[GRPO] checkpoint saved: {save_path}')
        
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
                    f"总损失={loss_value:.4f} "
                      f"GRPO损失={ls.get('grpo_loss', 0.0):.4f} "
                      f"奖励均值={ls.get('grpo_reward_mean', 0.0):.4f} "
                      f"奖励标准差={ls.get('grpo_reward_std', 0.0):.4f}")
            
            log_item = {
                'timestamp': datetime.datetime.now().isoformat(),
                'step': int(step),
                'loss': float(loss_value),
                'grpo_loss': float(ls.get('grpo_loss', 0.0)),
                'grpo_policy_loss_raw': float(ls.get('grpo_policy_loss_raw', 0.0)),
                'reward_mean': float(ls.get('grpo_reward_mean', 0.0)),
                'reward_std': float(ls.get('grpo_reward_std', 0.0)),
                'reward_best': float(ls.get('grpo_reward_best', 0.0)),
                'final_score_mean': float(ls.get('grpo_final_score_mean', ls.get('grpo_reward_mean', 0.0))),
                'final_score_best': float(ls.get('grpo_final_score_best', ls.get('grpo_reward_best', 0.0))),
                'delta_score_mean': float(ls.get('grpo_delta_score_mean', 0.0)),
                'smoothness_penalty_mean': float(ls.get('grpo_smoothness_penalty_mean', 0.0)),
                'adv_mean': float(ls.get('grpo_adv_mean', 0.0)),
                'adv_std': float(ls.get('grpo_adv_std', 0.0)),
                'approx_kl': float(ls.get('grpo_approx_kl', 0.0)),
                'clipfrac': float(ls.get('grpo_clipfrac', 0.0)),
                'kl_loss': float(ls.get('grpo_kl_loss', 0.0)),
                'ref_load_ratio': float(ls.get('grpo_ref_load_ratio', 0.0)),
                'diff_loss': float(ls.get('diff_loss', 0.0)),
                'diff_loss_scaled': float(ls.get('diff_loss_scaled', 0.0)),
                'grpo_pure_rl_loss': bool(getattr(cfg, 'grpo_pure_rl_loss', False)),
                'grpo_steps': int(cfg.grpo_steps),
                'grpo_k': int(cfg.grpo_k),
                'grpo_window_size': int(cfg.grpo_window_size),
                'grpo_window_range': tuple(getattr(cfg, 'grpo_window_range', (2, 12))),
                'grpo_loss_weight': float(getattr(cfg, 'grpo_loss_weight', 0.0)),
                'grpo_action_std': float(getattr(cfg, 'grpo_action_std', 0.0)),
                'grpo_seed': int(train_seed),
                'grpo_reward_image_scale': bool(getattr(cfg, 'grpo_reward_image_scale', False)),
                'grpo_reward_delta': bool(getattr(cfg, 'grpo_reward_delta', False)),
                'grpo_reward_abs_weight': float(getattr(cfg, 'grpo_reward_abs_weight', 0.0)),
                'grpo_reward_delta_weight': float(getattr(cfg, 'grpo_reward_delta_weight', 0.0)),
                'grpo_smoothness_weight': float(getattr(cfg, 'grpo_smoothness_weight', 0.0)),
                'grpo_adv_center': str(getattr(cfg, 'grpo_adv_center', 'group')),
                'grpo_reference_path': str(getattr(cfg, 'grpo_reference_path', '') or ''),
                'reward_w_region': float(getattr(cfg, 'reward_w_region', 0.0)),
                'reward_w_dice': float(getattr(cfg, 'reward_w_dice', 0.0)),
                'reward_w_iou': float(getattr(cfg, 'reward_w_iou', 0.0)),
                'base_ckpt': str(ckpt_path),
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
                    'loss': float(loss_value)
                }
                with open(str(log_path), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_item, ensure_ascii=False) + '\n')

        # 释放当前步的张量引用，降低长跑训练中的显存碎片和峰值压力。
        try:
            del batch, output, loss, loss_stats
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === 最终可视化 ===
    if is_main:
        infer_and_save(trainer, vis_batch if 'vis_batch' in locals() and vis_batch is not None else batch, tag='final')

    # === 保存GRPO微调后的模型 ===
    if is_main:
        save_path = _save_checkpoint(steps, final=True)
        print(f'[GRPO] 模型检查点已保存: {save_path}')

if __name__ == '__main__':
    main()
