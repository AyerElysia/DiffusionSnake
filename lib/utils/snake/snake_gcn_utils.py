import torch
import os
import numpy as np
from lib.utils.snake import snake_decode, snake_config
try:
    from lib.csrc.extreme_utils._ext import _ext as extreme_utils
except ImportError:
    # 彻底冗余检查，防止其它路径失败
    from lib.csrc.extreme_utils._ext import _ext as extreme_utils
from lib.utils import data_utils
import torch.nn.functional as F
import sys

# get_gcn_feature() 170行

def collect_training(poly, ct_01):
    batch_size = ct_01.size(0)
    poly = torch.cat([poly[i][ct_01[i]] for i in range(batch_size)], dim=0) # ct_01[i] 是一个布尔值（0 或 1），它指示是否选择 poly 中第 i 个元素
    return poly


def prepare_training_init(ret, batch):
    ct_01 = batch['ct_01'].bool()
    init = {}
    init.update({'i_it_4py': collect_training(batch['i_it_4py'], ct_01)})
    init.update({'c_it_4py': collect_training(batch['c_it_4py'], ct_01)})
    init.update({'i_gt_4py': collect_training(batch['i_gt_4py'], ct_01)})
    init.update({'c_gt_4py': collect_training(batch['c_gt_4py'], ct_01)})

    ct_num = batch['meta']['ct_num']
    init.update({'ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})

    return init


def prepare_testing(output):
    # 测试/推理阶段统一入口：优先使用外部传入的 SAM init；
    # 否则读取 output['detection'] 中的 bbox/score，走 prepare_testing_init。
    # 返回的 i_it_py/c_it_py/py_ind 会被 diffusion/flow evolution 当作初始 contour。
    if 'sam_i_it_py' in output and output['sam_i_it_py'] is not None:
        i_it_py = output['sam_i_it_py']
        c_it_py = output.get('sam_c_it_py', img_poly_to_can_poly(i_it_py))
        ind = output.get('sam_py_ind', torch.zeros((i_it_py.size(0),), dtype=torch.long, device=i_it_py.device))
        i_it_4py = i_it_py[:, :snake_config.init_poly_num] if i_it_py.numel() > 0 else i_it_py.new_zeros((0, snake_config.init_poly_num, 2))
        init = {'i_it_4py': i_it_4py, 'c_it_4py': i_it_4py.clone(), 'ind': ind}
        init.update({'i_it_py': i_it_py, 'c_it_py': c_it_py, 'py_ind': ind})
        return init

    box = output['detection'][..., :4]
    score = output['detection'][..., 4]
    init = prepare_testing_init(box, score)
    i_it_4py = init['i_it_4py']
    ind = init['ind']
    if i_it_4py.numel() == 0:
        i_it_py = torch.zeros([0, snake_config.poly_num, 2], device=i_it_4py.device, dtype=i_it_4py.dtype)
        c_it_py = torch.zeros_like(i_it_py)
    else:
        i_it_py = uniform_upsample(i_it_4py.unsqueeze(0), snake_config.poly_num)[0]
        c_it_py = img_poly_to_can_poly(i_it_py)
    init.update({'i_it_py': i_it_py, 'c_it_py': c_it_py, 'py_ind': ind})
    return init


def _get_box_from_extreme(ex):
    x_min = torch.min(ex[..., 0], dim=-1)[0]
    y_min = torch.min(ex[..., 1], dim=-1)[0]
    x_max = torch.max(ex[..., 0], dim=-1)[0]
    y_max = torch.max(ex[..., 1], dim=-1)[0]
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


def _get_evolve_init_from_extreme(ex):
    if snake_config.evolve_init == 'octagon':
        return snake_decode.get_octagon(ex)
    return snake_decode.get_box(_get_box_from_extreme(ex))


def _box_to_octagon_init(box, p_num):
    """Build a DeepSnake-style octagon from xyxy boxes and upsample it."""
    if box.numel() == 0:
        return box.new_zeros((0, int(p_num), 2))
    if box.dim() == 2:
        box_batched = box.unsqueeze(0)
        squeeze_batch = True
    elif box.dim() == 3:
        box_batched = box
        squeeze_batch = False
    else:
        raise ValueError(f'box must have shape [N,4] or [B,N,4], got {tuple(box.shape)}')

    ex = snake_decode.get_quadrangle(box_batched)
    octagon = snake_decode.get_octagon(ex)
    poly = uniform_upsample(octagon, int(p_num))
    return poly[0] if squeeze_batch else poly


def build_box_octagon_from_poly(gt_poly, p_num=None):
    """Use each GT contour bbox as the only initialization signal."""
    if p_num is None:
        p_num = snake_config.poly_num
    if gt_poly.numel() == 0:
        return gt_poly.new_zeros((0, int(p_num), 2))
    box = torch.cat([gt_poly.min(dim=1)[0], gt_poly.max(dim=1)[0]], dim=1)
    return _box_to_octagon_init(box, int(p_num))


def replace_training_init_with_gt_box_octagon(train_dict):
    """Make diffusion train from bbox-derived octagons instead of GT extremes."""
    if 'i_gt_py' not in train_dict:
        return train_dict
    i_gt_py = train_dict['i_gt_py']
    if not torch.is_tensor(i_gt_py) or i_gt_py.numel() == 0:
        return train_dict

    i_it_py = build_box_octagon_from_poly(i_gt_py, snake_config.poly_num)
    i_it_4py = build_box_octagon_from_poly(i_gt_py, snake_config.init_poly_num)
    train_dict = dict(train_dict)
    train_dict['i_it_py'] = i_it_py
    train_dict['c_it_py'] = img_poly_to_can_poly(i_it_py)
    train_dict['i_it_4py'] = i_it_4py
    train_dict['c_it_4py'] = img_poly_to_can_poly(i_it_4py)
    return train_dict


def prepare_testing_init(box, score):
    """Convert valid detector boxes to fixed-size Snake initialization polygons.

    Dense [B,N,6] detection tensors may contain all-zero padding. Padding must be
    removed before ``snake_decode.get_init`` so no degenerate contour is created.
    This function only normalizes the detector-to-initialization boundary; it
    does not change the downstream contour evolution algorithm.
    """
    if box.dim() == 2:
        box = box.unsqueeze(0)
    if score.dim() == 1:
        score = score.unsqueeze(0)
    if box.dim() != 3 or box.size(-1) != 4:
        raise ValueError('box must have shape [B,N,4]')
    if score.dim() != 2 or tuple(score.shape) != tuple(box.shape[:2]):
        raise ValueError('score must have shape [B,N] matching box')

    valid = score > 1e-4
    if box.numel() == 0 or score.numel() == 0 or box.size(1) == 0 or not bool(valid.any()):
        empty_poly = box.new_zeros((0, snake_config.init_poly_num, 2))
        empty_ind = torch.zeros((0,), dtype=torch.long, device=box.device)
        return {'i_it_4py': empty_poly, 'c_it_4py': empty_poly.clone(), 'ind': empty_ind}

    # Preserve batch ownership while compacting only valid box slots.
    batch_ind = valid.nonzero(as_tuple=False)[:, 0]
    valid_box = box[valid].unsqueeze(0)
    i_it_4pys = snake_decode.get_init(valid_box)
    if i_it_4pys.numel() == 0 or i_it_4pys.size(1) == 0:
        empty_poly = box.new_zeros((0, snake_config.init_poly_num, 2))
        empty_ind = torch.zeros((0,), dtype=torch.long, device=box.device)
        return {'i_it_4py': empty_poly, 'c_it_4py': empty_poly.clone(), 'ind': empty_ind}

    i_it_4pys = uniform_upsample(i_it_4pys, snake_config.init_poly_num)[0]
    c_it_4pys = img_poly_to_can_poly(i_it_4pys)
    i_it_4pys = i_it_4pys / 4.0
    c_it_4pys = c_it_4pys / 4.0
    return {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'ind': batch_ind}


def get_box_match_ind(pred_box, score, gt_poly):
    if gt_poly.size(0) == 0:
        return [], []

    gt_box = torch.cat([torch.min(gt_poly, dim=1)[0], torch.max(gt_poly, dim=1)[0]], dim=1)
    iou_matrix = data_utils.box_iou(pred_box, gt_box)
    iou, gt_ind = iou_matrix.max(dim=1)
    box_ind = ((iou > snake_config.box_iou) * (score > snake_config.confidence)).nonzero().view(-1)
    gt_ind = gt_ind[box_ind]

    ind = np.unique(gt_ind.detach().cpu().numpy(), return_index=True)[1]
    box_ind = box_ind[ind]
    gt_ind = gt_ind[ind]

    return box_ind, gt_ind


def prepare_training_box(ret, batch, init):
    box = ret['detection'][..., :4]
    score = ret['detection'][..., 4]
    batch_size = box.size(0)
    i_gt_4py = batch['i_gt_4py']
    ct_01 = batch['ct_01'].bool()
    ind = [get_box_match_ind(box[i], score[i], i_gt_4py[i][ct_01[i]]) for i in range(batch_size)]
    box_ind = [ind_[0] for ind_ in ind]
    gt_ind = [ind_[1] for ind_ in ind]

    i_it_4py = torch.cat([snake_decode.get_init(box[i][box_ind[i]][None]) for i in range(batch_size)], dim=1)
    if i_it_4py.size(1) == 0:
        return

    i_it_4py = uniform_upsample(i_it_4py, snake_config.init_poly_num)[0]
    c_it_4py = img_poly_to_can_poly(i_it_4py)
    i_gt_4py = torch.cat([batch['i_gt_4py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    c_gt_4py = torch.cat([batch['c_gt_4py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    init_4py = {'i_it_4py': i_it_4py, 'c_it_4py': c_it_4py, 'i_gt_4py': i_gt_4py, 'c_gt_4py': c_gt_4py}

    i_it_py = _get_evolve_init_from_extreme(i_gt_4py[None])
    i_it_py = uniform_upsample(i_it_py, snake_config.poly_num)[0]
    c_it_py = img_poly_to_can_poly(i_it_py)
    i_gt_py = torch.cat([batch['i_gt_py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    init_py = {'i_it_py': i_it_py, 'c_it_py': c_it_py, 'i_gt_py': i_gt_py}

    ind = torch.cat([torch.full([len(gt_ind[i])], i) for i in range(batch_size)], dim=0)

    if snake_config.train_pred_box_only:
        for k, v in init_4py.items():
            init[k] = v
        for k, v in init_py.items():
            init[k] = v
        init['4py_ind'] = ind
        init['py_ind'] = ind
    else:
        init.update({k: torch.cat([init[k], v], dim=0) for k, v in init_4py.items()})
        init.update({'4py_ind': torch.cat([init['4py_ind'], ind], dim=0)})
        init.update({k: torch.cat([init[k], v], dim=0) for k, v in init_py.items()})
        init.update({'py_ind': torch.cat([init['py_ind'], ind], dim=0)})


def prepare_training(ret, batch):
    ct_01 = batch['ct_01'].bool()
    init = {}
    init.update({'i_it_4py': collect_training(batch['i_it_4py'], ct_01)})
    init.update({'c_it_4py': collect_training(batch['c_it_4py'], ct_01)})
    init.update({'i_gt_4py': collect_training(batch['i_gt_4py'], ct_01)})
    init.update({'c_gt_4py': collect_training(batch['c_gt_4py'], ct_01)})

    init.update({'i_it_py': collect_training(batch['i_it_py'], ct_01)})
    init.update({'c_it_py': collect_training(batch['c_it_py'], ct_01)})
    init.update({'i_gt_py': collect_training(batch['i_gt_py'], ct_01)})
    init.update({'c_gt_py': collect_training(batch['c_gt_py'], ct_01)})
    if 'point_mask' in batch:
        init.update({'point_mask': collect_training(batch['point_mask'], ct_01)})

    ct_num = batch['meta']['ct_num']
    init.update({'4py_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})
    init.update({'py_ind': init['4py_ind']})

    if snake_config.train_pred_box:
        prepare_training_box(ret, batch, init)

    init['4py_ind'] = init['4py_ind'].to(ct_01.device)
    init['py_ind'] = init['py_ind'].to(ct_01.device)

    return init


def prepare_training_evolve(ex, init):
    if not snake_config.train_pred_ex:
        evolve = {'i_it_py': init['i_it_py'], 'c_it_py': init['c_it_py'], 'i_gt_py': init['i_gt_py']}
        return evolve

    i_gt_py = init['i_gt_py']

    if snake_config.train_nearest_gt:
        shift = -(ex[:, :1] - i_gt_py).pow(2).sum(2).argmin(1)
        i_gt_py = extreme_utils.roll_array(i_gt_py, shift)

    i_it_py = _get_evolve_init_from_extreme(ex[None])
    i_it_py = uniform_upsample(i_it_py, snake_config.poly_num)[0]
    c_it_py = img_poly_to_can_poly(i_it_py)
    evolve = {'i_it_py': i_it_py, 'c_it_py': c_it_py, 'i_gt_py': i_gt_py}

    return evolve


def prepare_testing_evolve(ex):
    if len(ex) == 0:
        i_it_pys = torch.zeros([0, snake_config.poly_num, 2]).to(ex)
        c_it_pys = torch.zeros_like(i_it_pys)
    else:
        i_it_pys = _get_evolve_init_from_extreme(ex[None])
        i_it_pys = uniform_upsample(i_it_pys, snake_config.poly_num)[0]
        c_it_pys = img_poly_to_can_poly(i_it_pys)
    evolve = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys}
    return evolve

# 从CNN-map中提取数据，为蛇演化提供信息，这个很重要！！！
# cnn_feature为 1,64,136,136.
_GCN_SAMPLE_CFG = None


def _gcn_sample_cfg():
    """Lazily read point-sampling options from the global config.

    Kept lazy so this module stays importable without a config, and cached
    because get_gcn_feature() is called many times per forward pass.
    """
    global _GCN_SAMPLE_CFG
    if _GCN_SAMPLE_CFG is None:
        try:
            from lib.config import cfg as _cfg
            mode = str(getattr(_cfg, 'gcn_sample_mode', 'legacy'))
            padding_mode = str(getattr(_cfg, 'gcn_sample_padding_mode', 'zeros'))
        except Exception:
            mode, padding_mode = 'legacy', 'zeros'
        if mode not in ('legacy', 'half_pixel'):
            raise ValueError(
                "cfg.gcn_sample_mode must be 'legacy' or 'half_pixel', got {!r}".format(mode)
            )
        if padding_mode not in ('zeros', 'border', 'reflection'):
            raise ValueError(
                "cfg.gcn_sample_padding_mode must be zeros/border/reflection, got {!r}".format(padding_mode)
            )
        _GCN_SAMPLE_CFG = (mode, padding_mode)
    return _GCN_SAMPLE_CFG


def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
    """Sample per-point features from a feature map at contour coordinates.

    ``img_poly`` holds continuous coordinates in feature-map pixel units.

    Two coordinate conventions are supported via ``cfg.gcn_sample_mode``:

    * ``legacy`` (default) reproduces the original ResNet/YOLOv8-era formula
      ``2x/w - 1`` with ``align_corners=False``. This is half a pixel short of
      the pixel-center convention, a bias that is negligible at stride 4 but
      not on a coarse MoonViT grid.
    * ``half_pixel`` uses ``(x + 0.5) * 2/w - 1`` with ``align_corners=False``,
      which maps integer coordinates onto pixel centers exactly and matches the
      ``align_corners=True`` grid built in apply_locate_feature_replacement.
    """
    mode, padding_mode = _gcn_sample_cfg()
    if mode == 'half_pixel':
        norm_x = (img_poly[..., 0] + 0.5) * (2.0 / w) - 1
        norm_y = (img_poly[..., 1] + 0.5) * (2.0 / h) - 1
    else:
        norm_x = img_poly[..., 0] / (w / 2.) - 1  #  大小放缩至 -1 到 1 之间
        norm_y = img_poly[..., 1] / (h / 2.) - 1
    img_poly = torch.stack([norm_x, norm_y], dim=-1)

    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)

        feature = torch.nn.functional.grid_sample(
            cnn_feature[i:i + 1], poly, padding_mode=padding_mode,
        )[0].permute(1, 0, 2)
        gcn_feature[ind == i] = feature
    #print(img_poly.size)  # 7,40,2    7个多边形，每个多边形上40个点，每个点两个坐标x,y
    #print(poly.size)  # 1,7,40,2
    #print(cnn_feature.size)  # 1,64,136,136   1个batch，64个channel，特征图大小为64X64
    #print(gcn_feature.size)  # 7, 64, 40   7个图形，每个图形上有40个点，每个点有64个特征值
    return gcn_feature


def get_gcn_feature_pro(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone()  # 避免修改原始数据
    channels = cnn_feature.size(1)  # channels:128
    # im_poly 的最后一个维度本来应该就是136 x 136,这里对数值进行取整
    img_poly[..., 0]  = torch.round(img_poly[..., 0])
    img_poly[..., 1] = torch.round(img_poly[..., 1])

    batch_size = cnn_feature.size(0)  # batch_size = 1
    gcn_feature = torch.zeros([img_poly.size(0), img_poly.size(1), cnn_feature.size(1)]).to(img_poly.device)  # gcn_feature (9,64,40)

    for i in range(batch_size):
        poly = img_poly[ind == i]  # ploy   9,40,2

        # 应用填充（防止截取卷积块时出现越界）
        pad = 3
        padded_feature = F.pad(cnn_feature, (pad, pad, pad, pad), mode='constant', value=0)  # 经过padding, feature 的形状变为了 1，64，140，140

        # 遍历所有蛇点     ploy 形状为 1,9,40,2
        for poly_idx in range(poly.size(0)):   # 遍历每个多边形
            polygon = poly[poly_idx, :, :]  # 获取一个多边形的所有顶点  polygon : tensor(40,2)

            # 遍历每个多边形的顶点
            for vertex_idx in range(polygon.size(0)):
                vertex = polygon[vertex_idx, :]  # 获取一个顶点的坐标  vertex : tensor(2,)

                # 这里访问 vertex 来执行所需的操作
                xy = vertex
                if torch.isnan(xy).any():
                    continue  # 如果检测到nan，就跳过
                x = int(xy[0])
                y = int(xy[1])   # 必须将x\y转化为整数（直接取出来是tensor）

                 # 截取卷积对象  Region of Interest, RoI
                ROI = padded_feature[:, :, x-2:x+3, y-2:y+3]   # 注意！切片操作不包括结尾   ROI的形状为 1， 64， 5， 5 ，很标准的张量形式
                if ROI.size(2) == 0 or ROI.size(3) == 0:
                    continue

                 # 应用3x3卷积
                import torch.nn as nn
                from lib.networks.PDC.pdc import Center_PDC, Zig_Zag_PDC, Step_PDC, PDC_Inception
                Inception = PDC_Inception(channels, channels).cuda()
                ROI_feature = Inception(ROI)

                # 将卷积结果放入gcn_feature中
                ROI_feature = ROI_feature.view(1,channels)
                gcn_feature[poly_idx, vertex_idx, :] = ROI_feature

                # (gcn_feature.size)  # 9, 64, 40   9个图形，每个图形上有40个点，每个点有64个特征值

    gcn_feature = gcn_feature.view([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    return gcn_feature


def get_adj_mat(n_adj, n_nodes, device):
    a = np.zeros([n_nodes, n_nodes])

    for i in range(n_nodes):
        for j in range(-n_adj // 2, n_adj // 2 + 1):
            if j != 0:
                a[i][(i + j) % n_nodes] = 1
                a[(i + j) % n_nodes][i] = 1

    a = torch.Tensor(a.astype(np.float32))
    return a.to(device)


def get_adj_ind(n_adj, n_nodes, device):
    ind = torch.LongTensor([i for i in range(-n_adj // 2, n_adj // 2 + 1) if i != 0])
    ind = (torch.arange(n_nodes)[:, None] + ind[None]) % n_nodes
    return ind.to(device)


def get_pconv_ind(n_adj, n_nodes, device):
    n_outer_nodes = snake_config.poly_num
    ind = torch.LongTensor([i for i in range(-n_adj // 2, n_adj // 2 + 1)])
    outer_ind = (torch.arange(n_outer_nodes)[:, None] + ind[None]) % n_outer_nodes
    inner_ind = outer_ind + n_outer_nodes
    ind = torch.cat([outer_ind, inner_ind], dim=1)
    return ind

# 将图像中的多边形顶点坐标转换为规范化（canonical）坐标系
def img_poly_to_can_poly(img_poly):
    if len(img_poly) == 0:
        return torch.zeros_like(img_poly)
    x_min = torch.min(img_poly[..., 0], dim=-1)[0]
    y_min = torch.min(img_poly[..., 1], dim=-1)[0]
    can_poly = torch.stack([
        img_poly[..., 0] - x_min[..., None],
        img_poly[..., 1] - y_min[..., None],
    ], dim=-1)
    # x_max = torch.max(img_poly[..., 0], dim=-1)[0]
    # y_max = torch.max(img_poly[..., 1], dim=-1)[0]
    # h, w = y_max - y_min + 1, x_max - x_min + 1
    # long_side = torch.max(h, w)
    # can_poly = can_poly / long_side[..., None, None]
    return can_poly


def _get_poly_resample_mode():
    return str(getattr(snake_config, 'poly_resample_mode', 'uniform')).strip().lower()


def _get_poly_resample_curvature_alpha():
    return float(getattr(snake_config, 'poly_resample_curvature_alpha', 1.5))


def _compute_resample_edge_scores(poly: torch.Tensor) -> torch.Tensor:
    next_poly = torch.roll(poly, -1, 2)
    edge_len = (next_poly - poly).pow(2).sum(3).sqrt()

    mode = _get_poly_resample_mode()
    alpha = _get_poly_resample_curvature_alpha()
    if mode == 'uniform' or alpha <= 0 or poly.size(2) < 3:
        return edge_len

    prev_poly = torch.roll(poly, 1, 2)
    prev_vec = poly - prev_poly
    next_vec = next_poly - poly
    prev_norm = prev_vec.pow(2).sum(3).sqrt()
    next_norm = next_vec.pow(2).sum(3).sqrt()
    denom = (prev_norm * next_norm).clamp(min=1e-6)
    cos_theta = (prev_vec * next_vec).sum(3) / denom
    cos_theta = cos_theta.clamp(min=-1.0, max=1.0)
    turn = torch.acos(cos_theta) / torch.pi
    edge_turn = 0.5 * (turn + torch.roll(turn, -1, 2))

    if mode == 'curvature':
        return edge_len * (1.0 + alpha * edge_turn)
    return edge_len


def uniform_upsample(poly, p_num):  # 初始4边形上采样，不用深究原理过程（有点抽象）
    # 1. assign point number for each edge
    # 2. calculate the coefficient for linear interpolation
    next_poly = torch.roll(poly, -1, 2)
    edge_score = _compute_resample_edge_scores(poly)
    edge_num = torch.round(edge_score * p_num / torch.sum(edge_score, dim=2)[..., None].clamp(min=1e-6)).long()
    edge_num = torch.clamp(edge_num, min=1)
    edge_num_sum = torch.sum(edge_num, dim=2)
    edge_idx_sort = torch.argsort(edge_num, dim=2, descending=True)

    use_cuda_ext = (os.environ.get('SNAKE_USE_CUDA_EXT', '0') == '1') and poly.is_cuda
    if use_cuda_ext:
        try:
            # Fast CUDA path (opt-in)
            extreme_utils.calculate_edge_num(edge_num, edge_num_sum, edge_idx_sort, p_num)
            edge_num_sum = torch.sum(edge_num, dim=2)
            assert torch.all(edge_num_sum == p_num)

            edge_start_idx = torch.cumsum(edge_num, dim=2) - edge_num
            weight, ind = extreme_utils.calculate_wnp(edge_num, edge_start_idx, p_num)
            poly1 = poly.gather(2, ind[..., 0:1].expand(ind.size(0), ind.size(1), ind.size(2), 2))
            poly2 = poly.gather(2, ind[..., 1:2].expand(ind.size(0), ind.size(1), ind.size(2), 2))
            poly = poly1 * (1 - weight) + poly2 * weight
            return poly
        except RuntimeError:
            # Fallback to PyTorch path silently
            pass

    # PyTorch fallback (default): robust CPU implementation
    B, N, V, C = poly.shape  # V can be 4 (quadrangle) or 8 (octagon)
    device = poly.device
    dtype = poly.dtype

    # Force PyTorch fallback for non-quadrangle to avoid CUDA implementation potential hardcoding
    if V == 4 and extreme_utils is not None:
        try:
            edge_num = torch.full([B, N, V], p_num // V, device=device, dtype=torch.int32)
            edge_start_idx = torch.cumsum(edge_num, dim=2) - edge_num
            weight, ind = extreme_utils.calculate_wnp(edge_num, edge_start_idx, p_num)
            poly1 = poly.gather(2, ind[..., 0:1].expand(ind.size(0), ind.size(1), ind.size(2), 2))
            poly2 = poly.gather(2, ind[..., 1:2].expand(ind.size(0), ind.size(1), ind.size(2), 2))
            poly = poly1 * (1 - weight) + poly2 * weight
            return poly
        except RuntimeError:
            pass

    out_list = []
    for b in range(B):
        polys_b = []
        for n in range(N):
            pts = poly[b, n]              # [V, 2]
            nxt = torch.roll(pts, -1, dims=0)
            elen = torch.sqrt(torch.sum((nxt - pts) ** 2, dim=1) + 1e-12)  # [V]
            turn_score = _compute_resample_edge_scores(pts.view(1, 1, V, C))[0, 0]
            total = torch.clamp(turn_score.sum(), min=1e-6)
            frac = (turn_score / total) * float(p_num)
            en = torch.clamp(torch.round(frac), min=1).to(torch.int64)      # [V]
            diff = int(p_num - int(en.sum().item()))
            if diff != 0:
                order = torch.argsort(turn_score, descending=True)
                idx = 0
                # Distribute residual to the longest edges first
                while diff != 0:
                    j = int(order[idx % V])
                    if diff > 0:
                        en[j] += 1
                        diff -= 1
                    else:  # diff < 0
                        if en[j] > 1:
                            en[j] -= 1
                            diff += 1
                    idx += 1

            samples = []
            for j in range(V):
                k = en[j].item()
                start = pts[j]
                end = nxt[j]
                if k <= 0:
                    continue
                if k == 1:
                    samples.append(start.unsqueeze(0))
                else:
                    # Sample exactly k points on this edge, including the start point
                    # and excluding the endpoint to avoid duplicates across edges.
                    t = torch.linspace(0.0, 1.0, steps=int(k) + 1, device=device, dtype=dtype)[:-1]
                    seg = start[None, :] * (1 - t[:, None]) + end[None, :] * t[:, None]
                    samples.append(seg)
            new_poly = torch.cat(samples, dim=0) if len(samples) else pts[:1]
            # Pad or trim to exact p_num
            if new_poly.size(0) < p_num:
                pad = p_num - new_poly.size(0)
                new_poly = torch.cat([new_poly, new_poly[-1:].repeat(pad, 1)], dim=0)
            elif new_poly.size(0) > p_num:
                new_poly = new_poly[:p_num]
            polys_b.append(new_poly)
        if len(polys_b) == 0:
            out_list.append(torch.zeros((0, p_num, 2), device=device, dtype=dtype))
        else:
            out_list.append(torch.stack(polys_b, dim=0))
    return torch.stack(out_list, dim=0)

# 对多边形顶点进行缩放操作，不重要
def zoom_poly(poly, scale):
    mean = (poly.min(dim=1, keepdim=True)[0] + poly.max(dim=1, keepdim=True)[0]) * 0.5
    poly = poly - mean
    poly = poly * scale + mean
    return poly
    
