import torch.nn as nn
import torch
from lib.utils import data_utils
# from lib.csrc.extreme_utils import _ext as extreme_utils
from lib.utils.snake import snake_config

# "Modified by Zhang Ruicheng on 2024.04.21".
# 116 line

def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat


def topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decode_ct_hm(ct_hm, wh, reg=None, K=100):
    batch, cat, height, width = ct_hm.size()
    ct_hm = nms(ct_hm)

    scores, inds, clses, ys, xs = topk(ct_hm, K=K)  # 从张量中选取 K 个最大值及其对应的索引。
    # ct_hm: 一个四维张量，其形状通常是 (batch_size, 分数？, height, width)，表示每个类别在每个位置的预测分数。
    # K: 一个整数，表示要从每个类别热力图中选取的 top-k 个最高分数的检测框
    wh = transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)

    if reg is not None:
        reg = transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1)
        ys = ys.view(batch, K, 1)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    ct = torch.cat([xs, ys], dim=2)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detection = torch.cat([bboxes, scores, clses], dim=2)

    return ct, detection
'''
假设 batch 为 2，K 为 5，那么 ct 和 detection 可能看起来像这样：

ct: 
tensor([[[ x11, y11],
         [ x12, y12],
         [ x13, y13],
         [ x14, y14],
         [ x15, y15]],

        [[ x21, y21],
         [ x22, y22],
         [ x23, y23],
         [ x24, y24],
         [ x25, y25]]])
detection:
tensor([[[ x11-a1, y11-b1, x11+a1, y11+b1, score1],
         [ x12-a2, y12-b2, x12+a2, y12+b2, score2],
         [ x13-a3, y13-b3, x13+a3, y13+b3, score3],
         [ x14-a4, y14-b4, x14+a4, y14+b4, score4],
         [ x15-a5, y15-b5, x15+a5, y15+b5, score5]],

        [[ x21-a1, y21-b1, x21+a1, y21+b1, score1],
         [ x22-a2, y22-b2, x22+a2, y22+b2, score2],
         [ x23-a3, y23-b3, x23+a3, y23+b3, score3],
         [ x24-a4, y24-b4, x24+a4, y24+b4, score4],
         [ x25-a5, y25-b5, x25+a5, y25+b5, score5]]])'''

def gaussian_radius(height, width, min_overlap=0.7):
    height = torch.ceil(height)
    width = torch.ceil(width)

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1.pow(2) - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2.pow(2) - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height

    r31 = torch.min(r1, r2)
    det = b3.pow(2) - 4 * a3 * c3
    sq3 = torch.sqrt(torch.clamp(det, min=0))
    r32 = (b3 + sq3) / 2
    r3_01 = (det < 0).float()
    r3 = r3_01 * r31 + (1 - r3_01) * r32

    radius = torch.clamp(torch.min(torch.min(r1, r2), r3), min=0) / 3
    return torch.round(radius).long()

'''
def decode_ext_hm(ext_hm, bbox, vote, ct):
    h, w = ext_hm.size(2), ext_hm.size(3)
    ext_hm = nms(ext_hm)
    bbox = data_utils.clip_to_image(bbox, h, w)
    radius = gaussian_radius(bbox[..., 2] - bbox[..., 0], bbox[..., 3] - bbox[..., 1])
    bbox = torch.round(bbox).long()
    extreme_point = extreme_utils.collect_extreme_point(ext_hm, bbox, radius, vote.permute(0, 2, 3, 1), ct)
    return extreme_point
'''

def get_quadrangle(box):
    x_min, y_min, x_max, y_max = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    quadrangle = [
        (x_min + x_max) / 2., y_min,
        x_min, (y_min + y_max) / 2.,
        (x_min + x_max) / 2., y_max,
        x_max, (y_min + y_max) / 2.
    ]
    quadrangle = torch.stack(quadrangle, dim=2).view(x_min.size(0), x_min.size(1), 4, 2)
    return quadrangle


def get_box(box):
    x_min, y_min, x_max, y_max = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    box = [
        x_min, y_min,
        x_min, y_max,
        x_max, y_max,
        x_max, y_min
    ]
    box = torch.stack(box, dim=2).view(x_min.size(0), x_min.size(1), 4, 2)
    return box


def get_init(box):
    # x_min, y_min, x_max, y_max = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    # # 计算每个检测框的宽高比
    # scale = (y_max - y_min) / (x_max - x_min)  # [batch, num]
    # # 根据宽高比选择初始化形状
    # # scale < 2.5 使用菱形，否则使用矩形
    # use_quadrangle = scale < 2.5  # [batch, num] 布尔张量
    # # 生成菱形和矩形的坐标
    # quadrangle = get_quadrangle(box)  # [batch, num, 4, 2]
    # rectangle = get_box(box)          # [batch, num, 4, 2]
    # # 根据条件选择对应的形状
    # # 使用 torch.where 进行条件选择
    # result = torch.where(
    #     use_quadrangle.unsqueeze(-1).unsqueeze(-1),  # 扩展维度以匹配 [batch, num, 4, 2]
    #     quadrangle,
    #     rectangle
    # )
    # return result
    if snake_config.init == 'quadrangle':
        return get_quadrangle(box)
    elif snake_config.init == 'octagon':
        ex = get_quadrangle(box)
        return get_octagon(ex)
    else:
        return get_box(box)



def get_octagon(ex):
    # ex shape: [..., 4, 2] -> Expected order: Top, Left, Bottom, Right
    # Official DeepSnake: extend each extreme point to a segment of 1/4 edge length.
    
    # 1. Calculate bounding box of extreme points to get w, h
    x_min = ex[..., :, 0].min(dim=-1)[0]
    x_max = ex[..., :, 0].max(dim=-1)[0]
    y_min = ex[..., :, 1].min(dim=-1)[0]
    y_max = ex[..., :, 1].max(dim=-1)[0]
    w = x_max - x_min
    h = y_max - y_min
    
    # 2. Extract extreme points: 0:Top, 1:Left, 2:Bottom, 3:Right
    t = ex[..., 0, :]
    l = ex[..., 1, :]
    b = ex[..., 2, :]
    r = ex[..., 3, :]
    
    x_ext = w / 8.0
    y_ext = h / 8.0
    
    # 3. Construct 8 points in strict clockwise order:
    # Top-segment (p1, p2), Right-segment (p3, p4), Bottom-segment (p5, p6), Left-segment (p7, p8)
    p1 = torch.stack([t[..., 0] - x_ext, t[..., 1]], dim=-1) # Top-Left
    p2 = torch.stack([t[..., 0] + x_ext, t[..., 1]], dim=-1) # Top-Right
    p3 = torch.stack([r[..., 0], r[..., 1] - y_ext], dim=-1) # Right-Top
    p4 = torch.stack([r[..., 0], r[..., 1] + y_ext], dim=-1) # Right-Bottom
    p5 = torch.stack([b[..., 0] + x_ext, b[..., 1]], dim=-1) # Bottom-Right
    p6 = torch.stack([b[..., 0] - x_ext, b[..., 1]], dim=-1) # Bottom-Left
    p7 = torch.stack([l[..., 0], l[..., 1] + y_ext], dim=-1) # Left-Bottom
    p8 = torch.stack([l[..., 0], l[..., 1] - y_ext], dim=-1) # Left-Top
    
    # 4. Concatenate points along the vertex dimension
    octagon = torch.stack([p1, p2, p3, p4, p5, p6, p7, p8], dim=-2)
    return octagon



def decode_ext_hm(ct_hm, ext, ae=None, K=100):
    batch, cat, height, width = ct_hm.size()
    ct_hm = nms(ct_hm)

    scores, inds, clses, ys, xs = topk(ct_hm, K=K)
    ext = transpose_and_gather_feat(ext, inds)
    ext = ext.view(batch, K, 4, 2)

    xs = xs.view(batch, K, 1) + 0.5
    ys = ys.view(batch, K, 1) + 0.5
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    ct = torch.cat([xs, ys], dim=2)

    extreme_point = ct[:, :, None] + ext

    xy_min, ind = torch.min(extreme_point, dim=2)
    l_ind, t_ind = ind[..., 0:1], ind[..., 1:2]
    l_ind = l_ind[..., None].expand(l_ind.size(0), l_ind.size(1), 1, 2)
    t_ind = t_ind[..., None].expand(t_ind.size(0), t_ind.size(1), 1, 2)
    ll = extreme_point.gather(2, l_ind)
    tt = extreme_point.gather(2, t_ind)

    xy_max, ind = torch.max(extreme_point, dim=2)
    r_ind, b_ind = ind[..., 0:1], ind[..., 1:2]
    r_ind = r_ind[..., None].expand(r_ind.size(0), r_ind.size(1), 1, 2)
    b_ind = b_ind[..., None].expand(b_ind.size(0), b_ind.size(1), 1, 2)
    rr = extreme_point.gather(2, r_ind)
    bb = extreme_point.gather(2, b_ind)

    extreme_point = torch.cat([tt, ll, bb, rr], dim=2)
    bboxes = torch.cat([xy_min, xy_max], dim=2)
    detection = torch.cat([bboxes, scores, clses], dim=2)

    if ae is not None:
        ae = transpose_and_gather_feat(ae, inds)
        detection = torch.cat([detection, ae], dim=2)

    return ct, extreme_point, detection
