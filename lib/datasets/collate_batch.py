from torch.utils.data.dataloader import default_collate
import torch
import numpy as np
from lib.config import cfg

# Modified by Zhang Ruicheng on 2024.04.21

def snake_collator(batch):
    ret = {'inp': default_collate([b['inp'] for b in batch])}
    # Keep original images as a Python list to handle variable sizes safely (used only for visualization)
    ret.update({'orig_img': [b['orig_img'] for b in batch]})
    ret.update({'img_path': default_collate([b['img_path'] for b in batch])})
    if any('locate_feat' in b for b in batch):
        missing = [b.get('img_path', '<unknown>') for b in batch if 'locate_feat' not in b]
        if missing:
            raise KeyError(f"locate_feat is missing for batch samples: {missing}")
        # Variable spatial sizes across slices (different CT geometries).
        # Keep as a Python list so batch_size>1 works without breaking grid_sample math.
        # The model loops per-sample in apply_locate_feature_replacement.
        ret.update({
            'locate_feat': [
                torch.as_tensor(b['locate_feat'], dtype=torch.float16) for b in batch
            ],
            'locate_feat_grid_hw': default_collate([b['locate_feat_grid_hw'] for b in batch]),
            'locate_feat_orig_hw': default_collate([b['locate_feat_orig_hw'] for b in batch]),
            'locate_feat_resized_hw': default_collate([b['locate_feat_resized_hw'] for b in batch]),
            'locate_feat_padded_hw': default_collate([b['locate_feat_padded_hw'] for b in batch]),
            'locate_feat_pad': default_collate([b['locate_feat_pad'] for b in batch]),
            'locate_feat_scale': default_collate([b['locate_feat_scale'] for b in batch]),
            'locate_feat_patch_size': default_collate([b['locate_feat_patch_size'] for b in batch]),
            'locate_feat_path': [b['locate_feat_path'] for b in batch],
        })

    meta = default_collate([b['meta'] for b in batch])

    # clamp max contour count to stabilize tensor shapes
    max_ct_cfg = getattr(getattr(cfg, 'train', cfg), 'max_ct_num', None)
    max_ct_num = int(max_ct_cfg) if max_ct_cfg is not None else None

    # truncate per-sample lists according to max_ct_num
    truncated = []
    ct_nums = []
    for b in batch:
        n = len(b['wh'])
        if max_ct_num is not None:
            n = min(n, max_ct_num)
        ct_nums.append(n)
        truncated.append({
            'wh': b['wh'][:n],
            'ct_cls': b['ct_cls'][:n],
            'ct_ind': b['ct_ind'][:n],
            'i_it_4py': b['i_it_4py'][:n],
            'c_it_4py': b['c_it_4py'][:n],
            'i_gt_4py': b['i_gt_4py'][:n],
            'c_gt_4py': b['c_gt_4py'][:n],
            'i_it_py': b['i_it_py'][:n],
            'c_it_py': b['c_it_py'][:n],
            'i_gt_py': b['i_gt_py'][:n],
            'c_gt_py': b['c_gt_py'][:n]
        })
        if 'eagle_i_gt_4py' in b and 'eagle_4py_mask' in b:
            truncated[-1]['eagle_i_gt_4py'] = b['eagle_i_gt_4py'][:n]
            truncated[-1]['eagle_4py_mask'] = b['eagle_4py_mask'][:n]

    meta['ct_num'] = torch.tensor(ct_nums, dtype=torch.int64)
    ret.update({'meta': meta})

    if 'vis_GT' in meta:
        return ret

    # detection
    ct_hm = default_collate([b['ct_hm'] for b in batch])

    batch_size = len(batch)
    ct_num = torch.max(meta['ct_num'])
    wh = torch.zeros([batch_size, ct_num, 2], dtype=torch.float)
    ct_cls = torch.zeros([batch_size, ct_num], dtype=torch.int64)
    ct_ind = torch.zeros([batch_size, ct_num], dtype=torch.int64)
    ct_01 = torch.zeros([batch_size, ct_num], dtype=torch.bool)
    for i in range(batch_size):
        ct_01[i, :meta['ct_num'][i]] = True

    if ct_num != 0:
        wh[ct_01] = torch.Tensor(sum([b['wh'] for b in truncated], []))
        ct_cls[ct_01] = torch.LongTensor(sum([b['ct_cls'] for b in truncated], []))
        ct_ind[ct_01] = torch.LongTensor(sum([b['ct_ind'] for b in truncated], []))

    detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'ct_01': ct_01.float()}
    ret.update(detection)

    from lib.utils.snake import snake_config

    # init
    i_it_4pys = torch.zeros([batch_size, ct_num, snake_config.init_poly_num, 2], dtype=torch.float)
    c_it_4pys = torch.zeros([batch_size, ct_num, snake_config.init_poly_num, 2], dtype=torch.float)
    i_gt_4pys = torch.zeros([batch_size, ct_num, 4, 2], dtype=torch.float)
    c_gt_4pys = torch.zeros([batch_size, ct_num, 4, 2], dtype=torch.float)
    if ct_num != 0:
        i_it_4pys[ct_01] = torch.Tensor(sum([b['i_it_4py'] for b in truncated], []))
        c_it_4pys[ct_01] = torch.Tensor(sum([b['c_it_4py'] for b in truncated], []))
        i_gt_4pys[ct_01] = torch.Tensor(sum([b['i_gt_4py'] for b in truncated], []))
        c_gt_4pys[ct_01] = torch.Tensor(sum([b['c_gt_4py'] for b in truncated], []))
    init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
    ret.update(init)

    has_eagle = any('eagle_i_gt_4py' in b for b in truncated)
    if has_eagle:
        eagle_i_gt_4pys = torch.zeros([batch_size, ct_num, 4, 2], dtype=torch.float)
        eagle_4py_mask = torch.zeros([batch_size, ct_num], dtype=torch.float)
        for bi, b in enumerate(truncated):
            if 'eagle_i_gt_4py' not in b:
                continue
            n = min(len(b['eagle_i_gt_4py']), int(meta['ct_num'][bi]))
            if n <= 0:
                continue
            eagle_i_gt_4pys[bi, :n] = torch.as_tensor(b['eagle_i_gt_4py'][:n], dtype=torch.float)
            eagle_4py_mask[bi, :n] = torch.as_tensor(b['eagle_4py_mask'][:n], dtype=torch.float)
        ret.update({'eagle_i_gt_4py': eagle_i_gt_4pys, 'eagle_4py_mask': eagle_4py_mask})

    # V3.10: evolution with adaptive points + mask
    if snake_config.adaptive_points_enabled:
        # Use fixed output slots so 64-point contours can occupy a contiguous prefix and pad the tail.
        max_points = int(max(
            getattr(snake_config, 'poly_num', 128),
            getattr(snake_config, 'max_points', 128),
            getattr(snake_config, 'adaptive_large_points', 128),
        ))
        small_points = int(getattr(snake_config, 'adaptive_small_points', 64))

        i_it_pys = torch.zeros([batch_size, ct_num, max_points, 2], dtype=torch.float)
        c_it_pys = torch.zeros([batch_size, ct_num, max_points, 2], dtype=torch.float)
        i_gt_pys = torch.zeros([batch_size, ct_num, max_points, 2], dtype=torch.float)
        c_gt_pys = torch.zeros([batch_size, ct_num, max_points, 2], dtype=torch.float)
        point_masks = torch.zeros([batch_size, ct_num, max_points], dtype=torch.float)

        if ct_num != 0:
            for bi, b in enumerate(truncated):
                num_ct = len(b['i_it_py'])
                for ci in range(num_ct):
                    n_pts = int(b['i_it_py'][ci].shape[0])
                    i_src = torch.as_tensor(b['i_it_py'][ci], dtype=torch.float)
                    c_src = torch.as_tensor(b['c_it_py'][ci], dtype=torch.float)
                    ig_src = torch.as_tensor(b['i_gt_py'][ci], dtype=torch.float)
                    cg_src = torch.as_tensor(b['c_gt_py'][ci], dtype=torch.float)

                    # V3.10: keep valid 64-point contours as a contiguous prefix.
                    # This preserves local point order and avoids artificial gaps.
                    if n_pts == small_points and max_points >= (small_points * 2):
                        i_it_pys[bi, ci, :small_points] = i_src[:small_points]
                        c_it_pys[bi, ci, :small_points] = c_src[:small_points]
                        i_gt_pys[bi, ci, :small_points] = ig_src[:small_points]
                        c_gt_pys[bi, ci, :small_points] = cg_src[:small_points]
                        point_masks[bi, ci, :small_points] = 1.0
                    else:
                        valid_n = min(n_pts, max_points)
                        i_it_pys[bi, ci, :valid_n] = i_src[:valid_n]
                        c_it_pys[bi, ci, :valid_n] = c_src[:valid_n]
                        i_gt_pys[bi, ci, :valid_n] = ig_src[:valid_n]
                        c_gt_pys[bi, ci, :valid_n] = cg_src[:valid_n]
                        point_masks[bi, ci, :valid_n] = 1.0

        evolution = {
            'i_it_py': i_it_pys,
            'c_it_py': c_it_pys,
            'i_gt_py': i_gt_pys,
            'c_gt_py': c_gt_pys,
            'point_mask': point_masks  # V3.10: add point mask
        }
    else:
        # Original fixed-size logic
        i_it_pys = torch.zeros([batch_size, ct_num, snake_config.poly_num, 2], dtype=torch.float)
        c_it_pys = torch.zeros([batch_size, ct_num, snake_config.poly_num, 2], dtype=torch.float)
        i_gt_pys = torch.zeros([batch_size, ct_num, snake_config.gt_poly_num, 2], dtype=torch.float)
        c_gt_pys = torch.zeros([batch_size, ct_num, snake_config.gt_poly_num, 2], dtype=torch.float)
        if ct_num != 0:
            i_it_pys[ct_01] = torch.Tensor(sum([b['i_it_py'] for b in truncated], []))
            c_it_pys[ct_01] = torch.Tensor(sum([b['c_it_py'] for b in truncated], []))
            i_gt_pys[ct_01] = torch.Tensor(sum([b['i_gt_py'] for b in truncated], []))
            c_gt_pys[ct_01] = torch.Tensor(sum([b['c_gt_py'] for b in truncated], []))
        evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}

    ret.update(evolution)

    # Aggregate YOLO targets if present in samples
    has_yolo = any(('bboxes' in b and 'cls' in b and 'batch_idx' in b) for b in batch)
    if has_yolo:
        all_bboxes = []
        all_cls = []
        all_batch = []
        for i, b in enumerate(batch):
            if 'bboxes' in b and b['bboxes'] is not None and b['bboxes'].numel() > 0:
                n = b['bboxes'].shape[0]
                all_bboxes.append(b['bboxes'])
                all_cls.append(b['cls'])
                # override batch_idx with actual image index i
                all_batch.append(torch.full((n, 1), float(i)))
        if len(all_bboxes) > 0:
            ret['bboxes'] = torch.cat(all_bboxes, dim=0)
            ret['cls'] = torch.cat(all_cls, dim=0)
            ret['batch_idx'] = torch.cat(all_batch, dim=0)
        else:
            ret['bboxes'] = torch.zeros((0, 4), dtype=torch.float32)
            ret['cls'] = torch.zeros((0, 1), dtype=torch.float32)
            ret['batch_idx'] = torch.zeros((0, 1), dtype=torch.float32)

    return ret


def dsnake_collator(batch):
    ret = snake_collator(batch)
    meta = ret['meta']

    # detection
    act_hm = default_collate([b['act_hm'] for b in batch])

    batch_size = len(batch)
    act_num = torch.max(meta['act_num'])
    awh = torch.zeros([batch_size, act_num, 2], dtype=torch.float)
    act_ind = torch.zeros([batch_size, act_num], dtype=torch.int64)
    act_01 = torch.zeros([batch_size, act_num], dtype=torch.uint8)
    for i in range(batch_size):
        act_01[i, :meta['act_num'][i]] = 1

    awh[act_01] = torch.Tensor(sum([b['awh'] for b in batch], []))
    act_ind[act_01] = torch.LongTensor(sum([b['act_ind'] for b in batch], []))

    adet = {'act_hm': act_hm, 'awh': awh, 'act_ind': act_ind, 'act_01': act_01.float()}
    ret.update(adet)

    return ret


# Modified by Zhang Ruicheng on 2024.04.21
'''
def rcnn_snake_collator(batch):
    ret = {'inp': default_collate([b['inp'] for b in batch])}

    meta = default_collate([b['meta'] for b in batch])
    ret.update({'meta': meta})

    if 'vis_GT' in meta:
        return ret

    # detection
    act_hm = default_collate([b['act_hm'] for b in batch])

    batch_size = len(batch)
    act_num = torch.max(meta['act_num'])
    awh = torch.zeros([batch_size, act_num, 2], dtype=torch.float)
    act_ind = torch.zeros([batch_size, act_num], dtype=torch.int64)
    act_01 = torch.zeros([batch_size, act_num], dtype=torch.uint8)
    for i in range(batch_size):
        act_01[i, :meta['act_num'][i]] = 1

    if act_num != 0:
        awh[act_01] = torch.Tensor(sum([b['awh'] for b in batch], []))
        act_ind[act_01] = torch.LongTensor(sum([b['act_ind'] for b in batch], []))

    detection = {'act_hm': act_hm, 'awh': awh, 'act_ind': act_ind, 'act_01': act_01.float()}
    ret.update(detection)

    from lib.utils.rcnn_snake import rcnn_snake_config as snake_config

    ct_num = torch.max(meta['ct_num'])
    ct_01 = torch.zeros([batch_size, ct_num], dtype=torch.uint8)
    for i in range(batch_size):
        ct_01[i, :meta['ct_num'][i]] = 1
    ret.update({'ct_01': ct_01.float()})

    # component detection
    cp_hm = default_collate(sum([[hm for hm in b['cp_hm']] for b in batch], []))

    cp_num_list = sum([[len(wh) for wh in b['cp_wh']] for b in batch], [])
    cp_num = max(cp_num_list)
    cp_wh = torch.zeros([len(cp_hm), cp_num, 2], dtype=torch.float)
    cp_ind = torch.zeros([len(cp_hm), cp_num], dtype=torch.int64)
    cp_01 = torch.zeros([len(cp_hm), cp_num], dtype=torch.uint8)
    for i in range(len(cp_hm)):
        cp_01[i, :cp_num_list[i]] = 1

    if cp_num != 0:
        cp_wh[cp_01] = torch.Tensor(sum(sum([b['cp_wh'] for b in batch], []), []))
        cp_ind[cp_01] = torch.LongTensor(sum(sum([b['cp_ind'] for b in batch], []), []))

    cp_hm_ = torch.zeros([batch_size, act_num, 1, snake_config.cp_h, snake_config.cp_w], dtype=torch.float)
    cp_wh_ = torch.zeros([batch_size, act_num, cp_num, 2], dtype=torch.float)
    cp_ind_ = torch.zeros([batch_size, act_num, cp_num], dtype=torch.int64)
    cp_01_ = torch.zeros([batch_size, act_num, cp_num], dtype=torch.uint8)

    cp_hm_[act_01] = cp_hm
    cp_wh_[act_01] = cp_wh
    cp_ind_[act_01] = cp_ind
    cp_01_[act_01] = cp_01

    cp_detection = {'cp_hm': cp_hm_, 'cp_wh': cp_wh_, 'cp_ind': cp_ind_, 'cp_01': cp_01_.float()}
    ret.update(cp_detection)

    # init
    i_it_4pys = torch.zeros([batch_size, ct_num, snake_config.init_poly_num, 2], dtype=torch.float)
    c_it_4pys = torch.zeros([batch_size, ct_num, snake_config.init_poly_num, 2], dtype=torch.float)
    i_gt_4pys = torch.zeros([batch_size, ct_num, 4, 2], dtype=torch.float)
    c_gt_4pys = torch.zeros([batch_size, ct_num, 4, 2], dtype=torch.float)
    if ct_num != 0:
        i_it_4pys[ct_01] = torch.Tensor(sum([b['i_it_4py'] for b in batch], []))
        c_it_4pys[ct_01] = torch.Tensor(sum([b['c_it_4py'] for b in batch], []))
        i_gt_4pys[ct_01] = torch.Tensor(sum([b['i_gt_4py'] for b in batch], []))
        c_gt_4pys[ct_01] = torch.Tensor(sum([b['c_gt_4py'] for b in batch], []))
    init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
    ret.update(init)

    # evolution
    i_it_pys = torch.zeros([batch_size, ct_num, snake_config.poly_num, 2], dtype=torch.float)
    c_it_pys = torch.zeros([batch_size, ct_num, snake_config.poly_num, 2], dtype=torch.float)
    i_gt_pys = torch.zeros([batch_size, ct_num, snake_config.gt_poly_num, 2], dtype=torch.float)
    c_gt_pys = torch.zeros([batch_size, ct_num, snake_config.gt_poly_num, 2], dtype=torch.float)
    if ct_num != 0:
        i_it_pys[ct_01] = torch.Tensor(sum([b['i_it_py'] for b in batch], []))
        c_it_pys[ct_01] = torch.Tensor(sum([b['c_it_py'] for b in batch], []))
        i_gt_pys[ct_01] = torch.Tensor(sum([b['i_gt_py'] for b in batch], []))
        c_gt_pys[ct_01] = torch.Tensor(sum([b['c_gt_py'] for b in batch], []))
    evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
    ret.update(evolution)

    return ret


def ext_snake_collator(batch):
    ret = {'inp': default_collate([b['inp'] for b in batch])}

    meta = default_collate([b['meta'] for b in batch])
    ret.update({'meta': meta})

    if 'vis_GT' in meta:
        return ret

    # detection
    ct_hm = default_collate([b['ct_hm'] for b in batch])

    batch_size = len(batch)
    ct_num = torch.max(meta['ct_num'])
    ext = torch.zeros([batch_size, ct_num, 8], dtype=torch.float)
    ct_cls = torch.zeros([batch_size, ct_num], dtype=torch.int64)
    ct_ind = torch.zeros([batch_size, ct_num], dtype=torch.int64)
    ct_01 = torch.zeros([batch_size, ct_num], dtype=torch.uint8)
    for i in range(batch_size):
        ct_01[i, :meta['ct_num'][i]] = 1

    ext[ct_01] = torch.Tensor(sum([b['ext'] for b in batch], []))
    ct_cls[ct_01] = torch.LongTensor(sum([b['ct_cls'] for b in batch], []))
    ct_ind[ct_01] = torch.LongTensor(sum([b['ct_ind'] for b in batch], []))

    detection = {'ct_hm': ct_hm, 'ext': ext, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'ct_01': ct_01.float()}
    ret.update(detection)

    from lib.utils.snake import snake_config

    # init
    i_it_4pys = torch.zeros([batch_size, ct_num, snake_config.init_poly_num, 2], dtype=torch.float)
    c_it_4pys = torch.zeros([batch_size, ct_num, snake_config.init_poly_num, 2], dtype=torch.float)
    i_gt_4pys = torch.zeros([batch_size, ct_num, 4, 2], dtype=torch.float)
    c_gt_4pys = torch.zeros([batch_size, ct_num, 4, 2], dtype=torch.float)
    i_it_4pys[ct_01] = torch.Tensor(sum([b['i_it_4py'] for b in batch], []))
    c_it_4pys[ct_01] = torch.Tensor(sum([b['c_it_4py'] for b in batch], []))
    i_gt_4pys[ct_01] = torch.Tensor(sum([b['i_gt_4py'] for b in batch], []))
    c_gt_4pys[ct_01] = torch.Tensor(sum([b['c_gt_4py'] for b in batch], []))
    init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
    ret.update(init)

    # evolution
    i_it_pys = torch.zeros([batch_size, ct_num, snake_config.poly_num, 2], dtype=torch.float)
    c_it_pys = torch.zeros([batch_size, ct_num, snake_config.poly_num, 2], dtype=torch.float)
    i_gt_pys = torch.zeros([batch_size, ct_num, snake_config.gt_poly_num, 2], dtype=torch.float)
    c_gt_pys = torch.zeros([batch_size, ct_num, snake_config.gt_poly_num, 2], dtype=torch.float)
    i_it_pys[ct_01] = torch.Tensor(sum([b['i_it_py'] for b in batch], []))
    c_it_pys[ct_01] = torch.Tensor(sum([b['c_it_py'] for b in batch], []))
    i_gt_pys[ct_01] = torch.Tensor(sum([b['i_gt_py'] for b in batch], []))
    c_gt_pys[ct_01] = torch.Tensor(sum([b['c_gt_py'] for b in batch], []))
    evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
    ret.update(evolution)

    return ret
'''

# Modified by Zhang Ruicheng on 2024.04.21
_collators = {
    'snake': snake_collator,
    'ct': snake_collator,
    #'rcnn_snake': rcnn_snake_collator,
    #'ct_rcnn': rcnn_snake_collator
}


def make_collator(cfg):
    if cfg.task in _collators:
        return _collators[cfg.task]
    else:
        return default_collate
