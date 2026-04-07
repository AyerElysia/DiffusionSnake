import os
from lib.utils.snake import snake_voc_utils, snake_config, visualize_utils
import cv2
import numpy as np
import math
from lib.utils import data_utils
import torch.utils.data as data
import torch
from pycocotools.coco import COCO
from lib.config import cfg
from lib.utils.getedge import binary_mask_to_polygon
import sys
import glob

class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        self.coco = None
        self.anns = np.array([], dtype=np.int64)
        self.json_category_id_to_contiguous_id = {}
        try:
            if ann_file is not None and str(ann_file).strip() and os.path.exists(ann_file):
                self.coco = COCO(ann_file)
                self.anns = np.array(sorted(self.coco.getImgIds()))
                self.anns = self.anns[:100] if split == 'mini' else self.anns
                self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}
        except Exception:
            self.coco = None
            self.anns = np.array([], dtype=np.int64)
            self.json_category_id_to_contiguous_id = {}

        # ===============================================================================================
        ### ours
        # ===============================================================================================
        self.train_images_path=[]
        self.train_masks_path=[]
        list_file = os.path.join(cfg.train.data_path, 'train_list.txt')
        with open(list_file, 'r', encoding='utf-8') as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]
        for name in lines:
            img_png = os.path.join(cfg.train.data_path, name)
            if not os.path.exists(img_png):
                raise FileNotFoundError(f"Image not found: {img_png}")
            
            self.train_images_path.append(img_png)
            base_name = os.path.basename(name).split('_')[0]
            mask_name = os.path.join(cfg.train.data_path, f"{base_name}_mask")
            self.train_masks_path.append(mask_name)  # root only

        print('======================')
        print('训练数据路径：', cfg.train.data_path)
        print('样本数（来自 train_list.txt）:', len(self.train_images_path))
        print('列表文件：', list_file)
        print('======================')
        # for i in range(1, cfg.train.data_num):
        #     self.train_images_path.append(cfg.train.data_path + '{}_image.jpg'.format(i))
        #     self.train_masks_path.append(cfg.train.data_path + '{}_mask'.format(i))

        try:
            self.per_contour = bool(getattr(cfg.train, 'per_contour'))
        except Exception:
            self.per_contour = bool(getattr(cfg, 'per_contour', False))
        self.samples = []
        if self.per_contour:
            for idx in range(len(self.train_images_path)):
                mroot = self.train_masks_path[idx]
                mpaths = glob.glob(mroot + "*")
                for mpath in mpaths:
                    polys = binary_mask_to_polygon(mpath)
                    if not polys:
                        continue
                    cls_id = int(os.path.basename(mpath).split('.')[0].split('_')[-1])
                    for p_idx in range(len(polys)):
                        self.samples.append((idx, mpath, cls_id, p_idx))


    def process_info(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)
        path = os.path.join(self.data_root, self.coco.loadImgs(int(img_id))[0]['file_name'])
        return anno, path, img_id

    def read_original_data(self, anno, path):
        img = cv2.imread(path)
        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in anno]
        cls_ids = [self.json_category_id_to_contiguous_id[obj['category_id']] for obj in anno]
        return img, instance_polys, cls_ids

    def transform_original_data(self, instance_polys, flipped, width, trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            polys = [poly.reshape(-1, 2) for poly in instance]

            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            polys = snake_voc_utils.transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def get_valid_polys(self, instance_polys, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            for poly in instance:
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
            polys = snake_voc_utils.filter_tiny_polys(instance)
            polys = snake_voc_utils.get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            instance_polys_.append(polys)
        return instance_polys_

    def get_extreme_points(self, instance_polys):
        extreme_points = []
        for instance in instance_polys:
            points = [snake_voc_utils.get_extreme_points(poly) for poly in instance]
            extreme_points.append(points)
        return extreme_points

    def prepare_detection(self, box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        ct = np.round(ct).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

        x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        decode_box = [x_min, y_min, x_max, y_max]

        return decode_box

    def prepare_detection_(self, box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        box_ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)

        x_min_int, y_min_int = int(x_min), int(y_min)
        h_int, w_int = math.ceil(y_max - y_min_int) + 1, math.ceil(x_max - x_min_int) + 1
        max_h, max_w = ct_hm.shape[0], ct_hm.shape[1]
        h_int, w_int = min(y_min_int + h_int, max_h) - y_min_int, min(x_min_int + w_int, max_w) - x_min_int

        mask_poly = poly - np.array([x_min_int, y_min_int])
        mask_ct = box_ct - np.array([x_min_int, y_min_int])
        ct, off, xy = snake_voc_utils.prepare_ct_off_mask(mask_poly, mask_ct, h_int, w_int)

        xy += np.array([x_min_int, y_min_int])
        ct += np.array([x_min_int, y_min_int])

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

    def prepare_init(self, box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, h, w):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        img_init_poly = snake_voc_utils.get_init(box)
        img_init_poly = snake_voc_utils.uniformsample(img_init_poly, snake_config.init_poly_num)
        can_init_poly = snake_voc_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)
        img_gt_poly = extreme_point
        can_gt_poly = snake_voc_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        i_it_4pys.append(img_init_poly)
        c_it_4pys.append(can_init_poly)
        i_gt_4pys.append(img_gt_poly)
        c_gt_4pys.append(can_gt_poly)

    def prepare_evolution(self, poly, extreme_point, img_init_4polys, img_init_polys, can_init_polys, img_gt_polys, can_gt_polys):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])
        bbox = [x_min, y_min, x_max, y_max]
        base_init_poly = snake_voc_utils.get_evolution_init(extreme_point, bbox)
        img_init_poly = snake_voc_utils.uniformsample(base_init_poly, snake_config.poly_num)
        can_init_poly = snake_voc_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)

        img_gt_poly = snake_voc_utils.uniformsample(poly, len(poly) * snake_config.gt_poly_num)
        tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]
        can_gt_poly = snake_voc_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)
        
    def prepare_merge(self, is_id, cls_id, cp_id, cp_cls):
        cp_id.append(is_id)
        cp_cls.append(cls_id)

    def __getitem__(self, index):
        ours=True  # FLAG

        if ours:
            if getattr(self, 'per_contour', False):
                img_idx, sel_mask_path, sel_cls_id, sel_poly_idx = self.samples[index]
                img_path = self.train_images_path[img_idx]
                img = cv2.imread(img_path)
                if img is None:
                    alt_path = None
                    if img_path.endswith('.jpg'):
                        alt_path = img_path[:-4] + '.png'
                    elif img_path.endswith('.png'):
                        alt_path = img_path[:-4] + '.jpg'
                    if alt_path and os.path.exists(alt_path):
                        img = cv2.imread(alt_path)
                        if img is not None:
                            img_path = alt_path
                if img is None:
                    raise FileNotFoundError('Image not found or cannot be read: {}'.format(img_path))
                height, width = img.shape[0], img.shape[1]
                instance_polys=[]
                cls_ids=[]
                cla_mask_num = []
                polys = binary_mask_to_polygon(sel_mask_path)
                if len(polys) == 0:
                    raise ValueError('No polygons found for mask: {}'.format(sel_mask_path))
                sel_poly = polys[sel_poly_idx]
                instance_polys.extend([sel_poly])
                cls_ids.append(sel_cls_id)
                cla_mask_num.append(1)
            else:
                img_path = self.train_images_path[index]
                mask_path_root = self.train_masks_path[index]
                mask_paths = glob.glob(mask_path_root+"*")
                img = cv2.imread(img_path)
                if img is None:
                    alt_path = None
                    if img_path.endswith('.jpg'):
                        alt_path = img_path[:-4] + '.png'
                    elif img_path.endswith('.png'):
                        alt_path = img_path[:-4] + '.jpg'
                    if alt_path and os.path.exists(alt_path):
                        img = cv2.imread(alt_path)
                        if img is not None:
                            img_path = alt_path
                if img is None:
                    raise FileNotFoundError('Image not found or cannot be read: {}'.format(img_path))
                height, width = img.shape[0], img.shape[1]
                instance_polys=[]
                cls_ids=[]
                cla_mask_num = []
                for mask_path in mask_paths:
                    polys = binary_mask_to_polygon(mask_path)
                    if not polys:
                        print(f"Warning: No polygons found for mask at {mask_path}. Skipping this mask.")
                    instance_polys.extend(polys)
                    cls_ids.append(int(os.path.basename(mask_path).split('.')[0].split('_')[-1]))
                    cla_mask_num.append(len(polys))

        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            snake_voc_utils.augment(
                img, self.split,
                snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
                snake_config.mean, snake_config.std, instance_polys
            )
        instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw)
        instance_polys = self.get_valid_polys(instance_polys, inp_out_hw)
        extreme_points = self.get_extreme_points(instance_polys)


        #ours


        # detection (Snake) + targets for YOLO loss
        output_h, output_w = inp_out_hw[2:]
        ct_hm = np.zeros([cfg.heads.ct_hm, output_h, output_w], dtype=np.float32)
        wh = []
        ct_cls = []
        ct_ind = []

        # YOLO targets (normalized xywh on input size)
        yolo_xywh = []
        yolo_cls = []

        # init
        i_it_4pys = []
        c_it_4pys = []
        i_gt_4pys = []
        c_gt_4pys = []

        # evolution
        i_it_pys = []
        c_it_pys = []
        i_gt_pys = []
        c_gt_pys = []
        k=0
        
            
        if ours:
            anno=cls_ids
        for i in range(len(anno)):
            cls_id = cls_ids[i]

            for m in range(cla_mask_num[i]):

                instance_poly = instance_polys[k]
                instance_points = extreme_points[k]
                k=k+1
                for j in range(len(instance_poly)):
                    poly = instance_poly[j]
                    extreme_point = instance_points[j]

                    x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                    x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                    bbox = [x_min, y_min, x_max, y_max]
                    h, w = y_max - y_min + 1, x_max - x_min + 1
                    if h <= 1 or w <= 1:
                        continue


                    self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind)
                    # Build YOLO targets: convert bbox (in output coords) to input coords, then to normalized xywh
                    inp_h, inp_w = inp_out_hw[0], inp_out_hw[1]
                    scale_x = float(inp_w) / float(output_w)
                    scale_y = float(inp_h) / float(output_h)
                    # center-x, center-y, width, height (in input pixel coords)
                    cx = (x_min + x_max) / 2.0 * scale_x
                    cy = (y_min + y_max) / 2.0 * scale_y
                    bw = (x_max - x_min) * scale_x
                    bh = (y_max - y_min) * scale_y
                    # normalize to [0,1] by input size
                    yolo_xywh.append([
                        cx / max(inp_w, 1e-6),
                        cy / max(inp_h, 1e-6),
                        bw / max(inp_w, 1e-6),
                        bh / max(inp_h, 1e-6),
                    ])
                    # 注意：YOLO 的类别标签必须为 0 基（范围 [0, nc-1]）。
                    # 我们只对 YOLO 的标签做减一（cls_id-1），保持 Snake 分支沿用原始 cls_id，避免与
                    # `ct_hm[cls_id]` 等索引产生冲突或越界。
                    yolo_cls.append(float(cls_id - 1))
                    self.prepare_init(bbox, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, output_h, output_w)
                    self.prepare_evolution(poly, extreme_point, i_it_4pys[-1], i_it_pys, c_it_pys, i_gt_pys, c_gt_pys)

        # Use augmented (affine+flip) image for visualization to stay consistent with flipped polys
        ret = {'inp': inp, 'orig_img': orig_img}
        detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
        evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
        
        ret.update(detection)
        ret.update(init)
        ret.update(evolution)
        ret.update({'img_path': img_path})

        # Attach YOLO target tensors expected by v8DetectionLoss
        if len(yolo_xywh) > 0:
            ret['bboxes'] = torch.tensor(yolo_xywh, dtype=torch.float32)
            ret['cls'] = torch.tensor(yolo_cls, dtype=torch.float32).unsqueeze(1)  # (N,1)
            ret['batch_idx'] = torch.zeros((len(yolo_xywh), 1), dtype=torch.float32)  # single-image per sample
        else:
            # empty tensors to keep shapes consistent
            ret['bboxes'] = torch.zeros((0, 4), dtype=torch.float32)    # xywh
            ret['cls'] = torch.zeros((0, 1), dtype=torch.float32)
            ret['batch_idx'] = torch.zeros((0, 1), dtype=torch.float32)

        if cfg.vis_zrc:
            # 中间结果可视化
            visualize_utils.visualize_snake_detection(orig_img, ret)
            # 进化后结果可视化
            visualize_utils.visualize_snake_evolution(orig_img, ret)

        ct_num = len(ct_ind)
        meta = {'center': center, 'scale': scale, 'ct_num': ct_num}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        if getattr(self, 'per_contour', False):
            return len(self.samples)
        return len(self.train_images_path)
