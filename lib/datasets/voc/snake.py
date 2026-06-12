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
import json
from pathlib import Path

class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split
        self.eagle_teacher = self._load_eagle_teacher()
        self.locate_feat_enabled = bool(
            getattr(cfg, 'locate_feat_inject', False)
            or getattr(cfg, 'locate_feat_replace', False)
            or getattr(cfg, 'use_locate_token_dit', False)
        )
        cache_root = getattr(cfg, 'locate_feat_cache_dir', '') or getattr(cfg, 'locate_feat_cache_root', 'data/locate_feat_cache')
        self.locate_feat_cache_root = str(cache_root or '')
        keys = getattr(cfg, 'locate_feat_keys', ['feat'])
        if isinstance(keys, str):
            keys = [x.strip() for x in keys.split(',') if x.strip()]
        self.locate_feat_keys = list(keys) if keys else ['feat']
        split_name = str(split).lower()
        self.locate_feat_split = 'test' if split_name in ('val', 'test') else 'train'

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

        # 按 split 选择数据根目录：训练/验证分离，避免推理误读训练集
        if str(split).lower() == 'val':
            data_path = str(getattr(cfg.test, 'img_path', '') or data_root)
            list_candidates = ['test_list.txt', 'val_list.txt', 'train_list.txt']
        elif str(split).lower() == 'mini':
            data_path = str(getattr(cfg.train, 'data_path', '') or data_root)
            list_candidates = ['mini_list.txt', 'train_list.txt', 'test_list.txt']
        else:
            data_path = str(getattr(cfg.train, 'data_path', '') or data_root)
            list_candidates = ['train_list.txt', 'test_list.txt']

        if not data_path:
            raise ValueError(f"Empty data path for split={split}")

        list_file = None
        for name in list_candidates:
            candidate = os.path.join(data_path, name)
            if os.path.exists(candidate):
                list_file = candidate
                break

        if list_file is not None:
            with open(list_file, 'r', encoding='utf-8') as f:
                lines = [x.strip() for x in f.readlines() if x.strip()]
        else:
            # 兼容没有 list 文件的目录：按 *_image.png 自动构建样本列表
            image_files = sorted(glob.glob(os.path.join(data_path, '*_image.png')))
            if not image_files:
                raise FileNotFoundError(
                    f"No list file and no *_image.png found under: {data_path}"
                )
            lines = [os.path.basename(p) for p in image_files]

        for name in lines:
            img_png = name if os.path.isabs(name) else os.path.join(data_path, name)
            if not os.path.exists(img_png):
                raise FileNotFoundError(f"Image not found: {img_png}")

            self.train_images_path.append(img_png)
            base_name = os.path.basename(name).split('_')[0]
            mask_name = os.path.join(data_path, f"{base_name}_mask")
            self.train_masks_path.append(mask_name)  # root only

        print('======================')
        print('数据 split：', split)
        print('数据路径：', data_path)
        print('样本数：', len(self.train_images_path))
        print('列表文件：', list_file if list_file is not None else '<auto-scan>')
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

        if self.locate_feat_enabled:
            print(
                f"[LocateFeat] enabled root={self.locate_feat_cache_root} "
                f"split={self.locate_feat_split} keys={self.locate_feat_keys}",
                flush=True,
            )

    @staticmethod
    def _path_keys(path):
        path = str(path)
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        return {path, os.path.abspath(path), base, stem}

    def _load_eagle_teacher(self):
        teacher_path = str(getattr(cfg, 'eagle_teacher_json', '') or '').strip()
        if not teacher_path:
            return {}
        if not os.path.exists(teacher_path):
            print(f"[EagleTeacher] teacher json not found: {teacher_path}")
            return {}
        with open(teacher_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)

        if isinstance(obj, dict) and isinstance(obj.get('samples'), list):
            records = obj['samples']
        elif isinstance(obj, list):
            records = obj
        elif isinstance(obj, dict):
            records = []
            for k, v in obj.items():
                if isinstance(v, dict):
                    rec = dict(v)
                    rec.setdefault('img_path', k)
                    records.append(rec)
                elif isinstance(v, list):
                    records.append({'img_path': k, 'instances': v})
        else:
            records = []

        teacher = {}
        for rec in records:
            if not isinstance(rec, dict):
                continue
            path = (
                rec.get('img_path') or rec.get('image_path') or rec.get('path')
                or rec.get('file_name') or rec.get('image') or rec.get('id')
            )
            if path is None:
                continue
            instances = rec.get('instances') or rec.get('objects') or rec.get('predictions') or rec.get('labels')
            if instances is None:
                instances = [rec]
            for key in self._path_keys(path):
                teacher[str(key)] = instances
        print(f"[EagleTeacher] loaded {len(records)} records from {teacher_path}")
        return teacher

    def _get_eagle_instances(self, img_path):
        if not self.eagle_teacher:
            return []
        for key in self._path_keys(img_path):
            if str(key) in self.eagle_teacher:
                return self.eagle_teacher[str(key)]
        return []

    def _locate_feat_path(self, img_path):
        stem = Path(str(img_path)).stem
        return os.path.join(self.locate_feat_cache_root, self.locate_feat_split, f'{stem}.npz')

    def _load_locate_feature(self, img_path):
        if not self.locate_feat_enabled:
            return {}
        feat_path = self._locate_feat_path(img_path)
        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                f"Locate feature cache missing for image={img_path}; expected npz={feat_path}"
            )
        try:
            with np.load(feat_path) as npz:
                def get_npz(key, default):
                    return npz[key] if key in npz.files else default

                missing_keys = [key for key in self.locate_feat_keys if key not in npz.files]
                if missing_keys:
                    raise KeyError(
                        f"missing feature keys={missing_keys}; available={list(npz.files)}"
                    )
                arrays = [np.asarray(npz[key], dtype=np.float16) for key in self.locate_feat_keys]
                shapes = {tuple(arr.shape[-2:]) for arr in arrays}
                if len(shapes) != 1:
                    raise ValueError(f"Locate feature spatial sizes differ: {[arr.shape for arr in arrays]}")
                feat = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
                meta = {
                    'locate_feat': feat,
                    'locate_feat_grid_hw': np.asarray(get_npz('grid_hw', feat.shape[-2:]), dtype=np.int32),
                    'locate_feat_orig_hw': np.asarray(get_npz('orig_hw', [0, 0]), dtype=np.int32),
                    'locate_feat_resized_hw': np.asarray(get_npz('resized_hw', [0, 0]), dtype=np.int32),
                    'locate_feat_padded_hw': np.asarray(get_npz('padded_hw', [0, 0]), dtype=np.int32),
                    'locate_feat_pad': np.asarray(get_npz('pad', [0, 0, 0, 0]), dtype=np.int32),
                    'locate_feat_scale': np.asarray(get_npz('scale', [1.0]), dtype=np.float32),
                    'locate_feat_patch_size': np.asarray(get_npz('patch_size', [14]), dtype=np.int32),
                    'locate_feat_path': feat_path,
                }
        except Exception as exc:
            raise RuntimeError(f"Failed to read Locate feature cache {feat_path}: {exc}") from exc
        return meta

    @staticmethod
    def _teacher_label(obj):
        for key in ('label_id', 'cls_id', 'class_id', 'category_id', 'label'):
            if key in obj:
                try:
                    return int(obj[key])
                except Exception:
                    return None
        return None

    @staticmethod
    def _teacher_extreme_points(obj):
        pts = (
            obj.get('extreme_points') or obj.get('extremes') or obj.get('points_4')
            or obj.get('extreme_4py') or obj.get('points')
        )
        if pts is None:
            return None
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 4:
            return None
        return pts[:4]

    def _transform_eagle_points(self, points, flipped, width, trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        poly = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if flipped:
            poly = poly.copy()
            poly[:, 0] = width - np.array(poly[:, 0]) - 1
        transformed = snake_voc_utils.transform_polys([poly], trans_output, output_h, output_w)
        if not transformed:
            return None
        transformed = transformed[0].astype(np.float32)
        transformed[:, 0] = np.clip(transformed[:, 0], 0, output_w - 1)
        transformed[:, 1] = np.clip(transformed[:, 1], 0, output_h - 1)
        return transformed[:4]

    def _build_eagle_targets(self, img_path, cls_ids_ordered, flipped, width, trans_output, inp_out_hw):
        instances = self._get_eagle_instances(img_path)
        if not instances:
            return None, None
        conf_thresh = float(getattr(cfg, 'eagle_teacher_conf_thresh', 0.0))
        queues = {}
        for obj in instances:
            if not isinstance(obj, dict):
                continue
            score = float(obj.get('confidence', obj.get('score', 1.0)))
            if score < conf_thresh:
                continue
            label = self._teacher_label(obj)
            pts = self._teacher_extreme_points(obj)
            if label is None or pts is None:
                continue
            pts = self._transform_eagle_points(pts, flipped, width, trans_output, inp_out_hw)
            if pts is None or pts.shape[0] < 4:
                continue
            queues.setdefault(label, []).append(pts)

        targets = []
        mask = []
        for cls_id in cls_ids_ordered:
            q = queues.get(int(cls_id), [])
            if q:
                targets.append(q.pop(0))
                mask.append(1.0)
            else:
                targets.append(np.zeros((4, 2), dtype=np.float32))
                mask.append(0.0)
        if not targets:
            return None, None
        return targets, mask


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
        num_points = self.compute_adaptive_points(bbox)
        base_init_poly = snake_voc_utils.get_evolution_init(extreme_point, bbox)
        img_init_poly = snake_voc_utils.uniformsample(base_init_poly, num_points)
        can_init_poly = snake_voc_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)

        img_gt_poly = snake_voc_utils.uniformsample(poly, len(poly) * num_points)
        tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]
        can_gt_poly = snake_voc_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)

    def compute_adaptive_points(self, bbox):
        """Compute contour point count from bbox when adaptive mode is enabled."""
        if not getattr(snake_config, 'adaptive_points_enabled', False):
            return int(snake_config.poly_num)

        w = float(bbox[2] - bbox[0])
        h = float(bbox[3] - bbox[1])

        use_threshold = bool(getattr(snake_config, 'adaptive_use_area_threshold', False))
        strategy = str(getattr(snake_config, 'point_strategy', 'perimeter')).strip().lower()
        if use_threshold or strategy in ('area_threshold', 'threshold'):
            area = max(w, 0.0) * max(h, 0.0)
            area_threshold = float(getattr(snake_config, 'adaptive_area_threshold', 4096.0))
            small_points = int(getattr(snake_config, 'adaptive_small_points', 64))
            large_points = int(getattr(snake_config, 'adaptive_large_points', 128))
            return int(small_points if area < area_threshold else large_points)

        target_density = float(getattr(snake_config, 'target_density', 2.5))
        min_points = int(getattr(snake_config, 'min_points', 32))
        max_points = int(getattr(snake_config, 'max_points', 512))
        round_to = max(1, int(getattr(snake_config, 'round_to_multiple', 8)))

        if strategy == 'perimeter':
            base_points = 2.0 * (w + h) / max(target_density, 1e-6)
        elif strategy == 'area':
            base_points = np.sqrt(max(w * h, 0.0)) * 1.5
        elif strategy == 'mixed':
            perimeter_factor = 2.0 * (w + h) / max(target_density, 1e-6)
            area_factor = np.sqrt(max(w * h, 0.0)) * 1.5
            base_points = 0.6 * perimeter_factor + 0.4 * area_factor
        else:
            base_points = 2.0 * (w + h) / max(target_density, 1e-6)

        num_points = int(np.clip(base_points, min_points, max_points))
        num_points = ((num_points + round_to - 1) // round_to) * round_to
        return int(min(max(num_points, min_points), max_points))
        
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
        eagle_cls_ordered = []
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


                    eagle_cls_ordered.append(cls_id)
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
        locate_feat = self._load_locate_feature(img_path)
        if locate_feat:
            ret.update(locate_feat)
        if bool(getattr(cfg, 'use_eagle_teacher_init', False)):
            eagle_4py, eagle_mask = self._build_eagle_targets(
                img_path,
                eagle_cls_ordered,
                flipped,
                width,
                trans_output,
                inp_out_hw,
            )
            if eagle_4py is not None and eagle_mask is not None:
                ret['eagle_i_gt_4py'] = eagle_4py
                ret['eagle_4py_mask'] = eagle_mask

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
        inv_trans_input = cv2.invertAffineTransform(trans_input).astype(np.float32)
        meta = {
            'center': center,
            'scale': scale,
            'ct_num': ct_num,
            'trans_input': trans_input.astype(np.float32),
            'inv_trans_input': inv_trans_input,
            'flipped': np.asarray([1 if flipped else 0], dtype=np.float32),
            'orig_hw': np.asarray([height, width], dtype=np.float32),
            'inp_out_hw': np.asarray(inp_out_hw, dtype=np.float32),
        }
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        if getattr(self, 'per_contour', False):
            return len(self.samples)
        return len(self.train_images_path)
