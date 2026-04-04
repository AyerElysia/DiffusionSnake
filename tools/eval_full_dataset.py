"""
在全椎骨数据集上评估三个指标（mIoU、mDice、mBoundF），并考虑类别。

用法示例：
    python tools/eval_full_dataset.py \
        --ckpt data/outputs/grpo_t/model_grpo_20251210_204316.pth \
        --img_root /home/medteam/Zhrch/Data_processed/1232processed \
        --save_dir /home/medteam/Zhrch/EnergeSnake1GRPO/data/outputs/grpo_t/eval_full

若不传参，将默认使用上述模型与数据路径（存在则用）。
"""

import argparse
import glob
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data
import tqdm

# 保证可以从仓库根目录加载 lib 包，并避免 lib.config 的 argparse 与本脚本冲突
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_orig_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]  # 让 lib.config 的 argparse 不读取本脚本参数
from lib.config import cfg
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.utils import data_utils
from lib.utils.snake import snake_config
sys.argv = _orig_argv


# --------------------------- 基础工具函数 ---------------------------

def poly2mask(ex):
    """将多边形转换为二值掩码。"""
    ex = ex[-1] if isinstance(ex, list) else ex
    ex = ex.detach().cpu().numpy() * 4  # 与 test_medical 保持一致的尺度
    img = np.zeros((512, 512))
    ex = np.array(ex).astype(np.int32)
    for i in range(ex.shape[0]):
        img = cv2.fillPoly(img, [ex[i]], 1)
    return img


def cal_iou(mask, gtmask):
    jiaoji = mask * gtmask
    bingji = ((mask + gtmask) != 0).astype(np.int16)
    return jiaoji.sum() / bingji.sum() if bingji.sum() > 0 else 0.0


def cal_dice(iou):
    return 2 * iou / (iou + 1) if iou + 1 > 0 else 0.0


def extract_contour(mask, tolerance=1):
    mask = (mask > 0.5).astype(np.uint8)
    result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(result) == 3:
        _, contours, _ = result
    else:
        contours, _ = result
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, contours, -1, 1, thickness=tolerance)
    return contour_mask.astype(np.float32)


def cal_boundary_dice(pred_mask, gt_mask, tolerance=1):
    pred_contour = extract_contour(pred_mask, tolerance)
    gt_contour = extract_contour(gt_mask, tolerance)
    intersection = (pred_contour * gt_contour).sum()
    union = pred_contour.sum() + gt_contour.sum()
    return 2.0 * intersection / union if union > 0 else 0.0


def cal_mBoundF(pred_masks, gt_masks, pred_classes, gt_classes):
    tolerances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    boundary_dices_per_tolerance = []
    for tolerance in tolerances:
        instance_boundary_dices = []
        for i, gt_cls in enumerate(gt_classes):
            gt_mask = gt_masks[i]
            if gt_mask.sum() == 0:
                continue
            best_boundary_dice = 0.0
            for j, pred_cls in enumerate(pred_classes):
                if pred_cls == gt_cls:
                    pred_mask = pred_masks[j]
                    if pred_mask.sum() > 0:
                        boundary_dice = cal_boundary_dice(pred_mask, gt_mask, tolerance)
                        best_boundary_dice = max(best_boundary_dice, boundary_dice)
            instance_boundary_dices.append(best_boundary_dice)
        if instance_boundary_dices:
            boundary_dices_per_tolerance.append(np.mean(instance_boundary_dices))
    return np.mean(boundary_dices_per_tolerance) if boundary_dices_per_tolerance else 0.0


def cal_mBoundF_agnostic(pred_mask, gt_mask):
    tolerances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    vals = []
    for t in tolerances:
        vals.append(cal_boundary_dice(pred_mask, gt_mask, tolerance=t))
    return float(np.mean(vals)) if len(vals) > 0 else 0.0


def cal_class_wise_metrics(pred_masks, gt_masks, pred_classes, gt_classes):
    class_wise_ious = {}
    class_wise_dices = {}
    all_classes = set(pred_classes + gt_classes)
    for class_id in all_classes:
        pred_mask_class = np.zeros_like(pred_masks[0])
        gt_mask_class = np.zeros_like(gt_masks[0])
        for i, pred_cls in enumerate(pred_classes):
            if pred_cls == class_id:
                pred_mask_class = np.maximum(pred_mask_class, pred_masks[i])
        for i, gt_cls in enumerate(gt_classes):
            if gt_cls == class_id:
                gt_mask_class = np.maximum(gt_mask_class, gt_masks[i])
        if gt_mask_class.sum() > 0:
            iou = cal_iou(pred_mask_class, gt_mask_class)
            dice = cal_dice(iou)
            class_wise_ious[class_id] = iou
            class_wise_dices[class_id] = dice
    mean_iou = np.mean(list(class_wise_ious.values())) if class_wise_ious else 0.0
    mean_dice = np.mean(list(class_wise_dices.values())) if class_wise_dices else 0.0
    return class_wise_ious, class_wise_dices, mean_iou, mean_dice


# --------------------------- 数据集定义 ---------------------------

class FullDataset(data.Dataset):
    def __init__(self, img_root):
        super().__init__()
        assert os.path.isdir(img_root), f"img_root 不存在: {img_root}"
        self.imgs = sorted(glob.glob(os.path.join(img_root, '*_image.png')))
        if len(self.imgs) == 0:
            self.imgs = sorted(glob.glob(os.path.join(img_root, '*_image.jpg')))
        if len(self.imgs) == 0:
            raise RuntimeError(f"在 {img_root} 未找到 *_image.(png|jpg) 文件")

    def normalize_image(self, inp):
        inp = (inp.astype(np.float32) / 255.0)
        inp = (inp - snake_config.mean) / snake_config.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = cv2.imread(img_path)
        width, height = img.shape[1], img.shape[0]
        center = np.array([width // 2, height // 2])
        scale = np.array([width, height])
        x = 32
        input_w = ((width + x - 1) // x) * x
        input_h = ((height + x - 1) // x) * x
        trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        inp = self.normalize_image(inp)
        ret = {'inp': inp, 'meta': {'center': center, 'scale': scale}}
        return ret, img_path

    def __len__(self):
        return len(self.imgs)


# --------------------------- 模型加载 ---------------------------

def strip_prefix_if_present(state_dict, prefix_list=('module.', 'net.')):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        nk = k
        for pre in prefix_list:
            if nk.startswith(pre):
                nk = nk[len(pre):]
        new_sd[nk] = v
    return new_sd


def load_checkpoint(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict) and 'net' in sd:
        sd = sd['net']
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    sd = strip_prefix_if_present(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"加载完成，missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


# --------------------------- 评估主流程 ---------------------------

def evaluate(ckpt_path, img_root, save_dir=None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 明确使用 Diffusion + Snake
    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True
    if hasattr(cfg, 'test'):
        cfg.test.img_path = img_root
        cfg.test.visual_save_root = save_dir or cfg.test.visual_save_root

    network = make_network(cfg).cuda()
    network.eval()
    load_checkpoint(network, ckpt_path)

    dataset = FullDataset(img_root)

    # 累计量
    iou_sum = 0
    dice_sum = 0
    mBoundF_agnostic_sum = 0
    counter = 0

    class_wise_iou_sum = {}
    class_wise_dice_sum = {}
    class_wise_count = {}
    mBoundF_sum = 0

    for batch, img_path in tqdm.tqdm(dataset, desc='Evaluating'):
        img = cv2.imread(img_path)
        batch['inp'] = torch.FloatTensor(batch['inp'])[None].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)

        poly = output['py']
        detection = output['detection']

        mask_paths = glob.glob(img_path.replace('_image.png', '_mask') + '*')
        gt_masks, gt_classes = [], []
        mask_gt_combined = np.zeros((512, 512))
        for maskpath in mask_paths:
            class_match = re.search(r'_mask_(\d+)\.png', maskpath)
            class_id = int(class_match.group(1)) if class_match else 1
            mask = cv2.imread(maskpath, 0)
            if mask is None:
                continue
            # 统一尺寸到 512x512，保证与预测掩码一致
            mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_resized > 0).astype(np.float32)
            gt_masks.append(mask_binary)
            gt_classes.append(class_id)
            mask_gt_combined = np.maximum(mask_gt_combined, mask_binary)
        if not gt_masks:
            gt_masks = [mask_gt_combined]
            gt_classes = [1]
        mask_gt_combined = np.clip(mask_gt_combined, 0, 1)

        # 预测掩码
        pred_masks = []
        pred_classes = []
        if poly[-1].shape[0] > 0:
            for j in range(poly[-1].shape[0]):
                single_poly = poly[-1][j:j+1]
                pred_mask = poly2mask(single_poly)
                pred_masks.append(pred_mask)
                pred_classes.append(int(detection[0, j, 5]) + 1)

        pred_mask_combined = np.zeros((512, 512))
        for m in pred_masks:
            pred_mask_combined = np.maximum(pred_mask_combined, m)

        iou_overall = cal_iou(pred_mask_combined, mask_gt_combined)
        dice_overall = cal_dice(iou_overall)
        mBoundF_agnostic = cal_mBoundF_agnostic(pred_mask_combined, mask_gt_combined)

        iou_sum += iou_overall
        dice_sum += dice_overall
        mBoundF_agnostic_sum += mBoundF_agnostic

        # 类别相关
        pred_masks_merged = []
        pred_classes_merged = []
        if len(pred_masks) > 0:
            merged_by_class = {}
            for m, c in zip(pred_masks, pred_classes):
                merged_by_class[c] = np.maximum(merged_by_class.get(c, np.zeros_like(m)), m)
            for c in sorted(merged_by_class.keys()):
                pred_classes_merged.append(c)
                pred_masks_merged.append(merged_by_class[c])

        if len(pred_masks_merged) > 0 and len(gt_masks) > 0:
            class_ious, class_dices, mean_iou, mean_dice = cal_class_wise_metrics(
                pred_masks_merged, gt_masks, pred_classes_merged, gt_classes)
            mBoundF = cal_mBoundF(pred_masks_merged, gt_masks, pred_classes_merged, gt_classes)
            mBoundF_sum += mBoundF
            for class_id, iou in class_ious.items():
                class_wise_iou_sum[class_id] = class_wise_iou_sum.get(class_id, 0) + iou
                class_wise_dice_sum[class_id] = class_wise_dice_sum.get(class_id, 0) + class_dices[class_id]
                class_wise_count[class_id] = class_wise_count.get(class_id, 0) + 1

        counter += 1

    # 汇总
    print('\n=== 类别无关指标 ===')
    print(f"Overall mIoU:  {iou_sum / counter:.4f}")
    print(f"Overall mDice: {dice_sum / counter:.4f}")
    print(f"Overall mBoundF: {mBoundF_agnostic_sum / counter:.4f}")

    print('\n=== 类别相关指标 ===')
    if class_wise_count:
        class_avg_ious = {cid: class_wise_iou_sum[cid] / class_wise_count[cid] for cid in class_wise_count}
        class_avg_dices = {cid: class_wise_dice_sum[cid] / class_wise_count[cid] for cid in class_wise_count}
        overall_class_miou = np.mean(list(class_avg_ious.values()))
        overall_class_mdice = np.mean(list(class_avg_dices.values()))
        overall_mBoundF = mBoundF_sum / counter if counter > 0 else 0.0
        print(f"Class-aware mIoU:  {overall_class_miou:.4f}")
        print(f"Class-aware mDice: {overall_class_mdice:.4f}")
        print(f"mBoundF: {overall_mBoundF:.4f}")
    else:
        print("无有效的类别相关检测/标注，未计算类别相关指标")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=os.path.join('data', 'outputs', 'grpo_t', 'model_grpo_20251210_204316.pth'))
    parser.add_argument('--img_root', default='/home/medteam/Zhrch/Data_processed/1232processed')
    parser.add_argument('--save_dir', default='')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    img_root = args.img_root if os.path.isdir(args.img_root) else cfg.test.img_path
    evaluate(args.ckpt, img_root, save_dir=args.save_dir or None)
