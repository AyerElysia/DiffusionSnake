import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def _parse_args():
    parser = argparse.ArgumentParser(description='Smoke-test BTCV detector checkpoint on a single sample.')
    parser.add_argument('--cfg', default='configs/btcv_yolo_detect_only.yaml')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--dataset', choices=['train', 'val'], default='train')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--score-thresh', type=float, default=0.01)
    parser.add_argument('--save-path', default='')
    parser.add_argument('--disable-aug', action='store_true')
    parser.add_argument('--yolo-num-classes', type=int, default=0)
    return parser.parse_args()


def _set_cfg_file(cfg_file: str):
    os.environ['CFG_FILE'] = cfg_file
    sys.argv = [sys.argv[0]]


def _iou_xyxy(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = max(area_a + area_b - inter, 1e-6)
    return inter / union


def _unwrap_network_state_dict(state_dict):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict

    sample_keys = list(state_dict.keys())
    if all(k.startswith('net.') for k in sample_keys):
        return {k[4:]: v for k, v in state_dict.items()}
    if all(k.startswith('module.net.') for k in sample_keys):
        return {k[len('module.net.'):]: v for k, v in state_dict.items()}
    return state_dict


def main():
    args = _parse_args()
    if args.disable_aug:
        os.environ['SNAKE_DISABLE_AUG'] = '1'
    _set_cfg_file(args.cfg)

    from lib.config import cfg
    from lib.datasets.collate_batch import make_collator
    from lib.datasets.make_dataset import make_dataset
    from lib.datasets.transforms import make_transforms
    from lib.networks import make_network

    cfg.train.data_path = '/home/medteam/Zhrch/Datasets/BTCV/btcv_png_new_snake'
    cfg.test.img_path = '/home/medteam/Zhrch/Datasets/BTCV/btcv_png_test_new_snake'
    if args.yolo_num_classes > 0:
        cfg.yolo_num_classes = args.yolo_num_classes

    dataset_name = cfg.train.dataset if args.dataset == 'train' else cfg.test.dataset
    transforms = make_transforms(cfg, is_train=(args.dataset == 'train'))
    dataset = make_dataset(cfg, dataset_name, transforms, is_train=(args.dataset == 'train'))
    sample = dataset[args.index]
    batch = make_collator(cfg)([sample])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = make_network(cfg).to(device).eval()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = _unwrap_network_state_dict(ckpt.get('state_dict', ckpt))
    network.load_state_dict(state_dict, strict=False)

    for key, value in list(batch.items()):
        if key in ('meta', 'orig_img', 'img_path'):
            continue
        if torch.is_tensor(value):
            batch[key] = value.to(device)

    with torch.no_grad():
        output = network(batch['inp'], batch)

    detections = output['detection'][0].detach().cpu().numpy()
    detections = detections[detections[:, 4] > args.score_thresh]
    detections = detections[np.argsort(-detections[:, 4])]

    inp_h, inp_w = sample['inp'].shape[1:]
    gt_boxes = []
    for box_xywh, cls_tensor in zip(sample['bboxes'].numpy(), sample['cls'].view(-1).numpy()):
        cx, cy, w, h = [float(x) for x in box_xywh]
        gt_boxes.append([
            (cx - w / 2.0) * inp_w,
            (cy - h / 2.0) * inp_h,
            (cx + w / 2.0) * inp_w,
            (cy + h / 2.0) * inp_h,
            int(cls_tensor),
        ])

    matches = []
    for gt_idx, gt_box in enumerate(gt_boxes):
        best_iou = 0.0
        best_pred_idx = -1
        best_score = 0.0
        for pred_idx, pred in enumerate(detections):
            if int(pred[5]) != gt_box[4]:
                continue
            iou = _iou_xyxy(gt_box[:4], pred[:4])
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx
                best_score = float(pred[4])
        matches.append({
            'gt_index': gt_idx,
            'gt_cls': gt_box[4],
            'best_pred_index': best_pred_idx,
            'best_iou': best_iou,
            'best_score': best_score,
        })

    canvas = np.zeros((inp_h, inp_w, 3), dtype=np.uint8)
    for gt in gt_boxes:
        x1, y1, x2, y2, cls_id = gt
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f'g{cls_id}',
            (int(x1), max(12, int(y1) - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )
    for pred in detections[:20]:
        x1, y1, x2, y2, score, cls_id = pred
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        cv2.putText(
            canvas,
            f'p{int(cls_id)}:{float(score):.2f}',
            (int(x1), max(12, int(y1) - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
        )

    save_path = args.save_path or str(
        Path(args.ckpt).resolve().parent.parent / f'{args.dataset}_sample{args.index}_pred_vs_gt.png'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, canvas)

    summary = {
        'dataset': args.dataset,
        'index': args.index,
        'num_gt': len(gt_boxes),
        'num_pred': int(len(detections)),
        'matches': matches,
        'save_path': save_path,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
