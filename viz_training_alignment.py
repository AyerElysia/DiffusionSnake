import os
import cv2
import torch
import numpy as np
from lib.config import cfg
from lib.datasets.make_dataset import make_dataset
from lib.datasets.transforms import make_transforms
from lib.datasets.collate_batch import make_collator
from lib.utils.snake import snake_config, snake_gcn_utils

def main():
    cfg.merge_from_file('configs/btcv_diffusion_dit_v3.yaml')
    cfg.train.data_path = '/home/medteam/Zhrch/Datasets/BTCV/btcv_png_new_snake'
    dataset = make_dataset(cfg, 'BtcvTrain', make_transforms(cfg, is_train=True), is_train=True)
    batch = make_collator(cfg)([dataset[0]])
    init_data = snake_gcn_utils.prepare_training({'detection': torch.zeros((1, 100, 6))}, batch)
    i_init = init_data['i_it_py'][0].numpy() * snake_config.down_ratio
    i_gt = init_data['i_gt_py'][0].numpy() * snake_config.down_ratio
    img = batch['orig_img'][0].numpy().astype(np.uint8)
    for i in range(0, 128, 8):
        cv2.arrowedLine(img, tuple(i_init[i].astype(int)), tuple(i_gt[i].astype(int)), (255, 255, 255), 1)
    cv2.polylines(img, [i_init.astype(int)], True, (0, 255, 255), 1)
    cv2.polylines(img, [i_gt.astype(int)], True, (255, 0, 0), 2)
    os.makedirs("visual", exist_ok=True)
    cv2.imwrite("visual/train_alignment_check.png", img)
    print("Saved to visual/train_alignment_check.png")

if __name__ == '__main__':
    main()
