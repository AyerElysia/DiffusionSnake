# 保存位置举例：/mnt/sdb1/leijh/EnergySnake1/EnergeSnake1/tools/crf/build_crf_mat.py
import os
import numpy as np
import scipy.io as sio
import torch

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def _xyxy_to_yxyx(det_xyxy):  # [N, 6] -> [N, 6] with yx order
    x1, y1, x2, y2, score, cls = det_xyxy.T
    return np.stack([y1, x1, y2, x2, cls, score], axis=1)

def _pad_to_len(arr, target_len, pad_value=0.0):
    if arr.shape[0] >= target_len:
        return arr[:target_len]
    pad_rows = target_len - arr.shape[0]
    pad = np.zeros((pad_rows, arr.shape[1]), dtype=arr.dtype) + pad_value
    return np.concatenate([arr, pad], axis=0)

def collect_for_train(train_samples, save_mat_path, max_dets=100):
    """
    train_samples: list of dict per image:
      - 'detection': np.ndarray or tensor of shape [1, N, 6], columns [x1,y1,x2,y2,score,class_id]
    save_mat_path: '/path/to/rpn_and_detections_train0.mat'
    输出键：
      - mrcnn_boxes_and_scores_all: [num_images, max_dets, 6] with [y1,x1,y2,x2,class_id,score]
    """
    mrcnn_list = []
    for sample in train_samples:
        det = _to_numpy(sample['detection'])  # [1,N,6]
        det = det[0] if det.ndim == 3 else det
        det = det.astype(np.float32)
        # 转列顺序并保留 class_id 为第5列
        det_yxyx = _xyxy_to_yxyx(det)  # [N,6] -> [y1,x1,y2,x2,cls,score]
        det_yxyx = _pad_to_len(det_yxyx, max_dets, pad_value=0.0)
        mrcnn_list.append(det_yxyx)
    mrcnn_arr = np.stack(mrcnn_list, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(save_mat_path), exist_ok=True)
    sio.savemat(save_mat_path, {'mrcnn_boxes_and_scores_all': mrcnn_arr})
    print(f'saved: {save_mat_path}, mrcnn_boxes_and_scores_all shape={mrcnn_arr.shape}')

def collect_for_test(test_samples, save_mat_path, max_rois=100, max_dets=100):
    """
    test_samples: list of dict per image:
      - 'detection': [1, N, 6]  (必需)
      - 'rpn': [M, 5] 可选，列 [x1,y1,x2,y2,rpn_score]；若没有，会用 detection 的 [score] 近似生成
    save_mat_path: '/path/to/rpn_and_detections_test0.mat'
    输出键：
      - rpn_rois_and_scores_all: [num_images, max_rois, 5] -> [y1,x1,y2,x2,rpn_score]
      - mrcnn_boxes_and_scores_all: [num_images, max_dets, 6] -> [y1,x1,y2,x2,class_id,score]
    """
    rpn_list, mrcnn_list = [], []
    for sample in test_samples:
        det = _to_numpy(sample['detection'])
        det = det[0] if det.ndim == 3 else det
        det = det.astype(np.float32)

        # mrcnn / yolo 最终检测 -> [y1,x1,y2,x2,cls,score]
        mrcnn_yxyx = _xyxy_to_yxyx(det)
        mrcnn_yxyx = _pad_to_len(mrcnn_yxyx, max_dets, pad_value=0.0)
        mrcnn_list.append(mrcnn_yxyx)

        # RPN candidates -> 如果你没有RPN，可以用检测框和其分数近似（或来自YOLO候选）
        if 'rpn' in sample and sample['rpn'] is not None:
            rpn = _to_numpy(sample['rpn']).astype(np.float32)  # [M,5] [x1,y1,x2,y2,score]
            x1, y1, x2, y2, s = rpn.T
            rpn_yxyx = np.stack([y1, x1, y2, x2, s], axis=1)
        else:
            # 用最终检测近似：去掉class列，只留得分
            x1, y1, x2, y2, s, cls = det.T
            rpn_yxyx = np.stack([y1, x1, y2, x2, s], axis=1)
        rpn_yxyx = _pad_to_len(rpn_yxyx, max_rois, pad_value=0.0)
        rpn_list.append(rpn_yxyx)

    rpn_arr = np.stack(rpn_list, axis=0).astype(np.float32)
    mrcnn_arr = np.stack(mrcnn_list, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(save_mat_path), exist_ok=True)
    sio.savemat(save_mat_path, {
        'rpn_rois_and_scores_all': rpn_arr,
        'mrcnn_boxes_and_scores_all': mrcnn_arr
    })
    print(f'saved: {save_mat_path}, rpn shape={rpn_arr.shape}, mrcnn shape={mrcnn_arr.shape}')