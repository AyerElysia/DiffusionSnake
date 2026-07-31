"""
Temporal contour initialization for sagittal slice inference.

Injects previous-frame predicted contours as Snake init, bypassing the
standard bbox-octagon.  Works by populating output['sam_i_it_py'] so that
snake_gcn_utils.prepare_testing() picks them up via the existing SAM path —
no changes to flow_matching_evolution required.

Cache format expected in batch['prev_contour_cache']:
    {cls_id (int): np.ndarray shape [P, 2], feature-map coordinates (image/4)}

output['py'] produced by the network is already in feature coords, so
contours can be cached and reused directly with no rescaling.
"""
import numpy as np
import torch

from lib.utils.snake import snake_config
from lib.utils.snake.snake_voc_utils import uniformsample
from lib.utils.snake.snake_gcn_utils import img_poly_to_can_poly, prepare_testing_init


def cache_previous_predictions(output):
    """Build a class-to-contour cache from one single-image inference output."""
    py = output.get('py') if isinstance(output, dict) else None
    if isinstance(py, (list, tuple)):
        py = py[-1] if py else None
    detection = output.get('detection') if isinstance(output, dict) else None
    if not torch.is_tensor(py) or not torch.is_tensor(detection) or py.numel() == 0:
        return {}
    if py.ndim == 4 and py.size(0) == 1:
        py = py[0]
    if py.ndim != 3 or py.shape[-1] != 2:
        raise RuntimeError('Expected py with shape [N,P,2], got {}'.format(tuple(py.shape)))
    if detection.ndim == 2:
        detection = detection.unsqueeze(0)
    if detection.ndim != 3 or detection.size(0) != 1 or detection.size(-1) != 6:
        raise RuntimeError(
            'Temporal propagation requires single-image detection [1,N,6], got {}'.format(
                tuple(detection.shape)
            )
        )
    valid = detection[0, :, 4] > 1e-4
    labels = detection[0, valid, 5].long()
    if labels.numel() != py.size(0):
        raise RuntimeError(
            'Cannot cache temporal contours: {} final contours for {} valid detections'.format(
                py.size(0), labels.numel()
            )
        )
    contours = py.detach().float().cpu().numpy()
    return {
        int(label): contours[index].copy()
        for index, label in enumerate(labels.detach().cpu().tolist())
    }


def attach_prev_contour_testing_init(output, batch, device):
    """Replace bbox-octagon init with cached previous-frame contours.

    For each valid detection in the current frame, if the class ID exists in
    batch['prev_contour_cache'] the cached contour is used as i_it_py; otherwise
    the standard octagon init is built from the detected bbox.

    The function only injects when at least one instance uses a cached contour.
    If no cache hits are found, output is returned unchanged and octagon is used.
    """
    prev_cache = batch.get('prev_contour_cache')  # {cls_id: np.ndarray [P, 2]}
    if not prev_cache:
        return output

    detection = output.get('detection')  # [B, N, 6]
    if not torch.is_tensor(detection) or detection.numel() == 0:
        return output

    poly_num = int(snake_config.poly_num)
    all_contours = []
    all_batch_idx = []
    has_any_prev = False

    for b in range(detection.size(0)):
        det_b = detection[b]             # [N, 6]
        valid = det_b[:, 4] > 1e-4
        det_valid = det_b[valid]         # [M, 6]
        if det_valid.size(0) == 0:
            continue

        for i in range(det_valid.size(0)):
            cls_id = int(det_valid[i, 5].item())
            if cls_id in prev_cache:
                # Resample cached contour to poly_num pts (already feature coords)
                prev_poly = prev_cache[cls_id].astype(np.float32)
                resampled = uniformsample(prev_poly, poly_num)
                t = torch.from_numpy(resampled).to(device=device, dtype=det_valid.dtype)
                all_contours.append(t)
                has_any_prev = True
            else:
                # Fallback: build octagon from detected bbox (in feature coords)
                box = det_valid[i:i + 1, :4].unsqueeze(0)   # [1, 1, 4]
                score = det_valid[i:i + 1, 4:5]             # [1, 1]
                init = prepare_testing_init(box, score)
                i4 = init['i_it_4py']                       # [1, init_poly_num, 2] feature coords
                if i4.numel() == 0:
                    # Degenerate box — skip; prepare_testing would skip it too
                    continue
                # Upsample 4-point init to poly_num
                from lib.utils.snake.snake_gcn_utils import uniform_upsample
                poly_t = uniform_upsample(i4.unsqueeze(0), poly_num)[0].squeeze(0)  # [poly_num, 2]
                all_contours.append(poly_t.to(device=device))
            all_batch_idx.append(b)

    if not has_any_prev or not all_contours:
        return output

    sam_i_it_py = torch.stack(all_contours, dim=0)                   # [M, poly_num, 2]
    sam_py_ind = torch.tensor(all_batch_idx, dtype=torch.long, device=device)
    output['sam_i_it_py'] = sam_i_it_py
    output['sam_py_ind'] = sam_py_ind
    return output
