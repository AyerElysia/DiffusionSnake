import os
import json
import math
import torch

# IMPORTANT: lib.config 会在 import 时解析 argv / 环境变量并加载 cfg。
# 因此请通过 --cfg_file 或 CFG_FILE 指定你要用的数据集配置。
from lib.config import cfg
from lib.datasets import make_data_loader
from lib.utils.snake import snake_gcn_utils


@torch.no_grad()
def _signed_area(poly: torch.Tensor) -> torch.Tensor:
    x = poly[..., 0]
    y = poly[..., 1]
    x1 = torch.roll(x, shifts=-1, dims=1)
    y1 = torch.roll(y, shifts=-1, dims=1)
    return 0.5 * torch.sum(x * y1 - x1 * y, dim=1)


@torch.no_grad()
def compute_disp_stats() -> dict:
    loader = make_data_loader(cfg, is_train=True, is_distributed=False)

    dx_min = math.inf
    dx_max = -math.inf
    dy_min = math.inf
    dy_max = -math.inf

    for batch in loader:
        init = snake_gcn_utils.prepare_training({}, batch)

        i_gt_py = init['i_gt_py']
        i_init_train_py = init['i_it_py']

        if not isinstance(i_gt_py, torch.Tensor) or i_gt_py.numel() == 0:
            continue

        device = i_gt_py.device
        i_init_train_py = i_init_train_py.to(device)

        # Orientation alignment + start-point alignment (must match training code)
        area_init = _signed_area(i_init_train_py)
        area_gt = _signed_area(i_gt_py)
        orient_mismatch = ((area_init >= 0) ^ (area_gt >= 0))
        if orient_mismatch.any():
            i_gt_py = i_gt_py.clone()
            i_gt_py[orient_mismatch] = torch.flip(i_gt_py[orient_mismatch], dims=[1])

        d2 = (i_init_train_py[:, :1, :] - i_gt_py).pow(2).sum(-1)
        nearest = torch.argmin(d2, dim=1)
        if i_gt_py.size(0) > 0:
            rolled = []
            for i in range(i_gt_py.size(0)):
                s = int(nearest[i].item())
                if s != 0:
                    rolled.append(torch.roll(i_gt_py[i], shifts=-s, dims=0))
                else:
                    rolled.append(i_gt_py[i])
            i_gt_py = torch.stack(rolled, dim=0)

        disp = i_gt_py - i_init_train_py  # [N,P,2]

        disp_cpu = disp.detach().float().cpu()
        dx_min = min(dx_min, float(disp_cpu[..., 0].min().item()))
        dx_max = max(dx_max, float(disp_cpu[..., 0].max().item()))
        dy_min = min(dy_min, float(disp_cpu[..., 1].min().item()))
        dy_max = max(dy_max, float(disp_cpu[..., 1].max().item()))

    if not (math.isfinite(dx_min) and math.isfinite(dx_max) and math.isfinite(dy_min) and math.isfinite(dy_max)):
        raise RuntimeError('Failed to compute disp stats: got non-finite min/max. Check your dataset and cfg.')

    return {
        'dx_min': dx_min,
        'dx_max': dx_max,
        'dy_min': dy_min,
        'dy_max': dy_max,
    }


def main():
    stats_path = str(getattr(cfg, 'diffusion_disp_stats', '') or '').strip()
    if not stats_path:
        raise ValueError('cfg.diffusion_disp_stats is empty. Please set it in your yaml (e.g. data/stats/btcv_disp_stats.json).')

    stats = compute_disp_stats()
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print('[disp_stats] saved to:', stats_path)
    print('[disp_stats] stats:', stats)


if __name__ == '__main__':
    main()
