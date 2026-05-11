import torch
import torch.nn as nn

from lib.config import cfg
from lib.utils.snake import snake_gcn_utils


class SAMSnakeRefine(nn.Module):
    def __init__(self, c_in=64, num_points=128, stride=4.0):
        super().__init__()
        self.num_points = int(num_points)
        self.stride = float(stride)
        self.feature_dim = 64
        self.max_disp_frac = float(getattr(cfg, "samsnake_refine_max_disp_frac", 0.0))

        self.trans_feature = nn.Sequential(
            nn.Conv2d(c_in, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.feature_dim, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.trans_poly = nn.Linear((self.num_points + 1) * self.feature_dim, self.num_points * 4, bias=False)
        self.trans_fuse = nn.Linear(self.num_points * 4, self.num_points * 2, bias=True)

        if bool(getattr(cfg, "samsnake_refine_zero_init", True)):
            nn.init.zeros_(self.trans_fuse.weight)
            nn.init.zeros_(self.trans_fuse.bias)

    @staticmethod
    def _contour_scale(polys: torch.Tensor) -> torch.Tensor:
        span_x = polys[..., 0].amax(dim=1) - polys[..., 0].amin(dim=1)
        span_y = polys[..., 1].amax(dim=1) - polys[..., 1].amin(dim=1)
        return torch.maximum(span_x, span_y).clamp_min(1.0).view(-1, 1, 1)

    @staticmethod
    def _clip_poly(polys: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return torch.stack([
            polys[..., 0].clamp(min=0, max=max(w - 1, 0)),
            polys[..., 1].clamp(min=0, max=max(h - 1, 0)),
        ], dim=-1)

    def _clamp_disp(self, disp: torch.Tensor, init_polys: torch.Tensor) -> torch.Tensor:
        if self.max_disp_frac <= 0:
            return disp
        limit = self._contour_scale(init_polys).to(disp) * self.max_disp_frac
        norm = disp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return disp * torch.clamp(limit / norm, max=1.0)

    def forward(self, feature: torch.Tensor, centers: torch.Tensor, init_polys: torch.Tensor, py_ind: torch.Tensor, ignore=False):
        if ignore or init_polys.numel() == 0:
            return init_polys

        h, w = feature.size(2), feature.size(3)
        feature = self.trans_feature(feature)

        centers = centers.to(device=init_polys.device, dtype=init_polys.dtype)
        py_ind = py_ind.to(device=init_polys.device, dtype=torch.long)
        center_points = centers.unsqueeze(1)
        points = torch.cat([center_points, init_polys], dim=1)
        point_features = snake_gcn_utils.get_gcn_feature(feature, points, py_ind, h, w).view(init_polys.size(0), -1)

        offsets = self.trans_fuse(self.trans_poly(point_features)).view(init_polys.size(0), self.num_points, 2)
        disp = self._clamp_disp(offsets * self.stride, init_polys)
        coarse_polys = init_polys.detach() + disp
        return self._clip_poly(coarse_polys, h, w)
