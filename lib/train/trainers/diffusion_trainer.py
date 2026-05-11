import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from lib.config import cfg
from lib.networks.YOLOV8.utils.loss import v8DetectionLoss
from lib.utils import net_utils

logger = logging.getLogger(__name__)


class DiffusionPretrainNetworkWrapper(nn.Module):
    """
    仅做预训练：YOLO 检测损失 + Diffusion 去噪损失（无 GRPO）。
    若 output 中包含 CMAM 先验矩阵 L 与 L_star，则附加 MSE 约束。
    """
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.detector_backend = str(getattr(cfg, 'detector_backend', 'yolo') or 'yolo').strip().lower()

        self.det_crit = None
        self.ct_crit = None
        self.wh_crit = None
        self.heatmap_wh_weight = float(getattr(cfg, 'heatmap_wh_weight', 0.1))
        if self.detector_backend == 'yolo':
            try:
                self.det_crit = self.net.yolo.init_criterion()
            except (AttributeError, RuntimeError) as e:
                logger.warning(f"Failed to init YOLO criterion, using default: {e}")
                self.det_crit = v8DetectionLoss(self.net.yolo)
        else:
            self.ct_crit = net_utils.FocalLoss()
            self.wh_crit = net_utils.IndL1Loss1d('smooth_l1')

        # 先验损失
        self.L_crit = F.mse_loss

        # 损失权重
        default_scales = {'det': 1.0, 'diff': 1.0, 'prior': 1.0}
        self.loss_scales = getattr(cfg, 'loss_scales', default_scales)
        for k, v in default_scales.items():
            if k not in self.loss_scales:
                self.loss_scales[k] = v

        # 若 freeze_snake=True, diffusion 分支不训练；若 freeze_yolo=True, YOLO 分支不训练
        self.freeze_snake = bool(getattr(cfg, 'freeze_snake', False))
        self.freeze_yolo = bool(getattr(cfg, 'freeze_yolo', False))

        if self.freeze_yolo and getattr(self.net, 'yolo', None) is not None:
            for p in self.net.yolo.parameters():
                p.requires_grad = False
            self.net.yolo.eval()

    @staticmethod
    def _masked_contour_regularizer(contours: torch.Tensor, point_mask: torch.Tensor, kind: str) -> torch.Tensor:
        """Compute a pointwise contour regularizer on compacted valid points only."""
        if contours.numel() == 0 or point_mask is None:
            return contours.sum() * 0.0

        losses = []
        for contour, mask in zip(contours, point_mask):
            valid = mask > 0.5
            if int(valid.sum().item()) < 3:
                continue

            pts = contour[valid]
            prev = torch.roll(pts, 1, dims=0)
            next = torch.roll(pts, -1, dims=0)

            if kind == 'smooth':
                diff = pts - (prev + next) * 0.5
            elif kind == 'curv':
                diff = next - 2.0 * pts + prev
            else:
                raise ValueError(f"Unsupported regularizer kind: {kind}")

            losses.append(torch.mean(diff ** 2))

        if not losses:
            return contours.sum() * 0.0
        return torch.stack(losses).mean()

    def forward(self, batch):
        # 仅调用原网络，保证不会进入 GRPO 分支
        output = self.net(batch['inp'], batch)

        # 总损失与标量统计
        base_device = output['detection'].device if isinstance(output.get('detection', None), torch.Tensor) else batch['inp'].device
        loss = torch.zeros(1, device=base_device).squeeze()
        scalar_stats = {}
        image_stats = {}

        # 1) YOLO 检测损失
        det_weight = float(self.loss_scales.get('det', 1.0))
        if (
            det_weight > 0.0
            and (not self.freeze_yolo)
            and self.detector_backend == 'yolo'
            and ('yolo_preds' in output)
            and (self.det_crit is not None)
        ):
            det_loss, det_items = self.det_crit(output['yolo_preds'], batch)
            box_l, cls_l, dfl_l = det_items[0], det_items[1], det_items[2]
            loss = loss + det_weight * det_loss
            bs = int(batch.get('inp').shape[0]) if isinstance(batch, dict) and isinstance(batch.get('inp', None), torch.Tensor) else 1
            bs = max(bs, 1)
            det_loss_log = det_loss / float(bs)
            scalar_stats.update({
                'det_loss': det_loss_log,
                'det_box': box_l,
                'det_cls': cls_l,
                'det_dfl': dfl_l,
                'det_loss_scaled': det_weight * det_loss_log,
            })
        elif (
            det_weight > 0.0
            and (not self.freeze_yolo)
            and self.ct_crit is not None
            and ('ct_hm' in output)
            and ('wh' in output)
        ):
            ct_target = batch['ct_hm'].to(output['ct_hm'].device)
            wh_target = batch['wh'].to(output['wh'].device)
            ct_ind = batch['ct_ind'].to(output['wh'].device)
            ct_mask = batch['ct_01'].to(output['wh'].device)

            ct_loss = self.ct_crit(output['ct_hm'], ct_target)
            wh_loss = self.wh_crit(output['wh'], wh_target, ct_ind, ct_mask)
            det_loss = ct_loss + self.heatmap_wh_weight * wh_loss
            loss = loss + det_weight * det_loss
            scalar_stats.update({
                'det_loss': det_loss,
                'det_ct': ct_loss,
                'det_wh': wh_loss,
                'det_loss_scaled': det_weight * det_loss,
            })
        else:
            scalar_stats.update({'det_loss': torch.tensor(0.0, device=base_device)})

        # 2) Diffusion 去噪损失
        if (not self.freeze_snake) and ('diff_loss' in output):
            diff_weight = float(getattr(cfg, 'diffusion_loss_weight', 1.0))
            loss = loss + diff_weight * output['diff_loss']
            scalar_stats.update({'diff_loss': output['diff_loss'], 'diff_loss_scaled': diff_weight * output['diff_loss']})
            for k_, v_ in output.items():
                if isinstance(k_, str) and k_.startswith('diff_loss'):
                    scalar_stats[k_] = v_
        else:
            scalar_stats.update({'diff_loss': torch.tensor(0.0, device=base_device)})

        # 4) Combined metric for monitoring: detection loss + diffusion loss (both unscaled)
        det_l = scalar_stats.get('det_loss', None)
        diff_l = scalar_stats.get('diff_loss', None)
        if isinstance(det_l, torch.Tensor) and isinstance(diff_l, torch.Tensor):
            scalar_stats['det_plus_diff_loss'] = det_l + diff_l

        # 3) CMAM 先验（可选）
        if 'L' in output and 'L_star' in output and (not self.freeze_snake):
            L_loss = self.L_crit(output['L'], output['L_star'])
            prior_weight = float(self.loss_scales.get('prior', 1.0))
            loss = loss + prior_weight * L_loss
            scalar_stats.update({'L_loss': L_loss, 'L_loss_scaled': prior_weight * L_loss})
        else:
            scalar_stats.update({'L_loss': torch.tensor(0.0, device=base_device)})

        # 4) Smoothness Loss (NEW in V3.3)
        if (not self.freeze_snake) and ('pred_contours' in output):
            smooth_weight = float(self.loss_scales.get('smooth', 0.0))
            curv_weight = float(self.loss_scales.get('curv', 0.0))
            point_mask = output.get('point_mask', None)

            if smooth_weight > 0 or curv_weight > 0:
                contours = output['pred_contours']  # (N, P, 2)
                base_contours = output.get('i_it_py', None)
                masked_contours = None

                # Regularize the predicted correction rather than absolute coordinates.
                # This keeps the penalty focused on local jaggedness and avoids
                # over-penalizing the coarse contour geometry itself.
                if isinstance(base_contours, torch.Tensor) and base_contours.shape == contours.shape:
                    contours = contours - base_contours
                if isinstance(point_mask, torch.Tensor) and point_mask.dim() == 2 and point_mask.shape[:2] == contours.shape[:2]:
                    point_mask = point_mask.to(device=contours.device, dtype=contours.dtype)
                    masked_contours = contours

                # Laplacian smoothness loss
                if smooth_weight > 0:
                    if masked_contours is not None:
                        smooth_loss = self._masked_contour_regularizer(masked_contours, point_mask, 'smooth')
                    else:
                        prev = torch.roll(contours, 1, dims=1)
                        next = torch.roll(contours, -1, dims=1)
                        laplacian = contours - (prev + next) / 2
                        smooth_loss = torch.mean(laplacian ** 2)
                    loss = loss + smooth_weight * smooth_loss
                    scalar_stats.update({
                        'smooth_loss': smooth_loss,
                        'smooth_loss_scaled': smooth_weight * smooth_loss
                    })
                else:
                    scalar_stats.update({'smooth_loss': torch.tensor(0.0, device=base_device)})

                # Curvature loss
                if curv_weight > 0:
                    # Cyclic second-order difference keeps the contour closed.
                    if masked_contours is not None:
                        curv_loss = self._masked_contour_regularizer(masked_contours, point_mask, 'curv')
                    else:
                        v2 = (
                            torch.roll(contours, -1, dims=1)
                            - 2.0 * contours
                            + torch.roll(contours, 1, dims=1)
                        )
                        curv_loss = torch.mean(v2 ** 2)
                    loss = loss + curv_weight * curv_loss
                    scalar_stats.update({
                        'curv_loss': curv_loss,
                        'curv_loss_scaled': curv_weight * curv_loss
                    })
                else:
                    scalar_stats.update({'curv_loss': torch.tensor(0.0, device=base_device)})
            else:
                scalar_stats.update({
                    'smooth_loss': torch.tensor(0.0, device=base_device),
                    'curv_loss': torch.tensor(0.0, device=base_device)
                })
        else:
            scalar_stats.update({
                'smooth_loss': torch.tensor(0.0, device=base_device),
                'curv_loss': torch.tensor(0.0, device=base_device)
            })

        scalar_stats.update({'loss': loss})
        for k, v in list(scalar_stats.items()):
            if isinstance(v, torch.Tensor):
                scalar_stats[k] = v.detach()

        # Clean up large intermediate outputs during training to save memory
        if self.training and isinstance(output, dict):
            output.pop('yolo_preds', None)
            # Return minimal output during training
            output = {}

        return output, loss, scalar_stats, image_stats
