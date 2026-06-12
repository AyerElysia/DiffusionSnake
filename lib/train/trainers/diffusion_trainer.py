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
        self.ex_crit = F.smooth_l1_loss

        # 损失权重
        default_scales = {'det': 1.0, 'mask': 0.0, 'ex': 0.0, 'diff': 1.0, 'prior': 1.0}
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

    @staticmethod
    def _poly_to_mask(poly: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if poly.numel() == 0 or poly.size(0) < 3:
            return torch.zeros((height, width), device=poly.device, dtype=torch.bool)

        yy, xx = torch.meshgrid(
            torch.arange(height, device=poly.device, dtype=poly.dtype) + 0.5,
            torch.arange(width, device=poly.device, dtype=poly.dtype) + 0.5,
            indexing='ij',
        )
        x0 = poly[:, 0].clamp(0, max(width - 1, 0))
        y0 = poly[:, 1].clamp(0, max(height - 1, 0))
        x1 = torch.roll(x0, shifts=-1, dims=0)
        y1 = torch.roll(y0, shifts=-1, dims=0)

        denom = y1 - y0
        eps = torch.full_like(denom, 1e-6)
        denom = torch.where(denom.abs() < 1e-6, eps, denom)
        crosses = (y0[:, None, None] > yy) != (y1[:, None, None] > yy)
        x_at_y = (x1 - x0)[:, None, None] * (yy - y0[:, None, None]) / denom[:, None, None] + x0[:, None, None]
        inside = crosses & (xx < x_at_y)
        return (inside.sum(dim=0) % 2) == 1

    def _build_heatmap_mask_target(self, batch, mask_logits: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = mask_logits.shape
        target = torch.zeros_like(mask_logits)
        if 'i_gt_py' not in batch or 'ct_cls' not in batch or 'ct_01' not in batch:
            return target

        polys = batch['i_gt_py'].to(device=mask_logits.device, dtype=mask_logits.dtype)
        ct_cls = batch['ct_cls'].to(device=mask_logits.device)
        ct_01 = batch['ct_01'].to(device=mask_logits.device).bool()
        point_mask = batch.get('point_mask', None)
        if isinstance(point_mask, torch.Tensor):
            point_mask = point_mask.to(device=mask_logits.device)
        class_offset = int(getattr(cfg, 'heatmap_class_offset', 0))

        with torch.no_grad():
            for b in range(min(bsz, polys.size(0))):
                valid_inds = ct_01[b].nonzero(as_tuple=False).flatten()
                for ind in valid_inds:
                    cls_id = int(ct_cls[b, ind].item()) - class_offset
                    if cls_id < 0 or cls_id >= channels:
                        continue
                    poly = polys[b, ind]
                    if isinstance(point_mask, torch.Tensor) and point_mask.dim() == 3:
                        valid_pts = point_mask[b, ind] > 0.5
                        poly = poly[valid_pts]
                    mask = self._poly_to_mask(poly, height, width).to(dtype=target.dtype)
                    target[b, cls_id] = torch.maximum(target[b, cls_id], mask)
        return target

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

        # 1.5) Class-aware polygon mask loss. This is used by V8.2 to give the
        # heatmap detector dense shape supervision in addition to center/box loss.
        mask_weight = float(self.loss_scales.get('mask', 0.0))
        if mask_weight > 0.0 and ('mask_logits' in output):
            mask_logits = output['mask_logits']
            mask_target = self._build_heatmap_mask_target(batch, mask_logits)
            pos_weight_value = float(getattr(cfg, 'heatmap_mask_pos_weight', 5.0))
            pos_weight = torch.full(
                (1, mask_logits.size(1), 1, 1),
                pos_weight_value,
                device=mask_logits.device,
                dtype=mask_logits.dtype,
            )
            mask_bce = F.binary_cross_entropy_with_logits(mask_logits, mask_target, pos_weight=pos_weight)
            mask_prob = torch.sigmoid(mask_logits)
            reduce_dims = (0, 2, 3)
            inter = (mask_prob * mask_target).sum(dim=reduce_dims)
            denom = mask_prob.sum(dim=reduce_dims) + mask_target.sum(dim=reduce_dims)
            active = mask_target.sum(dim=reduce_dims) > 0
            if bool(active.any()):
                mask_dice = (1.0 - (2.0 * inter[active] + 1.0) / (denom[active] + 1.0)).mean()
            else:
                mask_dice = mask_bce * 0.0
            dice_weight = float(getattr(cfg, 'heatmap_mask_dice_weight', 1.0))
            mask_loss = mask_bce + dice_weight * mask_dice
            loss = loss + mask_weight * mask_loss
            scalar_stats.update({
                'mask_loss': mask_loss,
                'mask_bce': mask_bce,
                'mask_dice': mask_dice,
                'mask_loss_scaled': mask_weight * mask_loss,
            })
        else:
            scalar_stats.update({'mask_loss': torch.tensor(0.0, device=base_device)})

        # 2) Extreme-point refinement loss. This aligns detector-side init with
        # the octagon initialization used by diffusion training/inference.
        ex_weight = float(self.loss_scales.get('ex', 0.0))
        if (
            ex_weight > 0.0
            and (not self.freeze_snake)
            and ('ex_pred' in output)
            and ('i_gt_4py' in output)
        ):
            ex_target = output['i_gt_4py'].to(device=output['ex_pred'].device, dtype=output['ex_pred'].dtype)
            ex_loss = self.ex_crit(output['ex_pred'], ex_target)
            loss = loss + ex_weight * ex_loss
            scalar_stats.update({'ex_loss': ex_loss, 'ex_loss_scaled': ex_weight * ex_loss})
        else:
            scalar_stats.update({'ex_loss': torch.tensor(0.0, device=base_device)})

        eagle_weight = float(getattr(cfg, 'eagle_teacher_loss_weight', 0.0))
        if (
            eagle_weight > 0.0
            and (not self.freeze_snake)
            and ('ex_pred' in output)
            and ('eagle_i_gt_4py' in batch)
            and ('eagle_4py_mask' in batch)
        ):
            ct_01 = batch['ct_01'].to(device=output['ex_pred'].device).bool()
            teacher = batch['eagle_i_gt_4py'].to(device=output['ex_pred'].device, dtype=output['ex_pred'].dtype)
            teacher_mask = batch['eagle_4py_mask'].to(device=output['ex_pred'].device).bool()
            teacher_flat = torch.cat([teacher[b][ct_01[b]] for b in range(ct_01.size(0))], dim=0)
            mask_flat = torch.cat([teacher_mask[b][ct_01[b]] for b in range(ct_01.size(0))], dim=0)
            if teacher_flat.numel() > 0 and bool(mask_flat.any()):
                eagle_loss = self.ex_crit(output['ex_pred'][mask_flat], teacher_flat[mask_flat])
                loss = loss + eagle_weight * eagle_loss
            else:
                eagle_loss = output['ex_pred'].sum() * 0.0
            scalar_stats.update({
                'eagle_ex_loss': eagle_loss,
                'eagle_ex_loss_scaled': eagle_weight * eagle_loss,
            })
        else:
            scalar_stats.update({'eagle_ex_loss': torch.tensor(0.0, device=base_device)})

        # 3) Diffusion 去噪损失
        if (not self.freeze_snake) and ('diff_loss' in output):
            diff_weight = float(getattr(cfg, 'diffusion_loss_weight', 1.0)) * float(self.loss_scales.get('diff', 1.0))
            loss = loss + diff_weight * output['diff_loss']
            scalar_stats.update({'diff_loss': output['diff_loss'], 'diff_loss_scaled': diff_weight * output['diff_loss']})
            for k_, v_ in output.items():
                if isinstance(k_, str) and k_.startswith('diff_loss'):
                    scalar_stats[k_] = v_
        else:
            scalar_stats.update({'diff_loss': torch.tensor(0.0, device=base_device)})

        for debug_key in (
            'ex_box_jitter_count',
            'pred_extreme_init_count',
            'gt_extreme_init_count',
            'pred_extreme_init_prob_effective',
            'locate_feat_residual_absmax',
            'locate_feat_adapter_last_absmax',
            'locate_feat_replace_absmax',
        ):
            if debug_key in output:
                value = output[debug_key]
                if isinstance(value, torch.Tensor):
                    scalar_stats[debug_key] = value.detach().cpu()
                else:
                    try:
                        scalar_stats[debug_key] = torch.tensor(float(value))
                    except Exception:
                        pass

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
