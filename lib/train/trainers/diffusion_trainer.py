import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from lib.networks.YOLOV8.utils.loss import v8DetectionLoss
import gc


class DiffusionPretrainNetworkWrapper(nn.Module):
    """
    仅做预训练：YOLO 检测损失 + Diffusion 去噪损失（无 GRPO）。
    若 output 中包含 CMAM 先验矩阵 L 与 L_star，则附加 MSE 约束。
    """
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

        # YOLO 损失
        try:
            self.det_crit = self.net.yolo.init_criterion()
        except Exception:
            self.det_crit = v8DetectionLoss(self.net.yolo)

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

        if self.freeze_yolo:
            try:
                for p in self.net.yolo.parameters():
                    p.requires_grad = False
                self.net.yolo.eval()
            except Exception:
                pass

    def forward(self, batch):
        # 仅调用原网络，保证不会进入 GRPO 分支
        output = self.net(batch['inp'], batch)

        # 总损失与标量统计
        base_device = output['detection'].device if isinstance(output.get('detection', None), torch.Tensor) else batch['inp'].device
        loss = torch.zeros(1, device=base_device).squeeze()
        scalar_stats = {}
        image_stats = {}

        # 1) YOLO 检测损失
        if (not self.freeze_yolo) and ('yolo_preds' in output) and (self.det_crit is not None):
            det_loss, det_items = self.det_crit(output['yolo_preds'], batch)
            box_l, cls_l, dfl_l = det_items[0], det_items[1], det_items[2]
            det_weight = float(self.loss_scales.get('det', 1.0))
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
        else:
            scalar_stats.update({'det_loss': torch.tensor(0.0, device=base_device)})

        # 2) Diffusion 去噪损失
        if (not self.freeze_snake) and ('diff_loss' in output):
            diff_weight = float(getattr(cfg, 'diffusion_loss_weight', 1.0))
            loss = loss + diff_weight * output['diff_loss']
            scalar_stats.update({'diff_loss': output['diff_loss'], 'diff_loss_scaled': diff_weight * output['diff_loss']})
            try:
                for k_, v_ in output.items():
                    if isinstance(k_, str) and k_.startswith('diff_loss'):
                        scalar_stats[k_] = v_
            except Exception:
                pass
        else:
            scalar_stats.update({'diff_loss': torch.tensor(0.0, device=base_device)})

        # 4) Combined metric for monitoring: detection loss + diffusion loss (both unscaled)
        try:
            det_l = scalar_stats.get('det_loss', None)
            diff_l = scalar_stats.get('diff_loss', None)
            if isinstance(det_l, torch.Tensor) and isinstance(diff_l, torch.Tensor):
                scalar_stats['det_plus_diff_loss'] = det_l + diff_l
        except Exception:
            pass

        # 3) CMAM 先验（可选）
        if 'L' in output and 'L_star' in output and (not self.freeze_snake):
            L_loss = self.L_crit(output['L'], output['L_star'])
            prior_weight = float(self.loss_scales.get('prior', 1.0))
            loss = loss + prior_weight * L_loss
            scalar_stats.update({'L_loss': L_loss, 'L_loss_scaled': prior_weight * L_loss})
        else:
            scalar_stats.update({'L_loss': torch.tensor(0.0, device=base_device)})

        scalar_stats.update({'loss': loss})
        try:
            for k, v in list(scalar_stats.items()):
                if isinstance(v, torch.Tensor):
                    scalar_stats[k] = v.detach()
        except Exception:
            pass
        if self.training:
            try:
                if isinstance(output, dict):
                    output.pop('yolo_preds', None)
            except Exception:
                pass
            full_output = output
            output = {}
            try:
                del full_output
            except Exception:
                pass
            try:
                gc.collect()
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
        return output, loss, scalar_stats, image_stats
