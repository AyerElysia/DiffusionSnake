import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import os
import numpy as np
import copy
from lib.config import cfg
from lib.networks.YOLOV8.utils.loss import v8DetectionLoss
from lib.utils.snake import snake_gcn_utils, snake_config, snake_decode
from lib.train.rewards.region_reward import compute_region_reward
from lib.networks.YOLOV8.utils.ops import non_max_suppression
from lib.networks.diffusion.ddpm_with_logprob import ddpm_step_with_logprob


class DiffusionGRPONetworkWrapper(nn.Module):
    """
    GRPO 训练专用：包含 YOLO 检测损失、Diffusion 去噪损失与 GRPO 策略损失。
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.disable_grpo = os.environ.get('DISABLE_GRPO', '1') != '0'
        raw_use_grpo = bool(getattr(cfg, 'use_grpo', False)) or bool(getattr(cfg, 'use_grpo_kl', False)) or bool(getattr(getattr(cfg, 'train', cfg), 'use_grpo', False))
        self.allow_grpo = raw_use_grpo and (not self.disable_grpo)

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

        # 冻结开关
        self.freeze_snake = bool(getattr(cfg, 'freeze_snake', False))
        self.freeze_yolo = bool(getattr(cfg, 'freeze_yolo', False))

        if self.freeze_yolo:
            try:
                for p in self.net.yolo.parameters():
                    p.requires_grad = False
                self.net.yolo.eval()
            except Exception:
                pass

        # GRPO KL & PPO settings (Flow-GRPO style names)
        beta_train = None
        try:
            beta_train = getattr(cfg.train, 'beta')
        except Exception:
            beta_train = None
        if beta_train is not None and float(beta_train) > 0.0:
            self.use_kl = True
            self.kl_beta = float(beta_train)
        else:
            self.use_kl = bool(getattr(cfg, 'use_grpo_kl', False))
            self.kl_beta = float(getattr(cfg, 'grpo_kl_beta', 0.0))
        try:
            self.clip_range = float(getattr(cfg.train, 'clip_range'))
        except Exception:
            self.clip_range = float(getattr(cfg, 'grpo_clip_range', 0.2))
        try:
            self.adv_clip_max = float(getattr(cfg.train, 'adv_clip_max'))
        except Exception:
            self.adv_clip_max = float(getattr(cfg, 'grpo_adv_clip_max', 5.0))
        self.ref_gcn = None  # lazy snapshot after weights are loaded

        # 若不允许 GRPO，则移除 gcn sampler，避免 logprob/采样
        if not self.allow_grpo:
            try:
                if hasattr(self.net, 'gcn') and hasattr(self.net.gcn, 'sampler'):
                    self.net.gcn.sampler = None
            except Exception:
                pass

    def forward(self, batch):
        use_grpo = self.allow_grpo
        if not hasattr(self, '_log_use_grpo_once'):
            try:
                print(f"[DEBUG] DiffusionGRPONetworkWrapper use_grpo={use_grpo}, freeze_yolo={self.freeze_yolo}, freeze_snake={self.freeze_snake}, disable_grpo_env={self.disable_grpo}")
            except Exception:
                pass
            self._log_use_grpo_once = True
        if use_grpo:
            # 惰性创建参考策略（冻结的GCN副本），仅在启用KL时使用
            if self.use_kl and self.ref_gcn is None:
                try:
                    self.ref_gcn = copy.deepcopy(self.net.gcn).cuda()  # 深拷贝GCN到GPU
                    for p in self.ref_gcn.parameters():
                        p.requires_grad = False
                    self.ref_gcn.eval()
                except Exception:
                    self.ref_gcn = None
            x = batch['inp']
            # 强制YOLO在eval+no_grad下运行，避免GRPO期间BN统计被修改
            if self.freeze_yolo:
                was_training = bool(self.net.yolo.training)
                self.net.yolo.eval()
                ctx = torch.no_grad()
            else:
                was_training = False
                ctx = contextlib.nullcontext()
            with ctx:
                yolo_out = self.net.yolo(x)
            if self.freeze_yolo and was_training:
                self.net.yolo.train()
            if isinstance(yolo_out, tuple) and len(yolo_out) >= 2:
                yolo_y, yolo_feats = yolo_out[0], yolo_out[1]
            else:
                yolo_y, yolo_feats = yolo_out, []
            p2 = yolo_feats[0] if isinstance(yolo_feats, (list, tuple)) and len(yolo_feats) > 0 else None
            if p2 is None:
                raise RuntimeError("YOLO head features are not available; expected a list with P2 at index 0.")
            if self.freeze_yolo:
                p2 = p2.detach()

            cnn_feature = self.net.cnn_proj(p2)
            h, w = cnn_feature.size(2), cnn_feature.size(3)
            h_img, w_img = h * int(snake_config.down_ratio), w * int(snake_config.down_ratio)
            raw_det = self.net.decode_detection_from_yolo(yolo_y, h_img, w_img)
            use_nms = bool(getattr(cfg, 'use_nms_for_snake', True))
            conf_thres = float(getattr(cfg, 'det_conf_thresh', 0.20))
            iou_thres = float(getattr(cfg, 'det_iou_thresh', 0.30))
            max_det = int(getattr(cfg, 'det_max_det', 300))
            per_class = bool(getattr(cfg, 'per_class_nms', True))
            if use_nms:
                y = yolo_y.permute(0, 2, 1).contiguous()
                xywh = y[..., :4]
                cls_logits = y[..., 4:]
                cls_prob = cls_logits.sigmoid()
                x_c, y_c, w_box, h_box = xywh.unbind(-1)
                x1 = x_c - w_box / 2
                y1 = y_c - h_box / 2
                x2 = x_c + w_box / 2
                y2 = y_c + h_box / 2
                boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                boxes = boxes.clamp(min=0)
                pred = torch.cat([boxes, cls_prob], dim=-1).permute(0, 2, 1).contiguous()
                nms_out = non_max_suppression(
                    pred,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                    classes=None,
                    agnostic=not per_class,
                    multi_label=False,
                    labels=(),
                    max_det=max_det,
                    nc=cls_prob.shape[-1],
                    in_place=True,
                    rotated=True,
                )
                max_len = max((d.size(0) for d in nms_out), default=0)
                if max_len == 0:
                    detection = raw_det.new_zeros((raw_det.size(0), 0, 6))
                else:
                    detection = raw_det.new_zeros((raw_det.size(0), max_len, 6))
                    for b, det_b in enumerate(nms_out):
                        if det_b is not None and det_b.size(0) > 0:
                            detection[b, :det_b.size(0)] = det_b[:, :6]
            else:
                detection = raw_det
            output = {'detection': detection, 'feat_hw': (h, w)}
            if getattr(cfg, 'use_gt_det', False):
                self.net.use_gt_detection(output, batch)
        else:
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

        # 2) Diffusion 去噪损失 或 GRPO 策略损失
        if use_grpo and (not self.freeze_snake):
            with torch.no_grad():
                init = snake_gcn_utils.prepare_training(output, batch)
            i_gt_4py = init['i_gt_4py']
            i_gt_py = init['i_gt_py']
            py_ind = init['py_ind']
            x_min = i_gt_4py[..., 0].min(dim=1)[0]
            y_min = i_gt_4py[..., 1].min(dim=1)[0]
            x_max = i_gt_4py[..., 0].max(dim=1)[0]
            y_max = i_gt_4py[..., 1].max(dim=1)[0]
            gt_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1).unsqueeze(0)
            gt_rect4 = snake_decode.get_box(gt_boxes)[0]
            i_init_train_py = snake_gcn_utils.uniform_upsample(
                gt_rect4.unsqueeze(0), snake_config.poly_num
            )[0]
            c_init_train_py = snake_gcn_utils.img_poly_to_can_poly(i_init_train_py)

            def _signed_area(poly: torch.Tensor) -> torch.Tensor:
                x = poly[..., 0]
                y = poly[..., 1]
                x1 = torch.roll(x, shifts=-1, dims=1)
                y1 = torch.roll(y, shifts=-1, dims=1)
                return 0.5 * torch.sum(x * y1 - x1 * y, dim=1)

            area_init = _signed_area(i_init_train_py)
            area_gt = _signed_area(i_gt_py)
            orient_mismatch = ((area_init >= 0) ^ (area_gt >= 0))
            if orient_mismatch.any():
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

            i_it_py = i_init_train_py
            c_it_py = c_init_train_py
            if bool(getattr(cfg, 'grpo_first_contour_only', False)) and i_it_py.size(0) > 0:
                with torch.no_grad():
                    uniq = []
                    keep_idx = []
                    for idx in range(py_ind.numel()):
                        b = int(py_ind[idx].item())
                        if b not in uniq:
                            uniq.append(b)
                            keep_idx.append(idx)
                    keep = torch.as_tensor(keep_idx, device=py_ind.device, dtype=torch.long)
                i_it_py = i_it_py.index_select(0, keep)
                c_it_py = c_it_py.index_select(0, keep)
                i_gt_py = i_gt_py.index_select(0, keep)
                py_ind = py_ind.index_select(0, keep)
            if i_it_py.size(0) == 0:
                zero_term = cnn_feature.sum() * 0.0
                loss = loss + zero_term
                scalar_stats.update({'grpo_loss': torch.tensor(0.0, device=base_device), 'grpo_reward_mean': torch.tensor(0.0, device=base_device), 'grpo_reward_std': torch.tensor(0.0, device=base_device)})
            else:
                steps = int(getattr(cfg, 'grpo_steps', 20))
                k = int(getattr(cfg, 'grpo_k', 4))
                window_size = int(getattr(cfg, 'grpo_window_size', 6))
                window_range = getattr(cfg, 'grpo_window_range', (2, 12))
                w1 = float(getattr(cfg, 'reward_w_region', 1.0))

                total_policy_loss = torch.zeros((), device=base_device)

                rewards_list = []
                old_logs_list = []
                trajs_list = []
                for _ in range(k):
                    with torch.no_grad():
                        ret = self.net.gcn.sampler.sample_with_logprob(
                            cnn_feature, i_it_py, c_it_py, py_ind,
                            steps=steps, window_size=window_size, window_range=window_range
                        )
                    if isinstance(ret['log_probs'], list) and len(ret['log_probs']) > 0:
                        lp_steps = torch.stack(ret['log_probs'], dim=0).transpose(0, 1).contiguous()
                        old_logs_list.append(lp_steps.detach())
                        disp = ret['disp']
                        rew = compute_region_reward(i_it_py, disp, i_gt_py, H=h, W=w, w1=w1)
                        rewards_list.append(rew.detach())
                        trajs_list.append((ret['timesteps'], ret['x_ts'], ret['x_prevs']))
                if len(rewards_list) == 0:
                    loss = loss + (cnn_feature.sum() * 0.0)
                    scalar_stats.update({'grpo_loss': torch.tensor(0.0, device=base_device), 'grpo_reward_mean': torch.tensor(0.0, device=base_device), 'grpo_reward_std': torch.tensor(0.0, device=base_device)})
                else:
                    rewards_k = torch.stack(rewards_list, dim=0)
                    adv_k = rewards_k - rewards_k.mean(dim=0, keepdim=True)
                    scalar_stats.update({
                        'grpo_reward_mean': rewards_k.mean(),
                        'grpo_reward_std': rewards_k.std(unbiased=False)
                    })

                    clip_eps = self.clip_range
                    adv_clip_max = self.adv_clip_max

                    pl_terms = []
                    kl_terms = []
                    num_traj = len(trajs_list)
                    for j in range(num_traj):
                        t_seq, x_ts, x_prevs = trajs_list[j]
                        old_logs_j = old_logs_list[j]
                        step_losses = []
                        kl_sum = torch.zeros((), device=base_device)
                        for s_idx in range(len(t_seq)):
                            t_scalar = t_seq[s_idx]
                            t_batch = torch.full((i_it_py.size(0),), int(t_scalar), device=base_device, dtype=torch.long)
                            x_t = x_ts[s_idx]
                            x_prev = x_prevs[s_idx]
                            eps_cur, _ = self.net.gcn.predict_eps(cnn_feature, i_it_py, c_it_py, py_ind, x_t, t_batch)
                            _, lp_cur, mean_cur, std_cur = ddpm_step_with_logprob(self.net.gcn.scheduler, eps_cur, t_scalar, x_t, prev_sample=x_prev)
                            ratio = torch.exp(lp_cur - old_logs_j[:, s_idx].to(base_device))
                            adv_j = adv_k[j].clamp(-adv_clip_max, adv_clip_max).detach()
                            unclipped = -adv_j * ratio
                            clipped = -adv_j * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                            step_losses.append(torch.maximum(unclipped, clipped).mean())
                            if self.use_kl and (self.ref_gcn is not None) and self.kl_beta > 0.0:
                                with torch.no_grad():
                                    eps_ref, _ = self.ref_gcn.predict_eps(cnn_feature, i_it_py, c_it_py, py_ind, x_t, t_batch)
                                _, _, mean_ref, _ = ddpm_step_with_logprob(self.net.gcn.scheduler, eps_ref, t_scalar, x_t, prev_sample=x_prev)
                                var = (std_cur ** 2).clamp_min(1e-12)
                                kl_step = ((mean_cur - mean_ref) ** 2) / (2.0 * var)
                                kl_sum = kl_sum + kl_step.mean()
                        pl_terms.append(torch.stack(step_losses).mean())
                        if self.use_kl and self.kl_beta > 0.0:
                            kl_terms.append(kl_sum / max(len(t_seq), 1))
                    if len(pl_terms) > 0:
                        total_policy_loss = torch.stack(pl_terms).mean()
                        if self.use_kl and self.kl_beta > 0.0 and len(kl_terms) > 0:
                            total_policy_loss = total_policy_loss + self.kl_beta * torch.stack(kl_terms).mean()
                        loss = loss + total_policy_loss
                        scalar_stats.update({'grpo_loss': total_policy_loss})
                    else:
                        loss = loss + (cnn_feature.sum() * 0.0)
                        scalar_stats.update({'grpo_loss': torch.tensor(0.0, device=base_device)})
        elif (not self.freeze_snake) and ('diff_loss' in output):
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

        # Combined metric for monitoring on wandb/json: detection + diffusion (or detection + grpo)
        try:
            det_l = scalar_stats.get('det_loss', None)
            diff_l = scalar_stats.get('diff_loss', None)
            grpo_l = scalar_stats.get('grpo_loss', None)
            if isinstance(det_l, torch.Tensor) and isinstance(diff_l, torch.Tensor):
                scalar_stats['det_plus_diff_loss'] = det_l + diff_l
            elif isinstance(det_l, torch.Tensor) and isinstance(grpo_l, torch.Tensor):
                scalar_stats['det_plus_diff_loss'] = det_l + grpo_l
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
        return output, loss, scalar_stats, image_stats
