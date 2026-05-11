import torch.nn as nn
from .evolve import Evolution
from lib.utils import net_utils, data_utils
from lib.utils.snake import snake_decode, snake_gcn_utils
import torch
import torch.nn.functional as F
from lib.config import cfg
import warnings
from lib.networks.YOLOV8.nn.tasks import DetectionModel, attempt_load_one_weight, yaml_model_load
import os
from torchvision import models as tv_models
from torchvision.ops import nms as tv_nms

warnings.filterwarnings("ignore")


# 网络拼接主程序（之前的都是准备）--------------------------------------------------------------------------------------------


def _make_torchvision_resnet(backbone_name, pretrained):
    backbone_name = str(backbone_name).strip().lower()
    if backbone_name not in ("resnet18", "resnet34"):
        raise ValueError(f"Unsupported heatmap backbone: {backbone_name}")

    constructor = getattr(tv_models, backbone_name)
    if not pretrained:
        try:
            return constructor(weights=None)
        except TypeError:
            return constructor(pretrained=False)

    try:
        weight_enum_name = "ResNet18_Weights" if backbone_name == "resnet18" else "ResNet34_Weights"
        weights = getattr(tv_models, weight_enum_name).IMAGENET1K_V1
        return constructor(weights=weights)
    except AttributeError:
        return constructor(pretrained=True)


class HeatmapResNetDetector(nn.Module):
    def __init__(self, num_classes, head_conv=256, backbone_name="resnet18", pretrained=False, feat_channels=64):
        super().__init__()

        backbone = _make_torchvision_resnet(backbone_name, pretrained)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.lat1 = nn.Conv2d(64, feat_channels, kernel_size=1, bias=False)
        self.lat2 = nn.Conv2d(128, feat_channels, kernel_size=1, bias=False)
        self.lat3 = nn.Conv2d(256, feat_channels, kernel_size=1, bias=False)
        self.lat4 = nn.Conv2d(512, feat_channels, kernel_size=1, bias=False)
        self.out = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

        self.ct_head = nn.Sequential(
            nn.Conv2d(feat_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes, kernel_size=1, bias=True),
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(feat_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, bias=True),
        )
        nn.init.constant_(self.ct_head[-1].bias, -2.19)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.lat4(c5)
        p4 = self.lat3(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.lat2(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lat1(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.out(p2)

        ct_hm = net_utils.sigmoid(self.ct_head(feat))
        wh = F.relu(self.wh_head(feat))
        return feat, ct_hm, wh

class Network(nn.Module):
    def __init__(self, num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
        super(Network, self).__init__()

        self.detector_backend = str(getattr(cfg, 'detector_backend', 'yolo') or 'yolo').strip().lower()
        self.down_ratio = float(down_ratio)
        nc = int(getattr(cfg, 'yolo_num_classes', 0) or heads.get('ct_hm', 1))
        self.freeze_snake = bool(getattr(cfg, 'freeze_snake', False))
        self.freeze_yolo = bool(getattr(cfg, 'freeze_yolo', False))
        self.use_p3_features = False
        self.yolo = None
        self.samsnake_dla = None
        self.samsnake_refine = None

        if self.detector_backend == 'yolo':
            # 使用本地 YOLOv8 检测模型替换 DLA，输出检测与特征
            # 选择包含 P2 的结构以获得 stride=4 的特征图，空间大小与原来 DLA 的 136x136 对齐（当输入是 544x544）
            yolo_yaml = 'lib/networks/YOLOV8/cfg/models/v8/yolov8-p2.yaml'
            yolo_cfg = yaml_model_load(yolo_yaml)
            yolo_scale = str(getattr(cfg, 'yolo_model_scale', '') or '').strip().lower()
            if yolo_scale:
                yolo_cfg['scale'] = yolo_scale
            self.yolo = DetectionModel(cfg=yolo_cfg, ch=3, nc=nc, verbose=False)

            try:
                actual_scale = str(getattr(self.yolo, 'yaml', {}).get('scale', ''))
                print(f"[YOLO] yaml={yolo_yaml} requested_scale={yolo_scale or 'default'} actual_scale={actual_scale or 'default'} nc={nc}")
            except Exception:
                pass

            # 加载 YOLO 预训练权重（测试阶段无需加载，统一依赖整体checkpoint；训练可通过开关启用）
            try:
                yolov8_pt = getattr(cfg, 'yolo_pretrained', None)
                load_yolo_pretrained = getattr(cfg, 'load_yolo_pretrained', False)
                if yolov8_pt and load_yolo_pretrained:
                    weights_model, _ = attempt_load_one_weight(yolov8_pt, device=None, inplace=True, fuse=False)
                    self.yolo.load(weights_model)
            except Exception as e:
                print(f"[WARN] Failed to load YOLO pretrained weights: {e}")

            # 将 P2 级别的特征通道压到 64，供 Snake 的 GCN 使用
            # YOLO Detect 头拼接后的通道数为 reg_max*4 + nc（默认 reg_max=16 -> 64）
            in_ch = 64 + nc
            self.cnn_proj = nn.Conv2d(in_ch, 64, kernel_size=1, bias=False)
            self.use_p3_features = bool(
                getattr(cfg, 'v3_4_use_p3_features', False)
                or getattr(cfg, 'v3_7_use_p3_features', False)
                or getattr(cfg, 'v4_use_p3_features', False)
                or getattr(cfg, 'v4_1_use_p3_features', False)
                or getattr(cfg, 'v4_2_use_p3_features', False)
            )
            if self.use_p3_features:
                self.cnn_proj_p3 = nn.Conv2d(in_ch, 64, kernel_size=1, bias=False)
                nn.init.zeros_(self.cnn_proj_p3.weight)
                print("[Snake] P3 feature fusion enabled with zero-init residual.")
        elif self.detector_backend.startswith('heatmap_'):
            self.heatmap_detector = HeatmapResNetDetector(
                num_classes=nc,
                head_conv=head_conv,
                backbone_name=str(getattr(cfg, 'heatmap_backbone', 'resnet18')),
                pretrained=bool(getattr(cfg, 'heatmap_pretrained', False)),
            )
            print(f"[Heatmap] backend={self.detector_backend} backbone={getattr(cfg, 'heatmap_backbone', 'resnet18')} nc={nc}")
        elif self.detector_backend == 'samsnake_fm':
            from SAMSnake.network.backbone.dla import DLASeg
            from .samsnake_refine import SAMSnakeRefine

            dla_layers = num_layers if int(num_layers) > 0 else 34
            self.samsnake_dla = DLASeg(
                f'dla{dla_layers}',
                heads,
                pretrained=bool(getattr(cfg, 'samsnake_dla_pretrained', False)),
                down_ratio=int(down_ratio),
                final_kernel=1,
                last_level=int(getattr(cfg, 'samsnake_dla_last_level', 5)),
                head_conv=head_conv,
                use_dcn=bool(getattr(cfg, 'samsnake_use_dcn', False)),
            )
            self.samsnake_refine = SAMSnakeRefine(
                c_in=64,
                num_points=int(getattr(cfg, 'poly_num', 128)),
                stride=float(getattr(cfg, 'samsnake_refine_stride', 4.0)),
            )
            self.samsnake_freeze_dla = bool(getattr(cfg, 'samsnake_freeze_dla', False))
            if self.samsnake_freeze_dla:
                self.samsnake_dla.eval()
                for p in self.samsnake_dla.parameters():
                    p.requires_grad = False
            print(
                f"[V5.2] backend=samsnake_fm DLA{dla_layers} "
                f"pretrained={getattr(cfg, 'samsnake_dla_pretrained', False)} "
                f"use_dcn={getattr(cfg, 'samsnake_use_dcn', False)} "
                f"freeze_dla={getattr(cfg, 'samsnake_freeze_dla', False)}"
            )
        else:
            raise ValueError(f"Unsupported detector backend: {self.detector_backend}")

        # Choose between original evolution and diffusion evolution
        use_diffusion = getattr(cfg, 'use_diffusion_evolution', False)

        if use_diffusion:
            # 延迟导入，避免与 diffusion.evolution -> snake.snake 的循环依赖
            from lib.networks.diffusion import make_evolution
            use_flow_matching = bool(
                getattr(cfg, 'use_flow_matching', False)
                or getattr(cfg, 'use_dit_v3_6', False)
            )
            self.gcn = make_evolution(
                use_grpo=getattr(cfg, 'use_grpo', False),
                state_dim=128,
                feature_dim=64,
                num_points=128,
                num_timesteps=getattr(cfg, 'diffusion_timesteps', 1000),
                use_ddim_inference=getattr(cfg, 'use_ddim_inference', True),
                loss_weight=getattr(cfg, 'diffusion_loss_weight', 1.0),
                loss_type=getattr(cfg, 'diffusion_loss_type', 'adaptive'),
                # DiT 去噪器参数
                use_dit_denoiser=getattr(cfg, 'use_dit_denoiser', False),
                use_flow_matching=use_flow_matching,
                flow_ode_steps=getattr(cfg, 'flow_ode_steps', 10),
                dit_num_layers=getattr(cfg, 'dit_num_layers', 6),
                dit_num_heads=getattr(cfg, 'dit_num_heads', 8),
                dit_state_dim=getattr(cfg, 'dit_state_dim', 256),
            )
            self.diffusion_loss_fn = None
        else:
            self.gcn = Evolution()
            self.diffusion_loss_fn = None

        # 冻结 Snake 相关模块（只训练 YOLO）
        if self.freeze_snake:
            if self.detector_backend == 'yolo':
                modules_to_freeze = [self.gcn, self.cnn_proj] if not use_diffusion else [self.cnn_proj]
                if hasattr(self, 'cnn_proj_p3'):
                    modules_to_freeze.append(self.cnn_proj_p3)
            elif self.detector_backend == 'samsnake_fm':
                modules_to_freeze = [self.gcn]
                if self.samsnake_refine is not None:
                    modules_to_freeze.append(self.samsnake_refine)
            else:
                modules_to_freeze = [self.gcn]
            for m in modules_to_freeze:
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
            # 确保 YOLO 可训练（除非显式要求冻结）
            if self.yolo is not None:
                for p in self.yolo.parameters():
                    p.requires_grad = not self.freeze_yolo

        # 单独的 YOLO 冻结开关
        if self.freeze_yolo and self.yolo is not None:
            for p in self.yolo.parameters():
                p.requires_grad = False

    # 注意：自定义 NMS/IoU 函数已移除，统一使用 YOLOv8 的 non_max_suppression，避免训练/测试不一致与重复实现。

    def decode_detection_from_yolo(self, yolo_y, h, w):
        # yolo_y: (B, no, HW) where first 4*reg_max decoded to xywh already by head, but here yolo_y stores [xywh, cls_logits]
        # 将其转置为 (B, HW, C)
        y = yolo_y.permute(0, 2, 1).contiguous()
        xywh = y[..., :4]
        cls_logits = y[..., 4:]
        # 分数与类别
        cls_prob = cls_logits.sigmoid()
        score, cls_idx = cls_prob.max(dim=-1, keepdim=True)
        # 转 xyxy 并裁剪
        x, y_c, w_box, h_box = xywh.unbind(-1)
        x1 = x - w_box / 2
        y1 = y_c - h_box / 2
        x2 = x + w_box / 2
        y2 = y_c + h_box / 2
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        boxes = data_utils.clip_to_image(boxes, h, w)
        detection = torch.cat([boxes, score, cls_idx.float()], dim=-1)
        return detection

    def decode_detection_from_heatmap(self, ct_hm, wh):
        max_det = max(int(getattr(cfg, 'det_max_det', 100)) * 4, 100)
        class_offset = int(getattr(cfg, 'heatmap_class_offset', 0))
        ct_hm_decode = ct_hm[:, class_offset:, :, :] if class_offset > 0 else ct_hm
        if ct_hm_decode.size(1) <= 0:
            raise ValueError(f"heatmap_class_offset={class_offset} removes all heatmap channels")
        ct, detection = snake_decode.decode_ct_hm(ct_hm_decode, wh, K=max_det)
        h, w = ct_hm.shape[-2:]
        h_img, w_img = int(round(h * self.down_ratio)), int(round(w * self.down_ratio))

        ct = ct * self.down_ratio
        detection = detection.clone()
        detection[..., :4] = detection[..., :4] * self.down_ratio
        detection[..., :4] = data_utils.clip_to_image(detection[..., :4], h_img, w_img)
        return ct, detection

    def filter_detection_candidates(self, detection):
        conf_thres = float(getattr(cfg, 'det_conf_thresh', 0.20))
        iou_thres = float(getattr(cfg, 'det_iou_thresh', 0.30))
        max_det = int(getattr(cfg, 'det_max_det', 300))
        per_class = bool(getattr(cfg, 'per_class_nms', True))
        use_nms = bool(getattr(cfg, 'use_nms_for_snake', True))

        filtered = []
        for det_b in detection:
            keep = det_b[:, 4] >= conf_thres
            det_b = det_b[keep]
            if det_b.numel() == 0:
                filtered.append(det_b.new_zeros((0, 6)))
                continue

            det_b = det_b[torch.argsort(det_b[:, 4], descending=True)]
            if use_nms:
                if per_class:
                    kept_idx = []
                    for cls_id in det_b[:, 5].unique():
                        cls_mask = det_b[:, 5] == cls_id
                        cls_inds = cls_mask.nonzero(as_tuple=False).squeeze(1)
                        cls_keep = tv_nms(det_b[cls_inds, :4], det_b[cls_inds, 4], iou_thres)
                        kept_idx.append(cls_inds[cls_keep])
                    keep_idx = torch.cat(kept_idx, dim=0) if kept_idx else det_b.new_zeros((0,), dtype=torch.long)
                else:
                    keep_idx = tv_nms(det_b[:, :4], det_b[:, 4], iou_thres)
                if keep_idx.numel() > 0:
                    keep_idx = keep_idx[torch.argsort(det_b[keep_idx, 4], descending=True)]
                    det_b = det_b[keep_idx]
            filtered.append(det_b[:max_det])

        max_len = max((d.size(0) for d in filtered), default=0)
        if max_len == 0:
            return detection.new_zeros((detection.size(0), 0, 6))

        packed = detection.new_zeros((detection.size(0), max_len, 6))
        for b, det_b in enumerate(filtered):
            if det_b.size(0) > 0:
                packed[b, :det_b.size(0)] = det_b
        return packed

    def use_gt_detection(self, output, batch):
        ct_01 = batch['ct_01'].bool()
        batch_size = int(ct_01.size(0))
        feat_h, feat_w = output.get('feat_hw', (0, 0))
        if not feat_h or not feat_w:
            feat_h, feat_w = batch['ct_hm'].shape[-2:]

        counts = ct_01.long().sum(dim=1)
        max_len = int(counts.max().item()) if counts.numel() > 0 else 0
        device = batch['inp'].device
        dtype = batch['inp'].dtype
        if max_len == 0:
            output['ct'] = torch.zeros((batch_size, 0, 2), device=device, dtype=dtype)
            output['detection'] = torch.zeros((batch_size, 0, 6), device=device, dtype=dtype)
            return output

        packed_ct = torch.zeros((batch_size, max_len, 2), device=device, dtype=dtype)
        packed_det = torch.zeros((batch_size, max_len, 6), device=device, dtype=dtype)
        dr = float(self.down_ratio)

        for b in range(batch_size):
            keep = ct_01[b]
            n = int(keep.sum().item())
            if n == 0:
                continue
            ct_ind = batch['ct_ind'][b, keep].to(device=device)
            xs = (ct_ind % int(feat_w)).to(dtype=dtype)
            ys = (ct_ind // int(feat_w)).to(dtype=dtype)
            ct_feat = torch.stack([xs, ys], dim=1)
            wh_feat = batch['wh'][b, keep].to(device=device, dtype=dtype)
            bboxes_feat = torch.cat([
                xs[:, None] - wh_feat[..., 0:1] / 2,
                ys[:, None] - wh_feat[..., 1:2] / 2,
                xs[:, None] + wh_feat[..., 0:1] / 2,
                ys[:, None] + wh_feat[..., 1:2] / 2,
            ], dim=1)
            bboxes = bboxes_feat * dr
            bboxes = data_utils.clip_to_image(
                bboxes,
                int(round(feat_h * dr)),
                int(round(feat_w * dr)),
            )
            score = torch.ones((n, 1), device=device, dtype=dtype)
            ct_cls = batch['ct_cls'][b, keep].to(device=device, dtype=dtype).view(n, 1)
            if self.detector_backend.startswith('heatmap_'):
                ct_cls = ct_cls - float(getattr(cfg, 'heatmap_class_offset', 0))
            packed_ct[b, :n] = ct_feat * dr
            packed_det[b, :n] = torch.cat([bboxes, score, ct_cls], dim=1)

        output['ct'] = packed_ct
        output['detection'] = packed_det

        return output

    def forward(self, x, batch=None):
        if self.detector_backend == 'samsnake_fm':
            if bool(getattr(cfg, 'samsnake_freeze_dla', False)):
                self.samsnake_dla.eval()
                with torch.no_grad():
                    dla_out, cnn_feature = self.samsnake_dla(x)
            else:
                dla_out, cnn_feature = self.samsnake_dla(x)
            h, w = cnn_feature.size(2), cnn_feature.size(3)
            ct_hm = net_utils.sigmoid(dla_out['ct_hm']) if 'ct_hm' in dla_out else None
            wh = F.relu(dla_out['wh']) if 'wh' in dla_out else None
            if ct_hm is not None and wh is not None:
                ct, raw_det = self.decode_detection_from_heatmap(ct_hm, wh)
                detection = self.filter_detection_candidates(raw_det)
            else:
                ct = torch.zeros((x.size(0), 0, 2), device=x.device, dtype=x.dtype)
                detection = torch.zeros((x.size(0), 0, 6), device=x.device, dtype=x.dtype)

            output = {
                'ct_hm': ct_hm,
                'wh': wh,
                'ct': ct,
                'detection': detection,
                'feat_hw': (h, w),
                'cnn_feature': cnn_feature,
            }
            if getattr(cfg, 'use_gt_det', False):
                self.use_gt_detection(output, batch)

            if batch is not None:
                from lib.utils.snake.sam_init import attach_sam_testing_init, sam_init_enabled
                if sam_init_enabled():
                    output = attach_sam_testing_init(output, batch, device=x.device)

            if (
                self.samsnake_refine is not None
                and bool(getattr(cfg, 'v5_2_use_samsnake_refine', True))
                and torch.is_tensor(output.get('sam_i_it_py', None))
            ):
                raw_init = output['sam_i_it_py']
                output['sam_raw_i_it_py'] = raw_init
                centers = output.get('sam_ct', raw_init.mean(dim=1))
                py_ind = output.get(
                    'sam_py_ind',
                    torch.zeros((raw_init.size(0),), dtype=torch.long, device=raw_init.device),
                )
                coarse = self.samsnake_refine(
                    cnn_feature,
                    centers,
                    raw_init,
                    py_ind,
                    ignore=bool(getattr(cfg, 'samsnake_refine_ignore', False)),
                )
                output['samsnake_coarse_i_it_py'] = coarse
                output['sam_i_it_py'] = coarse
                output['sam_c_it_py'] = snake_gcn_utils.img_poly_to_can_poly(coarse)

            if not self.freeze_snake:
                output = self.gcn(output, cnn_feature, batch)
            output['feat_hw'] = (h, w)
            output['cnn_feature'] = cnn_feature
            return output

        if self.detector_backend.startswith('heatmap_'):
            cnn_feature, ct_hm, wh = self.heatmap_detector(x)
            h, w = cnn_feature.size(2), cnn_feature.size(3)
            ct, raw_det = self.decode_detection_from_heatmap(ct_hm, wh)
            detection = self.filter_detection_candidates(raw_det)

            output = {
                'ct_hm': ct_hm,
                'wh': wh,
                'ct': ct,
                'detection': detection,
                'feat_hw': (h, w),
                'cnn_feature': cnn_feature,
            }
            if getattr(cfg, 'use_gt_det', False):
                self.use_gt_detection(output, batch)
            if (not self.training) and str(getattr(cfg, 'contour_init_method', 'octagon')).strip().lower() == 'sam':
                from lib.utils.snake.sam_init import attach_sam_testing_init
                output = attach_sam_testing_init(output, batch, device=x.device)
            if not self.freeze_snake:
                output = self.gcn(output, cnn_feature, batch)
            output['feat_hw'] = (h, w)
            output['cnn_feature'] = cnn_feature
            return output

        # YOLO 前向：返回 (y, feats)，其中 feats 为多尺度 head 特征列表
        yolo_out = self.yolo(x)
        # Detect 头推理默认返回 (y, feats)。y 是张量，feats 是多尺度特征列表
        if isinstance(yolo_out, tuple) and len(yolo_out) >= 2:
            yolo_y, yolo_feats = yolo_out[0], yolo_out[1]
        else:
            # 兼容返回单个张量的情况（导出/特殊路径）
            yolo_y, yolo_feats = yolo_out, []

        # 选择 P2 特征（最细一层，对应 stride=4，索引取 0）并压到 64 通道
        p2 = yolo_feats[0] if isinstance(yolo_feats, (list, tuple)) and len(yolo_feats) > 0 else None
        if p2 is None:
            raise RuntimeError("YOLO head features are not available; expected a list with P2 at index 0.")
        cnn_feature = self.cnn_proj(p2)
        if self.use_p3_features:
            p3 = yolo_feats[1] if isinstance(yolo_feats, (list, tuple)) and len(yolo_feats) > 1 else None
            if p3 is not None:
                p3_up = F.interpolate(p3, size=p2.shape[-2:], mode='bilinear', align_corners=False)
                cnn_feature = cnn_feature + self.cnn_proj_p3(p3_up)

        # 从 YOLO 输出构建 detection (B, N, 6) => [x1,y1,x2,y2,score,cls]
        # 并按配置执行阈值+NMS，确保训练/测试阶段一致地给 Snake 提供精简候选
        h, w = cnn_feature.size(2), cnn_feature.size(3) # 这个h,w是特征图的尺寸，相比图像尺寸缩小了4倍
        h_img, w_img = h*4, w*4
        raw_det = self.decode_detection_from_yolo(yolo_y, h_img, w_img)  # [B, HW, 6]

        use_nms = getattr(cfg, 'use_nms_for_snake', True)
        conf_thres = float(getattr(cfg, 'det_conf_thresh', 0.20))
        iou_thres = float(getattr(cfg, 'det_iou_thresh', 0.30))
        max_det = int(getattr(cfg, 'det_max_det', 300))
        per_class = bool(getattr(cfg, 'per_class_nms', True))

        # 直接使用 YOLO 内部 NMS，保证与推理后处理完全一致
        if use_nms:
            from lib.networks.YOLOV8.utils.ops import non_max_suppression

            # 从 yolo 输出构建 NMS 所需的 prediction 张量: (B, 4+nc, HW)
            y = yolo_y.permute(0, 2, 1).contiguous()  # (B, HW, 4+nc)
            xywh = y[..., :4]
            cls_logits = y[..., 4:]
            cls_prob = cls_logits.sigmoid()
            # 转为 xyxy 并裁剪
            x_c, y_c, w_box, h_box = xywh.unbind(-1)
            x1 = x_c - w_box / 2
            y1 = y_c - h_box / 2
            x2 = x_c + w_box / 2
            y2 = y_c + h_box / 2
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            boxes = data_utils.clip_to_image(boxes, h_img, w_img)

            pred = torch.cat([boxes, cls_prob], dim=-1).permute(0, 2, 1).contiguous()  # (B, 4+nc, HW)

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

            # 打包为 [B, M, 6]
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

        # 构造与下游一致的 output 字典
        output = {}
        output.update({'detection': detection})
        # 记录特征图尺寸，供可视化/坐标缩放使用（不再依赖 ct_hm）
        output['feat_hw'] = (h, w)
        output['cnn_feature'] = cnn_feature

        # 训练时可选择使用 GT 框替换
        if getattr(cfg, 'use_gt_det', False):
            self.use_gt_detection(output, batch)

        if (not self.training) and str(getattr(cfg, 'contour_init_method', 'octagon')).strip().lower() == 'sam':
            from lib.utils.snake.sam_init import attach_sam_testing_init
            output = attach_sam_testing_init(output, batch, device=x.device)

        # 传入 Snake 进行演化
        if not self.freeze_snake:
            output = self.gcn(output, cnn_feature, batch)
        output['feat_hw'] = (h, w)
        output['cnn_feature'] = cnn_feature
        # print("output",output.keys())
        # print("py_pred",len(output['py_pred']))
        # print("py_pred",output['py_pred'][-1].shape)
        # print("detection",output['detection'].shape)

        # 暴露 YOLO 原始预测供损失使用
        output['yolo_preds'] = (yolo_y, yolo_feats)

        return output


def get_network(num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
    network = Network(num_layers, heads, head_conv, down_ratio, det_dir)
    return network
