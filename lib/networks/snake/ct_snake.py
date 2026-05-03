import torch.nn as nn
from .evolve import Evolution
from lib.utils import net_utils, data_utils
from lib.utils.snake import snake_decode
import torch
import torch.nn.functional as F
from lib.config import cfg
import warnings
from lib.networks.YOLOV8.nn.tasks import DetectionModel, attempt_load_one_weight, yaml_model_load
import os

warnings.filterwarnings("ignore")


# 网络拼接主程序（之前的都是准备）--------------------------------------------------------------------------------------------

class Network(nn.Module):
    def __init__(self, num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
        super(Network, self).__init__()

        # 使用本地 YOLOv8 检测模型替换 DLA，输出检测与特征
        # 选择包含 P2 的结构以获得 stride=4 的特征图，空间大小与原来 DLA 的 136x136 对齐（当输入是 544x544）
        yolo_yaml = 'lib/networks/YOLOV8/cfg/models/v8/yolov8-p2.yaml'
        nc = int(getattr(cfg, 'yolo_num_classes', 0) or heads.get('ct_hm', 1))
        yolo_cfg = yaml_model_load(yolo_yaml)
        yolo_scale = str(getattr(cfg, 'yolo_model_scale', '') or '').strip().lower()
        if yolo_scale:
            yolo_cfg['scale'] = yolo_scale
        self.yolo = DetectionModel(cfg=yolo_cfg, ch=3, nc=nc, verbose=False)
        self.freeze_snake = bool(getattr(cfg, 'freeze_snake', False))
        self.freeze_yolo = bool(getattr(cfg, 'freeze_yolo', False))

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
            else:
                # 跳过单独加载YOLO预训练，避免覆盖整体checkpoint或增加测试耗时
                pass
        except Exception as e:
            print(f"[WARN] Failed to load YOLO pretrained weights: {e}")

        # 将 P2 级别的特征通道压到 64，供 Snake 的 GCN 使用
        # YOLO Detect 头拼接后的通道数为 reg_max*4 + nc（默认 reg_max=16 -> 64）
        in_ch = 64 + nc
        self.cnn_proj = nn.Conv2d(in_ch, 64, kernel_size=1, bias=False)
        self.use_p3_features = bool(
            getattr(cfg, 'v3_4_use_p3_features', False)
            or getattr(cfg, 'v3_7_use_p3_features', False)
        )
        if self.use_p3_features:
            self.cnn_proj_p3 = nn.Conv2d(in_ch, 64, kernel_size=1, bias=False)
            nn.init.zeros_(self.cnn_proj_p3.weight)
            print("[Snake] P3 feature fusion enabled with zero-init residual.")

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
            modules_to_freeze = [self.gcn, self.cnn_proj] if not use_diffusion else [self.cnn_proj]
            if hasattr(self, 'cnn_proj_p3'):
                modules_to_freeze.append(self.cnn_proj_p3)
            for m in modules_to_freeze:
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
            # 确保 YOLO 可训练（除非显式要求冻结）
            for p in self.yolo.parameters():
                p.requires_grad = not self.freeze_yolo

        # 单独的 YOLO 冻结开关
        if self.freeze_yolo:
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

    def use_gt_detection(self, output, batch):
        # 使用输入图像尺寸作为参考，避免依赖 DLA 的 ct_hm
        _, _, height, width = batch['inp'].size()
        ct_01 = batch['ct_01'].byte()

        ct_ind = batch['ct_ind'][ct_01]
        xs, ys = ct_ind % width, ct_ind // width
        xs, ys = xs[:, None].float(), ys[:, None].float()
        ct = torch.cat([xs, ys], dim=1)

        wh = batch['wh'][ct_01]
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=1)
        score = torch.ones([len(bboxes)]).to(bboxes)[:, None]
        ct_cls = batch['ct_cls'][ct_01].float()[:, None]
        detection = torch.cat([bboxes, score, ct_cls], dim=1)

        output['ct'] = ct[None]
        output['detection'] = detection[None]

        return output

    def forward(self, x, batch=None):
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

        # 训练时可选择使用 GT 框替换
        if getattr(cfg, 'use_gt_det', False):
            self.use_gt_detection(output, batch)

        # 传入 Snake 进行演化
        if not self.freeze_snake:
            output = self.gcn(output, cnn_feature, batch)
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
