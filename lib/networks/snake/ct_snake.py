import torch.nn as nn
from .evolve import Evolution
from lib.utils import net_utils, data_utils
from lib.utils.snake import snake_decode
import torch
from lib.config import cfg
import warnings
from lib.networks.YOLOV8.nn.tasks import DetectionModel, attempt_load_one_weight
import os
import time
import numpy as np
import cv2
# from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")
# from lib.utils.snake import snake_gcn_utils
# from torchvision.utils import save_image
# import sys
# import cv2
# from lib.networks.darnet.drn_contours import DRNContours
# from math import sqrt
# from lib.networks.classifier.wave_mlp import WaveMLP
# #from lib.networks.snake.evolve import Evolution


# 网络拼接主程序（之前的都是准备）--------------------------------------------------------------------------------------------

class Network(nn.Module):
    def __init__(self, num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
        super(Network, self).__init__()

        # 使用本地 YOLOv8 检测模型替换 DLA，输出检测与特征
        # 选择包含 P2 的结构以获得 stride=4 的特征图，空间大小与原来 DLA 的 136x136 对齐（当输入是 544x544）
        yolo_yaml = 'lib/networks/YOLOV8/cfg/models/v8/yolov8-p2.yaml'
        nc = heads.get('ct_hm', 1)
        self.yolo = DetectionModel(cfg=yolo_yaml, ch=3, nc=nc, verbose=False)
        self.freeze_snake = bool(getattr(cfg, 'freeze_snake', False))
        self.freeze_yolo = bool(getattr(cfg, 'freeze_yolo', False))

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

        # Choose between original evolution and diffusion evolution
        use_diffusion = getattr(cfg, 'use_diffusion_evolution', False)

        if use_diffusion:
            # 延迟导入，避免与 diffusion.evolution -> snake.snake 的循环依赖
            from lib.networks.diffusion import make_evolution
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
                use_dit_v2=getattr(cfg, 'use_dit_v2', False),
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

        # ClinicalBERT 部分
        # 加载 ClinicalBERT 模型和分词器
        # self.clinical_bert = AutoModel.from_pretrained(cfg.model_clinical_bert)
        # self.clinical_bert_tokenizer = AutoTokenizer.from_pretrained(cfg.model_clinical_bert)
        # # 冻结 ClinicalBERT 的参数
        # for param in self.clinical_bert.parameters():
        #     param.requires_grad = False
        # # 添加一个全连接层用于降维
        # self.bert_dim_reduction = nn.Linear(768, 64)

        # 双分类头部分
        # self.class_head = WaveMLP('M', num_classes=cfg.heads['ct_hm'])

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

        # 可视化并保存 cnn_feature（仅保存第一个 batch 的前若干通道为网格）
        # vis_root = "/mnt/sdb1/leijh/EnergySnake1/EnergeSnake1/zrc_visual/cnn_feature"
        # os.makedirs(vis_root, exist_ok=True)
        # feat = cnn_feature.detach().float().cpu()[0]  # [C, H, W]
        # C, H, W = feat.shape
        # max_ch = min(64, C)
        # cols = int(np.ceil(np.sqrt(max_ch)))
        # rows = int(np.ceil(max_ch / cols))
        # grid = np.zeros((rows * H, cols * W), dtype=np.uint8)
        # for idx in range(max_ch):
        #     r, c = divmod(idx, cols)
        #     ch = feat[idx].numpy()
        #     ch_min, ch_max = float(ch.min()), float(ch.max())
        #     if ch_max > ch_min:
        #         ch_norm = (ch - ch_min) / (ch_max - ch_min)
        #     else:
        #         ch_norm = np.zeros_like(ch)
        #     tile = (ch_norm * 255.0).astype(np.uint8)
        #     grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = tile
        # ts = time.strftime('%Y%m%d_%H%M%S')
        # out_path = os.path.join(vis_root, f'cnn_grid_{ts}.png')
        # cv2.imwrite(out_path, grid)
        # print(f"保存 cnn_feature 到 {out_path}")

        

        # print("cnn_feature.shape",cnn_feature.shape)  #[1, 64, 128, 128]

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

        #------------------------------
        #------------------------------
        # 简单可视化 YOLO 检测框：同一图像的检测框画在同一白板上（在可能的 GT 替换之前，确保是 YOLO 的预测）
        # 保存目录：/mnt/sdb1/leijh/EnergySnake1/EnergeSnake1/lib/zrc_visual/yolo_det
        # try:
        #     save_root = "/mnt/sdb1/leijh/EnergySnake1/EnergeSnake1/lib/ljh_visual/yolo_det"
        #     os.makedirs(save_root, exist_ok=True)

        #     det_vis = output.get('detection', None)
        #     feat_hw = output.get('feat_hw', None)
            
            
        #     if det_vis is not None and feat_hw is not None:
        #         Hf, Wf = int(feat_hw[0]), int(feat_hw[1])
        #         B = det_vis.size(0)
        #         # 获取输入图像作为背景
        #         input_images = batch['orig_img'][0]
        #         # 转换为numpy
        #         img_np = input_images.detach().cpu().numpy()
        #         if img_np.shape[0] == 3:
        #             img_np = np.transpose(img_np, (1, 2, 0)).astype(np.uint8)
        #         # 确保是BGR格式（OpenCV默认格式）
        #         if img_np.shape[2] == 3:
        #             # 如果是RGB格式，转换为BGR
        #             img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        #         # 计数器，确保文件名不冲突
        #         if not hasattr(self, '_det_vis_counter'):
        #             self._det_vis_counter = 0

        #         for b in range(B):
        #             # 生成白底画布（使用特征图尺寸坐标系，避免额外缩放）
        #             # canvas = np.ones((Hf*4, Wf*4, 3), dtype=np.uint8) * 255
        #             det_b = det_vis[b]
        #             # 过滤掉 padding 的全 0 行与低分框（>0 分即可，已做过 NMS）
        #             det_b = det_b.detach().float().cpu().numpy()
        #             if det_b.size == 0:
        #                 pass
        #             else:
        #                 # 保留 score > 0 的框
        #                 keep = det_b[:, 4] > 0
        #                 det_b = det_b[keep]

        #                 for box in det_b:
        #                     x1, y1, x2, y2, score, cls_id = box[:6]
        #                     x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        #                     # 绘制矩形与简单标签
        #                     color = (0, 255, 0)
        #                     cv2.rectangle(img_np, (x1i, y1i), (x2i, y2i), color, 1)
        #                     label = f"{int(cls_id)}:{score:.2f}"
        #                     # 放在框左上角，避免越界
        #                     tx, ty = max(0, x1i), max(0, y1i - 2)
        #                     cv2.putText(img_np, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

        #             fname = f"det_{self._det_vis_counter:07d}_b{b}.png"
        #             cv2.imwrite(os.path.join(save_root, fname), img_np)
        #         self._det_vis_counter += 1
        # except Exception as e:
        #     # 可视化失败不影响训练/推理
        #     print(e)
        #     pass
        #------------------------------
        #------------------------------


        # 训练时可选择使用 GT 框替换
        if getattr(cfg, 'use_gt_det', False):
            self.use_gt_detection(output, batch)

        #------------------------------
        # 测试 clinical_bert
        # input_text = "verba"
        # inputs = self.clinical_bert_tokenizer(input_text, return_tensors="pt").to('cuda')
        # with torch.no_grad():
        #     clinical_bert_output = self.clinical_bert(**inputs)
        # # 提取 ClinicalBERT 的 [CLS] 嵌入   last_hidden_state 结构：["[CLS]", ```, "[SEP]"]
        # clinical_bert_features = clinical_bert_output.last_hidden_state[:, 0, :]  # [CLS] 的嵌入
        # # 降维到 [1, 64]
        # clinical_bert_features = self.bert_dim_reduction(clinical_bert_features)
        # ------------------------------

        # 传入 Snake 进行演化
        if not self.freeze_snake:
            output = self.gcn(output, cnn_feature, batch)
        # print("output",output.keys())
        # print("py_pred",len(output['py_pred']))
        # print("py_pred",output['py_pred'][-1].shape)
        # print("detection",output['detection'].shape)

        # 暴露 YOLO 原始预测供损失使用
        output['yolo_preds'] = (yolo_y, yolo_feats)



        ## -------------------------------------------------------------------------------------------------------------
        # # 训练阶段使用双分类头一致性损失策略
        # if cfg.train_or_test == 'train':
        #     init = self.gcn.prepare_training(output, batch)
        #
        #     if 'py_pred' in output:
        #         py_pred = output['py_pred']
        #
        #         # 取最后一次演化结果
        #         if isinstance(py_pred, list):
        #             py_pred = py_pred[-1]
        #
        #         # 获取预测数量
        #         # num_preds = py_pred.size(0)  # 实际预测数量
        #
        #         detection = output['detection']
        #
        #         py_ind = init['py_ind']
        #         py_count = torch.bincount(py_ind)
        #
        #         topk_list = []
        #         for i in range(detection.shape[0]):
        #             count = py_count[i]
        #             # 提取第五维的值，并获取前 count 个最大值的索引
        #             topk_values, topk_indices = torch.topk(detection[i, :, 4], k=count)
        #             # 使用这些索引提取整个组的信息
        #             topk_groups = detection[i, topk_indices, :]
        #             topk_list.append(topk_groups)
        #
        #         if isinstance(topk_list, list):
        #             topk_tensor = torch.cat(topk_list,0)
        #         topk_class = topk_tensor[:,5]
        #         output["detect_classes"] = topk_class
        #
        #         # 下面是后分类头的代码
        #         # 取轮廓特征
        #         py_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, py_pred, init['py_ind'], 128, 128)
        #         #py_feature = torch.cat((py_feature, py_feature), dim=1)
        #         py_feature = torch.unsqueeze(py_feature, 1)
        #         py_feature = torch.cat((py_feature, py_feature, py_feature), dim=1)
        #         seg_class = self.class_head(py_feature)
        #
        #         output["seg_classes"] = seg_class
        ## -------------------------------------------------------------------------------------------------------------


            # #可视化CT——HM
            # from torchvision.utils import save_image
            # ct_hm = output['ct_hm']
            # ct_hm1 = ct_hm[0]
            # ct_hm_gt = batch['ct_hm']
            # ct_hm_gt1 = ct_hm_gt[0]
            # img = batch['inp']
            # img1 = img[0]
            # save_image(img1, '/home/ub/PycharmProjects/EnergeSnake/zrc_visual/230_ct_hm/img.png',
            #            normalize=True)
            # for i in range(ct_hm1.shape[0]):
            #     save_image(ct_hm1[i], f'/home/ub/PycharmProjects/EnergeSnake/zrc_visual/230_ct_hm/ct_hm_{i}.png', normalize=True)
            #     save_image(ct_hm_gt1[i], f'/home/ub/PycharmProjects/EnergeSnake/zrc_visual/230_ct_hm/ct_hm_gt_{i}.png', normalize=True)
            #
            # #可视化CNN-FEARTURE
            # cnn_feature1 = cnn_feature[0]
            # img = batch['inp']
            # img1 = img[0]
            # save_image(img1, '/home/ub/PycharmProjects/EnergeSnake/zrc_visual/230_cnn_feature/img.png',
            #            normalize=True)
            # for i in range(cnn_feature1.shape[0]):
            #     save_image(cnn_feature1[i], f'/home/ub/PycharmProjects/EnergeSnake/zrc_visual/230_cnn_feature/cnn_f_{i}.png', normalize=True)

        return output


def get_network(num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
    network = Network(num_layers, heads, head_conv, down_ratio, det_dir)
    return network


