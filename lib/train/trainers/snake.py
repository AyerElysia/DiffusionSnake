import torch.nn as nn
from lib.utils import net_utils
import torch
from lib.config import cfg
import sys
import torch.nn.functional as F
from lib.networks.YOLOV8.utils.loss import v8DetectionLoss  # 回退使用的 YOLOv8 损失实现
import os
import cv2
import numpy as np

# ==============================================================================
# ======================== 新增的可视化函数 ==============================
# ==============================================================================
def visualize_polygons(pred_polys_tensor, gt_polys_tensor, batch_meta, save_dir, vis_counter):
    """
    将预测和金标准多边形可视化并保存。

    Args:
        pred_polys_tensor (torch.Tensor): 预测的多边形张量，形状为 [N, 128, 2]。
        gt_polys_tensor (torch.Tensor): 金标准多边形张量，形状为 [N, 128, 2]。
        batch_meta (dict): 包含元信息的字典，用于生成文件名。
        save_dir (str): 保存可视化结果的根目录。
        vis_counter (int): 一个全局计数器，用于在元信息不足时生成唯一文件名。
    """
    # 1. 准备工作：将Tensor转为Numpy数组
    # 从GPU移动到CPU，并转换为numpy数组，坐标转为整数
    pred_polys = pred_polys_tensor.detach().cpu().numpy().astype(np.int32)
    gt_polys = gt_polys_tensor.detach().cpu().numpy().astype(np.int32)

    # `batch['meta']['ct_num']` 记录了每个batch item对应多少个polygon。
    # 我们需要根据这个信息，将扁平化的polygon列表重新分配给它们所属的原始图像。
    # 构造一个列表，其中每个元素代表一个polygon属于第几张原始图片
    # 例如：img_indices = [0, 0, 1, 2, 2, 2] 表示前2个poly属于第0张图，第3个属于第1张图...
    img_indices = []
    for i, num_polys in enumerate(batch_meta['ct_num']):
        img_indices.extend([i] * num_polys)

    # num_images_in_batch = len(batch_meta['ct_img_path'])

    # 2. 为批次中的每张原始图像生成一张可视化图
    for img_idx in range(3):
        # 创建一个512x512的白色画布 (您可以根据需要调整画布大小)
        canvas_size = (512, 512, 3)
        canvas = np.ones(canvas_size, dtype=np.uint8) * 255

        # 找出属于当前图像的所有多边形的索引
        poly_indices_for_this_image = [i for i, idx in enumerate(img_indices) if idx == img_idx]

        # 如果这张图没有多边形，则跳过
        if not poly_indices_for_this_image:
            continue 

        # 获取属于这张图的多边形，并放入列表中（cv2.polylines需要列表格式）
        current_pred_polys = [pred_polys[i] for i in poly_indices_for_this_image]
        current_gt_polys = [gt_polys[i] for i in poly_indices_for_this_image]

        # 3. 绘制多边形
        # 绘制金标准多边形 (绿色)
        cv2.polylines(canvas, current_gt_polys, isClosed=True, color=(0, 255, 0), thickness=2)
        # 绘制预测多边形 (红色)
        cv2.polylines(canvas, current_pred_polys, isClosed=True, color=(255, 0, 0), thickness=2)

        # 4. 生成保存路径和文件名
        try:
            # 优先使用epoch和原始图片名来创建唯一的文件路径
            epoch = batch_meta.get('epoch', 0)
            original_img_name = os.path.basename(batch_meta['ct_img_path'][img_idx])
            img_name_without_ext = os.path.splitext(original_img_name)[0]
            
            # 创建子文件夹
            epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch}")
            os.makedirs(epoch_save_dir, exist_ok=True)
            
            # 定义最终保存路径
            save_path = os.path.join(epoch_save_dir, f"{img_name_without_ext}_vis.png")
            print("保存路径：", save_path)

        except (KeyError, IndexError):
            # 如果元信息不全，则使用全局计数器作为备用方案
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"vis_{vis_counter}_{img_idx}.png")
            
        # 5. 保存图像
        cv2.imwrite(save_path, canvas)


def intermediate_signal(gtpoly):
    decerate = [0,0.25]
    circle_rate=1
    sup_polys=[]
    for rate in decerate:
        for i in range(gtpoly.shape[0]):
            gtpyiter = gtpoly[i,:,:]      #128*2
            center = torch.mean(gtpyiter,dim=0)
            center_vector = center.repeat(128,1)
            pdist = nn.PairwiseDistance(p=2)
            pdist_result = pdist(gtpyiter,center_vector)
            mean_pdist = torch.mean(pdist_result)
            vector_poly = gtpyiter-center
            pdist_result1 = pdist_result.unsqueeze(dim=1).repeat(1,2)         
            vector_poly_circle = mean_pdist*vector_poly/pdist_result1
            gap_dist = pdist_result-mean_pdist
            percentage1 = 1+ rate*gap_dist/mean_pdist
            percentage1 = percentage1.unsqueeze(dim=1)
            percentage1 = percentage1.repeat(1,2)
            percentage1 = circle_rate*percentage1
            if i==0:
                layer1_sup_poly = vector_poly_circle.mul(percentage1)+center_vector
                layer1_sup_poly = layer1_sup_poly.unsqueeze(dim=0)
            else:
                layer1_sup_poly_toappend = vector_poly_circle.mul(percentage1)+center_vector
                layer1_sup_poly_toappend = layer1_sup_poly_toappend.unsqueeze(dim=0)
                layer1_sup_poly = torch.cat((layer1_sup_poly,layer1_sup_poly_toappend),dim=0)
        sup_polys.append(layer1_sup_poly)
    return sup_polys

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        try:
            self.det_crit = self.net.yolo.init_criterion()
        except Exception:
            self.det_crit = v8DetectionLoss(self.net.yolo)
        self.ex_crit = torch.nn.functional.smooth_l1_loss
        self.py_crit = torch.nn.functional.smooth_l1_loss
        self.L_crit = F.mse_loss    # CMAM loss

        default_scales = {'det': 0.6, 'ex': 1.0, 'py': 1.0}
        self.loss_scales = getattr(cfg, 'loss_scales', default_scales)
        for k, v in default_scales.items():
            if k not in self.loss_scales:
                self.loss_scales[k] = v

        # 冻结开关
        self.freeze_snake = bool(getattr(cfg, 'freeze_snake', False))

        # 若完全冻结 Snake，可把 ex/py 的 loss scale 置 0（可选）
        if self.freeze_snake:
            self.loss_scales['ex'] = 0.0
            self.loss_scales['py'] = 0.0

        # 可视化计数器
        self._vis_counter = 0

    def forward(self, batch):
        output = self.net(batch['inp'], batch)

        scalar_stats = {}
        base_device = output['inp'].device if isinstance(output.get('inp', None), torch.Tensor) else output['detection'].device
        loss = torch.zeros(1, device=base_device).squeeze()

        if self.det_crit is not None and 'yolo_preds' in output:
            det_loss, det_items = self.det_crit(output['yolo_preds'], batch)
            box_l, cls_l, dfl_l = det_items[0], det_items[1], det_items[2]
            scalar_stats.update({'det_box': box_l, 'det_cls': cls_l, 'det_dfl': dfl_l})
            scalar_stats.update({'det_loss': det_loss})
            if det_loss.device != loss.device:
                det_loss = det_loss.to(loss.device)
            det_weight = float(self.loss_scales.get('det', 1.0))
            det_loss_scaled = det_weight * det_loss
            scalar_stats.update({'det_loss_scaled': det_loss_scaled})
            loss += det_loss_scaled

        # 演化损失
        if self.freeze_snake:
            zero = torch.zeros((), device=loss.device)
            scalar_stats.update({'ex_loss': zero, 'py_loss': zero})
            scalar_stats.update({'loss': loss})
            image_stats = {}
            return output, loss, scalar_stats, image_stats
        
        ex_target = output['i_gt_4py']
        if ex_target.device != output['ex_pred'].device:
            ex_target = ex_target.to(output['ex_pred'].device)
        ex_loss = self.ex_crit(output['ex_pred'], ex_target)
        if ex_loss.device != loss.device:
            ex_loss = ex_loss.to(loss.device)
        scalar_stats.update({'ex_loss': ex_loss})
        ex_weight = float(self.loss_scales.get('ex', 1.0))
        loss += ex_weight * ex_loss
        # ex_pred 的来源: 由演化模块的初始化 GCN 在 CNN 特征上，对“初始多边形”做一次更新得到，并非直接来自 YOLO 检测框。
        
        py_loss = 0
        if cfg.multistage:
            sup_polys=intermediate_signal(output['i_gt_py'])
            supsignal = [sup_polys[0],sup_polys[1],output['i_gt_py']]
            for i in range(len(output['py_pred'])):
                py_loss += self.py_crit(output['py_pred'][i], supsignal[2]) / len(output['py_pred'])
        else:
            output['py_pred'] = [output['py_pred'][-1]]
            for i in range(len(output['py_pred'])):
                py_loss += self.py_crit(output['py_pred'][i], output['i_gt_py']) / len(output['py_pred'])
        scalar_stats.update({'py_loss': py_loss})
        py_weight = float(self.loss_scales.get('py', 1.0))
        loss += py_weight * py_loss

        L_loss = self.L_crit(output['L'], output['L_star'])
        scalar_stats.update({'L_loss': L_loss})
        loss += L_loss



        # ==============================================================================
        # ======================== 在这里调用可视化函数 ==========================
        # ==============================================================================
        # 为了避免每一步都可视化导致训练过慢，可以通过配置文件中的开关来控制
        if 0:
            # 从 batch 中获取最后一个预测阶段的多边形
            pred_polys_to_vis = output['py_pred'][-1]
            gt_polys_to_vis = output['i_gt_py']
            
            save_directory = '/mnt/sdb1/leijh/EnergySnake1/EnergeSnake1/lib/ljh_visual/train'
            
            visualize_polygons(
                pred_polys_tensor=pred_polys_to_vis,
                gt_polys_tensor=gt_polys_to_vis,
                batch_meta=batch['meta'],
                save_dir=save_directory,
                vis_counter=self._vis_counter
            )
            # 更新计数器，确保即使元信息缺失文件名也不会重复
            self._vis_counter += 1
        
        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats