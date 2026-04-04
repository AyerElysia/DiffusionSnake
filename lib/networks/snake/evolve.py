import torch.nn as nn
from .snake import Snake
from lib.utils.snake import snake_gcn_utils, snake_config, snake_decode, active_spline
import torch
from lib.networks.vision_mamba2.mamba2 import VMAMBA2Block
import os
import numpy as np
import cv2



class Evolution(nn.Module):
    def __init__(self):
        super(Evolution, self).__init__()

        self.fuse = nn.Conv1d(128, 64, 1)
        self.state_compression = VMAMBA2Block(dim=64*2, input_resolution=128)
        self.init_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid')
        self.evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='vm2')
        self.iter = 2
        # 形态复杂度权重
        self.sigma = 1  
  
        for i in range(self.iter):
            evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='vm2')
            self.__setattr__('evolve_gcn'+str(i), evolve_gcn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def prepare_training(self, output, batch):
        init = snake_gcn_utils.prepare_training(output, batch)
        output.update({'i_it_4py': init['i_it_4py'], 'i_it_py': init['i_it_py']})
        output.update({'i_gt_4py': init['i_gt_4py'], 'i_gt_py': init['i_gt_py']})
        return init

    def prepare_training_evolve(self, output, batch, init):
        evolve = snake_gcn_utils.prepare_training_evolve(output['ex_pred'], init)
        output.update({'i_it_py': evolve['i_it_py'], 'c_it_py': evolve['c_it_py'], 'i_gt_py': evolve['i_gt_py']})
        evolve.update({'py_ind': init['py_ind']})
        return evolve

    def prepare_testing_init(self, output):
        init = snake_gcn_utils.prepare_testing_init(output['detection'][..., :4], output['detection'][..., 4])  # init = {'i_it_4py': i_it_4pys  （0，40，2）, 'c_it_4py': c_it_4pys   （0，40，2）, 'ind': ind}
        # output['detection'] = output['detection'][output['detection'][..., 4] > snake_config.ct_score]
        output.update({'it_ex': init['i_it_4py']})
        return init

    def prepare_testing_evolve(self, output, h, w):
        ex = output['ex']
        ex[..., 0] = torch.clamp(ex[..., 0], min=0, max=w-1)
        ex[..., 1] = torch.clamp(ex[..., 1], min=0, max=h-1)
        evolve = snake_gcn_utils.prepare_testing_evolve(ex)
        output.update({'it_py': evolve['i_it_py']})
        return evolve
    
    def compute_contour_entropy(self, points,):
        """计算轮廓熵"""
        batch, npoints = points.shape[:2]
        device = points.device
        
        # 1. 计算点间距
        dist = torch.norm(
            points.roll(-1, dims=1) - points,
            dim=-1
        )
        dist_entropy = -torch.softmax(dist, dim=-1) * torch.log_softmax(dist, dim=-1)
        
        # 2. 计算曲率
        prev = points.roll(1, dims=1)
        next = points.roll(-1, dims=1)
        v1 = prev - points
        v2 = next - points
        cos_angles = torch.sum(
            v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-6) * 
            v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-6),
            dim=-1
        )
        curvature = 1 - torch.clamp(cos_angles, -1, 1)
        curv_entropy = -torch.softmax(curvature, dim=-1) * torch.log_softmax(curvature, dim=-1)
        
        # 3. 计算平滑度
        angles = torch.atan2(v1[..., 1], v1[..., 0])
        angle_diff = (angles.roll(-1, dims=1) - angles).abs()
        smooth_entropy = -torch.softmax(angle_diff, dim=-1) * torch.log_softmax(angle_diff, dim=-1)
        
        # 组合三种熵
        contour_entropy = (dist_entropy + curv_entropy + smooth_entropy) / 3
        
        # 归一化
        contour_entropy = (contour_entropy - contour_entropy.mean(dim=1, keepdim=True)) / (
            contour_entropy.std(dim=1, keepdim=True) + 1e-6
        )

        return contour_entropy.unsqueeze(-1)  # [B, L, 1]
    

    def compute_prior_mask_batch(self, points, entropy, sigma, py_ind):
        """
        points:   [batch_size, B, L, 2] where B is contours per image (may include padding)
        entropy:  [batch_size, B, L, 1] 
        py_ind:   tensor indicating which image each contour belongs to
        return:   L_star [total_contours, H, L, L]
        """
        batch_size, B_per_image, L = points.shape[0], points.shape[1], points.shape[2]
        
        # 获取实际的轮廓总数（来自py_ind的长度）
        total_contours = len(py_ind)
        
        # 将points和entropy重塑为实际的总轮廓数
        # 注意：points和entropy可能包含填充，需要根据py_ind来获取实际轮廓
        if total_contours != batch_size * B_per_image:
            # 如果实际轮廓数不等于batch_size * B_per_image，说明有填充或变化
            # 我们需要重新组织数据以匹配py_ind
            points_flat = []
            entropy_flat = []
            
            for img_idx in range(batch_size):
                # 找到属于当前图片的轮廓
                img_mask = (py_ind == img_idx)
                num_img_contours = img_mask.sum().item()
                
                if num_img_contours > 0:
                    # 获取当前图片的所有轮廓（可能有填充）
                    img_points = points[img_idx]  # [B_per_image, L, 2]
                    img_entropy = entropy[img_idx]  # [B_per_image, L, 1]
                    
                    # 只取实际存在的轮廓
                    valid_img_points = img_points[:num_img_contours]  # [num_img_contours, L, 2]
                    valid_img_entropy = img_entropy[:num_img_contours]  # [num_img_contours, L, 1]
                    
                    points_flat.append(valid_img_points)
                    entropy_flat.append(valid_img_entropy)
            
            # 合并所有图片的轮廓
            points = torch.cat(points_flat, dim=0)  # [total_contours, L, 2]
            entropy = torch.cat(entropy_flat, dim=0)  # [total_contours, L, 1]
        else:
            # 如果没有填充，直接重塑
            points = points.reshape(-1, L, 2)  # [total_contours, L, 2]
            entropy = entropy.reshape(-1, L, 1)  # [total_contours, L, 1]
        
        # 去掉最后一维：[total_contours, L]
        entropy = entropy.squeeze(-1)  # [total_contours, L]

        # 初始化结果矩阵
        L_star = torch.zeros(total_contours, 8, L, L, device=points.device)

        # 获取批次中的图片数量
        unique_images = torch.unique(py_ind)
        
        # 对每张图片分别处理
        for img_idx in unique_images:
            # 找到属于当前图片的轮廓
            img_mask = (py_ind == img_idx)
            img_contours = torch.where(img_mask)[0]
            
            if len(img_contours) == 0:
                continue

            # 提取当前图片的轮廓点和熵
            img_points = points[img_mask]  # [num_contours_in_img, L, 2]
            img_entropy = entropy[img_mask]  # [num_contours_in_img, L]

            # 空间距离矩阵（仅在当前图片内的轮廓之间计算）
            p1 = img_points.unsqueeze(2)  # [num_contours_in_img, L, 1, 2]
            p2 = img_points.unsqueeze(1)  # [num_contours_in_img, 1, L, 2]
            dist2 = torch.sum((p1 - p2) ** 2, dim=-1)  # [num_contours_in_img, L, L]
            gaussian = torch.exp(-dist2 / (sigma ** 2 + 1e-6))  # [num_contours_in_img, L, L]

            # 形态复杂度项
            h_t1 = img_entropy.unsqueeze(2)  # [num_contours_in_img, L, 1]
            h_t2 = img_entropy.unsqueeze(1)  # [num_contours_in_img, 1, L]
            morph_term = (1 - h_t1) * h_t2  # [num_contours_in_img, L, L]

            # 组合得到 L*
            img_L_star = morph_term * gaussian  # [num_contours_in_img, L, L]

            # 添加 head 维度并存储到结果中
            img_L_star = img_L_star.unsqueeze(1).repeat(1, 8, 1, 1)  # [num_contours_in_img, 8, L, L]
            L_star[img_mask] = img_L_star

        return L_star

    def init_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
        #print(i_it_pprepare_testing_initoly.shape)
        if len(i_it_poly) == 0:
            return torch.zeros([0, 4, 2]).to(i_it_poly)  # 这个张量的形状可以被理解为一个包含 0 个四维向量的集合，其中每个四维向量本身又包含 2 个元素。

        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        center = (torch.min(i_it_poly, dim=1)[0] + torch.max(i_it_poly, dim=1)[0]) * 0.5
        ct_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, center[:, None], ind, h, w)
        init_feature = torch.cat([init_feature, ct_feature.expand_as(init_feature)], dim=1)
        init_feature = self.fuse(init_feature)

        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2), init_input.device)
        offset, _ = snake(init_input, adj, i_it_poly)
        i_poly = i_it_poly + offset.permute(0, 2, 1)
        i_poly = i_poly[:, ::snake_config.init_poly_num//4]

        return i_poly

    

    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind, later_poly=None):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)  #(一个batch里的点的个数，64, 128)
        if later_poly is not None:
            later_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, later_poly, ind, h, w)
            feature = torch.cat((init_feature, later_feature), dim=1)
            compressed_feature, _ = self.state_compression(feature)
            init_feature = torch.split(compressed_feature, 64, dim=1)[0]
        c_it_poly = c_it_poly * snake_config.ro
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2), init_input.device)
        offset, L = snake(init_input, adj, i_it_poly)
        i_poly = i_it_poly * snake_config.ro + offset.permute(0, 2, 1)
        return i_poly, L




    def forward(self, output, cnn_feature, batch=None):
        ret = output
        # batch = {dict : 2}，包括两个部分：batch['inp']=(1,3,544,544),这个应该是原输入图像，batch['meta']={dict:4}，其中{'ann':null, 'center':[256,256], 'scale':[512, 512], 'vis_GT':null}，这个记录的是batch的一些参数
        # output = {dict : 2}, 包括两个部分：ct_hm (1, 9, 136, 136) 和 wh(1, 2, 136, 136)，这两个信息看看怎么用
        # cnn_feature  = {tensor:[1, 64, 136, 136]}

        def to_cv_contours(polys):
            cv_polys = []
            for p in polys:
                if p is None:
                    continue
                p = np.asarray(p)
                if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 3:
                    continue
                p = np.round(p).astype(np.int32).reshape(-1, 1, 2)  # (K,1,2)
                cv_polys.append(p)
            return cv_polys
        
        def draw_init_polys_on_white(init_i_it_4py, save_path, canvas_size=(512, 512)):
            # init_i_it_4py: torch.Tensor, shape (N, 80, 2)
            polys = init_i_it_4py.detach().cpu().numpy().astype(np.float32)  # (N,80,2)
            polys_list = [polys[i] for i in range(polys.shape[0])]

            # 白底画布
            H, W = canvas_size
            canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

            # 可选：将坐标做简单平移/缩放以放进画布（不需要就注释掉）
            # 这里演示做一个平移，避免负坐标
            all_pts = polys.reshape(-1, 2)
            min_x, min_y = np.min(all_pts[:, 0]), np.min(all_pts[:, 1])
            shift = np.array([max(-min_x + 10, 0), max(-min_y + 10, 0)], dtype=np.float32)
            polys_list = [p + shift for p in polys_list]

            # 转为 OpenCV 轮廓格式并绘制
            cv_polys = to_cv_contours(polys_list)
            if len(cv_polys) > 0:
                cv2.polylines(canvas, cv_polys, isClosed=True, color=(0, 0, 255), thickness=2)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, canvas)

        if self.training:  #训练模式
            with torch.no_grad():
                init = self.prepare_training(output, batch)
                # save_root = "/mnt/sdb1/leijh/EnergySnake1/EnergeSnake1/lib/ljh_visual/yolo_det_1/test1.png"
                # draw_init_polys_on_white(init['i_it_4py']*3.75, save_root)   # 八边形
                # '/mnt/sdb1/leijh/EnergySnake1/Data_processed/230processed_filled/203_image.png'
                # save_name = batch['img_path'][0].split('/')[-1]
                # draw_init_polys_on_white(init['i_it_py']*3.75, "/mnt/sdb1/leijh/EnergySnake1/EnergeSnake1/lib/ljh_visual/it_py_visual_1/"+save_name)   # 蛇轮廓

            ex_pred = self.init_poly(self.init_gcn, cnn_feature, init['i_it_4py'], init['c_it_4py'], init['4py_ind'])
            ret.update({'ex_pred': ex_pred, 'i_gt_4py': output['i_gt_4py']})

            # with torch.no_grad():
            #     init = self.prepare_training_evolve(output, batch, init)

            py_pred, _ = self.evolve_poly(self.evolve_gcn, cnn_feature, init['i_it_py'], init['c_it_py'], init['py_ind'])
            py_preds = [py_pred]
        # img1 = img[0]
            for i in range(self.iter):
                py_pred = py_pred / snake_config.ro
                c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred)
                evolve_gcn = self.__getattr__('evolve_gcn'+str(i))
                py_pred, L = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred, init['py_ind'], py_preds[-1])
                py_preds.append(py_pred)
            ce = self.compute_contour_entropy(batch['i_gt_py'])
            # print("ce:",ce.shape)  # ce: torch.Size([1, 22, 128, 1])
            # print("L:",L.shape)  # L: torch.Size([22, 8, 128, 128]),一张图片中存在22个多边形
            L_star = self.compute_prior_mask_batch(batch['i_gt_py'], ce, self.sigma, init['py_ind'])  # L_star的形状要和L保持一致。
            # print("L_star:",L_star.shape)  # L_star: torch.Size([22, 8, 128, 128])
            
            ret.update({'py_pred': py_preds, 'i_gt_py': output['i_gt_py'] * snake_config.ro, 'L': L, 'L_star': L_star})



        if not self.training:

            with torch.no_grad():
                init = self.prepare_testing_init(output)
                # save_root = "/mnt/sdb1/leijh/EnergySnake1/EnergeSnake1/lib/ljh_visual/yolo_det_1/test.png"
                # draw_init_polys_on_white(init['i_it_4py']*3.75, save_root)
                
                ex = self.init_poly(self.init_gcn, cnn_feature, init['i_it_4py'], init['c_it_4py'], init['ind'])
                ret.update({'ex': ex})

                evolve = self.prepare_testing_evolve(output, cnn_feature.size(2), cnn_feature.size(3))
                # draw_init_polys_on_white(evolve['i_it_py']*3.75, "/mnt/sdb1/leijh/EnergySnake1/EnergeSnake1/lib/ljh_visual/yolo_det_1/pred_py.png")
                py, _ = self.evolve_poly(self.evolve_gcn, cnn_feature, evolve['i_it_py'], evolve['c_it_py'], init['ind'])
                pys = [py / snake_config.ro]
                for i in range(self.iter):
                    py = py / snake_config.ro
                    c_py = snake_gcn_utils.img_poly_to_can_poly(py)
                    evolve_gcn = self.__getattr__('evolve_gcn'+str(i))
                    py, _ = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['ind'], pys[-1])
                    pys.append(py / snake_config.ro)
                ret.update({'py': pys})


        return output
