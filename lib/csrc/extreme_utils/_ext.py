"""
临时替代lib.csrc.extreme_utils模块
由于C++扩展不可用，这里提供纯Python实现
"""

import torch
import numpy as np


class FakeExtremeUtils:
    """假的extreme_utils模块，提供必要的功能"""

    def roll_array(self, array, shift):
        """替代extreme_utils.roll_array"""
        # 简单的循环移位实现
        if shift == 0:
            return array

        shift = shift % array.size(0)  # 确保shift在有效范围内
        return torch.cat([array[shift:], array[:shift]], dim=0)

    def calculate_edge_num(self, edge_num, edge_num_sum, edge_idx_sort, p_num):
        """替代extreme_utils.calculate_edge_num"""
        # 简化实现：调整edge_num使其总和等于p_num
        for i in range(edge_num.size(0)):
            for j in range(edge_num.size(1)):
                if edge_num_sum[i, j] != p_num:
                    # 调整最大的edge使其满足总和要求
                    max_idx = edge_idx_sort[i, j, 0]
                    if edge_num[i, j, max_idx] > 1:
                        edge_num[i, j, max_idx] -= 1
                        edge_num_sum[i, j] -= 1

    def calculate_wnp(self, edge_num, edge_start_idx, p_num):
        """替代extreme_utils.calculate_wnp"""
        # 简化实现：返回均匀分布的权重和索引
        batch_size, num_polys = edge_num.size(0), edge_num.size(1)
        total_points = p_num

        ind = []
        weight = []

        for i in range(batch_size):
            for j in range(num_polys):
                poly_ind = []
                poly_weight = []

                current_edge = 0
                current_pos = 0

                for k in range(total_points):
                    # 找到当前点所属的边
                    while current_pos >= edge_num[i, j, current_edge]:
                        current_pos -= edge_num[i, j, current_edge]
                        current_edge += 1
                        current_pos = 0

                    # 计算在当前边上的位置
                    edge_points = edge_num[i, j, current_edge]
                    if edge_points > 1:
                        edge_progress = current_pos / (edge_points - 1)
                    else:
                        edge_progress = 0.0

                    # 线性插值权重
                    weight_val = edge_progress
                    poly_ind.append([current_edge, (current_edge + 1) % 4])  # 假设四边形
                    poly_weight.append(weight_val)

                    current_pos += 1

                ind.append(poly_ind)
                weight.append(poly_weight)

        # 转换为张量
        ind = torch.tensor(ind, dtype=torch.long)
        weight = torch.tensor(weight, dtype=torch.float32)

        return weight, ind


# 创建全局实例
_ext = FakeExtremeUtils()
