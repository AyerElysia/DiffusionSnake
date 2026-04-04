import torch
import numpy as np
import scipy.io as sio
from lib.config import cfg

class CRFDataConverter:
    def __init__(self):
        # 脊椎类别映射（根据你的配置文件）
        self.vertebrae_classes = {
            1: 0,   # S1 -> 0
            2: 1,   # L5 -> 1  
            3: 2,   # L4 -> 2
            4: 3,   # L3 -> 3
            5: 4,   # L2 -> 4
            6: 5,   # L1 -> 5
            7: 6,   # T12 -> 6
            8: 7,   # T11 -> 7
            9: 8,   # T10 -> 8
            # 可以根据需要添加更多类别
        }
        
    def extract_features_from_detection(self, detection, poly):
        """
        从EnergySnake1的检测结果中提取特征
        Args:
            detection: [1, N, 6] 检测结果 (x1, y1, x2, y2, score, class_id)
            poly: [N, 128, 2] 多边形坐标
        Returns:
            features: [N, 10] 特征向量
        """
        N = detection.shape[1]
        features = np.zeros((N, 10))
        
        for i in range(N):
            # 提取几何特征
            bbox = detection[0, i, :4].cpu().numpy()
            score = detection[0, i, 4].cpu().numpy()
            class_id = int(detection[0, i, 5].cpu().numpy())
            
            # 计算bbox特征
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height
            aspect_ratio = width / height if height > 0 else 1.0
            
            # 计算多边形特征
            poly_coords = poly[i].cpu().numpy()
            poly_area = self._calculate_polygon_area(poly_coords)
            poly_perimeter = self._calculate_polygon_perimeter(poly_coords)
            
            # 组合特征
            features[i] = [
                score,           # 置信度
                width,          # 宽度
                height,         # 高度
                area,           # 面积
                aspect_ratio,   # 宽高比
                poly_area,      # 多边形面积
                poly_perimeter, # 多边形周长
                x1,            # 位置x
                y1,            # 位置y
                class_id       # 类别ID
            ]
            
        return features
    
    def _calculate_polygon_area(self, coords):
        """计算多边形面积"""
        x, y = coords[:, 0], coords[:, 1]
        return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] 
                           for i in range(-1, len(x)-1)))
    
    def _calculate_polygon_perimeter(self, coords):
        """计算多边形周长"""
        coords = np.vstack([coords, coords[0]])  # 闭合
        return np.sum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)))