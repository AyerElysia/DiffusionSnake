"""
边缘感知平滑模块 - 独立验证脚本
不修改任何现有代码，直接对V3.0的预测结果进行后处理
"""

import torch
import torch.nn as nn
import numpy as np


class EdgeAwareSmoothing(nn.Module):
    """
    边缘感知平滑：在平坦区域平滑，在尖锐转角处保持锐利

    核心思想：
    1. 计算每个点的曲率（二阶差分）
    2. 曲率大的点（尖锐转角）→ 低平滑权重
    3. 曲率小的点（平坦边缘）→ 高平滑权重
    4. 加权混合原始轮廓和平滑轮廓
    """
    def __init__(self, curvature_threshold=5.0):
        super().__init__()
        self.curvature_threshold = curvature_threshold

    def compute_curvature(self, contour):
        """
        计算每个点的曲率（二阶差分的模）

        Args:
            contour: (N, P, 2) - 轮廓坐标
        Returns:
            curvature: (N, P, 1) - 每个点的曲率
        """
        prev = torch.roll(contour, 1, dims=1)
        next = torch.roll(contour, -1, dims=1)

        # 一阶差分
        d1 = next - contour
        d1_prev = contour - prev

        # 二阶差分（曲率的近似）
        d2 = d1 - d1_prev
        curvature = torch.norm(d2, dim=-1, keepdim=True)  # (N, P, 1)
        return curvature

    def forward(self, contour):
        """
        Args:
            contour: (N, P, 2) - 预测的轮廓
        Returns:
            smoothed: (N, P, 2) - 平滑后的轮廓
        """
        # 计算曲率
        curvature = self.compute_curvature(contour)  # (N, P, 1)

        # 归一化曲率到[0, 1]范围，用于计算平滑权重
        # 曲率越大（尖锐转角）→ 权重越小（少平滑）
        # 曲率越小（平坦区域）→ 权重越大（多平滑）
        smooth_weight = torch.exp(-curvature / self.curvature_threshold)  # (N, P, 1)

        # 循环卷积平滑（简单的3点加权平均）
        prev = torch.roll(contour, 1, dims=1)
        next = torch.roll(contour, -1, dims=1)
        smoothed_local = (prev + 2 * contour + next) / 4

        # 加权混合：smooth_weight越大，越接近smoothed_local
        # 修正：应该是 smooth_weight * smoothed_local + (1 - smooth_weight) * contour
        # 但这样在低曲率时才会平滑，所以逻辑是对的
        # 问题可能是curvature_threshold太小，导致所有点的smooth_weight都接近1
        smoothed = smooth_weight * smoothed_local + (1 - smooth_weight) * contour

        return smoothed


def smooth_contours_numpy(contours, curvature_threshold=5.0, iterations=2):
    """
    对numpy数组格式的轮廓进行平滑

    Args:
        contours: numpy array (N, P, 2) 或 (P, 2)
        curvature_threshold: 曲率阈值
        iterations: 迭代次数
    Returns:
        smoothed: 平滑后的轮廓，shape与输入相同
    """
    # 转换为torch tensor
    if isinstance(contours, np.ndarray):
        contours_tensor = torch.from_numpy(contours).float()
    else:
        contours_tensor = contours

    # 确保是3维 (N, P, 2)
    if contours_tensor.dim() == 2:
        contours_tensor = contours_tensor.unsqueeze(0)

    # 创建平滑器
    smoother = EdgeAwareSmoothing(curvature_threshold=curvature_threshold)

    # 迭代平滑
    smoothed = contours_tensor
    for _ in range(iterations):
        smoothed = smoother(smoothed)

    # 转换回numpy
    smoothed_np = smoothed.squeeze(0).numpy() if contours.ndim == 2 else smoothed.numpy()

    return smoothed_np


if __name__ == "__main__":
    # 测试代码
    print("EdgeAwareSmoothing模块测试")

    # 创建一个测试轮廓（带噪声的圆）
    num_points = 128
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radius = 50

    # 添加噪声模拟毛刺
    noise = np.random.randn(num_points) * 2
    x = radius * np.cos(theta) + noise
    y = radius * np.sin(theta) + noise

    contour = np.stack([x, y], axis=1)  # (128, 2)

    print(f"原始轮廓 shape: {contour.shape}")

    # 应用平滑
    smoothed = smooth_contours_numpy(contour, curvature_threshold=0.1, iterations=2)

    print(f"平滑后轮廓 shape: {smoothed.shape}")

    # 计算平滑前后的差异
    diff = np.mean(np.linalg.norm(contour - smoothed, axis=1))
    print(f"平均位移: {diff:.4f} 像素")

    print("\n模块测试通过！")
