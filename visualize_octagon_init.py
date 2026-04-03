import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Importing the get_octagon implementation from our decoder
from lib.utils.snake.snake_decode import get_octagon

def mock_organ_points():
    # 模拟一个稍微倾斜、带有不规则形态的类肝脏器官 (斜向生长)
    t = np.linspace(0, 2 * np.pi, 100)
    # 椭圆基座
    x = 100 * np.cos(t) + 150
    y = 50 * np.sin(t) + 150
    
    # 加入形态扭曲，使其不是规则正圆 (旋转)
    angle = np.pi / 4 # 45度倾斜
    x_rot = (x - 150) * np.cos(angle) - (y - 150) * np.sin(angle) + 150
    y_rot = (x - 150) * np.sin(angle) + (y - 150) * np.cos(angle) + 150
    
    pts = np.stack([x_rot, y_rot], axis=1)
    pts = torch.from_numpy(pts).float()
    return pts

def get_extreme_points(pts):
    # 寻找上下左右四个极值点
    # Top: min y
    t_idx = pts[:, 1].argmin()
    # Bottom: max y
    b_idx = pts[:, 1].argmax()
    # Left: min x
    l_idx = pts[:, 0].argmin()
    # Right: max x
    r_idx = pts[:, 0].argmax()
    
    ex = torch.stack([pts[t_idx], pts[l_idx], pts[b_idx], pts[r_idx]])
    return ex

def get_box(ex):
    # 根据极值点求 Box
    x_min, y_min = ex[:, 0].min(), ex[:, 1].min()
    x_max, y_max = ex[:, 0].max(), ex[:, 1].max()
    return torch.tensor([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])

def get_diamond(ex):
    # Box 中点连线 (简单的基于框的初始态)
    x_min, y_min = ex[:, 0].min(), ex[:, 1].min()
    x_max, y_max = ex[:, 0].max(), ex[:, 1].max()
    x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
    return torch.tensor([[x_mid, y_min], [x_min, y_mid], [x_mid, y_max], [x_max, y_mid]])

if __name__ == '__main__':
    pts = mock_organ_points()
    ex = get_extreme_points(pts)
    
    # ex 的 shape 需要为 [batch, num, 4, 2] 作为 get_octagon 的输入
    ex_input = ex.unsqueeze(0).unsqueeze(0)
    oct_pts = get_octagon(ex_input)
    oct_pts = oct_pts[0, 0] # [12, 2] 实际上这个函数返回12个点，因为每边有切角，严格说是12边形或更复杂的极点多边形
    
    box_pts = get_box(ex)
    diamond_pts = get_diamond(ex)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    titles = ["1. Bbox (Current YOLO)", "2. Diamond (Current Init)", "3. Octagon (Deep Snake Init)"]
    shapes = [box_pts, diamond_pts, oct_pts]
    colors = ['red', 'orange', 'green']
    
    for i, ax in enumerate(axes):
        ax.plot(pts[:, 0], pts[:, 1], 'k--', label='True Contour (Organ)')
        ax.scatter(ex[:, 0], ex[:, 1], c='blue', s=50, label='Extreme Points', zorder=5)
        
        # 闭合曲线
        shape_close = torch.cat([shapes[i], shapes[i][:1]], dim=0)
        ax.plot(shape_close[:, 0], shape_close[:, 1], color=colors[i], linewidth=2, label=titles[i].split('.')[1])
        
        ax.set_title(titles[i])
        ax.invert_yaxis()
        ax.legend()
        ax.axis('equal')
        
    plt.tight_layout()
    plt.savefig('visual/octagon_init_demo.png', dpi=150)
    print("Saved to visual/octagon_init_demo.png")
