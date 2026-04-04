import numpy as np
import torch, matplotlib.pyplot as plt

def show_heatmap(x, topk=50, clip_percentile=0.5, title="heatmap"):
    """
    x: np.ndarray 或 torch.Tensor（形状最好是 HxW；>2D 可先切片/聚合）
    topk: 标注前 k 大 & k 小
    clip_percentile: 颜色拉伸的分位裁剪(0~1，如0.5表示两端各裁剪0.5%)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    x2d = x
    if x2d.ndim > 2:
        raise ValueError("请先切片/聚合成2D矩阵再可视化")
    H, W = x2d.shape

    # 颜色拉伸，避免极端值压扁色阶
    lo, hi = np.percentile(x2d, [clip_percentile, 100-clip_percentile])
    lo = float(lo); hi = float(hi) if hi > lo else lo + 1e-6

    plt.figure()
    im = plt.imshow(x2d, vmin=lo, vmax=hi)
    plt.title(title)
    plt.colorbar()

    # 标注 top/bottom
    flat = x2d.reshape(-1)
    idx_top = np.argpartition(flat, -topk)[-topk:]
    idx_bot = np.argpartition(flat,  topk)[:topk]
    for idx in idx_top:
        y, z = divmod(int(idx), W)
        plt.scatter(z, y, s=10)           # 前k大
    for idx in idx_bot:
        y, z = divmod(int(idx), W)
        plt.scatter(z, y, s=10, marker="x")  # 前k小
    plt.show()



# 查看张量形状和数据类型
# print(f"Shape: {cls_prob.shape}, Dtype: {cls_prob.dtype}")

# # 查看每个类别的前10个最高置信度值及其索引
# for i in range(cls_prob.shape[0]):  # 遍历52个类别
#     values, indices = torch.topk(cls_prob[i], k=10, dim=0)
#     print(f"Class {i}: Top 10 values = {values.tolist()}")
#     print(f"Class {i}: Top 10 indices = {indices.tolist()}")