# V3.5边缘平滑改进方案

## 核心思路

**不要在训练时强制平滑，而是让模型学会预测平滑的边界**

具体策略：
1. 回归V3.0的简洁架构
2. 在输出空间（而非特征空间）进行边缘感知的后处理
3. 使用更智能的损失函数，而非简单的Laplacian

---

## 方案A：边缘感知平滑后处理（推荐★★★★★）

### 优点
- ✅ 不改变训练过程，保持V3.0的优势
- ✅ 仅在推理时应用，可以灵活调整
- ✅ 计算开销小，易于实现
- ✅ 可以快速验证效果（1-2天）
- ✅ 无需重新训练

### 实现细节

#### 核心模块：EdgeAwareSmoothing

```python
import torch
import torch.nn as nn

class EdgeAwareSmoothing(nn.Module):
    """
    边缘感知平滑：在平坦区域平滑，在尖锐转角处保持锐利
    
    核心思想：
    1. 计算每个点的曲率（二阶差分）
    2. 曲率大的点（尖锐转角）→ 低平滑权重
    3. 曲率小的点（平坦边缘）→ 高平滑权重
    4. 加权混合原始轮廓和平滑轮廓
    """
    def __init__(self, kernel_size=5, curvature_threshold=0.1):
        super().__init__()
        self.kernel_size = kernel_size
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
        
        # 曲率越大，权重越小（保持尖锐转角）
        # 曲率越小，权重越大（平滑平坦区域）
        smooth_weight = torch.exp(-curvature / self.curvature_threshold)  # (N, P, 1)
        
        # 循环卷积平滑（简单的3点平均）
        prev = torch.roll(contour, 1, dims=1)
        next = torch.roll(contour, -1, dims=1)
        smoothed_local = (prev + 2 * contour + next) / 4
        
        # 加权混合
        smoothed = smooth_weight * smoothed_local + (1 - smooth_weight) * contour
        
        return smoothed
```

#### 使用方式

在推理脚本中集成：

```python
# 在infer_v3_refinement.py中添加

from lib.utils.edge_smoothing import EdgeAwareSmoothing

# 初始化平滑模块
smoother = EdgeAwareSmoothing(
    kernel_size=5,
    curvature_threshold=0.1  # 可调参数
).to(device)

# 在推理循环中应用
for batch in dataloader:
    # ... 模型推理 ...
    pred_contours = model(...)  # (N, P, 2)
    
    # 应用边缘感知平滑（可迭代2-3次）
    smoothed_contours = pred_contours
    for _ in range(2):  # 迭代次数可调
        smoothed_contours = smoother(smoothed_contours)
    
    # 使用平滑后的轮廓
    final_contours = smoothed_contours
```

### 超参数调整

| 参数 | 默认值 | 作用 | 调整建议 |
|------|--------|------|----------|
| `curvature_threshold` | 0.1 | 控制曲率敏感度 | 越小越保留尖锐转角 |
| 迭代次数 | 2 | 平滑强度 | 1-3次，过多会过度平滑 |
| `kernel_size` | 5 | 平滑窗口大小 | 3/5/7，越大越平滑 |

### 验证步骤

1. **快速测试**（30分钟）
   ```bash
   cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30
   export CFG_FILE=configs/btcv_diffusion_dit_v3.yaml
   python infer_v3_refinement.py --ckpt data/outputs/btcv_diffusion_dit_v3/checkpoints/latest.pt --use_edge_smooth
   ```

2. **对比可视化**
   - 生成V3.0原始输出
   - 生成V3.0+EdgeAwareSmoothing输出
   - 对比边缘质量和形状保真度

3. **调整超参数**
   - 如果还有毛刺：增加迭代次数或降低curvature_threshold
   - 如果过度平滑：减少迭代次数或提高curvature_threshold

---

## 方案B：改进的训练时损失（备选★★★）

### 适用场景
- 方案A效果不理想
- 希望模型直接学会预测平滑边界
- 可以接受重新训练的时间成本

### 核心改进：边缘感知平滑损失

```python
def edge_aware_smoothness_loss(contour, curvature_threshold=0.1):
    """
    边缘感知平滑损失：在低曲率区域惩罚不平滑，在高曲率区域放松约束
    
    与V3.3b的Laplacian损失的区别：
    - V3.3b：所有点使用相同的平滑权重
    - 本方案：根据曲率自适应调整权重
    """
    # 计算曲率
    prev = torch.roll(contour, 1, dims=1)
    next = torch.roll(contour, -1, dims=1)
    d2 = (next - 2 * contour + prev)
    curvature = torch.norm(d2, dim=-1, keepdim=True)
    
    # 自适应权重：曲率大的点权重小
    weight = torch.exp(-curvature / curvature_threshold)
    
    # Laplacian
    laplacian = contour - (prev + next) / 2
    
    # 加权损失
    loss = torch.mean(weight * (laplacian ** 2))
    return loss
```

### 集成到训练器

在 `diffusion_trainer.py` 中修改：

```python
# 替换原来的smooth_loss计算（第117-126行）

if smooth_weight > 0:
    # 使用边缘感知平滑损失
    smooth_loss = edge_aware_smoothness_loss(
        contours, 
        curvature_threshold=0.1
    )
    loss = loss + smooth_weight * smooth_loss
    scalar_stats.update({
        'smooth_loss': smooth_loss,
        'smooth_loss_scaled': smooth_weight * smooth_loss
    })
```

### 配置文件

创建 `configs/btcv_diffusion_dit_v3_5.yaml`：

```yaml
# 基于V3.0，添加边缘感知平滑损失
model: 'sbd'
network: 'ro_34'
task: 'snake'
model_dir: "data/model/btcv_diffusion_dit_v3_5"

gpus: [0]  # 根据实际空闲GPU调整
resume: false

train_or_test: train

train:
    dataset: 'BtcvMini'
    data_path: '/mnt/sdb1/leijh/DiffusionSnake/Datasets/BTCV/btcv_png_new_snake'
    per_contour: false
    
    optim: 'adamw'
    lr: 5e-5
    batch_size: 4
    epoch: 10000
    save_ep: 100

# 关键：使用边缘感知平滑损失，权重较小
loss_scales: {det: 0, ex: 1.0, py: 1.2, edge_aware_smooth: 0.02}

use_diffusion_evolution: true
diffusion_timesteps: 1000
use_ddim_inference: true

# 使用V3.0架构（不使用V3.3的CircularConv）
use_dit_v3_3: false
dit_num_layers: 6
dit_num_heads: 8
dit_state_dim: 256
```

### 训练步骤

```bash
cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30
conda activate snake1

# 检查GPU
nvidia-smi

# 训练
export CFG_FILE=configs/btcv_diffusion_dit_v3_5.yaml
CUDA_VISIBLE_DEVICES=0 python diffusion_train.py
```

---

## 方案C：在输出空间应用CircularConv（备选★★）

### 核心改进

如果要保留CircularConv的思路，应该：

1. **移到输出空间**：在final_layer之后应用
2. **增大残差权重**：从0.1提升到0.3-0.5
3. **使用更大的卷积核**：从5提升到7或9

### 实现

```python
class OutputSpaceSmoothing(nn.Module):
    """
    在输出空间（2维坐标）应用循环卷积
    """
    def __init__(self, kernel_size=7, residual_weight=0.3):
        super().__init__()
        self.kernel_size = kernel_size
        self.residual_weight = residual_weight
        self.conv = nn.Conv1d(2, 2, kernel_size, padding=0)
        
    def forward(self, contour):
        """
        Args:
            contour: (N, P, 2) - 预测的轮廓坐标
        Returns:
            smoothed: (N, P, 2) - 平滑后的轮廓
        """
        # Transpose to (N, 2, P)
        x_t = contour.transpose(1, 2)
        
        # Circular padding
        pad_size = self.kernel_size // 2
        x_padded = F.pad(x_t, (pad_size, pad_size), mode='circular')
        
        # Apply convolution
        x_conv = self.conv(x_padded)
        
        # Transpose back
        x_out = x_conv.transpose(1, 2)
        
        # Residual with larger weight
        return contour + self.residual_weight * x_out
```

---

## 推荐实施路线

### 阶段1：快速验证（1-2天）★推荐优先

1. **实现EdgeAwareSmoothing模块**
   - 创建文件：`lib/utils/edge_smoothing.py`
   - 实现上述代码

2. **在V3.0的checkpoint上测试**
   - 使用现有的10k轮checkpoint
   - 对比平滑前后的效果
   - 调整超参数（curvature_threshold, 迭代次数）

3. **可视化对比**
   - 生成V3.0原始输出 vs V3.0+EdgeAwareSmoothing
   - 评估边缘质量和形状保真度

**预期结果**：
- 如果效果好：直接使用，无需重新训练
- 如果效果一般：进入阶段2

### 阶段2：训练验证（3-5天）

如果阶段1效果不理想，再考虑修改训练：

1. **实现edge_aware_smoothness_loss**
   - 修改`diffusion_trainer.py`
   - 添加新的损失项

2. **训练V3.5模型**
   - 基于V3.0架构（不使用V3.3的CircularConv）
   - 使用edge_aware_smooth loss
   - 权重从0.01开始尝试

3. **对比实验**
   - V3.0 baseline
   - V3.5 (edge_aware_smooth loss)
   - V3.0 + EdgeAwareSmoothing后处理

### 阶段3：进一步优化（可选）

1. **多尺度平滑**
   - 在不同的diffusion timestep应用不同强度的平滑
   - 早期timestep（粗糙阶段）：弱平滑
   - 后期timestep（精细阶段）：强平滑

2. **学习平滑参数**
   - 让模型自动学习每个点的最优平滑权重
   - 添加一个小型MLP预测smooth_weight

---

## 关键文件清单

### 需要创建的文件
- `lib/utils/edge_smoothing.py` - EdgeAwareSmoothing模块（方案A）
- `configs/btcv_diffusion_dit_v3_5.yaml` - V3.5配置（方案B）

### 需要修改的文件
- `infer_v3_refinement.py` - 集成EdgeAwareSmoothing（方案A）
- `lib/train/trainers/diffusion_trainer.py` - 添加edge_aware_smoothness_loss（方案B）

### 参考文件
- `lib/networks/diffusion/dit_denoiser_v3.py` - V3.0架构（保持不变）
- `lib/networks/diffusion/dit_denoiser_v3_3.py` - V3.3架构（作为反面教材）

---

## 验证标准

### 定性评估
- ✅ 边缘是否平滑，无明显锯齿
- ✅ 尖锐转角是否保留
- ✅ 整体形状是否准确
- ✅ 是否有过度平滑现象

### 定量评估
- **IoU**（与ground truth的交并比）：应该不低于V3.0
- **Hausdorff距离**（边界点的最大距离）：应该降低
- **边缘平滑度**（相邻点距离的方差）：应该降低

---

## 总结

推荐从**方案A（EdgeAwareSmoothing后处理）**开始：
- ✅ 最快速（1-2天验证）
- ✅ 最灵活（无需重新训练）
- ✅ 最安全（不破坏V3.0的优势）

如果方案A效果不理想，再考虑方案B（训练时损失）。

方案C不推荐，因为它本质上还是在做均匀平滑，缺乏自适应性。
