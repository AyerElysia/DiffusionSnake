# 自适应点数验证方案

## 目标

验证假设：**根据轮廓大小自适应调整点数，可以显著减少小轮廓的毛刺问题**

---

## 实验设计

### 阶段1：后处理快速验证（1-2天）⏰

#### 目的
不修改训练，仅在推理后调整点数，快速验证假设

#### 实现步骤

**步骤1：实现点数调整函数**

```python
def adaptive_resample(contour, target_density=2.5):
    """
    根据目标点密度重采样轮廓
    
    Args:
        contour: (N, 2) numpy array
        target_density: 目标点密度（像素/点）
    
    Returns:
        resampled: 重采样后的轮廓
        num_points: 使用的点数
    """
    # 计算周长
    dists = np.linalg.norm(np.diff(contour, axis=0, append=contour[:1]), axis=1)
    perimeter = np.sum(dists)
    
    # 计算目标点数
    target_points = int(perimeter / target_density)
    target_points = max(32, min(target_points, 256))  # 限制范围
    target_points = (target_points // 4) * 4  # 确保是4的倍数
    
    # 下采样
    if target_points < len(contour):
        # 使用均匀采样
        indices = np.linspace(0, len(contour)-1, target_points, dtype=int)
        downsampled = contour[indices]
    else:
        downsampled = contour
    
    # 上采样回128点（用于对比）
    upsampled = uniform_upsample_numpy(downsampled, 128)
    
    return upsampled, target_points


def uniform_upsample_numpy(contour, target_points):
    """numpy版本的均匀上采样"""
    from scipy import interpolate
    
    # 计算累积弧长
    dists = np.linalg.norm(np.diff(contour, axis=0, append=contour[:1]), axis=1)
    cumsum = np.concatenate([[0], np.cumsum(dists)])
    
    # 归一化到[0, 1]
    cumsum_norm = cumsum / cumsum[-1]
    
    # 创建插值函数
    fx = interpolate.interp1d(cumsum_norm, contour[:, 0], kind='linear')
    fy = interpolate.interp1d(cumsum_norm, contour[:, 1], kind='linear')
    
    # 均匀采样
    t = np.linspace(0, 1, target_points, endpoint=False)
    x_new = fx(t)
    y_new = fy(t)
    
    return np.stack([x_new, y_new], axis=1)
```

**步骤2：创建验证脚本**

```python
# verify_adaptive_points.py

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import json

# 加载预测结果
pred_polys = np.load('visual/burr_v3_4_full/pred_polys.npy')  # 需要先保存
gt_polys = np.load('visual/burr_v3_4_full/gt_polys.npy')

results = []

for i, pred in enumerate(pred_polys):
    # 原始预测（128点）
    curv_original = compute_curvature(pred)
    
    # 自适应重采样
    pred_adaptive, num_points = adaptive_resample(pred, target_density=2.5)
    curv_adaptive = compute_curvature(pred_adaptive)
    
    # 统计
    improvement = (curv_original.max() - curv_adaptive.max()) / curv_original.max() * 100
    
    results.append({
        'contour_id': i,
        'num_points_used': num_points,
        'curv_original': float(curv_original.max()),
        'curv_adaptive': float(curv_adaptive.max()),
        'improvement': float(improvement)
    })
    
    print(f"轮廓{i}: {num_points}点, 曲率 {curv_original.max():.2f} → {curv_adaptive.max():.2f} ({improvement:+.1f}%)")

# 保存结果
with open('visual/burr_v3_4_full/adaptive_points_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

**步骤3：可视化对比**

创建对比图：
- 原始128点预测
- 自适应点数预测
- 曲率热力图对比

#### 预期结果

| 轮廓 | 周长 | 建议点数 | 原始曲率 | 自适应曲率 | 改善 |
|------|------|---------|---------|-----------|------|
| 1 | 75.8 | 32 | 36.16 | 15-20 | 40-50% |
| 5 | 152.1 | 64 | 25.52 | 12-15 | 40-50% |
| 2 | 182.0 | 72 | 22.64 | 10-12 | 45-55% |
| 3 | 295.8 | 120 | 11.76 | 8-10 | 20-30% |
| 4 | 374.8 | 152 | 4.59 | 4-5 | 持平 |

#### 成功标准

- 小轮廓（1、5、2）曲率降低 > 30%
- 大轮廓（3、4）曲率变化 < 10%
- 视觉上毛刺明显减少

---

### 阶段2：训练时集成（1周）⏰

#### 目的
在训练流程中集成自适应点数，从根本上解决问题

#### 实现步骤

**步骤1：修改数据准备**

修改 `lib/datasets/btcv/snake.py`:

```python
class BtcvDataset(Dataset):
    def __init__(self, ...):
        self.adaptive_points = True  # 新增开关
        self.target_density = 2.5    # 目标点密度
    
    def prepare_evolution(self, poly, ...):
        # 原有代码...
        
        if self.adaptive_points:
            # 计算周长
            perimeter = self.compute_perimeter(poly)
            
            # 确定点数
            num_points = int(perimeter / self.target_density)
            num_points = max(32, min(num_points, 256))
            num_points = (num_points // 4) * 4
        else:
            num_points = 128  # 原有固定值
        
        # 重采样
        poly_resampled = snake_gcn_utils.uniform_upsample(poly, num_points)[0]
        
        return poly_resampled, num_points
```

**步骤2：修改collate函数**

修改 `lib/datasets/collate_batch.py`:

```python
def snake_collator(batch):
    # 原有代码...
    
    # 处理可变点数
    if 'num_points' in batch[0]:
        # 找到最大点数
        max_points = max([b['num_points'] for b in batch])
        
        # Padding到最大点数
        for b in batch:
            if b['num_points'] < max_points:
                # Padding轮廓
                pad_size = max_points - b['num_points']
                b['i_it_py'] = np.pad(b['i_it_py'], ((0, pad_size), (0, 0)), mode='edge')
                b['i_gt_py'] = np.pad(b['i_gt_py'], ((0, pad_size), (0, 0)), mode='edge')
                
                # 记录mask
                b['point_mask'] = np.concatenate([
                    np.ones(b['num_points']),
                    np.zeros(pad_size)
                ])
    
    # 原有代码...
```

**步骤3：修改模型**

修改 `lib/networks/snake/ct_snake.py`:

```python
class SnakeNet(nn.Module):
    def forward(self, batch):
        # 原有代码...
        
        # 获取point_mask
        if 'point_mask' in batch:
            point_mask = batch['point_mask']  # (B*M, P)
        else:
            point_mask = None
        
        # 在计算损失时使用mask
        if point_mask is not None:
            loss = loss * point_mask.unsqueeze(-1)
            loss = loss.sum() / point_mask.sum()
        
        # 原有代码...
```

**步骤4：训练配置**

修改 `configs/btcv_diffusion_dit_v3_4_adaptive.yaml`:

```yaml
# 新增配置
adaptive_points: true
target_point_density: 2.5
min_points: 32
max_points: 256
```

#### 训练步骤

1. **单样本验证**（2天）
   ```bash
   # 使用V3.4单样本过拟合快速验证
   python diffusion_train.py --cfg configs/btcv_diffusion_dit_v3_4_adaptive.yaml
   ```

2. **评估改善**（1天）
   ```bash
   python analyze_burr_v3_4_full.py --cfg configs/btcv_diffusion_dit_v3_4_adaptive.yaml
   ```

3. **对比分析**（1天）
   - 对比固定128点 vs 自适应点数
   - 生成详细报告

#### 成功标准

- 小轮廓平均曲率 < 15（当前28.1）
- 大轮廓平均曲率 < 10（当前8.2）
- 训练收敛稳定
- 推理速度下降 < 20%

---

### 阶段3：全面评估（2周）⏰

#### 目的
在完整数据集上验证效果

#### 实验步骤

1. **完整数据集训练**（1周）
   - 使用所有BTCV训练数据
   - 训练epoch数：根据收敛情况

2. **全面评估**（3天）
   - 在测试集上评估所有样本
   - 统计不同大小轮廓的改善

3. **消融实验**（2天）
   - 测试不同的target_density（2.0, 2.5, 3.0）
   - 测试不同的点数范围（32-256, 64-512）

4. **最终报告**（2天）
   - 生成完整的评估报告
   - 可视化改善效果

---

## 实现细节

### 关键函数实现

#### 1. 计算周长

```python
def compute_perimeter(contour):
    """计算轮廓周长"""
    dists = np.linalg.norm(
        np.diff(contour, axis=0, append=contour[:1]), 
        axis=1
    )
    return np.sum(dists)
```

#### 2. 自适应点数计算

```python
def compute_adaptive_num_points(perimeter, target_density=2.5, 
                                min_points=32, max_points=256):
    """
    根据周长计算自适应点数
    
    Args:
        perimeter: 轮廓周长
        target_density: 目标点密度（像素/点）
        min_points: 最小点数
        max_points: 最大点数
    
    Returns:
        num_points: 自适应点数
    """
    num_points = int(perimeter / target_density)
    num_points = max(min_points, min(num_points, max_points))
    num_points = (num_points // 4) * 4  # 确保是4的倍数
    return num_points
```

#### 3. 均匀重采样

```python
def uniform_resample(contour, target_points):
    """
    均匀重采样轮廓到目标点数
    
    使用弧长参数化确保均匀分布
    """
    from scipy import interpolate
    
    # 计算累积弧长
    dists = np.linalg.norm(
        np.diff(contour, axis=0, append=contour[:1]), 
        axis=1
    )
    cumsum = np.concatenate([[0], np.cumsum(dists)])
    cumsum_norm = cumsum / cumsum[-1]
    
    # 插值
    fx = interpolate.interp1d(cumsum_norm, contour[:, 0], kind='cubic')
    fy = interpolate.interp1d(cumsum_norm, contour[:, 1], kind='cubic')
    
    # 均匀采样
    t = np.linspace(0, 1, target_points, endpoint=False)
    x_new = fx(t)
    y_new = fy(t)
    
    return np.stack([x_new, y_new], axis=1)
```

---

## 评估指标

### 主要指标

| 指标 | 当前值 | 目标值 | 计算方法 |
|------|--------|--------|---------|
| 小轮廓平均曲率 | 28.1 | < 15 | 轮廓1、2、5的平均 |
| 大轮廓平均曲率 | 8.2 | < 10 | 轮廓3、4的平均 |
| 曲率标准差 | 12.5 | < 8 | 所有轮廓的std |
| 小轮廓尖锐角 | 166 | < 80 | 轮廓1、2、5的总和 |
| 大轮廓尖锐角 | 44 | < 50 | 轮廓3、4的总和 |

### 次要指标

- 训练时间变化
- 推理速度变化
- 模型参数量变化
- 内存占用变化

---

## 风险缓解

### 风险1：可变点数导致训练不稳定

**缓解措施：**
- 使用padding + mask确保batch一致性
- 逐步增加点数范围（先32-128，再32-256）
- 调整学习率和batch size

### 风险2：大轮廓效果变差

**缓解措施：**
- 设置max_points上限（如256）
- 监控大轮廓的指标变化
- 如果变差，调整target_density

### 风险3：实现复杂度高

**缓解措施：**
- 先实现后处理版本验证假设
- 再逐步集成到训练流程
- 保留固定128点的baseline对比

---

## 时间表

### 第1周

- **Day 1-2**: 实现后处理验证
  - 编写adaptive_resample函数
  - 创建验证脚本
  - 生成对比可视化

- **Day 3**: 分析结果
  - 如果改善 > 30%，继续
  - 如果改善 < 10%，重新评估假设

- **Day 4-5**: 准备训练集成
  - 修改数据准备代码
  - 修改collate函数
  - 单元测试

### 第2周

- **Day 1-3**: 单样本训练验证
  - 训练V3.4自适应版本
  - 监控训练曲线
  - 评估改善效果

- **Day 4-5**: 调优
  - 调整target_density
  - 调整点数范围
  - 优化实现

### 第3-4周

- **Week 3**: 完整数据集训练
- **Week 4**: 全面评估和报告

---

## 成功标准

### 必须达成（P0）

- ✓ 小轮廓曲率降低 > 30%
- ✓ 训练收敛稳定
- ✓ 推理速度下降 < 30%

### 期望达成（P1）

- ✓ 小轮廓曲率降低 > 50%
- ✓ 大轮廓曲率保持或改善
- ✓ 推理速度下降 < 20%

### 可选达成（P2）

- ✓ 所有轮廓曲率 < 15
- ✓ 曲率标准差 < 5
- ✓ 推理速度下降 < 10%

---

## 备选方案

如果自适应点数效果不佳或实现困难，考虑：

### 备选方案1：固定减少小轮廓点数

简化版本：
- 小轮廓（周长<150）：64点
- 中轮廓（周长150-300）：128点
- 大轮廓（周长>300）：128点

### 备选方案2：尺度归一化

在训练时将所有轮廓归一化到相同尺度：
```python
scale = 200 / perimeter
contour_normalized = contour * scale
```

### 备选方案3：多尺度训练

对小轮廓进行放大，对大轮廓进行缩小

---

## 总结

### 核心假设

**小轮廓点太密集（点密度<1.0），导致对噪声敏感，产生严重毛刺**

### 验证方法

**阶段1（1-2天）：** 后处理快速验证  
**阶段2（1周）：** 训练时集成  
**阶段3（2周）：** 全面评估

### 预期收益

- 小轮廓曲率降低：30-60%
- 整体毛刺改善：25-50%
- 为后续优化提供新思路

---

**方案制定时间：** 2026-04-18  
**预计完成时间：** 2026-05-02（2周）  
**负责人：** 待定  
**优先级：** P0（高优先级）
